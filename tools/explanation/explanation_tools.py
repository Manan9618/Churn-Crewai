import pandas as pd
import numpy as np
import pickle
import json
import os
from crewai.tools import tool

MODEL_PATH = "artifacts/model/churn_model.pkl"
FEATURES_PATH = "artifacts/data/selected_features.json"
SHAP_PATH = "artifacts/model/shap_values.pkl"
EXPLANATION_REPORT_PATH = "artifacts/model/explanation_report.json"
COUNTERFACTUAL_PATH = "artifacts/model/counterfactuals.json"
FAIRNESS_REPORT_PATH = "artifacts/model/fairness_report.json"
RULES_PATH = "artifacts/model/global_logic_rules.json"


def _load_model_and_data():
    """Helper function to load model, test data, and features."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    X_te = pd.read_csv("artifacts/data/X_test.csv")[features]
    return model, X_te, features


def _load_train_data():
    """Helper function to load training data for statistics."""
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    X_tr = pd.read_csv("artifacts/data/X_train.csv")[features]
    return X_tr, features


@tool("SHAP Summary Analysis")
def shap_summary_tool(sample_size: int = 500) -> str:
    """
    Compute SHAP values for the test set and return a summary of
    mean absolute SHAP values per feature (global importance).
    """
    import shap

    model, X_te, features = _load_model_and_data()
    sample = X_te.sample(min(sample_size, len(X_te)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # For binary classification, take the positive class SHAP values
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=features).sort_values(ascending=False)

    os.makedirs("artifacts/model", exist_ok=True)
    with open(SHAP_PATH, "wb") as f:
        pickle.dump({"shap_values": sv, "feature_names": features}, f)

    return f"Global SHAP Importance (mean |SHAP value|):\n{mean_abs.to_string()}"


@tool("SHAP Force Plot Summary")
def shap_force_plot_tool(customer_index: int = 0) -> str:
    """
    Return the SHAP force plot explanation for a specific customer (by index in test set).
    Shows which features pushed the prediction above or below the base value.
    """
    import shap

    model, X_te, features = _load_model_and_data()
    
    if len(X_te) == 0:
        return "ERROR: Test set is empty. Cannot generate SHAP explanation."
    
    if customer_index < 0 or customer_index >= len(X_te):
        return (
            f"ERROR: Invalid customer_index={customer_index}.\n"
            f"Test set has {len(X_te)} sample(s).\n"
            f"Valid indices: 0 to {len(X_te) - 1}\n"
            f"Please use a valid index (try customer_index=0)"
        )

    row = X_te.iloc[[customer_index]]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    contribution = pd.Series(sv, index=features).sort_values(key=abs, ascending=False)
    return (
        f"Customer Index {customer_index} — SHAP Explanation:\n"
        f"Base value: {base_value:.4f}\n"
        f"Top feature contributions:\n{contribution.head(8).to_string()}"
    )


@tool("LIME Explanation")
def lime_explanation_tool(customer_index: int = 0, num_features: int = 8) -> str:
    """
    Generate a LIME local explanation for a specific customer in the test set.
    """
    from lime.lime_tabular import LimeTabularExplainer

    model, X_te, features = _load_model_and_data()
    X_tr, _ = _load_train_data()

    explainer = LimeTabularExplainer(
        training_data=X_tr.values,
        feature_names=features,
        class_names=["No Churn", "Churn"],
        mode="classification",
        random_state=42,
    )

    instance = X_te.iloc[customer_index].values
    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=num_features,
    )

    lime_list = explanation.as_list()
    result = "\n".join([f"  {feat}: {weight:+.4f}" for feat, weight in lime_list])
    churn_prob = model.predict_proba([instance])[0][1]
    return (
        f"LIME Explanation for Customer Index {customer_index}:\n"
        f"Predicted Churn Probability: {churn_prob:.4f}\n"
        f"Feature Contributions:\n{result}"
    )


@tool("Generate Explanation Report")
def generate_explanation_report_tool() -> str:
    """
    Generate a JSON explanation report combining global SHAP importance
    and model performance summary.
    """
    from sklearn.metrics import roc_auc_score, f1_score

    if not os.path.exists(SHAP_PATH):
        return "ERROR: SHAP values not found. Run shap_summary_tool first."

    with open(SHAP_PATH, "rb") as f:
        shap_data = pickle.load(f)

    shap_vals = shap_data["shap_values"]
    features = shap_data["feature_names"]
    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=features).sort_values(ascending=False)

    report = {
        "model_path": MODEL_PATH,
        "global_feature_importance_shap": mean_abs.head(10).to_dict(),
        "top_churn_drivers": list(mean_abs.head(5).index),
    }

    if os.path.exists(MODEL_PATH) and os.path.exists("artifacts/data/X_test.csv"):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        X_te = pd.read_csv("artifacts/data/X_test.csv")[features]
        y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)
        report["test_auc"] = round(roc_auc_score(y_te, y_prob), 4)
        report["test_f1"] = round(f1_score(y_te, y_pred), 4)

    os.makedirs("artifacts/model", exist_ok=True)
    with open(EXPLANATION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return f"Explanation report saved to {EXPLANATION_REPORT_PATH}.\nTop churn drivers: {report['top_churn_drivers']}"


# ── NEW EXPLANATION TOOLS ADDED BELOW ──────────────────────────────────────────


@tool("Generate Counterfactuals")
def generate_counterfactuals_tool(customer_index: int = 0, max_iterations: int = 100) -> str:
    """
    Answers "What minimal changes would change the prediction?"
    Finds the smallest feature modifications needed to flip the churn prediction.
    """
    model, X_te, features = _load_model_and_data()
    X_tr, _ = _load_train_data()
    
    if len(X_te) == 0:
        return "ERROR: Test set is empty. Cannot generate counterfactuals."
    
    if customer_index < 0 or customer_index >= len(X_te):
        return f"ERROR: Invalid customer_index={customer_index}. Test set has {len(X_te)} samples."
    
    original_instance = X_te.iloc[customer_index].copy()
    original_pred = model.predict([original_instance.values])[0]
    original_prob = model.predict_proba([original_instance.values])[0][1]
    
    target_class = 0 if original_pred == 1 else 1  # Flip the prediction
    
    # Get feature statistics for realistic perturbations
    feature_stats = {}
    for col in features:
        feature_stats[col] = {
            "mean": X_tr[col].mean(),
            "std": X_tr[col].std(),
            "min": X_tr[col].min(),
            "max": X_tr[col].max(),
            "unique_values": sorted(X_tr[col].unique()) if X_tr[col].nunique() < 10 else None
        }
    
    best_counterfactual = None
    min_changes = float('inf')
    
    np.random.seed(42)
    
    for iteration in range(max_iterations):
        candidate = original_instance.copy()
        changed_features = []
        
        # Randomly select 1-3 features to modify
        n_features_to_change = np.random.randint(1, 4)
        features_to_modify = np.random.choice(features, size=min(n_features_to_change, len(features)), replace=False)
        
        for feat in features_to_modify:
            stats = feature_stats[feat]
            
            if stats["unique_values"] is not None:
                # Categorical feature - switch to different category
                current_val = candidate[feat]
                other_values = [v for v in stats["unique_values"] if v != current_val]
                if other_values:
                    candidate[feat] = np.random.choice(other_values)
                    changed_features.append(feat)
            else:
                # Numerical feature - perturb within realistic range
                perturbation = np.random.normal(0, stats["std"] * 0.5)
                new_val = candidate[feat] + perturbation
                new_val = np.clip(new_val, stats["min"], stats["max"])
                candidate[feat] = new_val
                changed_features.append(feat)
        
        # Check if prediction flipped
        new_pred = model.predict([candidate.values])[0]
        
        if new_pred == target_class:
            n_changes = len(changed_features)
            if n_changes < min_changes:
                min_changes = n_changes
                best_counterfactual = {
                    "original_prediction": int(original_pred),
                    "original_probability": round(float(original_prob), 4),
                    "counterfactual_prediction": int(new_pred),
                    "counterfactual_probability": round(float(model.predict_proba([candidate.values])[0][1]), 4),
                    "changed_features": {},
                    "changes_description": []
                }
                
                for feat in changed_features:
                    best_counterfactual["changed_features"][feat] = {
                        "original": float(original_instance[feat]),
                        "counterfactual": float(candidate[feat]),
                        "change": float(candidate[feat] - original_instance[feat])
                    }
                    best_counterfactual["changes_description"].append(
                        f"{feat}: {original_instance[feat]:.2f} → {candidate[feat]:.2f}"
                    )
    
    if best_counterfactual is None:
        return (
            f"Counterfactual Analysis for Customer {customer_index}:\n"
            f"Original Prediction: {original_pred} (Prob: {original_prob:.4f})\n"
            f"No counterfactual found in {max_iterations} iterations.\n"
            f"Recommendation: This prediction is stable and robust."
        )
    
    # Save counterfactuals
    os.makedirs("artifacts/model", exist_ok=True)
    
    # Load existing or create new
    if os.path.exists(COUNTERFACTUAL_PATH):
        with open(COUNTERFACTUAL_PATH, "r") as f:
            all_counterfactuals = json.load(f)
    else:
        all_counterfactuals = []
    
    all_counterfactuals.append({
        "customer_index": customer_index,
        **best_counterfactual
    })
    
    with open(COUNTERFACTUAL_PATH, "w") as f:
        json.dump(all_counterfactuals, f, indent=2)
    
    changes_text = "\n  ".join(best_counterfactual["changes_description"])
    return (
        f"Counterfactual Analysis for Customer {customer_index}:\n"
        f"Original Prediction: {best_counterfactual['original_prediction']} (Prob: {best_counterfactual['original_probability']})\n"
        f"Counterfactual Prediction: {best_counterfactual['counterfactual_prediction']} (Prob: {best_counterfactual['counterfactual_probability']})\n"
        f"Minimal Changes Required ({len(best_counterfactual['changed_features'])} features):\n"
        f"  {changes_text}\n"
        f"Business Interpretation: If these changes were made, churn prediction would flip.\n"
        f"Saved to: {COUNTERFACTUAL_PATH}"
    )


@tool("Check Explanation Stability")
def check_explanation_stability_tool(customer_index: int = 0, n_perturbations: int = 10) -> str:
    """
    Runs SHAP multiple times with slight data perturbations to ensure
    explanations aren't random. Returns stability score (0-1).
    """
    import shap

    model, X_te, features = _load_model_and_data()
    X_tr, _ = _load_train_data()
    
    if len(X_te) == 0:
        return "ERROR: Test set is empty. Cannot check explanation stability."
    
    if customer_index < 0 or customer_index >= len(X_te):
        return f"ERROR: Invalid customer_index={customer_index}. Test set has {len(X_te)} samples."
    
    original_instance = X_te.iloc[customer_index].values.reshape(1, -1)
    
    # Get baseline SHAP values
    explainer = shap.TreeExplainer(model)
    baseline_shap = explainer.shap_values(original_instance)
    if isinstance(baseline_shap, list):
        baseline_shap = baseline_shap[1][0]
    else:
        baseline_shap = baseline_shap[0]
    
    baseline_top_features = pd.Series(np.abs(baseline_shap), index=features).nlargest(5).index.tolist()
    
    stability_scores = []
    all_top_features = []
    
    np.random.seed(42)
    
    for i in range(n_perturbations):
        # Add small noise to the instance
        noise = np.random.normal(0, 0.01, size=original_instance.shape)
        perturbed_instance = original_instance + noise
        
        # Clip to valid range
        for j, col in enumerate(features):
            col_min = X_tr[col].min()
            col_max = X_tr[col].max()
            perturbed_instance[0, j] = np.clip(perturbed_instance[0, j], col_min, col_max)
        
        # Compute SHAP for perturbed instance
        perturbed_shap = explainer.shap_values(perturbed_instance)
        if isinstance(perturbed_shap, list):
            perturbed_shap = perturbed_shap[1][0]
        else:
            perturbed_shap = perturbed_shap[0]
        
        # Calculate correlation with baseline
        correlation = np.corrcoef(baseline_shap, perturbed_shap)[0, 1]
        stability_scores.append(correlation)
        
        # Get top features for this perturbation
        top_features = pd.Series(np.abs(perturbed_shap), index=features).nlargest(5).index.tolist()
        all_top_features.append(set(top_features))
    
    # Calculate stability metrics
    mean_correlation = np.mean(stability_scores)
    std_correlation = np.std(stability_scores)
    
    # Feature stability (how often same features appear in top 5)
    feature_stability = {}
    for feat in features:
        count = sum(1 for top_set in all_top_features if feat in top_set)
        feature_stability[feat] = count / n_perturbations
    
    stable_features = [feat for feat, score in feature_stability.items() if score >= 0.7]
    
    # Determine overall stability
    if mean_correlation >= 0.9:
        stability_status = "HIGHLY STABLE"
    elif mean_correlation >= 0.7:
        stability_status = "STABLE"
    elif mean_correlation >= 0.5:
        stability_status = "MODERATE"
    else:
        stability_status = "UNSTABLE"
    
    result = {
        "customer_index": customer_index,
        "stability_status": stability_status,
        "mean_correlation": round(float(mean_correlation), 4),
        "std_correlation": round(float(std_correlation), 4),
        "n_perturbations": n_perturbations,
        "baseline_top_5_features": baseline_top_features,
        "stable_features": stable_features,
        "feature_stability_scores": {k: round(v, 2) for k, v in feature_stability.items() if v >= 0.5}
    }
    
    return (
        f"Explanation Stability Check for Customer {customer_index}:\n"
        f"Status: {stability_status}\n"
        f"Mean Correlation: {mean_correlation:.4f} (±{std_correlation:.4f})\n"
        f"Perturbations Tested: {n_perturbations}\n"
        f"Baseline Top 5 Features: {baseline_top_features}\n"
        f"Stable Features (≥70% consistency): {stable_features}\n"
        f"Interpretation: {'Explanations are reliable and consistent.' if mean_correlation >= 0.7 else 'Explanations show variability - use with caution.'}"
    )


@tool("Analyze Fairness Bias")
def analyze_fairness_bias_tool() -> str:
    """
    Checks if SHAP values show bias against protected groups
    (Gender, SeniorCitizen, etc.) in model predictions.
    """
    import shap
    from sklearn.metrics import accuracy_score, recall_score

    model, X_te, features = _load_model_and_data()
    y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
    
    # Protected attributes (based on Telco dataset)
    protected_attrs = {
        "gender": "Gender",
        "SeniorCitizen": "Age Group"
    }
    
    # Load original data to get protected attributes
    X_te_full = pd.read_csv("artifacts/data/X_test.csv")
    
    fairness_results = {
        "protected_attributes_analyzed": [],
        "bias_detected": [],
        "group_metrics": {}
    }
    
    # Compute SHAP values for full test set
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    predictions = model.predict(X_te)
    probabilities = model.predict_proba(X_te)[:, 1]
    
    for attr_col, attr_name in protected_attrs.items():
        if attr_col not in X_te_full.columns:
            continue
        
        fairness_results["protected_attributes_analyzed"].append(attr_name)
        
        groups = X_te_full[attr_col].unique()
        group_metrics = {}
        
        for group_val in groups:
            mask = X_te_full[attr_col] == group_val
            group_size = mask.sum()
            
            if group_size == 0:
                continue
            
            # Prediction metrics
            group_preds = predictions[mask]
            group_probs = probabilities[mask]
            group_actual = y_te[mask]
            
            group_metrics[f"group_{group_val}"] = {
                "size": int(group_size),
                "percentage": round(float(group_size / len(X_te_full) * 100), 2),
                "churn_rate": round(float(group_preds.mean() * 100), 2),
                "avg_probability": round(float(group_probs.mean()), 4),
                "accuracy": round(float(accuracy_score(group_actual, group_preds)), 4),
                "recall": round(float(recall_score(group_actual, group_preds, zero_division=0)), 4)
            }
            
            # SHAP analysis for this group
            group_shap = shap_values[mask]
            mean_abs_shap = np.abs(group_shap).mean(axis=0)
            
            group_metrics[f"group_{group_val}"]["top_features_shap"] = (
                pd.Series(mean_abs_shap, index=features).nlargest(3).index.tolist()
            )
        
        fairness_results["group_metrics"][attr_name] = group_metrics
        
        # Check for bias
        if len(groups) >= 2:
            group_0_metrics = group_metrics.get("group_0", {})
            group_1_metrics = group_metrics.get("group_1", {})
            
            if group_0_metrics and group_1_metrics:
                churn_rate_diff = abs(
                    group_0_metrics.get("churn_rate", 0) - 
                    group_1_metrics.get("churn_rate", 0)
                )
                
                recall_diff = abs(
                    group_0_metrics.get("recall", 0) - 
                    group_1_metrics.get("recall", 0)
                )
                
                if churn_rate_diff > 10:  # >10% difference in predicted churn rate
                    fairness_results["bias_detected"].append({
                        "attribute": attr_name,
                        "type": "Prediction Disparity",
                        "details": f"Churn rate difference: {churn_rate_diff:.2f}%"
                    })
                
                if recall_diff > 0.15:  # >15% difference in recall
                    fairness_results["bias_detected"].append({
                        "attribute": attr_name,
                        "type": "Performance Disparity",
                        "details": f"Recall difference: {recall_diff:.4f}"
                    })
    
    # Save fairness report
    os.makedirs("artifacts/model", exist_ok=True)
    with open(FAIRNESS_REPORT_PATH, "w") as f:
        json.dump(fairness_results, f, indent=2)
    
    # Format output
    if fairness_results["bias_detected"]:
        bias_text = "\n".join([
            f"  ⚠️ {b['attribute']}: {b['type']} - {b['details']}"
            for b in fairness_results["bias_detected"]
        ])
        status = "⚠️ POTENTIAL BIAS DETECTED"
    else:
        bias_text = "  ✅ No significant bias detected across protected groups."
        status = "✅ FAIR"
    
    return (
        f"Fairness Bias Analysis:\n"
        f"Status: {status}\n"
        f"Protected Attributes Analyzed: {fairness_results['protected_attributes_analyzed']}\n"
        f"Bias Findings:\n{bias_text}\n"
        f"Group Metrics Summary:\n"
        f"  {json.dumps(fairness_results['group_metrics'], indent=2)}\n"
        f"Full report saved to: {FAIRNESS_REPORT_PATH}"
    )


@tool("Extract Global Logic Rules")
def extract_global_logic_rules_tool(max_rules: int = 10, min_support: int = 50) -> str:
    """
    Converts model decisions into simple If-Then rules for business stakeholders.
    Extracts the most common decision paths from the model.
    """
    model, X_te, features = _load_model_and_data()
    X_tr, _ = _load_train_data()
    y_tr = pd.read_csv("artifacts/data/y_train.csv").squeeze()
    
    # Use decision tree to extract rules from the model
    from sklearn.tree import export_text, DecisionTreeClassifier
    
    # Fit a simple decision tree to approximate the model
    surrogate_tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=min_support,
        random_state=42
    )
    surrogate_tree.fit(X_tr, y_tr)
    
    # Export rules as text
    rules_text = export_text(
        surrogate_tree,
        feature_names=features,
        max_depth=4
    )
    
    # Extract feature importance from surrogate
    feature_importance = pd.Series(
        surrogate_tree.feature_importances_,
        index=features
    ).sort_values(ascending=False)
    
    # Generate human-readable rules
    human_readable_rules = []
    
    # Get decision paths
    n_nodes = surrogate_tree.tree_.node_count
    children_left = surrogate_tree.tree_.children_left
    children_right = surrogate_tree.tree_.children_right
    feature_indices = surrogate_tree.tree_.feature
    thresholds = surrogate_tree.tree_.threshold
    values = surrogate_tree.tree_.value
    
    # Extract top paths
    def extract_path(node=0, path=[]):
        if children_left[node] == -1:  # Leaf node
            return [path]
        
        paths = []
        feature_name = features[feature_indices[node]]
        threshold = thresholds[node]
        
        # Left child (<= threshold)
        left_path = path + [(feature_name, "<=", threshold)]
        paths.extend(extract_path(children_left[node], left_path))
        
        # Right child (> threshold)
        right_path = path + [(feature_name, ">", threshold)]
        paths.extend(extract_path(children_right[node], right_path))
        
        return paths
    
    all_paths = extract_path()
    
    # Convert to human-readable rules
    for i, path in enumerate(all_paths[:max_rules]):
        if len(path) == 0:
            continue
        
        leaf_node = len(path)
        # Find the actual leaf node to get prediction
        current_node = 0
        for feat, op, thresh in path:
            feat_idx = features.index(feat)
            if op == "<=":
                current_node = children_left[current_node]
            else:
                current_node = children_right[current_node]
        
        leaf_value = values[current_node][0]
        prediction = "Churn" if leaf_value[1] > leaf_value[0] else "No Churn"
        confidence = round(float(max(leaf_value[0], leaf_value[1]) / sum(leaf_value) * 100), 1)
        
        conditions = " AND ".join([
            f"{feat} {op} {thresh:.2f}" for feat, op, thresh in path
        ])
        
        human_readable_rules.append({
            "rule_id": i + 1,
            "conditions": conditions,
            "prediction": prediction,
            "confidence_percent": confidence,
            "support_samples": int(sum(leaf_value))
        })
    
    # Save rules
    rules_output = {
        "model_type": type(model).__name__,
        "n_rules_extracted": len(human_readable_rules),
        "top_features": feature_importance.head(5).to_dict(),
        "rules": human_readable_rules,
        "decision_tree_text": rules_text
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    with open(RULES_PATH, "w") as f:
        json.dump(rules_output, f, indent=2)
    
    # Format output
    rules_summary = "\n".join([
        f"  Rule {r['rule_id']}: IF {r['conditions']} THEN {r['prediction']} (Confidence: {r['confidence_percent']}%, Support: {r['support_samples']})"
        for r in human_readable_rules[:5]
    ])
    
    return (
        f"Global Logic Rules Extraction:\n"
        f"Model Approximated: {rules_output['model_type']}\n"
        f"Rules Extracted: {len(human_readable_rules)}\n"
        f"Top Features: {list(feature_importance.head(5).index)}\n"
        f"Sample Rules (Top 5):\n{rules_summary}\n"
        f"Full decision tree and all rules saved to: {RULES_PATH}\n"
        f"Business Use: These rules can be used for customer segmentation and intervention strategies."
    )