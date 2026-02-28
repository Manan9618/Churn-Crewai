import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from crewai.tools import tool

MODEL_PATH = "artifacts/model/churn_model.pkl"
PREDICTIONS_PATH = "artifacts/data/predictions.csv"
BUSINESS_METRICS_PATH = "artifacts/model/business_metrics_report.json"
ADVERSARIAL_PATH = "artifacts/model/adversarial_robustness_report.json"
FAIRNESS_DISPARITY_PATH = "artifacts/model/fairness_disparity_report.json"
PROBABILITY_DIST_PATH = "artifacts/model/probability_distribution_report.json"

# Business cost assumptions (adjust based on your business context)
CUSTOMER_LIFETIME_VALUE = 500  # Average lifetime value per customer
CHURN_COST = 300  # Cost of losing a customer
RETENTION_CAMPAIGN_COST = 50  # Cost per retention campaign
FALSE_NEGATIVE_COST = 400  # Cost of missing a churner
FALSE_POSITIVE_COST = 50  # Cost of false alarm

# Fairness thresholds
MAX_FPR_DISPARITY = 0.10  # Max 10% difference in FPR between groups

# Probability distribution thresholds
MIN_ENTROPY_THRESHOLD = 0.5  # Minimum entropy for well-distributed probabilities
MAX_CONFIDENCE_RATIO = 0.80  # Max 80% of predictions should be >0.9 or <0.1


@tool("Validate Model Metrics")
def validate_model_metrics_tool(min_auc: float = 0.75, min_f1: float = 0.55) -> str:
    """Validate that the model meets minimum AUC and F1 thresholds on the test set."""
    from sklearn.metrics import roc_auc_score, f1_score

    for p in [MODEL_PATH, "artifacts/data/X_test.csv", "artifacts/data/y_test.csv"]:
        if not os.path.exists(p):
            return f"VALIDATION FAILED: Required file missing: {p}"

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    X_te = pd.read_csv("artifacts/data/X_test.csv")
    y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    f1 = f1_score(y_te, y_pred)

    errors = []
    if auc < min_auc:
        errors.append(f"AUC {auc:.4f} < minimum {min_auc}")
    if f1 < min_f1:
        errors.append(f"F1 {f1:.4f} < minimum {min_f1}")

    if errors:
        return "VALIDATION FAILED:\n" + "\n".join(errors)
    return f"Model metrics validation PASSED. AUC={auc:.4f}, F1={f1:.4f}."


@tool("Validate No Overfitting")
def validate_no_overfitting_tool(max_gap: float = 0.05) -> str:
    """Check that train-test AUC gap does not exceed max_gap."""
    from sklearn.metrics import roc_auc_score

    for p in [MODEL_PATH, "artifacts/data/X_train.csv", "artifacts/data/X_test.csv", "artifacts/data/y_train.csv", "artifacts/data/y_test.csv"]:
        if not os.path.exists(p):
            return f"VALIDATION FAILED: Required file missing: {p}"

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X_tr = pd.read_csv("artifacts/data/X_train.csv")
    y_tr = pd.read_csv("artifacts/data/y_train.csv").squeeze()
    X_te = pd.read_csv("artifacts/data/X_test.csv")
    y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()

    train_auc = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
    test_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    gap = train_auc - test_auc

    if gap > max_gap:
        return f"VALIDATION FAILED: Overfitting detected. Train AUC={train_auc:.4f}, Test AUC={test_auc:.4f}, Gap={gap:.4f} > {max_gap}"
    return f"Overfitting check PASSED. Train AUC={train_auc:.4f}, Test AUC={test_auc:.4f}, Gap={gap:.4f}."


@tool("Validate Model File Exists")
def validate_model_file_exists_tool() -> str:
    """Check that model file exists and is loadable."""
    if not os.path.exists(MODEL_PATH):
        return f"VALIDATION FAILED: Model file not found at {MODEL_PATH}."
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return f"Model file validation PASSED. Model type: {type(model).__name__}."
    except Exception as e:
        return f"VALIDATION FAILED: Cannot load model: {e}"


@tool("Validate Confusion Matrix")
def validate_confusion_matrix_tool(min_recall_churn: float = 0.60) -> str:
    """Validate that recall on the churn class (minority) is at least min_recall_churn."""
    from sklearn.metrics import recall_score, confusion_matrix

    for p in [MODEL_PATH, "artifacts/data/X_test.csv", "artifacts/data/y_test.csv"]:
        if not os.path.exists(p):
            return f"VALIDATION FAILED: Required file missing: {p}"

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    X_te = pd.read_csv("artifacts/data/X_test.csv")
    y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
    y_pred = model.predict(X_te)

    recall_churn = recall_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)

    if recall_churn < min_recall_churn:
        return (
            f"VALIDATION FAILED: Churn recall {recall_churn:.4f} < {min_recall_churn}.\n"
            f"Confusion Matrix:\n{cm}"
        )
    return (
        f"Confusion matrix validation PASSED. Churn recall={recall_churn:.4f}.\n"
        f"Confusion Matrix:\n{cm}"
    )


@tool("Validate Predictions File")
def validate_predictions_file_tool() -> str:
    """Validate that predictions file exists and has no missing values."""
    if not os.path.exists(PREDICTIONS_PATH):
        return f"VALIDATION FAILED: Predictions file not found at {PREDICTIONS_PATH}."
    df = pd.read_csv(PREDICTIONS_PATH)
    missing = df[["Churn_Predicted", "Churn_Probability"]].isnull().sum().sum()
    if missing > 0:
        return f"VALIDATION FAILED: {missing} missing prediction values."
    return f"Predictions file PASSED. Shape: {df.shape}, no missing values."


@tool("Validate Prediction Columns")
def validate_prediction_columns_tool() -> str:
    """Validate that required columns exist in predictions file."""
    required = ["Churn_Predicted", "Churn_Probability"]
    if not os.path.exists(PREDICTIONS_PATH):
        return f"VALIDATION FAILED: {PREDICTIONS_PATH} not found."
    df = pd.read_csv(PREDICTIONS_PATH)
    missing = [c for c in required if c not in df.columns]
    if missing:
        return f"VALIDATION FAILED: Missing columns: {missing}"
    return f"Prediction columns PASSED. Found: {required}"


@tool("Validate Churn Rate Range")
def validate_churn_rate_range_tool(min_rate: float = 0.10, max_rate: float = 0.35) -> str:
    """Validate that predicted churn rate is in the expected 10-35% range."""
    if not os.path.exists(PREDICTIONS_PATH):
        return f"VALIDATION FAILED: {PREDICTIONS_PATH} not found."
    df = pd.read_csv(PREDICTIONS_PATH)
    rate = df["Churn_Predicted"].mean()
    if not (min_rate <= rate <= max_rate):
        return f"VALIDATION FAILED: Predicted churn rate {rate:.2%} outside expected range [{min_rate:.0%}, {max_rate:.0%}]."
    return f"Churn rate validation PASSED. Predicted churn rate: {rate:.2%}."


@tool("Validate Probability Range")
def validate_probability_range_tool() -> str:
    """Validate that all Churn_Probability values are in [0, 1]."""
    if not os.path.exists(PREDICTIONS_PATH):
        return f"VALIDATION FAILED: {PREDICTIONS_PATH} not found."
    df = pd.read_csv(PREDICTIONS_PATH)
    invalid = df[(df["Churn_Probability"] < 0) | (df["Churn_Probability"] > 1)]
    if not invalid.empty:
        return f"VALIDATION FAILED: {len(invalid)} probabilities outside [0,1]."
    return "Probability range validation PASSED. All probabilities in [0, 1]."


# ── NEW MODEL VALIDATION TOOLS ADDED BELOW ────────────────────────────────────


@tool("Validate Business Metrics")
def validate_business_metrics_tool(
    customer_lifetime_value: float = CUSTOMER_LIFETIME_VALUE,
    churn_cost: float = CHURN_COST,
    retention_campaign_cost: float = RETENTION_CAMPAIGN_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    min_profit_per_customer: float = 0.0
) -> str:
    """
    Checks custom business metrics (e.g., Profit per Customer) not just Accuracy/F1.
    Translates model performance into financial impact for stakeholders.
    """
    from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
    
    for p in [MODEL_PATH, "artifacts/data/X_test.csv", "artifacts/data/y_test.csv"]:
        if not os.path.exists(p):
            return f"VALIDATION FAILED: Required file missing: {p}"
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    X_te = pd.read_csv("artifacts/data/X_test.csv")
    y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
    
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    total_customers = len(y_te)
    
    # Calculate business costs and savings
    # False Negatives: Missed churners (most costly)
    fn_cost = fn * false_negative_cost
    
    # False Positives: Unnecessary retention campaigns
    fp_cost = fp * false_positive_cost
    
    # True Positives: Successfully retained customers
    tp_savings = tp * (churn_cost - retention_campaign_cost)
    
    # True Negatives: Correctly identified non-churners (no action needed)
    tn_savings = tn * 0
    
    # Total impact
    total_cost = fn_cost + fp_cost
    total_savings = tp_savings + tn_savings
    net_profit = total_savings - total_cost
    profit_per_customer = net_profit / total_customers
    
    # Calculate additional business metrics
    auc = roc_auc_score(y_te, y_prob)
    f1 = f1_score(y_te, y_pred)
    
    # ROI calculation
    total_investment = fp * retention_campaign_cost  # Cost of campaigns
    roi = ((total_savings - total_investment) / total_investment * 100) if total_investment > 0 else 0
    
    # Save business metrics report
    business_report = {
        "timestamp": datetime.now().isoformat(),
        "test_set_size": total_customers,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "cost_breakdown": {
            "false_negative_cost": round(fn_cost, 2),
            "false_positive_cost": round(fp_cost, 2),
            "true_positive_savings": round(tp_savings, 2),
            "total_cost": round(total_cost, 2),
            "total_savings": round(total_savings, 2),
            "net_profit": round(net_profit, 2),
            "profit_per_customer": round(profit_per_customer, 4)
        },
        "cost_assumptions": {
            "customer_lifetime_value": customer_lifetime_value,
            "churn_cost": churn_cost,
            "retention_campaign_cost": retention_campaign_cost,
            "false_negative_cost": false_negative_cost,
            "false_positive_cost": false_positive_cost
        },
        "model_metrics": {
            "auc": round(auc, 4),
            "f1": round(f1, 4),
            "accuracy": round((tn + tp) / total_customers, 4)
        },
        "roi_percent": round(roi, 2),
        "meets_profit_threshold": profit_per_customer >= min_profit_per_customer
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    with open(BUSINESS_METRICS_PATH, "w") as f:
        json.dump(business_report, f, indent=2)
    
    # Determine validation status
    if profit_per_customer < min_profit_per_customer:
        status = "❌ FAIL"
        recommendation = f"Model generates ${profit_per_customer:.4f}/customer (below ${min_profit_per_customer} threshold). Consider improvements."
    else:
        status = "✅ PASS"
        recommendation = f"Model generates ${profit_per_customer:.4f}/customer profit. Ready for deployment."
    
    return (
        f"Business Metrics Validation:\n"
        f"Status: {status}\n"
        f"Test Set Size: {total_customers} customers\n"
        f"\nConfusion Matrix:\n"
        f"  True Negatives:  {tn} (correctly identified non-churners)\n"
        f"  False Positives: {fp} (unnecessary campaigns)\n"
        f"  False Negatives: {fn} (missed churners - HIGH COST)\n"
        f"  True Positives:  {tp} (successfully identified churners)\n"
        f"\nFinancial Impact:\n"
        f"  False Negative Cost:  ${fn_cost:,.2f} ({fn} × ${false_negative_cost})\n"
        f"  False Positive Cost:  ${fp_cost:,.2f} ({fp} × ${false_positive_cost})\n"
        f"  True Positive Savings: ${tp_savings:,.2f}\n"
        f"  ─────────────────────────────────────\n"
        f"  Total Cost:           ${total_cost:,.2f}\n"
        f"  Total Savings:        ${total_savings:,.2f}\n"
        f"  NET PROFIT:           ${net_profit:,.2f}\n"
        f"  PROFIT PER CUSTOMER:  ${profit_per_customer:.4f}\n"
        f"  ROI:                  {roi:.1f}%\n"
        f"\nProfit Threshold: ${min_profit_per_customer}\n"
        f"Recommendation: {recommendation}\n"
        f"Report saved to: {BUSINESS_METRICS_PATH}"
    )


@tool("Validate Adversarial Robustness")
def validate_adversarial_robustness_tool(
    perturbation_magnitude: float = 0.1,
    n_samples: int = 100,
    max_accuracy_drop: float = 0.05
) -> str:
    """
    Tests the model against slightly perturbed inputs to ensure it doesn't break easily.
    Simulates real-world data noise and potential adversarial attacks.
    """
    from sklearn.metrics import accuracy_score
    
    for p in [MODEL_PATH, "artifacts/data/X_test.csv", "artifacts/data/y_test.csv"]:
        if not os.path.exists(p):
            return f"VALIDATION FAILED: Required file missing: {p}"
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    X_te = pd.read_csv("artifacts/data/X_test.csv")
    y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
    
    # Select samples for testing
    n_samples = min(n_samples, len(X_te))
    X_sample = X_te.iloc[:n_samples].copy()
    y_sample = y_te.iloc[:n_samples]
    
    # Calculate baseline accuracy
    y_pred_baseline = model.predict(X_sample)
    baseline_accuracy = accuracy_score(y_sample, y_pred_baseline)
    
    # Test robustness against perturbations
    robustness_results = []
    np.random.seed(42)
    
    for i in range(5):  # Test 5 different perturbation levels
        perturbation = perturbation_magnitude * (i + 1) / 5
        
        # Add noise to features
        X_perturbed = X_sample.copy()
        noise = np.random.normal(0, perturbation, X_perturbed.shape)
        X_perturbed = X_perturbed + noise
        
        # Clip to valid range (0-1 for scaled features)
        X_perturbed = X_perturbed.clip(-3, 3)  # Assuming standardized features
        
        # Predict on perturbed data
        y_pred_perturbed = model.predict(X_perturbed)
        perturbed_accuracy = accuracy_score(y_sample, y_pred_perturbed)
        
        accuracy_drop = baseline_accuracy - perturbed_accuracy
        
        robustness_results.append({
            "perturbation_magnitude": round(perturbation, 4),
            "baseline_accuracy": round(baseline_accuracy, 4),
            "perturbed_accuracy": round(perturbed_accuracy, 4),
            "accuracy_drop": round(accuracy_drop, 4),
            "robust": accuracy_drop <= max_accuracy_drop
        })
    
    # Calculate robustness score (average accuracy retention)
    avg_accuracy_retention = np.mean([1 - r["accuracy_drop"] for r in robustness_results])
    all_robust = all(r["robust"] for r in robustness_results)
    
    # Save robustness report
    robustness_report = {
        "timestamp": datetime.now().isoformat(),
        "n_samples_tested": n_samples,
        "baseline_accuracy": round(baseline_accuracy, 4),
        "perturbation_magnitude_range": [perturbation_magnitude * 0.2, perturbation_magnitude],
        "max_allowed_accuracy_drop": max_accuracy_drop,
        "robustness_results": robustness_results,
        "average_accuracy_retention": round(avg_accuracy_retention, 4),
        "all_tests_robust": all_robust
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    with open(ADVERSARIAL_PATH, "w") as f:
        json.dump(robustness_report, f, indent=2)
    
    # Format output
    results_text = "\n".join([
        f"  Perturbation {r['perturbation_magnitude']:.4f}: "
        f"Accuracy {r['perturbed_accuracy']:.4f} (drop: {r['accuracy_drop']:.4f}) "
        f"{'✅' if r['robust'] else '❌'}"
        for r in robustness_results
    ])
    
    if all_robust:
        status = "✅ ROBUST"
        recommendation = "Model is stable against input perturbations. Safe for production."
    else:
        status = "⚠️ NOT ROBUST"
        recommendation = "Model shows sensitivity to input noise. Consider regularization or ensemble methods."
    
    return (
        f"Adversarial Robustness Validation:\n"
        f"Status: {status}\n"
        f"Samples Tested: {n_samples}\n"
        f"Baseline Accuracy: {baseline_accuracy:.4f}\n"
        f"Max Allowed Accuracy Drop: {max_accuracy_drop}\n"
        f"\nRobustness Test Results:\n{results_text}\n"
        f"\nAverage Accuracy Retention: {avg_accuracy_retention:.4f}\n"
        f"Recommendation: {recommendation}\n"
        f"Report saved to: {ADVERSARIAL_PATH}"
    )


@tool("Validate Fairness Disparity")
def validate_fairness_disparity_tool(
    max_fpr_disparity: float = MAX_FPR_DISPARITY,
    protected_attributes: str = "gender,SeniorCitizen"
) -> str:
    """
    Ensures False Positive Rates are similar across different demographic groups.
    Checks for bias in model predictions across protected attributes.
    
    Args:
        max_fpr_disparity: Maximum allowed FPR difference between groups (default: 0.10)
        protected_attributes: Comma-separated list of attribute names (e.g., "gender,SeniorCitizen")
    """
    """
    Ensures False Positive Rates are similar across different demographic groups.
    Checks for bias in model predictions across protected attributes.
    """
    from sklearn.metrics import false_positive_rate
    
    for p in [MODEL_PATH, "artifacts/data/X_test.csv", "artifacts/data/y_test.csv", PREDICTIONS_PATH]:
        if not os.path.exists(p):
            return f"VALIDATION FAILED: Required file missing: {p}"
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    X_te = pd.read_csv("artifacts/data/X_test.csv")
    y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    
    y_pred = predictions_df["Churn_Predicted"].values
    
    # Parse protected_attributes from comma-separated string
    if not protected_attributes or protected_attributes.strip() == "":
        protected_attributes_list = ["gender", "SeniorCitizen"]
    else:
        protected_attributes_list = [attr.strip() for attr in protected_attributes.split(",") if attr.strip()]
    
    # Check which protected attributes exist in the data
    available_attrs = [attr for attr in protected_attributes_list if attr in X_te.columns]
    
    if not available_attrs:
        return (
            f"VALIDATION SKIPPED: No protected attributes found.\n"
            f"Checked for: {protected_attributes_list}\n"
            f"Available columns: {list(X_te.columns)}"
        )
    
    fairness_results = []
    violations = []
    
    for attr in available_attrs:
        groups = X_te[attr].unique()
        
        if len(groups) < 2:
            continue
        
        group_metrics = []
        
        for group_val in groups:
            mask = X_te[attr] == group_val
            y_true_group = y_te[mask]
            y_pred_group = y_pred[mask]
            
            # Calculate False Positive Rate for this group
            # FPR = FP / (FP + TN) - rate of incorrectly predicting churn for non-churners
            tn = ((y_true_group == 0) & (y_pred_group == 0)).sum()
            fp = ((y_true_group == 0) & (y_pred_group == 1)).sum()
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            group_metrics.append({
                "group_value": group_val,
                "sample_size": int(mask.sum()),
                "false_positive_rate": round(fpr, 4)
            })
        
        # Calculate disparity
        fpr_values = [g["false_positive_rate"] for g in group_metrics]
        fpr_disparity = max(fpr_values) - min(fpr_values)
        
        is_fair = fpr_disparity <= max_fpr_disparity
        
        fairness_results.append({
            "attribute": attr,
            "groups": group_metrics,
            "fpr_disparity": round(fpr_disparity, 4),
            "max_fpr": round(max(fpr_values), 4),
            "min_fpr": round(min(fpr_values), 4),
            "is_fair": is_fair
        })
        
        if not is_fair:
            violations.append({
                "attribute": attr,
                "fpr_disparity": round(fpr_disparity, 4),
                "threshold": max_fpr_disparity
            })
    
    # Save fairness report
    fairness_report = {
        "timestamp": datetime.now().isoformat(),
        "protected_attributes_tested": available_attrs,
        "protected_attributes_input": protected_attributes_list,
        "max_allowed_fpr_disparity": max_fpr_disparity,
        "fairness_results": fairness_results,
        "violations": violations,
        "all_fair": len(violations) == 0
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    with open(FAIRNESS_DISPARITY_PATH, "w") as f:
        json.dump(fairness_report, f, indent=2)
    
    # Format output
    results_text = ""
    for result in fairness_results:
        results_text += (
            f"\n  {result['attribute']}:\n"
            f"    FPR Disparity: {result['fpr_disparity']:.4f} "
            f"{'✅' if result['is_fair'] else '❌'}\n"
            f"    Groups:\n"
        )
        for group in result["groups"]:
            results_text += (
                f"      {group['group_value']}: FPR={group['false_positive_rate']:.4f}, "
                f"N={group['sample_size']}\n"
            )
    
    if violations:
        status = "❌ FAIRNESS VIOLATIONS DETECTED"
        violations_text = "\n".join([
            f"  ⚠️ {v['attribute']}: FPR disparity {v['fpr_disparity']:.4f} > {v['threshold']}"
            for v in violations
        ])
        recommendation = "Review model for bias. Consider fairness-aware training or post-processing."
    else:
        status = "✅ FAIR"
        violations_text = "No fairness violations detected."
        recommendation = "Model shows fair treatment across demographic groups."
    
    return (
        f"Fairness Disparity Validation:\n"
        f"Status: {status}\n"
        f"Protected Attributes Tested: {available_attrs}\n"
        f"Max Allowed FPR Disparity: {max_fpr_disparity}\n"
        f"\nFairness Results:{results_text}\n"
        f"\nViolations:\n{violations_text}\n"
        f"Recommendation: {recommendation}\n"
        f"Report saved to: {FAIRNESS_DISPARITY_PATH}"
    )


@tool("Validate Probability Distribution")
def validate_probability_distribution_tool(
    min_entropy: float = MIN_ENTROPY_THRESHOLD,
    max_confidence_ratio: float = MAX_CONFIDENCE_RATIO
) -> str:
    """
    Checks that predicted probabilities aren't all clustered near 0 or 1.
    Overconfident predictions (all near 0 or 1) indicate poor calibration.
    """
    from scipy.stats import entropy
    
    if not os.path.exists(PREDICTIONS_PATH):
        return f"VALIDATION FAILED: {PREDICTIONS_PATH} not found."
    
    df = pd.read_csv(PREDICTIONS_PATH)
    
    if "Churn_Probability" not in df.columns:
        return f"VALIDATION FAILED: Churn_Probability column not found in predictions."
    
    probabilities = df["Churn_Probability"].values
    
    # Calculate distribution statistics
    mean_prob = probabilities.mean()
    std_prob = probabilities.std()
    min_prob = probabilities.min()
    max_prob = probabilities.max()
    median_prob = np.median(probabilities)
    
    # Calculate entropy (measure of uncertainty/diversity in predictions)
    # Bin probabilities into 10 bins for entropy calculation
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(probabilities, bins=bins)
    hist_normalized = hist / hist.sum()
    prob_entropy = entropy(hist_normalized, base=2)  # Binary entropy
    max_entropy = np.log2(len(bins) - 1)  # Maximum possible entropy
    normalized_entropy = prob_entropy / max_entropy if max_entropy > 0 else 0
    
    # Check for overconfidence (too many predictions near 0 or 1)
    very_confident = ((probabilities < 0.1) | (probabilities > 0.9)).sum()
    confidence_ratio = very_confident / len(probabilities)
    
    # Check for underconfidence (too many predictions near 0.5)
    uncertain = ((probabilities > 0.4) & (probabilities < 0.6)).sum()
    uncertain_ratio = uncertain / len(probabilities)
    
    # Determine validation status
    entropy_ok = normalized_entropy >= min_entropy
    confidence_ok = confidence_ratio <= max_confidence_ratio
    
    all_ok = entropy_ok and confidence_ok
    
    # Save probability distribution report
    distribution_report = {
        "timestamp": datetime.now().isoformat(),
        "n_predictions": len(probabilities),
        "statistics": {
            "mean": round(float(mean_prob), 4),
            "std": round(float(std_prob), 4),
            "min": round(float(min_prob), 4),
            "max": round(float(max_prob), 4),
            "median": round(float(median_prob), 4)
        },
        "entropy": {
            "raw_entropy": round(float(prob_entropy), 4),
            "max_entropy": round(float(max_entropy), 4),
            "normalized_entropy": round(float(normalized_entropy), 4),
            "min_threshold": min_entropy,
            "passes": entropy_ok
        },
        "confidence_analysis": {
            "very_confident_count": int(very_confident),
            "very_confident_ratio": round(float(confidence_ratio), 4),
            "uncertain_count": int(uncertain),
            "uncertain_ratio": round(float(uncertain_ratio), 4),
            "max_confidence_ratio": max_confidence_ratio,
            "passes": confidence_ok
        },
        "distribution_bins": {
            "0.0-0.1": int((probabilities < 0.1).sum()),
            "0.1-0.2": int(((probabilities >= 0.1) & (probabilities < 0.2)).sum()),
            "0.2-0.3": int(((probabilities >= 0.2) & (probabilities < 0.3)).sum()),
            "0.3-0.4": int(((probabilities >= 0.3) & (probabilities < 0.4)).sum()),
            "0.4-0.5": int(((probabilities >= 0.4) & (probabilities < 0.5)).sum()),
            "0.5-0.6": int(((probabilities >= 0.5) & (probabilities < 0.6)).sum()),
            "0.6-0.7": int(((probabilities >= 0.6) & (probabilities < 0.7)).sum()),
            "0.7-0.8": int(((probabilities >= 0.7) & (probabilities < 0.8)).sum()),
            "0.8-0.9": int(((probabilities >= 0.8) & (probabilities < 0.9)).sum()),
            "0.9-1.0": int((probabilities >= 0.9).sum())
        },
        "all_valid": all_ok
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    with open(PROBABILITY_DIST_PATH, "w") as f:
        json.dump(distribution_report, f, indent=2)
    
    # Format output
    bins_text = "\n".join([
        f"  {bin_range}: {count} ({count/len(probabilities)*100:.1f}%)"
        for bin_range, count in distribution_report["distribution_bins"].items()
    ])
    
    if all_ok:
        status = "✅ WELL-CALIBRATED"
        recommendation = "Probability distribution is healthy. Model is well-calibrated."
    else:
        status = "⚠️ CALIBRATION ISSUES"
        issues = []
        if not entropy_ok:
            issues.append(f"Low entropy ({normalized_entropy:.4f} < {min_entropy})")
        if not confidence_ok:
            issues.append(f"Overconfident ({confidence_ratio*100:.1f}% > {max_confidence_ratio*100}%)")
        recommendation = f"Issues detected: {', '.join(issues)}. Consider probability calibration."
    
    return (
        f"Probability Distribution Validation:\n"
        f"Status: {status}\n"
        f"Total Predictions: {len(probabilities)}\n"
        f"\nDistribution Statistics:\n"
        f"  Mean:   {mean_prob:.4f}\n"
        f"  Std:    {std_prob:.4f}\n"
        f"  Min:    {min_prob:.4f}\n"
        f"  Max:    {max_prob:.4f}\n"
        f"  Median: {median_prob:.4f}\n"
        f"\nEntropy Analysis:\n"
        f"  Normalized Entropy: {normalized_entropy:.4f} (min: {min_entropy}) "
        f"{'✅' if entropy_ok else '❌'}\n"
        f"\nConfidence Analysis:\n"
        f"  Overconfident (>0.9 or <0.1): {very_confident} ({confidence_ratio*100:.1f}%) "
        f"{'✅' if confidence_ok else '❌'}\n"
        f"  Uncertain (0.4-0.6): {uncertain} ({uncertain_ratio*100:.1f}%)\n"
        f"\nProbability Distribution:\n{bins_text}\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {PROBABILITY_DIST_PATH}"
    )