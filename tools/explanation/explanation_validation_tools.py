import pandas as pd
import numpy as np
import pickle
import json
import os
from crewai.tools import tool

SHAP_PATH = "artifacts/model/shap_values.pkl"
EXPLANATION_REPORT_PATH = "artifacts/model/explanation_report.json"
FAIRNESS_REPORT_PATH = "artifacts/model/fairness_report.json"
COUNTERFACTUAL_PATH = "artifacts/model/counterfactuals.json"
LIME_PATH = "artifacts/model/lime_explanations.json"
EXPECTED_TOP_FEATURES = {"Contract", "tenure", "MonthlyCharges", "TotalCharges", "InternetService"}

# Validation thresholds
FAIRNESS_CHURN_RATE_THRESHOLD = 10.0  # Max % difference in churn rate between groups
FAIRNESS_RECALL_THRESHOLD = 0.15  # Max difference in recall between groups
COUNTERFACTUAL_AGREEMENT_THRESHOLD = 2  # Min SHAP/LIME top features that must match


@tool("Validate SHAP Values")
def validate_shap_values_tool() -> str:
    """Validate that SHAP values file exists, has correct shape, and contains no NaNs."""
    if not os.path.exists(SHAP_PATH):
        return f"VALIDATION FAILED: SHAP values file not found at {SHAP_PATH}."
    with open(SHAP_PATH, "rb") as f:
        data = pickle.load(f)
    sv = data["shap_values"]
    features = data["feature_names"]
    if sv.shape[1] != len(features):
        return f"VALIDATION FAILED: SHAP shape {sv.shape} doesn't match {len(features)} features."
    nan_count = np.isnan(sv).sum()
    if nan_count > 0:
        return f"VALIDATION FAILED: {nan_count} NaN values in SHAP matrix."
    return f"SHAP values validation PASSED. Shape: {sv.shape}, No NaNs."


@tool("Validate Top Features in SHAP")
def validate_top_features_in_shap_tool(top_n: int = 5, min_overlap: int = 2) -> str:
    """
    Validate that known domain-important features appear in the top SHAP features.
    Requires at least min_overlap features from EXPECTED_TOP_FEATURES in the top_n.
    """
    if not os.path.exists(SHAP_PATH):
        return f"VALIDATION FAILED: SHAP values file not found at {SHAP_PATH}."
    with open(SHAP_PATH, "rb") as f:
        data = pickle.load(f)
    sv = data["shap_values"]
    features = data["feature_names"]
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=features).sort_values(ascending=False)
    top_features = set(mean_abs.head(top_n).index)
    overlap = top_features & EXPECTED_TOP_FEATURES
    if len(overlap) < min_overlap:
        return (
            f"VALIDATION FAILED: Only {len(overlap)} expected features in top {top_n} SHAP.\n"
            f"Top SHAP: {list(top_features)}\nExpected to find: {EXPECTED_TOP_FEATURES}"
        )
    return f"Top SHAP features validation PASSED. Domain features in top {top_n}: {overlap}"


@tool("Validate Explanation Report")
def validate_explanation_report_tool() -> str:
    """Validate that the explanation report exists and has required keys."""
    if not os.path.exists(EXPLANATION_REPORT_PATH):
        return f"VALIDATION FAILED: Report not found at {EXPLANATION_REPORT_PATH}."
    with open(EXPLANATION_REPORT_PATH) as f:
        report = json.load(f)
    required_keys = ["global_feature_importance_shap", "top_churn_drivers"]
    missing = [k for k in required_keys if k not in report]
    if missing:
        return f"VALIDATION FAILED: Missing keys in report: {missing}"
    return f"Explanation report validation PASSED. Top drivers: {report['top_churn_drivers']}"


# ── NEW EXPLANATION VALIDATION TOOLS ADDED BELOW ──────────────────────────────


@tool("Validate Consistency SHAP LIME")
def validate_consistency_shap_lime_tool(top_n: int = 3, min_agreement: int = 2) -> str:
    """
    Ensures SHAP and LIME agree on the top features.
    If they disagree wildly, the model might be unstable.
    Requires at least min_agreement features in common between top_n of both methods.
    """
    # Load SHAP top features
    if not os.path.exists(SHAP_PATH):
        return f"VALIDATION FAILED: SHAP values file not found at {SHAP_PATH}."
    
    with open(SHAP_PATH, "rb") as f:
        shap_data = pickle.load(f)
    
    shap_sv = shap_data["shap_values"]
    shap_features = shap_data["feature_names"]
    shap_mean_abs = pd.Series(np.abs(shap_sv).mean(axis=0), index=shap_features).sort_values(ascending=False)
    shap_top = set(shap_mean_abs.head(top_n).index)
    
    # Load LIME top features from explanation report or separate file
    lime_top = set()
    
    if os.path.exists(LIME_PATH):
        with open(LIME_PATH, "r") as f:
            lime_data = json.load(f)
        # Extract top features from LIME explanations
        all_lime_features = []
        for explanation in lime_data.get("explanations", []):
            features = [item[0] for item in explanation.get("feature_contributions", [])]
            all_lime_features.extend(features)
        
        # Get most frequent LIME features
        if all_lime_features:
            lime_freq = pd.Series(all_lime_features).value_counts()
            lime_top = set(lime_freq.head(top_n).index)
    
    elif os.path.exists(EXPLANATION_REPORT_PATH):
        with open(EXPLANATION_REPORT_PATH, "r") as f:
            report = json.load(f)
        
        # Try to extract LIME features from report
        if "lime_top_features" in report:
            lime_top = set(report["lime_top_features"][:top_n])
    
    if not lime_top:
        return (
            f"VALIDATION SKIPPED: LIME explanations not found.\n"
            f"Cannot validate SHAP-LIME consistency without LIME data.\n"
            f"Run lime_explanation_tool first and save to {LIME_PATH}"
        )
    
    # Calculate agreement
    agreement = shap_top & lime_top
    agreement_count = len(agreement)
    agreement_pct = (agreement_count / top_n) * 100
    
    if agreement_count < min_agreement:
        return (
            f"VALIDATION FAILED: SHAP and LIME disagree on top {top_n} features.\n"
            f"Agreement: {agreement_count}/{top_n} features ({agreement_pct:.1f}%)\n"
            f"SHAP Top {top_n}: {list(shap_top)}\n"
            f"LIME Top {top_n}: {list(lime_top)}\n"
            f"Common Features: {agreement}\n"
            f"WARNING: Model explanations may be unstable. Investigate further."
        )
    
    return (
        f"SHAP-LIME Consistency Validation PASSED.\n"
        f"Agreement: {agreement_count}/{top_n} features ({agreement_pct:.1f}%)\n"
        f"SHAP Top {top_n}: {list(shap_top)}\n"
        f"LIME Top {top_n}: {list(lime_top)}\n"
        f"Common Features: {agreement}\n"
        f"Interpretation: Explanations are consistent across methods."
    )


@tool("Validate Fairness Metrics")
def validate_fairness_metrics_tool(
    churn_rate_threshold: float = FAIRNESS_CHURN_RATE_THRESHOLD,
    recall_threshold: float = FAIRNESS_RECALL_THRESHOLD
) -> str:
    """
    Checks that explanation bias scores are below defined thresholds.
    Validates fairness across protected groups (Gender, SeniorCitizen).
    """
    if not os.path.exists(FAIRNESS_REPORT_PATH):
        return (
            f"VALIDATION FAILED: Fairness report not found at {FAIRNESS_REPORT_PATH}.\n"
            f"Run analyze_fairness_bias_tool first."
        )
    
    with open(FAIRNESS_REPORT_PATH, "r") as f:
        fairness_data = json.load(f)
    
    violations = []
    group_summary = []
    
    # Check for detected bias
    bias_detected = fairness_data.get("bias_detected", [])
    
    if bias_detected:
        for bias in bias_detected:
            violations.append(
                f"  ⚠️ {bias['attribute']}: {bias['type']} - {bias['details']}"
            )
    
    # Validate group metrics
    group_metrics = fairness_data.get("group_metrics", {})
    
    for attr_name, groups in group_metrics.items():
        group_rates = []
        
        for group_key, metrics in groups.items():
            if group_key.startswith("group_"):
                churn_rate = metrics.get("churn_rate", 0)
                recall = metrics.get("recall", 0)
                group_rates.append({
                    "group": group_key,
                    "churn_rate": churn_rate,
                    "recall": recall
                })
                group_summary.append(
                    f"  {attr_name} - {group_key}: Churn Rate={churn_rate}%, Recall={recall}"
                )
        
        # Check disparities between groups
        if len(group_rates) >= 2:
            churn_rates = [g["churn_rate"] for g in group_rates]
            recalls = [g["recall"] for g in group_rates]
            
            churn_diff = max(churn_rates) - min(churn_rates)
            recall_diff = max(recalls) - min(recalls)
            
            if churn_diff > churn_rate_threshold:
                violations.append(
                    f"  ⚠️ {attr_name}: Churn rate disparity {churn_diff:.2f}% > threshold {churn_rate_threshold}%"
                )
            
            if recall_diff > recall_threshold:
                violations.append(
                    f"  ⚠️ {attr_name}: Recall disparity {recall_diff:.4f} > threshold {recall_threshold}"
                )
    
    if violations:
        return (
            f"VALIDATION FAILED: Fairness threshold violations detected.\n"
            f"Thresholds: Churn Rate Diff ≤ {churn_rate_threshold}%, Recall Diff ≤ {recall_threshold}\n"
            f"Violations:\n" + "\n".join(violations) + "\n"
            f"Group Metrics:\n" + "\n".join(group_summary) + "\n"
            f"Recommendation: Investigate bias sources and consider fairness-aware training."
        )
    
    return (
        f"Fairness Metrics Validation PASSED.\n"
        f"Thresholds: Churn Rate Diff ≤ {churn_rate_threshold}%, Recall Diff ≤ {recall_threshold}\n"
        f"Protected Attributes Analyzed: {list(group_metrics.keys())}\n"
        f"Group Metrics:\n" + "\n".join(group_summary) + "\n"
        f"Status: ✅ FAIR - No significant bias detected across protected groups."
    )


@tool("Validate Counterfactual Feasibility")
def validate_counterfactual_feasibility_tool() -> str:
    """
    Ensures suggested counterfactuals are actually possible.
    Checks that feature changes are realistic (e.g., tenure can increase but not go negative,
    age cannot decrease, binary fields can only flip 0↔1).
    """
    if not os.path.exists(COUNTERFACTUAL_PATH):
        return (
            f"VALIDATION FAILED: Counterfactuals file not found at {COUNTERFACTUAL_PATH}.\n"
            f"Run generate_counterfactuals_tool first."
        )
    
    with open(COUNTERFACTUAL_PATH, "r") as f:
        counterfactuals = json.load(f)
    
    if not counterfactuals:
        return "VALIDATION FAILED: No counterfactuals found in file."
    
    # Define feasibility rules for Telco dataset features
    feasibility_rules = {
        "tenure": {"min_change": 0, "direction": "increase_only"},  # Can only increase tenure
        "SeniorCitizen": {"direction": "immutable"},  # Cannot change age group
        "gender": {"direction": "immutable"},  # Cannot change gender
        "Contract": {"allowed_values": [0, 1, 2]},  # Month-to-month, 1 year, 2 year
        "PaymentMethod": {"allowed_values": [0, 1, 2, 3]},  # Categorical
        "InternetService": {"allowed_values": [0, 1, 2]},  # None, DSL, Fiber
    }
    
    infeasible_changes = []
    feasible_count = 0
    
    for cf in counterfactuals:
        customer_idx = cf.get("customer_index", "unknown")
        changed_features = cf.get("changed_features", {})
        
        for feat, change_data in changed_features.items():
            original = change_data.get("original", 0)
            counterfactual = change_data.get("counterfactual", 0)
            delta = change_data.get("change", 0)
            
            is_feasible = True
            reason = ""
            
            # Apply feasibility rules
            if feat in feasibility_rules:
                rules = feasibility_rules[feat]
                
                if rules.get("direction") == "immutable":
                    is_feasible = False
                    reason = f"Cannot change immutable attribute '{feat}'"
                
                elif rules.get("direction") == "increase_only":
                    if delta < 0:
                        is_feasible = False
                        reason = f"Cannot decrease '{feat}' (original={original}, counterfactual={counterfactual})"
                
                elif "allowed_values" in rules:
                    if counterfactual not in rules["allowed_values"]:
                        is_feasible = False
                        reason = f"Invalid value for '{feat}': {counterfactual} not in {rules['allowed_values']}"
            
            # General feasibility checks
            if feat not in feasibility_rules:
                # Check for negative values in non-negative features
                if counterfactual < 0 and feat in ["tenure", "MonthlyCharges", "TotalCharges"]:
                    is_feasible = False
                    reason = f"Cannot have negative value for '{feat}': {counterfactual}"
                
                # Check for unrealistic changes (>3 std deviations)
                if abs(delta) > 3:  # Generic threshold for standardized features
                    is_feasible = False
                    reason = f"Unrealistic change for '{feat}': delta={delta:.2f}"
            
            if not is_feasible:
                infeasible_changes.append({
                    "customer_index": customer_idx,
                    "feature": feat,
                    "original": original,
                    "counterfactual": counterfactual,
                    "reason": reason
                })
            else:
                feasible_count += 1
    
    total_changes = feasible_count + len(infeasible_changes)
    feasibility_rate = (feasible_count / total_changes * 100) if total_changes > 0 else 0
    
    if infeasible_changes:
        infeasible_text = "\n".join([
            f"  ❌ Customer {ic['customer_index']}: {ic['feature']} - {ic['reason']}"
            for ic in infeasible_changes
        ])
        
        return (
            f"VALIDATION FAILED: Infeasible counterfactual changes detected.\n"
            f"Feasibility Rate: {feasibility_rate:.1f}% ({feasible_count}/{total_changes} changes valid)\n"
            f"Infeasible Changes:\n{infeasible_text}\n"
            f"Recommendation: Review counterfactual generation constraints."
        )
    
    return (
        f"Counterfactual Feasibility Validation PASSED.\n"
        f"Feasibility Rate: {feasibility_rate:.1f}% ({feasible_count}/{total_changes} changes valid)\n"
        f"Customers Analyzed: {len(counterfactuals)}\n"
        f"Status: ✅ All counterfactual suggestions are realistic and actionable."
    )