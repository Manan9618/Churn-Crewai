import pandas as pd
import numpy as np
import json
import os
from crewai.tools import tool

FEATURES_PATH = "artifacts/data/selected_features.json"
PROCESSED_PATH = "artifacts/data/processed_churn.csv"
LEAKAGE_COLUMNS = ["customerID", "Churn"]
VIF_REPORT_PATH = "artifacts/data/vif_report.json"
STABILITY_REPORT_PATH = "artifacts/data/feature_stability_report.json"
INFORMATION_VALUE_PATH = "artifacts/data/information_value_report.json"

# Validation thresholds
VIF_THRESHOLD = 5.0  # VIF > 5 indicates high multicollinearity
STABILITY_THRESHOLD = 0.3  # Max std dev of feature importance across CV folds
IV_THRESHOLD = 0.02  # Minimum Information Value for predictive power


@tool("Validate Feature Count")
def validate_feature_count_tool(min_features: int = 5) -> str:
    """Validate that at least min_features were selected."""
    if not os.path.exists(FEATURES_PATH):
        return f"VALIDATION FAILED: {FEATURES_PATH} not found."
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    if len(features) < min_features:
        return f"VALIDATION FAILED: Only {len(features)} features selected (minimum {min_features})."
    return f"Feature count validation PASSED. {len(features)} features selected."


@tool("Validate No Target Leakage")
def validate_no_target_leakage_tool() -> str:
    """Check that leakage columns are not in the selected feature list."""
    if not os.path.exists(FEATURES_PATH):
        return f"VALIDATION FAILED: {FEATURES_PATH} not found."
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    leakage = [c for c in features if c in LEAKAGE_COLUMNS]
    if leakage:
        return f"VALIDATION FAILED: Leakage columns found in features: {leakage}"
    return "Target leakage validation PASSED. No leakage columns in feature set."


@tool("Validate Feature Variance")
def validate_feature_variance_tool(path: str = PROCESSED_PATH, threshold: float = 0.001) -> str:
    """Ensure all selected features have variance above threshold."""
    if not os.path.exists(FEATURES_PATH):
        return f"VALIDATION FAILED: {FEATURES_PATH} not found."
    df = pd.read_csv(path)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    low_var = [f for f in features if f in df.columns and df[f].var() < threshold]
    if low_var:
        return f"VALIDATION FAILED: Low variance features: {low_var}"
    return f"Feature variance validation PASSED. All selected features have variance ≥ {threshold}."


@tool("Validate Selected Features Saved")
def validate_selected_features_saved_tool() -> str:
    """Confirm selected_features.json exists and is a non-empty list."""
    if not os.path.exists(FEATURES_PATH):
        return f"VALIDATION FAILED: {FEATURES_PATH} not found."
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    if not isinstance(features, list) or len(features) == 0:
        return "VALIDATION FAILED: selected_features.json is empty or invalid."
    return f"Selected features file PASSED. Contains {len(features)} features: {features}"


# ── NEW FEATURE VALIDATION TOOLS ADDED BELOW ──────────────────────────────────


@tool("Validate Multicollinearity VIF")
def validate_multicollinearity_vif_tool(
    path: str = PROCESSED_PATH,
    vif_threshold: float = VIF_THRESHOLD
) -> str:
    """
    Calculates Variance Inflation Factor (VIF) to ensure features aren't too correlated.
    VIF > 5 indicates high multicollinearity that can destabilize model coefficients.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    if not os.path.exists(FEATURES_PATH):
        return f"VALIDATION FAILED: {FEATURES_PATH} not found."
    
    df = pd.read_csv(path).dropna()
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    
    # Filter to only selected features that exist in dataframe
    X = df[[f for f in features if f in df.columns]]
    
    if X.shape[1] < 2:
        return "VALIDATION SKIPPED: Need at least 2 features to calculate VIF."
    
    # Add constant for VIF calculation
    X_with_const = pd.DataFrame(X.values, columns=X.columns)
    
    # Calculate VIF for each feature
    vif_results = []
    for i, feature in enumerate(X_with_const.columns):
        try:
            vif = variance_inflation_factor(X_with_const.values, i)
            vif_results.append({
                "feature": feature,
                "vif": round(float(vif), 4),
                "status": "HIGH" if vif > vif_threshold else "OK"
            })
        except Exception as e:
            vif_results.append({
                "feature": feature,
                "vif": None,
                "status": "ERROR",
                "error": str(e)
            })
    
    vif_df = pd.DataFrame(vif_results)
    high_vif_features = vif_df[vif_df["status"] == "HIGH"]["feature"].tolist()
    
    # Save VIF report
    os.makedirs("artifacts/data", exist_ok=True)
    vif_report = {
        "vif_threshold": vif_threshold,
        "n_features_analyzed": len(vif_results),
        "n_high_vif_features": len(high_vif_features),
        "high_vif_features": high_vif_features,
        "all_vif_scores": vif_results
    }
    
    with open(VIF_REPORT_PATH, "w") as f:
        json.dump(vif_report, f, indent=2)
    
    if high_vif_features:
        high_vif_text = "\n".join([
            f"  ⚠️ {r['feature']}: VIF = {r['vif']}"
            for r in vif_results if r["status"] == "HIGH"
        ])
        return (
            f"VALIDATION FAILED: High multicollinearity detected.\n"
            f"VIF Threshold: {vif_threshold}\n"
            f"Features with VIF > {vif_threshold}: {len(high_vif_features)}\n"
            f"High VIF Features:\n{high_vif_text}\n"
            f"All VIF Scores:\n{vif_df.to_string(index=False)}\n"
            f"Recommendation: Remove or combine highly correlated features.\n"
            f"Report saved to: {VIF_REPORT_PATH}"
        )
    
    avg_vif = vif_df["vif"].mean()
    return (
        f"Multicollinearity VIF Validation PASSED.\n"
        f"VIF Threshold: {vif_threshold}\n"
        f"Features Analyzed: {len(vif_results)}\n"
        f"High VIF Features: 0\n"
        f"Average VIF: {avg_vif:.4f}\n"
        f"Max VIF: {vif_df['vif'].max():.4f}\n"
        f"All VIF Scores:\n{vif_df.to_string(index=False)}\n"
        f"Status: ✅ No significant multicollinearity detected.\n"
        f"Report saved to: {VIF_REPORT_PATH}"
    )


@tool("Validate Feature Stability")
def validate_feature_stability_tool(
    path: str = PROCESSED_PATH,
    n_folds: int = 5,
    stability_threshold: float = STABILITY_THRESHOLD
) -> str:
    """
    Checks if feature importance remains consistent across different CV folds.
    Unstable features may indicate overfitting or noise.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    
    if not os.path.exists(FEATURES_PATH):
        return f"VALIDATION FAILED: {FEATURES_PATH} not found."
    
    df = pd.read_csv(path).dropna()
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    
    X = df[[f for f in features if f in df.columns]]
    y = df["Churn"]
    
    if X.shape[0] < n_folds * 10:
        return (
            f"VALIDATION SKIPPED: Insufficient samples ({X.shape[0]}) for {n_folds}-fold CV.\n"
            f"Need at least {n_folds * 10} samples."
        )
    
    # Cross-validation to get feature importance per fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_importances = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        fold_importance = pd.Series(rf.feature_importances_, index=X.columns)
        fold_importances.append(fold_importance)
    
    # Calculate stability metrics
    importance_df = pd.DataFrame(fold_importances)
    mean_importance = importance_df.mean()
    std_importance = importance_df.std()
    cv_importance = std_importance / (mean_importance + 0.001)  # Coefficient of variation
    
    # Identify unstable features
    unstable_features = std_importance[std_importance > stability_threshold].index.tolist()
    
    # Save stability report
    os.makedirs("artifacts/data", exist_ok=True)
    stability_report = {
        "n_folds": n_folds,
        "stability_threshold": stability_threshold,
        "n_features_analyzed": len(features),
        "n_unstable_features": len(unstable_features),
        "unstable_features": unstable_features,
        "mean_importance": mean_importance.to_dict(),
        "std_importance": std_importance.to_dict(),
        "cv_importance": cv_importance.to_dict(),
        "feature_stability": [
            {
                "feature": feat,
                "mean_importance": round(float(mean_importance[feat]), 6),
                "std_importance": round(float(std_importance[feat]), 6),
                "cv_importance": round(float(cv_importance[feat]), 4),
                "status": "UNSTABLE" if feat in unstable_features else "STABLE"
            }
            for feat in X.columns
        ]
    }
    
    with open(STABILITY_REPORT_PATH, "w") as f:
        json.dump(stability_report, f, indent=2)
    
    if unstable_features:
        unstable_text = "\n".join([
            f"  ⚠️ {feat}: std={std_importance[feat]:.6f}, CV={cv_importance[feat]:.4f}"
            for feat in unstable_features
        ])
        return (
            f"VALIDATION FAILED: Unstable feature importance detected.\n"
            f"CV Folds: {n_folds}\n"
            f"Stability Threshold (std): {stability_threshold}\n"
            f"Unstable Features: {len(unstable_features)}\n"
            f"Unstable Features Details:\n{unstable_text}\n"
            f"\nAll Feature Stability:\n"
            f"{pd.DataFrame(stability_report['feature_stability']).to_string(index=False)}\n"
            f"Recommendation: Consider removing unstable features or collecting more data.\n"
            f"Report saved to: {STABILITY_REPORT_PATH}"
        )
    
    avg_std = std_importance.mean()
    return (
        f"Feature Stability Validation PASSED.\n"
        f"CV Folds: {n_folds}\n"
        f"Stability Threshold (std): {stability_threshold}\n"
        f"Features Analyzed: {len(features)}\n"
        f"Unstable Features: 0\n"
        f"Average Std Dev: {avg_std:.6f}\n"
        f"Max Std Dev: {std_importance.max():.6f}\n"
        f"\nAll Feature Stability:\n"
        f"{pd.DataFrame(stability_report['feature_stability']).to_string(index=False)}\n"
        f"Status: ✅ Feature importance is consistent across CV folds.\n"
        f"Report saved to: {STABILITY_REPORT_PATH}"
    )


@tool("Validate Information Value")
def validate_information_value_tool(
    path: str = PROCESSED_PATH,
    iv_threshold: float = IV_THRESHOLD,
    n_bins: int = 10
) -> str:
    """
    Ensures selected features have sufficient predictive power using Weight of Evidence (WOE)
    and Information Value (IV) analysis. IV < 0.02 indicates useless predictors.
    """
    if not os.path.exists(FEATURES_PATH):
        return f"VALIDATION FAILED: {FEATURES_PATH} not found."
    
    df = pd.read_csv(path).dropna()
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    
    def calculate_woe_iv(df, feature, target):
        """Calculate WOE and IV for a feature."""
        # Bin the feature if numeric
        if df[feature].nunique() > n_bins:
            df_temp = df.copy()
            df_temp[f"{feature}_binned"] = pd.qcut(df_temp[feature], q=n_bins, duplicates='drop')
            feature_name = f"{feature}_binned"
        else:
            df_temp = df.copy()
            df_temp[f"{feature}_binned"] = df_temp[feature].astype(str)
            feature_name = f"{feature}_binned"
        
        # Calculate WOE and IV
        grouped = df_temp.groupby(feature_name)[target].agg(['sum', 'count'])
        grouped.columns = ['events', 'total']
        
        total_events = grouped['events'].sum()
        total_non_events = grouped['total'].sum() - total_events
        
        grouped['non_events'] = grouped['total'] - grouped['events']
        grouped['event_rate'] = grouped['events'] / total_events
        grouped['non_event_rate'] = grouped['non_events'] / total_non_events
        
        # Handle division by zero
        grouped['event_rate'] = grouped['event_rate'].replace(0, 0.0001)
        grouped['non_event_rate'] = grouped['non_event_rate'].replace(0, 0.0001)
        
        grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
        grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']
        
        return grouped['iv'].sum(), grouped[['woe', 'iv']]
    
    # Calculate IV for each feature
    iv_results = []
    weak_features = []
    
    for feature in features:
        if feature not in df.columns or feature == "Churn":
            continue
        
        try:
            iv_score, woe_table = calculate_woe_iv(df, feature, "Churn")
            
            # Determine predictive power
            if iv_score < 0.02:
                predictive_power = "Useless"
                weak_features.append(feature)
            elif iv_score < 0.1:
                predictive_power = "Weak"
            elif iv_score < 0.3:
                predictive_power = "Medium"
            else:
                predictive_power = "Strong"
            
            iv_results.append({
                "feature": feature,
                "information_value": round(float(iv_score), 6),
                "predictive_power": predictive_power,
                "status": "FAIL" if iv_score < iv_threshold else "PASS"
            })
        except Exception as e:
            iv_results.append({
                "feature": feature,
                "information_value": None,
                "predictive_power": "ERROR",
                "status": "ERROR",
                "error": str(e)
            })
    
    iv_df = pd.DataFrame(iv_results)
    
    # Save IV report
    os.makedirs("artifacts/data", exist_ok=True)
    iv_report = {
        "iv_threshold": iv_threshold,
        "n_features_analyzed": len(iv_results),
        "n_weak_features": len(weak_features),
        "weak_features": weak_features,
        "all_iv_scores": iv_results,
        "iv_summary": {
            "useless": len([r for r in iv_results if r["predictive_power"] == "Useless"]),
            "weak": len([r for r in iv_results if r["predictive_power"] == "Weak"]),
            "medium": len([r for r in iv_results if r["predictive_power"] == "Medium"]),
            "strong": len([r for r in iv_results if r["predictive_power"] == "Strong"])
        }
    }
    
    with open(INFORMATION_VALUE_PATH, "w") as f:
        json.dump(iv_report, f, indent=2)
    
    if weak_features:
        weak_text = "\n".join([
            f"  ⚠️ {r['feature']}: IV = {r['information_value']}"
            for r in iv_results if r["feature"] in weak_features
        ])
        return (
            f"VALIDATION FAILED: Features with insufficient predictive power detected.\n"
            f"IV Threshold: {iv_threshold}\n"
            f"Features with IV < {iv_threshold}: {len(weak_features)}\n"
            f"Weak Features:\n{weak_text}\n"
            f"\nAll IV Scores:\n{iv_df.to_string(index=False)}\n"
            f"\nIV Summary:\n"
            f"  Useless (IV < 0.02): {iv_report['iv_summary']['useless']}\n"
            f"  Weak (0.02 ≤ IV < 0.1): {iv_report['iv_summary']['weak']}\n"
            f"  Medium (0.1 ≤ IV < 0.3): {iv_report['iv_summary']['medium']}\n"
            f"  Strong (IV ≥ 0.3): {iv_report['iv_summary']['strong']}\n"
            f"Recommendation: Remove features with IV < 0.02.\n"
            f"Report saved to: {INFORMATION_VALUE_PATH}"
        )
    
    avg_iv = iv_df["information_value"].mean()
    return (
        f"Information Value Validation PASSED.\n"
        f"IV Threshold: {iv_threshold}\n"
        f"Features Analyzed: {len(iv_results)}\n"
        f"Weak Features (IV < {iv_threshold}): 0\n"
        f"Average IV: {avg_iv:.6f}\n"
        f"Max IV: {iv_df['information_value'].max():.6f}\n"
        f"\nAll IV Scores:\n{iv_df.to_string(index=False)}\n"
        f"\nIV Summary:\n"
        f"  Useless (IV < 0.02): {iv_report['iv_summary']['useless']}\n"
        f"  Weak (0.02 ≤ IV < 0.1): {iv_report['iv_summary']['weak']}\n"
        f"  Medium (0.1 ≤ IV < 0.3): {iv_report['iv_summary']['medium']}\n"
        f"  Strong (IV ≥ 0.3): {iv_report['iv_summary']['strong']}\n"
        f"Status: ✅ All selected features have sufficient predictive power.\n"
        f"Report saved to: {INFORMATION_VALUE_PATH}"
    )