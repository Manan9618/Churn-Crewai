import pandas as pd
import numpy as np
import pickle
import json
import os
from crewai.tools import tool

SHAP_PATH = "artifacts/model/shap_values.pkl"
EXPLANATION_REPORT_PATH = "artifacts/model/explanation_report.json"
EXPECTED_TOP_FEATURES = {"Contract", "tenure", "MonthlyCharges", "TotalCharges", "InternetService"}


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