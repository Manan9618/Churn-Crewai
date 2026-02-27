import pandas as pd
import json
import os
from crewai.tools import tool

FEATURES_PATH = "artifacts/data/selected_features.json"
PROCESSED_PATH = "artifacts/data/processed_churn.csv"
LEAKAGE_COLUMNS = ["customerID", "Churn"]


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