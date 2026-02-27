import pandas as pd
import os
from crewai.tools import tool

PYCARET_MODEL_PATH = "artifacts/model/pycaret_best_model.pkl"
PYCARET_RESULTS_PATH = "artifacts/model/pycaret_compare_results.csv"


@tool("Validate PyCaret Setup")
def validate_pycaret_setup_tool() -> str:
    """Validate that PyCaret processed data and features are ready for experiment setup."""
    required = ["artifacts/data/processed_churn.csv"]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        return f"VALIDATION FAILED: Missing files for PyCaret setup: {missing}"
    df = pd.read_csv("artifacts/data/processed_churn.csv")
    if "Churn" not in df.columns:
        return "VALIDATION FAILED: Target column 'Churn' missing from processed data."
    non_numeric = [c for c in df.columns if df[c].dtype == object]
    if non_numeric:
        return f"VALIDATION FAILED: Non-numeric columns present: {non_numeric}"
    return f"PyCaret setup pre-conditions PASSED. Data shape: {df.shape}."


@tool("Validate Compare Models Output")
def validate_compare_models_output_tool(min_auc: float = 0.75, min_f1: float = 0.55) -> str:
    """Validate that PyCaret model comparison produced a valid leaderboard meeting thresholds."""
    if not os.path.exists(PYCARET_RESULTS_PATH):
        return f"VALIDATION FAILED: Comparison results not found at {PYCARET_RESULTS_PATH}."
    df = pd.read_csv(PYCARET_RESULTS_PATH)
    if df.empty:
        return "VALIDATION FAILED: Comparison results are empty."
    best_auc = df["AUC"].max() if "AUC" in df.columns else 0
    best_f1 = df["F1"].max() if "F1" in df.columns else 0
    errors = []
    if best_auc < min_auc:
        errors.append(f"Best AUC {best_auc:.4f} < {min_auc}")
    if best_f1 < min_f1:
        errors.append(f"Best F1 {best_f1:.4f} < {min_f1}")
    if errors:
        return "VALIDATION FAILED:\n" + "\n".join(errors)
    return (
        f"Compare models validation PASSED.\n"
        f"Best AUC: {best_auc:.4f}, Best F1: {best_f1:.4f}\n"
        f"Models evaluated: {len(df)}"
    )


@tool("Validate Best Model Saved")
def validate_best_model_saved_tool() -> str:
    """Validate that the best PyCaret model was saved to disk."""
    if not os.path.exists(PYCARET_MODEL_PATH):
        return f"VALIDATION FAILED: PyCaret model not found at {PYCARET_MODEL_PATH}."
    size_kb = os.path.getsize(PYCARET_MODEL_PATH) / 1024
    return f"PyCaret model save validation PASSED. File: {PYCARET_MODEL_PATH} ({size_kb:.1f} KB)."