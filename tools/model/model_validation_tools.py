import pandas as pd
import numpy as np
import pickle
import os
from crewai.tools import tool

MODEL_PATH = "artifacts/model/churn_model.pkl"
PREDICTIONS_PATH = "artifacts/data/predictions.csv"


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