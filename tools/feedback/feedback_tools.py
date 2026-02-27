import pandas as pd
import json
import os
from datetime import datetime
from crewai.tools import tool

EXPERIMENT_LOG_PATH = "artifacts/model/experiment_log.json"
FEATURES_PATH = "artifacts/data/selected_features.json"


def _load_log():
    if os.path.exists(EXPERIMENT_LOG_PATH):
        with open(EXPERIMENT_LOG_PATH) as f:
            return json.load(f)
    return {"experiments": []}


@tool("Compare Metrics Across Runs")
def compare_metrics_tool() -> str:
    """Compare AUC, F1, and Recall across all logged experiment runs."""
    log = _load_log()
    if not log["experiments"]:
        return "No experiments logged yet."
    df = pd.DataFrame(log["experiments"])
    cols = [c for c in ["run_id", "timestamp", "auc", "f1", "recall", "model_type"] if c in df.columns]
    return f"Experiment Comparison:\n{df[cols].to_string(index=False)}"


@tool("Log Experiment")
def log_experiment_tool(model_type: str = "gradient_boosting") -> str:
    """Log current model metrics to the experiment log file."""
    from sklearn.metrics import roc_auc_score, f1_score, recall_score
    import pickle

    log = _load_log()
    run_id = len(log["experiments"]) + 1

    metrics = {"run_id": run_id, "timestamp": datetime.now().isoformat(), "model_type": model_type}

    model_path = "artifacts/model/churn_model.pkl"
    if os.path.exists(model_path) and os.path.exists("artifacts/data/X_test.csv"):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(FEATURES_PATH) as f:
            features = json.load(f)
        X_te = pd.read_csv("artifacts/data/X_test.csv")[features]
        y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        metrics["auc"] = round(roc_auc_score(y_te, y_prob), 4)
        metrics["f1"] = round(f1_score(y_te, y_pred), 4)
        metrics["recall"] = round(recall_score(y_te, y_pred), 4)

    log["experiments"].append(metrics)
    os.makedirs("model", exist_ok=True)
    with open(EXPERIMENT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return f"Experiment Run {run_id} logged: {metrics}"


@tool("Suggest Improvements")
def suggest_improvements_tool() -> str:
    """Analyse current metrics and suggest concrete improvements for the next iteration."""
    import pickle

    suggestions = []

    model_path = "artifacts/model/churn_model.pkl"
    if os.path.exists(model_path) and os.path.exists("artifacts/data/X_test.csv"):
        from sklearn.metrics import roc_auc_score, f1_score, recall_score
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(FEATURES_PATH) as f:
            features = json.load(f)
        X_te = pd.read_csv("artifacts/data/X_test.csv")[features]
        y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        f1 = f1_score(y_te, y_pred)
        recall = recall_score(y_te, y_pred)

        if auc < 0.80:
            suggestions.append("Try XGBoost or LightGBM for potentially higher AUC.")
        if f1 < 0.60:
            suggestions.append("Apply SMOTE or class_weight='balanced' to improve minority class F1.")
        if recall < 0.65:
            suggestions.append("Lower the classification threshold from 0.5 to 0.4 to improve churn recall.")
        if not suggestions:
            suggestions.append("Model is performing well. Consider feature engineering: add tenure_group, charges_ratio features.")
        suggestions.append("Run hyperparameter tuning with RandomizedSearchCV for further gains.")
    else:
        suggestions.append("Train a model first before requesting improvement suggestions.")

    return "Improvement Suggestions for Next Iteration:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(suggestions))


@tool("Update Feature List")
def update_feature_list_tool(add_features: str = "", remove_features: str = "") -> str:
    """
    Update the selected features list by adding or removing features.
    Pass comma-separated feature names.
    """
    if not os.path.exists(FEATURES_PATH):
        return f"ERROR: {FEATURES_PATH} not found. Run feature selection first."
    with open(FEATURES_PATH) as f:
        features = json.load(f)

    if add_features:
        new = [f.strip() for f in add_features.split(",") if f.strip()]
        features = list(set(features + new))

    if remove_features:
        rem = [f.strip() for f in remove_features.split(",") if f.strip()]
        features = [f for f in features if f not in rem]

    with open(FEATURES_PATH, "w") as f:
        json.dump(features, f)

    return f"Feature list updated. Current features ({len(features)}): {features}"