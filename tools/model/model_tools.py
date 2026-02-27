import pandas as pd
import numpy as np
import pickle
import json
import os
from crewai.tools import tool

PROCESSED_PATH = "artifacts/data/processed_churn.csv"
FEATURES_PATH = "artifacts/data/selected_features.json"
MODEL_PATH = "artifacts/model/churn_model.pkl"
PREDICTIONS_PATH = "artifacts/data/predictions.csv"


def _load_data_and_features():
    df = pd.read_csv(PROCESSED_PATH).dropna()
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    X = df[features]
    y = df["Churn"]
    return X, y, features


@tool("Train Churn Model")
def train_model_tool(model_type: str = "gradient_boosting") -> str:
    """
    Train a classification model for churn prediction.
    model_type options: 'logistic_regression', 'random_forest', 'gradient_boosting'
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    X, y, features = _load_data_and_features()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    model = models.get(model_type, models["gradient_boosting"])
    model.fit(X_tr, y_tr)

    # Save train/test splits
    os.makedirs("artifacts/data", exist_ok=True)
    pd.DataFrame(X_tr, columns=features).to_csv("artifacts/data/X_train.csv", index=False)
    pd.DataFrame(X_te, columns=features).to_csv("artifacts/data/X_test.csv", index=False)
    pd.Series(y_tr.values).to_csv("artifacts/data/y_train.csv", index=False)
    pd.Series(y_te.values).to_csv("artifacts/data/y_test.csv", index=False)

    os.makedirs("artifacts/model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return f"Model '{model_type}' trained and saved to {MODEL_PATH}. Train size: {len(X_tr)}, Test size: {len(X_te)}."


@tool("Cross Validate Model")
def cross_validate_tool(cv: int = 5) -> str:
    """Run stratified k-fold cross-validation and return mean AUC, F1, Recall."""
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import make_scorer, f1_score, recall_score, roc_auc_score

    if not os.path.exists(MODEL_PATH):
        return "ERROR: Model not found. Run train_model_tool first."

    X, y, _ = _load_data_and_features()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    scoring = {
        "auc": "roc_auc",
        "f1": make_scorer(f1_score),
        "recall": make_scorer(recall_score),
    }
    cv_results = cross_validate(model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), scoring=scoring)
    return (
        f"{cv}-Fold Cross-Validation Results:\n"
        f"  AUC:    {cv_results['test_auc'].mean():.4f} ± {cv_results['test_auc'].std():.4f}\n"
        f"  F1:     {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}\n"
        f"  Recall: {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}"
    )


@tool("Evaluate Model")
def evaluate_model_tool() -> str:
    """Evaluate the trained model on the test set and return classification report."""
    from sklearn.metrics import classification_report, roc_auc_score

    if not os.path.exists(MODEL_PATH):
        return "ERROR: Model not found. Run train_model_tool first."

    X_te = pd.read_csv("artifacts/data/X_test.csv")
    y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    report = classification_report(y_te, y_pred, target_names=["No Churn", "Churn"])
    return f"Test AUC: {auc:.4f}\n\nClassification Report:\n{report}"


@tool("Save Model")
def save_model_tool() -> str:
    """Confirm model is saved and return file size."""
    if not os.path.exists(MODEL_PATH):
        return f"ERROR: Model not found at {MODEL_PATH}."
    size = os.path.getsize(MODEL_PATH) / 1024
    return f"Model saved at {MODEL_PATH} ({size:.1f} KB)."


@tool("Load Model")
def load_model_tool() -> str:
    """Load the trained churn model and return a confirmation."""
    if not os.path.exists(MODEL_PATH):
        return f"ERROR: Model file not found at {MODEL_PATH}."
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return f"Model loaded successfully: {type(model).__name__}"


@tool("Predict Churn")
def predict_tool(path: str = PROCESSED_PATH) -> str:
    """Run churn predictions on the given dataset and return a preview."""
    if not os.path.exists(MODEL_PATH):
        return "ERROR: Model not found. Train the model first."

    df = pd.read_csv(path)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X = df[features]
    df["Churn_Predicted"] = model.predict(X)
    df["Churn_Probability"] = model.predict_proba(X)[:, 1].round(4)
    df.to_csv(PREDICTIONS_PATH, index=False)
    churn_rate = df["Churn_Predicted"].mean() * 100
    return (
        f"Predictions complete. Predicted churn rate: {churn_rate:.2f}%.\n"
        f"Saved to {PREDICTIONS_PATH}.\n"
        f"Preview:\n{df[['Churn_Predicted', 'Churn_Probability']].head(5).to_string()}"
    )


@tool("Save Predictions")
def save_predictions_tool() -> str:
    """Confirm predictions file exists."""
    if not os.path.exists(PREDICTIONS_PATH):
        return f"ERROR: Predictions file not found at {PREDICTIONS_PATH}."
    df = pd.read_csv(PREDICTIONS_PATH)
    return f"Predictions file confirmed at {PREDICTIONS_PATH}. Shape: {df.shape}."