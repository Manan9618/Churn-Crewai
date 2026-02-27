import pandas as pd
import numpy as np
import json
import os
from crewai.tools import tool

PROCESSED_PATH = "artifacts/data/processed_churn.csv"
FEATURES_PATH = "artifacts/data/selected_features.json"


@tool("Correlation Analysis")
def correlation_analysis_tool(path: str = PROCESSED_PATH, threshold: float = 0.05) -> str:
    """Compute correlation of each feature with the Churn target and return sorted results."""
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        return "ERROR: Churn column not found."
    corr = df.corr()["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
    significant = corr[corr.abs() >= threshold]
    return f"Features correlated with Churn (|r| >= {threshold}):\n{significant.to_string()}"


@tool("Feature Importance via Random Forest")
def feature_importance_tool(path: str = PROCESSED_PATH, top_n: int = 15) -> str:
    """Train a quick Random Forest and return top-N feature importances."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path).dropna()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return f"Top {top_n} Feature Importances:\n{importances.head(top_n).to_string()}"


@tool("Variance Threshold Filter")
def variance_threshold_tool(path: str = PROCESSED_PATH, threshold: float = 0.01) -> str:
    """Remove features with variance below the given threshold."""
    from sklearn.feature_selection import VarianceThreshold

    df = pd.read_csv(path).dropna()
    X = df.drop(columns=["Churn"])
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    removed = list(X.columns[~selector.get_support()])
    kept = list(X.columns[selector.get_support()])
    return f"Variance threshold={threshold}.\nRemoved ({len(removed)}): {removed}\nKept ({len(kept)}): {kept}"


@tool("Select Top Features")
def select_top_features_tool(path: str = PROCESSED_PATH, top_n: int = 12) -> str:
    """Select top N features by RF importance and save to selected_features.json."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path).dropna()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected = list(importances.head(top_n).index)
    os.makedirs("data", exist_ok=True)
    with open(FEATURES_PATH, "w") as f:
        json.dump(selected, f)
    return f"Selected top {top_n} features saved to {FEATURES_PATH}:\n{selected}"