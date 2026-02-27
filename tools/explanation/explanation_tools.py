import pandas as pd
import numpy as np
import pickle
import json
import os
from crewai.tools import tool

MODEL_PATH = "artifacts/model/churn_model.pkl"
FEATURES_PATH = "artifacts/data/selected_features.json"
SHAP_PATH = "artifacts/model/shap_values.pkl"
EXPLANATION_REPORT_PATH = "artifacts/model/explanation_report.json"


def _load_model_and_data():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    X_te = pd.read_csv("artifacts/data/X_test.csv")[features]
    return model, X_te, features


@tool("SHAP Summary Analysis")
def shap_summary_tool(sample_size: int = 500) -> str:
    """
    Compute SHAP values for the test set and return a summary of
    mean absolute SHAP values per feature (global importance).
    """
    import shap

    model, X_te, features = _load_model_and_data()
    sample = X_te.sample(min(sample_size, len(X_te)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # For binary classification, take the positive class SHAP values
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=features).sort_values(ascending=False)

    os.makedirs("model", exist_ok=True)
    with open(SHAP_PATH, "wb") as f:
        pickle.dump({"shap_values": sv, "feature_names": features}, f)

    return f"Global SHAP Importance (mean |SHAP value|):\n{mean_abs.to_string()}"


@tool("SHAP Force Plot Summary")
def shap_force_plot_tool(customer_index: int = 0) -> str:
    """
    Return the SHAP force plot explanation for a specific customer (by index in test set).
    Shows which features pushed the prediction above or below the base value.
    """
    import shap

    model, X_te, features = _load_model_and_data()
    
    # ✅ ADD THIS VALIDATION BLOCK
    if len(X_te) == 0:
        return "ERROR: Test set is empty. Cannot generate SHAP explanation."
    
    if customer_index < 0 or customer_index >= len(X_te):
        return (
            f"ERROR: Invalid customer_index={customer_index}.\n"
            f"Test set has {len(X_te)} sample(s).\n"
            f"Valid indices: 0 to {len(X_te) - 1}\n"
            f"Please use a valid index (try customer_index=0)"
        )
    # ✅ END VALIDATION BLOCK

    row = X_te.iloc[[customer_index]]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    contribution = pd.Series(sv, index=features).sort_values(key=abs, ascending=False)
    return (
        f"Customer Index {customer_index} — SHAP Explanation:\n"
        f"Base value: {base_value:.4f}\n"
        f"Top feature contributions:\n{contribution.head(8).to_string()}"
    )


@tool("LIME Explanation")
def lime_explanation_tool(customer_index: int = 0, num_features: int = 8) -> str:
    """
    Generate a LIME local explanation for a specific customer in the test set.
    """
    from lime.lime_tabular import LimeTabularExplainer

    model, X_te, features = _load_model_and_data()
    X_tr = pd.read_csv("artifacts/data/X_train.csv")[features]

    explainer = LimeTabularExplainer(
        training_data=X_tr.values,
        feature_names=features,
        class_names=["No Churn", "Churn"],
        mode="classification",
        random_state=42,
    )

    instance = X_te.iloc[customer_index].values
    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=num_features,
    )

    lime_list = explanation.as_list()
    result = "\n".join([f"  {feat}: {weight:+.4f}" for feat, weight in lime_list])
    churn_prob = model.predict_proba([instance])[0][1]
    return (
        f"LIME Explanation for Customer Index {customer_index}:\n"
        f"Predicted Churn Probability: {churn_prob:.4f}\n"
        f"Feature Contributions:\n{result}"
    )


@tool("Generate Explanation Report")
def generate_explanation_report_tool() -> str:
    """
    Generate a JSON explanation report combining global SHAP importance
    and model performance summary.
    """
    from sklearn.metrics import roc_auc_score, f1_score

    if not os.path.exists(SHAP_PATH):
        return "ERROR: SHAP values not found. Run shap_summary_tool first."

    with open(SHAP_PATH, "rb") as f:
        shap_data = pickle.load(f)

    shap_vals = shap_data["shap_values"]
    features = shap_data["feature_names"]
    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=features).sort_values(ascending=False)

    report = {
        "model_path": MODEL_PATH,
        "global_feature_importance_shap": mean_abs.head(10).to_dict(),
        "top_churn_drivers": list(mean_abs.head(5).index),
    }

    if os.path.exists(MODEL_PATH) and os.path.exists("artifacts/data/X_test.csv"):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        X_te = pd.read_csv("artifacts/data/X_test.csv")[features]
        y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)
        report["test_auc"] = round(roc_auc_score(y_te, y_prob), 4)
        report["test_f1"] = round(f1_score(y_te, y_pred), 4)

    os.makedirs("model", exist_ok=True)
    with open(EXPLANATION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return f"Explanation report saved to {EXPLANATION_REPORT_PATH}.\nTop churn drivers: {report['top_churn_drivers']}"