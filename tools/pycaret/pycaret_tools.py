import pandas as pd
import os
import json
from crewai.tools import tool

PROCESSED_PATH = "artifacts/data/processed_churn.csv"
FEATURES_PATH = "artifacts/data/selected_features.json"
PYCARET_MODEL_PATH = "artifacts/model/pycaret_best_model"
PYCARET_RESULTS_PATH = "artifacts/model/pycaret_compare_results.csv"

_pycaret_setup_done = False
_clf_setup = None
_best_model = None


@tool("PyCaret Setup")
def pycaret_setup_tool() -> str:
    """
    Initialise the PyCaret classification experiment with the processed churn dataset.
    Handles class imbalance via fix_imbalance=True and sets AUC as the sort metric.
    """
    global _clf_setup, _pycaret_setup_done
    from pycaret.classification import setup, get_config

    df = pd.read_csv(PROCESSED_PATH).dropna()
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            features = json.load(f)
        keep_cols = features + ["Churn"]
        df = df[[c for c in keep_cols if c in df.columns]]

    _clf_setup = setup(
        data=df,
        target="Churn",
        fix_imbalance=True,
        session_id=42,
        verbose=False,
        html=False,
    )
    _pycaret_setup_done = True
    return (
        "PyCaret setup complete.\n"
        f"Dataset shape: {df.shape}\n"
        f"Target: Churn\n"
        f"Imbalance fix: SMOTE enabled\n"
        f"Session ID: 42"
    )


@tool("PyCaret Compare Models")
def pycaret_compare_models_tool(n_select: int = 5) -> str:
    """
    Compare multiple classifiers and return the leaderboard sorted by AUC.
    Saves comparison results to CSV.
    """
    global _best_model
    from pycaret.classification import compare_models, pull

    if not _pycaret_setup_done:
        return "ERROR: Run pycaret_setup_tool first."

    best_models = compare_models(
        sort="AUC",
        n_select=n_select,
        verbose=False,
        errors="ignore",
    )
    results = pull()
    os.makedirs("model", exist_ok=True)
    results.to_csv(PYCARET_RESULTS_PATH, index=False)
    _best_model = best_models[0] if isinstance(best_models, list) else best_models
    return f"Model comparison complete. Leaderboard:\n{results[['Model', 'AUC', 'F1', 'Recall', 'Prec.']].head(10).to_string(index=False)}"


@tool("PyCaret Tune Best Model")
def pycaret_tune_model_tool(n_iter: int = 50) -> str:
    """Tune the best model from compare_models using random search."""
    global _best_model
    from pycaret.classification import tune_model, pull

    if _best_model is None:
        return "ERROR: Run pycaret_compare_models_tool first."

    tuned = tune_model(_best_model, n_iter=n_iter, optimize="AUC", verbose=False)
    results = pull()
    _best_model = tuned
    return (
        f"Model tuned. Best CV Results:\n"
        f"{results[['AUC', 'F1', 'Recall', 'Prec.']].to_string()}"
    )


@tool("PyCaret Save Model")
def pycaret_save_model_tool() -> str:
    """Save the best PyCaret model pipeline to disk."""
    from pycaret.classification import save_model

    if _best_model is None:
        return "ERROR: No model to save. Run compare/tune first."

    os.makedirs("model", exist_ok=True)
    save_model(_best_model, PYCARET_MODEL_PATH)
    return f"PyCaret best model saved to {PYCARET_MODEL_PATH}.pkl"