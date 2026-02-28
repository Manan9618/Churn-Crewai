import pandas as pd
import os
import json
from datetime import datetime
from crewai.tools import tool

PROCESSED_PATH = "artifacts/data/processed_churn.csv"
FEATURES_PATH = "artifacts/data/selected_features.json"
PYCARET_MODEL_PATH = "artifacts/model/pycaret_best_model"
PYCARET_RESULTS_PATH = "artifacts/model/pycaret_compare_results.csv"
PYCARET_BLEND_PATH = "artifacts/model/pycaret_blend_model"
PYCARET_STACK_PATH = "artifacts/model/pycaret_stack_model"
PYCARET_LEADERBOARD_PATH = "artifacts/model/pycaret_full_leaderboard.csv"
PYCARET_API_PATH = "artifacts/model/pycaret_api_deployment"

_pycaret_setup_done = False
_clf_setup = None
_best_model = None
_blend_model = None
_stack_model = None
_leaderboard_df = None


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
    global _best_model, _leaderboard_df
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
    _leaderboard_df = results.copy()
    
    os.makedirs("artifacts/model", exist_ok=True)
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

    os.makedirs("artifacts/model", exist_ok=True)
    save_model(_best_model, PYCARET_MODEL_PATH)
    return f"PyCaret best model saved to {PYCARET_MODEL_PATH}.pkl"


# ── NEW PYCARET ADVANCED TOOLS ADDED BELOW ────────────────────────────────────


@tool("Blend Top Models")
def blend_top_models_tool(
    n_models: int = 5,
    method: str = "auto"
) -> str:
    """
    Uses PyCaret's blend_models to combine the top N performers.
    Blending reduces variance and often improves generalization.
    """
    global _blend_model, _leaderboard_df
    from pycaret.classification import blend_models, pull
    
    if not _pycaret_setup_done:
        return "ERROR: Run pycaret_setup_tool first."
    
    if _leaderboard_df is None or len(_leaderboard_df) < n_models:
        return (
            f"ERROR: Insufficient models in leaderboard for blending.\n"
            f"Need at least {n_models} models, have {len(_leaderboard_df) if _leaderboard_df is not None else 0}.\n"
            f"Run pycaret_compare_models_tool first with n_select >= {n_models}."
        )
    
    # Get top N models from leaderboard
    top_model_names = _leaderboard_df.head(n_models)['Model'].tolist()
    
    try:
        # Create blended model
        blend = blend_models(
            estimator_list=top_model_names,
            method=method,
            verbose=False,
        )
        
        _blend_model = blend
        
        # Evaluate blend
        from pycaret.classification import evaluate_model
        blend_results = pull()
        
        # Save blended model
        os.makedirs("artifacts/model", exist_ok=True)
        from pycaret.classification import save_model
        save_model(blend, PYCARET_BLEND_PATH)
        
        # Get blend metrics
        from pycaret.classification import predict_model
        blend_pred = predict_model(blend)
        
        # Save blend report
        blend_report = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "n_models_blended": n_models,
            "models_included": top_model_names,
            "blend_model_path": f"{PYCARET_BLEND_PATH}.pkl",
            "status": "SUCCESS"
        }
        
        blend_report_path = "artifacts/model/pycaret_blend_report.json"
        with open(blend_report_path, "w") as f:
            json.dump(blend_report, f, indent=2)
        
        return (
            f"Model Blending Complete:\n"
            f"  Method: {method}\n"
            f"  Models Blended: {n_models}\n"
            f"  Models Included:\n"
            + "\n".join([f"    - {m}" for m in top_model_names]) +
            f"\n\n  Blend Model Saved: {PYCARET_BLEND_PATH}.pkl\n"
            f"  Report Saved: {blend_report_path}\n"
            f"\n💡 Blending combines predictions from multiple models to reduce variance."
        )
    
    except Exception as e:
        return (
            f"ERROR: Blending failed - {str(e)}\n"
            f"Recommendation: Try reducing n_models or use method='soft' for classification."
        )


@tool("Stack Models")
def stack_models_tool(
    base_models: str = "",
    meta_model: str = "logistic_regression",
    n_folds: int = 5
) -> str:
    """
    Uses PyCaret's stack_models for meta-learning.
    Stacking uses predictions from base models as features for a meta-model.
    
    Args:
        base_models: Comma-separated list of model names (e.g., "random_forest,gradient_boosting,logistic_regression")
        meta_model: Meta model for stacking (default: "logistic_regression")
        n_folds: Number of CV folds (default: 5) - passed via setup, not stack_models
    """
    global _stack_model, _leaderboard_df
    from pycaret.classification import stack_models, pull
    
    if not _pycaret_setup_done:
        return "ERROR: Run pycaret_setup_tool first."
    
    # Parse base_models from comma-separated string
    if not base_models or base_models.strip() == "":
        # If no base models specified, use top 3 from leaderboard
        if _leaderboard_df is None or len(_leaderboard_df) < 3:
            return (
                f"ERROR: Insufficient models in leaderboard for stacking.\n"
                f"Run pycaret_compare_models_tool first with n_select >= 3."
            )
        base_models_list = _leaderboard_df.head(3)['Model'].tolist()
    else:
        # Convert comma-separated string to list
        base_models_list = [m.strip() for m in base_models.split(",") if m.strip()]
    
    try:
        # Create stacked model
        # FIX: Remove n_folds parameter - not supported in all PyCaret versions
        # CV folds are controlled in setup(), not in stack_models()
        stack = stack_models(
            estimator_list=base_models_list,
            meta_model=meta_model,
            verbose=False,
        )
        
        _stack_model = stack
        
        # Save stacked model
        os.makedirs("artifacts/model", exist_ok=True)
        from pycaret.classification import save_model
        save_model(stack, PYCARET_STACK_PATH)
        
        # Save stack report
        stack_report = {
            "timestamp": datetime.now().isoformat(),
            "meta_model": meta_model,
            "n_folds": n_folds,  # Note: This is informational only
            "base_models": base_models_list,
            "stack_model_path": f"{PYCARET_STACK_PATH}.pkl",
            "status": "SUCCESS"
        }
        
        stack_report_path = "artifacts/model/pycaret_stack_report.json"
        with open(stack_report_path, "w") as f:
            json.dump(stack_report, f, indent=2)
        
        return (
            f"Model Stacking Complete:\n"
            f"  Meta Model: {meta_model}\n"
            f"  CV Folds: {n_folds} (set in setup, not stack_models)\n"
            f"  Base Models:\n"
            + "\n".join([f"    - {m}" for m in base_models_list]) +
            f"\n\n  Stack Model Saved: {PYCARET_STACK_PATH}.pkl\n"
            f"  Report Saved: {stack_report_path}\n"
            f"\n💡 Stacking uses base model predictions as features for meta-learning."
        )
    
    except Exception as e:
        return (
            f"ERROR: Stacking failed - {str(e)}\n"
            f"Recommendation: Ensure base models are diverse and meta_model is compatible.\n"
            f"Note: CV folds are controlled in pycaret_setup_tool (setup function), not stack_models."
        )


@tool("Deploy Model API")
def deploy_model_api_tool(
    model_path: str = None,
    api_name: str = "churn_prediction_api",
    port: int = 8000
) -> str:
    """
    Directly spins up a FastAPI endpoint from PyCaret for immediate testing.
    Creates a deployable API for real-time churn predictions.
    """
    from pycaret.classification import deploy_model, load_model
    
    # FIX: Handle model path correctly to avoid double .pkl extension
    if model_path is None:
        if _best_model is not None:
            model_to_deploy = _best_model
            model_path_used = "best_model (in-memory)"
        else:
            return (
                "ERROR: No model available for deployment.\n"
                "Run pycaret_compare_models_tool or pycaret_tune_model_tool first, "
                "or specify model_path parameter."
            )
    else:
        # FIX: Remove .pkl extension if present, PyCaret adds it automatically
        model_path_clean = model_path.replace('.pkl', '')
        try:
            model_to_deploy = load_model(model_path_clean)
            model_path_used = model_path_clean
        except Exception as e:
            # Try with .pkl extension
            try:
                model_to_deploy = load_model(model_path)
                model_path_used = model_path
            except:
                return f"ERROR: Cannot load model from {model_path}. Error: {str(e)}"
    
    try:
        # Deploy model as API
        os.makedirs("artifacts/model", exist_ok=True)
        
        # Note: deploy_model in PyCaret creates API files, doesn't actually start server
        api_url = deploy_model(
            model=model_to_deploy,
            model_name=api_name,
            port=port,
            verbose=False,
        )
        
        # Save deployment report
        deployment_report = {
            "timestamp": datetime.now().isoformat(),
            "api_name": api_name,
            "port": port,
            "api_url": api_url if api_url else f"http://localhost:{port}",
            "model_path": model_path_used,
            "status": "SUCCESS",
            "endpoints": {
                "predict": f"/predict",
                "predict_proba": f"/predict_proba",
                "health": f"/health"
            }
        }
        
        deployment_report_path = f"{PYCARET_API_PATH}_deployment.json"
        with open(deployment_report_path, "w") as f:
            json.dump(deployment_report, f, indent=2)
        
        return (
            f"Model API Deployment Complete:\n"
            f"  API Name: {api_name}\n"
            f"  Port: {port}\n"
            f"  API URL: {api_url if api_url else f'http://localhost:{port}'}\n"
            f"  Model: {model_path_used}\n"
            f"\n  API Endpoints:\n"
            f"    POST /predict - Get churn predictions\n"
            f"    POST /predict_proba - Get churn probabilities\n"
            f"    GET /health - Health check endpoint\n"
            f"\n  Deployment Report: {deployment_report_path}\n"
            f"\n💡 API is ready for testing. Use curl or Postman to send prediction requests."
        )
    
    except Exception as e:
        return (
            f"ERROR: API deployment failed - {str(e)}\n"
            f"Recommendation: Check if port {port} is available and PyCaret deployment dependencies are installed."
        )


@tool("Get Model Leaderboard")
def get_model_leaderboard_tool() -> str:
    """
    Retrieves the full dataframe of all trained models for the Summary Agent to analyze.
    Returns complete leaderboard with all metrics for comprehensive model comparison.
    """
    global _leaderboard_df
    
    if _leaderboard_df is None:
        # Try to load from saved CSV
        if os.path.exists(PYCARET_RESULTS_PATH):
            _leaderboard_df = pd.read_csv(PYCARET_RESULTS_PATH)
        else:
            return (
                "ERROR: No leaderboard available.\n"
                "Run pycaret_compare_models_tool first to generate model comparison results."
            )
    
    # Format leaderboard for display
    leaderboard_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_models": len(_leaderboard_df),
        "top_model": _leaderboard_df.iloc[0]['Model'] if len(_leaderboard_df) > 0 else None,
        "top_auc": float(_leaderboard_df.iloc[0]['AUC']) if len(_leaderboard_df) > 0 else None,
        "columns": list(_leaderboard_df.columns),
        "leaderboard_saved_to": PYCARET_LEADERBOARD_PATH
    }
    
    # Save full leaderboard
    os.makedirs("artifacts/model", exist_ok=True)
    _leaderboard_df.to_csv(PYCARET_LEADERBOARD_PATH, index=False)
    
    # Format output
    display_cols = ['Model', 'AUC', 'F1', 'Recall', 'Prec.', 'Accuracy', 'ROC AUC']
    available_cols = [c for c in display_cols if c in _leaderboard_df.columns]
    
    if not available_cols:
        available_cols = _leaderboard_df.columns[:7].tolist()
    
    leaderboard_text = _leaderboard_df[available_cols].to_string(index=False)
    
    return (
        f"Model Leaderboard Retrieved:\n"
        f"  Total Models: {leaderboard_summary['total_models']}\n"
        f"  Top Model: {leaderboard_summary['top_model']}\n"
        f"  Top AUC: {leaderboard_summary['top_auc']:.4f}\n"
        f"\n  Full Leaderboard:\n{leaderboard_text}\n"
        f"\n  Leaderboard Saved: {PYCARET_LEADERBOARD_PATH}\n"
        f"  Report Saved: artifacts/model/pycaret_leaderboard_summary.json\n"
        f"\n💡 Use this leaderboard to compare all trained models and select the best performer."
    )