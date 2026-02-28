import pandas as pd
import numpy as np
import pickle
import json
import os
import time
from datetime import datetime
from crewai.tools import tool

PROCESSED_PATH = "artifacts/data/processed_churn.csv"
FEATURES_PATH = "artifacts/data/selected_features.json"
MODEL_PATH = "artifacts/model/churn_model.pkl"
PREDICTIONS_PATH = "artifacts/data/predictions.csv"
THRESHOLD_PATH = "artifacts/model/optimal_threshold.json"
CALIBRATION_PATH = "artifacts/model/calibration_report.json"
ENSEMBLE_PATH = "artifacts/model/ensemble_models.json"
ONNX_PATH = "artifacts/model/churn_model.onnx"
LATENCY_PATH = "artifacts/model/inference_latency_report.json"


def _load_data_and_features():
    """Helper function to load data and features."""
    df = pd.read_csv(PROCESSED_PATH).dropna()
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    X = df[features]
    y = df["Churn"]
    return X, y, features


def _load_model():
    """Helper function to load the trained model."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


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


# ── NEW MODEL OPTIMIZATION & DEPLOYMENT TOOLS ADDED BELOW ─────────────────────


@tool("Tune Classification Threshold")
def tune_classification_threshold_tool(
    metric_to_optimize: str = "f1",
    threshold_range: str = "0.1,0.9,0.01"  # ✅ FIXED: Changed from tuple to str
) -> str:
    """
    Finds the optimal probability threshold for churn (not just 0.5) based on
    Precision/Recall trade-off. Optimizes for F1, Recall, or custom business metric.
    
    Args:
        metric_to_optimize: Metric to optimize ('f1', 'recall', 'precision')
        threshold_range: Comma-separated string: min,max,step (e.g., "0.1,0.9,0.01")
    """
    from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
    
    if not os.path.exists(MODEL_PATH):
        return "ERROR: Model not found. Run train_model_tool first."
    
    # FIX: Parse threshold_range from comma-separated string
    try:
        threshold_parts = [float(x.strip()) for x in threshold_range.split(",")]
        if len(threshold_parts) != 3:
            raise ValueError("threshold_range must have 3 values: min,max,step")
        min_thresh, max_thresh, step = threshold_parts
    except Exception as e:
        return f"ERROR: Invalid threshold_range format. Use 'min,max,step' (e.g., '0.1,0.9,0.01'). Error: {str(e)}"
    
    # Load test data
    X_te = pd.read_csv("artifacts/data/X_test.csv")
    y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    # Get predicted probabilities
    y_prob = model.predict_proba(X_te)[:, 1]
    
    # Search for optimal threshold
    thresholds = np.arange(min_thresh, max_thresh + step, step)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        precision = precision_score(y_te, y_pred, zero_division=0)
        recall = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        
        results.append({
            "threshold": round(thresh, 3),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold based on metric
    if metric_to_optimize == "f1":
        best_idx = results_df["f1"].idxmax()
    elif metric_to_optimize == "recall":
        best_idx = results_df["recall"].idxmax()
    elif metric_to_optimize == "precision":
        best_idx = results_df["precision"].idxmax()
    else:
        # Custom: maximize F1 with recall >= 0.6
        filtered = results_df[results_df["recall"] >= 0.6]
        if len(filtered) > 0:
            best_idx = filtered["f1"].idxmax()
        else:
            best_idx = results_df["f1"].idxmax()
    
    best_result = results_df.loc[best_idx]
    default_result = results_df.loc[results_df["threshold"].sub(0.5).abs().idxmin()]
    
    # Calculate improvement over default 0.5 threshold
    improvement = {
        "f1": best_result["f1"] - default_result["f1"],
        "recall": best_result["recall"] - default_result["recall"],
        "precision": best_result["precision"] - default_result["precision"]
    }
    
    # Save optimal threshold
    threshold_report = {
        "timestamp": datetime.now().isoformat(),
        "metric_optimized": metric_to_optimize,
        "threshold_range": {"min": min_thresh, "max": max_thresh, "step": step},
        "optimal_threshold": round(float(best_result["threshold"]), 3),
        "default_threshold": 0.5,
        "optimal_metrics": {
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"]
        },
        "default_metrics": {
            "precision": default_result["precision"],
            "recall": default_result["recall"],
            "f1": default_result["f1"]
        },
        "improvement": {
            "f1_delta": round(float(improvement["f1"]), 4),
            "recall_delta": round(float(improvement["recall"]), 4),
            "precision_delta": round(float(improvement["precision"]), 4)
        },
        "all_thresholds": results
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(threshold_report, f, indent=2)
    
    return (
        f"Classification Threshold Tuning Complete:\n"
        f"Metric Optimized: {metric_to_optimize}\n"
        f"Threshold Range: {min_thresh} to {max_thresh} (step: {step})\n"
        f"\nOptimal Threshold: {best_result['threshold']:.3f}\n"
        f"Default Threshold: 0.500\n"
        f"\nOptimal Metrics:\n"
        f"  Precision: {best_result['precision']:.4f}\n"
        f"  Recall:    {best_result['recall']:.4f}\n"
        f"  F1 Score:  {best_result['f1']:.4f}\n"
        f"\nDefault (0.5) Metrics:\n"
        f"  Precision: {default_result['precision']:.4f}\n"
        f"  Recall:    {default_result['recall']:.4f}\n"
        f"  F1 Score:  {default_result['f1']:.4f}\n"
        f"\nImprovement over Default:\n"
        f"  F1:       {improvement['f1']:+.4f}\n"
        f"  Recall:   {improvement['recall']:+.4f}\n"
        f"  Precision: {improvement['precision']:+.4f}\n"
        f"\nRecommendation: Use threshold {best_result['threshold']:.3f} for production.\n"
        f"Report saved to: {THRESHOLD_PATH}"
    )


@tool("Calibrate Model Probabilities")
def calibrate_model_probabilities_tool(method: str = "isotonic") -> str:
    """
    Applies Platt Scaling or Isotonic Regression to ensure predicted probabilities
    are accurate and well-calibrated (P(churn) = actual churn rate).
    """
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.metrics import brier_score_loss
    
    if not os.path.exists(MODEL_PATH):
        return "ERROR: Model not found. Run train_model_tool first."
    
    # Load data
    X_te = pd.read_csv("artifacts/data/X_test.csv")
    y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
    X_tr = pd.read_csv("artifacts/data/X_train.csv")
    y_tr = pd.read_csv("artifacts/data/y_train.csv").squeeze()
    
    with open(MODEL_PATH, "rb") as f:
        base_model = pickle.load(f)
    
    # Calibrate model
    if method == "isotonic":
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    else:  # sigmoid (Platt scaling)
        calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
    
    calibrated_model.fit(X_tr, y_tr)
    
    # Get predictions from both models
    y_prob_base = base_model.predict_proba(X_te)[:, 1]
    y_prob_calib = calibrated_model.predict_proba(X_te)[:, 1]
    
    # Calculate calibration metrics
    brier_base = brier_score_loss(y_te, y_prob_base)
    brier_calib = brier_score_loss(y_te, y_prob_calib)
    
    # Calibration curve
    prob_true, prob_pred_base = calibration_curve(y_te, y_prob_base, n_bins=10)
    prob_true, prob_pred_calib = calibration_curve(y_te, y_prob_calib, n_bins=10)
    
    # Calculate calibration error (mean absolute difference)
    calib_error_base = np.mean(np.abs(prob_pred_base - prob_true))
    calib_error_calib = np.mean(np.abs(prob_pred_calib - prob_true))
    
    # Save calibrated model
    os.makedirs("artifacts/model", exist_ok=True)
    calibrated_model_path = "artifacts/model/churn_model_calibrated.pkl"
    with open(calibrated_model_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    
    # Save calibration report
    calibration_report = {
        "timestamp": datetime.now().isoformat(),
        "calibration_method": method,
        "base_model_brier_score": round(float(brier_base), 6),
        "calibrated_model_brier_score": round(float(brier_calib), 6),
        "brier_score_improvement": round(float(brier_base - brier_calib), 6),
        "base_model_calibration_error": round(float(calib_error_base), 6),
        "calibrated_model_calibration_error": round(float(calib_error_calib), 6),
        "calibration_error_improvement": round(float(calib_error_base - calib_error_calib), 6),
        "calibration_curve_data": {
            "probability_bins": prob_true.tolist(),
            "base_model_predicted": prob_pred_base.tolist(),
            "calibrated_model_predicted": prob_pred_calib.tolist()
        },
        "calibrated_model_path": calibrated_model_path
    }
    
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(calibration_report, f, indent=2)
    
    improvement_pct = ((brier_base - brier_calib) / brier_base) * 100 if brier_base > 0 else 0
    
    return (
        f"Model Probability Calibration Complete:\n"
        f"Method: {method}\n"
        f"\nBrier Score (lower is better):\n"
        f"  Base Model:      {brier_base:.6f}\n"
        f"  Calibrated Model: {brier_calib:.6f}\n"
        f"  Improvement:      {brier_base - brier_calib:.6f} ({improvement_pct:.1f}%)\n"
        f"\nCalibration Error (mean |predicted - actual|):\n"
        f"  Base Model:      {calib_error_base:.6f}\n"
        f"  Calibrated Model: {calib_error_calib:.6f}\n"
        f"  Improvement:      {calib_error_base - calib_error_calib:.6f}\n"
        f"\nCalibrated model saved to: {calibrated_model_path}\n"
        f"Report saved to: {CALIBRATION_PATH}\n"
        f"\n💡 {'✅ Calibration improved probability accuracy' if brier_calib < brier_base else '⚠️ Minimal calibration improvement'}"
    )


@tool("Create Model Ensemble")
def create_model_ensemble_tool(
    model_types: str = "random_forest,gradient_boosting,logistic_regression",
    voting: str = "soft"
) -> str:
    """
    Averages predictions from top 3 models to reduce variance and improve robustness.
    Supports soft voting (probability averaging) or hard voting (majority vote).
    
    Args:
        model_types: Comma-separated list of model types (e.g., "random_forest,gradient_boosting,logistic_regression")
        voting: Voting method - "soft" or "hard" (default: "soft")
    """
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score
    
    # Parse model_types from comma-separated string
    if not model_types or model_types.strip() == "":
        model_types_list = ["random_forest", "gradient_boosting", "logistic_regression"]
    else:
        model_types_list = [m.strip() for m in model_types.split(",") if m.strip()]
    
    X, y, features = _load_data_and_features()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define base models
    base_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    
    # Train individual models and collect results
    individual_results = {}
    estimators = []
    
    for model_name in model_types_list:
        if model_name not in base_models:
            continue
        
        model = base_models[model_name]
        model.fit(X_tr, y_tr)
        
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)
        
        individual_results[model_name] = {
            "auc": round(float(roc_auc_score(y_te, y_prob)), 4),
            "f1": round(float(f1_score(y_te, y_pred)), 4)
        }
        
        estimators.append((model_name, model))  # model_types_list used above
    
    # Create ensemble
    ensemble = VotingClassifier(estimators=estimators, voting=voting)
    ensemble.fit(X_tr, y_tr)
    
    # Evaluate ensemble
    y_prob_ensemble = ensemble.predict_proba(X_te)[:, 1]
    y_pred_ensemble = ensemble.predict(X_te)
    
    ensemble_auc = round(float(roc_auc_score(y_te, y_prob_ensemble)), 4)
    ensemble_f1 = round(float(f1_score(y_te, y_pred_ensemble)), 4)
    
    # Find best individual model
    best_model = max(individual_results.items(), key=lambda x: x[1]["auc"])
    
    # Save ensemble
    os.makedirs("artifacts/model", exist_ok=True)
    ensemble_path = "artifacts/model/churn_model_ensemble.pkl"
    with open(ensemble_path, "wb") as f:
        pickle.dump(ensemble, f)
    
    # Save ensemble report
    ensemble_report = {
        "timestamp": datetime.now().isoformat(),
        "voting_method": voting,
        "models_included": model_types,
        "individual_model_performance": individual_results,
        "best_individual_model": best_model[0],
        "best_individual_auc": best_model[1]["auc"],
        "ensemble_performance": {
            "auc": ensemble_auc,
            "f1": ensemble_f1
        },
        "improvement_over_best": {
            "auc_delta": round(ensemble_auc - best_model[1]["auc"], 4),
            "f1_delta": round(ensemble_f1 - best_model[1]["f1"], 4)
        },
        "ensemble_model_path": ensemble_path
    }
    
    with open(ENSEMBLE_PATH, "w") as f:
        json.dump(ensemble_report, f, indent=2)
    
    improvement = ensemble_auc - best_model[1]["auc"]
    
    return (
        f"Model Ensemble Created:\n"
        f"Voting Method: {voting}\n"
        f"Models Included: {model_types}\n"
        f"\nIndividual Model Performance:\n"
        + "\n".join([f"  {name}: AUC={perf['auc']}, F1={perf['f1']}" 
                    for name, perf in individual_results.items()]) +
        f"\n\nBest Individual Model: {best_model[0]} (AUC={best_model[1]['auc']})\n"
        f"\nEnsemble Performance:\n"
        f"  AUC: {ensemble_auc}\n"
        f"  F1:  {ensemble_f1}\n"
        f"\nImprovement over Best Individual:\n"
        f"  AUC Delta: {improvement:+.4f}\n"
        f"  F1 Delta:  {ensemble_f1 - best_model[1]['f1']:+.4f}\n"
        f"\nEnsemble model saved to: {ensemble_path}\n"
        f"Report saved to: {ENSEMBLE_PATH}\n"
        f"\n💡 {'✅ Ensemble improved performance' if improvement > 0 else '⚠️ Ensemble similar to best individual'}"
    )


@tool("Export Model ONNX")
def export_model_onnx_tool() -> str:
    """
    Saves the model in ONNX (Open Neural Network Exchange) format for deployment
    in non-Python environments (Java, C#, JavaScript, mobile, edge devices).
    """
    try:
        import skl2onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        return (
            "ERROR: skl2onnx package not installed.\n"
            "Install with: pip install skl2onnx\n"
            "ONNX export requires this package for model conversion."
        )
    
    if not os.path.exists(MODEL_PATH):
        return "ERROR: Model not found. Run train_model_tool first."
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    # Load features to get input shape
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    
    n_features = len(features)
    
    # Define initial type for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    try:
        # Convert to ONNX
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # Save ONNX model
        os.makedirs("artifacts/model", exist_ok=True)
        with open(ONNX_PATH, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        onnx_size = os.path.getsize(ONNX_PATH) / 1024
        
        # Verify ONNX model
        import onnx
        onnx.checker.check_model(onnx_model)
        
        export_report = {
            "timestamp": datetime.now().isoformat(),
            "original_model_path": MODEL_PATH,
            "onnx_model_path": ONNX_PATH,
            "original_model_type": type(model).__name__,
            "n_features": n_features,
            "onnx_file_size_kb": round(onnx_size, 2),
            "onnx_version": onnx.__version__,
            "skl2onnx_version": skl2onnx.__version__,
            "export_status": "SUCCESS"
        }
        
        return (
            f"Model Exported to ONNX Format:\n"
            f"Original Model: {type(model).__name__}\n"
            f"Number of Features: {n_features}\n"
            f"\nONNX Model Details:\n"
            f"  Path: {ONNX_PATH}\n"
            f"  Size: {onnx_size:.2f} KB\n"
            f"  ONNX Version: {onnx.__version__}\n"
            f"  skl2onnx Version: {skl2onnx.__version__}\n"
            f"\n✅ Model successfully exported and validated.\n"
            f"\nDeployment Options:\n"
            f"  - Java: Use onnxruntime-java\n"
            f"  - C#: Use Microsoft.ML.OnnxRuntime\n"
            f"  - JavaScript: Use onnxruntime-web\n"
            f"  - Mobile: Use onnxruntime-mobile\n"
            f"  - Edge: Use onnxruntime for IoT devices\n"
            f"\nReport saved to: {ONNX_PATH}"
        )
    
    except Exception as e:
        return (
            f"ERROR: ONNX export failed.\n"
            f"Model type {type(model).__name__} may not be supported.\n"
            f"Error: {str(e)}\n"
            f"\nSupported models: RandomForest, GradientBoosting, LogisticRegression, etc."
        )


@tool("Measure Inference Latency")
def measure_inference_latency_tool(n_iterations: int = 100) -> str:
    """
    Times how long a prediction takes to ensure it meets SLA requirements.
    Measures both single prediction and batch prediction latency.
    """
    if not os.path.exists(MODEL_PATH):
        return "ERROR: Model not found. Run train_model_tool first."
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    
    # Load test data
    X_te = pd.read_csv("artifacts/data/X_test.csv")[features]
    
    # Warm up
    _ = model.predict(X_te.iloc[:1])
    
    # Measure single prediction latency
    single_latencies = []
    for i in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X_te.iloc[[i % len(X_te)]])
        end = time.perf_counter()
        single_latencies.append((end - start) * 1000)  # Convert to milliseconds
    
    # Measure batch prediction latency
    batch_latencies = []
    batch_size = min(100, len(X_te))
    for i in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X_te.iloc[i:i+batch_size])
        end = time.perf_counter()
        batch_latencies.append((end - start) * 1000)
    
    # Calculate statistics
    single_stats = {
        "mean_ms": round(np.mean(single_latencies), 4),
        "std_ms": round(np.std(single_latencies), 4),
        "min_ms": round(np.min(single_latencies), 4),
        "max_ms": round(np.max(single_latencies), 4),
        "p50_ms": round(np.percentile(single_latencies, 50), 4),
        "p95_ms": round(np.percentile(single_latencies, 95), 4),
        "p99_ms": round(np.percentile(single_latencies, 99), 4)
    }
    
    batch_stats = {
        "mean_ms": round(np.mean(batch_latencies), 4),
        "std_ms": round(np.std(batch_latencies), 4),
        "min_ms": round(np.min(batch_latencies), 4),
        "max_ms": round(np.max(batch_latencies), 4),
        "p50_ms": round(np.percentile(batch_latencies, 50), 4),
        "p95_ms": round(np.percentile(batch_latencies, 95), 4),
        "p99_ms": round(np.percentile(batch_latencies, 99), 4),
        "batch_size": batch_size,
        "per_sample_ms": round(np.mean(batch_latencies) / batch_size, 6)
    }
    
    # SLA assessment (typical SLA: <100ms for single prediction)
    sla_threshold_ms = 100
    meets_sla = bool(single_stats["p95_ms"] < sla_threshold_ms)  # FIX: Convert numpy bool to Python bool
    
    # Save latency report
    latency_report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "n_iterations": n_iterations,
        "single_prediction_latency": single_stats,
        "batch_prediction_latency": batch_stats,
        "sla_threshold_ms": sla_threshold_ms,
        "meets_sla": bool(meets_sla),
        "sla_status": "✅ PASS" if meets_sla else "❌ FAIL"  # meets_sla is now Python bool
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    with open(LATENCY_PATH, "w") as f:
        json.dump(latency_report, f, indent=2)
    
    return (
        f"Inference Latency Measurement Complete:\n"
        f"Iterations: {n_iterations}\n"
        f"\nSingle Prediction Latency:\n"
        f"  Mean:   {single_stats['mean_ms']:.4f} ms\n"
        f"  Std:    {single_stats['std_ms']:.4f} ms\n"
        f"  Min:    {single_stats['min_ms']:.4f} ms\n"
        f"  Max:    {single_stats['max_ms']:.4f} ms\n"
        f"  P50:    {single_stats['p50_ms']:.4f} ms\n"
        f"  P95:    {single_stats['p95_ms']:.4f} ms\n"
        f"  P99:    {single_stats['p99_ms']:.4f} ms\n"
        f"\nBatch Prediction Latency (batch_size={batch_size}):\n"
        f"  Mean:        {batch_stats['mean_ms']:.4f} ms\n"
        f"  Per Sample:  {batch_stats['per_sample_ms']:.6f} ms\n"
        f"  P95:         {batch_stats['p95_ms']:.4f} ms\n"
        f"\nSLA Assessment (threshold: {sla_threshold_ms}ms at P95):\n"
        f"  Status: {single_stats['p95_ms']:.4f} ms {'<=' if meets_sla else '>'} {sla_threshold_ms} ms\n"
        f"  Result: {'✅ PASS - Meets SLA requirements' if meets_sla else '❌ FAIL - Exceeds SLA threshold'}\n"
        f"\nReport saved to: {LATENCY_PATH}\n"
        f"\n💡 {'Model is production-ready for real-time inference' if meets_sla else 'Consider model optimization for production'}"
    )