import pandas as pd
import os
import json
import pickle
import requests
import numpy as np
from crewai.tools import tool

PYCARET_MODEL_PATH = "artifacts/model/pycaret_best_model.pkl"
PYCARET_RESULTS_PATH = "artifacts/model/pycaret_compare_results.csv"
PYCARET_API_DEPLOYMENT_PATH = "artifacts/model/pycaret_api_deployment.json"
PYCARET_FULL_LEADERBOARD_PATH = "artifacts/model/pycaret_full_leaderboard.csv"
PYCARET_BLEND_MODEL_PATH = "artifacts/model/pycaret_blend_model.pkl"
PYCARET_STACK_MODEL_PATH = "artifacts/model/pycaret_stack_model.pkl"
X_TEST_PATH = "artifacts/data/X_test.csv"
Y_TEST_PATH = "artifacts/data/y_test.csv"
PROCESSED_PATH = "artifacts/data/processed_churn.csv"

# Validation thresholds
API_TIMEOUT_SECONDS = 10
API_HEALTH_ENDPOINT = "/health"
CONSISTENCY_TOLERANCE = 0.0001  # Max difference between PyCaret and sklearn predictions


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


# ── NEW PYCARET VALIDATION TOOLS ADDED BELOW ──────────────────────────────────


@tool("Validate Deployment Health")
def validate_deployment_health_tool(
    api_url: str = None,
    timeout: int = API_TIMEOUT_SECONDS,
    health_endpoint: str = API_HEALTH_ENDPOINT
) -> str:
    """
    Pings the deployed API endpoint to ensure it returns 200 OK.
    Validates that the deployed model API is healthy and responding.
    """
    # Try to load API deployment info
    if api_url is None:
        if os.path.exists(PYCARET_API_DEPLOYMENT_PATH):
            with open(PYCARET_API_DEPLOYMENT_PATH, "r") as f:
                deployment_info = json.load(f)
            api_url = deployment_info.get("api_url", f"http://localhost:{deployment_info.get('port', 8000)}")
        else:
            return (
                f"VALIDATION SKIPPED: API deployment info not found at {PYCARET_API_DEPLOYMENT_PATH}.\n"
                f"Run deploy_model_api_tool first before validating deployment health."
            )
    
    # Construct health endpoint URL
    health_url = f"{api_url.rstrip('/')}{health_endpoint}"
    
    validation_results = []
    all_healthy = True
    
    # Check 1: Health endpoint
    try:
        response = requests.get(health_url, timeout=timeout)
        if response.status_code == 200:
            validation_results.append({
                "endpoint": health_endpoint,
                "status_code": response.status_code,
                "status": "HEALTHY",
                "response_time_ms": round(response.elapsed.total_seconds() * 1000, 2)
            })
        else:
            validation_results.append({
                "endpoint": health_endpoint,
                "status_code": response.status_code,
                "status": "UNHEALTHY",
                "response_time_ms": round(response.elapsed.total_seconds() * 1000, 2)
            })
            all_healthy = False
    except requests.exceptions.ConnectionError as e:
        validation_results.append({
            "endpoint": health_endpoint,
            "status_code": "N/A",
            "status": "CONNECTION_FAILED",
            "error": str(e)
        })
        all_healthy = False
    except requests.exceptions.Timeout as e:
        validation_results.append({
            "endpoint": health_endpoint,
            "status_code": "N/A",
            "status": "TIMEOUT",
            "error": f"Request timed out after {timeout}s"
        })
        all_healthy = False
    except Exception as e:
        validation_results.append({
            "endpoint": health_endpoint,
            "status_code": "N/A",
            "status": "ERROR",
            "error": str(e)
        })
        all_healthy = False
    
    # Check 2: Predict endpoint (if health is OK)
    if all_healthy and os.path.exists(X_TEST_PATH):
        try:
            # Load sample data for prediction test
            X_test = pd.read_csv(X_TEST_PATH).head(1)
            sample_data = X_test.to_dict(orient="records")[0]
            
            predict_url = f"{api_url.rstrip('/')}/predict"
            response = requests.post(predict_url, json=sample_data, timeout=timeout)
            
            if response.status_code == 200:
                prediction_result = response.json()
                validation_results.append({
                    "endpoint": "/predict",
                    "status_code": response.status_code,
                    "status": "WORKING",
                    "prediction": prediction_result
                })
            else:
                validation_results.append({
                    "endpoint": "/predict",
                    "status_code": response.status_code,
                    "status": "FAILED",
                    "error": response.text
                })
                all_healthy = False
        except Exception as e:
            validation_results.append({
                "endpoint": "/predict",
                "status_code": "N/A",
                "status": "ERROR",
                "error": str(e)
            })
            all_healthy = False
    
    # Save health check report
    health_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "api_url": api_url,
        "timeout_seconds": timeout,
        "all_healthy": all_healthy,
        "endpoint_checks": validation_results
    }
    
    health_report_path = "artifacts/model/api_health_check_report.json"
    os.makedirs("artifacts/model", exist_ok=True)
    with open(health_report_path, "w") as f:
        json.dump(health_report, f, indent=2)
    
    # Format output
    checks_text = "\n".join([
        f"  {r['endpoint']}: {r['status']} "
        f"({'✅' if r['status'] in ['HEALTHY', 'WORKING'] else '❌'}) "
        f"- Status Code: {r.get('status_code', 'N/A')}"
        + (f", Response Time: {r.get('response_time_ms', 'N/A')}ms" if 'response_time_ms' in r else "")
        for r in validation_results
    ])
    
    if all_healthy:
        status = "✅ API HEALTHY"
        recommendation = "Deployed API is responding correctly. Ready for production use."
    else:
        status = "❌ API UNHEALTHY"
        recommendation = "API deployment has issues. Check server logs and redeploy if necessary."
    
    return (
        f"Deployment Health Validation:\n"
        f"Status: {status}\n"
        f"API URL: {api_url}\n"
        f"Timeout: {timeout}s\n"
        f"\nEndpoint Checks:\n{checks_text}\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {health_report_path}"
    )


@tool("Validate Inference Consistency")
def validate_inference_consistency_tool(
    tolerance: float = CONSISTENCY_TOLERANCE,
    n_samples: int = 10
) -> str:
    """
    Compares PyCaret prediction vs. raw Scikit-Learn prediction to ensure
    wrapper isn't breaking logic. Validates prediction consistency.
    """
    # Check required files
    required_files = {
        "PyCaret model": PYCARET_MODEL_PATH,
        "Test features": X_TEST_PATH,
        "Test labels": Y_TEST_PATH,
    }
    
    missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
    if missing_files:
        return (
            f"VALIDATION FAILED: Missing required files: {', '.join(missing_files)}\n"
            f"Ensure PyCaret model is saved and test data exists."
        )
    
    # Load PyCaret model
    try:
        with open(PYCARET_MODEL_PATH, "rb") as f:
            pycaret_model = pickle.load(f)
    except Exception as e:
        return f"VALIDATION FAILED: Cannot load PyCaret model: {str(e)}"
    
    # Load test data
    X_test = pd.read_csv(X_TEST_PATH).head(n_samples)
    y_test = pd.read_csv(Y_TEST_PATH).head(n_samples)
    
    # Get PyCaret predictions
    try:
        from pycaret.classification import predict_model
        pycaret_pred = predict_model(pycaret_model, data=X_test)
        pycaret_labels = pycaret_pred["prediction_label"].values
        pycaret_scores = pycaret_pred["prediction_score"].values if "prediction_score" in pycaret_pred.columns else None
    except Exception as e:
        return f"VALIDATION FAILED: PyCaret prediction failed: {str(e)}"
    
    # Get raw sklearn predictions (if model has underlying estimator)
    sklearn_labels = None
    sklearn_scores = None
    sklearn_comparison_possible = False
    
    try:
        # Try to access underlying sklearn model from PyCaret pipeline
        if hasattr(pycaret_model, 'named_steps') and 'trained_model' in pycaret_model.named_steps:
            sklearn_model = pycaret_model.named_steps['trained_model']
        elif hasattr(pycaret_model, '_final_estimator'):
            sklearn_model = pycaret_model._final_estimator
        elif hasattr(pycaret_model, 'estimator_'):
            sklearn_model = pycaret_model.estimator_
        else:
            # PyCaret model might already be the sklearn model
            sklearn_model = pycaret_model
        
        # Make predictions
        sklearn_labels = sklearn_model.predict(X_test)
        if hasattr(sklearn_model, 'predict_proba'):
            sklearn_scores = sklearn_model.predict_proba(X_test)[:, 1]
        
        sklearn_comparison_possible = True
    except Exception as e:
        sklearn_comparison_possible = False
        sklearn_error = str(e)
    
    # Compare predictions
    consistency_results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_samples_tested": len(X_test),
        "tolerance": tolerance,
        "pycaret_predictions": {
            "labels": pycaret_labels.tolist(),
            "scores": pycaret_scores.tolist() if pycaret_scores is not None else None
        },
        "sklearn_predictions": {
            "labels": sklearn_labels.tolist() if sklearn_labels is not None else None,
            "scores": sklearn_scores.tolist() if sklearn_scores is not None else None,
            "comparison_possible": sklearn_comparison_possible
        },
        "consistency_check": {
            "labels_match": None,
            "scores_max_diff": None,
            "all_within_tolerance": None
        }
    }
    
    # Check label consistency
    if sklearn_labels is not None:
        labels_match = np.array_equal(pycaret_labels, sklearn_labels)
        consistency_results["consistency_check"]["labels_match"] = labels_match
        
        # Check score consistency
        if pycaret_scores is not None and sklearn_scores is not None:
            max_diff = np.max(np.abs(pycaret_scores - sklearn_scores))
            all_within_tolerance = max_diff <= tolerance
            consistency_results["consistency_check"]["scores_max_diff"] = round(float(max_diff), 6)
            consistency_results["consistency_check"]["all_within_tolerance"] = all_within_tolerance
        else:
            consistency_results["consistency_check"]["scores_max_diff"] = "N/A"
            consistency_results["consistency_check"]["all_within_tolerance"] = "N/A"
    else:
        consistency_results["consistency_check"]["labels_match"] = "SKLEARN_NOT_AVAILABLE"
        consistency_results["consistency_check"]["scores_max_diff"] = "N/A"
        consistency_results["consistency_check"]["all_within_tolerance"] = "N/A"
    
    # Save consistency report
    consistency_report_path = "artifacts/model/inference_consistency_report.json"
    os.makedirs("artifacts/model", exist_ok=True)
    with open(consistency_report_path, "w") as f:
        json.dump(consistency_results, f, indent=2)
    
    # Determine validation status
    if sklearn_comparison_possible:
        labels_ok = consistency_results["consistency_check"]["labels_match"]
        scores_ok = consistency_results["consistency_check"]["all_within_tolerance"]
        
        if labels_ok and (scores_ok == "N/A" or scores_ok):
            status = "✅ CONSISTENT"
            recommendation = "PyCaret wrapper is producing consistent predictions with underlying sklearn model."
        else:
            status = "❌ INCONSISTENT"
            issues = []
            if not labels_ok:
                issues.append("Prediction labels don't match")
            if scores_ok is False:
                issues.append(f"Score difference ({consistency_results['consistency_check']['scores_max_diff']}) > tolerance ({tolerance})")
            recommendation = f"Issues detected: {', '.join(issues)}. Review PyCaret preprocessing pipeline."
    else:
        status = "⚠️ SKLEARN COMPARISON NOT POSSIBLE"
        recommendation = f"Cannot access underlying sklearn model: {sklearn_error}. PyCaret predictions generated successfully."
    
    # Format output
    output_lines = [
        f"Inference Consistency Validation:",
        f"Status: {status}",
        f"Samples Tested: {len(X_test)}",
        f"Tolerance: {tolerance}",
        f"",
        f"PyCaret Predictions:",
        f"  Labels: {pycaret_labels.tolist()}",
        f"  Scores: {pycaret_scores.tolist() if pycaret_scores is not None else 'N/A'}",
    ]
    
    if sklearn_comparison_possible:
        output_lines.extend([
            f"",
            f"Sklearn Predictions:",
            f"  Labels: {sklearn_labels.tolist()}",
            f"  Scores: {sklearn_scores.tolist() if sklearn_scores is not None else 'N/A'}",
            f"",
            f"Consistency Check:",
            f"  Labels Match: {'✅ YES' if labels_ok else '❌ NO'}",
            f"  Max Score Difference: {consistency_results['consistency_check']['scores_max_diff']}",
            f"  Within Tolerance: {'✅ YES' if scores_ok == True else '❌ NO' if scores_ok == False else 'N/A'}",
        ])
    else:
        output_lines.extend([
            f"",
            f"Sklearn Comparison: Not possible ({sklearn_error})",
        ])
    
    output_lines.extend([
        f"",
        f"Recommendation: {recommendation}",
        f"Report saved to: {consistency_report_path}"
    ])
    
    return "\n".join(output_lines)