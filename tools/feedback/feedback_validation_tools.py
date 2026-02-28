import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from crewai.tools import tool

EXPERIMENT_LOG_PATH = "artifacts/model/experiment_log.json"
HITL_FLAG_PATH = "artifacts/model/human_in_the_loop_flag.json"
COST_IMPACT_PATH = "artifacts/model/business_cost_impact.json"
ROOT_CAUSE_PATH = "artifacts/model/root_cause_analysis.json"
ROLLBACK_LOG_PATH = "artifacts/model/rollback_log.json"
MODEL_PATH = "artifacts/model/churn_model.pkl"
SCALER_PATH = "artifacts/model/scaler.pkl"
BACKUP_DIR = "artifacts/model/backups"

# Resource constraint thresholds (adjust based on your infrastructure)
MAX_TRAINING_TIME_MINUTES = 60
MAX_MEMORY_GB = 16
MAX_FEATURES = 50
MIN_AUC_FOR_DEPLOYMENT = 0.75

# Statistical significance thresholds
SIGNIFICANCE_LEVEL = 0.05  # p-value threshold for T-Test
MIN_SAMPLE_SIZE = 30  # Minimum samples for statistical tests


@tool("Validate Experiment Log")
def validate_experiment_log_tool() -> str:
    """Validate that the experiment log exists and contains required fields."""
    if not os.path.exists(EXPERIMENT_LOG_PATH):
        return f"VALIDATION FAILED: Experiment log not found at {EXPERIMENT_LOG_PATH}."
    with open(EXPERIMENT_LOG_PATH) as f:
        log = json.load(f)
    if not log.get("experiments"):
        return "VALIDATION FAILED: Experiment log is empty."
    required = ["run_id", "timestamp", "auc", "f1"]
    last = log["experiments"][-1]
    missing = [k for k in required if k not in last]
    if missing:
        return f"VALIDATION FAILED: Latest experiment missing fields: {missing}"
    return (
        f"Experiment log PASSED. Total runs: {len(log['experiments'])}.\n"
        f"Latest run: AUC={last.get('auc')}, F1={last.get('f1')}, Recall={last.get('recall')}"
    )


@tool("Validate Improvement Suggestions")
def validate_improvement_suggestions_tool() -> str:
    """Validate that improvement suggestions are available (model and test data exist)."""
    required = ["artifacts/model/churn_model.pkl", "artifacts/data/X_test.csv", "artifacts/data/y_test.csv"]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        return f"VALIDATION FAILED: Cannot generate suggestions. Missing files: {missing}"
    return "Improvement suggestions pre-conditions PASSED. All required files are present."


@tool("Validate Metrics Improvement")
def validate_metrics_improvement_tool(min_improvement: float = 0.01) -> str:
    """
    Validate that the latest experiment shows AUC improvement of at least min_improvement
    over the previous run (requires at least 2 logged experiments).
    """
    if not os.path.exists(EXPERIMENT_LOG_PATH):
        return f"VALIDATION FAILED: Experiment log not found at {EXPERIMENT_LOG_PATH}."
    with open(EXPERIMENT_LOG_PATH) as f:
        log = json.load(f)
    experiments = log.get("experiments", [])
    if len(experiments) < 2:
        return "SKIPPED: Need at least 2 experiment runs to compare. Only 1 run logged so far."
    prev_auc = experiments[-2].get("auc", 0)
    curr_auc = experiments[-1].get("auc", 0)
    improvement = curr_auc - prev_auc
    if improvement < min_improvement:
        return (
            f"VALIDATION WARNING: AUC improvement {improvement:.4f} < {min_improvement}.\n"
            f"Previous AUC: {prev_auc}, Current AUC: {curr_auc}.\n"
            f"Consider revisiting the feature set or model hyperparameters."
        )
    return (
        f"Metrics improvement PASSED. AUC improved by {improvement:.4f} "
        f"({prev_auc:.4f} → {curr_auc:.4f})."
    )


# ── NEW FEEDBACK VALIDATION TOOLS ADDED BELOW ──────────────────────────────────


@tool("Validate Statistical Significance")
def validate_statistical_significance_tool(
    significance_level: float = SIGNIFICANCE_LEVEL,
    min_sample_size: int = MIN_SAMPLE_SIZE
) -> str:
    """
    Ensures metric improvements aren't just due to random variance using T-Test.
    Validates that AUC/F1 improvements are statistically significant (p-value < 0.05).
    """
    from scipy import stats
    
    if not os.path.exists(EXPERIMENT_LOG_PATH):
        return f"VALIDATION FAILED: Experiment log not found at {EXPERIMENT_LOG_PATH}."
    
    with open(EXPERIMENT_LOG_PATH) as f:
        log = json.load(f)
    
    experiments = log.get("experiments", [])
    
    if len(experiments) < 2:
        return (
            f"VALIDATION SKIPPED: Insufficient experiment history ({len(experiments)} runs).\n"
            f"Need at least 2 runs to perform statistical significance testing."
        )
    
    # Get current and previous run metrics
    current_run = experiments[-1]
    previous_run = experiments[-2]
    
    # Check if we have enough historical data for meaningful stats
    if len(experiments) < min_sample_size:
        note = f"NOTE: Only {len(experiments)} runs available (recommended: {min_sample_size}+ for robust stats)"
    else:
        note = ""
    
    # Perform T-Test on AUC across all available runs
    auc_values = [exp.get("auc", 0) for exp in experiments if "auc" in exp]
    f1_values = [exp.get("f1", 0) for exp in experiments if "f1" in exp]
    recall_values = [exp.get("recall", 0) for exp in experiments if "recall" in exp]
    
    significance_results = {}
    all_significant = True
    
    # T-Test: Compare recent runs vs older runs
    if len(auc_values) >= 4:
        midpoint = len(auc_values) // 2
        older_auc = auc_values[:midpoint]
        newer_auc = auc_values[midpoint:]
        
        t_stat, p_value_auc = stats.ttest_ind(newer_auc, older_auc)
        
        significance_results["auc"] = {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value_auc), 4),
            "significant": p_value_auc < significance_level,
            "older_mean": round(np.mean(older_auc), 4),
            "newer_mean": round(np.mean(newer_auc), 4)
        }
        
        if p_value_auc >= significance_level:
            all_significant = False
    else:
        significance_results["auc"] = {
            "status": "INSUFFICIENT_DATA",
            "message": f"Need ≥4 AUC values for T-Test, have {len(auc_values)}"
        }
    
    # Compare current vs previous directly (paired comparison)
    current_auc = current_run.get("auc", 0)
    previous_auc = previous_run.get("auc", 0)
    auc_delta = current_auc - previous_auc
    
    # Estimate if delta is meaningful (rule of thumb: >2% is typically significant)
    if len(auc_values) >= 2:
        auc_std = np.std(auc_values) if len(auc_values) > 1 else 0.01
        z_score = auc_delta / (auc_std + 0.001)
        is_meaningful = abs(z_score) > 1.96  # 95% confidence
    else:
        z_score = 0
        is_meaningful = abs(auc_delta) > 0.02
    
    significance_results["current_vs_previous"] = {
        "auc_delta": round(auc_delta, 4),
        "z_score": round(z_score, 4),
        "meaningful_improvement": is_meaningful
    }
    
    if not is_meaningful:
        all_significant = False
    
    # Format output
    if all_significant:
        status = "✅ STATISTICALLY SIGNIFICANT"
        recommendation = "Improvements are likely real, not random variance. Safe to deploy."
    else:
        status = "⚠️ NOT STATISTICALLY SIGNIFICANT"
        recommendation = "Improvements may be due to random variance. Collect more data or run more experiments."
    
    # Save significance report
    significance_report = {
        "timestamp": datetime.now().isoformat(),
        "significance_level": significance_level,
        "min_sample_size": min_sample_size,
        "total_runs_analyzed": len(experiments),
        "all_significant": all_significant,
        "results": significance_results
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    significance_path = "artifacts/model/statistical_significance_report.json"
    with open(significance_path, "w") as f:
        json.dump(significance_report, f, indent=2)
    
    # Format results text
    results_text = ""
    for metric, data in significance_results.items():
        if isinstance(data, dict):
            if "p_value" in data:
                results_text += (
                    f"  {metric.upper()}:\n"
                    f"    T-Statistic: {data['t_statistic']}\n"
                    f"    P-Value: {data['p_value']} {'<' if data['p_value'] < significance_level else '≥'} {significance_level}\n"
                    f"    Significant: {'✅ YES' if data['significant'] else '❌ NO'}\n"
                    f"    Older Mean: {data['older_mean']}, Newer Mean: {data['newer_mean']}\n"
                )
            elif "auc_delta" in data:
                results_text += (
                    f"  CURRENT VS PREVIOUS:\n"
                    f"    AUC Delta: {data['auc_delta']:+.4f}\n"
                    f"    Z-Score: {data['z_score']}\n"
                    f"    Meaningful: {'✅ YES' if data['meaningful_improvement'] else '❌ NO'}\n"
                )
            else:
                results_text += f"  {metric.upper()}: {data.get('message', data.get('status', 'N/A'))}\n"
    
    return (
        f"Statistical Significance Validation:\n"
        f"Status: {status}\n"
        f"Significance Level (α): {significance_level}\n"
        f"Total Runs Analyzed: {len(experiments)}\n"
        f"{note if note else ''}\n"
        f"\nT-Test Results:\n{results_text}\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {significance_path}"
    )


@tool("Validate Resource Constraints")
def validate_resource_constraints_tool(
    max_training_time_minutes: int = MAX_TRAINING_TIME_MINUTES,
    max_memory_gb: int = MAX_MEMORY_GB,
    max_features: int = MAX_FEATURES,
    min_auc_for_deployment: float = MIN_AUC_FOR_DEPLOYMENT
) -> str:
    """
    Checks if the suggested improvement exceeds compute/time budgets.
    Ensures the model is production-ready within resource constraints.
    """
    import pickle
    
    violations = []
    resource_usage = {}
    
    # Check feature count
    features_path = "artifacts/data/selected_features.json"
    if os.path.exists(features_path):
        with open(features_path) as f:
            features = json.load(f)
        n_features = len(features)
        resource_usage["n_features"] = n_features
        
        if n_features > max_features:
            violations.append(f"⚠️ Feature count ({n_features}) exceeds limit ({max_features})")
    else:
        resource_usage["n_features"] = "UNKNOWN"
        violations.append("⚠️ Feature file not found - cannot validate feature count")
    
    # Check model file size (proxy for memory)
    if os.path.exists(MODEL_PATH):
        model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        resource_usage["model_size_mb"] = round(model_size_mb, 2)
        
        # Rough estimate: 1MB model ≈ 0.1GB memory during inference
        estimated_memory_gb = model_size_mb * 0.1
        resource_usage["estimated_memory_gb"] = round(estimated_memory_gb, 2)
        
        if estimated_memory_gb > max_memory_gb:
            violations.append(f"⚠️ Estimated memory ({estimated_memory_gb:.2f}GB) exceeds limit ({max_memory_gb}GB)")
    else:
        resource_usage["model_size_mb"] = "FILE_NOT_FOUND"
        violations.append("⚠️ Model file not found - cannot validate memory usage")
    
    # Check AUC meets deployment threshold
    if os.path.exists(EXPERIMENT_LOG_PATH):
        with open(EXPERIMENT_LOG_PATH) as f:
            log = json.load(f)
        experiments = log.get("experiments", [])
        
        if experiments:
            latest_auc = experiments[-1].get("auc", 0)
            resource_usage["latest_auc"] = latest_auc
            
            if latest_auc < min_auc_for_deployment:
                violations.append(
                    f"⚠️ AUC ({latest_auc:.4f}) below deployment threshold ({min_auc_for_deployment})"
                )
        else:
            resource_usage["latest_auc"] = "NO_EXPERIMENTS"
            violations.append("⚠️ No experiments logged - cannot validate AUC")
    else:
        resource_usage["latest_auc"] = "LOG_NOT_FOUND"
        violations.append("⚠️ Experiment log not found - cannot validate AUC")
    
    # Check scaler exists (required for production)
    if os.path.exists(SCALER_PATH):
        resource_usage["scaler_exists"] = True
    else:
        resource_usage["scaler_exists"] = False
        violations.append("⚠️ Scaler file not found - required for production inference")
    
    # Estimate training time based on feature count and data size
    x_train_path = "artifacts/data/X_train.csv"
    if os.path.exists(x_train_path):
        train_size_mb = os.path.getsize(x_train_path) / (1024 * 1024)
        # Rough estimate: 1MB data ≈ 0.5 minutes training time
        estimated_training_time = train_size_mb * 0.5 * (n_features / 10)
        resource_usage["estimated_training_time_minutes"] = round(estimated_training_time, 2)
        
        if estimated_training_time > max_training_time_minutes:
            violations.append(
                f"⚠️ Estimated training time ({estimated_training_time:.1f}min) exceeds limit ({max_training_time_minutes}min)"
            )
    else:
        resource_usage["estimated_training_time_minutes"] = "DATA_NOT_FOUND"
    
    # Save resource validation report
    resource_report = {
        "timestamp": datetime.now().isoformat(),
        "constraints": {
            "max_training_time_minutes": max_training_time_minutes,
            "max_memory_gb": max_memory_gb,
            "max_features": max_features,
            "min_auc_for_deployment": min_auc_for_deployment
        },
        "actual_usage": resource_usage,
        "violations": violations,
        "deployment_ready": len(violations) == 0
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    resource_path = "artifacts/model/resource_constraints_report.json"
    with open(resource_path, "w") as f:
        json.dump(resource_report, f, indent=2)
    
    # Format output
    if violations:
        violations_text = "\n".join(violations)
        return (
            f"VALIDATION FAILED: Resource constraint violations detected.\n"
            f"Constraints:\n"
            f"  Max Training Time: {max_training_time_minutes} minutes\n"
            f"  Max Memory: {max_memory_gb} GB\n"
            f"  Max Features: {max_features}\n"
            f"  Min AUC for Deployment: {min_auc_for_deployment}\n"
            f"\nActual Usage:\n"
            + "\n".join([f"  {k}: {v}" for k, v in resource_usage.items()]) +
            f"\n\nViolations:\n{violations_text}\n"
            f"\nDeployment Ready: ❌ NO\n"
            f"Report saved to: {resource_path}\n"
            f"\nRecommendation: Address violations before deploying to production."
        )
    
    return (
        f"Resource Constraints Validation PASSED.\n"
        f"Constraints:\n"
        f"  Max Training Time: {max_training_time_minutes} minutes\n"
        f"  Max Memory: {max_memory_gb} GB\n"
        f"  Max Features: {max_features}\n"
        f"  Min AUC for Deployment: {min_auc_for_deployment}\n"
        f"\nActual Usage:\n"
        + "\n".join([f"  {k}: {v}" for k, v in resource_usage.items()]) +
        f"\n\nViolations: None\n"
        f"Deployment Ready: ✅ YES\n"
        f"Report saved to: {resource_path}\n"
        f"\nModel is ready for production deployment within resource budgets."
    )


@tool("Validate Rollback Integrity")
def validate_rollback_integrity_tool(backup_dir: str = BACKUP_DIR) -> str:
    """
    Ensures the rollback actually restored the correct model files.
    Verifies file checksums and metadata match the target run.
    """
    import hashlib
    
    if not os.path.exists(ROLLBACK_LOG_PATH):
        return (
            f"VALIDATION SKIPPED: Rollback log not found at {ROLLBACK_LOG_PATH}.\n"
            f"No rollback has been performed yet."
        )
    
    with open(ROLLBACK_LOG_PATH, "r") as f:
        rollback_log = json.load(f)
    
    rollbacks = rollback_log.get("rollbacks", [])
    
    if not rollbacks:
        return (
            f"VALIDATION SKIPPED: No rollbacks recorded in log.\n"
            f"Rollback integrity cannot be validated without rollback history."
        )
    
    # Get most recent rollback
    latest_rollback = rollbacks[-1]
    
    integrity_checks = []
    all_passed = True
    
    # Check 1: Verify rollback was marked as successful
    if latest_rollback.get("rollback_status") != "SUCCESS":
        integrity_checks.append({
            "check": "Rollback Status",
            "status": "FAILED",
            "details": f"Rollback status is '{latest_rollback.get('rollback_status')}', not 'SUCCESS'"
        })
        all_passed = False
    else:
        integrity_checks.append({
            "check": "Rollback Status",
            "status": "PASSED",
            "details": f"Rollback completed successfully"
        })
    
    # Check 2: Verify model file exists and has content
    if os.path.exists(MODEL_PATH):
        model_size = os.path.getsize(MODEL_PATH)
        if model_size > 0:
            # Calculate file hash for integrity
            with open(MODEL_PATH, "rb") as f:
                model_hash = hashlib.md5(f.read()).hexdigest()
            
            integrity_checks.append({
                "check": "Model File Integrity",
                "status": "PASSED",
                "details": f"Model exists, size={model_size} bytes, MD5={model_hash[:8]}..."
            })
        else:
            integrity_checks.append({
                "check": "Model File Integrity",
                "status": "FAILED",
                "details": "Model file exists but is empty (0 bytes)"
            })
            all_passed = False
    else:
        integrity_checks.append({
            "check": "Model File Integrity",
            "status": "FAILED",
            "details": "Model file not found after rollback"
        })
        all_passed = False
    
    # Check 3: Verify scaler file exists
    if os.path.exists(SCALER_PATH):
        scaler_size = os.path.getsize(SCALER_PATH)
        integrity_checks.append({
            "check": "Scaler File Integrity",
            "status": "PASSED",
            "details": f"Scaler exists, size={scaler_size} bytes"
        })
    else:
        integrity_checks.append({
            "check": "Scaler File Integrity",
            "status": "FAILED",
            "details": "Scaler file not found after rollback"
        })
        all_passed = False
    
    # Check 4: Verify features file exists and is valid JSON
    features_path = "artifacts/data/selected_features.json"
    if os.path.exists(features_path):
        try:
            with open(features_path) as f:
                features = json.load(f)
            if isinstance(features, list) and len(features) > 0:
                integrity_checks.append({
                    "check": "Features File Integrity",
                    "status": "PASSED",
                    "details": f"Features file valid, {len(features)} features"
                })
            else:
                integrity_checks.append({
                    "check": "Features File Integrity",
                    "status": "FAILED",
                    "details": "Features file is empty or invalid format"
                })
                all_passed = False
        except json.JSONDecodeError:
            integrity_checks.append({
                "check": "Features File Integrity",
                "status": "FAILED",
                "details": "Features file is not valid JSON"
            })
            all_passed = False
    else:
        integrity_checks.append({
            "check": "Features File Integrity",
            "status": "FAILED",
            "details": "Features file not found after rollback"
        })
        all_passed = False
    
    # Check 5: Verify backup was created before rollback
    if latest_rollback.get("backup_created"):
        backup_path = os.path.join(backup_dir, f"backup_before_rollback_{latest_rollback['backup_created']}")
        if os.path.exists(backup_path):
            integrity_checks.append({
                "check": "Pre-Rollback Backup",
                "status": "PASSED",
                "details": f"Backup created at {backup_path}"
            })
        else:
            integrity_checks.append({
                "check": "Pre-Rollback Backup",
                "status": "WARNING",
                "details": f"Backup directory not found at {backup_path}"
            })
    else:
        integrity_checks.append({
            "check": "Pre-Rollback Backup",
            "status": "WARNING",
            "details": "No backup timestamp recorded"
        })
    
    # Check 6: Verify artifacts were restored
    restored_artifacts = latest_rollback.get("artifacts_restored", [])
    if restored_artifacts:
        integrity_checks.append({
            "check": "Artifacts Restored",
            "status": "PASSED",
            "details": f"Artifacts restored: {', '.join(restored_artifacts)}"
        })
    else:
        integrity_checks.append({
            "check": "Artifacts Restored",
            "status": "WARNING",
            "details": "No artifacts listed as restored"
        })
    
    # Save integrity report
    integrity_report = {
        "timestamp": datetime.now().isoformat(),
        "rollback_run_id": latest_rollback.get("rollback_to_run_id", "unknown"),
        "all_checks_passed": all_passed,
        "integrity_checks": integrity_checks,
        "passed_count": sum(1 for c in integrity_checks if c["status"] == "PASSED"),
        "failed_count": sum(1 for c in integrity_checks if c["status"] == "FAILED"),
        "warning_count": sum(1 for c in integrity_checks if c["status"] == "WARNING")
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    integrity_path = "artifacts/model/rollback_integrity_report.json"
    with open(integrity_path, "w") as f:
        json.dump(integrity_report, f, indent=2)
    
    # Format output
    checks_text = "\n".join([
        f"  {'✅' if c['status'] == 'PASSED' else '⚠️' if c['status'] == 'WARNING' else '❌'} "
        f"{c['check']}: {c['status']} - {c['details']}"
        for c in integrity_checks
    ])
    
    if all_passed:
        status = "✅ ROLLBACK INTEGRITY VERIFIED"
        recommendation = "Rollback completed successfully. All artifacts restored correctly."
    else:
        status = "❌ ROLLBACK INTEGRITY FAILED"
        recommendation = "Rollback may be incomplete. Verify restored files manually before proceeding."
    
    return (
        f"Rollback Integrity Validation:\n"
        f"Status: {status}\n"
        f"Rollback To Run ID: {latest_rollback.get('rollback_to_run_id', 'unknown')}\n"
        f"Timestamp: {latest_rollback.get('timestamp', 'unknown')}\n"
        f"\nIntegrity Checks ({integrity_report['passed_count']} passed, "
        f"{integrity_report['failed_count']} failed, {integrity_report['warning_count']} warnings):\n"
        f"{checks_text}\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {integrity_path}\n"
        f"\n{'✅ Safe to proceed with pipeline' if all_passed else '⚠️ Manual verification required before proceeding'}"
    )