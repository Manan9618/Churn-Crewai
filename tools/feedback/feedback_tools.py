import pandas as pd
import json
import os
import shutil
from datetime import datetime
from crewai.tools import tool

EXPERIMENT_LOG_PATH = "artifacts/model/experiment_log.json"
FEATURES_PATH = "artifacts/data/selected_features.json"
HITL_FLAG_PATH = "artifacts/model/human_in_the_loop_flag.json"
COST_IMPACT_PATH = "artifacts/model/business_cost_impact.json"
ROLLBACK_LOG_PATH = "artifacts/model/rollback_log.json"
ROOT_CAUSE_PATH = "artifacts/model/root_cause_analysis.json"
BACKUP_DIR = "artifacts/model/backups"

# Business cost assumptions (adjust based on your business context)
CUSTOMER_LIFETIME_VALUE = 500  # Average lifetime value per customer
CHURN_COST = 300  # Cost of losing a customer (acquisition + lost revenue)
RETENTION_CAMPAIGN_COST = 50  # Cost per retention campaign
FALSE_NEGATIVE_COST = 400  # Cost of missing a churner (high priority)
FALSE_POSITIVE_COST = 50  # Cost of false alarm (unnecessary campaign)


def _load_log():
    """Helper function to load experiment log."""
    if os.path.exists(EXPERIMENT_LOG_PATH):
        with open(EXPERIMENT_LOG_PATH) as f:
            return json.load(f)
    return {"experiments": []}


def _save_log(log):
    """Helper function to save experiment log."""
    os.makedirs("artifacts/model", exist_ok=True)
    with open(EXPERIMENT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


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
    _save_log(log)

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


# ── NEW FEEDBACK LOOP TOOLS ADDED BELOW ──────────────────────────────────────


@tool("Trigger Human In The Loop")
def trigger_human_in_the_loop_tool(
    auc_threshold: float = 0.75,
    recall_threshold: float = 0.60,
    max_degradation_pct: float = 5.0
) -> str:
    """
    If metrics degrade beyond a threshold, this tool flags a human for review
    instead of auto-retrying. Prevents automated pipelines from making things worse.
    """
    log = _load_log()
    
    if len(log["experiments"]) < 2:
        return (
            "HUMAN REVIEW: NOT TRIGGERED\n"
            f"Reason: Insufficient experiment history ({len(log['experiments'])} runs).\n"
            f"Need at least 2 runs to detect degradation.\n"
            f"Recommendation: Continue automated pipeline until more data is collected."
        )
    
    # Get current and previous run metrics
    current_run = log["experiments"][-1]
    previous_run = log["experiments"][-2]
    
    # Calculate degradation
    degradation = {}
    triggers = []
    
    for metric in ["auc", "f1", "recall"]:
        if metric in current_run and metric in previous_run:
            current_val = current_run[metric]
            previous_val = previous_run[metric]
            pct_change = ((current_val - previous_val) / previous_val) * 100
            
            degradation[metric] = {
                "current": current_val,
                "previous": previous_val,
                "change_pct": round(pct_change, 2)
            }
            
            # Check absolute thresholds
            if metric == "auc" and current_val < auc_threshold:
                triggers.append(f"⚠️ AUC below threshold: {current_val} < {auc_threshold}")
            
            if metric == "recall" and current_val < recall_threshold:
                triggers.append(f"⚠️ Recall below threshold: {current_val} < {recall_threshold}")
            
            # Check degradation threshold
            if pct_change < -max_degradation_pct:
                triggers.append(f"⚠️ {metric.upper()} degraded by {abs(pct_change):.2f}% (max allowed: {max_degradation_pct}%)")
    
    # Create HITL flag
    hitl_flag = {
        "triggered": len(triggers) > 0,
        "timestamp": datetime.now().isoformat(),
        "current_run_id": current_run.get("run_id", "unknown"),
        "previous_run_id": previous_run.get("run_id", "unknown"),
        "triggers": triggers,
        "degradation_details": degradation,
        "recommended_action": "MANUAL_REVIEW" if len(triggers) > 0 else "CONTINUE_AUTO"
    }
    
    # Save HITL flag
    os.makedirs("artifacts/model", exist_ok=True)
    with open(HITL_FLAG_PATH, "w") as f:
        json.dump(hitl_flag, f, indent=2)
    
    if triggers:
        triggers_text = "\n".join(triggers)
        return (
            f"🚨 HUMAN IN THE LOOP TRIGGERED 🚨\n"
            f"Timestamp: {hitl_flag['timestamp']}\n"
            f"Current Run: {hitl_flag['current_run_id']}\n"
            f"Previous Run: {hitl_flag['previous_run_id']}\n"
            f"\nDegradation Detected:\n{triggers_text}\n"
            f"\nDegradation Details:\n"
            + "\n".join([f"  {k}: {v['previous']} → {v['current']} ({v['change_pct']:+.2f}%)" 
                        for k, v in degradation.items()]) +
            f"\n\nRecommended Action: {hitl_flag['recommended_action']}\n"
            f"Flag saved to: {HITL_FLAG_PATH}\n"
            f"\n⚠️ AUTOMATED PIPELINE PAUSED - Manual review required before proceeding."
        )
    
    return (
        f"✅ HUMAN IN THE LOOP: NOT TRIGGERED\n"
        f"Timestamp: {hitl_flag['timestamp']}\n"
        f"Current Run: {hitl_flag['current_run_id']}\n"
        f"Previous Run: {hitl_flag['previous_run_id']}\n"
        f"\nNo significant degradation detected.\n"
        f"Degradation Details:\n"
        + "\n".join([f"  {k}: {v['previous']} → {v['current']} ({v['change_pct']:+.2f}%)" 
                    for k, v in degradation.items()]) +
        f"\n\nRecommended Action: {hitl_flag['recommended_action']}\n"
        f"Flag saved to: {HITL_FLAG_PATH}\n"
        f"\n✅ Automated pipeline can continue."
    )


@tool("Calculate Business Cost Impact")
def calculate_business_cost_impact_tool(
    customer_lifetime_value: float = CUSTOMER_LIFETIME_VALUE,
    churn_cost: float = CHURN_COST,
    retention_campaign_cost: float = RETENTION_CAMPAIGN_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST
) -> str:
    """
    Translates metric changes (e.g., Recall +5%) into estimated dollar value/cost.
    Helps business stakeholders understand the financial impact of model improvements.
    """
    import pickle
    from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score
    
    log = _load_log()
    
    # Get current model metrics
    model_path = "artifacts/model/churn_model.pkl"
    x_test_path = "artifacts/data/X_test.csv"
    y_test_path = "artifacts/data/y_test.csv"
    
    if not (os.path.exists(model_path) and os.path.exists(x_test_path) and os.path.exists(y_test_path)):
        return (
            "ERROR: Cannot calculate business impact.\n"
            f"Missing files: model={os.path.exists(model_path)}, "
            f"X_test={os.path.exists(x_test_path)}, y_test={os.path.exists(y_test_path)}\n"
            "Train a model first before calculating business impact."
        )
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    
    X_te = pd.read_csv(x_test_path)[features]
    y_te = pd.read_csv(y_test_path).squeeze()
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    
    # Calculate business costs
    # False Negatives: Missed churners (most costly)
    fn_cost = fn * false_negative_cost
    
    # False Positives: Unnecessary retention campaigns
    fp_cost = fp * false_positive_cost
    
    # True Positives: Successfully retained customers
    tp_savings = tp * (churn_cost - retention_campaign_cost)
    
    # True Negatives: Correctly identified non-churners (no action needed)
    tn_savings = tn * 0  # No cost, no savings
    
    # Total impact
    total_cost = fn_cost + fp_cost
    total_savings = tp_savings + tn_savings
    net_impact = total_savings - total_cost
    
    # Calculate per-customer metrics
    total_customers = len(y_te)
    churn_rate_actual = y_te.sum() / total_customers
    churn_rate_predicted = y_pred.sum() / total_customers
    
    # Compare with previous run if available
    comparison = {}
    if len(log["experiments"]) >= 2:
        current_run = log["experiments"][-1]
        previous_run = log["experiments"][-2]
        
        for metric in ["auc", "f1", "recall"]:
            if metric in current_run and metric in previous_run:
                delta = current_run[metric] - previous_run[metric]
                comparison[metric] = {
                    "current": current_run[metric],
                    "previous": previous_run[metric],
                    "delta": round(delta, 4)
                }
        
        # Estimate financial impact of metric changes
        if "recall" in comparison:
            recall_delta = comparison["recall"]["delta"]
            # Each 1% recall improvement ≈ preventing 1% more churners from leaving
            estimated_churners_saved = int(total_customers * churn_rate_actual * recall_delta)
            estimated_savings = estimated_churners_saved * churn_cost
            comparison["recall"]["estimated_financial_impact"] = round(estimated_savings, 2)
    
    # Save cost impact report
    cost_report = {
        "timestamp": datetime.now().isoformat(),
        "test_set_size": total_customers,
        "actual_churn_rate": round(churn_rate_actual, 4),
        "predicted_churn_rate": round(churn_rate_predicted, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "cost_breakdown": {
            "false_negative_cost": round(fn_cost, 2),
            "false_positive_cost": round(fp_cost, 2),
            "true_positive_savings": round(tp_savings, 2),
            "total_cost": round(total_cost, 2),
            "total_savings": round(total_savings, 2),
            "net_business_impact": round(net_impact, 2)
        },
        "cost_assumptions": {
            "customer_lifetime_value": customer_lifetime_value,
            "churn_cost": churn_cost,
            "retention_campaign_cost": retention_campaign_cost,
            "false_negative_cost": false_negative_cost,
            "false_positive_cost": false_positive_cost
        },
        "metric_comparison": comparison
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    with open(COST_IMPACT_PATH, "w") as f:
        json.dump(cost_report, f, indent=2)
    
    # Format output
    comparison_text = ""
    if comparison:
        comparison_text = "\nMetric Changes vs Previous Run:\n"
        for metric, data in comparison.items():
            financial = f" (Est. Financial Impact: ${data.get('estimated_financial_impact', 0):,.2f})" if "estimated_financial_impact" in data else ""
            comparison_text += f"  {metric}: {data['previous']} → {data['current']} ({data['delta']:+.4f}){financial}\n"
    
    return (
        f"Business Cost Impact Analysis:\n"
        f"Timestamp: {cost_report['timestamp']}\n"
        f"Test Set Size: {total_customers} customers\n"
        f"Actual Churn Rate: {churn_rate_actual*100:.2f}%\n"
        f"Predicted Churn Rate: {churn_rate_predicted*100:.2f}%\n"
        f"\nConfusion Matrix:\n"
        f"  True Negatives:  {tn} (correctly identified non-churners)\n"
        f"  False Positives: {fp} (unnecessary campaigns)\n"
        f"  False Negatives: {fn} (missed churners - HIGH COST)\n"
        f"  True Positives:  {tp} (successfully identified churners)\n"
        f"\nFinancial Impact:\n"
        f"  False Negative Cost:  ${fn_cost:,.2f} ({fn} × ${false_negative_cost})\n"
        f"  False Positive Cost:  ${fp_cost:,.2f} ({fp} × ${false_positive_cost})\n"
        f"  True Positive Savings: ${tp_savings:,.2f} ({tp} × ${churn_cost - retention_campaign_cost})\n"
        f"  ─────────────────────────────────────\n"
        f"  Total Cost:           ${total_cost:,.2f}\n"
        f"  Total Savings:        ${total_savings:,.2f}\n"
        f"  NET BUSINESS IMPACT:  ${net_impact:,.2f}\n"
        f"{comparison_text}"
        f"\nCost Assumptions:\n"
        f"  Customer Lifetime Value: ${customer_lifetime_value}\n"
        f"  Cost of Churn: ${churn_cost}\n"
        f"  Retention Campaign Cost: ${retention_campaign_cost}\n"
        f"\nReport saved to: {COST_IMPACT_PATH}\n"
        f"\n💡 Business Recommendation: "
        f"{'✅ Model generates positive ROI' if net_impact > 0 else '⚠️ Model costs exceed savings - consider improvements'}"
    )


@tool("Rollback To Previous Version")
def rollback_to_previous_version_tool(
    backup_dir: str = BACKUP_DIR,
    restore_run_id: int = None
) -> str:
    """
    If the new run is worse, this tool reverts artifacts to the last known good state.
    Creates backups before rollback for audit trail.
    """
    log = _load_log()
    
    if len(log["experiments"]) < 2:
        return (
            "ROLLBACK: NOT POSSIBLE\n"
            f"Reason: Insufficient experiment history ({len(log['experiments'])} runs).\n"
            f"Need at least 2 runs to rollback to previous version."
        )
    
    # Determine which run to rollback to
    if restore_run_id is None:
        # Default to previous run
        target_run = log["experiments"][-2]
    else:
        # Find specific run
        target_run = None
        for exp in log["experiments"]:
            if exp.get("run_id") == restore_run_id:
                target_run = exp
                break
        
        if target_run is None:
            return f"ERROR: Run ID {restore_run_id} not found in experiment log."
    
    # Define artifacts to rollback
    artifacts_to_rollback = {
        "model": "artifacts/model/churn_model.pkl",
        "scaler": "artifacts/model/scaler.pkl",
        "features": "artifacts/data/selected_features.json",
        "predictions": "artifacts/data/predictions.csv"
    }
    
    # Create backup of current state
    os.makedirs(backup_dir, exist_ok=True)
    backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_backup_dir = os.path.join(backup_dir, f"backup_before_rollback_{backup_timestamp}")
    os.makedirs(current_backup_dir, exist_ok=True)
    
    backed_up = []
    for artifact_name, artifact_path in artifacts_to_rollback.items():
        if os.path.exists(artifact_path):
            # Copy to backup
            backup_path = os.path.join(current_backup_dir, os.path.basename(artifact_path))
            if os.path.isfile(artifact_path):
                shutil.copy2(artifact_path, backup_path)
            else:
                shutil.copytree(artifact_path, backup_path)
            backed_up.append(artifact_name)
    
    # Check if backup exists for target run
    backup_run_dir = os.path.join(backup_dir, f"run_{target_run['run_id']}")
    
    rollback_result = {
        "timestamp": datetime.now().isoformat(),
        "current_run_id": log["experiments"][-1].get("run_id", "unknown"),
        "rollback_to_run_id": target_run["run_id"],
        "backup_created": backup_timestamp,
        "backed_up_artifacts": backed_up,
        "rollback_status": "PENDING",
        "artifacts_restored": []
    }
    
    # Attempt rollback if backup exists
    if os.path.exists(backup_run_dir):
        restored = []
        for artifact_name, artifact_path in artifacts_to_rollback.items():
            backup_file = os.path.join(backup_run_dir, os.path.basename(artifact_path))
            if os.path.exists(backup_file):
                # Create directory if needed
                os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
                
                if os.path.isfile(backup_file):
                    shutil.copy2(backup_file, artifact_path)
                else:
                    if os.path.exists(artifact_path):
                        shutil.rmtree(artifact_path)
                    shutil.copytree(backup_file, artifact_path)
                
                restored.append(artifact_name)
        
        rollback_result["rollback_status"] = "SUCCESS"
        rollback_result["artifacts_restored"] = restored
        
        rollback_message = (
            f"✅ ROLLBACK SUCCESSFUL\n"
            f"Timestamp: {rollback_result['timestamp']}\n"
            f"Rolled back from Run {rollback_result['current_run_id']} to Run {rollback_result['rollback_to_run_id']}\n"
            f"Backup of current state saved to: {current_backup_dir}\n"
            f"Artifacts Restored:\n"
            + "\n".join([f"  ✅ {a}" for a in restored]) +
            f"\n\n⚠️ Remember to re-run validation tests after rollback."
        )
    else:
        rollback_result["rollback_status"] = "NO_BACKUP_FOUND"
        rollback_message = (
            f"⚠️ ROLLBACK: NO BACKUP FOUND\n"
            f"Timestamp: {rollback_result['timestamp']}\n"
            f"Target Run: {rollback_result['rollback_to_run_id']}\n"
            f"Backup Directory: {backup_run_dir}\n"
            f"\nNo backup found for Run {target_run['run_id']}.\n"
            f"Backups are created automatically during experiment runs.\n"
            f"Current state backed up to: {current_backup_dir}\n"
            f"\nRecommendation: Manually restore from previous deployment or retrain model."
        )
    
    # Save rollback log
    os.makedirs("artifacts/model", exist_ok=True)
    
    # Load existing rollback log or create new
    if os.path.exists(ROLLBACK_LOG_PATH):
        with open(ROLLBACK_LOG_PATH, "r") as f:
            rollback_log = json.load(f)
    else:
        rollback_log = {"rollbacks": []}
    
    rollback_log["rollbacks"].append(rollback_result)
    
    with open(ROLLBACK_LOG_PATH, "w") as f:
        json.dump(rollback_log, f, indent=2)
    
    return rollback_message + f"\n\nRollback log saved to: {ROLLBACK_LOG_PATH}"


@tool("Generate Root Cause Hypothesis")
def generate_root_cause_hypothesis_tool() -> str:
    """
    Analyzes logs to suggest why performance dropped (e.g., "Data drift detected in Column X").
    Helps teams quickly identify and fix issues.
    """
    import pickle
    from sklearn.metrics import roc_auc_score, f1_score, recall_score
    
    log = _load_log()
    
    if len(log["experiments"]) < 2:
        return (
            "ROOT CAUSE ANALYSIS: NOT POSSIBLE\n"
            f"Reason: Insufficient experiment history ({len(log['experiments'])} runs).\n"
            f"Need at least 2 runs to compare and identify root causes."
        )
    
    current_run = log["experiments"][-1]
    previous_run = log["experiments"][-2]
    
    # Calculate metric changes
    metric_changes = {}
    degradation_detected = False
    
    for metric in ["auc", "f1", "recall"]:
        if metric in current_run and metric in previous_run:
            delta = current_run[metric] - previous_run[metric]
            pct_change = (delta / previous_run[metric]) * 100 if previous_run[metric] > 0 else 0
            
            metric_changes[metric] = {
                "current": current_run[metric],
                "previous": previous_run[metric],
                "delta": round(delta, 4),
                "pct_change": round(pct_change, 2)
            }
            
            if delta < -0.02:  # More than 2% degradation
                degradation_detected = True
    
    # Generate hypotheses
    hypotheses = []
    evidence = []
    
    # Hypothesis 1: Data Drift
    drift_evidence = []
    baseline_path = "artifacts/data/baseline_churn.csv"
    current_path = "artifacts/data/processed_churn.csv"
    
    if os.path.exists(baseline_path) and os.path.exists(current_path):
        try:
            baseline_df = pd.read_csv(baseline_path)
            current_df = pd.read_csv(current_path)
            
            # Check for distribution shifts in key features
            key_features = ["tenure", "MonthlyCharges", "TotalCharges"]
            for feat in key_features:
                if feat in baseline_df.columns and feat in current_df.columns:
                    baseline_mean = baseline_df[feat].mean()
                    current_mean = current_df[feat].mean()
                    pct_shift = abs((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
                    
                    if pct_shift > 10:  # More than 10% shift
                        drift_evidence.append(f"{feat}: {baseline_mean:.2f} → {current_mean:.2f} ({pct_shift:.1f}% shift)")
            
            if drift_evidence:
                hypotheses.append({
                    "cause": "Data Drift",
                    "confidence": "HIGH" if len(drift_evidence) > 2 else "MEDIUM",
                    "evidence": drift_evidence,
                    "recommendation": "Retrain model with recent data or implement drift monitoring"
                })
        except Exception as e:
            evidence.append(f"Drift analysis error: {str(e)}")
    
    # Hypothesis 2: Feature Changes
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            current_features = json.load(f)
        
        # Check if feature list changed
        prev_features = previous_run.get("features", [])
        if prev_features and set(current_features) != set(prev_features):
            added = set(current_features) - set(prev_features)
            removed = set(prev_features) - set(current_features)
            
            hypotheses.append({
                "cause": "Feature Set Changes",
                "confidence": "HIGH",
                "evidence": [
                    f"Features added: {list(added)}" if added else "No features added",
                    f"Features removed: {list(removed)}" if removed else "No features removed"
                ],
                "recommendation": "Review feature engineering changes and validate new features"
            })
    
    # Hypothesis 3: Model Change
    if current_run.get("model_type") != previous_run.get("model_type"):
        hypotheses.append({
            "cause": "Model Algorithm Change",
            "confidence": "MEDIUM",
            "evidence": [
                f"Previous model: {previous_run.get('model_type', 'unknown')}",
                f"Current model: {current_run.get('model_type', 'unknown')}"
            ],
            "recommendation": "Compare model hyperparameters and validate new algorithm"
        })
    
    # Hypothesis 4: Sample Size Issues
    x_test_path = "artifacts/data/X_test.csv"
    if os.path.exists(x_test_path):
        test_df = pd.read_csv(x_test_path)
        if len(test_df) < 100:
            hypotheses.append({
                "cause": "Small Test Set",
                "confidence": "MEDIUM",
                "evidence": [f"Test set size: {len(test_df)} samples (recommended: ≥100)"],
                "recommendation": "Increase test set size for more reliable metrics"
            })
    
    # Hypothesis 5: Class Imbalance
    y_test_path = "artifacts/data/y_test.csv"
    if os.path.exists(y_test_path):
        y_te = pd.read_csv(y_test_path).squeeze()
        churn_rate = y_te.mean()
        
        if churn_rate < 0.20 or churn_rate > 0.50:
            hypotheses.append({
                "cause": "Class Imbalance",
                "confidence": "MEDIUM",
                "evidence": [f"Test set churn rate: {churn_rate*100:.2f}% (optimal: 20-50%)"],
                "recommendation": "Apply SMOTE or class weights to handle imbalance"
            })
    
    # If no specific hypotheses, provide general recommendations
    if not hypotheses:
        hypotheses.append({
            "cause": "Unknown / Multiple Factors",
            "confidence": "LOW",
            "evidence": ["No single clear cause identified"],
            "recommendation": "Review all pipeline stages: data quality, feature engineering, model training"
        })
    
    # Save root cause analysis
    root_cause_report = {
        "timestamp": datetime.now().isoformat(),
        "current_run_id": current_run.get("run_id", "unknown"),
        "previous_run_id": previous_run.get("run_id", "unknown"),
        "degradation_detected": degradation_detected,
        "metric_changes": metric_changes,
        "hypotheses": hypotheses,
        "evidence": evidence
    }
    
    os.makedirs("artifacts/model", exist_ok=True)
    with open(ROOT_CAUSE_PATH, "w") as f:
        json.dump(root_cause_report, f, indent=2)
    
    # Format output
    metric_text = "\n".join([
        f"  {k}: {v['previous']} → {v['current']} ({v['delta']:+.4f}, {v['pct_change']:+.2f}%)"
        for k, v in metric_changes.items()
    ])
    
    hypotheses_text = "\n".join([
        f"\n🔍 Hypothesis {i+1}: {h['cause']} (Confidence: {h['confidence']})\n"
        f"  Evidence:\n" + "\n".join([f"    - {e}" for e in h['evidence']]) +
        f"  Recommendation: {h['recommendation']}"
        for i, h in enumerate(hypotheses)
    ])
    
    return (
        f"Root Cause Analysis Report:\n"
        f"Timestamp: {root_cause_report['timestamp']}\n"
        f"Comparing Run {root_cause_report['previous_run_id']} → Run {root_cause_report['current_run_id']}\n"
        f"\nDegradation Detected: {'⚠️ YES' if degradation_detected else '✅ NO'}\n"
        f"\nMetric Changes:\n{metric_text}\n"
        f"\nRoot Cause Hypotheses:{hypotheses_text}\n"
        f"\nReport saved to: {ROOT_CAUSE_PATH}\n"
        f"\n💡 Next Steps: Review hypotheses in order of confidence and test recommended fixes."
    )