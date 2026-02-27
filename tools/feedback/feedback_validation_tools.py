import pandas as pd
import json
import os
from crewai.tools import tool

EXPERIMENT_LOG_PATH = "artifact/model/experiment_log.json"


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