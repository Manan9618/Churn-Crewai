import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from crewai.tools import tool

SUMMARY_REPORT_PATH = "artifacts/model/pipeline_summary_report.json"
SUMMARY_MD_PATH = "artifacts/model/pipeline_summary_report.md"


def _safe_read_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _safe_read_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@tool("Generate Data Summary")
def generate_data_summary_tool() -> str:
    """
    Summarise data pipeline outputs: raw dataset info, processed dataset info,
    class distribution, train/test split sizes, and selected features.
    """
    summary = {}

    # Raw data
    raw_path = "data/Customer_Churn.csv"
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        summary["raw_data"] = {
            "rows": len(raw_df),
            "columns": len(raw_df.columns),
            "churn_rate_pct": round((raw_df["Churn"] == "Yes").mean() * 100, 2),
        }

    # Processed data
    proc_path = "artifacts/data/processed_churn.csv"
    if os.path.exists(proc_path):
        proc_df = pd.read_csv(proc_path)
        summary["processed_data"] = {
            "rows": len(proc_df),
            "columns": len(proc_df.columns),
            "missing_values": int(proc_df.isnull().sum().sum()),
        }

    # Train/test splits
    for split in ["X_train", "X_test", "y_train", "y_test"]:
        path = f"artifacts/data/{split}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            summary[split] = {"rows": len(df), "cols": len(df.columns)}

    # Selected features
    feat_path = "artifacts/data/selected_features.json"
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            features = json.load(f)
        summary["selected_features"] = {"count": len(features), "features": features}

    result = json.dumps(summary, indent=2)
    return f"Data Summary:\n{result}"


@tool("Generate Model Performance Summary")
def generate_model_performance_summary_tool() -> str:
    """
    Summarise model training results: cross-validation metrics, test metrics,
    and PyCaret comparison results if available.
    """
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
    )

    summary = {}

    model_path = "artifacts/model/churn_model.pkl"
    feat_path = "artifacts/data/selected_features.json"
    if (os.path.exists(model_path)
            and os.path.exists("artifacts/data/X_test.csv")
            and os.path.exists(feat_path)):

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(feat_path) as f:
            features = json.load(f)

        X_te = pd.read_csv("artifacts/data/X_test.csv")[features]
        y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        summary["model_type"] = type(model).__name__
        summary["test_metrics"] = {
            "auc":       round(roc_auc_score(y_te, y_prob), 4),
            "f1":        round(f1_score(y_te, y_pred), 4),
            "precision": round(precision_score(y_te, y_pred), 4),
            "recall":    round(recall_score(y_te, y_pred), 4),
            "accuracy":  round(accuracy_score(y_te, y_pred), 4),
        }

    # PyCaret results
    pc_results_path = "artifacts/model/pycaret_compare_results.csv"
    if os.path.exists(pc_results_path):
        pc_df = pd.read_csv(pc_results_path)
        if not pc_df.empty:
            top = pc_df.iloc[0]
            summary["pycaret_best"] = {
                "model": str(top.get("Model", "N/A")),
                "auc":   round(float(top.get("AUC", 0)), 4),
                "f1":    round(float(top.get("F1", 0)), 4),
            }

    result = json.dumps(summary, indent=2)
    return f"Model Performance Summary:\n{result}"


@tool("Generate Prediction Summary")
def generate_prediction_summary_tool() -> str:
    """
    Summarise prediction outputs: total predictions, churn rate,
    probability distribution, and segment breakdown.
    """
    pred_path = "artifacts/data/predictions.csv"
    if not os.path.exists(pred_path):
        return "ERROR: Predictions file not found. Run prediction agent first."

    df = pd.read_csv(pred_path)
    summary = {
        "total_customers": len(df),
        "predicted_churners": int(df["Churn_Predicted"].sum()),
        "predicted_non_churners": int((df["Churn_Predicted"] == 0).sum()),
        "churn_rate_pct": round(df["Churn_Predicted"].mean() * 100, 2),
        "avg_churn_probability": round(df["Churn_Probability"].mean(), 4),
        "probability_distribution": {
            "min":    round(df["Churn_Probability"].min(), 4),
            "p25":    round(df["Churn_Probability"].quantile(0.25), 4),
            "median": round(df["Churn_Probability"].median(), 4),
            "p75":    round(df["Churn_Probability"].quantile(0.75), 4),
            "max":    round(df["Churn_Probability"].max(), 4),
        },
    }

    if "Risk_Segment" in df.columns:
        summary["risk_segments"] = df["Risk_Segment"].value_counts().to_dict()

    result = json.dumps(summary, indent=2)
    return f"Prediction Summary:\n{result}"


@tool("Generate Retention Summary")
def generate_retention_summary_tool() -> str:
    """
    Summarise retention strategy outputs: customer counts per segment,
    offer types assigned, and estimated business impact.
    """
    retention_path = "artifacts/data/retention_plan.csv"
    if not os.path.exists(retention_path):
        return "ERROR: Retention plan not found. Run retention strategy agent first."

    df = pd.read_csv(retention_path)
    summary = {
        "total_at_risk_customers": len(df),
    }

    if "Risk_Segment" in df.columns:
        summary["by_segment"] = df["Risk_Segment"].value_counts().to_dict()

    if "Retention_Offer" in df.columns:
        summary["offers_assigned"] = int(df["Retention_Offer"].notna().sum())
        summary["unique_offers"] = df["Retention_Offer"].nunique()

    if "Priority_Score" in df.columns:
        summary["priority_stats"] = {
            "avg": round(df["Priority_Score"].mean(), 1),
            "max": round(df["Priority_Score"].max(), 1),
            "min": round(df["Priority_Score"].min(), 1),
        }

    result = json.dumps(summary, indent=2)
    return f"Retention Summary:\n{result}"


@tool("Generate Full Pipeline Summary Report")
def generate_full_pipeline_summary_tool() -> str:
    """
    Compile a comprehensive summary of the entire pipeline run — covering
    data, model metrics, predictions, explanations, and retention strategy.
    Saves a JSON report and a human-readable Markdown report.
    """
    from sklearn.metrics import roc_auc_score, f1_score, recall_score

    report = {
        "pipeline": "Customer Churn Prediction — CrewAI",
        "generated_at": datetime.now().isoformat(),
        "sections": {},
    }

    # ── Data section ──────────────────────────────────────────────────────
    raw_path = "data/Customer_Churn.csv"
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        report["sections"]["data"] = {
            "total_customers": len(raw_df),
            "features": len(raw_df.columns) - 1,
            "raw_churn_rate_pct": round((raw_df["Churn"] == "Yes").mean() * 100, 2),
        }

    feat_path = "artifacts/data/selected_features.json"
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            features = json.load(f)
        report["sections"]["feature_selection"] = {
            "selected_features": features,
            "count": len(features),
        }

    # ── Model section ─────────────────────────────────────────────────────
    model_path = "artifacts/model/churn_model.pkl"
    if os.path.exists(model_path) and os.path.exists("artifacts/data/X_test.csv") and os.path.exists(feat_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(feat_path) as f:
            features = json.load(f)
        X_te = pd.read_csv("artifacts/data/X_test.csv")[features]
        y_te = pd.read_csv("artifacts/data/y_test.csv").squeeze()
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        report["sections"]["model_performance"] = {
            "model_type": type(model).__name__,
            "auc":       round(roc_auc_score(y_te, y_prob), 4),
            "f1":        round(f1_score(y_te, y_pred), 4),
            "recall":    round(recall_score(y_te, y_pred), 4),
        }

    # ── Predictions section ───────────────────────────────────────────────
    pred_path = "artifacts/data/predictions.csv"
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        report["sections"]["predictions"] = {
            "total": len(pred_df),
            "churners": int(pred_df["Churn_Predicted"].sum()),
            "churn_rate_pct": round(pred_df["Churn_Predicted"].mean() * 100, 2),
        }

    # ── Explanation section ───────────────────────────────────────────────
    explanation_report = _safe_read_json("artifacts/model/explanation_report.json")
    if explanation_report:
        report["sections"]["explanability"] = {
            "top_churn_drivers": explanation_report.get("top_churn_drivers", []),
        }

    # ── Retention section ─────────────────────────────────────────────────
    retention_path = "artifacts/data/retention_plan.csv"
    if os.path.exists(retention_path):
        ret_df = pd.read_csv(retention_path)
        seg_counts = ret_df["Risk_Segment"].value_counts().to_dict() if "Risk_Segment" in ret_df.columns else {}
        report["sections"]["retention_strategy"] = {
            "total_at_risk": len(ret_df),
            "segments": seg_counts,
        }

    # ── Experiment log ────────────────────────────────────────────────────
    exp_log = _safe_read_json("artifacts/model/experiment_log.json")
    if exp_log.get("experiments"):
        report["sections"]["experiment_log"] = {
            "total_runs": len(exp_log["experiments"]),
            "latest_run": exp_log["experiments"][-1],
        }

    # ── Save JSON report ──────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)
    with open(SUMMARY_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # ── Save Markdown report ──────────────────────────────────────────────
    md_lines = [
        "# Customer Churn Prediction — Pipeline Summary Report",
        f"\n**Generated:** {report['generated_at']}\n",
    ]

    for section, content in report["sections"].items():
        md_lines.append(f"\n## {section.replace('_', ' ').title()}\n")
        for k, v in content.items():
            md_lines.append(f"- **{k}**: {v}")

    with open(SUMMARY_MD_PATH, "w") as f:
        f.write("\n".join(md_lines))

    return (
        f"Full pipeline summary saved.\n"
        f"  JSON : {SUMMARY_REPORT_PATH}\n"
        f"  MD   : {SUMMARY_MD_PATH}\n"
        f"  Sections: {list(report['sections'].keys())}"
    )