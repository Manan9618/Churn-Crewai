import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from crewai.tools import tool

SUMMARY_REPORT_PATH = "artifacts/model/pipeline_summary_report.json"
SUMMARY_MD_PATH = "artifacts/model/pipeline_summary_report.md"
DASHBOARD_JSON_PATH = "artifacts/model/executive_dashboard.json"
ALERT_MESSAGE_PATH = "artifacts/model/alert_message.txt"
STAKEHOLDER_SUMMARY_PATH = "artifacts/model/stakeholder_summary.md"
ARTIFACT_LINKS_PATH = "artifacts/model/artifact_links.json"

# Thresholds for alerts
ALERT_THRESHOLDS = {
    "min_auc": 0.75,
    "min_f1": 0.55,
    "min_recall": 0.60,
    "max_budget": 10000.00,
    "min_high_risk_coverage": 0.80,
}


def _safe_read_json(path: str) -> dict:
    """Helper function to safely read JSON files."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _safe_read_csv(path: str) -> pd.DataFrame:
    """Helper function to safely read CSV files."""
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
    os.makedirs("artifacts/model", exist_ok=True)
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


# ── NEW SUMMARY TOOLS ADDED BELOW ─────────────────────────────────────────


@tool("Generate Executive Dashboard JSON")
def generate_executive_dashboard_json_tool() -> str:
    """
    Creates a JSON structure specifically for a frontend dashboard (charts/metrics).
    Optimized for visualization libraries like Chart.js, D3.js, or Plotly.
    """
    dashboard = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "dashboard_version": "1.0",
            "refresh_interval_seconds": 300
        },
        "kpi_cards": [],
        "charts": [],
        "alerts": [],
        "status": "UNKNOWN"
    }

    # ── KPI Cards ─────────────────────────────────────────────────────
    kpis = []

    # Model Performance KPI
    model_perf = _safe_read_json(SUMMARY_REPORT_PATH)
    if model_perf.get("sections", {}).get("model_performance"):
        perf = model_perf["sections"]["model_performance"]
        auc = perf.get("auc", 0)
        f1 = perf.get("f1", 0)
        recall = perf.get("recall", 0)

        kpis.append({
            "id": "model_auc",
            "title": "Model AUC",
            "value": auc,
            "suffix": "",
            "target": ALERT_THRESHOLDS["min_auc"],
            "status": "✅ GOOD" if auc >= ALERT_THRESHOLDS["min_auc"] else "⚠️ BELOW TARGET",
            "trend": "stable"
        })

        kpis.append({
            "id": "model_f1",
            "title": "Model F1 Score",
            "value": f1,
            "suffix": "",
            "target": ALERT_THRESHOLDS["min_f1"],
            "status": "✅ GOOD" if f1 >= ALERT_THRESHOLDS["min_f1"] else "⚠️ BELOW TARGET",
            "trend": "stable"
        })

    # Prediction KPI
    pred_path = "artifacts/data/predictions.csv"
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        churn_rate = round(pred_df["Churn_Predicted"].mean() * 100, 2)
        total_customers = len(pred_df)

        kpis.append({
            "id": "predicted_churn_rate",
            "title": "Predicted Churn Rate",
            "value": churn_rate,
            "suffix": "%",
            "target": 26.0,  # Industry average
            "status": "✅ NORMAL" if 10 <= churn_rate <= 35 else "⚠️ UNUSUAL",
            "trend": "stable"
        })

        kpis.append({
            "id": "total_customers",
            "title": "Total Customers Scored",
            "value": total_customers,
            "suffix": "",
            "target": None,
            "status": "✅ COMPLETE",
            "trend": "stable"
        })

    # Retention KPI
    retention_path = "artifacts/data/retention_plan.csv"
    if os.path.exists(retention_path):
        ret_df = pd.read_csv(retention_path)
        at_risk = len(ret_df)
        high_risk = len(ret_df[ret_df["Risk_Segment"] == "High Risk"]) if "Risk_Segment" in ret_df.columns else 0

        kpis.append({
            "id": "at_risk_customers",
            "title": "At-Risk Customers",
            "value": at_risk,
            "suffix": "",
            "target": None,
            "status": "ℹ️ INFO",
            "trend": "stable"
        })

        kpis.append({
            "id": "high_risk_customers",
            "title": "High Risk Customers",
            "value": high_risk,
            "suffix": "",
            "target": None,
            "status": "⚠️ ACTION REQUIRED" if high_risk > 0 else "✅ NONE",
            "trend": "stable"
        })

    dashboard["kpi_cards"] = kpis

    # ── Charts Data ───────────────────────────────────────────────────
    charts = []

    # Churn Distribution Chart
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        churn_dist = pred_df["Churn_Predicted"].value_counts().to_dict()
        charts.append({
            "id": "churn_distribution",
            "title": "Predicted Churn Distribution",
            "type": "pie",
            "data": {
                "labels": ["Not Churning", "Churning"],
                "values": [churn_dist.get(0, 0), churn_dist.get(1, 0)]
            }
        })

        # Probability Histogram Data
        prob_bins = pd.cut(pred_df["Churn_Probability"], bins=10)
        prob_dist = prob_bins.value_counts().sort_index().to_dict()
        charts.append({
            "id": "probability_distribution",
            "title": "Churn Probability Distribution",
            "type": "bar",
            "data": {
                "labels": [str(k) for k in prob_dist.keys()],
                "values": list(prob_dist.values())
            }
        })

    # Risk Segment Distribution
    if os.path.exists(retention_path):
        ret_df = pd.read_csv(retention_path)
        if "Risk_Segment" in ret_df.columns:
            seg_dist = ret_df["Risk_Segment"].value_counts().to_dict()
            charts.append({
                "id": "risk_segments",
                "title": "Customer Risk Segments",
                "type": "bar",
                "data": {
                    "labels": list(seg_dist.keys()),
                    "values": list(seg_dist.values())
                }
            })

    dashboard["charts"] = charts

    # ── Alerts ────────────────────────────────────────────────────────
    alerts = []

    # Check model performance alerts
    if model_perf.get("sections", {}).get("model_performance"):
        perf = model_perf["sections"]["model_performance"]
        if perf.get("auc", 0) < ALERT_THRESHOLDS["min_auc"]:
            alerts.append({
                "severity": "HIGH",
                "message": f"Model AUC ({perf.get('auc', 0):.4f}) below threshold ({ALERT_THRESHOLDS['min_auc']})",
                "action": "Review model training or retrain with more data"
            })
        if perf.get("recall", 0) < ALERT_THRESHOLDS["min_recall"]:
            alerts.append({
                "severity": "MEDIUM",
                "message": f"Model Recall ({perf.get('recall', 0):.4f}) below threshold ({ALERT_THRESHOLDS['min_recall']})",
                "action": "Consider adjusting classification threshold or using SMOTE"
            })

    dashboard["alerts"] = alerts

    # Overall Status
    if len(alerts) == 0:
        dashboard["status"] = "✅ ALL SYSTEMS OPERATIONAL"
    elif any(a["severity"] == "HIGH" for a in alerts):
        dashboard["status"] = "🚨 CRITICAL ISSUES DETECTED"
    else:
        dashboard["status"] = "⚠️ WARNINGS PRESENT"

    # Save dashboard JSON
    os.makedirs("artifacts/model", exist_ok=True)
    with open(DASHBOARD_JSON_PATH, "w") as f:
        json.dump(dashboard, f, indent=2)

    return (
        f"Executive Dashboard JSON Generated:\n"
        f"  KPI Cards: {len(kpis)}\n"
        f"  Charts: {len(charts)}\n"
        f"  Alerts: {len(alerts)}\n"
        f"  Overall Status: {dashboard['status']}\n"
        f"\nDashboard saved to: {DASHBOARD_JSON_PATH}\n"
        f"\n💡 Import this JSON into your frontend dashboard for real-time visualization."
    )


@tool("Generate Alert Message")
def generate_alert_message_tool(
    channel: str = "slack",
    include_metrics: bool = True
) -> str:
    """
    Creates a Slack/Email alert text if critical thresholds are breached.
    Supports multiple channels: slack, email, teams.
    """
    # Load model performance
    model_perf = _safe_read_json(SUMMARY_REPORT_PATH)
    perf = model_perf.get("sections", {}).get("model_performance", {})

    # Load prediction summary
    pred_path = "artifacts/data/predictions.csv"
    pred_df = pd.read_csv(pred_path) if os.path.exists(pred_path) else pd.DataFrame()

    # Load retention summary
    retention_path = "artifacts/data/retention_plan.csv"
    ret_df = pd.read_csv(retention_path) if os.path.exists(retention_path) else pd.DataFrame()

    # Check for alert conditions
    alerts = []
    severity = "INFO"

    # Model Performance Alerts
    auc = perf.get("auc", 0)
    f1 = perf.get("f1", 0)
    recall = perf.get("recall", 0)

    if auc < ALERT_THRESHOLDS["min_auc"]:
        alerts.append(f"❌ Model AUC ({auc:.4f}) below threshold ({ALERT_THRESHOLDS['min_auc']})")
        severity = "CRITICAL"
    elif auc < ALERT_THRESHOLDS["min_auc"] + 0.05:
        alerts.append(f"⚠️ Model AUC ({auc:.4f}) approaching threshold ({ALERT_THRESHOLDS['min_auc']})")
        if severity != "CRITICAL":
            severity = "WARNING"

    if recall < ALERT_THRESHOLDS["min_recall"]:
        alerts.append(f"❌ Model Recall ({recall:.4f}) below threshold ({ALERT_THRESHOLDS['min_recall']})")
        severity = "CRITICAL"

    # Budget Alert
    budget_report = _safe_read_json("artifacts/data/campaign_cost_estimate.json")
    if budget_report:
        if not budget_report.get("within_budget", True):
            alerts.append(f"❌ Campaign budget exceeded by ${abs(budget_report.get('budget_remaining', 0)):.2f}")
            severity = "CRITICAL"

    # Coverage Alert
    coverage_report = _safe_read_json("artifacts/data/coverage_rate_validation.json")
    if coverage_report:
        if not coverage_report.get("meets_minimum", True):
            alerts.append(f"⚠️ High Risk coverage ({coverage_report.get('coverage_rate_percent', 0):.1f}%) below minimum ({ALERT_THRESHOLDS['min_high_risk_coverage']*100:.0f}%)")
            if severity != "CRITICAL":
                severity = "WARNING"

    # Generate message based on channel
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if channel == "slack":
        if alerts:
            message = (
                f":{('red_circle' if severity == 'CRITICAL' else 'warning' if severity == 'WARNING' else 'white_check_mark')}: "
                f"*Churn Prediction Pipeline Alert*\n"
                f"*Severity:* {severity}\n"
                f"*Timestamp:* {timestamp}\n\n"
                f"*Issues Detected:*\n"
                + "\n".join([f"• {alert}" for alert in alerts]) +
                f"\n\n*Quick Metrics:*\n"
                f"• Model AUC: `{auc:.4f}`\n"
                f"• Model F1: `{f1:.4f}`\n"
                f"• Model Recall: `{recall:.4f}`\n"
                f"• Predicted Churners: `{len(pred_df[pred_df['Churn_Predicted']==1]) if not pred_df.empty else 'N/A'}`\n"
                f"• At-Risk Customers: `{len(ret_df) if not ret_df.empty else 'N/A'}`\n\n"
                f":mag: *Action Required:* Review pipeline outputs and address issues above."
            )
        else:
            message = (
                f":white_check_mark: *Churn Prediction Pipeline — All Clear*\n"
                f"*Timestamp:* {timestamp}\n\n"
                f"*All metrics within acceptable thresholds:*\n"
                f"• Model AUC: `{auc:.4f}` (≥ {ALERT_THRESHOLDS['min_auc']})\n"
                f"• Model F1: `{f1:.4f}` (≥ {ALERT_THRESHOLDS['min_f1']})\n"
                f"• Model Recall: `{recall:.4f}` (≥ {ALERT_THRESHOLDS['min_recall']})\n\n"
                f":tada: Pipeline completed successfully!"
            )

    elif channel == "email":
        subject = f"[{severity}] Churn Prediction Pipeline Alert — {timestamp}"
        body = (
            f"<html><body>"
            f"<h2>Churn Prediction Pipeline Alert</h2>"
            f"<p><strong>Severity:</strong> {severity}</p>"
            f"<p><strong>Timestamp:</strong> {timestamp}</p>"
            f"<hr/>"
            f"<h3>Issues Detected:</h3>"
            f"<ul>" + ("".join([f"<li>{alert}</li>" for alert in alerts]) if alerts else "<li>None - All systems operational</li>") + "</ul>"
            f"<h3>Key Metrics:</h3>"
            f"<ul>"
            f"<li>Model AUC: {auc:.4f}</li>"
            f"<li>Model F1: {f1:.4f}</li>"
            f"<li>Model Recall: {recall:.4f}</li>"
            f"<li>Predicted Churners: {len(pred_df[pred_df['Churn_Predicted']==1]) if not pred_df.empty else 'N/A'}</li>"
            f"<li>At-Risk Customers: {len(ret_df) if not ret_df.empty else 'N/A'}</li>"
            f"</ul>"
            f"<hr/>"
            f"<p><em>This is an automated alert from the Churn Prediction Pipeline.</em></p>"
            f"</body></html>"
        )
        message = f"Subject: {subject}\n\n{body}"

    elif channel == "teams":
        message = (
            f"{'🚨' if severity == 'CRITICAL' else '⚠️' if severity == 'WARNING' else '✅'} "
            f"**Churn Prediction Pipeline Alert**\n\n"
            f"**Severity:** {severity}\n"
            f"**Timestamp:** {timestamp}\n\n"
            f"**Issues:**\n" + ("\n".join([f"- {alert}" for alert in alerts]) if alerts else "- None") +
            f"\n\n**Metrics:**\n"
            f"- AUC: {auc:.4f}\n"
            f"- F1: {f1:.4f}\n"
            f"- Recall: {recall:.4f}"
        )
    else:
        message = f"Unsupported channel: {channel}. Use 'slack', 'email', or 'teams'."

    # Save alert message
    os.makedirs("artifacts/model", exist_ok=True)
    with open(ALERT_MESSAGE_PATH, "w") as f:
        f.write(message)

    alert_status = "ALERTS TRIGGERED" if alerts else "ALL CLEAR"

    return (
        f"Alert Message Generated ({channel.upper()}):\n"
        f"Status: {alert_status}\n"
        f"Severity: {severity}\n"
        f"Alerts: {len(alerts)}\n"
        f"\nMessage Preview:\n{message[:500]}{'...' if len(message) > 500 else ''}\n"
        f"\nFull message saved to: {ALERT_MESSAGE_PATH}\n"
        f"\n💡 Copy this message to your {channel} channel or email recipients."
    )


@tool("Create Stakeholder Summary")
def create_stakeholder_summary_tool() -> str:
    """
    Generates a non-technical summary for business managers (hides technical metrics).
    Focuses on business impact, ROI, and actionable recommendations.
    """
    # Load all relevant data
    model_perf = _safe_read_json(SUMMARY_REPORT_PATH)
    pred_df = _safe_read_csv("artifacts/data/predictions.csv")
    ret_df = _safe_read_csv("artifacts/data/retention_plan.csv")
    cost_report = _safe_read_json("artifacts/data/campaign_cost_estimate.json")
    budget_opt = _safe_read_json("artifacts/data/budget_optimization_report.json")

    # Generate business-friendly summary
    summary = {
        "title": "Customer Churn Prediction — Business Summary",
        "generated_at": datetime.now().strftime("%B %d, %Y"),
        "executive_overview": "",
        "key_findings": [],
        "financial_impact": {},
        "recommendations": [],
        "next_steps": []
    }

    # Executive Overview
    perf = model_perf.get("sections", {}).get("model_performance", {})
    auc = perf.get("auc", 0)
    model_accuracy = "high" if auc >= 0.80 else "good" if auc >= 0.75 else "moderate"

    total_customers = len(pred_df) if not pred_df.empty else 0
    predicted_churners = int(pred_df["Churn_Predicted"].sum()) if not pred_df.empty else 0
    churn_rate = (predicted_churners / total_customers * 100) if total_customers > 0 else 0

    at_risk = len(ret_df) if not ret_df.empty else 0
    high_risk = len(ret_df[ret_df["Risk_Segment"] == "High Risk"]) if "Risk_Segment" in ret_df.columns and not ret_df.empty else 0

    summary["executive_overview"] = (
        f"Our AI-powered churn prediction model has analyzed {total_customers:,} customers "
        f"and identified {predicted_churners:,} ({churn_rate:.1f}%) at risk of churning in the next period. "
        f"The model demonstrates {model_accuracy} predictive accuracy (AUC: {auc:.2f}), enabling us to "
        f"proactively retain valuable customers through targeted interventions."
    )

    # Key Findings
    summary["key_findings"] = [
        f"{high_risk:,} customers are classified as High Risk (immediate action required)",
        f"{at_risk - high_risk:,} customers are Medium/Low Risk (monitoring recommended)",
        f"Estimated campaign cost: ${cost_report.get('total_cost', 0):,.2f}" if cost_report else "Campaign cost pending calculation",
        f"Expected ROI: {budget_opt.get('roi_percent', 0):.1f}%" if budget_opt else "ROI analysis pending"
    ]

    # Financial Impact
    if cost_report and budget_opt:
        summary["financial_impact"] = {
            "campaign_investment": f"${cost_report.get('total_cost', 0):,.2f}",
            "expected_revenue_saved": f"${budget_opt.get('expected_revenue', 0):,.2f}",
            "projected_roi": f"{budget_opt.get('roi_percent', 0):.1f}%",
            "customers_expected_retained": f"{budget_opt.get('expected_retained_customers', 0):.0f}"
        }
    else:
        summary["financial_impact"] = {
            "campaign_investment": "Pending",
            "expected_revenue_saved": "Pending",
            "projected_roi": "Pending",
            "customers_expected_retained": "Pending"
        }

    # Recommendations
    summary["recommendations"] = [
        "Prioritize outreach to High Risk customers within the next 7 days",
        "Allocate budget to personalized retention offers based on customer segment",
        "Monitor campaign effectiveness through A/B testing (Control vs Treatment groups)",
        "Review model performance monthly and retrain if accuracy drops below 75%"
    ]

    # Next Steps
    summary["next_steps"] = [
        "CRM team: Begin outreach to High Risk segment this week",
        "Marketing: Prepare personalized offer communications",
        "Analytics: Track retention rates over next 30 days",
        "Leadership: Review campaign ROI at month-end"
    ]

    # Save stakeholder summary
    os.makedirs("artifacts/model", exist_ok=True)

    # Save as JSON
    with open(STAKEHOLDER_SUMMARY_PATH.replace(".md", ".json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save as Markdown (business-friendly format)
    md_content = f"""# {summary['title']}

**Generated:** {summary['generated_at']}

---

## Executive Overview

{summary['executive_overview']}

---

## Key Findings

""" + "\n".join([f"- {finding}" for finding in summary['key_findings']]) + f"""

---

## Financial Impact

| Metric | Value |
|--------|-------|
| Campaign Investment | {summary['financial_impact']['campaign_investment']} |
| Expected Revenue Saved | {summary['financial_impact']['expected_revenue_saved']} |
| Projected ROI | {summary['financial_impact']['projected_roi']} |
| Customers Expected Retained | {summary['financial_impact']['customers_expected_retained']} |

---

## Recommendations

""" + "\n".join([f"✅ {rec}" for rec in summary['recommendations']]) + f"""

---

## Next Steps

""" + "\n".join([f"📋 {step}" for step in summary['next_steps']]) + f"""

---

*This summary is intended for business stakeholders. Technical details available in full pipeline report.*
"""

    with open(STAKEHOLDER_SUMMARY_PATH, "w") as f:
        f.write(md_content)

    return (
        f"Stakeholder Summary Created:\n"
        f"  Title: {summary['title']}\n"
        f"  Generated: {summary['generated_at']}\n"
        f"  Key Findings: {len(summary['key_findings'])}\n"
        f"  Recommendations: {len(summary['recommendations'])}\n"
        f"\nFiles Saved:\n"
        f"  Markdown: {STAKEHOLDER_SUMMARY_PATH}\n"
        f"  JSON: {STAKEHOLDER_SUMMARY_PATH.replace('.md', '.json')}\n"
        f"\n💡 Share this summary with business managers and leadership teams."
    )


@tool("Link Artifacts")
def link_artifacts_tool(
    base_url: str = "file://",
    include_hashes: bool = True
) -> str:
    """
    Generates clickable links to the specific model files, logs, and data versions in the report.
    Enables quick navigation to pipeline artifacts for auditing and debugging.
    """
    artifacts = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "base_url": base_url,
            "include_hashes": include_hashes
        },
        "categories": {
            "data": [],
            "models": [],
            "reports": [],
            "logs": [],
            "configs": []
        }
    }

    # Define artifact paths
    artifact_paths = {
        "data": [
            ("Raw Data", "data/Customer_Churn.csv"),
            ("Processed Data", "artifacts/data/processed_churn.csv"),
            ("Training Features", "artifacts/data/X_train.csv"),
            ("Test Features", "artifacts/data/X_test.csv"),
            ("Training Labels", "artifacts/data/y_train.csv"),
            ("Test Labels", "artifacts/data/y_test.csv"),
            ("Selected Features", "artifacts/data/selected_features.json"),
            ("Predictions", "artifacts/data/predictions.csv"),
            ("Retention Plan", "artifacts/data/retention_plan.csv"),
        ],
        "models": [
            ("Main Model", "artifacts/model/churn_model.pkl"),
            ("Scaler", "artifacts/model/scaler.pkl"),
            ("PyCaret Best Model", "artifacts/model/pycaret_best_model.pkl"),
            ("PyCaret Blend Model", "artifacts/model/pycaret_blend_model.pkl"),
            ("PyCaret Stack Model", "artifacts/model/pycaret_stack_model.pkl"),
            ("ONNX Model", "artifacts/model/churn_model.onnx"),
        ],
        "reports": [
            ("Pipeline Summary (JSON)", "artifacts/model/pipeline_summary_report.json"),
            ("Pipeline Summary (Markdown)", "artifacts/model/pipeline_summary_report.md"),
            ("Executive Dashboard", "artifacts/model/executive_dashboard.json"),
            ("Stakeholder Summary", "artifacts/model/stakeholder_summary.md"),
            ("Explanation Report", "artifacts/model/explanation_report.json"),
            ("Fairness Report", "artifacts/model/fairness_report.json"),
        ],
        "logs": [
            ("Experiment Log", "artifacts/model/experiment_log.json"),
            ("Pipeline State", "artifacts/data/pipeline_state.json"),
            ("Alert Message", "artifacts/model/alert_message.txt"),
        ],
        "configs": [
            ("Feature Config", "artifacts/data/selected_features.json"),
            ("Campaign Cost", "artifacts/data/campaign_cost_estimate.json"),
            ("Budget Optimization", "artifacts/data/budget_optimization_report.json"),
            ("A/B Test Groups", "artifacts/data/ab_test_groups.json"),
        ]
    }

    # Build artifact links
    for category, items in artifact_paths.items():
        for name, path in items:
            artifact_info = {
                "name": name,
                "path": path,
                "url": f"{base_url}{os.path.abspath(path)}",
                "exists": os.path.exists(path),
                "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
                "last_modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if os.path.exists(path) else None
            }

            # Add file hash if requested
            if include_hashes and os.path.exists(path):
                import hashlib
                with open(path, "rb") as f:
                    artifact_info["md5_hash"] = hashlib.md5(f.read()).hexdigest()

            artifacts["categories"][category].append(artifact_info)

    # Calculate summary statistics
    total_artifacts = sum(len(items) for items in artifacts["categories"].values())
    existing_artifacts = sum(1 for cat in artifacts["categories"].values() for item in cat if item["exists"])
    total_size = sum(item["size_bytes"] for cat in artifacts["categories"].values() for item in cat)

    artifacts["summary"] = {
        "total_artifacts": total_artifacts,
        "existing_artifacts": existing_artifacts,
        "missing_artifacts": total_artifacts - existing_artifacts,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 ** 2), 2)
    }

    # Save artifact links
    os.makedirs("artifacts/model", exist_ok=True)
    with open(ARTIFACT_LINKS_PATH, "w") as f:
        json.dump(artifacts, f, indent=2)

    # Generate markdown version for easy viewing
    md_links = [
        "# Pipeline Artifact Links",
        f"\n**Generated:** {artifacts['metadata']['generated_at']}\n",
        f"**Summary:** {existing_artifacts}/{total_artifacts} artifacts found ({artifacts['summary']['total_size_mb']:.2f} MB)\n",
    ]

    for category, items in artifacts["categories"].items():
        md_links.append(f"\n## {category.title()}\n")
        for item in items:
            status = "✅" if item["exists"] else "❌"
            size = f"{item['size_bytes'] / 1024:.1f} KB" if item["size_bytes"] > 0 else "N/A"
            md_links.append(
                f"- {status} **{item['name']}**\n"
                f"  - Path: `{item['path']}`\n"
                f"  - URL: [{item['url']}]({item['url']})\n"
                f"  - Size: {size}\n"
                + (f"  - Hash: `{item['md5_hash'][:16]}...`\n" if include_hashes and item.get('md5_hash') else "")
            )

    # Save markdown version
    md_path = ARTIFACT_LINKS_PATH.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_links))

    return (
        f"Artifact Links Generated:\n"
        f"  Total Artifacts: {total_artifacts}\n"
        f"  Existing: {existing_artifacts}\n"
        f"  Missing: {total_artifacts - existing_artifacts}\n"
        f"  Total Size: {artifacts['summary']['total_size_mb']:.2f} MB\n"
        f"\nFiles Saved:\n"
        f"  JSON: {ARTIFACT_LINKS_PATH}\n"
        f"  Markdown: {md_path}\n"
        f"\n💡 Use these links to quickly navigate to pipeline artifacts for auditing."
    )