import pandas as pd
import json
import os
from crewai.tools import tool

SUMMARY_REPORT_PATH = "artifact/model/pipeline_summary_report.json"
SUMMARY_MD_PATH = "artifact/model/pipeline_summary_report.md"

REQUIRED_SECTIONS = [
    "data",
    "feature_selection",
    "model_performance",
    "predictions",
    "explanability",
    "retention_strategy",
]

MINIMUM_MODEL_AUC = 0.75
MINIMUM_CHURN_RATE = 10.0
MAXIMUM_CHURN_RATE = 35.0


@tool("Validate Summary Report Exists")
def validate_summary_report_exists_tool() -> str:
    """Validate that both the JSON and Markdown summary report files exist."""
    errors = []
    for path in [SUMMARY_REPORT_PATH, SUMMARY_MD_PATH]:
        if not os.path.exists(path):
            errors.append(f"Missing: {path}")
        elif os.path.getsize(path) == 0:
            errors.append(f"Empty file: {path}")
    if errors:
        return "VALIDATION FAILED:\n" + "\n".join(errors)
    return f"Summary report files PASSED. Both JSON and Markdown reports exist."


@tool("Validate Summary Sections Complete")
def validate_summary_sections_tool() -> str:
    """
    Validate that the pipeline summary JSON report contains all required sections:
    data, feature_selection, model_performance, predictions, explanability,
    retention_strategy.
    """
    if not os.path.exists(SUMMARY_REPORT_PATH):
        return f"VALIDATION FAILED: Summary report not found at {SUMMARY_REPORT_PATH}."

    with open(SUMMARY_REPORT_PATH) as f:
        report = json.load(f)

    sections = report.get("sections", {})
    missing = [s for s in REQUIRED_SECTIONS if s not in sections]

    if missing:
        return (
            f"VALIDATION FAILED: Missing sections in summary report: {missing}\n"
            f"Present sections: {list(sections.keys())}"
        )
    return (
        f"Summary sections PASSED. All {len(REQUIRED_SECTIONS)} required sections present:\n"
        f"  {REQUIRED_SECTIONS}"
    )


@tool("Validate Summary Model Metrics")
def validate_summary_model_metrics_tool() -> str:
    """
    Validate that the model performance section in the summary report
    meets minimum AUC threshold.
    """
    if not os.path.exists(SUMMARY_REPORT_PATH):
        return f"VALIDATION FAILED: Summary report not found at {SUMMARY_REPORT_PATH}."

    with open(SUMMARY_REPORT_PATH) as f:
        report = json.load(f)

    model_section = report.get("sections", {}).get("model_performance", {})
    if not model_section:
        return "VALIDATION FAILED: model_performance section missing from summary."

    auc = model_section.get("auc", 0)
    if auc < MINIMUM_MODEL_AUC:
        return f"VALIDATION FAILED: Summary AUC {auc} < minimum {MINIMUM_MODEL_AUC}."

    return (
        f"Summary model metrics PASSED.\n"
        f"  Model : {model_section.get('model_type', 'N/A')}\n"
        f"  AUC   : {auc}\n"
        f"  F1    : {model_section.get('f1', 'N/A')}\n"
        f"  Recall: {model_section.get('recall', 'N/A')}"
    )


@tool("Validate Summary Prediction Stats")
def validate_summary_prediction_stats_tool() -> str:
    """
    Validate that the predictions section in the summary has a realistic
    churn rate (10%–35%) and non-zero total customers.
    """
    if not os.path.exists(SUMMARY_REPORT_PATH):
        return f"VALIDATION FAILED: Summary report not found at {SUMMARY_REPORT_PATH}."

    with open(SUMMARY_REPORT_PATH) as f:
        report = json.load(f)

    pred_section = report.get("sections", {}).get("predictions", {})
    if not pred_section:
        return "VALIDATION FAILED: predictions section missing from summary."

    total = pred_section.get("total", 0)
    churn_rate = pred_section.get("churn_rate_pct", 0)

    errors = []
    if total == 0:
        errors.append("Total customers is 0.")
    if not (MINIMUM_CHURN_RATE <= churn_rate <= MAXIMUM_CHURN_RATE):
        errors.append(
            f"Churn rate {churn_rate}% outside expected range "
            f"[{MINIMUM_CHURN_RATE}%, {MAXIMUM_CHURN_RATE}%]."
        )
    if errors:
        return "VALIDATION FAILED:\n" + "\n".join(errors)

    return (
        f"Prediction stats PASSED.\n"
        f"  Total customers   : {total}\n"
        f"  Predicted churners: {pred_section.get('churners', 'N/A')}\n"
        f"  Churn rate        : {churn_rate}%"
    )


@tool("Validate Summary Retention Section")
def validate_summary_retention_tool() -> str:
    """
    Validate that the retention strategy section exists and reports
    at least one at-risk customer with valid segments.
    """
    if not os.path.exists(SUMMARY_REPORT_PATH):
        return f"VALIDATION FAILED: Summary report not found at {SUMMARY_REPORT_PATH}."

    with open(SUMMARY_REPORT_PATH) as f:
        report = json.load(f)

    ret_section = report.get("sections", {}).get("retention_strategy", {})
    if not ret_section:
        return "VALIDATION FAILED: retention_strategy section missing from summary."

    total_at_risk = ret_section.get("total_at_risk", 0)
    if total_at_risk == 0:
        return "VALIDATION FAILED: retention_strategy reports 0 at-risk customers."

    valid_segments = {"High Risk", "Medium Risk", "Low Risk"}
    segments = set(ret_section.get("segments", {}).keys())
    invalid = segments - valid_segments
    if invalid:
        return f"VALIDATION FAILED: Invalid segment labels in summary: {invalid}"

    return (
        f"Retention section PASSED.\n"
        f"  At-risk customers: {total_at_risk}\n"
        f"  Segments: {ret_section.get('segments', {})}"
    )


@tool("Validate Markdown Report Readable")
def validate_markdown_report_tool() -> str:
    """Validate that the Markdown report is well-formed and contains all major headings."""
    if not os.path.exists(SUMMARY_MD_PATH):
        return f"VALIDATION FAILED: Markdown report not found at {SUMMARY_MD_PATH}."

    with open(SUMMARY_MD_PATH) as f:
        content = f.read()

    required_headings = [
        "# Customer Churn",
        "## Data",
        "## Model Performance",
        "## Predictions",
        "## Retention",
    ]
    missing = [h for h in required_headings if h not in content]
    if missing:
        return f"VALIDATION FAILED: Missing headings in Markdown report: {missing}"

    word_count = len(content.split())
    return f"Markdown report PASSED. {word_count} words, all required headings present."