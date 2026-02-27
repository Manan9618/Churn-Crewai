import pandas as pd
import os
from crewai.tools import tool

RETENTION_PLAN_PATH = "artifacts/data/retention_plan.csv"
PREDICTIONS_PATH = "artifacts/data/predictions.csv"
VALID_SEGMENTS = {"High Risk", "Medium Risk", "Low Risk"}


@tool("Validate Retention Plan File")
def validate_retention_plan_file_tool() -> str:
    """Validate that retention plan file exists and is non-empty."""
    if not os.path.exists(RETENTION_PLAN_PATH):
        return f"VALIDATION FAILED: Retention plan not found at {RETENTION_PLAN_PATH}."
    df = pd.read_csv(RETENTION_PLAN_PATH)
    if df.empty:
        return "VALIDATION FAILED: Retention plan is empty."
    return f"Retention plan file PASSED. {len(df)} at-risk customers in plan."


@tool("Validate Customer Segments")
def validate_customer_segments_tool() -> str:
    """Validate that all Risk_Segment values are from the expected set."""
    if not os.path.exists(PREDICTIONS_PATH):
        return f"VALIDATION FAILED: {PREDICTIONS_PATH} not found."
    df = pd.read_csv(PREDICTIONS_PATH)
    if "Risk_Segment" not in df.columns:
        return "VALIDATION FAILED: Risk_Segment column missing from predictions."
    invalid = set(df["Risk_Segment"].dropna().unique()) - VALID_SEGMENTS
    if invalid:
        return f"VALIDATION FAILED: Invalid segment labels found: {invalid}"
    return f"Customer segments PASSED. Segments: {df['Risk_Segment'].value_counts().to_dict()}"


@tool("Validate Offers Assigned")
def validate_offers_assigned_tool() -> str:
    """Validate that all high-risk customers (probability > 0.7) have a retention offer."""
    if not os.path.exists(PREDICTIONS_PATH):
        return f"VALIDATION FAILED: {PREDICTIONS_PATH} not found."
    df = pd.read_csv(PREDICTIONS_PATH)
    high_risk = df[df["Churn_Probability"] > 0.7]
    if "Retention_Offer" not in df.columns:
        return "VALIDATION FAILED: Retention_Offer column missing."
    missing_offers = high_risk["Retention_Offer"].isna().sum()
    if missing_offers > 0:
        return f"VALIDATION FAILED: {missing_offers} high-risk customers have no retention offer."
    return f"Retention offers PASSED. All {len(high_risk)} high-risk customers have offers."


@tool("Validate Priority Scores")
def validate_priority_scores_tool() -> str:
    """Validate that Priority_Score values are in [0, 100] range."""
    if not os.path.exists(PREDICTIONS_PATH):
        return f"VALIDATION FAILED: {PREDICTIONS_PATH} not found."
    df = pd.read_csv(PREDICTIONS_PATH)
    if "Priority_Score" not in df.columns:
        return "VALIDATION FAILED: Priority_Score column missing."
    invalid = df[(df["Priority_Score"] < 0) | (df["Priority_Score"] > 100)]
    if not invalid.empty:
        return f"VALIDATION FAILED: {len(invalid)} scores outside [0, 100]."
    return f"Priority scores PASSED. Range: [{df['Priority_Score'].min():.1f}, {df['Priority_Score'].max():.1f}]."