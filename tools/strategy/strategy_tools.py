import pandas as pd
import numpy as np
import json
import os
from crewai.tools import tool

PREDICTIONS_PATH = "artifacts/data/predictions.csv"
RETENTION_PLAN_PATH = "artifacts/data/retention_plan.csv"

SEGMENT_RULES = {
    "High Risk": (0.70, 1.01),
    "Medium Risk": (0.40, 0.70),
    "Low Risk": (0.00, 0.40),
}

RETENTION_OFFERS = {
    "High Risk": [
        "Upgrade to 2-year contract with 20% discount",
        "Free TechSupport + OnlineSecurity bundle for 6 months",
        "Dedicated account manager outreach call",
    ],
    "Medium Risk": [
        "Offer month-to-month to 1-year contract switch with 10% discount",
        "Free StreamingTV/Movies add-on for 3 months",
        "Personalised usage report with cost-saving tips",
    ],
    "Low Risk": [
        "Loyalty reward points programme invitation",
        "Newsletter with new product features",
    ],
}


@tool("Segment At-Risk Customers")
def segment_at_risk_customers_tool() -> str:
    """Segment customers by churn probability into High/Medium/Low Risk buckets."""
    if not os.path.exists(PREDICTIONS_PATH):
        return f"ERROR: Predictions file not found at {PREDICTIONS_PATH}."
    df = pd.read_csv(PREDICTIONS_PATH)
    df["Risk_Segment"] = pd.cut(
        df["Churn_Probability"],
        bins=[0, 0.40, 0.70, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        right=True,
    )
    counts = df["Risk_Segment"].value_counts().sort_index()
    return f"Customer Segments:\n{counts.to_string()}"


@tool("Generate Retention Offers")
def generate_retention_offers_tool() -> str:
    """Assign retention offers to each customer based on their risk segment."""
    if not os.path.exists(PREDICTIONS_PATH):
        return f"ERROR: Predictions file not found at {PREDICTIONS_PATH}."
    df = pd.read_csv(PREDICTIONS_PATH)
    df["Risk_Segment"] = pd.cut(
        df["Churn_Probability"],
        bins=[0, 0.40, 0.70, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        right=True,
    ).astype(str)

    def assign_offer(segment):
        offers = RETENTION_OFFERS.get(segment, ["Standard loyalty offer"])
        return offers[0]

    df["Retention_Offer"] = df["Risk_Segment"].apply(assign_offer)
    df.to_csv(PREDICTIONS_PATH, index=False)
    offer_counts = df.groupby("Risk_Segment")["Retention_Offer"].first()
    return f"Offers assigned:\n{offer_counts.to_string()}"


@tool("Prioritise Customers")
def prioritize_customers_tool() -> str:
    """Add priority scores (0-100) and sort customers by priority for CRM outreach."""
    if not os.path.exists(PREDICTIONS_PATH):
        return f"ERROR: Predictions file not found at {PREDICTIONS_PATH}."
    df = pd.read_csv(PREDICTIONS_PATH)
    df["Priority_Score"] = (df["Churn_Probability"] * 100).round(1)
    df_sorted = df.sort_values("Priority_Score", ascending=False)
    df_sorted.to_csv(PREDICTIONS_PATH, index=False)
    return (
        f"Priority scores assigned. Top 5 at-risk customers:\n"
        f"{df_sorted[['Churn_Probability', 'Priority_Score', 'Risk_Segment']].head(5).to_string()}"
    )


@tool("Save Retention Plan")
def save_retention_plan_tool() -> str:
    """Save the final retention plan (at-risk customers with offers and priorities) to CSV."""
    if not os.path.exists(PREDICTIONS_PATH):
        return f"ERROR: Predictions file not found at {PREDICTIONS_PATH}."
    df = pd.read_csv(PREDICTIONS_PATH)
    at_risk = df[df["Churn_Predicted"] == 1].copy()
    at_risk = at_risk.sort_values("Priority_Score", ascending=False)
    at_risk.to_csv(RETENTION_PLAN_PATH, index=False)
    return (
        f"Retention plan saved to {RETENTION_PLAN_PATH}.\n"
        f"Total at-risk customers: {len(at_risk)}\n"
        f"High Risk: {(at_risk['Risk_Segment'] == 'High Risk').sum()}\n"
        f"Medium Risk: {(at_risk['Risk_Segment'] == 'Medium Risk').sum()}\n"
        f"Low Risk: {(at_risk['Risk_Segment'] == 'Low Risk').sum()}"
    )