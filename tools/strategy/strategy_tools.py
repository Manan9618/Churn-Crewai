import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from crewai.tools import tool

PREDICTIONS_PATH = "artifacts/data/predictions.csv"
RETENTION_PLAN_PATH = "artifacts/data/retention_plan.csv"
CAMPAIGN_COST_PATH = "artifacts/data/campaign_cost_estimate.json"
COMMUNICATION_CHANNEL_PATH = "artifacts/data/communication_channel_assignment.json"
AB_TEST_PATH = "artifacts/data/ab_test_groups.json"
BUDGET_OPTIMIZATION_PATH = "artifacts/data/budget_optimization_report.json"

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

# Cost assumptions for retention offers (adjust based on your business)
OFFER_COSTS = {
    "Upgrade to 2-year contract with 20% discount": 50.00,
    "Free TechSupport + OnlineSecurity bundle for 6 months": 30.00,
    "Dedicated account manager outreach call": 25.00,
    "Offer month-to-month to 1-year contract switch with 10% discount": 20.00,
    "Free StreamingTV/Movies add-on for 3 months": 15.00,
    "Personalised usage report with cost-saving tips": 5.00,
    "Loyalty reward points programme invitation": 10.00,
    "Newsletter with new product features": 2.00,
    "Standard loyalty offer": 5.00,
}

# Communication channel costs and effectiveness
CHANNEL_INFO = {
    "Email": {"cost": 0.50, "effectiveness": 0.15, "preferred_by": ["PaperlessBilling"]},
    "SMS": {"cost": 1.00, "effectiveness": 0.25, "preferred_by": ["PhoneService"]},
    "Call": {"cost": 5.00, "effectiveness": 0.45, "preferred_by": ["SeniorCitizen"]},
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


# ── NEW RETENTION STRATEGY TOOLS ADDED BELOW ──────────────────────────────────


@tool("Estimate Campaign Cost")
def estimate_campaign_cost_tool(
    budget_limit: float = 10000.00,
    include_all_segments: bool = True
) -> str:
    """
    Calculates total cost of proposed retention offers against a budget.
    Helps determine if the retention campaign is financially feasible.
    """
    if not os.path.exists(RETENTION_PLAN_PATH):
        return (
            f"ERROR: Retention plan not found at {RETENTION_PLAN_PATH}.\n"
            f"Run save_retention_plan_tool first before estimating campaign cost."
        )
    
    df = pd.read_csv(RETENTION_PLAN_PATH)
    
    # Calculate cost per customer based on their assigned offer
    def get_offer_cost(offer):
        return OFFER_COSTS.get(offer, OFFER_COSTS.get("Standard loyalty offer", 5.00))
    
    df["Offer_Cost"] = df["Retention_Offer"].apply(get_offer_cost)
    
    # Filter segments if needed
    if not include_all_segments:
        df = df[df["Risk_Segment"].isin(["High Risk", "Medium Risk"])]
    
    # Calculate totals
    total_customers = len(df)
    total_cost = df["Offer_Cost"].sum()
    avg_cost_per_customer = df["Offer_Cost"].mean()
    
    # Segment breakdown
    segment_costs = df.groupby("Risk_Segment").agg({
        "Offer_Cost": ["count", "sum", "mean"]
    }).round(2)
    segment_costs.columns = ["customer_count", "total_cost", "avg_cost"]
    
    # Budget analysis
    within_budget = total_cost <= budget_limit
    budget_remaining = budget_limit - total_cost
    budget_utilization_pct = (total_cost / budget_limit) * 100 if budget_limit > 0 else 0
    
    # Calculate potential ROI (assuming 30% retention success rate)
    estimated_retained = int(total_customers * 0.30)
    customer_lifetime_value = 500.00  # Assumption
    estimated_revenue_saved = estimated_retained * customer_lifetime_value
    estimated_roi = ((estimated_revenue_saved - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    # Save cost estimate report
    cost_report = {
        "timestamp": datetime.now().isoformat(),
        "budget_limit": budget_limit,
        "include_all_segments": include_all_segments,
        "total_customers": int(total_customers),
        "total_cost": round(float(total_cost), 2),
        "avg_cost_per_customer": round(float(avg_cost_per_customer), 2),
        "within_budget": within_budget,
        "budget_remaining": round(float(budget_remaining), 2),
        "budget_utilization_percent": round(float(budget_utilization_pct), 2),
        "segment_breakdown": segment_costs.to_dict("index"),
        "roi_estimate": {
            "estimated_retained_customers": estimated_retained,
            "customer_lifetime_value": customer_lifetime_value,
            "estimated_revenue_saved": round(float(estimated_revenue_saved), 2),
            "estimated_roi_percent": round(float(estimated_roi), 2)
        }
    }
    
    os.makedirs("artifacts/data", exist_ok=True)
    with open(CAMPAIGN_COST_PATH, "w") as f:
        json.dump(cost_report, f, indent=2)
    
    # Format output
    segment_text = "\n".join([
        f"  {seg}: {data['customer_count']} customers, "
        f"Total: ${data['total_cost']:.2f}, Avg: ${data['avg_cost']:.2f}"
        for seg, data in cost_report["segment_breakdown"].items()
    ])
    
    if within_budget:
        status = "✅ WITHIN BUDGET"
        recommendation = f"Campaign is financially feasible. Budget remaining: ${budget_remaining:.2f}"
    else:
        status = "❌ EXCEEDS BUDGET"
        recommendation = (
            f"Campaign exceeds budget by ${abs(budget_remaining):.2f}. "
            f"Consider targeting only High Risk customers or reducing offer costs."
        )
    
    return (
        f"Campaign Cost Estimate:\n"
        f"Status: {status}\n"
        f"Budget Limit: ${budget_limit:,.2f}\n"
        f"\nCost Breakdown:\n"
        f"  Total Customers: {total_customers}\n"
        f"  Total Cost: ${total_cost:,.2f}\n"
        f"  Avg Cost per Customer: ${avg_cost_per_customer:.2f}\n"
        f"  Budget Utilization: {budget_utilization_pct:.1f}%\n"
        f"\nSegment Breakdown:\n{segment_text}\n"
        f"\nROI Estimate:\n"
        f"  Estimated Retained: {estimated_retained} customers (30% success rate)\n"
        f"  Revenue Saved: ${estimated_revenue_saved:,.2f}\n"
        f"  Estimated ROI: {estimated_roi:.1f}%\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {CAMPAIGN_COST_PATH}"
    )


@tool("Assign Communication Channel")
def assign_communication_channel_tool(
    processed_data_path: str = "artifacts/data/processed_churn.csv"
) -> str:
    """
    Recommends Email vs. SMS vs. Call based on customer preference history.
    Optimizes channel selection for cost and effectiveness.
    """
    if not os.path.exists(RETENTION_PLAN_PATH):
        return (
            f"ERROR: Retention plan not found at {RETENTION_PLAN_PATH}.\n"
            f"Run save_retention_plan_tool first before assigning communication channels."
        )
    
    retention_df = pd.read_csv(RETENTION_PLAN_PATH)
    
    # Load processed data for customer preferences
    if os.path.exists(processed_data_path):
        processed_df = pd.read_csv(processed_data_path)
        
        # Merge to get customer preferences
        # Note: Adjust merge key based on your dataset (customerID or index)
        if "customerID" in retention_df.columns and "customerID" in processed_df.columns:
            merged = retention_df.merge(processed_df[["customerID", "PaperlessBilling", "PhoneService", "SeniorCitizen"]], 
                                       on="customerID", how="left")
        else:
            # Fallback: use index-based merge
            merged = retention_df.copy()
            for col in ["PaperlessBilling", "PhoneService", "SeniorCitizen"]:
                if col in processed_df.columns:
                    merged[col] = processed_df[col].values[:len(merged)]
                else:
                    merged[col] = 0
    else:
        merged = retention_df.copy()
        merged["PaperlessBilling"] = 0
        merged["PhoneService"] = 0
        merged["SeniorCitizen"] = 0
    
    # Assign communication channel based on preferences and risk segment
    def assign_channel(row):
        risk = row.get("Risk_Segment", "Medium Risk")
        paperless = row.get("PaperlessBilling", 0)
        phone = row.get("PhoneService", 0)
        senior = row.get("SeniorCitizen", 0)
        
        # Scoring for each channel
        email_score = 0
        sms_score = 0
        call_score = 0
        
        # Base scores by risk segment
        if risk == "High Risk":
            call_score += 3  # High risk deserves personal touch
            sms_score += 2
        elif risk == "Medium Risk":
            sms_score += 3
            email_score += 2
        else:
            email_score += 3  # Low risk can be automated
        
        # Preference-based adjustments
        if paperless == 1:
            email_score += 2
        if phone == 1:
            sms_score += 1
            call_score += 1
        if senior == 1:
            call_score += 2  # Seniors prefer calls
        
        # Select best channel
        scores = {"Email": email_score, "SMS": sms_score, "Call": call_score}
        best_channel = max(scores, key=scores.get)
        
        # Add cost info
        channel_cost = CHANNEL_INFO[best_channel]["cost"]
        channel_effectiveness = CHANNEL_INFO[best_channel]["effectiveness"]
        
        return pd.Series({
            "Communication_Channel": best_channel,
            "Channel_Cost": channel_cost,
            "Channel_Effectiveness": channel_effectiveness,
            "Channel_Score": scores[best_channel]
        })
    
    # Apply channel assignment
    channel_assignments = merged.apply(assign_channel, axis=1)
    merged["Communication_Channel"] = channel_assignments["Communication_Channel"]
    merged["Channel_Cost"] = channel_assignments["Channel_Cost"]
    merged["Channel_Effectiveness"] = channel_assignments["Channel_Effectiveness"]
    merged["Channel_Score"] = channel_assignments["Channel_Score"]
    
    # Save updated retention plan with channels
    merged.to_csv(RETENTION_PLAN_PATH, index=False)
    
    # Calculate channel distribution
    channel_distribution = merged["Communication_Channel"].value_counts().to_dict()
    total_channel_cost = merged["Channel_Cost"].sum()
    avg_effectiveness = merged["Channel_Effectiveness"].mean()
    
    # Save channel assignment report
    channel_report = {
        "timestamp": datetime.now().isoformat(),
        "total_customers": len(merged),
        "channel_distribution": channel_distribution,
        "total_channel_cost": round(float(total_channel_cost), 2),
        "avg_channel_cost": round(float(merged["Channel_Cost"].mean()), 2),
        "avg_effectiveness_score": round(float(avg_effectiveness), 4),
        "channel_info": CHANNEL_INFO
    }
    
    os.makedirs("artifacts/data", exist_ok=True)
    with open(COMMUNICATION_CHANNEL_PATH, "w") as f:
        json.dump(channel_report, f, indent=2)
    
    # Format output
    distribution_text = "\n".join([
        f"  {channel}: {count} customers ({count/len(merged)*100:.1f}%)"
        for channel, count in channel_distribution.items()
    ])
    
    return (
        f"Communication Channel Assignment Complete:\n"
        f"Total Customers: {len(merged)}\n"
        f"\nChannel Distribution:\n{distribution_text}\n"
        f"\nCost Analysis:\n"
        f"  Total Channel Cost: ${total_channel_cost:.2f}\n"
        f"  Avg Cost per Customer: ${merged['Channel_Cost'].mean():.2f}\n"
        f"  Avg Effectiveness Score: {avg_effectiveness:.4f}\n"
        f"\nChannel Assignment Rules:\n"
        f"  High Risk → Prefer Call (personal touch)\n"
        f"  Medium Risk → Prefer SMS (balanced)\n"
        f"  Low Risk → Prefer Email (automated)\n"
        f"  PaperlessBilling=1 → Boost Email\n"
        f"  SeniorCitizen=1 → Boost Call\n"
        f"\nReport saved to: {COMMUNICATION_CHANNEL_PATH}\n"
        f"Updated retention plan saved to: {RETENTION_PLAN_PATH}\n"
        f"\n💡 Channels assigned based on customer preferences and risk segment."
    )


@tool("Generate A/B Test Groups")
def generate_ab_test_groups_tool(
    treatment_pct: float = 0.80,
    random_state: int = 42
) -> str:
    """
    Splits the retention list into Control and Treatment groups for validation.
    Enables measurement of retention campaign effectiveness.
    """
    if not os.path.exists(RETENTION_PLAN_PATH):
        return (
            f"ERROR: Retention plan not found at {RETENTION_PLAN_PATH}.\n"
            f"Run save_retention_plan_tool first before generating A/B test groups."
        )
    
    df = pd.read_csv(RETENTION_PLAN_PATH)
    
    # Stratified split by risk segment to ensure balance
    np.random.seed(random_state)
    
    control_indices = []
    treatment_indices = []
    
    for segment in df["Risk_Segment"].unique():
        segment_mask = df["Risk_Segment"] == segment
        segment_indices = df[segment_mask].index.tolist()
        
        n_treatment = int(len(segment_indices) * treatment_pct)
        
        np.random.shuffle(segment_indices)
        treatment_indices.extend(segment_indices[:n_treatment])
        control_indices.extend(segment_indices[n_treatment:])
    
    # Assign groups
    df["Test_Group"] = "Control"
    df.loc[treatment_indices, "Test_Group"] = "Treatment"
    
    # Save updated retention plan
    df.to_csv(RETENTION_PLAN_PATH, index=False)
    
    # Calculate group statistics
    control_count = len(control_indices)
    treatment_count = len(treatment_indices)
    total_count = len(df)
    
    control_pct = control_count / total_count * 100
    treatment_pct_actual = treatment_count / total_count * 100
    
    # Segment distribution in each group
    control_segment_dist = df[df["Test_Group"] == "Control"]["Risk_Segment"].value_counts().to_dict()
    treatment_segment_dist = df[df["Test_Group"] == "Treatment"]["Risk_Segment"].value_counts().to_dict()
    
    # Save A/B test report
    ab_test_report = {
        "timestamp": datetime.now().isoformat(),
        "random_state": random_state,
        "treatment_percentage_target": treatment_pct,
        "total_customers": total_count,
        "control_group": {
            "count": control_count,
            "percentage": round(control_pct, 2),
            "segment_distribution": control_segment_dist
        },
        "treatment_group": {
            "count": treatment_count,
            "percentage": round(treatment_pct_actual, 2),
            "segment_distribution": treatment_segment_dist
        },
        "stratified_by": "Risk_Segment",
        "measurement_plan": {
            "primary_metric": "Retention Rate (1 - actual churn after campaign)",
            "secondary_metrics": ["Offer redemption rate", "Customer satisfaction score"],
            "test_duration_days": 30,
            "statistical_significance_level": 0.05
        }
    }
    
    os.makedirs("artifacts/data", exist_ok=True)
    with open(AB_TEST_PATH, "w") as f:
        json.dump(ab_test_report, f, indent=2)
    
    # Format output
    control_segments = "\n".join([f"    {seg}: {count}" for seg, count in control_segment_dist.items()])
    treatment_segments = "\n".join([f"    {seg}: {count}" for seg, count in treatment_segment_dist.items()])
    
    return (
        f"A/B Test Groups Generated:\n"
        f"Random State: {random_state}\n"
        f"Target Treatment %: {treatment_pct*100:.0f}%\n"
        f"\nGroup Sizes:\n"
        f"  Control:   {control_count} customers ({control_pct:.1f}%)\n"
        f"  Treatment: {treatment_count} customers ({treatment_pct_actual:.1f}%)\n"
        f"\nControl Group Segment Distribution:\n{control_segments}\n"
        f"\nTreatment Group Segment Distribution:\n{treatment_segments}\n"
        f"\nMeasurement Plan:\n"
        f"  Primary Metric: Retention Rate\n"
        f"  Test Duration: 30 days\n"
        f"  Significance Level: 0.05\n"
        f"\nReport saved to: {AB_TEST_PATH}\n"
        f"Updated retention plan saved to: {RETENTION_PLAN_PATH}\n"
        f"\n💡 Control group receives no intervention to measure baseline churn."
    )


@tool("Optimize Budget Allocation")
def optimize_budget_allocation_tool(
    total_budget: float = 10000.00,
    min_customers_per_segment: int = 10
) -> str:
    """
    Uses simple optimization to maximize saved customers within a fixed budget.
    Allocates budget across segments for maximum ROI.
    """
    if not os.path.exists(RETENTION_PLAN_PATH):
        return (
            f"ERROR: Retention plan not found at {RETENTION_PLAN_PATH}.\n"
            f"Run save_retention_plan_tool first before optimizing budget allocation."
        )
    
    df = pd.read_csv(RETENTION_PLAN_PATH)
    
    # Calculate cost per customer based on offer
    def get_offer_cost(offer):
        return OFFER_COSTS.get(offer, OFFER_COSTS.get("Standard loyalty offer", 5.00))
    
    df["Offer_Cost"] = df["Retention_Offer"].apply(get_offer_cost)
    
    # Estimate retention probability by segment (based on historical data or assumptions)
    retention_rates = {
        "High Risk": 0.35,  # High risk but high intervention effectiveness
        "Medium Risk": 0.45,  # Medium risk, good response to offers
        "Low Risk": 0.20,  # Low risk, less likely to churn anyway
    }
    
    df["Estimated_Retention_Rate"] = df["Risk_Segment"].map(retention_rates)
    df["Expected_Value"] = df["Estimated_Retention_Rate"] * 500.00  # Customer lifetime value
    df["ROI_Score"] = df["Expected_Value"] / (df["Offer_Cost"] + 0.01)  # Avoid division by zero
    
    # Optimization: Greedy allocation by ROI score
    df_sorted = df.sort_values("ROI_Score", ascending=False).reset_index(drop=True)
    
    remaining_budget = total_budget
    selected_customers = []
    segment_counts = {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0}
    
    for idx, row in df_sorted.iterrows():
        segment = row["Risk_Segment"]
        cost = row["Offer_Cost"]
        
        # Check minimum segment requirements
        if segment_counts[segment] < min_customers_per_segment:
            # Must include at least min_customers_per_segment from each segment
            if remaining_budget >= cost:
                selected_customers.append(idx)
                remaining_budget -= cost
                segment_counts[segment] += 1
        else:
            # Allocate by ROI
            if remaining_budget >= cost:
                selected_customers.append(idx)
                remaining_budget -= cost
                segment_counts[segment] += 1
    
    # Create optimized allocation
    df["Included_in_Optimized_Plan"] = False
    df.loc[selected_customers, "Included_in_Optimized_Plan"] = True
    
    # Calculate optimized metrics
    optimized_df = df[df["Included_in_Optimized_Plan"] == True]
    
    total_customers_optimized = len(optimized_df)
    total_cost_optimized = optimized_df["Offer_Cost"].sum()
    expected_retained = (optimized_df["Estimated_Retention_Rate"] * optimized_df["Expected_Value"]).sum() / 500.00
    expected_revenue = optimized_df["Expected_Value"].sum()
    roi = ((expected_revenue - total_cost_optimized) / total_cost_optimized * 100) if total_cost_optimized > 0 else 0
    
    # Segment breakdown
    segment_optimization = optimized_df.groupby("Risk_Segment").agg({
        "Offer_Cost": ["count", "sum"],
        "Expected_Value": "sum",
        "Estimated_Retention_Rate": "mean"
    }).round(2)
    segment_optimization.columns = ["customers", "total_cost", "expected_revenue", "avg_retention_rate"]
    
    # Save optimization report
    optimization_report = {
        "timestamp": datetime.now().isoformat(),
        "total_budget": total_budget,
        "min_customers_per_segment": min_customers_per_segment,
        "original_customer_count": len(df),
        "optimized_customer_count": total_customers_optimized,
        "budget_used": round(float(total_cost_optimized), 2),
        "budget_remaining": round(float(total_budget - total_cost_optimized), 2),
        "budget_utilization_percent": round(float(total_cost_optimized / total_budget * 100), 2),
        "expected_retained_customers": round(float(expected_retained), 1),
        "expected_revenue": round(float(expected_revenue), 2),
        "roi_percent": round(float(roi), 2),
        "segment_allocation": segment_optimization.to_dict("index"),
        "optimization_method": "Greedy ROI-based with segment minimums"
    }
    
    os.makedirs("artifacts/data", exist_ok=True)
    with open(BUDGET_OPTIMIZATION_PATH, "w") as f:
        json.dump(optimization_report, f, indent=2)
    
    # Save optimized retention plan
    optimized_plan_path = "artifacts/data/retention_plan_optimized.csv"
    optimized_df.to_csv(optimized_plan_path, index=False)
    
    # Format output
    segment_text = "\n".join([
        f"  {seg}: {data['customers']} customers, "
        f"Cost: ${data['total_cost']:.2f}, "
        f"Expected Revenue: ${data['expected_revenue']:.2f}"
        for seg, data in optimization_report["segment_allocation"].items()
    ])
    
    return (
        f"Budget Optimization Complete:\n"
        f"Total Budget: ${total_budget:,.2f}\n"
        f"Optimization Method: Greedy ROI-based with segment minimums\n"
        f"\nResults:\n"
        f"  Original Customers: {len(df)}\n"
        f"  Optimized Customers: {total_customers_optimized}\n"
        f"  Budget Used: ${total_cost_optimized:,.2f}\n"
        f"  Budget Remaining: ${total_budget - total_cost_optimized:,.2f}\n"
        f"  Budget Utilization: {total_cost_optimized / total_budget * 100:.1f}%\n"
        f"\nExpected Outcomes:\n"
        f"  Expected Retained: {expected_retained:.1f} customers\n"
        f"  Expected Revenue: ${expected_revenue:,.2f}\n"
        f"  ROI: {roi:.1f}%\n"
        f"\nSegment Allocation:\n{segment_text}\n"
        f"\nFiles Saved:\n"
        f"  Optimization Report: {BUDGET_OPTIMIZATION_PATH}\n"
        f"  Optimized Plan: {optimized_plan_path}\n"
        f"\n💡 Budget allocated to maximize retained customers within budget constraints."
    )