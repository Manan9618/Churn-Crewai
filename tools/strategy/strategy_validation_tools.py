import pandas as pd
import numpy as np
import json
import os
from crewai.tools import tool

RETENTION_PLAN_PATH = "artifacts/data/retention_plan.csv"
PREDICTIONS_PATH = "artifacts/data/predictions.csv"
CAMPAIGN_COST_PATH = "artifacts/data/campaign_cost_estimate.json"
BUDGET_OPTIMIZATION_PATH = "artifacts/data/budget_optimization_report.json"
PROCESSED_DATA_PATH = "artifacts/data/processed_churn.csv"

VALID_SEGMENTS = {"High Risk", "Medium Risk", "Low Risk"}

# Validation thresholds
DEFAULT_BUDGET_LIMIT = 10000.00
MIN_HIGH_RISK_COVERAGE = 0.80  # At least 80% of high-risk customers should be included
ETHICAL_SENSITIVE_COLUMNS = ["gender", "SeniorCitizen", "Partner", "Dependents"]


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


# ── NEW RETENTION STRATEGY VALIDATION TOOLS ADDED BELOW ───────────────────────


@tool("Validate Budget Compliance")
def validate_budget_compliance_tool(
    budget_limit: float = DEFAULT_BUDGET_LIMIT
) -> str:
    """
    Ensures total offer cost is within the total budget.
    Validates financial feasibility of the retention campaign.
    """
    # Check if retention plan exists
    if not os.path.exists(RETENTION_PLAN_PATH):
        return (
            f"VALIDATION FAILED: Retention plan not found at {RETENTION_PLAN_PATH}.\n"
            f"Run save_retention_plan_tool first before validating budget compliance."
        )
    
    df = pd.read_csv(RETENTION_PLAN_PATH)
    
    # Try to load cost estimate report first
    if os.path.exists(CAMPAIGN_COST_PATH):
        with open(CAMPAIGN_COST_PATH, "r") as f:
            cost_report = json.load(f)
        
        total_cost = cost_report.get("total_cost", 0)
        within_budget = cost_report.get("within_budget", False)
        budget_remaining = cost_report.get("budget_remaining", 0)
        budget_utilization = cost_report.get("budget_utilization_percent", 0)
    else:
        # Calculate cost from retention plan directly
        # Cost assumptions (should match strategy_tools.py)
        offer_costs = {
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
        
        def get_offer_cost(offer):
            return offer_costs.get(offer, 5.00)
        
        if "Retention_Offer" in df.columns:
            df["Offer_Cost"] = df["Retention_Offer"].apply(get_offer_cost)
            total_cost = df["Offer_Cost"].sum()
        else:
            return (
                f"VALIDATION FAILED: Cannot calculate costs.\n"
                f"Missing 'Retention_Offer' column in retention plan AND "
                f"campaign_cost_estimate.json not found."
            )
        
        within_budget = total_cost <= budget_limit
        budget_remaining = budget_limit - total_cost
        budget_utilization = (total_cost / budget_limit * 100) if budget_limit > 0 else 0
    
    # Segment breakdown
    if "Risk_Segment" in df.columns and "Offer_Cost" in df.columns:
        segment_costs = df.groupby("Risk_Segment")["Offer_Cost"].agg(["count", "sum"]).to_dict()
    else:
        segment_costs = {}
    
    # Save validation report
    validation_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "budget_limit": budget_limit,
        "total_cost": round(float(total_cost), 2),
        "within_budget": within_budget,
        "budget_remaining": round(float(budget_remaining), 2),
        "budget_utilization_percent": round(float(budget_utilization), 2),
        "segment_costs": segment_costs
    }
    
    validation_report_path = "artifacts/data/budget_compliance_validation.json"
    os.makedirs("artifacts/data", exist_ok=True)
    with open(validation_report_path, "w") as f:
        json.dump(validation_report, f, indent=2)
    
    # Format output
    if within_budget:
        status = "✅ WITHIN BUDGET"
        recommendation = f"Campaign is financially feasible. Budget remaining: ${budget_remaining:.2f}"
    else:
        status = "❌ EXCEEDS BUDGET"
        recommendation = (
            f"Campaign exceeds budget by ${abs(budget_remaining):.2f}. "
            f"Consider reducing offer costs or targeting fewer customers."
        )
    
    segment_text = ""
    if segment_costs:
        segment_text = "\nSegment Breakdown:\n" + "\n".join([
            f"  {seg}: {data['count']} customers, Total Cost: ${data['sum']:.2f}"
            for seg, data in segment_costs.items()
        ])
    
    return (
        f"Budget Compliance Validation:\n"
        f"Status: {status}\n"
        f"Budget Limit: ${budget_limit:,.2f}\n"
        f"Total Campaign Cost: ${total_cost:,.2f}\n"
        f"Budget Utilization: {budget_utilization:.1f}%\n"
        f"Budget Remaining: ${budget_remaining:,.2f}\n"
        f"{segment_text}\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {validation_report_path}"
    )


@tool("Validate Ethical Compliance")
def validate_ethical_compliance_tool(
    processed_data_path: str = PROCESSED_DATA_PATH,
    significance_threshold: float = 0.10
) -> str:
    """
    Checks that offers don't discriminate based on demographics.
    Ensures fair treatment across gender, age, and other sensitive attributes.
    """
    if not os.path.exists(RETENTION_PLAN_PATH):
        return (
            f"VALIDATION FAILED: Retention plan not found at {RETENTION_PLAN_PATH}.\n"
            f"Run save_retention_plan_tool first before validating ethical compliance."
        )
    
    retention_df = pd.read_csv(RETENTION_PLAN_PATH)
    
    # Check if we have demographic data
    if os.path.exists(processed_data_path):
        processed_df = pd.read_csv(processed_data_path)
        
        # Merge to get demographic data
        if "customerID" in retention_df.columns and "customerID" in processed_df.columns:
            merged = retention_df.merge(
                processed_df[["customerID"] + ETHICAL_SENSITIVE_COLUMNS],
                on="customerID",
                how="left"
            )
        else:
            # Fallback: use index-based merge
            merged = retention_df.copy()
            for col in ETHICAL_SENSITIVE_COLUMNS:
                if col in processed_df.columns:
                    merged[col] = processed_df[col].values[:len(merged)]
                else:
                    merged[col] = None
    else:
        merged = retention_df.copy()
        for col in ETHICAL_SENSITIVE_COLUMNS:
            merged[col] = None
    
    # Check for discrimination in offer assignment
    ethical_issues = []
    fairness_results = {}
    
    for col in ETHICAL_SENSITIVE_COLUMNS:
        if col not in merged.columns or merged[col].isna().all():
            fairness_results[col] = {
                "status": "SKIPPED",
                "reason": "Column not available"
            }
            continue
        
        # Group by demographic attribute
        groups = merged[col].unique()
        
        if len(groups) < 2:
            fairness_results[col] = {
                "status": "SKIPPED",
                "reason": "Only one group present"
            }
            continue
        
        # Check offer distribution across groups
        group_offer_stats = merged.groupby(col).agg({
            "Retention_Offer": lambda x: x.value_counts().to_dict(),
            "Churn_Probability": "mean",
            "Priority_Score": "mean" if "Priority_Score" in merged.columns else lambda x: None
        }).to_dict()
        
        # Check if high-value offers are disproportionately assigned to specific groups
        high_value_offers = [
            "Upgrade to 2-year contract with 20% discount",
            "Free TechSupport + OnlineSecurity bundle for 6 months",
            "Dedicated account manager outreach call"
        ]
        
        group_high_value_rates = {}
        for group in groups:
            group_data = merged[merged[col] == group]
            high_value_count = group_data["Retention_Offer"].isin(high_value_offers).sum()
            high_value_rate = high_value_count / len(group_data) if len(group_data) > 0 else 0
            group_high_value_rates[group] = high_value_rate
        
        # Check for significant disparity
        if len(group_high_value_rates) >= 2:
            rates = list(group_high_value_rates.values())
            max_rate = max(rates)
            min_rate = min(rates)
            disparity = max_rate - min_rate
            
            if disparity > significance_threshold:
                ethical_issues.append({
                    "attribute": col,
                    "issue": "Offer disparity",
                    "details": f"High-value offer rate varies by {disparity:.2%} across {col} groups",
                    "group_rates": group_high_value_rates
                })
        
        fairness_results[col] = {
            "status": "PASSED" if disparity <= significance_threshold else "FAILED",
            "disparity": round(float(disparity), 4),
            "threshold": significance_threshold,
            "group_rates": group_high_value_rates
        }
    
    # Check priority score distribution across groups
    if "Priority_Score" in merged.columns:
        for col in ETHICAL_SENSITIVE_COLUMNS:
            if col in merged.columns and not merged[col].isna().all():
                groups = merged[col].unique()
                if len(groups) >= 2:
                    group_priority = merged.groupby(col)["Priority_Score"].mean()
                    priority_disparity = group_priority.max() - group_priority.min()
                    
                    if priority_disparity > 10:  # More than 10 point difference
                        ethical_issues.append({
                            "attribute": col,
                            "issue": "Priority score disparity",
                            "details": f"Priority scores vary by {priority_disparity:.1f} points across {col} groups"
                        })
    
    # Save ethical compliance report
    compliance_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "attributes_checked": ETHICAL_SENSITIVE_COLUMNS,
        "significance_threshold": significance_threshold,
        "ethical_issues": ethical_issues,
        "fairness_results": fairness_results,
        "all_compliant": len(ethical_issues) == 0
    }
    
    compliance_report_path = "artifacts/data/ethical_compliance_validation.json"
    os.makedirs("artifacts/data", exist_ok=True)
    with open(compliance_report_path, "w") as f:
        json.dump(compliance_report, f, indent=2)
    
    # Format output
    if ethical_issues:
        status = "❌ ETHICAL ISSUES DETECTED"
        issues_text = "\n".join([
            f"  ⚠️ {issue['attribute']}: {issue['issue']} - {issue['details']}"
            for issue in ethical_issues
        ])
        recommendation = "Review offer assignment logic to ensure fair treatment across demographics."
    else:
        status = "✅ ETHICALLY COMPLIANT"
        issues_text = "No discriminatory patterns detected."
        recommendation = "Retention offers are fairly distributed across demographic groups."
    
    fairness_text = "\n".join([
        f"  {col}: {result['status']} "
        f"({result.get('disparity', 'N/A'):.4f} disparity)" if 'disparity' in result else f"  {col}: {result['status']}"
        for col, result in fairness_results.items()
    ])
    
    return (
        f"Ethical Compliance Validation:\n"
        f"Status: {status}\n"
        f"Attributes Checked: {ETHICAL_SENSITIVE_COLUMNS}\n"
        f"Significance Threshold: {significance_threshold}\n"
        f"\nFairness Results:\n{fairness_text}\n"
        f"\nEthical Issues:\n{issues_text}\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {compliance_report_path}"
    )


@tool("Validate Coverage Rate")
def validate_coverage_rate_tool(
    min_coverage_pct: float = MIN_HIGH_RISK_COVERAGE
) -> str:
    """
    Ensures a minimum percentage of High Risk customers are included in the retention plan.
    Validates that the most at-risk customers are not being excluded.
    """
    if not os.path.exists(PREDICTIONS_PATH):
        return (
            f"VALIDATION FAILED: Predictions file not found at {PREDICTIONS_PATH}.\n"
            f"Ensure predictions have been generated before validating coverage rate."
        )
    
    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    
    # Get all high-risk customers
    high_risk_customers = predictions_df[predictions_df["Risk_Segment"] == "High Risk"]
    total_high_risk = len(high_risk_customers)
    
    if total_high_risk == 0:
        return (
            f"VALIDATION SKIPPED: No High Risk customers in predictions.\n"
            f"Cannot validate coverage rate without High Risk segment."
        )
    
    # Check retention plan
    if os.path.exists(RETENTION_PLAN_PATH):
        retention_df = pd.read_csv(RETENTION_PLAN_PATH)
        
        # Check if retention plan includes high-risk customers
        if "Risk_Segment" in retention_df.columns:
            high_risk_in_plan = retention_df[retention_df["Risk_Segment"] == "High Risk"]
            high_risk_covered = len(high_risk_in_plan)
        else:
            # Fallback: assume all in retention plan are at-risk
            high_risk_covered = len(retention_df)
    else:
        high_risk_covered = 0
    
    # Calculate coverage rate
    coverage_rate = high_risk_covered / total_high_risk if total_high_risk > 0 else 0
    meets_minimum = coverage_rate >= min_coverage_pct
    
    # Check optimized plan if available
    optimized_coverage = None
    if os.path.exists("artifacts/data/retention_plan_optimized.csv"):
        optimized_df = pd.read_csv("artifacts/data/retention_plan_optimized.csv")
        if "Risk_Segment" in optimized_df.columns:
            optimized_high_risk = optimized_df[optimized_df["Risk_Segment"] == "High Risk"]
            optimized_coverage = len(optimized_high_risk) / total_high_risk if total_high_risk > 0 else 0
    
    # Save coverage report
    coverage_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_high_risk_customers": total_high_risk,
        "high_risk_in_plan": high_risk_covered,
        "coverage_rate": round(float(coverage_rate), 4),
        "coverage_rate_percent": round(float(coverage_rate) * 100, 2),
        "minimum_required": min_coverage_pct,
        "meets_minimum": meets_minimum,
        "optimized_plan_coverage": round(float(optimized_coverage), 4) if optimized_coverage is not None else None
    }
    
    coverage_report_path = "artifacts/data/coverage_rate_validation.json"
    os.makedirs("artifacts/data", exist_ok=True)
    with open(coverage_report_path, "w") as f:
        json.dump(coverage_report, f, indent=2)
    
    # Format output
    if meets_minimum:
        status = "✅ COVERAGE ADEQUATE"
        recommendation = f"High Risk coverage ({coverage_rate*100:.1f}%) meets minimum requirement ({min_coverage_pct*100:.0f}%)."
    else:
        status = "❌ COVERAGE INSUFFICIENT"
        recommendation = (
            f"High Risk coverage ({coverage_rate*100:.1f}%) below minimum ({min_coverage_pct*100:.0f}%). "
            f"Missing {total_high_risk - high_risk_covered} High Risk customers. "
            f"Consider including more High Risk customers in the retention plan."
        )
    
    optimized_text = ""
    if optimized_coverage is not None:
        optimized_text = f"\nOptimized Plan Coverage: {optimized_coverage*100:.1f}%"
    
    return (
        f"Coverage Rate Validation:\n"
        f"Status: {status}\n"
        f"Total High Risk Customers: {total_high_risk}\n"
        f"High Risk in Plan: {high_risk_covered}\n"
        f"Coverage Rate: {coverage_rate*100:.1f}%\n"
        f"Minimum Required: {min_coverage_pct*100:.0f}%\n"
        f"{optimized_text}\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {coverage_report_path}"
    )