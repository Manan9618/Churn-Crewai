from crewai import Agent
from tools.strategy.strategy_tools import (
    segment_at_risk_customers_tool,
    generate_retention_offers_tool,
    prioritize_customers_tool,
    save_retention_plan_tool,
    estimate_campaign_cost_tool,           # NEW
    assign_communication_channel_tool,     # NEW
    generate_ab_test_groups_tool,          # NEW
    optimize_budget_allocation_tool,       # NEW
)


def create_retention_strategy_agent(llm) -> Agent:
    return Agent(
        role="Customer Retention Strategist",
        goal=(
            "Analyse the churn predictions and model explanations to design targeted "
            "retention strategies — segmenting at-risk customers by churn probability "
            "and key churn drivers, recommending personalised retention offers, estimating "
            "campaign costs against budget, assigning optimal communication channels, "
            "generating A/B test groups for validation, and optimizing budget allocation "
            "to maximize saved customers within financial constraints."
        ),
        backstory=(
            "You are a CRM and customer success strategist with 10+ years of experience "
            "in the telecom industry. You know that month-to-month contract customers "
            "with high monthly charges and no online security are most at risk, and you "
            "design cost-effective retention campaigns tailored to each customer segment. "
            "You go beyond basic segmentation by calculating campaign costs and ROI to "
            "ensure financial feasibility, assigning communication channels (Email/SMS/Call) "
            "based on customer preferences for maximum effectiveness, creating A/B test "
            "groups (Control/Treatment) to measure campaign impact, and using ROI-based "
            "optimization to allocate budget where it saves the most customers. You deliver "
            "retention strategies that are not just targeted, but also financially sound, "
            "measurable, and optimized for maximum business impact."
        ),
        tools=[
            segment_at_risk_customers_tool,
            generate_retention_offers_tool,
            prioritize_customers_tool,
            save_retention_plan_tool,
            estimate_campaign_cost_tool,           # NEW
            assign_communication_channel_tool,     # NEW
            generate_ab_test_groups_tool,          # NEW
            optimize_budget_allocation_tool,       # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )