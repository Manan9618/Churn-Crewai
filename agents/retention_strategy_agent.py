from crewai import Agent
from tools.strategy.strategy_tools import (
    segment_at_risk_customers_tool,
    generate_retention_offers_tool,
    prioritize_customers_tool,
    save_retention_plan_tool,
)


def create_retention_strategy_agent(llm) -> Agent:
    return Agent(
        role="Customer Retention Strategist",
        goal=(
            "Analyse the churn predictions and model explanations to design targeted "
            "retention strategies — segmenting at-risk customers by churn probability "
            "and key churn drivers, then recommending personalised retention offers."
        ),
        backstory=(
            "You are a CRM and customer success strategist with 10+ years of experience "
            "in the telecom industry. You know that month-to-month contract customers "
            "with high monthly charges and no online security are most at risk, and you "
            "design cost-effective retention campaigns tailored to each customer segment."
        ),
        tools=[
            segment_at_risk_customers_tool,
            generate_retention_offers_tool,
            prioritize_customers_tool,
            save_retention_plan_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )