from crewai import Agent
from tools.strategy.strategy_validation_tools import (
    validate_retention_plan_file_tool,
    validate_customer_segments_tool,
    validate_offers_assigned_tool,
    validate_priority_scores_tool,
)


def create_retention_strategy_validator(llm) -> Agent:
    return Agent(
        role="Retention Plan Validator",
        goal=(
            "Confirm that the retention strategy plan is complete, all at-risk customers "
            "have been assigned to a segment, every segment has at least one retention "
            "offer, and priority scores are normalised correctly."
        ),
        backstory=(
            "You are a CRM operations auditor who ensures retention campaigns are "
            "actionable. You check that no high-risk customer (Churn_Probability > 0.7) "
            "was left without an offer, that segment labels are consistent, and that the "
            "retention plan CSV can be imported directly into the CRM system."
        ),
        tools=[
            validate_retention_plan_file_tool,
            validate_customer_segments_tool,
            validate_offers_assigned_tool,
            validate_priority_scores_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )