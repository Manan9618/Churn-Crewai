from crewai import Agent
from tools.feedback.feedback_validation_tools import (
    validate_experiment_log_tool,
    validate_improvement_suggestions_tool,
    validate_metrics_improvement_tool,
)


def create_feedback_loop_validator(llm) -> Agent:
    return Agent(
        role="Feedback Loop Validator",
        goal=(
            "Ensure that experiment logs are correctly recorded, improvement suggestions "
            "are concrete and actionable, and that consecutive iterations show measurable "
            "metric improvement."
        ),
        backstory=(
            "You are an MLOps continuous improvement auditor. You verify that each "
            "feedback cycle: logs all key metrics (AUC, F1, Recall), provides at least "
            "one actionable recommendation, and that when applied, recommendations result "
            "in at least 0.01 AUC improvement over the previous iteration."
        ),
        tools=[
            validate_experiment_log_tool,
            validate_improvement_suggestions_tool,
            validate_metrics_improvement_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )