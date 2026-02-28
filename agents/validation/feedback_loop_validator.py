from crewai import Agent
from tools.feedback.feedback_validation_tools import (
    validate_experiment_log_tool,
    validate_improvement_suggestions_tool,
    validate_metrics_improvement_tool,
    validate_statistical_significance_tool,      # NEW
    validate_resource_constraints_tool,           # NEW
    validate_rollback_integrity_tool,             # NEW
)


def create_feedback_loop_validator(llm) -> Agent:
    return Agent(
        role="Feedback Loop Validator",
        goal=(
            "Ensure that experiment logs are correctly recorded, improvement suggestions "
            "are concrete and actionable, and that consecutive iterations show measurable "
            "metric improvement. Validate that improvements are statistically significant "
            "(not random variance), that the model fits within resource constraints, and "
            "that any rollbacks were executed with full integrity."
        ),
        backstory=(
            "You are an MLOps continuous improvement auditor. You verify that each "
            "feedback cycle: logs all key metrics (AUC, F1, Recall), provides at least "
            "one actionable recommendation, and that when applied, recommendations result "
            "in at least 0.01 AUC improvement over the previous iteration. You also "
            "perform statistical significance testing (T-Test, p-value < 0.05) to ensure "
            "improvements aren't due to random variance. You validate resource constraints "
            "(features ≤50, memory ≤16GB, training time ≤60min, AUC ≥0.75) to ensure "
            "production feasibility. If a rollback was performed, you verify all artifacts "
            "were restored correctly with file integrity checks. You provide a final "
            "FEEDBACK LOOP VALIDATED or FEEDBACK LOOP FAILED verdict."
        ),
        tools=[
            validate_experiment_log_tool,
            validate_improvement_suggestions_tool,
            validate_metrics_improvement_tool,
            validate_statistical_significance_tool,      # NEW
            validate_resource_constraints_tool,           # NEW
            validate_rollback_integrity_tool,             # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )