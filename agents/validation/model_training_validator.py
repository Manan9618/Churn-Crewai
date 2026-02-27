from crewai import Agent
from tools.model.model_validation_tools import (
    validate_model_metrics_tool,
    validate_no_overfitting_tool,
    validate_model_file_exists_tool,
    validate_confusion_matrix_tool,
)


def create_model_training_validator(llm) -> Agent:
    return Agent(
        role="Model Performance Validator",
        goal=(
            "Validate that the trained churn model meets minimum performance thresholds, "
            "is not overfitting, and the serialised model artifact exists and is loadable."
        ),
        backstory=(
            "You are a model review board member who signs off on ML models before "
            "deployment. You check AUC ≥ 0.75, F1 ≥ 0.55, train-test AUC gap ≤ 0.05, "
            "and that the confusion matrix shows acceptable recall on the minority (churn) "
            "class — recall ≥ 0.60 for churners."
        ),
        tools=[
            validate_model_metrics_tool,
            validate_no_overfitting_tool,
            validate_model_file_exists_tool,
            validate_confusion_matrix_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )