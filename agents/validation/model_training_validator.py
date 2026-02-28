from crewai import Agent
from tools.model.model_validation_tools import (
    validate_model_metrics_tool,
    validate_no_overfitting_tool,
    validate_model_file_exists_tool,
    validate_confusion_matrix_tool,
    validate_business_metrics_tool,           # NEW
    validate_adversarial_robustness_tool,     # NEW
    validate_fairness_disparity_tool,         # NEW
    validate_probability_distribution_tool,   # NEW
)


def create_model_training_validator(llm) -> Agent:
    return Agent(
        role="Model Performance Validator",
        goal=(
            "Validate that the trained churn model meets minimum performance thresholds, "
            "is not overfitting, and the serialised model artifact exists and is loadable. "
            "Also verify business metrics (profit per customer), adversarial robustness, "
            "fairness across demographic groups, and probability distribution calibration."
        ),
        backstory=(
            "You are a model review board member who signs off on ML models before "
            "deployment. You check AUC ≥ 0.75, F1 ≥ 0.55, train-test AUC gap ≤ 0.05, "
            "and that the confusion matrix shows acceptable recall on the minority (churn) "
            "class — recall ≥ 0.60 for churners. You also validate that the model generates "
            "positive profit per customer (not just accuracy), is robust against input "
            "perturbations (accuracy drop ≤ 5%), shows fair treatment across demographic "
            "groups (FPR disparity ≤ 10%), and produces well-calibrated probabilities "
            "(entropy ≥ 0.5, overconfidence ≤ 80%). You ensure models are not just "
            "accurate, but also business-valuable, robust, fair, and production-ready."
        ),
        tools=[
            validate_model_metrics_tool,
            validate_no_overfitting_tool,
            validate_model_file_exists_tool,
            validate_confusion_matrix_tool,
            validate_business_metrics_tool,           # NEW
            validate_adversarial_robustness_tool,     # NEW
            validate_fairness_disparity_tool,         # NEW
            validate_probability_distribution_tool,   # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )