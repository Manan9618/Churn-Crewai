from crewai import Agent
from tools.feature.feature_validation_tools import (
    validate_feature_count_tool,
    validate_no_target_leakage_tool,
    validate_feature_variance_tool,
    validate_selected_features_saved_tool,
)


def create_feature_discovery_validator(llm) -> Agent:
    return Agent(
        role="Feature Selection Validator",
        goal=(
            "Ensure the feature selection process produced a valid, leakage-free, "
            "non-degenerate feature subset that is persisted correctly for model training."
        ),
        backstory=(
            "You are a feature engineering auditor. You check that: at least 5 features "
            "were selected, no direct leakage columns (like customerID) are included, "
            "all selected features have non-zero variance, and the feature list file "
            "was saved to disk for reproducibility."
        ),
        tools=[
            validate_feature_count_tool,
            validate_no_target_leakage_tool,
            validate_feature_variance_tool,
            validate_selected_features_saved_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )