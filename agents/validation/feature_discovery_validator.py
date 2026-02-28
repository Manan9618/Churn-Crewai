from crewai import Agent
from tools.feature.feature_validation_tools import (
    validate_feature_count_tool,
    validate_no_target_leakage_tool,
    validate_feature_variance_tool,
    validate_selected_features_saved_tool,
    validate_multicollinearity_vif_tool,      # NEW
    validate_feature_stability_tool,           # NEW
    validate_information_value_tool,           # NEW
)


def create_feature_discovery_validator(llm) -> Agent:
    return Agent(
        role="Feature Selection Validator",
        goal=(
            "Ensure the feature selection process produced a valid, leakage-free, "
            "non-degenerate feature subset that is persisted correctly for model training. "
            "Verify features have low multicollinearity (VIF), stable importance across "
            "CV folds, and sufficient predictive power (Information Value)."
        ),
        backstory=(
            "You are a feature engineering auditor. You check that: at least 5 features "
            "were selected, no direct leakage columns (like customerID) are included, "
            "all selected features have non-zero variance, and the feature list file "
            "was saved to disk for reproducibility. You also validate that features don't "
            "suffer from multicollinearity (VIF ≤ 5.0), have stable importance across "
            "5-fold cross-validation (std ≤ 0.3), and possess sufficient predictive power "
            "(Information Value ≥ 0.02) to ensure the model will generalize well."
        ),
        tools=[
            validate_feature_count_tool,
            validate_no_target_leakage_tool,
            validate_feature_variance_tool,
            validate_selected_features_saved_tool,
            validate_multicollinearity_vif_tool,      # NEW
            validate_feature_stability_tool,           # NEW
            validate_information_value_tool,           # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )