from crewai import Agent
from tools.feature.feature_tools import (
    correlation_analysis_tool,
    feature_importance_tool,
    variance_threshold_tool,
    select_top_features_tool,
    generate_interaction_features_tool,      # NEW
    recursive_feature_elimination_tool,      # NEW
    calculate_mutual_information_tool,       # NEW
    cluster_features_tool,                   # NEW
)


def create_feature_discovery_agent(llm) -> Agent:
    return Agent(
        role="Feature Discovery Analyst",
        goal=(
            "Identify the most predictive features for customer churn by performing "
            "correlation analysis, computing feature importances, detecting non-linear "
            "relationships via mutual information, generating interaction features, "
            "clustering correlated features to reduce multicollinearity, applying "
            "recursive feature elimination, and selecting an optimal subset that "
            "maximizes model performance while reducing noise and ensuring interpretability."
        ),
        backstory=(
            "You are a feature engineering expert with a track record of improving "
            "churn model AUC by carefully selecting and crafting features. You understand "
            "multicollinearity, target leakage, and the importance of interpretable "
            "feature sets for business stakeholders. You leverage multiple techniques "
            "including correlation analysis, Random Forest importance, mutual information "
            "for non-linear relationships, polynomial interactions for feature combinations, "
            "hierarchical clustering to reduce redundant features, and recursive feature "
            "elimination to find the optimal feature subset. You ensure the final feature "
            "set is both predictive and business-interpretable."
        ),
        tools=[
            correlation_analysis_tool,
            feature_importance_tool,
            variance_threshold_tool,
            select_top_features_tool,
            generate_interaction_features_tool,      # NEW
            recursive_feature_elimination_tool,      # NEW
            calculate_mutual_information_tool,       # NEW
            cluster_features_tool,                   # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )