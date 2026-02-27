from crewai import Agent
from tools.feature.feature_tools import (
    correlation_analysis_tool,
    feature_importance_tool,
    variance_threshold_tool,
    select_top_features_tool,
)


def create_feature_discovery_agent(llm) -> Agent:
    return Agent(
        role="Feature Discovery Analyst",
        goal=(
            "Identify the most predictive features for customer churn by performing "
            "correlation analysis, computing feature importances, and selecting an "
            "optimal subset that maximizes model performance while reducing noise."
        ),
        backstory=(
            "You are a feature engineering expert with a track record of improving "
            "churn model AUC by carefully selecting and crafting features. You understand "
            "multicollinearity, target leakage, and the importance of interpretable "
            "feature sets for business stakeholders."
        ),
        tools=[
            correlation_analysis_tool,
            feature_importance_tool,
            variance_threshold_tool,
            select_top_features_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )