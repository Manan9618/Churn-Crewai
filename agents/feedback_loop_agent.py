from crewai import Agent
from tools.feedback.feedback_tools import (
    compare_metrics_tool,
    log_experiment_tool,
    suggest_improvements_tool,
    update_feature_list_tool,
)


def create_feedback_loop_agent(llm) -> Agent:
    return Agent(
        role="Continuous Improvement Coach",
        goal=(
            "Review model metrics across iterations, log experiments, identify "
            "performance bottlenecks, and recommend concrete improvements to the "
            "feature set, preprocessing pipeline, or model configuration."
        ),
        backstory=(
            "You are an MLOps engineer who ensures churn models don't degrade over "
            "time. You compare experiment runs, track drift, and systematically suggest "
            "the next best action — whether that's retraining, re-sampling, or revisiting "
            "feature engineering — to keep model performance at its peak."
        ),
        tools=[
            compare_metrics_tool,
            log_experiment_tool,
            suggest_improvements_tool,
            update_feature_list_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )