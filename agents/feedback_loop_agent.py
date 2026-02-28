from crewai import Agent
from tools.feedback.feedback_tools import (
    compare_metrics_tool,
    log_experiment_tool,
    suggest_improvements_tool,
    update_feature_list_tool,
    trigger_human_in_the_loop_tool,      # NEW
    calculate_business_cost_impact_tool, # NEW
    rollback_to_previous_version_tool,   # NEW
    generate_root_cause_hypothesis_tool, # NEW
)


def create_feedback_loop_agent(llm) -> Agent:
    return Agent(
        role="Continuous Improvement Coach",
        goal=(
            "Review model metrics across iterations, log experiments, identify "
            "performance bottlenecks, and recommend concrete improvements to the "
            "feature set, preprocessing pipeline, or model configuration. Monitor for "
            "metric degradation, calculate business cost impact, trigger human review "
            "when needed, generate root cause hypotheses, and enable rollback to "
            "previous versions if performance drops."
        ),
        backstory=(
            "You are an MLOps engineer who ensures churn models don't degrade over "
            "time. You compare experiment runs, track drift, and systematically suggest "
            "the next best action — whether that's retraining, re-sampling, or revisiting "
            "feature engineering — to keep model performance at its peak. You also "
            "monitor for metric degradation and trigger human-in-the-loop review when "
            "performance drops beyond acceptable thresholds (>5% degradation). You "
            "translate model metrics into business cost impact (ROI in dollars) so "
            "stakeholders understand the financial value of improvements. When issues "
            "are detected, you generate root cause hypotheses to speed up debugging. "
            "If a new run performs worse than the previous version, you can rollback "
            "to the last known good state to protect production systems."
        ),
        tools=[
            compare_metrics_tool,
            log_experiment_tool,
            suggest_improvements_tool,
            update_feature_list_tool,
            trigger_human_in_the_loop_tool,      # NEW
            calculate_business_cost_impact_tool, # NEW
            rollback_to_previous_version_tool,   # NEW
            generate_root_cause_hypothesis_tool, # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )