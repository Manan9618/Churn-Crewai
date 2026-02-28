from crewai import Agent
from tools.explanation.explanation_validation_tools import (
    validate_shap_values_tool,
    validate_explanation_report_tool,
    validate_top_features_in_shap_tool,
    validate_consistency_shap_lime_tool,      # NEW
    validate_fairness_metrics_tool,           # NEW
    validate_counterfactual_feasibility_tool, # NEW
)


def create_explanation_validator(llm) -> Agent:
    return Agent(
        role="Explainability Output Validator",
        goal=(
            "Verify that SHAP/LIME explanations were generated successfully, that the "
            "explanation report is coherent, that known important churn drivers appear "
            "among the top SHAP features, that SHAP and LIME explanations are consistent, "
            "that fairness metrics show no significant bias against protected groups, and "
            "that counterfactual suggestions are realistic and actionable."
        ),
        backstory=(
            "You are an AI ethics and transparency auditor. You ensure that model "
            "explanations are technically valid (correct SHAP shapes, no NaN values) and "
            "align with domain knowledge — e.g., Contract type should always rank highly "
            "for a telecom churn model. You also verify that multiple explanation methods "
            "(SHAP and LIME) agree on top features to ensure explanation stability, check "
            "for fairness bias across protected groups (Gender, SeniorCitizen), and validate "
            "that counterfactual recommendations are realistic and actionable for customers "
            "(e.g., don't suggest changing immutable attributes like age or gender)."
        ),
        tools=[
            validate_shap_values_tool,
            validate_top_features_in_shap_tool,
            validate_explanation_report_tool,
            validate_consistency_shap_lime_tool,      # NEW
            validate_fairness_metrics_tool,           # NEW
            validate_counterfactual_feasibility_tool, # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )