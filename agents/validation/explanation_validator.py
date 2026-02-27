from crewai import Agent
from tools.explanation.explanation_validation_tools import (
    validate_shap_values_tool,
    validate_explanation_report_tool,
    validate_top_features_in_shap_tool,
)


def create_explanation_validator(llm) -> Agent:
    return Agent(
        role="Explainability Output Validator",
        goal=(
            "Verify that SHAP/LIME explanations were generated successfully, that the "
            "explanation report is coherent, and that known important churn drivers "
            "(Contract type, MonthlyCharges, tenure) appear among the top SHAP features."
        ),
        backstory=(
            "You are an AI ethics and transparency auditor. You ensure that model "
            "explanations are technically valid (correct SHAP shapes, no NaN values) and "
            "align with domain knowledge — e.g., Contract type should always rank highly "
            "for a telecom churn model."
        ),
        tools=[
            validate_shap_values_tool,
            validate_top_features_in_shap_tool,
            validate_explanation_report_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )