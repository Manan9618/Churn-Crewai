from crewai import Agent
from tools.explanation.explanation_tools import (
    shap_summary_tool,
    shap_force_plot_tool,
    lime_explanation_tool,
    generate_explanation_report_tool,
)


def create_explanation_agent(llm) -> Agent:
    return Agent(
        role="Model Explainability Expert",
        goal=(
            "Explain the churn model's predictions using SHAP and LIME so that "
            "business stakeholders understand which features drive individual and "
            "global churn decisions."
        ),
        backstory=(
            "You are an AI explainability consultant who translates black-box model "
            "outputs into clear, actionable narratives. You use SHAP values for global "
            "feature importance and LIME for local instance explanations, and you "
            "produce reports that non-technical audiences can understand."
        ),
        tools=[
            shap_summary_tool,
            shap_force_plot_tool,
            lime_explanation_tool,
            generate_explanation_report_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )