from crewai import Agent
from tools.explanation.explanation_tools import (
    shap_summary_tool,
    shap_force_plot_tool,
    lime_explanation_tool,
    generate_explanation_report_tool,
    generate_counterfactuals_tool,      # NEW
    check_explanation_stability_tool,   # NEW
    analyze_fairness_bias_tool,         # NEW
    extract_global_logic_rules_tool,    # NEW
)


def create_explanation_agent(llm) -> Agent:
    return Agent(
        role="Model Explainability Expert",
        goal=(
            "Explain the churn model's predictions using SHAP, LIME, counterfactuals, "
            "fairness analysis, and logic rules so that business stakeholders understand "
            "which features drive individual and global churn decisions, verify explanation "
            "reliability, detect potential bias, and extract actionable business rules."
        ),
        backstory=(
            "You are an AI explainability consultant who translates black-box model "
            "outputs into clear, actionable narratives. You use SHAP values for global "
            "feature importance, LIME for local instance explanations, counterfactuals "
            "for what-if analysis, stability checks for explanation reliability, fairness "
            "analysis to detect bias against protected groups, and logic rule extraction "
            "to produce business-friendly If-Then rules that non-technical audiences can "
            "understand and act upon."
        ),
        tools=[
            shap_summary_tool,
            shap_force_plot_tool,
            lime_explanation_tool,
            generate_explanation_report_tool,
            generate_counterfactuals_tool,      # NEW
            check_explanation_stability_tool,   # NEW
            analyze_fairness_bias_tool,         # NEW
            extract_global_logic_rules_tool,    # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )