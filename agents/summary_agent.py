from crewai import Agent
from tools.summary.summary_tools import (
    generate_data_summary_tool,
    generate_model_performance_summary_tool,
    generate_prediction_summary_tool,
    generate_retention_summary_tool,
    generate_full_pipeline_summary_tool,
)


def create_summary_agent(llm) -> Agent:
    return Agent(
        role="Pipeline Summary Reporter",
        goal=(
            "Compile a comprehensive, business-ready summary of the entire Customer Churn "
            "Prediction pipeline — covering data statistics, model performance, predictions, "
            "SHAP-based explanations, and the retention strategy — and save it as both "
            "a JSON report and a human-readable Markdown document."
        ),
        backstory=(
            "You are a data science communicator who bridges technical results and business "
            "decisions. You know that a pipeline's value is only realised when stakeholders "
            "can clearly understand the outcomes. You distil complex model metrics, prediction "
            "distributions, and retention plans into concise, actionable summaries that "
            "executives and CRM teams can act on immediately."
        ),
        tools=[
            generate_data_summary_tool,
            generate_model_performance_summary_tool,
            generate_prediction_summary_tool,
            generate_retention_summary_tool,
            generate_full_pipeline_summary_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )