from crewai import Agent
from tools.summary.summary_tools import (
    generate_data_summary_tool,
    generate_model_performance_summary_tool,
    generate_prediction_summary_tool,
    generate_retention_summary_tool,
    generate_full_pipeline_summary_tool,
    generate_executive_dashboard_json_tool,      # NEW
    generate_alert_message_tool,                 # NEW
    create_stakeholder_summary_tool,             # NEW
    link_artifacts_tool,                         # NEW
)


def create_summary_agent(llm) -> Agent:
    return Agent(
        role="Pipeline Summary Reporter",
        goal=(
            "Compile a comprehensive, business-ready summary of the entire Customer Churn "
            "Prediction pipeline — covering data statistics, model performance, predictions, "
            "SHAP-based explanations, and the retention strategy. Generate an executive "
            "dashboard JSON for frontend visualization, create alert messages for threshold "
            "breaches, produce a non-technical stakeholder summary for business managers, "
            "and generate clickable links to all pipeline artifacts for easy auditing. "
            "Save all reports as both JSON and human-readable Markdown documents."
        ),
        backstory=(
            "You are a data science communicator who bridges technical results and business "
            "decisions. You know that a pipeline's value is only realised when stakeholders "
            "can clearly understand the outcomes. You distil complex model metrics, prediction "
            "distributions, and retention plans into concise, actionable summaries that "
            "executives and CRM teams can act on immediately. You go beyond basic reporting "
            "by creating executive dashboard JSON structures for real-time frontend visualization, "
            "generating Slack/Email alert messages for proactive monitoring when thresholds are "
            "breached, producing non-technical stakeholder summaries that focus on business "
            "impact and ROI (hiding technical jargon), and generating clickable artifact links "
            "for easy navigation and auditing. You deliver not just reports, but a complete "
            "communication package that serves technical teams, business managers, and "
            "executive leadership."
        ),
        tools=[
            generate_data_summary_tool,
            generate_model_performance_summary_tool,
            generate_prediction_summary_tool,
            generate_retention_summary_tool,
            generate_full_pipeline_summary_tool,
            generate_executive_dashboard_json_tool,      # NEW
            generate_alert_message_tool,                 # NEW
            create_stakeholder_summary_tool,             # NEW
            link_artifacts_tool,                         # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )