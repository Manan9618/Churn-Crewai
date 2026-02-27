from crewai import Agent
from tools.summary.summary_validation_tools import (
    validate_summary_report_exists_tool,
    validate_summary_sections_tool,
    validate_summary_model_metrics_tool,
    validate_summary_prediction_stats_tool,
    validate_summary_retention_tool,
    validate_markdown_report_tool,
)


def create_summary_validator(llm) -> Agent:
    return Agent(
        role="Pipeline Summary Validator",
        goal=(
            "Validate the final pipeline summary report by checking that both JSON and "
            "Markdown files exist, all required sections are present, model metrics meet "
            "thresholds, prediction stats are realistic, the retention section is complete, "
            "and the Markdown report is well-formed."
        ),
        backstory=(
            "You are a final QA checkpoint for the entire churn prediction pipeline. "
            "Nothing leaves the system without your sign-off. You ensure the summary "
            "report is complete, accurate, and ready to be presented to the business. "
            "A missing section or an out-of-range metric means the pipeline must be "
            "revisited before the report can be delivered."
        ),
        tools=[
            validate_summary_report_exists_tool,
            validate_summary_sections_tool,
            validate_summary_model_metrics_tool,
            validate_summary_prediction_stats_tool,
            validate_summary_retention_tool,
            validate_markdown_report_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )