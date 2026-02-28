from crewai import Agent
from tools.summary.summary_validation_tools import (
    validate_summary_report_exists_tool,
    validate_summary_sections_tool,
    validate_summary_model_metrics_tool,
    validate_summary_prediction_stats_tool,
    validate_summary_retention_tool,
    validate_markdown_report_tool,
    validate_link_integrity_tool,              # NEW
    validate_data_freshness_tool,              # NEW
    validate_sensitive_data_redaction_tool,    # NEW
)


def create_summary_validator(llm) -> Agent:
    return Agent(
        role="Pipeline Summary Validator",
        goal=(
            "Validate the final pipeline summary report by checking that both JSON and "
            "Markdown files exist, all required sections are present, model metrics meet "
            "thresholds, prediction stats are realistic, the retention section is complete, "
            "and the Markdown report is well-formed. Also verify that all hyperlinks resolve "
            "correctly, the summary data is fresh (not stale), and no sensitive PII data "
            "has leaked into the summary documents."
        ),
        backstory=(
            "You are a final QA checkpoint for the entire churn prediction pipeline. "
            "Nothing leaves the system without your sign-off. You ensure the summary "
            "report is complete, accurate, and ready to be presented to the business. "
            "A missing section or an out-of-range metric means the pipeline must be "
            "revisited before the report can be delivered. You also validate that all "
            "hyperlinks in the report resolve correctly (no broken links to artifacts), "
            "that the summary is generated from fresh data (within 24 hours, not stale), "
            "and that no sensitive PII data (emails, phone numbers, SSNs, credit cards) "
            "has leaked into the summary documents. You provide a final PIPELINE COMPLETE "
            "or PIPELINE INCOMPLETE verdict based on all 9 validation checks."
        ),
        tools=[
            validate_summary_report_exists_tool,
            validate_summary_sections_tool,
            validate_summary_model_metrics_tool,
            validate_summary_prediction_stats_tool,
            validate_summary_retention_tool,
            validate_markdown_report_tool,
            validate_link_integrity_tool,              # NEW
            validate_data_freshness_tool,              # NEW
            validate_sensitive_data_redaction_tool,    # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )