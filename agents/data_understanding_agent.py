from crewai import Agent
from tools.data.data_tools import (
    load_dataset_tool,
    dataset_info_tool,
    missing_values_tool,
    class_distribution_tool,
    descriptive_stats_tool,
    detect_pii_fields_tool,  # NEW
)


def create_data_understanding_agent(llm) -> Agent:
    return Agent(
        role="Data Understanding Specialist",
        goal=(
            "Thoroughly explore and understand the Customer Churn dataset by analyzing "
            "its structure, feature types, missing values, class distribution, statistical "
            "properties, and potential PII fields to ensure compliance and provide a solid "
            "foundation for downstream tasks."
        ),
        backstory=(
            "You are an expert data analyst with deep experience in telecom customer "
            "behavior datasets. You excel at uncovering data quality issues, spotting "
            "imbalances, identifying PII compliance risks, and summarizing key statistics "
            "that guide preprocessing and modeling decisions."
        ),
        tools=[
            load_dataset_tool,
            dataset_info_tool,
            missing_values_tool,
            class_distribution_tool,
            descriptive_stats_tool,
            detect_pii_fields_tool,  # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )