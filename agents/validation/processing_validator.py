from crewai import Agent
from tools.processing.processing_validation_tools import (
    validate_split_files_exist_tool,
    validate_split_ratio_tool,
    validate_no_data_leakage_tool,
    validate_stratification_tool,
    validate_feature_alignment_tool,
    validate_pipeline_state_tool,
)


def create_processing_validator(llm) -> Agent:
    return Agent(
        role="Data Processing Quality Validator",
        goal=(
            "Validate the train/test splitting and SMOTE processing outputs by checking: "
            "all split files exist, the split ratio is correct, there is no data leakage "
            "between splits, the class distribution is stratified, feature columns are "
            "aligned, and the pipeline state is persisted."
        ),
        backstory=(
            "You are a data pipeline QA specialist who ensures that every split is "
            "statistically sound. You check for data leakage at the row level, verify "
            "that stratification kept the churn rate consistent across splits, and "
            "confirm the pipeline state file accurately reflects what was generated."
        ),
        tools=[
            validate_split_files_exist_tool,
            validate_split_ratio_tool,
            validate_no_data_leakage_tool,
            validate_stratification_tool,
            validate_feature_alignment_tool,
            validate_pipeline_state_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )