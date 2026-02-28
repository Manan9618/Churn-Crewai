from crewai import Agent
from tools.processing.processing_validation_tools import (
    validate_split_files_exist_tool,
    validate_split_ratio_tool,
    validate_no_data_leakage_tool,
    validate_stratification_tool,
    validate_feature_alignment_tool,
    validate_pipeline_state_tool,
    validate_target_distribution_per_fold_tool,      # NEW
    validate_pipeline_determinism_tool,               # NEW
    validate_memory_footprint_tool,                   # NEW
)


def create_processing_validator(llm) -> Agent:
    return Agent(
        role="Data Processing Quality Validator",
        goal=(
            "Validate the train/test splitting and SMOTE processing outputs by checking: "
            "all split files exist, the split ratio is correct, there is no data leakage "
            "between splits, the class distribution is stratified, feature columns are "
            "aligned, the pipeline state is persisted, each CV fold has representative "
            "churn rate distribution, the pipeline is deterministic (reproducible), and "
            "the memory footprint fits within production limits."
        ),
        backstory=(
            "You are a data pipeline QA specialist who ensures that every split is "
            "statistically sound. You check for data leakage at the row level, verify "
            "that stratification kept the churn rate consistent across splits, and "
            "confirm the pipeline state file accurately reflects what was generated. "
            "You also validate that cross-validation folds have representative target "
            "distributions (churn rate variance ≤ 5%), that the pipeline is deterministic "
            "with proper random seeds and recorded file hashes for reproducibility, and "
            "that the memory footprint of all artifacts fits within production container "
            "limits (≤16GB with 20% safety margin). You ensure the data pipeline is not "
            "just functional, but also production-ready, reproducible, and resource-efficient."
        ),
        tools=[
            validate_split_files_exist_tool,
            validate_split_ratio_tool,
            validate_no_data_leakage_tool,
            validate_stratification_tool,
            validate_feature_alignment_tool,
            validate_pipeline_state_tool,
            validate_target_distribution_per_fold_tool,      # NEW
            validate_pipeline_determinism_tool,               # NEW
            validate_memory_footprint_tool,                   # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )