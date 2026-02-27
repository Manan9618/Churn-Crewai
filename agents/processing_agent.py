from crewai import Agent
from tools.processing.processing_tools import (
    train_test_split_tool,
    apply_smote_tool,
    prepare_feature_matrix_tool,
    save_pipeline_state_tool,
    load_pipeline_state_tool,
)


def create_processing_agent(llm) -> Agent:
    return Agent(
        role="ML Data Processing Engineer",
        goal=(
            "Create a clean, stratified train/test split from the preprocessed churn "
            "dataset, apply SMOTE oversampling to the training set to address class "
            "imbalance, verify feature matrix alignment, and persist the pipeline state."
        ),
        backstory=(
            "You are an ML pipeline engineer who specialises in reliable data splitting "
            "and sampling strategies. You know that improper splits cause data leakage "
            "and that SMOTE must only be applied to the training set — never the test set. "
            "You ensure every pipeline run is reproducible by saving its state to disk."
        ),
        tools=[
            train_test_split_tool,
            apply_smote_tool,
            prepare_feature_matrix_tool,
            save_pipeline_state_tool,
            load_pipeline_state_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )