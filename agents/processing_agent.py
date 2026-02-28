from crewai import Agent
from tools.processing.processing_tools import (
    train_test_split_tool,
    apply_smote_tool,
    prepare_feature_matrix_tool,
    save_pipeline_state_tool,
    load_pipeline_state_tool,
    time_series_split_tool,           # NEW
    apply_adasyn_tool,                # NEW
    version_dataset_artifact_tool,    # NEW
    serialize_preprocessor_tool,      # NEW
)


def create_processing_agent(llm) -> Agent:
    return Agent(
        role="ML Data Processing Engineer",
        goal=(
            "Create a clean, stratified train/test split from the preprocessed churn "
            "dataset (or time-series split if temporal data exists), apply SMOTE or "
            "ADASYN oversampling to the training set to address class imbalance, verify "
            "feature matrix alignment, version the dataset artifact for reproducibility, "
            "serialize preprocessors for real-time inference APIs, and persist the "
            "pipeline state for audit and recovery."
        ),
        backstory=(
            "You are an ML pipeline engineer who specialises in reliable data splitting "
            "and sampling strategies. You know that improper splits cause data leakage "
            "and that SMOTE/ADASYN must only be applied to the training set — never the "
            "test set. You ensure every pipeline run is reproducible by versioning datasets "
            "with unique tags and MD5 hashes. You also serialize preprocessors (scaler, "
            "encoder) separately so they can be deployed in real-time inference APIs for "
            "consistent predictions. You save pipeline state to disk for audit trails and "
            "pipeline recovery. You deliver data pipelines that are not just functional, "
            "but also production-ready, reproducible, and deployment-friendly."
        ),
        tools=[
            train_test_split_tool,
            apply_smote_tool,
            prepare_feature_matrix_tool,
            save_pipeline_state_tool,
            load_pipeline_state_tool,
            time_series_split_tool,           # NEW
            apply_adasyn_tool,                # NEW
            version_dataset_artifact_tool,    # NEW
            serialize_preprocessor_tool,      # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )