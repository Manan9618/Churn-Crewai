from crewai import Agent
from tools.data.data_validation_tools import (
    validate_no_missing_after_preprocessing_tool,
    validate_encoding_tool,
    validate_scaling_tool,
    validate_train_test_split_tool,
    validate_constant_features_tool,   # NEW
    validate_pii_removal_tool,         # NEW
)


def create_preprocessing_validator(llm) -> Agent:
    return Agent(
        role="Preprocessing Quality Validator",
        goal=(
            "Confirm that the preprocessing pipeline has correctly handled all missing "
            "values, removed PII fields, eliminated constant features, applied proper "
            "encoding to categorical variables, scaled numerical features, and produced "
            "a clean train/test split with no data leakage."
        ),
        backstory=(
            "You are a meticulous QA engineer for ML pipelines. You check that after "
            "preprocessing: no NaNs remain, all PII fields are removed, no constant "
            "features exist (single unique value), all categorical columns are numeric, "
            "feature ranges are standardised, and the target distribution is maintained "
            "across train and test splits. You ensure the data is production-ready."
        ),
        tools=[
            validate_no_missing_after_preprocessing_tool,
            validate_encoding_tool,
            validate_scaling_tool,
            validate_train_test_split_tool,
            validate_constant_features_tool,   # NEW
            validate_pii_removal_tool,         # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )