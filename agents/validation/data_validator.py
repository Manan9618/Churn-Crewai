from crewai import Agent
from tools.data.data_validation_tools import (
    validate_schema_tool,
    validate_missing_threshold_tool,
    validate_target_column_tool,
    validate_data_types_tool,
    validate_cardinality_limits_tool,      # NEW
    validate_distribution_stability_tool,  # NEW
)


def create_data_validator(llm) -> Agent:
    return Agent(
        role="Data Quality Validator",
        goal=(
            "Validate the raw Customer Churn dataset against expected schema, data types, "
            "missing value thresholds, target column integrity, categorical cardinality limits, "
            "and distribution stability before any preprocessing begins. Raise clear errors "
            "if validation fails."
        ),
        backstory=(
            "You are a data governance specialist who acts as the first line of defence "
            "against bad data. You enforce strict contracts: the dataset must have all 21 "
            "expected columns, TotalCharges must be coercible to float, Churn must only "
            "contain 'Yes'/'No', missing values must not exceed 1% per column, categorical "
            "columns must not have excessive unique values (prevents one-hot explosion), "
            "and the data distribution should match historical baselines."
        ),
        tools=[
            validate_schema_tool,
            validate_data_types_tool,
            validate_missing_threshold_tool,
            validate_target_column_tool,
            validate_cardinality_limits_tool,      # NEW
            validate_distribution_stability_tool,  # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )