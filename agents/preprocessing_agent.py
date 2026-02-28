from crewai import Agent
from tools.data.data_tools import (
    load_dataset_tool,
    encode_categoricals_tool,
    handle_missing_values_tool,
    scale_features_tool,
    save_processed_data_tool,
    handle_outliers_iqr_tool,       # NEW
    downcast_memory_tool,           # NEW
    detect_duplicate_records_tool,  # NEW
)


def create_preprocessing_agent(llm) -> Agent:
    return Agent(
        role="Data Preprocessing Engineer",
        goal=(
            "Clean and transform the raw Customer Churn dataset into a model-ready "
            "format by handling missing values, removing duplicates, detecting/capping "
            "outliers, encoding categorical variables, scaling numerical features, and "
            "optimizing memory usage."
        ),
        backstory=(
            "You are a seasoned ML engineer who specializes in data wrangling. "
            "You know exactly how to deal with mixed-type telecom datasets, fix "
            "TotalCharges whitespace issues, remove duplicate records, handle outliers "
            "using IQR methods, label-encode binary columns, apply appropriate scalers, "
            "and optimize memory to prevent OOM errors. You ensure no information leaks "
            "into the test set."
        ),
        tools=[
            load_dataset_tool,
            handle_missing_values_tool,
            detect_duplicate_records_tool,  # NEW
            handle_outliers_iqr_tool,       # NEW
            encode_categoricals_tool,
            scale_features_tool,
            downcast_memory_tool,           # NEW
            save_processed_data_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )