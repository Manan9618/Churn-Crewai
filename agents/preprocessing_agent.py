from crewai import Agent
from tools.data.data_tools import (
    load_dataset_tool,
    encode_categoricals_tool,
    handle_missing_values_tool,
    scale_features_tool,
    save_processed_data_tool,
)


def create_preprocessing_agent(llm) -> Agent:
    return Agent(
        role="Data Preprocessing Engineer",
        goal=(
            "Clean and transform the raw Customer Churn dataset into a model-ready "
            "format by handling missing values, encoding categorical variables, "
            "and scaling numerical features."
        ),
        backstory=(
            "You are a seasoned ML engineer who specializes in data wrangling. "
            "You know exactly how to deal with mixed-type telecom datasets, fix "
            "TotalCharges whitespace issues, label-encode binary columns, and "
            "apply appropriate scalers so that no information leaks into the test set."
        ),
        tools=[
            load_dataset_tool,
            handle_missing_values_tool,
            encode_categoricals_tool,
            scale_features_tool,
            save_processed_data_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )