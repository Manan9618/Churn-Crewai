from crewai import Agent
from tools.model.model_validation_tools import (
    validate_predictions_file_tool,
    validate_prediction_columns_tool,
    validate_churn_rate_range_tool,
    validate_probability_range_tool,
)


def create_prediction_validator(llm) -> Agent:
    return Agent(
        role="Prediction Output Validator",
        goal=(
            "Ensure the prediction output file is complete, contains the required columns "
            "(Churn_Predicted, Churn_Probability), has a realistic churn rate, and that "
            "all probability scores fall in [0, 1]."
        ),
        backstory=(
            "You are a data delivery auditor who validates model outputs before they are "
            "handed off to business teams. You check for missing predictions, out-of-range "
            "probabilities, and sanity-check that predicted churn rates are in the expected "
            "10-35% range typical of telecom datasets. You ensure prediction files are "
            "complete, accurate, and ready for business consumption by the CRM and "
            "retention teams."
        ),
        tools=[
            validate_predictions_file_tool,
            validate_prediction_columns_tool,
            validate_probability_range_tool,
            validate_churn_rate_range_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )