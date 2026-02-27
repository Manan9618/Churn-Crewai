from crewai import Agent
from tools.model.model_tools import (
    train_model_tool,
    cross_validate_tool,
    evaluate_model_tool,
    save_model_tool,
)


def create_model_training_agent(llm) -> Agent:
    return Agent(
        role="Machine Learning Model Trainer",
        goal=(
            "Train and evaluate the best classification model for customer churn "
            "prediction using cross-validation, report key metrics (AUC, F1, Precision, "
            "Recall), and save the trained model artifact."
        ),
        backstory=(
            "You are a machine learning engineer experienced in building production-grade "
            "churn models. You know how to handle class imbalance, select the right "
            "evaluation metrics for business impact, and ensure the model generalises "
            "well beyond the training set."
        ),
        tools=[
            train_model_tool,
            cross_validate_tool,
            evaluate_model_tool,
            save_model_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )