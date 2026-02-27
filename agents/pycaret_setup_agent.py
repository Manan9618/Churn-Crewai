from crewai import Agent
from tools.pycaret.pycaret_tools import (
    pycaret_setup_tool,
    pycaret_compare_models_tool,
    pycaret_tune_model_tool,
    pycaret_save_model_tool,
)


def create_pycaret_setup_agent(llm) -> Agent:
    return Agent(
        role="PyCaret AutoML Orchestrator",
        goal=(
            "Initialize the PyCaret classification environment with the preprocessed "
            "churn dataset, compare multiple classifiers, tune the best model, and "
            "persist the final pipeline for downstream prediction."
        ),
        backstory=(
            "You are an AutoML specialist who leverages PyCaret to rapidly benchmark "
            "classification algorithms. You configure experiments carefully — handling "
            "class imbalance via SMOTE, setting the correct metric (AUC / F1), and "
            "ensuring reproducibility through fixed random seeds."
        ),
        tools=[
            pycaret_setup_tool,
            pycaret_compare_models_tool,
            pycaret_tune_model_tool,
            pycaret_save_model_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )