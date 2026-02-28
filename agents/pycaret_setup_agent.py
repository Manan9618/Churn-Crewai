from crewai import Agent
from tools.pycaret.pycaret_tools import (
    pycaret_setup_tool,
    pycaret_compare_models_tool,
    pycaret_tune_model_tool,
    pycaret_save_model_tool,
    blend_top_models_tool,           # NEW
    stack_models_tool,               # NEW
    deploy_model_api_tool,           # NEW
    get_model_leaderboard_tool,      # NEW
)


def create_pycaret_setup_agent(llm) -> Agent:
    return Agent(
        role="PyCaret AutoML Orchestrator",
        goal=(
            "Initialize the PyCaret classification environment with the preprocessed "
            "churn dataset, compare multiple classifiers, tune the best model, create "
            "blended and stacked ensembles for improved performance, deploy a prediction "
            "API for immediate testing, retrieve the full model leaderboard for analysis, "
            "and persist the final pipeline for downstream prediction."
        ),
        backstory=(
            "You are an AutoML specialist who leverages PyCaret to rapidly benchmark "
            "classification algorithms. You configure experiments carefully — handling "
            "class imbalance via SMOTE, setting the correct metric (AUC / F1), and "
            "ensuring reproducibility through fixed random seeds. You go beyond basic "
            "model selection by creating blended ensembles (combining top 5 models) and "
            "stacked ensembles (meta-learning with base + meta model) to maximize predictive "
            "performance. You also deploy models as FastAPI endpoints for immediate real-time "
            "testing and retrieve complete model leaderboards for comprehensive analysis. "
            "You deliver not just a single model, but a complete production-ready ML pipeline "
            "with ensembles, APIs, and full model comparison reports."
        ),
        tools=[
            pycaret_setup_tool,
            pycaret_compare_models_tool,
            pycaret_tune_model_tool,
            pycaret_save_model_tool,
            blend_top_models_tool,           # NEW
            stack_models_tool,               # NEW
            deploy_model_api_tool,           # NEW
            get_model_leaderboard_tool,      # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )