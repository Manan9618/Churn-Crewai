from crewai import Agent
from tools.pycaret.pycaret_validation_tools import (
    validate_pycaret_setup_tool,
    validate_compare_models_output_tool,
    validate_best_model_saved_tool,
)


def create_pycaret_validator(llm) -> Agent:
    return Agent(
        role="PyCaret Experiment Validator",
        goal=(
            "Verify that the PyCaret experiment was set up correctly, that model "
            "comparison produced a valid leaderboard, and that the best model "
            "was saved with acceptable AUC and F1 scores."
        ),
        backstory=(
            "You are an AutoML quality auditor who ensures PyCaret pipelines are "
            "reproducible and produce models meeting minimum performance thresholds "
            "(AUC ≥ 0.75, F1 ≥ 0.55) before they are passed to downstream tasks."
        ),
        tools=[
            validate_pycaret_setup_tool,
            validate_compare_models_output_tool,
            validate_best_model_saved_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )