from crewai import Agent
from tools.pycaret.pycaret_validation_tools import (
    validate_pycaret_setup_tool,
    validate_compare_models_output_tool,
    validate_best_model_saved_tool,
    validate_deployment_health_tool,           # NEW
    validate_inference_consistency_tool,       # NEW
)


def create_pycaret_validator(llm) -> Agent:
    return Agent(
        role="PyCaret Experiment Validator",
        goal=(
            "Verify that the PyCaret experiment was set up correctly, that model "
            "comparison produced a valid leaderboard, and that the best model "
            "was saved with acceptable AUC and F1 scores. Also validate that any "
            "deployed API is healthy and responding, and that PyCaret predictions "
            "are consistent with underlying sklearn model outputs."
        ),
        backstory=(
            "You are an AutoML quality auditor who ensures PyCaret pipelines are "
            "reproducible and produce models meeting minimum performance thresholds "
            "(AUC ≥ 0.75, F1 ≥ 0.55) before they are passed to downstream tasks. "
            "You also validate that deployed model APIs are healthy (returning 200 OK "
            "on /health and /predict endpoints) and that PyCaret's wrapper layer produces "
            "predictions consistent with the underlying sklearn model (tolerance ≤ 0.0001). "
            "You ensure the PyCaret pipeline is not just accurate, but also deployment-ready, "
            "operationally healthy, and logically consistent. You provide a final "
            "PYCARET PIPELINE VALIDATED or PYCARET PIPELINE FAILED verdict."
        ),
        tools=[
            validate_pycaret_setup_tool,
            validate_compare_models_output_tool,
            validate_best_model_saved_tool,
            validate_deployment_health_tool,           # NEW
            validate_inference_consistency_tool,       # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )