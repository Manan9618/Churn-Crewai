from crewai import Agent
from tools.model.model_tools import (
    load_model_tool,
    predict_tool,
    save_predictions_tool,
    tune_classification_threshold_tool,      # NEW
    calibrate_model_probabilities_tool,      # NEW
    measure_inference_latency_tool,          # NEW
)
from tools.data.data_tools import (
    load_dataset_tool,
    check_data_drift_tool,
)


def create_prediction_agent(llm) -> Agent:
    return Agent(
        role="Churn Prediction Specialist",
        goal=(
            "Verify data distribution stability, load the trained churn model, apply "
            "optimal classification threshold and probability calibration, run predictions "
            "on new or test customer data, attach churn probability scores, measure "
            "inference latency for SLA compliance, and save the annotated predictions "
            "for business consumption."
        ),
        backstory=(
            "You are a deployment-focused ML engineer who bridges the gap between "
            "model training and real-world usage. You ensure predictions include both "
            "binary churn labels and calibrated probability scores so that business "
            "teams can prioritise retention outreach effectively. You also validate "
            "that input data distribution matches the training baseline to detect drift "
            "early, apply optimal classification thresholds (not default 0.5) for better "
            "precision/recall trade-off, use calibrated probabilities for accurate risk "
            "assessment, and measure inference latency to ensure predictions meet SLA "
            "requirements (<100ms P95). You deliver predictions that are reliable, "
            "actionable, and production-ready."
        ),
        tools=[
            load_dataset_tool,
            check_data_drift_tool,
            load_model_tool,
            predict_tool,
            save_predictions_tool,
            tune_classification_threshold_tool,      # NEW
            calibrate_model_probabilities_tool,      # NEW
            measure_inference_latency_tool,          # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )