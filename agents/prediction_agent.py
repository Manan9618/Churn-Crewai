from crewai import Agent
from tools.model.model_tools import load_model_tool
from tools.data.data_tools import load_dataset_tool
from tools.model.model_tools import predict_tool, save_predictions_tool


def create_prediction_agent(llm) -> Agent:
    return Agent(
        role="Churn Prediction Specialist",
        goal=(
            "Load the trained churn model, run predictions on new or test customer "
            "data, attach churn probability scores, and save the annotated predictions "
            "for business consumption."
        ),
        backstory=(
            "You are a deployment-focused ML engineer who bridges the gap between "
            "model training and real-world usage. You ensure predictions include both "
            "binary churn labels and calibrated probability scores so that business "
            "teams can prioritise retention outreach effectively."
        ),
        tools=[
            load_dataset_tool,
            load_model_tool,
            predict_tool,
            save_predictions_tool,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )