from crewai import Agent
from tools.model.model_tools import (
    train_model_tool,
    cross_validate_tool,
    evaluate_model_tool,
    save_model_tool,
    load_model_tool,
    predict_tool,
    save_predictions_tool,
    tune_classification_threshold_tool,      # NEW
    calibrate_model_probabilities_tool,      # NEW
    create_model_ensemble_tool,              # NEW
    export_model_onnx_tool,                  # NEW
    measure_inference_latency_tool,          # NEW
)


def create_model_training_agent(llm) -> Agent:
    return Agent(
        role="Machine Learning Model Trainer",
        goal=(
            "Train and evaluate the best classification model for customer churn "
            "prediction using cross-validation, report key metrics (AUC, F1, Precision, "
            "Recall), optimize the classification threshold, calibrate probabilities, "
            "create ensemble models, export to ONNX format, measure inference latency, "
            "and save all trained model artifacts for production deployment."
        ),
        backstory=(
            "You are a machine learning engineer experienced in building production-grade "
            "churn models. You know how to handle class imbalance, select the right "
            "evaluation metrics for business impact, and ensure the model generalises "
            "well beyond the training set. You also optimize models for production by "
            "tuning classification thresholds (not just 0.5), calibrating probabilities "
            "for accurate risk scores, creating ensembles to reduce variance, exporting "
            "to ONNX for cross-platform deployment, and measuring inference latency to "
            "ensure SLA compliance (<100ms P95). You deliver models that are not just "
            "accurate, but also production-ready and business-actionable."
        ),
        tools=[
            train_model_tool,
            cross_validate_tool,
            evaluate_model_tool,
            save_model_tool,
            load_model_tool,
            predict_tool,
            save_predictions_tool,
            tune_classification_threshold_tool,      # NEW
            calibrate_model_probabilities_tool,      # NEW
            create_model_ensemble_tool,              # NEW
            export_model_onnx_tool,                  # NEW
            measure_inference_latency_tool,          # NEW
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )