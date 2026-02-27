from crewai import Task
from prompts.prompts import PROMPTS


def create_tasks(agents: dict) -> list:
    """
    Create and return all CrewAI tasks in pipeline order.
    Pipeline: data_understanding → preprocessing → feature_discovery
              → processing → pycaret_setup → model_training
              → prediction → explanation → retention_strategy
              → summary → feedback_loop
    Each stage has an agent task + validator task.
    """

    # ── 1. Data Understanding ──────────────────────────────────────────────────
    task_data_understanding = Task(
        description=PROMPTS["data_understanding"]["task"],
        expected_output=PROMPTS["data_understanding"]["expected_output"],
        agent=agents["data_understanding"],
    )
    task_data_validation = Task(
        description=PROMPTS["data_validation"]["task"],
        expected_output=PROMPTS["data_validation"]["expected_output"],
        agent=agents["data_validator"],
        context=[task_data_understanding],
    )

    # ── 2. Preprocessing ───────────────────────────────────────────────────────
    task_preprocessing = Task(
        description=PROMPTS["preprocessing"]["task"],
        expected_output=PROMPTS["preprocessing"]["expected_output"],
        agent=agents["preprocessing"],
        context=[task_data_validation],
    )
    task_preprocessing_validation = Task(
        description=PROMPTS["preprocessing_validation"]["task"],
        expected_output=PROMPTS["preprocessing_validation"]["expected_output"],
        agent=agents["preprocessing_validator"],
        context=[task_preprocessing],
    )

    # ── 3. Feature Discovery ───────────────────────────────────────────────────
    task_feature_discovery = Task(
        description=PROMPTS["feature_discovery"]["task"],
        expected_output=PROMPTS["feature_discovery"]["expected_output"],
        agent=agents["feature_discovery"],
        context=[task_preprocessing_validation],
    )
    task_feature_validation = Task(
        description=PROMPTS["feature_validation"]["task"],
        expected_output=PROMPTS["feature_validation"]["expected_output"],
        agent=agents["feature_validator"],
        context=[task_feature_discovery],
    )

    # ── 4. Processing (Train/Test Split + SMOTE) ───────────────────────────────
    task_processing = Task(
        description=PROMPTS["processing"]["task"],
        expected_output=PROMPTS["processing"]["expected_output"],
        agent=agents["processing"],
        context=[task_feature_validation],
    )
    task_processing_validation = Task(
        description=PROMPTS["processing_validation"]["task"],
        expected_output=PROMPTS["processing_validation"]["expected_output"],
        agent=agents["processing_validator"],
        context=[task_processing],
    )

    # ── 5. PyCaret Setup ───────────────────────────────────────────────────────
    task_pycaret_setup = Task(
        description=PROMPTS["pycaret_setup"]["task"],
        expected_output=PROMPTS["pycaret_setup"]["expected_output"],
        agent=agents["pycaret_setup"],
        context=[task_processing_validation],
    )
    task_pycaret_validation = Task(
        description=PROMPTS["pycaret_validation"]["task"],
        expected_output=PROMPTS["pycaret_validation"]["expected_output"],
        agent=agents["pycaret_validator"],
        context=[task_pycaret_setup],
    )

    # ── 6. Model Training ─────────────────────────────────────────────────────
    task_model_training = Task(
        description=PROMPTS["model_training"]["task"],
        expected_output=PROMPTS["model_training"]["expected_output"],
        agent=agents["model_training"],
        context=[task_pycaret_validation],
    )
    task_model_validation = Task(
        description=PROMPTS["model_validation"]["task"],
        expected_output=PROMPTS["model_validation"]["expected_output"],
        agent=agents["model_validator"],
        context=[task_model_training],
    )

    # ── 7. Prediction ─────────────────────────────────────────────────────────
    task_prediction = Task(
        description=PROMPTS["prediction"]["task"],
        expected_output=PROMPTS["prediction"]["expected_output"],
        agent=agents["prediction"],
        context=[task_model_validation],
    )
    task_prediction_validation = Task(
        description=PROMPTS["prediction_validation"]["task"],
        expected_output=PROMPTS["prediction_validation"]["expected_output"],
        agent=agents["prediction_validator"],
        context=[task_prediction],
    )

    # ── 8. Explanation ────────────────────────────────────────────────────────
    task_explanation = Task(
        description=PROMPTS["explanation"]["task"],
        expected_output=PROMPTS["explanation"]["expected_output"],
        agent=agents["explanation"],
        context=[task_prediction_validation],
    )
    task_explanation_validation = Task(
        description=PROMPTS["explanation_validation"]["task"],
        expected_output=PROMPTS["explanation_validation"]["expected_output"],
        agent=agents["explanation_validator"],
        context=[task_explanation],
    )

    # ── 9. Retention Strategy ─────────────────────────────────────────────────
    task_retention_strategy = Task(
        description=PROMPTS["retention_strategy"]["task"],
        expected_output=PROMPTS["retention_strategy"]["expected_output"],
        agent=agents["retention_strategy"],
        context=[task_explanation_validation],
    )
    task_retention_validation = Task(
        description=PROMPTS["retention_validation"]["task"],
        expected_output=PROMPTS["retention_validation"]["expected_output"],
        agent=agents["retention_validator"],
        context=[task_retention_strategy],
    )

    # ── 10. Summary ───────────────────────────────────────────────────────────
    task_summary = Task(
        description=PROMPTS["summary"]["task"],
        expected_output=PROMPTS["summary"]["expected_output"],
        agent=agents["summary"],
        context=[task_retention_validation],
    )
    task_summary_validation = Task(
        description=PROMPTS["summary_validation"]["task"],
        expected_output=PROMPTS["summary_validation"]["expected_output"],
        agent=agents["summary_validator"],
        context=[task_summary],
    )

    # ── 11. Feedback Loop ──────────────────────────────────────────────────────
    task_feedback_loop = Task(
        description=PROMPTS["feedback_loop"]["task"],
        expected_output=PROMPTS["feedback_loop"]["expected_output"],
        agent=agents["feedback_loop"],
        context=[task_summary_validation],
    )
    task_feedback_validation = Task(
        description=PROMPTS["feedback_validation"]["task"],
        expected_output=PROMPTS["feedback_validation"]["expected_output"],
        agent=agents["feedback_validator"],
        context=[task_feedback_loop],
    )

    return [
        task_data_understanding,    task_data_validation,
        task_preprocessing,         task_preprocessing_validation,
        task_feature_discovery,     task_feature_validation,
        task_processing,            task_processing_validation,
        task_pycaret_setup,         task_pycaret_validation,
        task_model_training,        task_model_validation,
        task_prediction,            task_prediction_validation,
        task_explanation,           task_explanation_validation,
        task_retention_strategy,    task_retention_validation,
        task_summary,               task_summary_validation,
        task_feedback_loop,         task_feedback_validation,
    ]