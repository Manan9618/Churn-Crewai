from crewai import Crew, Process
from langchain_openai import ChatOpenAI

# ── Core Pipeline Agents ───────────────────────────────────────────────────────
from agents.data_understanding_agent import create_data_understanding_agent
from agents.preprocessing_agent import create_preprocessing_agent
from agents.feature_discovery_agent import create_feature_discovery_agent
from agents.processing_agent import create_processing_agent
from agents.pycaret_setup_agent import create_pycaret_setup_agent
from agents.model_training_agent import create_model_training_agent
from agents.prediction_agent import create_prediction_agent
from agents.explanation_agent import create_explanation_agent
from agents.retention_strategy_agent import create_retention_strategy_agent
from agents.summary_agent import create_summary_agent
from agents.feedback_loop_agent import create_feedback_loop_agent

# ── Validator Agents ───────────────────────────────────────────────────────────
from agents.validation.data_validator import create_data_validator
from agents.validation.preprocessing_validator import create_preprocessing_validator
from agents.validation.feature_discovery_validator import create_feature_discovery_validator
from agents.validation.processing_validator import create_processing_validator
from agents.validation.pycaret_validator import create_pycaret_validator
from agents.validation.model_training_validator import create_model_training_validator
from agents.validation.prediction_validator import create_prediction_validator
from agents.validation.explanation_validator import create_explanation_validator
from agents.validation.retention_strategy_validator import create_retention_strategy_validator
from agents.validation.summary_validator import create_summary_validator
from agents.validation.feedback_loop_validator import create_feedback_loop_validator

# ── Tasks ─────────────────────────────────────────────────────────────────────
from tasks.tasks import create_tasks


def build_crew(model_name: str = "gpt-4o-mini", temperature: float = 0.3) -> Crew:
    """
    Assemble the full Customer Churn Prediction CrewAI pipeline.

    Pipeline stages (each with agent + validator):
      1.  Data Understanding
      2.  Preprocessing
      3.  Feature Discovery
      4.  Processing (Train/Test Split + SMOTE)
      5.  PyCaret AutoML Setup
      6.  Model Training
      7.  Prediction
      8.  Explanation (SHAP + LIME)
      9.  Retention Strategy
      10. Summary Report
      11. Feedback Loop

    Key fixes applied:
      - max_iter=25     : Gives agents enough iterations to call ALL tools
                          (default was 5-10 which caused agents to stop early)
      - max_retry_limit=3 : Retries failed tool calls up to 3 times
      - max_rpm=10      : Rate limit guard to prevent silent OpenAI failures
    """
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    agents = {
        # ── Core pipeline agents ──────────────────────────────────────────────
        "data_understanding" : create_data_understanding_agent(llm),
        "preprocessing"      : create_preprocessing_agent(llm),
        "feature_discovery"  : create_feature_discovery_agent(llm),
        "processing"         : create_processing_agent(llm),
        "pycaret_setup"      : create_pycaret_setup_agent(llm),
        "model_training"     : create_model_training_agent(llm),
        "prediction"         : create_prediction_agent(llm),
        "explanation"        : create_explanation_agent(llm),
        "retention_strategy" : create_retention_strategy_agent(llm),
        "summary"            : create_summary_agent(llm),
        "feedback_loop"      : create_feedback_loop_agent(llm),

        # ── Validator agents ──────────────────────────────────────────────────
        "data_validator"          : create_data_validator(llm),
        "preprocessing_validator" : create_preprocessing_validator(llm),
        "feature_validator"       : create_feature_discovery_validator(llm),
        "processing_validator"    : create_processing_validator(llm),
        "pycaret_validator"       : create_pycaret_validator(llm),
        "model_validator"         : create_model_training_validator(llm),
        "prediction_validator"    : create_prediction_validator(llm),
        "explanation_validator"   : create_explanation_validator(llm),
        "retention_validator"     : create_retention_strategy_validator(llm),
        "summary_validator"       : create_summary_validator(llm),
        "feedback_validator"      : create_feedback_loop_validator(llm),
    }

    # ── Apply max_iter and max_retry_limit to every agent ─────────────────────
    # Must be set AFTER agent creation because factory functions
    # do not expose these as parameters.
    for agent in agents.values():
        agent.max_iter        = 25   # enough iterations to call all tools
        agent.max_retry_limit = 3    # retry failed tool calls up to 3 times

    tasks = create_tasks(agents)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
        max_rpm=10,    # max 10 requests/min — prevents silent OpenAI rate limit failures
    )

    return crew