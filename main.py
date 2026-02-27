"""
main.py — Entry point for the Customer Churn Prediction CrewAI Pipeline.

Usage:
    python main.py

Setup:
    1. Fill in OPENAI_API_KEY in your .env file
    2. Place Customer_Churn.csv in the data/ directory
    3. pip install -r requirements.txt
    4. python main.py

Artifacts Structure (auto-created on every run):
    artifacts/
    ├── data/
    │   ├── processed_churn.csv
    │   ├── selected_features.json
    │   ├── X_train.csv
    │   ├── X_test.csv
    │   ├── y_train.csv
    │   ├── y_test.csv
    │   ├── predictions.csv
    │   ├── retention_plan.csv
    │   └── pipeline_state.json
    └── model/
        ├── churn_model.pkl
        ├── scaler.pkl
        ├── pycaret_best_model.pkl
        ├── pycaret_compare_results.csv
        ├── shap_values.pkl
        ├── explanation_report.json
        ├── experiment_log.json
        ├── pipeline_summary_report.json
        └── pipeline_summary_report.md
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load .env first — must happen before anything else
load_dotenv()

from crew import build_crew


# =============================================================================
# ARTIFACT PATHS
# =============================================================================

ARTIFACTS_DIR       = os.environ.get("ARTIFACTS_DIR", "artifacts")
ARTIFACTS_DATA_DIR  = os.path.join(ARTIFACTS_DIR, "data")
ARTIFACTS_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")


def setup_artifact_dirs():
    """Create artifacts/data/ and artifacts/model/ if they don't exist."""
    for d in [ARTIFACTS_DIR, ARTIFACTS_DATA_DIR, ARTIFACTS_MODEL_DIR]:
        os.makedirs(d, exist_ok=True)


def get_artifact_paths() -> dict:
    """
    All output file paths rooted under artifacts/.
    These are injected into os.environ so every tool automatically
    reads the correct artifact path via os.environ.get(...).
    """
    return {
        # ── Data outputs ──────────────────────────────────────────────────
        "PROCESSED_DATA_PATH"     : os.path.join(ARTIFACTS_DATA_DIR,  "processed_churn.csv"),
        "FEATURES_PATH"           : os.path.join(ARTIFACTS_DATA_DIR,  "selected_features.json"),
        "X_TRAIN_PATH"            : os.path.join(ARTIFACTS_DATA_DIR,  "X_train.csv"),
        "X_TEST_PATH"             : os.path.join(ARTIFACTS_DATA_DIR,  "X_test.csv"),
        "Y_TRAIN_PATH"            : os.path.join(ARTIFACTS_DATA_DIR,  "y_train.csv"),
        "Y_TEST_PATH"             : os.path.join(ARTIFACTS_DATA_DIR,  "y_test.csv"),
        "PREDICTIONS_PATH"        : os.path.join(ARTIFACTS_DATA_DIR,  "predictions.csv"),
        "RETENTION_PLAN_PATH"     : os.path.join(ARTIFACTS_DATA_DIR,  "retention_plan.csv"),
        "PIPELINE_STATE_PATH"     : os.path.join(ARTIFACTS_DATA_DIR,  "pipeline_state.json"),

        # ── Model outputs ─────────────────────────────────────────────────
        "MODEL_PATH"              : os.path.join(ARTIFACTS_MODEL_DIR, "churn_model.pkl"),
        "SCALER_PATH"             : os.path.join(ARTIFACTS_MODEL_DIR, "scaler.pkl"),
        "PYCARET_MODEL_PATH"      : os.path.join(ARTIFACTS_MODEL_DIR, "pycaret_best_model"),
        "PYCARET_RESULTS_PATH"    : os.path.join(ARTIFACTS_MODEL_DIR, "pycaret_compare_results.csv"),
        "SHAP_PATH"               : os.path.join(ARTIFACTS_MODEL_DIR, "shap_values.pkl"),
        "EXPLANATION_REPORT_PATH" : os.path.join(ARTIFACTS_MODEL_DIR, "explanation_report.json"),
        "EXPERIMENT_LOG_PATH"     : os.path.join(ARTIFACTS_MODEL_DIR, "experiment_log.json"),
        "SUMMARY_REPORT_PATH"     : os.path.join(ARTIFACTS_MODEL_DIR, "pipeline_summary_report.json"),
        "SUMMARY_MD_PATH"         : os.path.join(ARTIFACTS_MODEL_DIR, "pipeline_summary_report.md"),
    }


def inject_artifact_paths(paths: dict):
    """
    Inject artifact paths into os.environ so every tool that calls
    os.environ.get() automatically uses the correct artifacts/ path.
    Only sets values not already defined in .env to avoid overrides.
    """
    for key, value in paths.items():
        if not os.environ.get(key):
            os.environ[key] = value


def print_artifact_summary(paths: dict):
    """Print a structured ✓ / ✗ summary of every generated artifact file."""

    # friendly display names
    labels = {
        "PROCESSED_DATA_PATH"     : "Processed data",
        "FEATURES_PATH"           : "Selected features",
        "X_TRAIN_PATH"            : "Train features",
        "X_TEST_PATH"             : "Test features",
        "Y_TRAIN_PATH"            : "Train labels",
        "Y_TEST_PATH"             : "Test labels",
        "PREDICTIONS_PATH"        : "Predictions",
        "RETENTION_PLAN_PATH"     : "Retention plan",
        "PIPELINE_STATE_PATH"     : "Pipeline state",
        "MODEL_PATH"              : "Trained model",
        "SCALER_PATH"             : "Scaler",
        "PYCARET_MODEL_PATH"      : "PyCaret model",
        "PYCARET_RESULTS_PATH"    : "PyCaret results",
        "SHAP_PATH"               : "SHAP values",
        "EXPLANATION_REPORT_PATH" : "Explanation report",
        "EXPERIMENT_LOG_PATH"     : "Experiment log",
        "SUMMARY_REPORT_PATH"     : "Summary (JSON)",
        "SUMMARY_MD_PATH"         : "Summary (Markdown)",
    }

    print("\n" + "=" * 65)
    print("  ARTIFACTS GENERATED")
    print("=" * 65)

    # ── artifacts/data/ ───────────────────────────────────────────────────
    data_keys = [
        "PROCESSED_DATA_PATH", "FEATURES_PATH",
        "X_TRAIN_PATH", "X_TEST_PATH",
        "Y_TRAIN_PATH", "Y_TEST_PATH",
        "PREDICTIONS_PATH", "RETENTION_PLAN_PATH", "PIPELINE_STATE_PATH",
    ]
    print(f"\n  📁 {ARTIFACTS_DATA_DIR}/")
    for key in data_keys:
        path   = paths[key]
        status = "✓" if os.path.exists(path) else "✗"
        print(f"    [{status}] {labels[key]:<24} → {os.path.basename(path)}")

    # ── artifacts/model/ ──────────────────────────────────────────────────
    model_keys = [
        "MODEL_PATH", "SCALER_PATH",
        "PYCARET_MODEL_PATH", "PYCARET_RESULTS_PATH",
        "SHAP_PATH", "EXPLANATION_REPORT_PATH",
        "EXPERIMENT_LOG_PATH", "SUMMARY_REPORT_PATH", "SUMMARY_MD_PATH",
    ]
    print(f"\n  📁 {ARTIFACTS_MODEL_DIR}/")
    for key in model_keys:
        path       = paths[key]
        # PyCaret appends .pkl automatically — check both
        check_path = path if os.path.exists(path) else path + ".pkl"
        status     = "✓" if os.path.exists(check_path) else "✗"
        print(f"    [{status}] {labels[key]:<24} → {os.path.basename(path)}")

    print(f"\n  📂 Full path: {os.path.abspath(ARTIFACTS_DIR)}/")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():

    # ── 1. Check API key ───────────────────────────────────────────────────
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Add it to your .env file: OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)

    model_name  = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
    temperature = float(os.environ.get("TEMPERATURE", "0.3"))

    # ── 2. Check raw dataset exists ────────────────────────────────────────
    data_path = os.environ.get("RAW_DATA_PATH", "data/Customer_Churn.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at '{data_path}'.")
        print("Please place Customer_Churn.csv in the data/ directory.")
        sys.exit(1)

    # ── 3. Setup artifacts/ folder & inject paths into os.environ ─────────
    artifact_paths = get_artifact_paths()
    inject_artifact_paths(artifact_paths)
    setup_artifact_dirs()

    # ── 4. Print run header ────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  Customer Churn Prediction — CrewAI Pipeline")
    print("=" * 65)
    print(f"  Model      : {model_name}")
    print(f"  Temperature: {temperature}")
    print(f"  Dataset    : {data_path}")
    print(f"  Artifacts  : {os.path.abspath(ARTIFACTS_DIR)}/")
    print(f"  Timestamp  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    # ── 5. Build and kick off the crew ─────────────────────────────────────
    crew    = build_crew(model_name=model_name, temperature=temperature)
    start   = time.time()
    result  = crew.kickoff()
    elapsed = time.time() - start

    # ── 6. Print completion banner ─────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  Pipeline Complete  ✓  ({elapsed:.1f}s)")
    print("=" * 65)
    print()
    print("Final Output:")
    print(result)

    # ── 7. Print artifact file summary ─────────────────────────────────────
    print_artifact_summary(artifact_paths)


if __name__ == "__main__":
    main()