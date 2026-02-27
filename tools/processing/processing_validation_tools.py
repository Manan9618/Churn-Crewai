import pandas as pd
import numpy as np
import json
import os
from crewai.tools import tool

X_TRAIN_PATH = "artifacts/data/X_train.csv"
X_TEST_PATH = "artifacts/data/X_test.csv"
Y_TRAIN_PATH = "artifacts/data/y_train.csv"
Y_TEST_PATH = "artifacts/data/y_test.csv"
FEATURES_PATH = "artifacts/data/selected_features.json"
PIPELINE_STATE_PATH = "artifacts/data/pipeline_state.json"


@tool("Validate Split Files Exist")
def validate_split_files_exist_tool() -> str:
    """Validate that all 4 train/test split files exist and are non-empty."""
    required = {
        "X_train": X_TRAIN_PATH,
        "X_test": X_TEST_PATH,
        "y_train": Y_TRAIN_PATH,
        "y_test": Y_TEST_PATH,
    }
    errors = []
    for name, path in required.items():
        if not os.path.exists(path):
            errors.append(f"Missing: {path}")
        else:
            df = pd.read_csv(path)
            if df.empty:
                errors.append(f"Empty file: {path}")
    if errors:
        return "VALIDATION FAILED:\n" + "\n".join(errors)
    return "Split files validation PASSED. All 4 split files exist and are non-empty."


@tool("Validate Split Ratio")
def validate_split_ratio_tool(expected_test_pct: float = 20.0, tolerance: float = 2.0) -> str:
    """
    Validate that the train/test split ratio matches the expected percentage
    within the given tolerance (default: 20% test ± 2%).
    """
    for p in [X_TRAIN_PATH, X_TEST_PATH]:
        if not os.path.exists(p):
            return f"VALIDATION FAILED: {p} not found."

    n_train = len(pd.read_csv(X_TRAIN_PATH))
    n_test = len(pd.read_csv(X_TEST_PATH))
    total = n_train + n_test
    actual_test_pct = n_test / total * 100

    if abs(actual_test_pct - expected_test_pct) > tolerance:
        return (
            f"VALIDATION FAILED: Test split {actual_test_pct:.1f}% deviates from "
            f"expected {expected_test_pct:.1f}% by more than {tolerance}%."
        )
    return (
        f"Split ratio PASSED. Train={n_train}, Test={n_test}, "
        f"Test%={actual_test_pct:.1f}% (expected ~{expected_test_pct:.1f}%)"
    )


@tool("Validate No Data Leakage Between Splits")
def validate_no_data_leakage_tool() -> str:
    """
    Validate there is no overlap between train and test sets by
    checking row-level uniqueness across both feature matrices.
    """
    for p in [X_TRAIN_PATH, X_TEST_PATH]:
        if not os.path.exists(p):
            return f"VALIDATION FAILED: {p} not found."

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    # Convert rows to tuples for intersection check
    train_rows = set(map(tuple, X_train.values.tolist()))
    test_rows = set(map(tuple, X_test.values.tolist()))
    overlap = len(train_rows & test_rows)

    if overlap > 0:
        return f"VALIDATION FAILED: {overlap} duplicate rows found between train and test sets (data leakage)."
    return f"Data leakage check PASSED. Zero overlapping rows between train ({len(X_train)}) and test ({len(X_test)})."


@tool("Validate Stratification")
def validate_stratification_tool(tolerance: float = 3.0) -> str:
    """
    Validate that the Churn class distribution is preserved in both
    train and test splits within the given tolerance (%).
    """
    for p in [Y_TRAIN_PATH, Y_TEST_PATH]:
        if not os.path.exists(p):
            return f"VALIDATION FAILED: {p} not found."

    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    train_churn_pct = y_train.mean() * 100
    test_churn_pct = y_test.mean() * 100
    diff = abs(train_churn_pct - test_churn_pct)

    if diff > tolerance:
        return (
            f"VALIDATION FAILED: Churn rate difference between splits ({diff:.2f}%) "
            f"exceeds tolerance ({tolerance}%).\n"
            f"Train: {train_churn_pct:.2f}% | Test: {test_churn_pct:.2f}%"
        )
    return (
        f"Stratification PASSED. Train churn: {train_churn_pct:.2f}%, "
        f"Test churn: {test_churn_pct:.2f}%, Difference: {diff:.2f}%."
    )


@tool("Validate Feature Alignment")
def validate_feature_alignment_tool() -> str:
    """Validate that X_train and X_test have identical columns in the same order."""
    for p in [X_TRAIN_PATH, X_TEST_PATH]:
        if not os.path.exists(p):
            return f"VALIDATION FAILED: {p} not found."

    train_cols = list(pd.read_csv(X_TRAIN_PATH, nrows=1).columns)
    test_cols = list(pd.read_csv(X_TEST_PATH, nrows=1).columns)

    if train_cols != test_cols:
        extra_train = set(train_cols) - set(test_cols)
        extra_test = set(test_cols) - set(train_cols)
        return (
            f"VALIDATION FAILED: Column mismatch.\n"
            f"  Only in train: {extra_train}\n"
            f"  Only in test : {extra_test}"
        )

    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            selected = json.load(f)
        unlisted = [c for c in train_cols if c not in selected]
        if unlisted:
            return f"VALIDATION WARNING: Columns in splits not in selected_features.json: {unlisted}"

    return f"Feature alignment PASSED. Both splits have identical {len(train_cols)} columns."


@tool("Validate Pipeline State")
def validate_pipeline_state_tool() -> str:
    """Validate that the pipeline state file exists and all expected files are present."""
    if not os.path.exists(PIPELINE_STATE_PATH):
        return f"VALIDATION FAILED: Pipeline state file not found at {PIPELINE_STATE_PATH}."

    with open(PIPELINE_STATE_PATH) as f:
        state = json.load(f)

    files_status = state.get("files_generated", {})
    missing = [k for k, v in files_status.items() if not v]

    if missing:
        return f"VALIDATION FAILED: Pipeline state reports missing files: {missing}"

    return (
        f"Pipeline state PASSED.\n"
        f"  Stage       : {state.get('current_stage', 'unknown')}\n"
        f"  Train size  : {state.get('train_size', 'N/A')}\n"
        f"  Test size   : {state.get('test_size', 'N/A')}\n"
        f"  Feature count: {state.get('feature_count', 'N/A')}"
    )