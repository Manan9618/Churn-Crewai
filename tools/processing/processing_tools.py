import pandas as pd
import numpy as np
import pickle
import json
import os
from crewai.tools import tool

PROCESSED_PATH = "artifacts/data/processed_churn.csv"
FEATURES_PATH = "artifacts/data/selected_features.json"
X_TRAIN_PATH = "artifacts/data/X_train.csv"
X_TEST_PATH = "artifacts/data/X_test.csv"
Y_TRAIN_PATH = "artifacts/data/y_train.csv"
Y_TEST_PATH = "artifacts/data/y_test.csv"
PIPELINE_STATE_PATH = "artifacts/data/pipeline_state.json"


@tool("Train Test Split")
def train_test_split_tool(test_size: float = 0.2, random_state: int = 42) -> str:
    """
    Split the processed dataset into stratified train and test sets.
    Saves X_train, X_test, y_train, y_test CSVs to the data/ directory.
    """
    from sklearn.model_selection import train_test_split

    if not os.path.exists(PROCESSED_PATH):
        return f"ERROR: Processed data not found at {PROCESSED_PATH}."

    df = pd.read_csv(PROCESSED_PATH).dropna()

    if not os.path.exists(FEATURES_PATH):
        return f"ERROR: Selected features not found at {FEATURES_PATH}. Run feature selection first."

    with open(FEATURES_PATH) as f:
        features = json.load(f)

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        return f"ERROR: Features missing from dataset: {missing_features}"

    X = df[features]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    os.makedirs("data", exist_ok=True)
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.reset_index(drop=True).to_csv(Y_TRAIN_PATH, index=False)
    y_test.reset_index(drop=True).to_csv(Y_TEST_PATH, index=False)

    train_churn_rate = y_train.mean() * 100
    test_churn_rate = y_test.mean() * 100

    return (
        f"Train/test split complete.\n"
        f"  Train size : {len(X_train)} samples | Churn rate: {train_churn_rate:.2f}%\n"
        f"  Test size  : {len(X_test)} samples  | Churn rate: {test_churn_rate:.2f}%\n"
        f"  Features   : {len(features)}\n"
        f"  Saved to   : {X_TRAIN_PATH}, {X_TEST_PATH}, {Y_TRAIN_PATH}, {Y_TEST_PATH}"
    )


@tool("Apply SMOTE Oversampling")
def apply_smote_tool(random_state: int = 42) -> str:
    """
    Apply SMOTE to the training set to handle class imbalance.
    Overwrites X_train.csv and y_train.csv with the balanced versions.
    """
    from imblearn.over_sampling import SMOTE

    for p in [X_TRAIN_PATH, Y_TRAIN_PATH]:
        if not os.path.exists(p):
            return f"ERROR: {p} not found. Run train_test_split_tool first."

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

    before_counts = y_train.value_counts().to_dict()
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    pd.DataFrame(X_resampled, columns=X_train.columns).to_csv(X_TRAIN_PATH, index=False)
    pd.Series(y_resampled, name="Churn").to_csv(Y_TRAIN_PATH, index=False)

    after_counts = pd.Series(y_resampled).value_counts().to_dict()
    return (
        f"SMOTE applied to training set.\n"
        f"  Before: {before_counts}\n"
        f"  After : {after_counts}\n"
        f"  New train size: {len(X_resampled)}"
    )


@tool("Prepare Final Feature Matrix")
def prepare_feature_matrix_tool() -> str:
    """
    Verify and report the final feature matrix dimensions for both
    train and test sets. Ensures feature alignment between splits.
    """
    for p in [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]:
        if not os.path.exists(p):
            return f"ERROR: Required file missing: {p}. Run train_test_split_tool first."

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    # Check feature alignment
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    misaligned = train_cols.symmetric_difference(test_cols)
    if misaligned:
        return f"ERROR: Feature mismatch between train and test: {misaligned}"

    return (
        f"Feature matrix ready.\n"
        f"  X_train : {X_train.shape} | y_train: {y_train.shape}\n"
        f"  X_test  : {X_test.shape}  | y_test : {y_test.shape}\n"
        f"  Features: {list(X_train.columns)}\n"
        f"  Train churn rate: {y_train.mean() * 100:.2f}%\n"
        f"  Test churn rate : {y_test.mean() * 100:.2f}%"
    )


@tool("Save Pipeline State")
def save_pipeline_state_tool(stage: str = "processing") -> str:
    """
    Save the current pipeline processing stage and metadata to a state file.
    Useful for resuming or auditing pipeline runs.
    """
    state = {
        "current_stage": stage,
        "files_generated": {},
    }

    file_checks = {
        "processed_data": PROCESSED_PATH,
        "selected_features": FEATURES_PATH,
        "X_train": X_TRAIN_PATH,
        "X_test": X_TEST_PATH,
        "y_train": Y_TRAIN_PATH,
        "y_test": Y_TEST_PATH,
    }

    for key, path in file_checks.items():
        state["files_generated"][key] = os.path.exists(path)

    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            state["feature_count"] = len(json.load(f))

    if os.path.exists(X_TRAIN_PATH):
        df = pd.read_csv(X_TRAIN_PATH)
        state["train_size"] = len(df)

    if os.path.exists(X_TEST_PATH):
        df = pd.read_csv(X_TEST_PATH)
        state["test_size"] = len(df)

    os.makedirs("data", exist_ok=True)
    with open(PIPELINE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

    return (
        f"Pipeline state saved to {PIPELINE_STATE_PATH}.\n"
        f"Stage: {stage}\n"
        f"Files present: {state['files_generated']}"
    )


@tool("Load Pipeline State")
def load_pipeline_state_tool() -> str:
    """Load and display the current pipeline state from disk."""
    if not os.path.exists(PIPELINE_STATE_PATH):
        return f"No pipeline state file found at {PIPELINE_STATE_PATH}. Pipeline not yet started."
    with open(PIPELINE_STATE_PATH) as f:
        state = json.load(f)
    lines = [f"Pipeline State:"]
    for key, val in state.items():
        lines.append(f"  {key}: {val}")
    return "\n".join(lines)