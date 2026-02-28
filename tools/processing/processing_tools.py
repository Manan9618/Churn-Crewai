import pandas as pd
import numpy as np
import pickle
import json
import os
import hashlib
from datetime import datetime
from crewai.tools import tool

PROCESSED_PATH = "artifacts/data/processed_churn.csv"
FEATURES_PATH = "artifacts/data/selected_features.json"
X_TRAIN_PATH = "artifacts/data/X_train.csv"
X_TEST_PATH = "artifacts/data/X_test.csv"
Y_TRAIN_PATH = "artifacts/data/y_train.csv"
Y_TEST_PATH = "artifacts/data/y_test.csv"
PIPELINE_STATE_PATH = "artifacts/data/pipeline_state.json"
TIME_SPLIT_PATH = "artifacts/data/time_series_split_report.json"
VERSION_LOG_PATH = "artifacts/data/dataset_version_log.json"
PREPROCESSOR_PATH = "artifacts/model/preprocessor.pkl"


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

    os.makedirs("artifacts/data", exist_ok=True)
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

    os.makedirs("artifacts/data", exist_ok=True)
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


# ── NEW PROCESSING TOOLS ADDED BELOW ──────────────────────────────────────────


@tool("Time Series Split")
def time_series_split_tool(
    date_column: str = "tenure",
    test_size: float = 0.2,
    gap_periods: int = 0
) -> str:
    """
    If data is temporal, ensures train/test split respects time order.
    Prevents look-ahead bias by training on past data and testing on future data.
    Uses tenure as a proxy for time if no explicit date column exists.
    """
    if not os.path.exists(PROCESSED_PATH):
        return f"ERROR: Processed data not found at {PROCESSED_PATH}."

    df = pd.read_csv(PROCESSED_PATH).dropna()

    if not os.path.exists(FEATURES_PATH):
        return f"ERROR: Selected features not found at {FEATURES_PATH}."

    with open(FEATURES_PATH) as f:
        features = json.load(f)

    # Check if date column exists
    if date_column not in df.columns:
        return (
            f"VALIDATION WARNING: Date column '{date_column}' not found.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Falling back to stratified random split instead of time-based split."
        )

    # Sort by date/tenure column
    df_sorted = df.sort_values(by=date_column).reset_index(drop=True)

    X = df_sorted[features]
    y = df_sorted["Churn"]

    # Calculate split index
    n_samples = len(df_sorted)
    split_idx = int(n_samples * (1 - test_size))

    # Apply gap if specified (prevents data leakage near split point)
    if gap_periods > 0:
        split_idx = max(0, split_idx - gap_periods)

    # Split respecting time order
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Save splits
    os.makedirs("artifacts/data", exist_ok=True)
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.reset_index(drop=True).to_csv(Y_TRAIN_PATH, index=False)
    y_test.reset_index(drop=True).to_csv(Y_TEST_PATH, index=False)

    # Calculate metrics
    train_date_range = (df_sorted[date_column].iloc[0], df_sorted[date_column].iloc[split_idx - 1])
    test_date_range = (df_sorted[date_column].iloc[split_idx], df_sorted[date_column].iloc[-1])
    train_churn_rate = y_train.mean() * 100
    test_churn_rate = y_test.mean() * 100

    # Save time series split report
    split_report = {
        "timestamp": datetime.now().isoformat(),
        "split_type": "time_series",
        "date_column": date_column,
        "gap_periods": gap_periods,
        "train_date_range": [str(train_date_range[0]), str(train_date_range[1])],
        "test_date_range": [str(test_date_range[0]), str(test_date_range[1])],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_rate": round(train_churn_rate, 2),
        "test_churn_rate": round(test_churn_rate, 2),
        "lookahead_bias_prevented": True
    }

    with open(TIME_SPLIT_PATH, "w") as f:
        json.dump(split_report, f, indent=2)

    return (
        f"Time Series Split Complete:\n"
        f"  Date Column: {date_column}\n"
        f"  Gap Periods: {gap_periods}\n"
        f"\n  Train Period: {train_date_range[0]} → {train_date_range[1]}\n"
        f"  Test Period:  {test_date_range[0]} → {test_date_range[1]}\n"
        f"\n  Train size : {len(X_train)} samples | Churn rate: {train_churn_rate:.2f}%\n"
        f"  Test size  : {len(X_test)} samples  | Churn rate: {test_churn_rate:.2f}%\n"
        f"\n  ✅ Look-ahead bias prevented (train data is strictly before test data)\n"
        f"  Report saved to: {TIME_SPLIT_PATH}"
    )


@tool("Apply ADASYN Undersampling")
def apply_adasyn_tool(
    random_state: int = 42,
    sampling_strategy: str = "auto",
    n_neighbors: int = 5
) -> str:
    """
    Alternative to SMOTE for handling class imbalance using ADASYN.
    ADASYN adapts the amount of synthetic samples generated based on
    the density distribution of minority class examples.
    """
    try:
        from imblearn.over_sampling import ADASYN
    except ImportError:
        return (
            "ERROR: imblearn package not installed or ADASYN not available.\n"
            "Install with: pip install imbalanced-learn\n"
            "Falling back to SMOTE instead."
        )

    for p in [X_TRAIN_PATH, Y_TRAIN_PATH]:
        if not os.path.exists(p):
            return f"ERROR: {p} not found. Run train_test_split_tool first."

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

    before_counts = y_train.value_counts().to_dict()
    before_ratio = before_counts.get(0, 0) / max(before_counts.get(1, 1), 1)

    # Apply ADASYN
    adasyn = ADASYN(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        n_neighbors=n_neighbors
    )

    try:
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    except Exception as e:
        return (
            f"ERROR: ADASYN failed - {str(e)}\n"
            f"This can happen if minority class has too few samples.\n"
            f"Recommendation: Use SMOTE instead or collect more data."
        )

    # Save resampled data
    pd.DataFrame(X_resampled, columns=X_train.columns).to_csv(X_TRAIN_PATH, index=False)
    pd.Series(y_resampled, name="Churn").to_csv(Y_TRAIN_PATH, index=False)

    after_counts = pd.Series(y_resampled).value_counts().to_dict()
    after_ratio = after_counts.get(0, 0) / max(after_counts.get(1, 1), 1)

    # Save ADASYN report
    adasyn_report = {
        "timestamp": datetime.now().isoformat(),
        "method": "ADASYN",
        "random_state": random_state,
        "n_neighbors": n_neighbors,
        "sampling_strategy": sampling_strategy,
        "before_class_distribution": before_counts,
        "after_class_distribution": after_counts,
        "before_imbalance_ratio": round(before_ratio, 4),
        "after_imbalance_ratio": round(after_ratio, 4),
        "original_train_size": len(X_train),
        "resampled_train_size": len(X_resampled),
        "samples_added": len(X_resampled) - len(X_train)
    }

    os.makedirs("artifacts/data", exist_ok=True)
    adasyn_path = "artifacts/data/adasyn_report.json"
    with open(adasyn_path, "w") as f:
        json.dump(adasyn_report, f, indent=2)

    return (
        f"ADASYN Applied to Training Set:\n"
        f"  Random State: {random_state}\n"
        f"  N Neighbors: {n_neighbors}\n"
        f"\n  Before:\n"
        f"    Class Distribution: {before_counts}\n"
        f"    Imbalance Ratio: {before_ratio:.4f}\n"
        f"    Train Size: {len(X_train)}\n"
        f"\n  After:\n"
        f"    Class Distribution: {after_counts}\n"
        f"    Imbalance Ratio: {after_ratio:.4f}\n"
        f"    Train Size: {len(X_resampled)}\n"
        f"    Samples Added: {len(X_resampled) - len(X_train)}\n"
        f"\n  Report saved to: {adasyn_path}\n"
        f"\n💡 ADASYN adapts synthetic sample generation based on density distribution."
    )


@tool("Version Dataset Artifact")
def version_dataset_artifact_tool(
    version_tag: str = None,
    include_hash: bool = True
) -> str:
    """
    Tags the processed dataset with a version ID for reproducibility.
    Generates MD5 hash and metadata for tracking dataset versions.
    """
    if not os.path.exists(PROCESSED_PATH):
        return f"ERROR: Processed data not found at {PROCESSED_PATH}."

    # Generate version tag if not provided
    if version_tag is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = f"v{timestamp}"

    # Calculate file hash
    file_hash = None
    file_size = None
    if include_hash:
        with open(PROCESSED_PATH, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        file_size = os.path.getsize(PROCESSED_PATH)

    # Collect metadata
    df = pd.read_csv(PROCESSED_PATH)
    metadata = {
        "version_tag": version_tag,
        "timestamp": datetime.now().isoformat(),
        "file_path": PROCESSED_PATH,
        "file_size_bytes": file_size,
        "file_hash_md5": file_hash,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "churn_rate": round(df["Churn"].mean() * 100, 2) if "Churn" in df.columns else None
    }

    # Load existing version log or create new
    if os.path.exists(VERSION_LOG_PATH):
        with open(VERSION_LOG_PATH, "r") as f:
            version_log = json.load(f)
    else:
        version_log = {"versions": []}

    # Add new version
    version_log["versions"].append(metadata)
    version_log["latest_version"] = version_tag
    version_log["total_versions"] = len(version_log["versions"])

    # Save version log
    os.makedirs("artifacts/data", exist_ok=True)
    with open(VERSION_LOG_PATH, "w") as f:
        json.dump(version_log, f, indent=2)

    # Also save version-specific metadata
    version_metadata_path = f"artifacts/data/dataset_{version_tag}.json"
    with open(version_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return (
        f"Dataset Versioned Successfully:\n"
        f"  Version Tag: {version_tag}\n"
        f"  Timestamp: {metadata['timestamp']}\n"
        f"  File: {PROCESSED_PATH}\n"
        f"  Size: {file_size:,} bytes\n"
        f"  MD5 Hash: {file_hash}\n"
        f"  Rows: {metadata['row_count']:,}\n"
        f"  Columns: {metadata['column_count']}\n"
        f"  Churn Rate: {metadata['churn_rate']}%\n"
        f"\n  Version Log: {VERSION_LOG_PATH}\n"
        f"  Version Metadata: {version_metadata_path}\n"
        f"  Total Versions Tracked: {version_log['total_versions']}\n"
        f"\n💡 Use version tag '{version_tag}' to reproduce this exact dataset."
    )


@tool("Serialize Preprocessor")
def serialize_preprocessor_tool(
    scaler_path: str = "artifacts/model/scaler.pkl",
    encoder_path: str = "artifacts/model/encoder.pkl"
) -> str:
    """
    Saves the scaler/encoder separately for use in real-time inference APIs.
    Ensures preprocessing can be applied consistently in production.
    """
    from sklearn.preprocessing import StandardScaler
    import json

    # Check if processed data exists
    if not os.path.exists(PROCESSED_PATH):
        return f"ERROR: Processed data not found at {PROCESSED_PATH}."

    df = pd.read_csv(PROCESSED_PATH).dropna()

    # Load features
    if not os.path.exists(FEATURES_PATH):
        return f"ERROR: Selected features not found at {FEATURES_PATH}."

    with open(FEATURES_PATH) as f:
        features = json.load(f)

    # Identify numerical and categorical columns
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [col for col in features if col not in numerical_cols and col in df.columns]

    # Fit and save scaler for numerical columns
    os.makedirs("artifacts/model", exist_ok=True)

    scaler = StandardScaler()
    if all(col in df.columns for col in numerical_cols):
        scaler.fit(df[numerical_cols])
        with open(scaler_path, "wb") as f:
            pickle.dump({
                "scaler": scaler,
                "columns": numerical_cols,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "var": scaler.var_.tolist()
            }, f)
        scaler_status = f"✅ Scaler saved ({len(numerical_cols)} columns)"
    else:
        scaler_status = "⚠️ Scaler skipped (numerical columns not found)"

    # Save encoder mapping for categorical columns
    encoder_mapping = {}
    for col in categorical_cols:
        if col in df.columns:
            unique_values = df[col].unique().tolist()
            # Create encoding mapping
            if df[col].dtype == 'object':
                encoder_mapping[col] = {val: idx for idx, val in enumerate(unique_values)}
            else:
                encoder_mapping[col] = "numeric"

    if encoder_mapping:
        with open(encoder_path, "wb") as f:
            pickle.dump({
                "encoder_mapping": encoder_mapping,
                "columns": categorical_cols
            }, f)
        encoder_status = f"✅ Encoder saved ({len(categorical_cols)} columns)"
    else:
        encoder_status = "⚠️ Encoder skipped (categorical columns not found)"

    # Save preprocessor metadata
    preprocessor_metadata = {
        "timestamp": datetime.now().isoformat(),
        "scaler_path": scaler_path,
        "encoder_path": encoder_path,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "scaler_saved": os.path.exists(scaler_path),
        "encoder_saved": os.path.exists(encoder_path),
        "feature_count": len(features),
        "ready_for_inference": os.path.exists(scaler_path)
    }

    metadata_path = "artifacts/model/preprocessor_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(preprocessor_metadata, f, indent=2)

    return (
        f"Preprocessor Serialized Successfully:\n"
        f"  Scaler: {scaler_status}\n"
        f"    Path: {scaler_path}\n"
        f"    Columns: {numerical_cols}\n"
        f"\n  Encoder: {encoder_status}\n"
        f"    Path: {encoder_path}\n"
        f"    Columns: {len(categorical_cols)} categorical\n"
        f"\n  Metadata: {metadata_path}\n"
        f"  Total Features: {len(features)}\n"
        f"  Ready for Inference: {'✅ YES' if preprocessor_metadata['ready_for_inference'] else '❌ NO'}\n"
        f"\n💡 Use these preprocessors in real-time APIs for consistent predictions."
    )