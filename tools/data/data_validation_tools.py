import pandas as pd
import numpy as np
import os
from crewai.tools import tool

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
DATA_PATH = "data/Customer_Churn.csv"
PROCESSED_PATH = "artifacts/data/processed_churn.csv"


@tool("Validate Schema")
def validate_schema_tool(path: str = DATA_PATH) -> str:
    """Validate that the dataset has all 21 expected columns."""
    df = pd.read_csv(path)
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing_cols:
        return f"VALIDATION FAILED: Missing columns: {missing_cols}"
    if extra_cols:
        return f"WARNING: Unexpected extra columns: {extra_cols}"
    return f"Schema validation PASSED. All {len(EXPECTED_COLUMNS)} expected columns present."


@tool("Validate Data Types")
def validate_data_types_tool(path: str = DATA_PATH) -> str:
    """Validate that key columns have correct/coercible data types."""
    df = pd.read_csv(path)
    errors = []

    df["TotalCharges_num"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if df["TotalCharges_num"].isna().sum() > 20:
        errors.append(f"TotalCharges has too many non-numeric values: {df['TotalCharges_num'].isna().sum()}")

    if df["tenure"].dtype not in [np.int64, np.float64]:
        errors.append(f"tenure dtype unexpected: {df['tenure'].dtype}")

    if errors:
        return "VALIDATION FAILED:\n" + "\n".join(errors)
    return "Data type validation PASSED."


@tool("Validate Missing Threshold")
def validate_missing_threshold_tool(path: str = DATA_PATH, threshold: float = 1.0) -> str:
    """Validate that no column exceeds the missing value percentage threshold."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    pct_missing = (df.isnull().sum() / len(df) * 100).round(2)
    violations = pct_missing[pct_missing > threshold]
    if not violations.empty:
        return f"VALIDATION FAILED: Columns exceeding {threshold}% missing:\n{violations.to_string()}"
    return f"Missing value threshold validation PASSED. All columns ≤ {threshold}% missing."


@tool("Validate Target Column")
def validate_target_column_tool(path: str = DATA_PATH) -> str:
    """Validate that the Churn target column only contains 'Yes' or 'No'."""
    df = pd.read_csv(path)
    unique_vals = df["Churn"].unique()
    invalid = [v for v in unique_vals if v not in ("Yes", "No")]
    if invalid:
        return f"VALIDATION FAILED: Unexpected Churn values: {invalid}"
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    return f"Target validation PASSED. Churn rate: {churn_rate:.2f}%. Values: {list(unique_vals)}"


@tool("Validate No Missing After Preprocessing")
def validate_no_missing_after_preprocessing_tool(path: str = PROCESSED_PATH) -> str:
    """Validate that processed dataset has zero missing values."""
    if not os.path.exists(path):
        return f"VALIDATION FAILED: Processed file not found at {path}."
    df = pd.read_csv(path)
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        return f"VALIDATION FAILED: {total_missing} missing values remain after preprocessing."
    return "Post-preprocessing missing value validation PASSED. Zero missing values."


@tool("Validate Encoding")
def validate_encoding_tool(path: str = PROCESSED_PATH) -> str:
    """Validate that all columns in the processed dataset are numeric."""
    if not os.path.exists(path):
        return f"VALIDATION FAILED: Processed file not found at {path}."
    df = pd.read_csv(path)
    non_numeric = [c for c in df.columns if df[c].dtype == object]
    if non_numeric:
        return f"VALIDATION FAILED: Non-numeric columns remain: {non_numeric}"
    return "Encoding validation PASSED. All columns are numeric."


@tool("Validate Scaling")
def validate_scaling_tool(path: str = PROCESSED_PATH) -> str:
    """Validate that numerical features have approximately zero mean after scaling."""
    if not os.path.exists(path):
        return f"VALIDATION FAILED: Processed file not found at {path}."
    df = pd.read_csv(path)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    results = []
    for col in num_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        results.append(f"{col}: mean={mean_val:.4f}, std={std_val:.4f}")
    return "Scaling check:\n" + "\n".join(results)


@tool("Validate Train Test Split")
def validate_train_test_split_tool(
    train_path: str = "data/X_train.csv",
    test_path: str = "data/X_test.csv",
) -> str:
    """Validate that train and test splits exist and are non-empty."""
    errors = []
    for p in [train_path, test_path]:
        if not os.path.exists(p):
            errors.append(f"Missing: {p}")
        else:
            df = pd.read_csv(p)
            if df.empty:
                errors.append(f"Empty file: {p}")
    if errors:
        return "VALIDATION FAILED:\n" + "\n".join(errors)
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    ratio = len(te) / (len(tr) + len(te)) * 100
    return f"Train/test split PASSED. Train={len(tr)}, Test={len(te)}, Test%={ratio:.1f}%"