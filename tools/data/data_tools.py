import pandas as pd
import numpy as np
import pickle
import json
import os
from crewai.tools import tool

# ── Paths from environment (injected by main.py at startup) ───────────────────
DATA_PATH      = os.environ.get("RAW_DATA_PATH",       "data/Customer_Churn.csv")
PROCESSED_PATH = os.environ.get("PROCESSED_DATA_PATH", "artifacts/data/processed_churn.csv")
SCALER_PATH    = os.environ.get("SCALER_PATH",         "artifacts/model/scaler.pkl")


@tool("Load Dataset")
def load_dataset_tool(path: str = DATA_PATH) -> str:
    """Load the Customer Churn CSV dataset and return a summary."""
    df = pd.read_csv(path)
    return (
        f"Dataset loaded successfully.\n"
        f"Shape: {df.shape}\n"
        f"Columns: {list(df.columns)}\n"
        f"First 3 rows:\n{df.head(3).to_string()}"
    )


@tool("Dataset Info")
def dataset_info_tool(path: str = DATA_PATH) -> str:
    """Return dtypes, non-null counts and memory usage of the dataset."""
    df = pd.read_csv(path)
    buf = []
    for col in df.columns:
        buf.append(f"{col}: dtype={df[col].dtype}, non-null={df[col].notna().sum()}")
    return "\n".join(buf)


@tool("Missing Values Analysis")
def missing_values_tool(path: str = DATA_PATH) -> str:
    """Return count and percentage of missing values per column."""
    df = pd.read_csv(path)
    # TotalCharges has whitespace that acts as missing
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    result  = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    result  = result[result["missing_count"] > 0]
    if result.empty:
        return "No missing values detected (after numeric coercion)."
    return result.to_string()


@tool("Class Distribution")
def class_distribution_tool(path: str = DATA_PATH) -> str:
    """Return class distribution of the Churn target variable."""
    df     = pd.read_csv(path)
    counts = df["Churn"].value_counts()
    pct    = (counts / len(df) * 100).round(2)
    return f"Churn Distribution:\n{counts.to_string()}\n\nPercentage:\n{pct.to_string()}"


@tool("Descriptive Statistics")
def descriptive_stats_tool(path: str = DATA_PATH) -> str:
    """Return descriptive statistics for numerical columns."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.describe().to_string()


@tool("Handle Missing Values")
def handle_missing_values_tool(path: str = DATA_PATH) -> str:
    """
    Handle missing values in the dataset:
    - Convert TotalCharges to numeric (coerce whitespace to NaN)
    - Fill NaN TotalCharges with median
    Saves intermediate result to artifacts/data/processed_churn.csv.
    """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    before = df["TotalCharges"].isna().sum()
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Ensure artifacts/data/ directory exists before saving
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    return (
        f"Handled {before} missing TotalCharges values (filled with median).\n"
        f"Saved to {PROCESSED_PATH}."
    )


@tool("Encode Categorical Variables")
def encode_categoricals_tool() -> str:
    """
    Encode categorical variables from artifacts/data/processed_churn.csv:
    - Binary columns (Yes/No, Male/Female) → 0/1
    - Multi-class columns → integer codes
    - Drop customerID
    Overwrites the processed file in artifacts/data/.
    """
    df = pd.read_csv(PROCESSED_PATH)
    df.drop(columns=["customerID"], errors="ignore", inplace=True)

    binary_map  = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    for col in multi_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes

    df.to_csv(PROCESSED_PATH, index=False)
    return (
        f"Encoding complete.\n"
        f"Binary encoded : {binary_cols}\n"
        f"Label encoded  : {multi_cols}\n"
        f"Saved to       : {PROCESSED_PATH}"
    )


@tool("Scale Features")
def scale_features_tool() -> str:
    """
    Apply StandardScaler to numerical features: tenure, MonthlyCharges, TotalCharges.
    - Saves fitted scaler to artifacts/model/scaler.pkl
    - Overwrites the processed file in artifacts/data/
    """
    from sklearn.preprocessing import StandardScaler

    df       = pd.read_csv(PROCESSED_PATH)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler   = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Save scaler to artifacts/model/
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    df.to_csv(PROCESSED_PATH, index=False)
    return (
        f"Scaled {num_cols} using StandardScaler.\n"
        f"Scaler saved to : {SCALER_PATH}\n"
        f"Data saved to   : {PROCESSED_PATH}"
    )


@tool("Save Processed Data")
def save_processed_data_tool() -> str:
    """Confirm the processed data file exists in artifacts/data/ and return its shape."""
    if not os.path.exists(PROCESSED_PATH):
        return f"ERROR: Processed data file not found at {PROCESSED_PATH}."
    df = pd.read_csv(PROCESSED_PATH)
    return f"Processed data confirmed at {PROCESSED_PATH}. Shape: {df.shape}."