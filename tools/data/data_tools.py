import pandas as pd
import numpy as np
import pickle
import json
import os
import re
from crewai.tools import tool

# ── Paths from environment (injected by main.py at startup) ───────────────────
DATA_PATH      = os.environ.get("RAW_DATA_PATH",       "data/Customer_Churn.csv")
PROCESSED_PATH = os.environ.get("PROCESSED_DATA_PATH", "artifacts/data/processed_churn.csv")
SCALER_PATH    = os.environ.get("SCALER_PATH",         "artifacts/model/scaler.pkl")
BASELINE_PATH  = os.environ.get("BASELINE_PATH",       "artifacts/data/baseline_churn.csv")


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


# ── NEW TOOLS ADDED BELOW ──────────────────────────────────────────────────────

@tool("Detect PII Fields")
def detect_pii_fields_tool(path: str = DATA_PATH) -> str:
    """
    Scans column names and sample values for Personally Identifiable Information (PII).
    Checks for patterns like Email, Phone Numbers, SSN, and generic IDs.
    """
    df = pd.read_csv(path)
    pii_keywords = ["email", "phone", "ssn", "social", "password", "credit_card", "id", "name"]
    pii_patterns = {
        "Email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "Phone": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "SSN":   r"\d{3}-\d{2}-\d{4}"
    }
    
    detected_pii = []
    
    # Check column names
    for col in df.columns:
        if any(keyword in col.lower() for keyword in pii_keywords):
            detected_pii.append(f"Column Name Match: '{col}'")
    
    # Check sample values (first 10 rows) for regex patterns
    sample = df.head(10).astype(str)
    for col in sample.columns:
        for pattern_name, pattern in pii_patterns.items():
            if sample[col].str.contains(pattern, regex=True).any():
                detected_pii.append(f"Value Pattern Match ({pattern_name}): '{col}'")
                
    if not detected_pii:
        return "No obvious PII fields or patterns detected in column names or sample values."
    
    return f"Potential PII detected:\n" + "\n".join([f"- {item}" for item in detected_pii])


@tool("Handle Outliers IQR")
def handle_outliers_iqr_tool(path: str = PROCESSED_PATH) -> str:
    """
    Detects and caps outliers using Interquartile Range (IQR) on numerical columns.
    Values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are capped to the boundary.
    Overwrites the processed file.
    """
    if not os.path.exists(path):
        return f"ERROR: File not found at {path}. Run preprocessing steps first."
        
    df = pd.read_csv(path)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    caps_applied = 0
    details = []
    
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers > 0:
            # Cap values (Winsorize)
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            caps_applied += outliers
            details.append(f"{col}: {outliers} values capped")
            
    df.to_csv(path, index=False)
    
    if caps_applied == 0:
        return "No outliers detected based on IQR method."
    
    return (
        f"Outlier handling complete.\n"
        f"Total values capped: {caps_applied}\n"
        f"Details:\n" + "\n".join([f"- {d}" for d in details]) +
        f"\nSaved to: {path}"
    )


@tool("Downcast Memory Optimization")
def downcast_memory_tool(path: str = PROCESSED_PATH) -> str:
    """
    Reduces dataframe memory usage by downcasting numeric types and converting 
    low-cardinality object columns to categories.
    Overwrites the processed file.
    """
    if not os.path.exists(path):
        return f"ERROR: File not found at {path}."
        
    df = pd.read_csv(path)
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2  # MB
    
    # Downcast numerics
    for col in df.select_dtypes(include=['int', 'float']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if str(df[col].dtype).startswith('int'):
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
                
        elif str(df[col].dtype).startswith('float'):
            if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    
    # Convert objects to category if cardinality is low
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')
            
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2  # MB
    reduction = ((start_mem - end_mem) / start_mem) * 100
    
    df.to_csv(path, index=False)
    
    return (
        f"Memory optimization complete.\n"
        f"Before: {start_mem:.2f} MB\n"
        f"After:  {end_mem:.2f} MB\n"
        f"Reduction: {reduction:.2f}%\n"
        f"Saved to: {path}"
    )


@tool("Detect Duplicate Records")
def detect_duplicate_records_tool(path: str = PROCESSED_PATH) -> str:
    """
    Identifies exact duplicate rows. Removes them if found to prevent data leakage.
    Overwrites the processed file.
    """
    if not os.path.exists(path):
        return f"ERROR: File not found at {path}."
        
    df = pd.read_csv(path)
    original_shape = df.shape
    
    duplicates = df.duplicated().sum()
    
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        df.to_csv(path, index=False)
        return (
            f"Duplicate detection complete.\n"
            f"Found {duplicates} duplicate rows.\n"
            f"Original Shape: {original_shape}\n"
            f"New Shape:      {df.shape}\n"
            f"Duplicates removed and saved to: {path}"
        )
    else:
        return f"No duplicate records found in {path}. Shape: {original_shape}."


@tool("Check Data Drift")
def check_data_drift_tool(current_path: str = PROCESSED_PATH, baseline_path: str = BASELINE_PATH) -> str:
    """
    Compares current input data distribution against a reference baseline file.
    Calculates mean shift percentage for numerical columns.
    """
    if not os.path.exists(current_path):
        return f"ERROR: Current data file not found at {current_path}."
    
    if not os.path.exists(baseline_path):
        return (
            f"WARNING: Baseline file not found at {baseline_path}.\n"
            f"Cannot perform drift detection without a historical baseline dataset."
        )
    
    df_current = pd.read_csv(current_path)
    df_baseline = pd.read_csv(baseline_path)
    
    # Identify common numerical columns
    num_cols = df_current.select_dtypes(include=[np.number]).columns
    common_cols = [c for c in num_cols if c in df_baseline.columns]
    
    if not common_cols:
        return "No common numerical columns found between current and baseline data."
    
    drift_report = []
    significant_drift = False
    
    for col in common_cols:
        mean_base = df_baseline[col].mean()
        mean_curr = df_current[col].mean()
        
        if mean_base == 0:
            shift_pct = 0
        else:
            shift_pct = abs((mean_curr - mean_base) / mean_base) * 100
            
        drift_report.append(f"{col}: {shift_pct:.2f}% mean shift")
        
        if shift_pct > 10.0: # Threshold for significant drift
            significant_drift = True
            
    status = "⚠️ SIGNIFICANT DRIFT DETECTED" if significant_drift else "✅ Data distribution stable"
    
    return (
        f"Data Drift Check ({status}):\n"
        f"Comparing: {current_path} vs {baseline_path}\n"
        f"Columns Analyzed: {len(common_cols)}\n"
        f"Details:\n" + "\n".join([f"- {r}" for r in drift_report])
    )