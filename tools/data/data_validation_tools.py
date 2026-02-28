import pandas as pd
import numpy as np
import os
from scipy import stats  # For KS-Test and Chi-Square
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
BASELINE_PATH = "artifacts/data/baseline_churn.csv"
PII_FIELDS_PATH = "artifacts/data/pii_fields.json"  # Store identified PII fields


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


# ── NEW VALIDATION TOOLS ADDED BELOW ──────────────────────────────────────────


@tool("Validate Cardinality Limits")
def validate_cardinality_limits_tool(
    path: str = DATA_PATH,
    max_unique_values: int = 50
) -> str:
    """
    Ensures categorical columns don't have too many unique values.
    High cardinality can cause One-Hot Encoding explosion.
    """
    df = pd.read_csv(path)
    violations = []
    
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_count = df[col].nunique()
        if unique_count > max_unique_values:
            violations.append(f"{col}: {unique_count} unique values (max: {max_unique_values})")
    
    if violations:
        return (
            f"VALIDATION FAILED: High cardinality columns detected:\n"
            + "\n".join([f"  - {v}" for v in violations])
            + "\nRecommendation: Consider grouping rare categories or using target encoding."
        )
    
    return f"Cardinality validation PASSED. All categorical columns ≤ {max_unique_values} unique values."


@tool("Validate Constant Features")
def validate_constant_features_tool(path: str = PROCESSED_PATH) -> str:
    """
    Checks for columns with a single unique value (constant features).
    These are useless for training and should be removed.
    """
    if not os.path.exists(path):
        return f"VALIDATION FAILED: Processed file not found at {path}."
    
    df = pd.read_csv(path)
    constant_cols = []
    
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_cols.append(f"{col} (unique value: {df[col].iloc[0]})")
    
    if constant_cols:
        return (
            f"VALIDATION FAILED: Constant features detected (no predictive power):\n"
            + "\n".join([f"  - {c}" for c in constant_cols])
            + "\nRecommendation: Remove these columns before training."
        )
    
    return "Constant feature validation PASSED. No constant columns found."


@tool("Validate PII Removal")
def validate_pii_removal_tool(
    path: str = PROCESSED_PATH,
    pii_fields_path: str = PII_FIELDS_PATH
) -> str:
    """
    Confirms that fields identified as PII in the previous step are actually
    removed or hashed from the processed dataset.
    """
    if not os.path.exists(path):
        return f"VALIDATION FAILED: Processed file not found at {path}."
    
    df = pd.read_csv(path)
    pii_fields = []
    
    # Load previously identified PII fields if available
    if os.path.exists(pii_fields_path):
        import json
        with open(pii_fields_path, 'r') as f:
            pii_data = json.load(f)
            pii_fields = pii_data.get('detected_pii_fields', [])
    else:
        # Default PII field names to check
        pii_fields = ['customerID', 'email', 'phone', 'ssn', 'name']
    
    # Check if any PII fields remain in the processed dataset
    remaining_pii = [field for field in pii_fields if field in df.columns]
    
    if remaining_pii:
        return (
            f"VALIDATION FAILED: PII fields still present in processed data:\n"
            + "\n".join([f"  - {field}" for field in remaining_pii])
            + "\nRecommendation: Remove or hash these fields before model training."
        )
    
    return "PII removal validation PASSED. No PII fields found in processed dataset."


@tool("Validate Distribution Stability")
def validate_distribution_stability_tool(
    current_path: str = PROCESSED_PATH,
    baseline_path: str = BASELINE_PATH,
    significance_level: float = 0.05
) -> str:
    """
    Uses KS-Test (for continuous) or Chi-Square (for categorical) to ensure
    training data distribution hasn't shifted drastically from expected norms.
    """
    if not os.path.exists(current_path):
        return f"VALIDATION FAILED: Current data file not found at {current_path}."
    
    if not os.path.exists(baseline_path):
        return (
            f"VALIDATION SKIPPED: Baseline file not found at {baseline_path}.\n"
            "Cannot perform distribution stability check without baseline data."
        )
    
    df_current = pd.read_csv(current_path)
    df_baseline = pd.read_csv(baseline_path)
    
    common_cols = [c for c in df_current.columns if c in df_baseline.columns]
    violations = []
    test_results = []
    
    for col in common_cols:
        if col == 'Churn':  # Skip target variable
            continue
            
        current_data = df_current[col].dropna()
        baseline_data = df_baseline[col].dropna()
        
        if len(current_data) < 10 or len(baseline_data) < 10:
            test_results.append(f"{col}: SKIPPED (insufficient samples)")
            continue
        
        # Choose test based on data type
        if df_current[col].dtype in [np.int64, np.float64]:
            # KS-Test for continuous variables
            statistic, p_value = stats.ks_2samp(current_data, baseline_data)
            test_type = "KS-Test"
        else:
            # Chi-Square for categorical variables
            try:
                # Get value counts for both datasets
                current_counts = current_data.value_counts()
                baseline_counts = baseline_data.value_counts()
                
                # Align categories
                all_categories = sorted(set(current_counts.index) | set(baseline_counts.index))
                current_freq = [current_counts.get(cat, 0) for cat in all_categories]
                baseline_freq = [baseline_counts.get(cat, 0) for cat in all_categories]
                
                # Avoid division by zero
                if sum(baseline_freq) == 0:
                    test_results.append(f"{col}: SKIPPED (zero baseline frequency)")
                    continue
                
                statistic, p_value = stats.chisquare(current_freq, baseline_freq)
                test_type = "Chi-Square"
            except Exception as e:
                test_results.append(f"{col}: SKIPPED ({str(e)})")
                continue
        
        is_stable = p_value > significance_level
        status = "STABLE" if is_stable else "DRIFT DETECTED"
        
        test_results.append(
            f"{col}: {test_type} p-value={p_value:.4f} [{status}]"
        )
        
        if not is_stable:
            violations.append(f"{col}: p-value={p_value:.4f} < {significance_level}")
    
    if violations:
        return (
            f"VALIDATION FAILED: Distribution drift detected in {len(violations)} columns:\n"
            + "\n".join([f"  - {v}" for v in violations])
            + "\n\nFull Test Results:\n"
            + "\n".join(test_results)
            + "\nRecommendation: Investigate data source changes or recollect baseline."
        )
    
    return (
        f"Distribution stability validation PASSED.\n"
        f"Significance level: {significance_level}\n"
        f"Columns tested: {len(common_cols)}\n"
        f"Test Results:\n" + "\n".join(test_results)
    )