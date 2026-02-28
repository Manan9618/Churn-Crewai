"""
Centralized prompt definitions for all CrewAI agents and tasks
in the Customer Churn Prediction pipeline.

Pipeline Order (11 stages, each with agent task + validator task = 22 total tasks):
  1.  Data Understanding
  2.  Preprocessing
  3.  Feature Discovery
  4.  Processing        (Train/Test Split + SMOTE)
  5.  PyCaret Setup     (AutoML model comparison + tuning)
  6.  Model Training    (Final model training + evaluation)
  7.  Prediction        (Generate churn predictions)
  8.  Explanation       (SHAP + LIME interpretability)
  9.  Retention Strategy
  10. Summary           (Full pipeline report)
  11. Feedback Loop     (Continuous improvement)
"""

PROMPTS = {

    # =========================================================================
    # 1. DATA UNDERSTANDING
    # =========================================================================

    "data_understanding": {
        "task": (
            "Load the Customer Churn dataset from 'data/Customer_Churn.csv' and perform "
            "a thorough exploratory data analysis covering: "
            "(1) dataset shape and full column list with data types, "
            "(2) missing value counts and percentages per column - pay special attention "
            "to TotalCharges which contains whitespace entries that act as missing values, "
            "(3) Churn target class distribution with counts and percentages, "
            "(4) descriptive statistics (mean, std, min, max, quartiles) for numerical "
            "features: tenure, MonthlyCharges, TotalCharges, SeniorCitizen, "
            "(5) unique value counts for key categorical features: Contract, "
            "InternetService, PaymentMethod, "
            "(6) **PII compliance scan** - detect any columns or values containing Personally "
            "Identifiable Information such as emails, phone numbers, SSNs, or sensitive IDs. "
            "Summarise all findings in a structured report with key observations "
            "about data quality, compliance risks, and potential preprocessing needs."
        ),
        "expected_output": (
            "A structured EDA report containing: "
            "(1) dataset shape (rows x columns) and complete column list with dtypes, "
            "(2) missing value table showing count and percentage per column, "
            "(3) Churn class distribution - count and percentage for Yes and No, "
            "(4) descriptive statistics table for numerical features, "
            "(5) unique value summary for categorical features, "
            "(6) **PII detection results** - list any columns or patterns flagged as potential PII, "
            "(7) a bullet-point list of key data quality observations, compliance status, and "
            "recommended preprocessing actions."
        ),
    },

    "data_validation": {
        "task": (
            "Validate the raw Customer Churn dataset against expected quality standards: "
            "(1) Schema check - confirm all 21 expected columns are present: "
            "customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, "
            "MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, "
            "TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, "
            "PaymentMethod, MonthlyCharges, TotalCharges, Churn, "
            "(2) Data type check - confirm tenure and MonthlyCharges are numeric, "
            "TotalCharges is coercible to float (whitespace entries allowed), "
            "(3) Missing value threshold - no column should exceed 1 percent missing values "
            "after coercing TotalCharges to numeric, "
            "(4) Target integrity - the Churn column must contain only 'Yes' or 'No', "
            "(5) **Cardinality limits** - confirm no categorical column has more than 50 unique values "
            "(prevents one-hot encoding explosion), "
            "(6) **Distribution stability** - compare against baseline to detect significant drift "
            "using KS-Test or Chi-Square tests. "
            "Report PASSED or FAILED with specific details for each check."
        ),
        "expected_output": (
            "A validation report with clear PASSED or FAILED status for each of the "
            "6 checks: schema validation, data type validation, missing value threshold, "
            "target column integrity, cardinality limits, and distribution stability. "
            "Include specific details on any failures such as missing column names, invalid values, "
            "threshold violations, high-cardinality columns, or distribution drift."
        ),
    },

    # =========================================================================
    # 2. PREPROCESSING
    # =========================================================================

    "preprocessing": {
        "task": (
            "Preprocess the raw Customer Churn dataset into a clean, model-ready format: "
            "(1) Convert TotalCharges to numeric using pd.to_numeric with errors='coerce', "
            "then fill any resulting NaN values with the column median, "
            "(2) **Detect and remove duplicate records** to prevent data leakage, "
            "(3) **Handle outliers** using IQR method - cap values outside "
            "[Q1 - 1.5*IQR, Q3 + 1.5*IQR] for all numerical columns, "
            "(4) Drop the customerID column as it is a non-predictive identifier, "
            "(5) Encode binary categorical columns to 0/1: "
            "gender (Male=1, Female=0), Partner, Dependents, PhoneService, "
            "PaperlessBilling, Churn (Yes=1, No=0), "
            "(6) Encode multi-class categorical columns to integer codes: "
            "MultipleLines, InternetService, OnlineSecurity, OnlineBackup, "
            "DeviceProtection, TechSupport, StreamingTV, StreamingMovies, "
            "Contract, PaymentMethod, "
            "(7) Apply StandardScaler to numerical features: "
            "tenure, MonthlyCharges, TotalCharges - save the fitted scaler "
            "to 'model/scaler.pkl' for later inverse transforms, "
            "(8) **Optimize memory usage** by downcasting numeric types and converting "
            "low-cardinality object columns to categories. "
            "Save the fully processed dataset to 'data/processed_churn.csv'."
        ),
        "expected_output": (
            "A preprocessing completion report confirming: "
            "(1) TotalCharges converted and NaN values filled with median, "
            "(2) **duplicate records detected and removed** with count, "
            "(3) **outliers handled** with count of values capped per column, "
            "(4) customerID dropped successfully, "
            "(5) list of binary-encoded columns with their mapping, "
            "(6) list of multi-class encoded columns, "
            "(7) confirmation that StandardScaler was applied and scaler.pkl was saved, "
            "(8) **memory optimization results** showing reduction percentage, "
            "(9) final processed dataset shape saved to data/processed_churn.csv."
        ),
    },

    "preprocessing_validation": {
        "task": (
            "Validate the preprocessed dataset at 'data/processed_churn.csv': "
            "(1) Zero missing values - confirm df.isnull().sum().sum() == 0, "
            "(2) All numeric - confirm no object dtype columns remain in the dataset, "
            "(3) Scaling check - confirm that tenure, MonthlyCharges, and TotalCharges "
            "have a mean close to 0 (absolute mean < 0.1) after StandardScaler was applied, "
            "(4) Column presence - confirm customerID was dropped and Churn column exists "
            "with only 0 and 1 values, "
            "(5) **Constant features** - confirm no columns have only 1 unique value "
            "(useless for training), "
            "(6) **PII removal** - confirm no PII fields (customerID, email, phone, SSN) "
            "remain in the processed dataset. "
            "Report PASSED or FAILED with specific details for each check."
        ),
        "expected_output": (
            "A preprocessing validation report with PASSED/FAILED status for: "
            "(1) missing values check, "
            "(2) encoding completeness (all columns numeric), "
            "(3) scaling verification (near-zero mean for scaled columns), "
            "(4) column integrity check (no customerID, binary Churn), "
            "(5) **constant feature check** (list any constant columns found), "
            "(6) **PII removal verification** (confirm no sensitive fields remain). "
            "Report the final dataset shape and any anomalies found."
        ),
    },

    # =========================================================================
    # 3. FEATURE DISCOVERY
    # =========================================================================

    "feature_discovery": {
        "task": (
            "Identify the most predictive features for customer churn from the "
            "preprocessed dataset 'artifacts/data/processed_churn.csv': "
            "(1) Correlation analysis - compute Pearson correlation of every feature "
            "with the Churn target column, list all features with absolute correlation >= 0.05, "
            "(2) Random Forest importance - train a RandomForestClassifier "
            "(n_estimators=100, random_state=42) and extract feature_importances_, "
            "(3) Variance threshold filtering - use VarianceThreshold(threshold=0.01) "
            "to identify and remove near-constant features, "
            "(4) **Mutual Information** - calculate mutual information scores to capture "
            "non-linear dependencies that correlation misses, highlight features where "
            "MI is 2x+ higher than correlation, "
            "(5) **Generate interaction features** - create polynomial/interaction terms "
            "(e.g., Tenure * MonthlyCharges) for top 8 features, select top 10 interactions "
            "by correlation with target, save to 'artifacts/data/interaction_features.json', "
            "(6) **Cluster features** - group highly correlated features (threshold=0.85) "
            "using hierarchical clustering, select one representative per cluster to reduce "
            "multicollinearity, save to 'artifacts/data/clustered_features.json', "
            "(7) **Recursive Feature Elimination** - apply RFE with Random Forest estimator "
            "to iteratively remove least important features, find optimal subset of 10 features, "
            "report cross-validation AUC, save to 'artifacts/data/rfe_results.json', "
            "(8) Select and save - pick the top 12 features ranked by combined scores "
            "(RF importance + mutual information), save the list as a JSON array to "
            "'artifacts/data/selected_features.json'. "
            "Report the final selected features with their importance scores and provide "
            "a business explanation of why the top features drive churn."
        ),
        "expected_output": (
            "A feature discovery report containing: "
            "(1) correlation table showing features with absolute r >= 0.05 sorted by absolute r, "
            "(2) Random Forest feature importance ranking for all features, "
            "(3) variance threshold results (features removed vs kept), "
            "(4) **mutual information scores** for all features with top 15 highlighted, "
            "(5) **interaction features created** - list top 10 interactions with correlation scores, "
            "(6) **feature clusters** - show cluster assignments and representative features selected, "
            "(7) **RFE results** - optimal feature subset with CV AUC score, "
            "(8) final top-12 selected features with importance scores, "
            "(9) confirmation that all artifacts were saved: "
            "artifacts/data/selected_features.json, artifacts/data/interaction_features.json, "
            "artifacts/data/clustered_features.json, artifacts/data/rfe_results.json, "
            "artifacts/data/mutual_information.json, "
            "(10) brief business explanation of why the top features drive churn."
        ),
    },

    "feature_validation": {
        "task": (
            "Validate the feature selection output: "
            "(1) Feature count - confirm at least 5 features were selected "
            "(load artifacts/data/selected_features.json and check length), "
            "(2) No target leakage - confirm neither 'customerID' nor 'Churn' "
            "appear in the selected feature list, "
            "(3) Variance check - for each selected feature, confirm its variance "
            "in the processed dataset is >= 0.001, "
            "(4) File persistence - confirm artifacts/data/selected_features.json exists, "
            "is a valid JSON file, and contains a non-empty list, "
            "(5) **Multicollinearity check** - calculate VIF for all selected features, "
            "confirm no feature has VIF > 5.0 (high multicollinearity), "
            "(6) **Feature stability** - run 5-fold CV and confirm feature importance "
            "standard deviation < 0.3 across folds (unstable features indicate noise), "
            "(7) **Information Value** - calculate WOE/IV for all selected features, "
            "confirm all features have IV >= 0.02 (sufficient predictive power). "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A feature validation report with PASSED/FAILED status for: "
            "(1) minimum feature count check (report actual count), "
            "(2) target leakage check (list any leakage columns found), "
            "(3) feature variance check (list any low-variance features found), "
            "(4) file persistence and format check, "
            "(5) **multicollinearity VIF check** (report max VIF, avg VIF, and any "
            "features with VIF > 5.0), "
            "(6) **feature stability check** (report avg/std importance across CV folds "
            "and any unstable features with std > 0.3), "
            "(7) **information value check** (report IV scores, predictive power summary: "
            "Useless/Weak/Medium/Strong, and any features with IV < 0.02). "
            "Include the full list of validated selected features and a final "
            "FEATURES VALIDATED or FEATURES FAILED verdict."
        ),
    },

    # =========================================================================
    # 4. PROCESSING (TRAIN/TEST SPLIT + SMOTE)
    # =========================================================================

    "processing": {
        "task": (
            "Prepare the final train and test datasets for model training: "
            "(1) **Time series split** - check if data has temporal column (e.g., tenure, date), "
            "if yes, split respecting time order to prevent look-ahead bias; otherwise use "
            "stratified random split 80/20 with random_state=42 to preserve Churn class ratio. "
            "Save X_train, X_test, y_train, y_test as separate CSVs to artifacts/data/ directory, "
            "(2) **Apply sampling** - apply SMOTE or ADASYN (random_state=42) to X_train and y_train "
            "ONLY (never the test set) to balance the Churn classes in the training data. "
            "Overwrite X_train.csv and y_train.csv with the resampled versions, "
            "(3) Feature matrix validation - confirm X_train and X_test have identical "
            "column names and counts, "
            "(4) **Version dataset** - tag the processed dataset with a version ID (e.g., v20240115_143022) "
            "and generate MD5 hash for reproducibility tracking, save to "
            "'artifacts/data/dataset_version_log.json', "
            "(5) **Serialize preprocessor** - save the fitted StandardScaler and encoder mapping "
            "separately to 'artifacts/model/scaler.pkl' and 'artifacts/model/encoder.pkl' "
            "for use in real-time inference APIs, "
            "(6) Save pipeline state - persist current stage metadata including file paths, "
            "split sizes, feature count, and version tag to 'artifacts/data/pipeline_state.json'. "
            "Report split method, train/test sizes, class distribution before/after sampling, "
            "version tag, and preprocessor paths."
        ),
        "expected_output": (
            "A processing report confirming: "
            "(1) split method used (time-series or stratified random) with exact row counts, "
            "(2) train and test churn rates before and after sampling, "
            "(3) **sampling method applied** (SMOTE or ADASYN) with before/after class distribution "
            "and imbalance ratio, "
            "(4) feature column alignment confirmed between train and test, "
            "(5) **dataset version tag** assigned (e.g., v20240115_143022) with MD5 hash, "
            "(6) **preprocessor serialized** - scaler path, encoder path, columns covered, "
            "(7) pipeline_state.json saved with stage metadata. "
            "Include train churn rate, test churn rate, total samples after resampling, "
            "version tag, and confirmation that preprocessors are ready for inference API deployment."
        ),
    },

    "processing_validation": {
        "task": (
            "Validate the train/test split and sampling processing outputs: "
            "(1) Split files exist - confirm all 4 files exist and are non-empty: "
            "artifacts/data/X_train.csv, artifacts/data/X_test.csv, artifacts/data/y_train.csv, "
            "artifacts/data/y_test.csv, "
            "(2) Split ratio - confirm the test set is approximately 20 percent of total data "
            "within a tolerance of plus or minus 2 percent, "
            "(3) No data leakage - check that zero rows are duplicated between "
            "X_train and X_test (no overlapping customer records), "
            "(4) Stratification - confirm Churn rate in y_test matches y_train "
            "within plus or minus 3 percent tolerance, "
            "(5) Feature alignment - confirm X_train and X_test have identical columns "
            "in the same order, "
            "(6) Pipeline state - confirm pipeline_state.json exists with all files present, "
            "(7) **Target distribution per fold** - perform 5-fold stratified CV and confirm "
            "each fold has representative churn rate (std dev ≤ 5%, min 30 samples per fold), "
            "(8) **Pipeline determinism** - verify random seeds are documented and output file "
            "hashes are recorded for reproducibility tracking, "
            "(9) **Memory footprint** - confirm all artifacts fit within memory limits "
            "(≤16GB total with 20% safety margin for in-memory operations). "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A processing validation report with PASSED/FAILED status for all 9 checks: "
            "(1) split file existence and non-empty check, "
            "(2) split ratio check with actual percentages, "
            "(3) data leakage check (overlapping row count), "
            "(4) stratification check with train vs test churn rates, "
            "(5) feature alignment check, "
            "(6) pipeline state check, "
            "(7) **target distribution per fold** (report churn rate per fold, std dev, "
            "and variance status), "
            "(8) **pipeline determinism** (report file hashes, seed documentation status), "
            "(9) **memory footprint** (report total file size, estimated in-memory size, "
            "and limit compliance). "
            "Include final train and test sizes, and a final "
            "PROCESSING VALIDATED or PROCESSING FAILED verdict."
        ),
    },

    # =========================================================================
    # 5. PYCARET SETUP
    # =========================================================================

    "pycaret_setup": {
        "task": (
            "Run an AutoML model comparison and selection using PyCaret: "
            "(1) PyCaret setup - initialise the classification experiment using the "
            "processed dataset with target='Churn', fix_imbalance=True (SMOTE), "
            "session_id=42, verbose=False, html=False, "
            "(2) Compare models - run compare_models(sort='AUC', n_select=5) to rank "
            "at least 5 classifiers. Display the leaderboard with AUC, F1, Recall, "
            "Precision columns. Save the results to 'artifacts/model/pycaret_compare_results.csv', "
            "(3) **Get leaderboard** - retrieve the full model leaderboard with all metrics "
            "for comprehensive analysis, save to 'artifacts/model/pycaret_full_leaderboard.csv', "
            "(4) Tune best model - call tune_model() on the top-ranked model with "
            "n_iter=50 and optimize='AUC', "
            "(5) **Blend models** - combine top 5 models using blend_models() to reduce "
            "variance and improve generalization, save to 'artifacts/model/pycaret_blend_model.pkl', "
            "(6) **Stack models** - create stacked ensemble with stack_models() using "
            "top 3 base models and logistic regression meta-model for meta-learning, "
            "save to 'artifacts/model/pycaret_stack_model.pkl', "
            "(7) **Deploy API** - create FastAPI endpoint using deploy_model() for "
            "real-time churn predictions on port 8000, "
            "(8) Save pipeline - call save_model() to persist the full PyCaret pipeline "
            "including preprocessing to 'artifacts/model/pycaret_best_model.pkl'. "
            "Report the leaderboard, ensemble results, API deployment status, and all saved artifacts."
        ),
        "expected_output": (
            "A PyCaret experiment report containing: "
            "(1) leaderboard table showing all models ranked by AUC with "
            "AUC, F1, Recall, Precision values, "
            "(2) name of the best-performing model with tuned metrics, "
            "(3) **full model leaderboard** saved to CSV with all trained models, "
            "(4) tuned model cross-validation metrics (AUC, F1, Recall), "
            "(5) **blended model results** - list of models included, save path, "
            "(6) **stacked model results** - base models, meta model, save path, "
            "(7) **API deployment status** - endpoint URLs, port, deployment report path, "
            "(8) confirmation that all artifacts were saved: "
            "artifacts/model/pycaret_compare_results.csv, "
            "artifacts/model/pycaret_full_leaderboard.csv, "
            "artifacts/model/pycaret_best_model.pkl, "
            "artifacts/model/pycaret_blend_model.pkl, "
            "artifacts/model/pycaret_stack_model.pkl. "
            "Include model comparison summary and deployment readiness status."
        ),
    },

    "pycaret_validation": {
        "task": (
            "Validate the PyCaret AutoML experiment outputs: "
            "(1) Setup pre-conditions - confirm artifacts/data/processed_churn.csv exists, "
            "contains the Churn column, and has no non-numeric columns, "
            "(2) Leaderboard quality - load artifacts/model/pycaret_compare_results.csv and "
            "confirm the best model achieves AUC >= 0.75 and F1 >= 0.55, "
            "(3) Model persistence - confirm artifacts/model/pycaret_best_model.pkl exists "
            "and is loadable without errors, "
            "(4) **Deployment health** - ping the deployed API endpoint at the URL specified "
            "in artifacts/model/pycaret_api_deployment.json to confirm it returns 200 OK "
            "on /health and /predict endpoints (timeout: 10s), "
            "(5) **Inference consistency** - compare PyCaret predictions against raw sklearn "
            "predictions on test data to ensure wrapper isn't breaking logic "
            "(tolerance: 0.0001, samples: 10). "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A PyCaret validation report with PASSED/FAILED status for all 5 checks: "
            "(1) data readiness pre-conditions (report data shape, column status), "
            "(2) leaderboard quality thresholds (report best AUC and F1 values, model count), "
            "(3) model file persistence and loadability (report file size), "
            "(4) **deployment health check** (report API URL, endpoint status codes, "
            "response times for /health and /predict), "
            "(5) **inference consistency check** (report samples tested, label match status, "
            "max score difference, tolerance compliance). "
            "Include the best model name, its AUC/F1 scores, API health status, "
            "consistency status, and a final PYCARET PIPELINE VALIDATED or "
            "PYCARET PIPELINE FAILED verdict."
        ),
    },

    # =========================================================================
    # 6. MODEL TRAINING
    # =========================================================================

    "model_training": {
        "task": (
            "Train, evaluate, and optimize the final churn prediction model: "
            "(1) Train model - load X_train/y_train from artifacts/data/, train a "
            "GradientBoostingClassifier(n_estimators=200, random_state=42) "
            "on the selected features from artifacts/data/selected_features.json, "
            "(2) Cross-validation - run 5-fold StratifiedKFold cross-validation "
            "and report mean +/- std for AUC, F1, and Recall, "
            "(3) Test set evaluation - evaluate on X_test/y_test, produce the full "
            "sklearn classification_report, compute AUC from predict_proba, "
            "(4) **Tune classification threshold** - find optimal probability threshold "
            "(not 0.5) that maximizes F1 score with recall >= 0.60, save to "
            "'artifacts/model/optimal_threshold.json', "
            "(5) **Calibrate probabilities** - apply isotonic regression to ensure "
            "predicted probabilities match actual churn rates, save calibrated model "
            "to 'artifacts/model/churn_model_calibrated.pkl', "
            "(6) **Create ensemble** - combine top 3 models (RandomForest, GradientBoosting, "
            "LogisticRegression) using soft voting to reduce variance, save to "
            "'artifacts/model/churn_model_ensemble.pkl', "
            "(7) **Export ONNX** - save model in ONNX format for deployment in "
            "non-Python environments, save to 'artifacts/model/churn_model.onnx', "
            "(8) **Measure inference latency** - time 100 predictions to ensure P95 "
            "latency < 100ms for SLA compliance, "
            "(9) Save artifacts - save the trained model to 'artifacts/model/churn_model.pkl'. "
            "Report all metrics, optimization results, and deployment readiness."
        ),
        "expected_output": (
            "A comprehensive model training report with: "
            "(1) 5-fold CV results: AUC mean +/- std, F1 mean +/- std, Recall mean +/- std, "
            "(2) test set metrics: AUC, Accuracy, Precision, Recall, F1, "
            "(3) full classification report with per-class precision/recall/f1/support, "
            "(4) **optimal threshold** with improvement over default 0.5 (F1, Recall, Precision deltas), "
            "(5) **calibration report** with Brier score before/after and calibration error, "
            "(6) **ensemble performance** vs individual models (AUC, F1 comparison), "
            "(7) **ONNX export status** (file size, version, deployment options), "
            "(8) **inference latency** (mean, P95, P99, SLA compliance status), "
            "(9) confirmation that model/churn_model.pkl was saved with file size. "
            "Highlight whether the model meets minimum thresholds: AUC >= 0.75, F1 >= 0.55, "
            "Recall >= 0.60, Latency P95 < 100ms."
        ),
    },

    "model_validation": {
        "task": (
            "Validate the trained churn model against quality thresholds: "
            "(1) Performance thresholds - load the model, evaluate on X_test/y_test, "
            "confirm test AUC >= 0.75 and F1 >= 0.55, "
            "(2) Overfitting check - compare train AUC vs test AUC, "
            "confirm the gap is <= 0.05, "
            "(3) Model file integrity - confirm model/churn_model.pkl exists "
            "and can be loaded via pickle without errors, "
            "(4) Churn recall - confirm recall on the positive (Churn=1) class >= 0.60, "
            "since missing actual churners is costly for the business, "
            "(5) **Business metrics** - confirm model generates positive profit per "
            "customer (revenue from retained customers > cost of campaigns + missed churners), "
            "(6) **Adversarial robustness** - confirm accuracy drop < 5% when inputs are "
            "perturbed with noise (simulates real-world data quality issues), "
            "(7) **Fairness disparity** - confirm False Positive Rate difference < 10% "
            "across protected groups (gender, SeniorCitizen), "
            "(8) **Probability distribution** - confirm predictions are well-calibrated "
            "(entropy >= 0.5, overconfidence ratio <= 80%). "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A model validation report with PASSED/FAILED status for: "
            "(1) AUC and F1 threshold checks with actual values, "
            "(2) overfitting check with train AUC, test AUC, and gap, "
            "(3) model file integrity check, "
            "(4) churn class recall check with actual recall value, "
            "(5) **business metrics check** (report profit per customer, ROI, and "
            "cost breakdown), "
            "(6) **adversarial robustness check** (report accuracy retention under "
            "perturbation), "
            "(7) **fairness disparity check** (report FPR for each demographic group "
            "and disparity percentage), "
            "(8) **probability distribution check** (report entropy, confidence ratio, "
            "and distribution bins). "
            "Include a final overall MODEL APPROVED or MODEL REJECTED verdict for deployment."
        ),
    },

    # =========================================================================
    # 7. PREDICTION
    # =========================================================================

    "prediction": {
        "task": (
            "Generate churn predictions for all customers in the dataset: "
            "(1) **Check data drift** - compare current data distribution against baseline "
            "at 'data/baseline_churn.csv' to detect significant distribution shifts "
            "(mean shift > 10% triggers warning), "
            "(2) Load model - load the trained model from 'model/churn_model.pkl' "
            "and the selected features from 'data/selected_features.json', "
            "(3) Load data - load 'data/processed_churn.csv' and select only the "
            "feature columns for inference, "
            "(4) Predict - run model.predict() for binary labels (0/1) and "
            "model.predict_proba()[:,1] for calibrated churn probability scores, "
            "(5) Annotate and save - add columns 'Churn_Predicted' and "
            "'Churn_Probability' (rounded to 4 decimal places) to the dataframe "
            "and save to 'data/predictions.csv'. "
            "Report the predicted churn rate, drift status, and a preview of results."
        ),
        "expected_output": (
            "A prediction completion report showing: "
            "(1) **data drift check results** - status (stable/drift detected) and "
            "mean shift percentages per numerical column, "
            "(2) total number of customers scored, "
            "(3) count and percentage of predicted churners vs non-churners, "
            "(4) churn probability distribution summary (min, mean, median, max), "
            "(5) confirmation that data/predictions.csv was saved with "
            "Churn_Predicted and Churn_Probability columns, "
            "(6) a 5-row preview of the predictions output."
        ),
    },

    "prediction_validation": {
        "task": (
            "Validate the prediction output file 'artifacts/data/predictions.csv': "
            "(1) File completeness - confirm the file exists, is non-empty, "
            "and has zero missing values in Churn_Predicted and Churn_Probability, "
            "(2) Required columns - confirm both Churn_Predicted and Churn_Probability "
            "columns are present in the output file, "
            "(3) Probability range - confirm all Churn_Probability values are "
            "within [0.0, 1.0] with no out-of-range scores, "
            "(4) Churn rate sanity - confirm the predicted churn rate (mean of "
            "Churn_Predicted) falls within the expected 10 to 35 percent range "
            "typical for telecom churn datasets, "
            "(5) **Drift warning check** - if data drift was detected during prediction, "
            "confirm a warning was logged and flagged in the prediction report. "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A prediction validation report with PASSED/FAILED status for: "
            "(1) file completeness and missing values check (report shape and missing count), "
            "(2) required column presence check (list all columns found), "
            "(3) probability range check (report any out-of-range count), "
            "(4) churn rate sanity check with actual predicted churn rate percentage, "
            "(5) **data drift warning verification** (report drift status: STABLE or DRIFT DETECTED). "
            "Include total prediction count, predicted churn rate percentage, drift status, "
            "and a final PREDICTIONS VALIDATED or PREDICTIONS FAILED verdict."
        ),
    },

    # =========================================================================
    # 8. EXPLANATION (SHAP + LIME)
    # =========================================================================
    "explanation": {
        "task": (
            "Generate comprehensive model explanations using multiple interpretability techniques: "
            "(1) Global SHAP importance - use shap.TreeExplainer on the trained model, "
            "compute SHAP values on the test set (use all available samples if less than 500), "
            "calculate mean absolute SHAP value per feature to produce global importance ranking. "
            "Save SHAP values to 'artifacts/model/shap_values.pkl', "
            "(2) SHAP force plots - generate SHAP explanations for available high-risk customers. "
            "IMPORTANT: Check how many test samples exist first. If test set has only 1 sample, "
            "use customer_index=0. If 2+ samples exist, analyze up to 2 highest-risk customers. "
            "Do NOT use invalid indices - always verify the test set size first. "
            "(3) LIME local explanation - generate a LIME explanation for the highest-risk "
            "available customer (use customer_index=0 if only 1 sample exists) showing top 5-8 feature contributions, "
            "(4) **Counterfactual analysis** - generate what-if scenarios for 2-3 high-risk customers "
            "showing the minimal feature changes needed to flip their churn prediction (e.g., 'If tenure "
            "was 5 months higher, churn prediction would change from 1 to 0'). Save to "
            "'artifacts/model/counterfactuals.json', "
            "(5) **Explanation stability check** - run SHAP multiple times with slight perturbations "
            "(10 iterations) to verify explanations aren't random. Calculate stability score (correlation) "
            "for 2 customers. Stability score ≥0.7 indicates reliable explanations. "
            "(6) **Fairness bias analysis** - check if SHAP values and predictions show bias against "
            "protected groups (Gender, SeniorCitizen). Flag any group with >10% churn rate disparity "
            "or >15% recall difference. Save to 'artifacts/model/fairness_report.json', "
            "(7) **Global logic rules** - extract business-friendly If-Then rules from the model "
            "(minimum 5 rules with confidence scores and support counts). These rules should be "
            "actionable for business stakeholders. Save to 'artifacts/model/global_logic_rules.json', "
            "(8) Save report - compile top SHAP importances, top churn drivers, fairness status, "
            "and stability scores into 'artifacts/model/explanation_report.json'. "
            "Provide a business-friendly narrative of the key churn drivers, fairness status, "
            "explanation reliability, and actionable retention rules."
        ),
        "expected_output": (
            "A comprehensive explainability report containing: "
            "(1) global feature importance table (mean absolute SHAP) ranked from most to least important, "
            "(2) SHAP explanations for available high-risk customers (adjust based on test set size), "
            "(3) LIME explanation for the highest-risk available customer with top 5-8 features, "
            "(4) **counterfactual analysis** for 2-3 customers showing minimal changes to flip predictions "
            "(include original vs counterfactual probability and specific feature changes), "
            "(5) **explanation stability scores** for 2 customers (must report mean correlation and status: "
            "HIGHLY STABLE/STABLE/MODERATE/UNSTABLE), "
            "(6) **fairness bias report** with group metrics for Gender and SeniorCitizen, "
            "including churn rate per group and any detected disparities (status: FAIR or POTENTIAL BIAS DETECTED), "
            "(7) **global logic rules** (minimum 5 If-Then rules with confidence % and support count), "
            "(8) confirmation that all artifacts were saved: "
            "artifacts/model/shap_values.pkl, artifacts/model/explanation_report.json, "
            "artifacts/model/counterfactuals.json, artifacts/model/fairness_report.json, "
            "artifacts/model/global_logic_rules.json, "
            "(9) a business narrative explaining: "
            "  - Top 5 churn drivers in plain English, "
            "  - Explanation reliability status (based on stability scores), "
            "  - Fairness status (any bias concerns or all clear), "
            "  - Top 3 actionable retention rules from the logic extraction."
        ),
    },

    "explanation_validation": {
        "task": (
            "Validate the explainability outputs for correctness, reliability, and domain alignment: "
            "(1) SHAP integrity - load 'artifacts/model/shap_values.pkl', confirm the SHAP matrix "
            "shape matches (n_samples x n_features), and verify there are zero NaN values, "
            "(2) Domain feature presence - confirm that at least 2 of the known domain-important "
            "features (Contract, tenure, MonthlyCharges, TotalCharges, InternetService) "
            "appear in the top 5 SHAP features by mean absolute importance, "
            "(3) **SHAP-LIME consistency** - confirm at least 2 out of 3 top features match "
            "between SHAP and LIME explanations (agreement ≥67%), "
            "(4) **Fairness metrics** - confirm no protected group (Gender, SeniorCitizen) has "
            ">10% churn rate disparity or >15% recall difference, "
            "(5) **Counterfactual feasibility** - confirm all suggested counterfactual changes "
            "are realistic (no immutable attributes changed, no negative values, feasible deltas), "
            "(6) Report completeness - load 'artifacts/model/explanation_report.json' and confirm "
            "it contains the keys: 'global_feature_importance_shap', 'top_churn_drivers', "
            "'test_auc', 'test_f1'. "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "An explainability validation report with PASSED/FAILED status for: "
            "(1) SHAP matrix shape and NaN check (report actual shape), "
            "(2) domain feature presence in top SHAP features "
            "(list which expected features appeared in top 5), "
            "(3) **SHAP-LIME consistency check** (report agreement % and common features), "
            "(4) **fairness metrics check** (report any disparities found with percentages, "
            "status: FAIR or BIAS DETECTED), "
            "(5) **counterfactual feasibility check** (report feasibility rate % and any "
            "infeasible changes found), "
            "(6) explanation report key completeness check. "
            "Include the actual top 5 SHAP features found, fairness status, counterfactual "
            "feasibility rate, and a final EXPLANATION VALIDATED or EXPLANATION FAILED verdict."
        ),
    },

    # =========================================================================
    # 9. RETENTION STRATEGY
    # =========================================================================

    "retention_strategy": {
        "task": (
            "Design a data-driven customer retention strategy based on churn predictions: "
            "(1) Segment customers - load 'artifacts/data/predictions.csv' and segment all customers "
            "by Churn_Probability into: High Risk (probability > 0.70), Medium Risk (probability "
            "0.40 to 0.70), Low Risk (probability < 0.40), "
            "(2) Assign retention offers - map each segment to personalised retention offers: "
            "High Risk: 2-year contract upgrade with 20% discount plus free TechSupport bundle, "
            "Medium Risk: 1-year contract switch with 10% discount plus free streaming add-on, "
            "Low Risk: loyalty rewards programme invitation, "
            "(3) Compute priority scores - add a Priority_Score column (0 to 100) equal to "
            "Churn_Probability multiplied by 100, sort customers descending by priority, "
            "(4) **Estimate campaign cost** - calculate total cost of all proposed retention offers "
            "against budget ($10,000 default), report budget utilization and ROI estimate "
            "(assume 30% retention success rate, $500 customer lifetime value), "
            "(5) **Assign communication channels** - recommend Email, SMS, or Call for each customer "
            "based on preferences (PaperlessBilling → Email, PhoneService → SMS, SeniorCitizen → Call), "
            "include channel cost and effectiveness score, "
            "(6) **Generate A/B test groups** - split retention list into Control (20%, no intervention) "
            "and Treatment (80%, receives offers) groups with stratified sampling by risk segment "
            "for campaign effectiveness measurement, "
            "(7) **Optimize budget allocation** - use ROI-based greedy optimization to select customers "
            "that maximize expected retained customers within budget constraint, ensure minimum "
            "10 customers per segment, "
            "(8) Save retention plan - save all predicted churners (Churn_Predicted=1) with their "
            "segment, offer, priority score, communication channel, test group, and optimization "
            "status to 'artifacts/data/retention_plan.csv'. Provide a business summary of the strategy."
        ),
        "expected_output": (
            "A retention strategy report containing: "
            "(1) customer count per risk segment (High/Medium/Low Risk) with percentages, "
            "(2) retention offer assigned to each segment with cost per offer, "
            "(3) priority score distribution (top 10 highest-priority customers), "
            "(4) **campaign cost estimate** - total cost, budget utilization %, ROI estimate, "
            "budget remaining or overage amount, "
            "(5) **communication channel distribution** - count and % for Email/SMS/Call, "
            "total channel cost, average effectiveness score, "
            "(6) **A/B test group sizes** - Control vs Treatment counts with segment distribution "
            "in each group, measurement plan (primary metric, test duration), "
            "(7) **optimized budget allocation** - customers selected, budget used, expected retained "
            "customers, expected revenue, ROI %, segment breakdown, "
            "(8) confirmation that artifacts/data/retention_plan.csv was saved with all fields "
            "(segment, offer, priority, channel, test_group, optimized), "
            "(9) confirmation that artifacts/data/retention_plan_optimized.csv was saved with "
            "optimized customer selection, "
            "(10) a business summary paragraph recommending how the CRM team should action each "
            "segment with specific outreach instructions."
        ),
    },

    "retention_validation": {
        "task": (
            "Validate the retention strategy plan at 'artifacts/data/retention_plan.csv': "
            "(1) File existence - confirm retention_plan.csv exists and is non-empty, "
            "(2) Segment integrity - confirm Risk_Segment column exists and all values "
            "are one of: High Risk, Medium Risk, Low Risk, "
            "(3) Offer coverage - confirm every customer with Churn_Probability > 0.70 "
            "has a non-null Retention_Offer assigned, "
            "(4) Priority score range - confirm all Priority_Score values are in [0, 100] "
            "with no nulls or out-of-range values, "
            "(5) **Budget compliance** - confirm total campaign cost is within budget "
            "($10,000 default), report total cost, budget utilization %, and remaining budget, "
            "(6) **Ethical compliance** - confirm offers don't discriminate across "
            "demographics (gender, SeniorCitizen, Partner, Dependents) with disparity < 10%, "
            "(7) **Coverage rate** - confirm at least 80% of High Risk customers are "
            "included in the retention plan. "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A retention plan validation report with PASSED/FAILED status for: "
            "(1) file existence and non-empty check (report customer count), "
            "(2) segment label integrity check (report segment distribution), "
            "(3) offer coverage for high-risk customers (report count with offers), "
            "(4) priority score range validation (report min/max scores), "
            "(5) **budget compliance check** (report total cost, budget limit, utilization %, "
            "remaining or overage amount), "
            "(6) **ethical compliance check** (report any discriminatory patterns found, "
            "disparity percentages per attribute), "
            "(7) **coverage rate check** (report High Risk coverage % vs minimum required). "
            "Include total at-risk customer count, breakdown per segment, and a final "
            "RETENTION PLAN VALIDATED or RETENTION PLAN FAILED verdict."
        ),
    },

    # =========================================================================
    # 10. SUMMARY
    # =========================================================================

    "summary": {
        "task": (
            "Compile a comprehensive end-to-end pipeline summary report: "
            "(1) Data summary - report raw dataset shape, feature count, and original "
            "churn rate. Report processed dataset shape and selected feature list, "
            "(2) Model performance - report the final model type, test AUC, F1, "
            "Recall, Precision. Include PyCaret best model name and AUC if available, "
            "(3) Prediction summary - report total customers scored, predicted churner "
            "count, predicted churn rate percent, and Churn_Probability distribution "
            "(min, median, mean, max), "
            "(4) Retention summary - report total at-risk customers, customer count per "
            "risk segment (High/Medium/Low), and number of customers with offers assigned, "
            "(5) **Executive dashboard** - generate JSON structure optimized for frontend "
            "dashboard libraries (Chart.js, D3.js, Plotly) with KPI cards, charts data, "
            "and alert status, save to 'artifacts/model/executive_dashboard.json', "
            "(6) **Alert messages** - create Slack/Email alert text for critical threshold "
            "breaches (AUC < 0.75, F1 < 0.55, budget exceeded, coverage < 80%), "
            "save to 'artifacts/model/alert_message.txt', "
            "(7) **Stakeholder summary** - generate non-technical summary for business "
            "managers that hides technical metrics and focuses on business impact, ROI, "
            "and actionable recommendations, save to 'artifacts/model/stakeholder_summary.md', "
            "(8) **Artifact links** - generate clickable links to all model files, logs, "
            "and data versions with file hashes for auditing, save to "
            "'artifacts/model/artifact_links.json', "
            "(9) Generate full report - compile all sections into 'artifacts/model/pipeline_summary_report.json' "
            "and 'artifacts/model/pipeline_summary_report.md'. "
            "Deliver a business-friendly executive summary paragraph."
        ),
        "expected_output": (
            "A complete pipeline summary report containing: "
            "(1) data overview (raw rows, features, churn rate, selected features), "
            "(2) model performance table (model type, AUC, F1, Recall, Precision), "
            "(3) prediction statistics (total scored, churner count, churn rate percent, "
            "probability distribution), "
            "(4) retention strategy overview (at-risk count by segment), "
            "(5) **executive dashboard JSON** (KPI cards, charts data, alerts, overall status), "
            "(6) **alert messages** (Slack and Email format with severity levels), "
            "(7) **stakeholder summary** (executive overview, key findings, financial impact, "
            "recommendations, next steps - all non-technical), "
            "(8) **artifact links** (clickable URLs to all pipeline files with hashes), "
            "(9) confirmation that all reports were saved: "
            "artifacts/model/pipeline_summary_report.json, "
            "artifacts/model/pipeline_summary_report.md, "
            "artifacts/model/executive_dashboard.json, "
            "artifacts/model/alert_message.txt, "
            "artifacts/model/stakeholder_summary.md, "
            "artifacts/model/artifact_links.json, "
            "(10) a 3 to 5 sentence executive summary paragraph for business stakeholders."
        ),
    },

    "summary_validation": {
        "task": (
            "Validate the final pipeline summary report outputs: "
            "(1) File existence - confirm both 'artifacts/model/pipeline_summary_report.json' "
            "and 'artifacts/model/pipeline_summary_report.md' exist and are non-empty, "
            "(2) Section completeness - load the JSON report and confirm all 6 required "
            "sections are present: data, feature_selection, model_performance, "
            "predictions, explanability, retention_strategy, "
            "(3) Model metric threshold - confirm the AUC value in the model_performance "
            "section is >= 0.75, "
            "(4) Prediction stat range - confirm the churn_rate_pct in the predictions "
            "section is between 10.0 and 35.0 percent, "
            "(5) Retention section - confirm at-risk customers > 0 with valid segment labels, "
            "(6) Markdown readability - confirm all major headings are present in the "
            "Markdown report, "
            "(7) **Link integrity** - verify all hyperlinks in reports resolve correctly "
            "(check artifact_links.json and Markdown links, no broken links), "
            "(8) **Data freshness** - confirm summary timestamp is within 24 hours "
            "(not running on stale data, check all key artifacts), "
            "(9) **Sensitive data redaction** - scan all summary reports for PII leakage "
            "(emails, phone numbers, SSNs, credit card numbers). "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A summary validation report with PASSED/FAILED status for all 9 checks: "
            "(1) file existence and non-empty check (report both file paths), "
            "(2) section completeness check (list all 6 sections), "
            "(3) model AUC threshold check (report actual AUC value), "
            "(4) prediction churn rate range check (report actual rate), "
            "(5) retention section check (report at-risk count and segments), "
            "(6) Markdown readability check (report word count and headings), "
            "(7) **link integrity check** (report total links, working, broken count), "
            "(8) **data freshness check** (report summary age in hours, artifact ages), "
            "(9) **sensitive data redaction check** (report PII types found or none). "
            "Include a final PIPELINE COMPLETE or PIPELINE INCOMPLETE verdict."
        ),
    },

    # =========================================================================
    # 11. FEEDBACK LOOP
    # =========================================================================

    "feedback_loop": {
        "task": (
            "Analyze the full pipeline results and generate a continuous improvement plan: "
            "(1) Log experiment - call log_experiment_tool to record the current run's "
            "AUC, F1, Recall, model type, and timestamp to 'artifacts/model/experiment_log.json', "
            "(2) Compare runs - call compare_metrics_tool to compare current AUC against "
            "previous best and compute the improvement delta, "
            "(3) **Trigger human review** - call trigger_human_in_the_loop_tool to check if "
            "metrics degraded >5% from previous run; if so, flag for manual review before "
            "proceeding, "
            "(4) **Calculate business cost impact** - call calculate_business_cost_impact_tool "
            "to translate metric changes into estimated dollar value (false negative cost, "
            "false positive cost, net ROI), "
            "(5) **Generate root cause hypothesis** - call generate_root_cause_hypothesis_tool "
            "to analyze logs and suggest why performance changed (data drift, feature changes, "
            "model change, etc.), "
            "(6) Generate improvement suggestions - call suggest_improvements_tool to produce "
            "at least 3 specific, actionable recommendations from these categories: "
            "a) Feature engineering: suggest new derived features such as tenure_group, "
            "monthly_to_total_charge_ratio, or num_services_subscribed, "
            "b) Model tuning: suggest hyperparameter changes or alternative algorithms "
            "such as XGBoost, LightGBM, or adjusting the decision threshold from 0.5 to 0.4, "
            "c) Sampling strategy: suggest SMOTE variants or cost-sensitive learning "
            "if recall on the churn class is below 0.65, "
            "(7) **Rollback capability** - if new run is worse than previous, call "
            "rollback_to_previous_version_tool to revert artifacts to last known good state, "
            "(8) Update feature list - if new features are recommended, call "
            "update_feature_list_tool to add them to 'artifacts/data/selected_features.json'. "
            "Provide a structured improvement roadmap."
        ),
        "expected_output": (
            "A feedback and improvement report containing: "
            "(1) current experiment run metrics (AUC, F1, Recall, model type), "
            "(2) comparison with previous runs showing AUC delta (or First Run if none), "
            "(3) **human review status** (TRIGGERED or NOT TRIGGERED with degradation details), "
            "(4) **business cost impact** (confusion matrix, cost breakdown, net ROI in dollars), "
            "(5) **root cause hypotheses** with confidence levels (HIGH/MEDIUM/LOW) and "
            "evidence for each hypothesis, "
            "(6) at least 3 specific improvement recommendations each with: "
            "the recommendation, the reasoning, and the expected impact, "
            "(7) **rollback status** (SUCCESS/PENDING/NOT_NEEDED with artifacts restored), "
            "(8) updated feature list if new features were added, "
            "(9) a prioritised roadmap stating what the next iteration should focus on "
            "listing improvements in order of expected impact. "
            "All reports saved to artifacts/model/ directory."
        ),
    },

        "feedback_validation": {
        "task": (
            "Validate the feedback loop outputs for completeness and quality: "
            "(1) Experiment log integrity - confirm 'artifacts/model/experiment_log.json' exists, "
            "is valid JSON, and the latest entry contains all required fields: "
            "run_id, timestamp, auc, f1, recall, "
            "(2) Improvement suggestions prerequisites - confirm that model/churn_model.pkl, "
            "artifacts/data/X_test.csv, and artifacts/data/y_test.csv all exist so suggestions "
            "can be generated, "
            "(3) Metrics improvement check - if 2 or more experiment runs exist, confirm "
            "the latest run improved AUC by at least 0.01 over the previous run. "
            "If only 1 run exists, report this check as SKIPPED not FAILED, "
            "(4) **Statistical significance** - perform T-Test to confirm improvements aren't "
            "due to random variance (p-value < 0.05 required for significance), "
            "(5) **Resource constraints** - confirm model fits within production budgets: "
            "features ≤50, estimated memory ≤16GB, estimated training time ≤60 minutes, "
            "AUC ≥0.75 for deployment, "
            "(6) **Rollback integrity** - if rollback was performed (check rollback_log.json), "
            "verify all artifacts were restored correctly with file checksums and metadata. "
            "Report PASSED, SKIPPED, or FAILED for each check."
        ),
        "expected_output": (
            "A feedback validation report with PASSED/SKIPPED/FAILED status for: "
            "(1) experiment log existence and required field check (report total runs), "
            "(2) improvement suggestion prerequisites check (list any missing files), "
            "(3) inter-run AUC improvement check (with actual delta or SKIPPED reason), "
            "(4) **statistical significance check** (report T-statistic, p-value, and "
            "significance status for AUC/F1/Recall), "
            "(5) **resource constraints check** (report actual usage vs limits for features, "
            "memory, training time, and AUC), "
            "(6) **rollback integrity check** (report file checksums, restoration status, "
            "and any failed checks). "
            "Include total number of experiment runs logged, latest run metrics "
            "(AUC, F1, Recall), statistical significance summary, resource usage summary, "
            "rollback status, and a final FEEDBACK LOOP VALIDATED or FEEDBACK LOOP FAILED verdict."
        ),
    },
}