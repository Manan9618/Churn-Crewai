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
            "InternetService, PaymentMethod. "
            "Summarise all findings in a structured report with key observations "
            "about data quality and potential preprocessing needs."
        ),
        "expected_output": (
            "A structured EDA report containing: "
            "(1) dataset shape (rows x columns) and complete column list with dtypes, "
            "(2) missing value table showing count and percentage per column, "
            "(3) Churn class distribution - count and percentage for Yes and No, "
            "(4) descriptive statistics table for numerical features, "
            "(5) unique value summary for categorical features, "
            "(6) a bullet-point list of key data quality observations and "
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
            "(4) Target integrity - the Churn column must contain only 'Yes' or 'No'. "
            "Report PASSED or FAILED with specific details for each check."
        ),
        "expected_output": (
            "A validation report with clear PASSED or FAILED status for each of the "
            "4 checks: schema validation, data type validation, missing value threshold, "
            "and target column integrity. Include specific details on any failures "
            "such as missing column names, invalid values, or threshold violations."
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
            "(2) Drop the customerID column as it is a non-predictive identifier, "
            "(3) Encode binary categorical columns to 0/1: "
            "gender (Male=1, Female=0), Partner, Dependents, PhoneService, "
            "PaperlessBilling, Churn (Yes=1, No=0), "
            "(4) Encode multi-class categorical columns to integer codes: "
            "MultipleLines, InternetService, OnlineSecurity, OnlineBackup, "
            "DeviceProtection, TechSupport, StreamingTV, StreamingMovies, "
            "Contract, PaymentMethod, "
            "(5) Apply StandardScaler to numerical features: "
            "tenure, MonthlyCharges, TotalCharges - save the fitted scaler "
            "to 'model/scaler.pkl' for later inverse transforms. "
            "Save the fully processed dataset to 'data/processed_churn.csv'."
        ),
        "expected_output": (
            "A preprocessing completion report confirming: "
            "(1) TotalCharges converted and NaN values filled with median, "
            "(2) customerID dropped successfully, "
            "(3) list of binary-encoded columns with their mapping, "
            "(4) list of multi-class encoded columns, "
            "(5) confirmation that StandardScaler was applied and scaler.pkl was saved, "
            "(6) final processed dataset shape saved to data/processed_churn.csv."
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
            "with only 0 and 1 values. "
            "Report PASSED or FAILED with specific details for each check."
        ),
        "expected_output": (
            "A preprocessing validation report with PASSED/FAILED status for: "
            "(1) missing values check, "
            "(2) encoding completeness (all columns numeric), "
            "(3) scaling verification (near-zero mean for scaled columns), "
            "(4) column integrity check (no customerID, binary Churn). "
            "Report the final dataset shape and any anomalies found."
        ),
    },

    # =========================================================================
    # 3. FEATURE DISCOVERY
    # =========================================================================

    "feature_discovery": {
        "task": (
            "Identify the most predictive features for customer churn from the "
            "preprocessed dataset 'data/processed_churn.csv': "
            "(1) Correlation analysis - compute Pearson correlation of every feature "
            "with the Churn target column, list all features with absolute correlation >= 0.05, "
            "(2) Random Forest importance - train a RandomForestClassifier "
            "(n_estimators=100, random_state=42) and extract feature_importances_, "
            "(3) Variance threshold filtering - use VarianceThreshold(threshold=0.01) "
            "to identify and remove near-constant features, "
            "(4) Select and save - pick the top 12 features ranked by Random Forest "
            "importance, save the list as a JSON array to 'data/selected_features.json'. "
            "Report the final selected features with their importance scores."
        ),
        "expected_output": (
            "A feature discovery report containing: "
            "(1) correlation table showing features with absolute r >= 0.05 sorted by absolute r, "
            "(2) Random Forest feature importance ranking for all features, "
            "(3) list of any features removed by variance thresholding, "
            "(4) final top-12 selected features with importance scores, "
            "(5) confirmation that data/selected_features.json was saved, "
            "(6) brief business explanation of why the top features drive churn."
        ),
    },

    "feature_validation": {
        "task": (
            "Validate the feature selection output: "
            "(1) Feature count - confirm at least 5 features were selected "
            "(load data/selected_features.json and check length), "
            "(2) No target leakage - confirm neither 'customerID' nor 'Churn' "
            "appear in the selected feature list, "
            "(3) Variance check - for each selected feature, confirm its variance "
            "in the processed dataset is >= 0.001, "
            "(4) File persistence - confirm data/selected_features.json exists, "
            "is a valid JSON file, and contains a non-empty list. "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A feature validation report with PASSED/FAILED status for: "
            "(1) minimum feature count check, "
            "(2) target leakage check, "
            "(3) feature variance check (list any low-variance features found), "
            "(4) file persistence and format check. "
            "Include the full list of validated selected features."
        ),
    },

    # =========================================================================
    # 4. PROCESSING (TRAIN/TEST SPLIT + SMOTE)
    # =========================================================================

    "processing": {
        "task": (
            "Prepare the final train and test datasets for model training: "
            "(1) Train/test split - load 'data/processed_churn.csv', load the selected "
            "features from 'data/selected_features.json', then split 80/20 using "
            "stratified splitting with random_state=42 to preserve the Churn class ratio. "
            "Save X_train, X_test, y_train, y_test as separate CSVs to the data/ directory, "
            "(2) SMOTE oversampling - apply SMOTE (random_state=42) to X_train and y_train "
            "ONLY (never the test set) to balance the Churn classes in the training data. "
            "Overwrite X_train.csv and y_train.csv with the resampled versions, "
            "(3) Feature matrix validation - confirm X_train and X_test have identical "
            "column names and counts, "
            "(4) Save pipeline state - persist current stage metadata including file paths, "
            "split sizes, and feature count to 'data/pipeline_state.json'."
        ),
        "expected_output": (
            "A processing report confirming: "
            "(1) X_train, X_test, y_train, y_test saved with exact row counts, "
            "(2) Churn class distribution before and after SMOTE on the training set, "
            "(3) feature column alignment confirmed between train and test, "
            "(4) pipeline_state.json saved with stage metadata. "
            "Include train churn rate, test churn rate, and total SMOTE-resampled size."
        ),
    },

    "processing_validation": {
        "task": (
            "Validate the train/test split and SMOTE processing outputs: "
            "(1) Split files exist - confirm all 4 files exist and are non-empty: "
            "data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv, "
            "(2) Split ratio - confirm the test set is approximately 20 percent of total data "
            "within a tolerance of plus or minus 2 percent, "
            "(3) No data leakage - check that zero rows are duplicated between "
            "X_train and X_test (no overlapping customer records), "
            "(4) Stratification - confirm Churn rate in y_test matches the original "
            "dataset Churn rate within plus or minus 3 percent tolerance, "
            "(5) Feature alignment - confirm X_train and X_test have identical columns "
            "in the same order. "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A processing validation report with PASSED/FAILED status for: "
            "(1) split file existence and non-empty check, "
            "(2) split ratio check with actual percentages, "
            "(3) data leakage check (overlapping row count), "
            "(4) stratification check with train vs test churn rates, "
            "(5) feature alignment check. "
            "Report final train and test sizes."
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
            "Precision columns. Save the results to 'model/pycaret_compare_results.csv', "
            "(3) Tune best model - call tune_model() on the top-ranked model with "
            "n_iter=50 and optimize='AUC', "
            "(4) Save pipeline - call save_model() to persist the full PyCaret pipeline "
            "including preprocessing to 'model/pycaret_best_model'. "
            "Report the leaderboard and final tuned metrics."
        ),
        "expected_output": (
            "A PyCaret experiment report containing: "
            "(1) leaderboard table showing top 5 models ranked by AUC with "
            "AUC, F1, Recall, Precision values, "
            "(2) name of the best-performing model, "
            "(3) tuned model cross-validation metrics (AUC, F1, Recall), "
            "(4) confirmation that pycaret_compare_results.csv and "
            "pycaret_best_model.pkl were saved to the model/ directory."
        ),
    },

    "pycaret_validation": {
        "task": (
            "Validate the PyCaret AutoML experiment outputs: "
            "(1) Setup pre-conditions - confirm data/processed_churn.csv exists, "
            "contains the Churn column, and has no non-numeric columns, "
            "(2) Leaderboard quality - load model/pycaret_compare_results.csv and "
            "confirm the best model achieves AUC >= 0.75 and F1 >= 0.55, "
            "(3) Model persistence - confirm model/pycaret_best_model.pkl exists "
            "and is loadable without errors. "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A PyCaret validation report with PASSED/FAILED status for: "
            "(1) data readiness pre-conditions, "
            "(2) leaderboard quality thresholds (best AUC and F1 values reported), "
            "(3) model file persistence and loadability. "
            "Include the best model name and its AUC/F1 scores."
        ),
    },

    # =========================================================================
    # 6. MODEL TRAINING
    # =========================================================================

    "model_training": {
        "task": (
            "Train, evaluate, and persist the final churn prediction model: "
            "(1) Train model - load X_train/y_train from data/, train a "
            "GradientBoostingClassifier(n_estimators=200, random_state=42) "
            "on the selected features from data/selected_features.json, "
            "(2) Cross-validation - run 5-fold StratifiedKFold cross-validation "
            "and report mean +/- std for AUC, F1, and Recall, "
            "(3) Test set evaluation - evaluate on X_test/y_test, produce the full "
            "sklearn classification_report, compute AUC from predict_proba, "
            "(4) Save artifacts - save the trained model to 'model/churn_model.pkl'. "
            "Report all metrics in a clear, structured format."
        ),
        "expected_output": (
            "A comprehensive model training report with: "
            "(1) 5-fold CV results: AUC mean +/- std, F1 mean +/- std, Recall mean +/- std, "
            "(2) test set metrics: AUC, Accuracy, Precision, Recall, F1, "
            "(3) full classification report with per-class precision/recall/f1/support, "
            "(4) confirmation that model/churn_model.pkl was saved with file size. "
            "Highlight whether the model meets minimum thresholds: AUC >= 0.75, F1 >= 0.55."
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
            "since missing actual churners is costly for the business. "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A model validation report with PASSED/FAILED status for: "
            "(1) AUC and F1 threshold checks with actual values, "
            "(2) overfitting check with train AUC, test AUC, and gap, "
            "(3) model file integrity check, "
            "(4) churn class recall check with actual recall value. "
            "Include a final overall APPROVED or REJECTED verdict for the model."
        ),
    },

    # =========================================================================
    # 7. PREDICTION
    # =========================================================================

    "prediction": {
        "task": (
            "Generate churn predictions for all customers in the dataset: "
            "(1) Load model - load the trained model from 'model/churn_model.pkl' "
            "and the selected features from 'data/selected_features.json', "
            "(2) Load data - load 'data/processed_churn.csv' and select only the "
            "feature columns for inference, "
            "(3) Predict - run model.predict() for binary labels (0/1) and "
            "model.predict_proba()[:,1] for calibrated churn probability scores, "
            "(4) Annotate and save - add columns 'Churn_Predicted' and "
            "'Churn_Probability' (rounded to 4 decimal places) to the dataframe "
            "and save to 'data/predictions.csv'. "
            "Report the predicted churn rate and a preview of results."
        ),
        "expected_output": (
            "A prediction completion report showing: "
            "(1) total number of customers scored, "
            "(2) count and percentage of predicted churners vs non-churners, "
            "(3) churn probability distribution summary (min, mean, median, max), "
            "(4) confirmation that data/predictions.csv was saved with "
            "Churn_Predicted and Churn_Probability columns, "
            "(5) a 5-row preview of the predictions output."
        ),
    },

    "prediction_validation": {
        "task": (
            "Validate the prediction output file 'data/predictions.csv': "
            "(1) File completeness - confirm the file exists, is non-empty, "
            "and has zero missing values in Churn_Predicted and Churn_Probability, "
            "(2) Required columns - confirm both Churn_Predicted and Churn_Probability "
            "columns are present in the output file, "
            "(3) Probability range - confirm all Churn_Probability values are "
            "within [0.0, 1.0] with no out-of-range scores, "
            "(4) Churn rate sanity - confirm the predicted churn rate (mean of "
            "Churn_Predicted) falls within the expected 10 to 35 percent range "
            "typical for telecom churn datasets. "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A prediction validation report with PASSED/FAILED status for: "
            "(1) file completeness and missing values check, "
            "(2) required column presence check, "
            "(3) probability range check (report any out-of-range count), "
            "(4) churn rate sanity check with actual predicted churn rate. "
            "Include total prediction count and the predicted churn rate percentage."
        ),
    },

    # =========================================================================
    # 8. EXPLANATION (SHAP + LIME)
    # =========================================================================
    "explanation": {
        "task": (
            "Generate model explanations using SHAP and LIME interpretability techniques: "
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
            "(4) Save report - compile top SHAP importances and top churn drivers "
            "into 'artifacts/model/explanation_report.json'. "
            "Provide a business-friendly narrative of the key churn drivers."
        ),
        "expected_output": (
            "An explainability report containing: "
            "(1) global feature importance table (mean absolute SHAP) ranked from most to least important, "
            "(2) SHAP explanations for available high-risk customers (adjust based on test set size), "
            "(3) LIME explanation for the highest-risk available customer with top features, "
            "(4) confirmation that artifacts/model/shap_values.pkl and artifacts/model/explanation_report.json "
            "were saved, "
            "(5) a business narrative explaining each top feature in plain English."
        ),
    },

    "explanation_validation": {
        "task": (
            "Validate the explainability outputs for correctness and domain alignment: "
            "(1) SHAP integrity - load 'model/shap_values.pkl', confirm the SHAP matrix "
            "shape matches (n_samples x n_features), and verify there are zero NaN values, "
            "(2) Domain feature presence - confirm that at least 2 of the known domain-important "
            "features (Contract, tenure, MonthlyCharges, TotalCharges, InternetService) "
            "appear in the top 5 SHAP features by mean absolute importance, "
            "(3) Report completeness - load 'model/explanation_report.json' and confirm "
            "it contains the keys: 'global_feature_importance_shap' and 'top_churn_drivers'. "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "An explainability validation report with PASSED/FAILED status for: "
            "(1) SHAP matrix shape and NaN check, "
            "(2) domain feature presence in top SHAP features "
            "(list which expected features appeared), "
            "(3) explanation report key completeness check. "
            "Include the actual top 5 SHAP features found."
        ),
    },

    # =========================================================================
    # 9. RETENTION STRATEGY
    # =========================================================================

    "retention_strategy": {
        "task": (
            "Design a data-driven customer retention strategy based on churn predictions: "
            "(1) Segment customers - load 'data/predictions.csv' and segment all customers "
            "by Churn_Probability into: "
            "High Risk (probability > 0.70), "
            "Medium Risk (probability 0.40 to 0.70), "
            "Low Risk (probability < 0.40), "
            "(2) Assign retention offers - map each segment to personalised retention offers: "
            "High Risk: 2-year contract upgrade with 20 percent discount plus free TechSupport bundle, "
            "Medium Risk: 1-year contract switch with 10 percent discount plus free streaming add-on, "
            "Low Risk: loyalty rewards programme invitation, "
            "(3) Compute priority scores - add a Priority_Score column (0 to 100) equal to "
            "Churn_Probability multiplied by 100, sort customers descending by priority, "
            "(4) Save retention plan - save all predicted churners (Churn_Predicted=1) "
            "with their segment, offer, and priority score to 'data/retention_plan.csv'. "
            "Provide a business summary of the strategy."
        ),
        "expected_output": (
            "A retention strategy report containing: "
            "(1) customer count per risk segment (High/Medium/Low Risk) with percentages, "
            "(2) retention offer assigned to each segment, "
            "(3) top 10 highest-priority customers with their Churn_Probability and assigned offer, "
            "(4) confirmation that data/retention_plan.csv was saved with all at-risk customers, "
            "(5) a business summary paragraph recommending how the CRM team should "
            "action each segment."
        ),
    },

    "retention_validation": {
        "task": (
            "Validate the retention strategy plan at 'data/retention_plan.csv': "
            "(1) File existence - confirm retention_plan.csv exists and is non-empty, "
            "(2) Segment integrity - confirm Risk_Segment column exists and all values "
            "are one of: High Risk, Medium Risk, Low Risk, "
            "(3) Offer coverage - confirm every customer with Churn_Probability > 0.70 "
            "has a non-null Retention_Offer assigned, "
            "(4) Priority score range - confirm all Priority_Score values are in [0, 100] "
            "with no nulls or out-of-range values. "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A retention plan validation report with PASSED/FAILED status for: "
            "(1) file existence and non-empty check, "
            "(2) segment label integrity check, "
            "(3) offer coverage for high-risk customers, "
            "(4) priority score range validation. "
            "Include total at-risk customer count and breakdown per segment."
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
            "(5) Generate full report - call generate_full_pipeline_summary_tool to compile "
            "all sections into 'model/pipeline_summary_report.json' and "
            "'model/pipeline_summary_report.md'. "
            "Deliver a business-friendly executive summary paragraph."
        ),
        "expected_output": (
            "A complete pipeline summary report containing: "
            "(1) data overview (raw rows, features, churn rate, selected features), "
            "(2) model performance table (model type, AUC, F1, Recall, Precision), "
            "(3) prediction statistics (total scored, churner count, churn rate percent, "
            "probability distribution), "
            "(4) retention strategy overview (at-risk count by segment), "
            "(5) confirmation that both pipeline_summary_report.json and "
            "pipeline_summary_report.md were saved, "
            "(6) a 3 to 5 sentence executive summary paragraph for business stakeholders."
        ),
    },

    "summary_validation": {
        "task": (
            "Validate the final pipeline summary report outputs: "
            "(1) File existence - confirm both 'model/pipeline_summary_report.json' "
            "and 'model/pipeline_summary_report.md' exist and are non-empty, "
            "(2) Section completeness - load the JSON report and confirm all 6 required "
            "sections are present: data, feature_selection, model_performance, "
            "predictions, explanability, retention_strategy, "
            "(3) Model metric threshold - confirm the AUC value in the model_performance "
            "section is >= 0.75, "
            "(4) Prediction stat range - confirm the churn_rate_pct in the predictions "
            "section is between 10.0 and 35.0 percent, "
            "(5) Markdown readability - confirm the .md file contains all required "
            "section headings: Customer Churn, Data, Model Performance, "
            "Predictions, Retention. "
            "Report PASSED or FAILED for each check."
        ),
        "expected_output": (
            "A summary validation report with PASSED/FAILED status for: "
            "(1) JSON and Markdown file existence and non-empty check, "
            "(2) all 6 required sections present in JSON report, "
            "(3) model AUC threshold check (actual AUC value reported), "
            "(4) prediction churn rate range check (actual rate reported), "
            "(5) Markdown headings readability check. "
            "Include a final PIPELINE COMPLETE or PIPELINE INCOMPLETE verdict."
        ),
    },

    # =========================================================================
    # 11. FEEDBACK LOOP
    # =========================================================================

    "feedback_loop": {
        "task": (
            "Analyse the full pipeline results and generate a continuous improvement plan: "
            "(1) Log experiment - call log_experiment_tool to record the current run's "
            "AUC, F1, Recall, model type, and timestamp to 'model/experiment_log.json', "
            "(2) Compare runs - if previous experiments exist in the log, compare the "
            "current AUC against the previous best and compute the improvement delta, "
            "(3) Generate improvement suggestions - analyse current metrics and produce "
            "at least 3 specific, actionable recommendations from these categories: "
            "a) Feature engineering: suggest new derived features such as tenure_group, "
            "monthly_to_total_charge_ratio, or num_services_subscribed, "
            "b) Model tuning: suggest hyperparameter changes or alternative algorithms "
            "such as XGBoost, LightGBM, or adjusting the decision threshold from 0.5 to 0.4, "
            "c) Sampling strategy: suggest SMOTE variants or cost-sensitive learning "
            "if recall on the churn class is below 0.65, "
            "(4) Update feature list - if new features are recommended, call "
            "update_feature_list_tool to add them to 'data/selected_features.json'. "
            "Provide a structured improvement roadmap."
        ),
        "expected_output": (
            "A feedback and improvement report containing: "
            "(1) current experiment run metrics (AUC, F1, Recall, model type), "
            "(2) comparison with previous runs showing AUC delta (or First Run if none), "
            "(3) at least 3 specific improvement recommendations each with: "
            "the recommendation, the reasoning, and the expected impact, "
            "(4) updated feature list if new features were added, "
            "(5) a prioritised roadmap stating what the next iteration should focus on "
            "listing improvements in order of expected impact."
        ),
    },

    "feedback_validation": {
        "task": (
            "Validate the feedback loop outputs for completeness and quality: "
            "(1) Experiment log integrity - confirm 'model/experiment_log.json' exists, "
            "is valid JSON, and the latest entry contains all required fields: "
            "run_id, timestamp, auc, f1, recall, "
            "(2) Improvement suggestions prerequisites - confirm that model/churn_model.pkl, "
            "data/X_test.csv, and data/y_test.csv all exist so suggestions can be generated, "
            "(3) Metrics improvement check - if 2 or more experiment runs exist, confirm "
            "the latest run improved AUC by at least 0.01 over the previous run. "
            "If only 1 run exists, report this check as SKIPPED not FAILED. "
            "Report PASSED, SKIPPED, or FAILED for each check."
        ),
        "expected_output": (
            "A feedback validation report with PASSED/SKIPPED/FAILED status for: "
            "(1) experiment log existence and required field check, "
            "(2) improvement suggestion prerequisites check, "
            "(3) inter-run AUC improvement check (with actual delta or SKIPPED reason). "
            "Include total number of experiment runs logged and the latest run metrics."
        ),
    },
}