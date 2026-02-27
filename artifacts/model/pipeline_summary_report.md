# Customer Churn Prediction — Pipeline Summary Report

**Generated:** 2026-02-27T15:42:04.483789


## Data

- **total_customers**: 7043
- **features**: 20
- **raw_churn_rate_pct**: 26.54

## Feature Selection

- **selected_features**: ['TotalCharges', 'MonthlyCharges', 'tenure', 'Contract', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'gender', 'InternetService', 'OnlineBackup', 'PaperlessBilling', 'MultipleLines']
- **count**: 12

## Model Performance

- **model_type**: GradientBoostingClassifier
- **auc**: 0.8394
- **f1**: 0.5896
- **recall**: 0.5321

## Predictions

- **total**: 7043
- **churners**: 1476
- **churn_rate_pct**: 20.96

## Explanability

- **top_churn_drivers**: ['Contract', 'tenure', 'MonthlyCharges', 'OnlineSecurity', 'TotalCharges']

## Retention Strategy

- **total_at_risk**: 1476
- **segments**: {'Medium Risk': 815, 'High Risk': 661}

## Experiment Log

- **total_runs**: 3
- **latest_run**: {'run_id': 3, 'timestamp': '2026-02-27T14:56:32.313165', 'model_type': 'gradient_boosting', 'auc': 0.8394, 'f1': 0.5896, 'recall': 0.5321}