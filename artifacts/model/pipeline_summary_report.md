# Customer Churn Prediction — Pipeline Summary Report

**Generated:** 2026-02-28T21:40:46.542295


## Data

- **total_customers**: 7043
- **features**: 20
- **raw_churn_rate_pct**: 26.54

## Feature Selection

- **selected_features**: ['TotalCharges', 'MonthlyCharges', 'tenure', 'Contract', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'InternetService', 'gender', 'OnlineBackup', 'PaperlessBilling', 'MultipleLines']
- **count**: 12

## Model Performance

- **model_type**: GradientBoostingClassifier
- **auc**: 0.8393
- **f1**: 0.5875
- **recall**: 0.5294

## Predictions

- **total**: 7043
- **churners**: 1475
- **churn_rate_pct**: 20.94

## Explanability

- **top_churn_drivers**: ['Contract', 'tenure', 'MonthlyCharges', 'OnlineSecurity', 'TotalCharges']

## Retention Strategy

- **total_at_risk**: 1475
- **segments**: {'Medium Risk': 814, 'High Risk': 661}