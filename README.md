# Hospital Readmission Risk Prediction

## Project Overview
This project aims to predict hospital readmission risk for patients with chronic conditions using machine learning techniques. The system analyzes healthcare data to identify patients at high risk of readmission within 30 days and provides actionable insights to reduce unnecessary readmissions.

## Objectives
1. Identify which patients are likely to be readmitted (Binary Classification)
2. Identify key factors causing readmission (Feature Importance Analysis)
3. Generate business insights to reduce unnecessary readmissions

## Features

### Data Processing
- Load and analyze healthcare dataset
- Feature engineering to create derived features:
  - Length_of_Stay: Difference between Admission and Discharge Date
  - Readmitted_within_30: Binary target for 30-day readmission
  - Previous_Visits: Count of earlier records for same patient
  - Age_Group: Categorization into Child, Adult, Senior
  - Billing_Category: Binned billing amount ranges
  - High_Risk_Doctor: Based on historical readmission rates
  - Test_Result_Score: Quantified test results

### Modeling
- Random Forest Classification for readmission prediction
- Feature importance analysis using SHAP values
- Model evaluation with ROC curves and classification metrics

### Model Tuning
- Hyperparameter optimization for multiple model types:
  - Random Forest, Gradient Boosting, XGBoost, Logistic Regression
- Class imbalance handling with SMOTE (Synthetic Minority Over-sampling Technique)
- Feature selection using Recursive Feature Elimination and SelectFromModel
- Model comparison with performance metrics visualization
- Integration of tuned models into the main prediction pipeline

### Visualization & Insights
- Readmission rates by hospital, doctor, and medical condition
- Correlation analysis between patient factors and readmission risk
- Actionable recommendations for reducing readmissions
- Automated report generation with model performance metrics

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the main analysis pipeline:
   ```
   python src/main.py
   ```

3. Run the model tuning pipeline:
   ```
   python src/model_tuning_demo.py
   ```

4. Run the integrated pipeline with model tuning:
   ```
   python src/main_with_tuning.py
   ```

5. View generated visualizations and insights in the `output` directory

### Output

All results are saved to the `output` directory, including:

- Visualization plots (readmission rates by various factors)
- Model performance metrics
- Business insights report with actionable recommendations
- Feature importance analysis

## Project Structure

```
├── src/
│   ├── data_processing/
│   │   ├── load_data.py         # Data loading functions
│   │   └── feature_engineering.py # Feature creation
│   ├── models/
│   │   ├── readmission_prediction.py # ML models
│   │   └── model_tuning.py      # Model optimization
│   ├── visualization/
│   │   └── readmission_insights.py # Visualization
│   ├── main.py                  # Main execution script
│   ├── model_tuning_demo.py     # Model tuning demo
│   └── main_with_tuning.py      # Integrated pipeline with model tuning
├── output/                      # Generated visualizations
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## Project Structure
- `data/`: Contains raw and processed data
- `notebooks/`: Jupyter notebooks for analysis
- `src/`: Source code for the project
  - `data_processing/`: Data loading and cleaning scripts
  - `feature_engineering/`: Feature creation and selection
  - `modeling/`: Machine learning models
  - `visualization/`: Data visualization scripts

## Team Members 
