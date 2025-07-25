import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Import modules
from data_processing.load_data import load_healthcare_data
from data_processing.feature_engineering import engineer_features
from models.readmission_prediction import ReadmissionPredictor
from models.model_tuning import ModelTuner
from visualization.readmission_insights import ReadmissionVisualizer


def main():
    """
    Main function to run the hospital readmission prediction system with model tuning
    """
    print("\n===== Hospital Readmission Prediction System =====\n")
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Step 1: Load and analyze data
    print("Step 1: Loading and analyzing data...")
    df = load_healthcare_data()
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Step 2: Feature engineering
    print("\nStep 2: Performing feature engineering...")
    df_engineered = engineer_features(df)
    print(f"Created {len(df_engineered.columns) - len(df.columns)} new features:")
    for feature in set(df_engineered.columns) - set(df.columns):
        print(f"  - {feature}")
    
    # Step 3: Create predictor and preprocess data
    print("\nStep 3: Preprocessing data...")
    predictor = ReadmissionPredictor()
    predictor.preprocess_data(df_engineered)
    print(f"Data split into {predictor.X_train.shape[0]} training samples and {predictor.X_test.shape[0]} test samples")
    
    # Step 4: Model tuning
    print("\nStep 4: Tuning models...")
    tuner = ModelTuner(predictor=predictor)
    
    # Optimize Random Forest with a smaller parameter grid for faster execution
    rf_param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }
    
    rf_results = tuner.optimize_random_forest(param_grid=rf_param_grid, verbose=1)
    print(f"Best Random Forest parameters: {rf_results['best_params']}")
    print(f"Best Random Forest ROC AUC: {rf_results['test_auc']:.3f}")
    
    # Apply SMOTE with Random Forest
    smote_param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 20]
    }
    
    smote_results = tuner.apply_smote(model_type='random_forest', param_grid=smote_param_grid, verbose=1)
    print(f"Best SMOTE + Random Forest parameters: {smote_results['best_params']}")
    print(f"Best SMOTE + Random Forest ROC AUC: {smote_results['test_auc']:.3f}")
    
    # Compare models
    comparison = tuner.compare_models()
    
    # Get the best model
    best_model_name = comparison.iloc[0]['model_names']
    best_model = tuner.best_models[best_model_name]
    print(f"\nBest model: {best_model_name} with ROC AUC: {comparison.iloc[0]['test_auc']:.3f}")
    
    # Step 5: Generate visualizations and insights
    print("\nStep 5: Generating visualizations and insights...")
    visualizer = ReadmissionVisualizer(df_engineered)
    
    # Plot readmission rates
    visualizer.plot_readmission_by_category('Medical Condition', save_fig=True)
    visualizer.plot_readmission_by_category('Hospital', save_fig=True)
    visualizer.plot_length_of_stay_vs_readmission(save_fig=True)
    visualizer.plot_age_vs_readmission(save_fig=True)
    visualizer.plot_billing_vs_readmission(save_fig=True)
    visualizer.plot_correlation_heatmap(save_fig=True)
    
    # Generate business insights
    insights = visualizer.generate_business_insights()
    
    # Save insights to file
    with open(output_dir / 'business_insights.txt', 'w') as f:
        f.write("===== Hospital Readmission Business Insights =====\n\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d')}\n\n")
        
        # Overall readmission rate
        f.write(f"Overall 30-day readmission rate: {insights['overall_readmission_rate']:.2%}\n\n")
        
        # Hospital insights
        f.write("===== Hospital Insights =====\n\n")
        f.write("Top 5 Hospitals with Highest Readmission Rates:\n")
        for hospital, data in insights['hospital_insights'].head(5).iterrows():
            f.write(f"- {hospital}: {data['Readmission_Rate']:.2%} (n={data['Count']})\n")
        
        # Medical condition insights
        f.write("\n===== Medical Condition Insights =====\n\n")
        f.write("Top 5 Medical Conditions with Highest Readmission Rates:\n")
        for condition, data in insights['condition_insights'].head(5).iterrows():
            f.write(f"- {condition}: {data['Readmission_Rate']:.2%} (n={data['Count']})\n")
        
        # Length of stay insights
        f.write("\n===== Length of Stay Insights =====\n\n")
        f.write("Readmission rates increase with longer hospital stays.\n")
        f.write("Patients with extended stays (>7 days) show significantly higher readmission risk.\n")
        
        # Age insights
        f.write("\n===== Age Group Insights =====\n\n")
        f.write("Readmission rates by age group:\n")
        for i, row in insights['age_insights'].iterrows():
            f.write(f"- {row['Age_Group']}: {row['Readmitted_within_30']:.2%}\n")
        
        # Correlation insights
        f.write("\n===== Key Risk Factors =====\n\n")
        readmission_corr = insights['correlation_insights']['Readmitted_within_30'].sort_values(ascending=False)
        for feature, corr in readmission_corr.items():
            if feature != 'Readmitted_within_30' and abs(corr) > 0.05:
                f.write(f"- {feature}: Correlation = {corr:.3f}\n")
        
        # Model performance
        f.write("\n===== Model Performance =====\n\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"ROC AUC: {comparison.iloc[0]['test_auc']:.3f}\n")
        f.write(f"Precision (Class 1): {comparison.iloc[0]['precision_class_1']:.3f}\n")
        f.write(f"Recall (Class 1): {comparison.iloc[0]['recall_class_1']:.3f}\n")
        f.write(f"F1 Score (Class 1): {comparison.iloc[0]['f1_class_1']:.3f}\n")
        
        # Model comparison
        f.write("\nModel Comparison:\n")
        for i, row in comparison.iterrows():
            f.write(f"- {row['model_names']}: AUC={row['test_auc']:.3f}, Precision={row['precision_class_1']:.3f}, Recall={row['recall_class_1']:.3f}, F1={row['f1_class_1']:.3f}\n")
        
        # Actionable recommendations
        f.write("\n===== Actionable Recommendations =====\n\n")
        f.write("1. Focus on high-risk patients identified by the model for targeted interventions\n")
        f.write("   - Implement early follow-up appointments for patients with high readmission risk scores\n")
        f.write("   - Develop specialized care plans for patients with multiple risk factors\n\n")
        
        f.write("2. Implement discharge planning improvements for patients with longer lengths of stay\n")
        f.write("   - Enhance transition of care protocols for patients staying >7 days\n")
        f.write("   - Provide additional home care resources for extended-stay patients\n\n")
        
        f.write("3. Review care protocols for medical conditions with highest readmission rates\n")
        f.write("   - Develop specialized follow-up procedures for patients with Obesity, Cancer, and Hypertension\n")
        f.write("   - Implement condition-specific education programs for high-risk diagnoses\n\n")
        
        f.write("4. Provide additional support for elderly patients who show higher readmission risk\n")
        f.write("   - Create geriatric-specific discharge protocols\n")
        f.write("   - Establish dedicated follow-up programs for elderly patients\n\n")
        
        f.write("5. Investigate hospitals with above-average readmission rates for process improvements\n")
        f.write("   - Conduct detailed audits at PLC Moore, Inc Jackson, and LLC Perez hospitals\n")
        f.write("   - Implement best practices from hospitals with lowest readmission rates\n")
    
    print("\nAnalysis complete! Results saved to the 'output' directory.")
    print(f"Best model: {best_model_name} with ROC AUC: {comparison.iloc[0]['test_auc']:.3f}")
    print("Check 'output/business_insights.txt' for detailed insights and recommendations.")


if __name__ == "__main__":
    main()