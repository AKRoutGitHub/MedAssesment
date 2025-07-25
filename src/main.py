import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Import project modules
from data_processing.load_data import load_healthcare_data, initial_data_analysis
from data_processing.feature_engineering import engineer_features
from models.readmission_prediction import ReadmissionPredictor
from visualization.readmission_insights import ReadmissionVisualizer


def main():
    """
    Main function to run the hospital readmission prediction and analysis pipeline
    """
    print("\n===== HOSPITAL READMISSION PREDICTION SYSTEM =====\n")
    
    # Step 1: Load and analyze raw data
    print("Step 1: Loading and analyzing raw data...")
    try:
        df_raw = load_healthcare_data()
        analysis = initial_data_analysis(df_raw)
        print(f"Data loaded successfully! Total records: {analysis['total_records']}")
        print(f"Data types: {analysis['data_types']}")
        print(f"Missing values: {analysis['missing_values']}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Step 2: Apply feature engineering
    print("\nStep 2: Applying feature engineering...")
    try:
        df_engineered = engineer_features(df_raw)
        print("Feature engineering completed!")
        print(f"Original columns: {len(df_raw.columns)}")
        print(f"New columns: {len(df_engineered.columns)}")
        print("\nNew features created:")
        new_features = ['Length_of_Stay', 'Readmitted_within_30', 'Previous_Visits', 
                       'Age_Group', 'Billing_Category', 'High_Risk_Doctor', 'Test_Result_Score']
        for feature in new_features:
            print(f"- {feature}")
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return
    
    # Step 3: Train and evaluate prediction model
    print("\nStep 3: Training and evaluating prediction model...")
    try:
        # Create output directory
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # Create predictor
        predictor = ReadmissionPredictor()
        
        # Preprocess data
        X_train, X_test, y_train, y_test = predictor.preprocess_data(df_engineered)
        print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets")
        
        # Train model
        model = predictor.train_model(model_type='random_forest')
        print("Model trained successfully!")
        
        # Evaluate model
        evaluation = predictor.evaluate_model()
        print(f"Model performance: ROC AUC = {evaluation['roc_auc']:.3f}")
        print("Classification Report:")
        for label, metrics in evaluation['classification_report'].items():
            if isinstance(metrics, dict):
                print(f"  {label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Analyze feature importance
        importance = predictor.analyze_feature_importance()
        print("Feature importance analysis completed!")
        
        # Plot ROC curve and feature importance
        predictor.plot_roc_curve(save_path=output_dir / 'roc_curve.png')
        
        # Only plot feature importance if available
        if importance['feature_importance'] is not None:
            predictor.plot_feature_importance(save_path=output_dir / 'feature_importance.png')
        else:
            print("Feature importance plot not available for this model type.")
    except Exception as e:
        print(f"Error in model training and evaluation: {e}")
        return
    
    # Step 4: Generate visualizations and business insights
    print("\nStep 4: Generating visualizations and business insights...")
    try:
        # Create visualizer
        visualizer = ReadmissionVisualizer(df=df_engineered)
        
        # Generate business insights
        insights = visualizer.generate_business_insights()
        
        # Print actionable recommendations
        print("\n===== ACTIONABLE RECOMMENDATIONS =====\n")
        
        # Recommendation 1: Focus on high-risk hospitals
        print("1. Focus on High-Risk Hospitals:")
        high_risk_hospitals = insights['hospital_insights'].head(3).index.tolist()
        print(f"   - Implement targeted interventions at: {', '.join(high_risk_hospitals)}")
        print("   - Conduct audits to identify process gaps and standardize discharge protocols")
        
        # Recommendation 2: Address high-risk medical conditions
        print("\n2. Address High-Risk Medical Conditions:")
        high_risk_conditions = insights['condition_insights'].head(3).index.tolist()
        print(f"   - Develop specialized care pathways for: {', '.join(high_risk_conditions)}")
        print("   - Implement post-discharge monitoring programs for these conditions")
        
        # Recommendation 3: Optimize length of stay
        print("\n3. Optimize Length of Stay:")
        print("   - Review cases with very short or very long stays")
        print("   - Develop standardized protocols for common conditions")
        
        # Recommendation 4: Improve care coordination
        print("\n4. Improve Care Coordination:")
        print("   - Implement medication reconciliation at discharge")
        print("   - Schedule follow-up appointments before discharge")
        print("   - Provide clear discharge instructions to patients and caregivers")
        
        # Recommendation 5: Monitor high-risk doctors
        print("\n5. Monitor and Support High-Risk Doctors:")
        high_risk_doctors = insights['doctor_insights'].head(3).index.tolist()
        print(f"   - Provide additional resources and training to: {', '.join(high_risk_doctors)}")
        print("   - Implement peer review and mentoring programs")
        
    except Exception as e:
        print(f"Error in generating visualizations and insights: {e}")
        return
    
    print("\n===== ANALYSIS COMPLETE =====\n")
    print("All outputs have been saved to the 'output' directory.")
    print("Use these insights to implement targeted interventions for reducing unnecessary readmissions.")


if __name__ == "__main__":
    main()