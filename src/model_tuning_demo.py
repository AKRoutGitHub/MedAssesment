import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import modules
from models.model_tuning import ModelTuner
from models.readmission_prediction import ReadmissionPredictor
from data_processing.load_data import load_healthcare_data
from data_processing.feature_engineering import engineer_features


def main():
    """
    Main function to demonstrate model tuning capabilities
    """
    print("\n===== Hospital Readmission Prediction Model Tuning =====\n")
    
    # Step 1: Load and prepare data
    print("Step 1: Loading and preparing data...")
    start_time = time.time()
    
    # Load raw data
    df = load_healthcare_data()
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Engineer features
    df = engineer_features(df)
    print(f"Dataset after feature engineering: {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"New features created: {list(set(df.columns) - set(load_healthcare_data().columns))}")
    
    # Create predictor and preprocess data
    predictor = ReadmissionPredictor()
    predictor.preprocess_data(df)
    
    print(f"Data preparation completed in {time.time() - start_time:.2f} seconds\n")
    
    # Step 2: Create model tuner
    print("Step 2: Creating model tuner...")
    output_dir = Path('output/model_tuning')
    output_dir.mkdir(exist_ok=True, parents=True)
    tuner = ModelTuner(predictor=predictor, output_dir=output_dir)
    print(f"Model tuner created with output directory: {output_dir}\n")
    
    # Step 3: Optimize Random Forest (baseline model)
    print("Step 3: Optimizing Random Forest model...")
    start_time = time.time()
    
    # Use a smaller parameter grid for faster execution
    rf_param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }
    
    rf_results = tuner.optimize_random_forest(param_grid=rf_param_grid, verbose=1)
    print(f"Random Forest optimization completed in {time.time() - start_time:.2f} seconds\n")
    
    # Step 4: Apply SMOTE to handle class imbalance
    print("Step 4: Applying SMOTE with Random Forest...")
    start_time = time.time()
    
    # Use a smaller parameter grid for faster execution
    smote_param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 20]
    }
    
    smote_results = tuner.apply_smote(model_type='random_forest', param_grid=smote_param_grid, verbose=1)
    print(f"SMOTE with Random Forest completed in {time.time() - start_time:.2f} seconds\n")
    
    # Step 5: Feature selection
    print("Step 5: Performing feature selection...")
    start_time = time.time()
    
    feature_selection_results = tuner.perform_feature_selection(model_type='random_forest', method='sfm')
    print(f"Feature selection completed in {time.time() - start_time:.2f} seconds\n")
    
    # Step 6: Compare models
    print("Step 6: Comparing models...")
    comparison = tuner.compare_models()
    
    # Step 7: Print summary
    print("\n===== Model Tuning Summary =====\n")
    print(f"Best Random Forest model:")
    print(f"  - Best parameters: {rf_results['best_params']}")
    print(f"  - ROC AUC: {rf_results['test_auc']:.3f}")
    
    print(f"\nBest SMOTE + Random Forest model:")
    print(f"  - Best parameters: {smote_results['best_params']}")
    print(f"  - ROC AUC: {smote_results['test_auc']:.3f}")
    
    print(f"\nFeature selection results:")
    print(f"  - Number of selected features: {feature_selection_results['n_features']}")
    print(f"  - ROC AUC with selected features: {feature_selection_results['test_auc']:.3f}")
    
    print("\nModel comparison:")
    if comparison is not None:
        for i, row in comparison.iterrows():
            print(f"  - {row['model_names']}: AUC={row['test_auc']:.3f}, Precision={row['precision_class_1']:.3f}, Recall={row['recall_class_1']:.3f}, F1={row['f1_class_1']:.3f}")
    
    print("\nModel tuning complete! Check the 'output/model_tuning' directory for results.")


if __name__ == "__main__":
    main()