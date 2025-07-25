import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import shap
import os
from pathlib import Path

# Add parent directory to path to import from data_processing
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.load_data import load_healthcare_data
from data_processing.feature_engineering import engineer_features


class ReadmissionPredictor:
    """
    Class for predicting hospital readmissions and analyzing feature importance
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.shap_values = None
        
    def load_and_prepare_data(self):
        """
        Load healthcare data and apply feature engineering
        
        Returns:
            pd.DataFrame: Processed dataset with engineered features
        """
        # Load raw data
        df = load_healthcare_data()
        
        # Apply feature engineering
        df = engineer_features(df)
        
        return df
    
    def preprocess_data(self, df, target='Readmitted_within_30', test_size=0.25, random_state=42):
        """
        Preprocess data for model training
        
        Args:
            df (pd.DataFrame): Input dataset with engineered features
            target (str): Target variable name
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Define features to use
        features = [
            'Age', 'Length_of_Stay', 'Previous_Visits', 'Billing Amount', 
            'Test_Result_Score', 'High_Risk_Doctor', 'Gender', 'Medical Condition',
            'Admission Type', 'Doctor', 'Hospital', 'Insurance Provider'
        ]
        
        # Store feature names for later use
        self.feature_names = features
        
        # Split data into features and target
        X = df[features]
        y = df[target]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Store splits for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Identify numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessor
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Store preprocessor for later use
        self.preprocessor = preprocessor
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_type='random_forest'):
        """
        Train a model to predict readmissions
        
        Args:
            model_type (str): Type of model to train ('random_forest', 'gradient_boosting', or 'logistic')
            
        Returns:
            object: Trained model
        """
        # Create model based on type
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline with preprocessor and model
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(self.X_train, self.y_train)
        
        # Store model
        self.model = pipeline
        
        return pipeline
    
    def evaluate_model(self):
        """
        Evaluate the trained model
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_prob)
        
        # Create evaluation dictionary
        evaluation = {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': auc
        }
        
        return evaluation
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance using model's feature_importances_ attribute
        
        Returns:
            dict: Dictionary containing feature importance information
        """
        # Extract the model from the pipeline
        model = self.model.named_steps['model']
        
        # For tree-based models, we can get feature importance directly
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            
            # Create feature importance dictionary
            importance = {
                'feature_importance': feature_importance
            }
            
            return importance
        else:
            # For non-tree-based models
            print("Feature importance is only available for tree-based models.")
            return {'feature_importance': None}
    
    def predict_single(self, patient_dict):
        """
        Predict readmission risk for a single patient and provide SHAP explanation.

        Args:
            patient_dict (dict): Dictionary of patient features matching model input
        Returns:
            dict: { 'prediction': 0/1, 'probability': float, 'explanation': str }
        """
        import shap
        import pandas as pd
        # Ensure model is trained
        if self.model is None:
            raise ValueError("Model is not trained.")
        # Create DataFrame for single patient
        X_single = pd.DataFrame([patient_dict])[self.feature_names]
        # Predict
        pred = self.model.predict(X_single)[0]
        prob = self.model.predict_proba(X_single)[0, 1]
        # SHAP explanation
        explainer = shap.TreeExplainer(self.model.named_steps['model'])
        shap_values = explainer.shap_values(self.model.named_steps['preprocessor'].transform(X_single))
        # Get top features contributing to risk
        feature_impact = list(zip(self.feature_names, shap_values[0]))
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_impact[:3]
        explanation = ", ".join([
            f"{name} ({'+' if val > 0 else '-'}{abs(val):.2f})" for name, val in top_features
        ])
        explanation = f"Top risk factors: {explanation}"
        return {
            'prediction': int(pred),
            'probability': float(prob),
            'explanation': explanation
        }
    
    def generate_insights(self, df):
        """
        Generate business insights based on the model and data
        
        Args:
            df (pd.DataFrame): Input dataset with engineered features
            
        Returns:
            dict: Dictionary containing business insights
        """
        # Calculate readmission rate by hospital
        hospital_readmission = df.groupby('Hospital')['Readmitted_within_30'].mean().sort_values(ascending=False)
        
        # Calculate readmission rate by doctor
        doctor_readmission = df.groupby('Doctor')['Readmitted_within_30'].mean().sort_values(ascending=False)
        
        # Calculate readmission rate by medical condition
        condition_readmission = df.groupby('Medical Condition')['Readmitted_within_30'].mean().sort_values(ascending=False)
        
        # Calculate readmission rate by length of stay
        los_readmission = df.groupby('Length_of_Stay')['Readmitted_within_30'].mean()
        
        # Calculate readmission rate by age group
        age_readmission = df.groupby('Age_Group')['Readmitted_within_30'].mean()
        
        # Create insights dictionary
        insights = {
            'hospital_readmission': hospital_readmission,
            'doctor_readmission': doctor_readmission,
            'condition_readmission': condition_readmission,
            'los_readmission': los_readmission,
            'age_readmission': age_readmission
        }
        
        return insights
    
    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC curve for the trained model
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        # Make predictions
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        auc = roc_auc_score(self.y_test, y_prob)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def plot_feature_importance(self, top_n=10, save_path=None):
        """
        Plot feature importance
        
        Args:
            top_n (int): Number of top features to plot
            save_path (str, optional): Path to save the plot
        """
        # Extract the model from the pipeline
        model = self.model.named_steps['model']
        
        # Get feature names after preprocessing
        preprocessor = self.model.named_steps['preprocessor']
        
        # For tree-based models, we can get feature importance directly
        if hasattr(model, 'feature_importances_'):
            # Get feature names
            feature_names = self.feature_names
            
            # Get feature importance
            feature_importance = model.feature_importances_
            
            # Create dataframe for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title(f'Top {top_n} Feature Importance')
            plt.tight_layout()
            
            # Save plot if path is provided
            if save_path:
                plt.savefig(save_path)
                
            plt.show()
        else:
            print("Feature importance plot is only available for tree-based models.")


def main():
    """
    Main function to demonstrate the ReadmissionPredictor class
    """
    # Create output directory for plots
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Create predictor
    predictor = ReadmissionPredictor()
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = predictor.load_and_prepare_data()
    
    # Preprocess data
    print("Preprocessing data...")
    predictor.preprocess_data(df)
    
    # Train model
    print("Training model...")
    predictor.train_model(model_type='random_forest')
    
    # Evaluate model
    print("Evaluating model...")
    evaluation = predictor.evaluate_model()
    print(f"ROC AUC: {evaluation['roc_auc']:.3f}")
    print("Classification Report:")
    for label, metrics in evaluation['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"  {label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance = predictor.analyze_feature_importance()
    
    # Generate insights
    print("\nGenerating business insights...")
    insights = predictor.generate_insights(df)
    
    # Print top hospitals by readmission rate
    print("\nTop 5 Hospitals by Readmission Rate:")
    print(insights['hospital_readmission'].head(5))
    
    # Print top medical conditions by readmission rate
    print("\nTop 5 Medical Conditions by Readmission Rate:")
    print(insights['condition_readmission'].head(5))
    
    # Plot ROC curve
    print("\nPlotting ROC curve...")
    predictor.plot_roc_curve(save_path=output_dir / 'roc_curve.png')
    
    # Plot feature importance
    print("\nPlotting feature importance...")
    predictor.plot_feature_importance(save_path=output_dir / 'feature_importance.png')
    
    print("\nAnalysis complete! Check the 'output' directory for plots.")


if __name__ == "__main__":
    main()