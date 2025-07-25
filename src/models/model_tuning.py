import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import sys

# Add parent directory to path to import from other modules
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.load_data import load_healthcare_data
from data_processing.feature_engineering import engineer_features
from models.readmission_prediction import ReadmissionPredictor


class ModelTuner:
    """
    Class for fine-tuning machine learning models for hospital readmission prediction
    """
    
    def __init__(self, predictor=None, output_dir=None):
        """
        Initialize the model tuner
        
        Args:
            predictor (ReadmissionPredictor, optional): Predictor instance with data already loaded
            output_dir (Path, optional): Directory to save output files
        """
        self.predictor = predictor
        self.output_dir = output_dir or Path('output/model_tuning')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.best_models = {}
        self.feature_importance = {}
        
        # If predictor is not provided, create one
        if self.predictor is None:
            self.predictor = ReadmissionPredictor()
            df = self.predictor.load_and_prepare_data()
            self.predictor.preprocess_data(df)
    
    def optimize_random_forest(self, param_grid=None, cv=5, n_jobs=-1, verbose=1):
        """
        Optimize Random Forest hyperparameters using GridSearchCV
        
        Args:
            param_grid (dict, optional): Parameter grid for GridSearchCV
            cv (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs
            verbose (int): Verbosity level
            
        Returns:
            dict: Best parameters and scores
        """
        print("\nOptimizing Random Forest hyperparameters...")
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 'log2', None]
            }
        
        # Create pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', self.predictor.preprocessor),
            ('model', RandomForestClassifier(random_state=42))
        ])
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        # Fit GridSearchCV
        start_time = time.time()
        grid_search.fit(self.predictor.X_train, self.predictor.y_train)
        end_time = time.time()
        
        # Print results
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # Evaluate best model on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.predictor.X_test)
        y_prob = best_model.predict_proba(self.predictor.X_test)[:, 1]
        
        # Calculate metrics
        test_auc = roc_auc_score(self.predictor.y_test, y_prob)
        report = classification_report(self.predictor.y_test, y_pred, output_dict=True)
        
        print(f"\nTest set ROC AUC: {test_auc:.3f}")
        print("Classification Report:")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"  {label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Store best model
        self.best_models['random_forest'] = best_model
        
        # Extract feature importance
        rf_model = best_model.named_steps['model']
        self.feature_importance['random_forest'] = rf_model.feature_importances_
        
        # Plot feature importance
        self._plot_feature_importance(
            rf_model.feature_importances_,
            'Random Forest Feature Importance',
            'random_forest_importance.png'
        )
        
        # Plot ROC curve
        self._plot_roc_curve(
            self.predictor.y_test,
            y_prob,
            'Random Forest ROC Curve',
            'random_forest_roc.png'
        )
        
        # Return results
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_auc': test_auc,
            'classification_report': report,
            'feature_importance': rf_model.feature_importances_
        }
        
        return results
    
    def optimize_gradient_boosting(self, param_grid=None, cv=5, n_jobs=-1, verbose=1):
        """
        Optimize Gradient Boosting hyperparameters using GridSearchCV
        
        Args:
            param_grid (dict, optional): Parameter grid for GridSearchCV
            cv (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs
            verbose (int): Verbosity level
            
        Returns:
            dict: Best parameters and scores
        """
        print("\nOptimizing Gradient Boosting hyperparameters...")
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2]
            }
        
        # Create pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', self.predictor.preprocessor),
            ('model', GradientBoostingClassifier(random_state=42))
        ])
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        # Fit GridSearchCV
        start_time = time.time()
        grid_search.fit(self.predictor.X_train, self.predictor.y_train)
        end_time = time.time()
        
        # Print results
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # Evaluate best model on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.predictor.X_test)
        y_prob = best_model.predict_proba(self.predictor.X_test)[:, 1]
        
        # Calculate metrics
        test_auc = roc_auc_score(self.predictor.y_test, y_prob)
        report = classification_report(self.predictor.y_test, y_pred, output_dict=True)
        
        print(f"\nTest set ROC AUC: {test_auc:.3f}")
        print("Classification Report:")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"  {label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Store best model
        self.best_models['gradient_boosting'] = best_model
        
        # Extract feature importance
        gb_model = best_model.named_steps['model']
        self.feature_importance['gradient_boosting'] = gb_model.feature_importances_
        
        # Plot feature importance
        self._plot_feature_importance(
            gb_model.feature_importances_,
            'Gradient Boosting Feature Importance',
            'gradient_boosting_importance.png'
        )
        
        # Plot ROC curve
        self._plot_roc_curve(
            self.predictor.y_test,
            y_prob,
            'Gradient Boosting ROC Curve',
            'gradient_boosting_roc.png'
        )
        
        # Return results
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_auc': test_auc,
            'classification_report': report,
            'feature_importance': gb_model.feature_importances_
        }
        
        return results
    
    def optimize_xgboost(self, param_grid=None, cv=5, n_jobs=-1, verbose=1):
        """
        Optimize XGBoost hyperparameters using GridSearchCV
        
        Args:
            param_grid (dict, optional): Parameter grid for GridSearchCV
            cv (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs
            verbose (int): Verbosity level
            
        Returns:
            dict: Best parameters and scores
        """
        print("\nOptimizing XGBoost hyperparameters...")
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__min_child_weight': [1, 3, 5],
                'model__gamma': [0, 0.1, 0.2],
                'model__subsample': [0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        # Create pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', self.predictor.preprocessor),
            ('model', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
        ])
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        # Fit GridSearchCV
        start_time = time.time()
        grid_search.fit(self.predictor.X_train, self.predictor.y_train)
        end_time = time.time()
        
        # Print results
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # Evaluate best model on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.predictor.X_test)
        y_prob = best_model.predict_proba(self.predictor.X_test)[:, 1]
        
        # Calculate metrics
        test_auc = roc_auc_score(self.predictor.y_test, y_prob)
        report = classification_report(self.predictor.y_test, y_pred, output_dict=True)
        
        print(f"\nTest set ROC AUC: {test_auc:.3f}")
        print("Classification Report:")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"  {label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Store best model
        self.best_models['xgboost'] = best_model
        
        # Extract feature importance
        xgb_model = best_model.named_steps['model']
        self.feature_importance['xgboost'] = xgb_model.feature_importances_
        
        # Plot feature importance
        self._plot_feature_importance(
            xgb_model.feature_importances_,
            'XGBoost Feature Importance',
            'xgboost_importance.png'
        )
        
        # Plot ROC curve
        self._plot_roc_curve(
            self.predictor.y_test,
            y_prob,
            'XGBoost ROC Curve',
            'xgboost_roc.png'
        )
        
        # Return results
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_auc': test_auc,
            'classification_report': report,
            'feature_importance': xgb_model.feature_importances_
        }
        
        return results
    
    def optimize_logistic_regression(self, param_grid=None, cv=5, n_jobs=-1, verbose=1):
        """
        Optimize Logistic Regression hyperparameters using GridSearchCV
        
        Args:
            param_grid (dict, optional): Parameter grid for GridSearchCV
            cv (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs
            verbose (int): Verbosity level
            
        Returns:
            dict: Best parameters and scores
        """
        print("\nOptimizing Logistic Regression hyperparameters...")
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l1', 'l2', 'elasticnet', None],
                'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'model__class_weight': [None, 'balanced']
            }
        
        # Create pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', self.predictor.preprocessor),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        # Fit GridSearchCV
        start_time = time.time()
        grid_search.fit(self.predictor.X_train, self.predictor.y_train)
        end_time = time.time()
        
        # Print results
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # Evaluate best model on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.predictor.X_test)
        y_prob = best_model.predict_proba(self.predictor.X_test)[:, 1]
        
        # Calculate metrics
        test_auc = roc_auc_score(self.predictor.y_test, y_prob)
        report = classification_report(self.predictor.y_test, y_pred, output_dict=True)
        
        print(f"\nTest set ROC AUC: {test_auc:.3f}")
        print("Classification Report:")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"  {label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Store best model
        self.best_models['logistic_regression'] = best_model
        
        # Plot ROC curve
        self._plot_roc_curve(
            self.predictor.y_test,
            y_prob,
            'Logistic Regression ROC Curve',
            'logistic_regression_roc.png'
        )
        
        # Return results
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_auc': test_auc,
            'classification_report': report
        }
        
        return results
    
    def apply_smote(self, model_type='random_forest', param_grid=None, cv=5, n_jobs=-1, verbose=1):
        """
        Apply SMOTE to handle class imbalance and optimize model hyperparameters
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'gradient_boosting', 'xgboost', 'logistic')
            param_grid (dict, optional): Parameter grid for GridSearchCV
            cv (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs
            verbose (int): Verbosity level
            
        Returns:
            dict: Best parameters and scores
        """
        print(f"\nApplying SMOTE and optimizing {model_type} hyperparameters...")
        
        # Create model based on type
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            if param_grid is None:
                param_grid = {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [None, 20],
                    'model__min_samples_split': [2, 5],
                    'model__min_samples_leaf': [1, 2]
                }
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
            if param_grid is None:
                param_grid = {
                    'model__n_estimators': [100, 200],
                    'model__learning_rate': [0.1, 0.2],
                    'model__max_depth': [3, 5]
                }
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            if param_grid is None:
                param_grid = {
                    'model__n_estimators': [100, 200],
                    'model__learning_rate': [0.1, 0.2],
                    'model__max_depth': [3, 5]
                }
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
            if param_grid is None:
                param_grid = {
                    'model__C': [0.1, 1, 10],
                    'model__penalty': ['l2', None],
                    'model__class_weight': [None, 'balanced']
                }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline with SMOTE, preprocessor, and model
        pipeline = ImbPipeline([
            ('preprocessor', self.predictor.preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        # Fit GridSearchCV
        start_time = time.time()
        grid_search.fit(self.predictor.X_train, self.predictor.y_train)
        end_time = time.time()
        
        # Print results
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # Evaluate best model on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.predictor.X_test)
        y_prob = best_model.predict_proba(self.predictor.X_test)[:, 1]
        
        # Calculate metrics
        test_auc = roc_auc_score(self.predictor.y_test, y_prob)
        report = classification_report(self.predictor.y_test, y_pred, output_dict=True)
        
        print(f"\nTest set ROC AUC: {test_auc:.3f}")
        print("Classification Report:")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"  {label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Store best model
        self.best_models[f'smote_{model_type}'] = best_model
        
        # Extract feature importance for tree-based models
        if model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model_instance = best_model.named_steps['model']
            self.feature_importance[f'smote_{model_type}'] = model_instance.feature_importances_
            
            # Plot feature importance
            self._plot_feature_importance(
                model_instance.feature_importances_,
                f'SMOTE {model_type.replace("_", " ").title()} Feature Importance',
                f'smote_{model_type}_importance.png'
            )
        
        # Plot ROC curve
        self._plot_roc_curve(
            self.predictor.y_test,
            y_prob,
            f'SMOTE {model_type.replace("_", " ").title()} ROC Curve',
            f'smote_{model_type}_roc.png'
        )
        
        # Return results
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_auc': test_auc,
            'classification_report': report
        }
        
        if model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            results['feature_importance'] = model_instance.feature_importances_
        
        return results
    
    def perform_feature_selection(self, model_type='random_forest', method='rfe', cv=5):
        """
        Perform feature selection using Recursive Feature Elimination or SelectFromModel
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'gradient_boosting', 'xgboost', 'logistic')
            method (str): Method to use for feature selection ('rfe' or 'sfm')
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Selected features and scores
        """
        print(f"\nPerforming feature selection using {method.upper()} with {model_type}...")
        
        # Create model based on type
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Apply preprocessor to get numeric features
        X_train_preprocessed = self.predictor.preprocessor.fit_transform(self.predictor.X_train)
        X_test_preprocessed = self.predictor.preprocessor.transform(self.predictor.X_test)
        
        # Perform feature selection
        if method == 'rfe':
            # Recursive Feature Elimination with Cross-Validation
            selector = RFECV(
                estimator=model,
                step=1,
                cv=cv,
                scoring='roc_auc',
                min_features_to_select=1,
                n_jobs=-1
            )
            selector.fit(X_train_preprocessed, self.predictor.y_train)
            
            # Get selected features
            n_features = selector.n_features_
            support = selector.support_
            ranking = selector.ranking_
            
            print(f"\nNumber of selected features: {n_features}")
            print(f"Feature ranking: {ranking}")
            
            # Plot number of features vs. CV score
            plt.figure(figsize=(10, 6))
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (ROC AUC)")
            plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
            plt.title(f"Recursive Feature Elimination with {model_type.replace('_', ' ').title()}")
            plt.tight_layout()
            plt.savefig(self.output_dir / f'rfe_{model_type}_scores.png')
            plt.close()
            
            # Train model with selected features
            X_train_selected = X_train_preprocessed[:, support]
            X_test_selected = X_test_preprocessed[:, support]
            
            model.fit(X_train_selected, self.predictor.y_train)
            y_pred = model.predict(X_test_selected)
            y_prob = model.predict_proba(X_test_selected)[:, 1]
            
            # Calculate metrics
            test_auc = roc_auc_score(self.predictor.y_test, y_prob)
            report = classification_report(self.predictor.y_test, y_pred, output_dict=True)
            
            print(f"\nTest set ROC AUC with selected features: {test_auc:.3f}")
            print("Classification Report:")
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    print(f"  {label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
            
            # Store results
            results = {
                'n_features': n_features,
                'support': support,
                'ranking': ranking,
                'grid_scores': selector.grid_scores_,
                'test_auc': test_auc,
                'classification_report': report
            }
            
        elif method == 'sfm':
            # SelectFromModel
            selector = SelectFromModel(estimator=model, threshold='median')
            selector.fit(X_train_preprocessed, self.predictor.y_train)
            
            # Get selected features
            support = selector.get_support()
            n_features = sum(support)
            
            print(f"\nNumber of selected features: {n_features}")
            
            # Train model with selected features
            X_train_selected = selector.transform(X_train_preprocessed)
            X_test_selected = selector.transform(X_test_preprocessed)
            
            model.fit(X_train_selected, self.predictor.y_train)
            y_pred = model.predict(X_test_selected)
            y_prob = model.predict_proba(X_test_selected)[:, 1]
            
            # Calculate metrics
            test_auc = roc_auc_score(self.predictor.y_test, y_prob)
            report = classification_report(self.predictor.y_test, y_pred, output_dict=True)
            
            print(f"\nTest set ROC AUC with selected features: {test_auc:.3f}")
            print("Classification Report:")
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    print(f"  {label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
            
            # Store results
            results = {
                'n_features': n_features,
                'support': support,
                'test_auc': test_auc,
                'classification_report': report
            }
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        return results
    
    def compare_models(self):
        """
        Compare all trained models and plot ROC curves
        
        Returns:
            dict: Comparison results
        """
        print("\nComparing all trained models...")
        
        # Check if models have been trained
        if not self.best_models:
            print("No models have been trained yet. Please train models first.")
            return None
        
        # Initialize results dictionary
        results = {
            'model_names': [],
            'test_auc': [],
            'precision_class_1': [],
            'recall_class_1': [],
            'f1_class_1': []
        }
        
        # Create figure for ROC curves
        plt.figure(figsize=(10, 8))
        
        # Evaluate each model
        for model_name, model in self.best_models.items():
            # Make predictions
            y_prob = model.predict_proba(self.predictor.X_test)[:, 1]
            y_pred = model.predict(self.predictor.X_test)
            
            # Calculate metrics
            test_auc = roc_auc_score(self.predictor.y_test, y_prob)
            report = classification_report(self.predictor.y_test, y_pred, output_dict=True)
            
            # Store results
            results['model_names'].append(model_name)
            results['test_auc'].append(test_auc)
            results['precision_class_1'].append(report['1']['precision'])
            results['recall_class_1'].append(report['1']['recall'])
            results['f1_class_1'].append(report['1']['f1-score'])
            
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(self.predictor.y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {test_auc:.3f})')
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'k--')
        
        # Set plot properties
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / 'model_comparison_roc.png')
        plt.close()
        
        # Create dataframe for results
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('test_auc', ascending=False)
        
        # Print results
        print("\nModel Comparison Results:")
        print(results_df)
        
        # Plot comparison bar chart
        plt.figure(figsize=(12, 8))
        
        # Plot AUC
        plt.subplot(2, 2, 1)
        sns.barplot(x='model_names', y='test_auc', data=results_df)
        plt.title('ROC AUC')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.9, 1.0)  # Adjust as needed
        
        # Plot Precision
        plt.subplot(2, 2, 2)
        sns.barplot(x='model_names', y='precision_class_1', data=results_df)
        plt.title('Precision (Class 1)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.5, 1.0)  # Adjust as needed
        
        # Plot Recall
        plt.subplot(2, 2, 3)
        sns.barplot(x='model_names', y='recall_class_1', data=results_df)
        plt.title('Recall (Class 1)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.5, 1.0)  # Adjust as needed
        
        # Plot F1
        plt.subplot(2, 2, 4)
        sns.barplot(x='model_names', y='f1_class_1', data=results_df)
        plt.title('F1 Score (Class 1)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.5, 1.0)  # Adjust as needed
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison_metrics.png')
        plt.close()
        
        return results_df
    
    def _plot_feature_importance(self, importance, title, filename):
        """
        Plot feature importance
        
        Args:
            importance (array): Feature importance values
            title (str): Plot title
            filename (str): Filename to save plot
        """
        try:
            # Get feature names
            feature_names = self.predictor.feature_names
            
            # Check if lengths match
            if len(feature_names) != len(importance):
                print(f"Warning: Feature names length ({len(feature_names)}) doesn't match importance length ({len(importance)})")
                # Create generic feature names if lengths don't match
                feature_names = [f"Feature {i}" for i in range(len(importance))]
            
            # Create dataframe for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot top 15 features (or fewer if less available)
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(min(15, len(importance_df))))
            plt.title(title)
            plt.tight_layout()
            plt.savefig(self.output_dir / filename)
            plt.close()
        except Exception as e:
            print(f"Error plotting feature importance: {e}")
            # Create a simple bar plot without feature names as fallback
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importance)), sorted(importance, reverse=True)[:15])
            plt.title(f"{title} (Generic Features)")
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig(self.output_dir / filename)
            plt.close()
    
    def _plot_roc_curve(self, y_true, y_prob, title, filename):
        """
        Plot ROC curve
        
        Args:
            y_true (array): True labels
            y_prob (array): Predicted probabilities
            title (str): Plot title
            filename (str): Filename to save plot
        """
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()


def main():
    """
    Main function to demonstrate the ModelTuner class
    """
    # Create predictor and load data
    print("Loading and preparing data...")
    predictor = ReadmissionPredictor()
    df = predictor.load_and_prepare_data()
    predictor.preprocess_data(df)
    
    # Create model tuner
    tuner = ModelTuner(predictor=predictor)
    
    # Optimize Random Forest (default model)
    rf_results = tuner.optimize_random_forest()
    
    # Apply SMOTE with Random Forest
    smote_rf_results = tuner.apply_smote(model_type='random_forest')
    
    # Optimize XGBoost
    xgb_results = tuner.optimize_xgboost()
    
    # Compare models
    comparison = tuner.compare_models()
    
    print("\nModel tuning complete! Check the 'output/model_tuning' directory for results.")


if __name__ == "__main__":
    main()