"""
Credibility scoring model using XGBoost.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from config import (
    XGBOOST_PARAMS, CV_FOLDS, TEST_SIZE, RANDOM_STATE,
    MIN_CREDIBILITY_SCORE, MAX_CREDIBILITY_SCORE, MODELS_DIR, RESULTS_DIR
)
from credibility_preprocessor import CredibilityFeaturePreprocessor

class CredibilityScorer:
    """XGBoost-based credibility scoring model."""
    
    def __init__(self, params=None):
        self.params = params or XGBOOST_PARAMS.copy()
        self.model = None
        self.preprocessor = CredibilityFeaturePreprocessor()
        self.is_fitted = False
        self.feature_importance = None
        self.training_metrics = {}
        
    def prepare_data(self, df, target_column='credibility_score'):
        """Prepare data for training."""
        print("🔄 Preparing data...")
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")
        
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_column]).copy()
        print(f"   Data shape after removing missing targets: {df_clean.shape}")
        
        # Validate target range
        target_min, target_max = df_clean[target_column].min(), df_clean[target_column].max()
        if target_min < MIN_CREDIBILITY_SCORE or target_max > MAX_CREDIBILITY_SCORE:
            print(f"   ⚠️  Target values outside expected range [{MIN_CREDIBILITY_SCORE}, {MAX_CREDIBILITY_SCORE}]: [{target_min}, {target_max}]")
        
        # Preprocess features
        X, y = self.preprocessor.fit_transform(df_clean, target_column)
        
        print(f"   Final feature matrix shape: {X.shape}")
        print(f"   Target statistics: mean={y.mean():.2f}, std={y.std():.2f}")
        
        return X, y
    
    def train(self, df, target_column='credibility_score', test_size=TEST_SIZE):
        """Train the credibility scoring model."""
        print("🚀 Training credibility scoring model...")
        
        # Prepare data
        X, y = self.prepare_data(df, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=None
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # Train model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Clip predictions to valid range
        train_pred = np.clip(train_pred, MIN_CREDIBILITY_SCORE, MAX_CREDIBILITY_SCORE)
        test_pred = np.clip(test_pred, MIN_CREDIBILITY_SCORE, MAX_CREDIBILITY_SCORE)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_percentage_error
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Accuracy-like metrics for regression
        try:
            train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100
            test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100
        except:
            train_mape = test_mape = 0
        
        # Accuracy within tolerance (±0.5 and ±1.0)
        train_acc_05 = np.mean(np.abs(y_train - train_pred) <= 0.5) * 100
        test_acc_05 = np.mean(np.abs(y_test - test_pred) <= 0.5) * 100
        train_acc_10 = np.mean(np.abs(y_train - train_pred) <= 1.0) * 100
        test_acc_10 = np.mean(np.abs(y_test - test_pred) <= 1.0) * 100
        
        self.training_metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'val_mape': test_mape,  # For compatibility
            'train_acc_05': train_acc_05,
            'test_acc_05': test_acc_05,
            'val_acc_05': test_acc_05,  # For compatibility
            'train_acc_10': train_acc_10,
            'test_acc_10': test_acc_10,
            'val_acc_10': test_acc_10,  # For compatibility
            'val_mae': test_mae,  # For compatibility
            'val_rmse': test_rmse,  # For compatibility
            'val_r2': test_r2,  # For compatibility
            'train_samples': len(y_train),
            'test_samples': len(y_test)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=CV_FOLDS, scoring='neg_mean_squared_error'
        )
        self.training_metrics['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
        self.training_metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
        
        # Feature importance
        self.feature_importance = self.model.feature_importances_
        
        self.is_fitted = True
        
        # Print results
        self._print_training_results()
        
        return {
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def _print_training_results(self):
        """Print training results."""
        print("\n📊 Training Results:")
        print(f"   Train RMSE: {self.training_metrics['train_rmse']:.3f}")
        print(f"   Test RMSE:  {self.training_metrics['test_rmse']:.3f}")
        print(f"   Train MAE:  {self.training_metrics['train_mae']:.3f}")
        print(f"   Test MAE:   {self.training_metrics['test_mae']:.3f}")
        print(f"   Train R²:   {self.training_metrics['train_r2']:.3f}")
        print(f"   Test R²:    {self.training_metrics['test_r2']:.3f}")
        print(f"   CV RMSE:    {self.training_metrics['cv_rmse_mean']:.3f} ± {self.training_metrics['cv_rmse_std']:.3f}")
    
    def predict(self, df):
        """Predict credibility scores for new data."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess data
        X = self.preprocessor.transform(df)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Clip to valid range
        predictions = np.clip(predictions, MIN_CREDIBILITY_SCORE, MAX_CREDIBILITY_SCORE)
        
        return predictions
    
    def predict_with_confidence(self, df, n_estimators_for_std=None):
        """Predict with confidence intervals using model uncertainty."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get base predictions
        predictions = self.predict(df)
        
        # Estimate uncertainty using feature importance and prediction variance
        # This is a simplified approach - for better uncertainty, consider using ensemble methods
        X = self.preprocessor.transform(df)
        
        # Use the model's built-in prediction intervals if available
        # For XGBoost, we can use the standard deviation of leaf values as a proxy
        try:
            # Get prediction contributions for uncertainty estimation
            pred_contribs = self.model.predict(X, pred_contribs=True)
            uncertainty = np.std(pred_contribs, axis=1)
            
            # Scale uncertainty to reasonable confidence intervals
            confidence_interval = uncertainty * 1.96  # ~95% confidence
            
            lower_bound = np.clip(predictions - confidence_interval, 
                                MIN_CREDIBILITY_SCORE, MAX_CREDIBILITY_SCORE)
            upper_bound = np.clip(predictions + confidence_interval, 
                                MIN_CREDIBILITY_SCORE, MAX_CREDIBILITY_SCORE)
            
            return {
                'predictions': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'uncertainty': uncertainty
            }
        except:
            # Fallback: use training RMSE as uncertainty estimate
            uncertainty = np.full(len(predictions), self.training_metrics.get('test_rmse', 1.0))
            confidence_interval = uncertainty * 1.96
            
            lower_bound = np.clip(predictions - confidence_interval, 
                                MIN_CREDIBILITY_SCORE, MAX_CREDIBILITY_SCORE)
            upper_bound = np.clip(predictions + confidence_interval, 
                                MIN_CREDIBILITY_SCORE, MAX_CREDIBILITY_SCORE)
            
            return {
                'predictions': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'uncertainty': uncertainty
            }
    
    def get_feature_importance(self, top_n=None):
        """Get feature importance rankings."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        feature_names = self.preprocessor.feature_names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def plot_feature_importance(self, top_n=15, save_path=None):
        """Plot feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {len(importance_df)} Feature Importance - Credibility Scoring')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save(self, model_path=None, preprocessor_path=None):
        """Save the trained model and preprocessor."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Cannot save.")
        
        if model_path is None:
            model_path = MODELS_DIR / "credibility_model.pkl"
        if preprocessor_path is None:
            preprocessor_path = MODELS_DIR / "credibility_preprocessor.pkl"
        
        # Save model
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, model_path)
        
        # Save preprocessor
        self.preprocessor.save(preprocessor_path)
        
        print(f"Model saved to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")
    
    def load(self, model_path=None, preprocessor_path=None):
        """Load a trained model and preprocessor."""
        if model_path is None:
            model_path = MODELS_DIR / "credibility_model.pkl"
        if preprocessor_path is None:
            preprocessor_path = MODELS_DIR / "credibility_preprocessor.pkl"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']
        self.is_fitted = model_data['is_fitted']
        
        # Load preprocessor
        self.preprocessor.load(preprocessor_path)
        
        print(f"Model loaded from {model_path}")
        print(f"Preprocessor loaded from {preprocessor_path}")
    
    def hyperparameter_tuning(self, df, target_column='credibility_score', param_grid=None):
        """Perform hyperparameter tuning using GridSearchCV."""
        print("🔧 Starting hyperparameter tuning...")
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        # Prepare data
        X, y = self.prepare_data(df, target_column)
        
        # Grid search
        base_model = xgb.XGBRegressor(random_state=RANDOM_STATE, objective='reg:squarederror')
        grid_search = GridSearchCV(
            base_model, param_grid, 
            cv=CV_FOLDS, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update parameters
        self.params.update(grid_search.best_params_)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.3f}")
        
        return grid_search.best_params_