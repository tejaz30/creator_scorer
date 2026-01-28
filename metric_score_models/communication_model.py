"""
Communication Scoring Model

This module implements an XGBoost regression model to predict communication scores (1-10)
for Instagram creators based on their content features.

Features used:
- eye_contact_avg_score_0_10: Direct eye contact with audience
- avg_captioned_reels: Use of captions for accessibility
- gemini_is_marketing: Marketing content ratio (inverse relationship)
- gemini_has_humour: Humor usage for engagement
- mean_face_density: Face presence for personal connection
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import (
    XGBOOST_PARAMS, CV_FOLDS, TEST_SIZE, RANDOM_STATE,
    MIN_COMMUNICATION_SCORE, MAX_COMMUNICATION_SCORE, MODELS_DIR, RESULTS_DIR
)
from communication_preprocessor import CommunicationPreprocessor

class CommunicationScorer:
    """XGBoost-based communication scoring model."""
    
    def __init__(self, params=None):
        self.params = params or XGBOOST_PARAMS.copy()
        self.model = None
        self.preprocessor = CommunicationPreprocessor()
        self.is_fitted = False
        self.feature_importance = None
        self.training_metrics = {}
        
    def prepare_data(self, df, target_column='communication'):
        """Prepare data for training."""
        print("🔄 Preparing communication data...")
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")
        
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_column]).copy()
        print(f"   Data shape after removing missing targets: {df_clean.shape}")
        
        # Validate target range
        target_min, target_max = df_clean[target_column].min(), df_clean[target_column].max()
        if target_min < MIN_COMMUNICATION_SCORE or target_max > MAX_COMMUNICATION_SCORE:
            print(f"   ⚠️  Target values outside expected range [{MIN_COMMUNICATION_SCORE}, {MAX_COMMUNICATION_SCORE}]: [{target_min}, {target_max}]")
        
        # Preprocess features using CommunicationPreprocessor
        df_processed = self.preprocessor.prepare_for_training(df_clean, target_column)
        
        # Extract features (exclude target and non-feature columns)
        exclude_columns = [target_column, 'creator', 'aspirational', 'relatable', 'cool', 'credible', 'communication', 'story_telling']
        feature_columns = [col for col in df_processed.columns if col not in exclude_columns]
        X = df_processed[feature_columns]
        y = df_processed[target_column]
        
        print(f"   Final feature matrix shape: {X.shape}")
        print(f"   Target statistics: mean={y.mean():.2f}, std={y.std():.2f}")
        
        return X, y, df_clean
    
    def train(self, df, target_column='communication', test_size=None):
        """Train the communication scoring model."""
        print("🚀 Training Communication Scoring Model...")
        
        test_size = test_size or TEST_SIZE
        
        # Prepare data
        X, y, df_clean = self.prepare_data(df, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=None
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # Initialize and train model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Clip predictions to valid range
        y_pred_train = np.clip(y_pred_train, MIN_COMMUNICATION_SCORE, MAX_COMMUNICATION_SCORE)
        y_pred_test = np.clip(y_pred_test, MIN_COMMUNICATION_SCORE, MAX_COMMUNICATION_SCORE)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, cv=CV_FOLDS, scoring='r2', n_jobs=-1
        )
        
        # Store metrics
        self.training_metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        
        # Print results
        print(f"\n📊 Communication Model Performance:")
        print(f"   Train RMSE: {train_rmse:.3f}")
        print(f"   Test RMSE: {test_rmse:.3f}")
        print(f"   Train R²: {train_r2:.3f}")
        print(f"   Test R²: {test_r2:.3f}")
        print(f"   CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        print(f"\n🔍 Top 5 Feature Importances:")
        for _, row in self.feature_importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        return self.training_metrics
    
    def predict(self, df):
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess features using CommunicationPreprocessor
        df_processed = self.preprocessor.prepare_for_prediction(df)
        
        # Extract features (exclude target and non-feature columns)
        exclude_columns = ['communication', 'communication_score', 'creator', 'aspirational', 'relatable', 'cool', 'credible', 'story_telling']
        feature_columns = [col for col in df_processed.columns if col not in exclude_columns]
        X = df_processed[feature_columns]
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Clip to valid range
        predictions = np.clip(predictions, MIN_COMMUNICATION_SCORE, MAX_COMMUNICATION_SCORE)
        
        return predictions
    
    def predict_with_confidence(self, df, n_estimators=None):
        """Make predictions with confidence intervals using quantile regression."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Get base predictions
        predictions = self.predict(df)
        
        # Estimate uncertainty based on training performance
        test_rmse = self.training_metrics.get('test_rmse', 1.0)
        
        # Simple confidence intervals (±1.96 * RMSE for ~95% confidence)
        uncertainty = 1.96 * test_rmse
        lower_bound = np.clip(predictions - uncertainty, MIN_COMMUNICATION_SCORE, MAX_COMMUNICATION_SCORE)
        upper_bound = np.clip(predictions + uncertainty, MIN_COMMUNICATION_SCORE, MAX_COMMUNICATION_SCORE)
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': np.full_like(predictions, uncertainty)
        }
    
    def save(self, model_path=None):
        """Save the trained model and preprocessor."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        if model_path is None:
            model_path = MODELS_DIR / "communication_scorer.joblib"
        
        # Save model and preprocessor together
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'params': self.params
        }
        
        joblib.dump(model_data, model_path)
        print(f"   ✅ Communication model saved to {model_path}")
    
    def load(self, model_path=None):
        """Load a trained model and preprocessor."""
        if model_path is None:
            model_path = MODELS_DIR / "communication_scorer.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.feature_importance = model_data.get('feature_importance')
        self.training_metrics = model_data.get('training_metrics', {})
        self.params = model_data.get('params', self.params)
        self.is_fitted = True
        
        print(f"   ✅ Communication model loaded from {model_path}")
    
    def get_feature_importance(self):
        """Get feature importance as a dictionary."""
        if self.feature_importance is None:
            return {}
        
        return dict(zip(self.feature_importance['feature'], self.feature_importance['importance']))
    
    def evaluate_predictions(self, df, target_column='communication'):
        """Evaluate model predictions against true values."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        predictions = self.predict(df)
        true_values = df[target_column].values
        
        # Remove any NaN values
        mask = ~(np.isnan(predictions) | np.isnan(true_values))
        predictions = predictions[mask]
        true_values = true_values[mask]
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'mae': mean_absolute_error(true_values, predictions),
            'r2': r2_score(true_values, predictions),
            'n_samples': len(predictions)
        }
        
        return metrics


class CommunicationModel:
    """Legacy class for backward compatibility."""
    
    def __init__(self):
        self.scorer = CommunicationScorer()
        
    def __getattr__(self, name):
        return getattr(self.scorer, name)