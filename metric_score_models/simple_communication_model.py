"""
Simple Communication Model that uses only basic features available in both training and test data.
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import config parameters
from config import (
    XGBOOST_PARAMS, CV_FOLDS, TEST_SIZE, RANDOM_STATE,
    COMMUNICATION_FEATURES, PERSONALITY_SCORE_RANGES
)

class SimpleCommunicationScorer:
    """Simple XGBoost-based communication scoring model using only core features."""
    
    def __init__(self):
        # Use features from config
        self.core_features = COMMUNICATION_FEATURES
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance = None
        self.training_metrics = {}
        self.score_range = PERSONALITY_SCORE_RANGES['communication']
        
    def prepare_features(self, df):
        """Prepare features for training/prediction."""
        df_processed = df.copy()
        
        # Ensure all required features exist with defaults
        for feature in self.core_features:
            if feature not in df_processed.columns:
                print(f"  Adding missing feature {feature} with default value")
                if feature == 'eye_contact_avg_score_0_10':
                    df_processed[feature] = 5.0
                elif feature == 'avg_captioned_reels':
                    df_processed[feature] = 0.0
                elif feature in ['gemini_is_marketing', 'gemini_has_humour']:
                    df_processed[feature] = 0.0
                elif feature == 'mean_face_density':
                    df_processed[feature] = 2.0
        
        # Fill missing values
        for feature in self.core_features:
            if df_processed[feature].isnull().any():
                if feature == 'eye_contact_avg_score_0_10':
                    df_processed[feature] = df_processed[feature].fillna(5.0)
                elif feature == 'avg_captioned_reels':
                    df_processed[feature] = df_processed[feature].fillna(0.0)
                elif feature in ['gemini_is_marketing', 'gemini_has_humour']:
                    df_processed[feature] = df_processed[feature].fillna(0.0)
                elif feature == 'mean_face_density':
                    df_processed[feature] = df_processed[feature].fillna(2.0)
        
        # Clip values to reasonable ranges
        df_processed['eye_contact_avg_score_0_10'] = np.clip(df_processed['eye_contact_avg_score_0_10'], 0, 10)
        df_processed['avg_captioned_reels'] = np.clip(df_processed['avg_captioned_reels'], 0, 1)
        df_processed['gemini_is_marketing'] = np.clip(df_processed['gemini_is_marketing'], 0, 1)
        df_processed['gemini_has_humour'] = np.clip(df_processed['gemini_has_humour'], 0, 1)
        df_processed['mean_face_density'] = np.clip(df_processed['mean_face_density'], 0, 10)
        
        return df_processed[self.core_features]
    
    def train(self, df, target_column='communication', test_size=None):
        """Train the simple communication scoring model."""
        print("🚀 Training Simple Communication Scoring Model...")
        
        test_size = test_size or TEST_SIZE
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df[target_column].copy()
        
        # Remove rows with missing target
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        print(f"   Using {len(X)} samples with {len(self.core_features)} features")
        print(f"   Features: {self.core_features}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Clip predictions to valid range (1-10)
        y_pred_train = np.clip(y_pred_train, *self.score_range)
        y_pred_test = np.clip(y_pred_test, *self.score_range)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Accuracy-like metrics for regression
        try:
            from sklearn.metrics import mean_absolute_percentage_error
            train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100
            test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
        except:
            train_mape = test_mape = 0
        
        # Accuracy within tolerance (±0.5 and ±1.0)
        train_acc_05 = np.mean(np.abs(y_train - y_pred_train) <= 0.5) * 100
        test_acc_05 = np.mean(np.abs(y_test - y_pred_test) <= 0.5) * 100
        train_acc_10 = np.mean(np.abs(y_train - y_pred_train) <= 1.0) * 100
        test_acc_10 = np.mean(np.abs(y_test - y_pred_test) <= 1.0) * 100
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=CV_FOLDS, scoring='r2'
        )
        
        # Store metrics
        self.training_metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'val_rmse': test_rmse,  # For compatibility
            'train_r2': train_r2,
            'test_r2': test_r2,
            'val_r2': test_r2,  # For compatibility
            'train_mae': train_mae,
            'test_mae': test_mae,
            'val_mae': test_mae,  # For compatibility
            'train_mape': train_mape,
            'test_mape': test_mape,
            'val_mape': test_mape,  # For compatibility
            'train_acc_05': train_acc_05,
            'test_acc_05': test_acc_05,
            'val_acc_05': test_acc_05,  # For compatibility
            'train_acc_10': train_acc_10,
            'test_acc_10': test_acc_10,
            'val_acc_10': test_acc_10,  # For compatibility
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'n_features': len(self.core_features),
            'n_samples': len(X),
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_test)
        }
        
        # Feature importance
        self.feature_importance = dict(zip(self.core_features, self.model.feature_importances_))
        
        self.is_fitted = True
        
        # Print results
        print(f"\n📊 Simple Communication Model Performance:")
        print(f"   Train RMSE: {train_rmse:.3f}")
        print(f"   Test RMSE: {test_rmse:.3f}")
        print(f"   Train R²: {train_r2:.3f}")
        print(f"   Test R²: {test_r2:.3f}")
        print(f"   Train MAE: {train_mae:.3f}")
        print(f"   Test MAE: {test_mae:.3f}")
        print(f"   Test MAPE: {test_mape:.2f}%")
        print(f"   Test Accuracy ±0.5: {test_acc_05:.1f}%")
        print(f"   Test Accuracy ±1.0: {test_acc_10:.1f}%")
        print(f"   CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        print(f"\n🔍 Feature Importances:")
        for feature, importance in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"   {feature}: {importance:.3f}")
        
        return self.training_metrics
    
    def predict(self, df):
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Clip to valid range
        predictions = np.clip(predictions, *self.score_range)
        
        return predictions
    
    def save(self, filepath):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'core_features': self.core_features,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"   ✅ Simple communication model saved to {filepath}")
    
    def load(self, filepath):
        """Load a trained model."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.core_features = model_data['core_features']
        self.feature_importance = model_data.get('feature_importance', {})
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_fitted = model_data.get('is_fitted', True)
        
        print(f"   ✅ Simple communication model loaded from {filepath}")

# Create global instance
simple_communication_model = SimpleCommunicationScorer()