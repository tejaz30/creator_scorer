"""
Relatable Scoring Model
Predicts how relatable a creator appears based on content analysis.

Features used:
- eye_contact_avg_score_0_10 (positive)
- avg_captioned_reels (positive)
- gemini_comment_sentiment_counts.neutral (positive)
- gemini_has_humour (positive)
- gemini_is_educational (negative)
- gemini_comment_sentiment_counts.appreciating (negative)
- mean_scene_score (additional)
- mean_face_density (additional)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from relatable_preprocessor import relatable_preprocessor
import warnings
warnings.filterwarnings('ignore')

# Import config parameters
from config import (
    RANDOM_FOREST_PARAMS, CV_FOLDS, TEST_SIZE, RANDOM_STATE,
    RELATABLE_FEATURES, PERSONALITY_SCORE_RANGES
)

class RelatableModel:
    """Model for predicting relatable scores."""
    
    def __init__(self):
        self.model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
        self.feature_names = RELATABLE_FEATURES
        self.preprocessor = relatable_preprocessor
        self.is_fitted = False
        self.score_range = PERSONALITY_SCORE_RANGES['relatable']
        
    def prepare_features(self, df):
        """Prepare features using the preprocessor."""
        return self.preprocessor.transform(df)
    
    def fit(self, df, target_col='relatable_score', test_size=None):
        """Train the model with train/validation split."""
        print("🔧 Training relatable model...")
        
        test_size = test_size or TEST_SIZE
        
        # Prepare data
        X = self.preprocessor.fit_transform(df)
        y = df[target_col]
        
        # Remove missing targets
        mask = ~pd.isna(y)
        X = X[mask]
        y = y[mask]
        
        print(f"   Training on {len(X)} samples with {X.shape[1]} features")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_percentage_error
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        
        # Accuracy-like metrics for regression
        try:
            train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100
            val_mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100
        except:
            train_mape = val_mape = 0
        
        # Accuracy within tolerance (±0.5 and ±1.0)
        train_acc_05 = np.mean(np.abs(y_train - y_pred_train) <= 0.5) * 100
        val_acc_05 = np.mean(np.abs(y_val - y_pred_val) <= 0.5) * 100
        train_acc_10 = np.mean(np.abs(y_train - y_pred_train) <= 1.0) * 100
        val_acc_10 = np.mean(np.abs(y_val - y_pred_val) <= 1.0) * 100
        
        # Cross-validation on training set
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=CV_FOLDS, scoring='r2')
        
        # Get feature importance with engineered feature names
        feature_names = self.preprocessor.get_feature_names()
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        return {
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': val_r2,  # For compatibility
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'test_rmse': val_rmse,  # For compatibility
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_mape': train_mape,
            'val_mape': val_mape,
            'train_acc_05': train_acc_05,
            'val_acc_05': val_acc_05,
            'train_acc_10': train_acc_10,
            'val_acc_10': val_acc_10,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val)
        }
    
    def predict(self, df):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self.prepare_features(df)
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save the model and preprocessor."""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        
        # Save preprocessor separately
        preprocessor_path = str(filepath).replace('.joblib', '_preprocessor.joblib')
        self.preprocessor.save(preprocessor_path)
    
    def load(self, filepath):
        """Load the model and preprocessor."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        # Load preprocessor
        preprocessor_path = str(filepath).replace('.joblib', '_preprocessor.joblib')
        self.preprocessor.load(preprocessor_path)

# Global model instance
relatable_model = RelatableModel()