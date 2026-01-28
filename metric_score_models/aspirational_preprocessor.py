"""
Preprocessor for Aspirational scoring model.
Handles feature engineering and data preparation for aspirational personality scoring.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

class AspirationalPreprocessor:
    """Preprocessor for aspirational scoring features."""
    
    def __init__(self):
        self.feature_names = [
            'mean_hist_score',
            'gemini_genz_word_count', 
            'gemini_comment_sentiment_counts.appreciating',
            'series_reel_mean',
            'avg_captioned_reels',
            'mean_scene_score',
            'mean_face_density',
            'outlier_2sigma_ratio'
        ]
        
        self.scaler = RobustScaler()  # More robust to outliers
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def engineer_features(self, df):
        """Engineer features for aspirational scoring."""
        df_processed = df.copy()
        
        # 1. Handle missing values first
        for feature in self.feature_names:
            if feature not in df_processed.columns:
                print(f"⚠️ Missing feature: {feature}, setting to 0")
                df_processed[feature] = 0.0
        
        # 2. Feature engineering
        
        # Aspirational content indicators
        df_processed['high_hist_score'] = (df_processed['mean_hist_score'] > df_processed['mean_hist_score'].quantile(0.75)).astype(int)
        
        # Gen-Z engagement (normalized)
        max_genz = df_processed['gemini_genz_word_count'].max()
        if max_genz > 0:
            df_processed['genz_word_ratio'] = df_processed['gemini_genz_word_count'] / max_genz
        else:
            df_processed['genz_word_ratio'] = 0.0
        
        # Appreciation engagement (normalized)
        max_appreciation = df_processed['gemini_comment_sentiment_counts.appreciating'].max()
        if max_appreciation > 0:
            df_processed['appreciation_ratio'] = df_processed['gemini_comment_sentiment_counts.appreciating'] / max_appreciation
        else:
            df_processed['appreciation_ratio'] = 0.0
        
        # Anti-aspirational indicators (inverted)
        df_processed['non_series_score'] = 1.0 - df_processed['series_reel_mean']  # Less series = more aspirational
        df_processed['non_captioned_score'] = 1.0 - df_processed['avg_captioned_reels']  # Less captions = more visual impact
        
        # Visual quality indicators
        df_processed['scene_quality_high'] = (df_processed['mean_scene_score'] > df_processed['mean_scene_score'].quantile(0.6)).astype(int)
        df_processed['face_presence_good'] = (df_processed['mean_face_density'] > 0.1).astype(int)
        
        # Content variety (outlier ratio as uniqueness indicator)
        df_processed['content_uniqueness'] = np.clip(df_processed['outlier_2sigma_ratio'], 0, 1)
        
        # Composite aspirational indicators
        df_processed['visual_appeal'] = (
            df_processed['mean_hist_score'] * 0.4 +
            df_processed['mean_scene_score'] * 0.3 +
            df_processed['mean_face_density'] * 0.3
        )
        
        df_processed['engagement_quality'] = (
            df_processed['appreciation_ratio'] * 0.6 +
            df_processed['genz_word_ratio'] * 0.4
        )
        
        # Final feature set for model
        self.engineered_features = [
            'mean_hist_score',
            'genz_word_ratio',
            'appreciation_ratio',
            'non_series_score',
            'non_captioned_score',
            'mean_scene_score',
            'mean_face_density',
            'content_uniqueness',
            'visual_appeal',
            'engagement_quality',
            'high_hist_score',
            'scene_quality_high',
            'face_presence_good'
        ]
        
        return df_processed[self.engineered_features]
    
    def fit(self, df):
        """Fit the preprocessor on training data."""
        print("🔧 Fitting aspirational preprocessor...")
        
        # Engineer features
        X_engineered = self.engineer_features(df)
        
        # Fit imputer
        X_imputed = self.imputer.fit_transform(X_engineered)
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        self.is_fitted = True
        print(f"   ✅ Fitted on {len(df)} samples with {len(self.engineered_features)} features")
        
        return X_scaled
    
    def transform(self, df):
        """Transform new data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Engineer features
        X_engineered = self.engineer_features(df)
        
        # Apply imputer
        X_imputed = self.imputer.transform(X_engineered)
        
        # Apply scaler
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def fit_transform(self, df):
        """Fit and transform in one step."""
        return self.fit(df)
    
    def get_feature_names(self):
        """Get names of engineered features."""
        if not hasattr(self, 'engineered_features'):
            return self.feature_names
        return self.engineered_features
    
    def save(self, filepath):
        """Save the fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        preprocessor_data = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'engineered_features': self.engineered_features,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"💾 Aspirational preprocessor saved to {filepath}")
    
    def load(self, filepath):
        """Load a fitted preprocessor."""
        preprocessor_data = joblib.load(filepath)
        
        self.scaler = preprocessor_data['scaler']
        self.imputer = preprocessor_data['imputer']
        self.feature_names = preprocessor_data['feature_names']
        self.engineered_features = preprocessor_data['engineered_features']
        self.is_fitted = preprocessor_data['is_fitted']
        
        print(f"📂 Aspirational preprocessor loaded from {filepath}")
    
    def get_feature_importance_names(self):
        """Get feature names for importance analysis."""
        return self.get_feature_names()

# Global preprocessor instance
aspirational_preprocessor = AspirationalPreprocessor()