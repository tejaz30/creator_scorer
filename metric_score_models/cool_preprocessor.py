"""
Preprocessor for Cool scoring model.
Handles feature engineering and data preparation for cool personality scoring.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

class CoolPreprocessor:
    """Preprocessor for cool scoring features."""
    
    def __init__(self):
        self.feature_names = [
            'mean_hist_score',
            'gemini_comment_sentiment_counts.appreciating',
            'avg_english_pct_non_music',
            'gemini_has_humour',
            'series_reel_mean',
            'gemini_is_educational',
            'mean_scene_score',
            'mean_face_density'
        ]
        
        self.scaler = RobustScaler()  # More robust to outliers
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def engineer_features(self, df):
        """Engineer features for cool scoring."""
        df_processed = df.copy()
        
        # 1. Handle missing values first
        for feature in self.feature_names:
            if feature not in df_processed.columns:
                print(f"⚠️ Missing feature: {feature}, setting to 0")
                df_processed[feature] = 0.0
        
        # 2. Feature engineering
        
        # Cool content indicators
        df_processed['high_visual_quality'] = (df_processed['mean_hist_score'] > df_processed['mean_hist_score'].quantile(0.7)).astype(int)
        
        # Appreciation engagement (normalized)
        max_appreciation = df_processed['gemini_comment_sentiment_counts.appreciating'].max()
        if max_appreciation > 0:
            df_processed['appreciation_ratio'] = df_processed['gemini_comment_sentiment_counts.appreciating'] / max_appreciation
        else:
            df_processed['appreciation_ratio'] = 0.0
        
        # English content ratio (cool factor for international appeal)
        df_processed['english_content_ratio'] = df_processed['avg_english_pct_non_music'] / 100.0  # Normalize to 0-1
        
        # Humor presence (binary to continuous)
        df_processed['humor_score'] = df_processed['gemini_has_humour'].astype(float)
        
        # Anti-cool indicators (inverted)
        df_processed['non_series_score'] = 1.0 - df_processed['series_reel_mean']  # Less series = more spontaneous/cool
        df_processed['non_educational_score'] = 1.0 - df_processed['gemini_is_educational']  # Less educational = more entertaining/cool
        
        # Visual and presence indicators
        df_processed['scene_quality_score'] = df_processed['mean_scene_score']
        df_processed['face_presence_score'] = np.clip(df_processed['mean_face_density'], 0, 1)
        
        # Composite cool indicators
        df_processed['visual_coolness'] = (
            df_processed['mean_hist_score'] * 0.5 +
            df_processed['mean_scene_score'] * 0.3 +
            df_processed['face_presence_score'] * 0.2
        )
        
        df_processed['content_coolness'] = (
            df_processed['humor_score'] * 0.4 +
            df_processed['english_content_ratio'] * 0.3 +
            df_processed['non_educational_score'] * 0.3
        )
        
        df_processed['engagement_coolness'] = (
            df_processed['appreciation_ratio'] * 0.6 +
            df_processed['non_series_score'] * 0.4
        )
        
        # Interaction features
        df_processed['humor_visual_combo'] = df_processed['humor_score'] * df_processed['high_visual_quality']
        df_processed['english_humor_combo'] = df_processed['english_content_ratio'] * df_processed['humor_score']
        
        # Final feature set for model
        self.engineered_features = [
            'mean_hist_score',
            'appreciation_ratio',
            'english_content_ratio',
            'humor_score',
            'non_series_score',
            'non_educational_score',
            'scene_quality_score',
            'face_presence_score',
            'visual_coolness',
            'content_coolness',
            'engagement_coolness',
            'high_visual_quality',
            'humor_visual_combo',
            'english_humor_combo'
        ]
        
        return df_processed[self.engineered_features]
    
    def fit(self, df):
        """Fit the preprocessor on training data."""
        print("🔧 Fitting cool preprocessor...")
        
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
        print(f"💾 Cool preprocessor saved to {filepath}")
    
    def load(self, filepath):
        """Load a fitted preprocessor."""
        preprocessor_data = joblib.load(filepath)
        
        self.scaler = preprocessor_data['scaler']
        self.imputer = preprocessor_data['imputer']
        self.feature_names = preprocessor_data['feature_names']
        self.engineered_features = preprocessor_data['engineered_features']
        self.is_fitted = preprocessor_data['is_fitted']
        
        print(f"📂 Cool preprocessor loaded from {filepath}")
    
    def get_feature_importance_names(self):
        """Get feature names for importance analysis."""
        return self.get_feature_names()

# Global preprocessor instance
cool_preprocessor = CoolPreprocessor()