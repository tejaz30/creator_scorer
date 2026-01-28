"""
Preprocessor for Relatable scoring model.
Handles feature engineering and data preparation for relatable personality scoring.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

class RelatablePreprocessor:
    """Preprocessor for relatable scoring features."""
    
    def __init__(self):
        self.feature_names = [
            'eye_contact_avg_score_0_10',
            'avg_captioned_reels',
            'gemini_comment_sentiment_counts.neutral',
            'gemini_has_humour',
            'gemini_is_educational',
            'gemini_comment_sentiment_counts.appreciating',
            'mean_scene_score',
            'mean_face_density'
        ]
        
        self.scaler = RobustScaler()  # More robust to outliers
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def engineer_features(self, df):
        """Engineer features for relatable scoring."""
        df_processed = df.copy()
        
        # 1. Handle missing values first
        for feature in self.feature_names:
            if feature not in df_processed.columns:
                print(f"⚠️ Missing feature: {feature}, setting to 0")
                df_processed[feature] = 0.0
        
        # 2. Feature engineering
        
        # Relatable connection indicators
        df_processed['eye_contact_score'] = df_processed['eye_contact_avg_score_0_10'] / 10.0  # Normalize to 0-1
        df_processed['high_eye_contact'] = (df_processed['eye_contact_avg_score_0_10'] > 6.0).astype(int)
        
        # Accessibility indicators
        df_processed['captioned_ratio'] = df_processed['avg_captioned_reels']
        df_processed['high_captions'] = (df_processed['avg_captioned_reels'] > 0.5).astype(int)
        
        # Neutral engagement (relatability through thoughtful responses)
        max_neutral = df_processed['gemini_comment_sentiment_counts.neutral'].max()
        if max_neutral > 0:
            df_processed['neutral_ratio'] = df_processed['gemini_comment_sentiment_counts.neutral'] / max_neutral
        else:
            df_processed['neutral_ratio'] = 0.0
        
        # Humor presence (relatability through humor)
        df_processed['humor_score'] = df_processed['gemini_has_humour'].astype(float)
        
        # Anti-relatable indicators (inverted)
        df_processed['non_educational_score'] = 1.0 - df_processed['gemini_is_educational']  # Less educational = more casual/relatable
        
        # Appreciation can be anti-relatable (too polished/aspirational)
        max_appreciation = df_processed['gemini_comment_sentiment_counts.appreciating'].max()
        if max_appreciation > 0:
            appreciation_ratio = df_processed['gemini_comment_sentiment_counts.appreciating'] / max_appreciation
            df_processed['low_appreciation_score'] = 1.0 - appreciation_ratio  # Less appreciation = more relatable
        else:
            df_processed['low_appreciation_score'] = 1.0
        
        # Visual relatability indicators
        df_processed['scene_quality_score'] = df_processed['mean_scene_score']
        df_processed['face_presence_score'] = np.clip(df_processed['mean_face_density'], 0, 1)
        df_processed['good_face_presence'] = (df_processed['mean_face_density'] > 0.2).astype(int)
        
        # Composite relatable indicators
        df_processed['personal_connection'] = (
            df_processed['eye_contact_score'] * 0.5 +
            df_processed['face_presence_score'] * 0.3 +
            df_processed['humor_score'] * 0.2
        )
        
        df_processed['accessibility_score'] = (
            df_processed['captioned_ratio'] * 0.6 +
            df_processed['neutral_ratio'] * 0.4
        )
        
        df_processed['authenticity_score'] = (
            df_processed['non_educational_score'] * 0.5 +
            df_processed['low_appreciation_score'] * 0.3 +
            df_processed['humor_score'] * 0.2
        )
        
        # Interaction features
        df_processed['eye_contact_humor_combo'] = df_processed['eye_contact_score'] * df_processed['humor_score']
        df_processed['captions_neutral_combo'] = df_processed['captioned_ratio'] * df_processed['neutral_ratio']
        df_processed['face_humor_combo'] = df_processed['face_presence_score'] * df_processed['humor_score']
        
        # Final feature set for model
        self.engineered_features = [
            'eye_contact_score',
            'captioned_ratio',
            'neutral_ratio',
            'humor_score',
            'non_educational_score',
            'low_appreciation_score',
            'scene_quality_score',
            'face_presence_score',
            'personal_connection',
            'accessibility_score',
            'authenticity_score',
            'high_eye_contact',
            'high_captions',
            'good_face_presence',
            'eye_contact_humor_combo',
            'captions_neutral_combo',
            'face_humor_combo'
        ]
        
        return df_processed[self.engineered_features]
    
    def fit(self, df):
        """Fit the preprocessor on training data."""
        print("🔧 Fitting relatable preprocessor...")
        
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
        print(f"💾 Relatable preprocessor saved to {filepath}")
    
    def load(self, filepath):
        """Load a fitted preprocessor."""
        preprocessor_data = joblib.load(filepath)
        
        self.scaler = preprocessor_data['scaler']
        self.imputer = preprocessor_data['imputer']
        self.feature_names = preprocessor_data['feature_names']
        self.engineered_features = preprocessor_data['engineered_features']
        self.is_fitted = preprocessor_data['is_fitted']
        
        print(f"📂 Relatable preprocessor loaded from {filepath}")
    
    def get_feature_importance_names(self):
        """Get feature names for importance analysis."""
        return self.get_feature_names()

# Global preprocessor instance
relatable_preprocessor = RelatablePreprocessor()