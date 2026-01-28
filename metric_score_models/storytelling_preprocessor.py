"""
Feature preprocessing for storytelling scoring model.
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
import joblib
from pathlib import Path

from config import STORYTELLING_FEATURES, MODELS_DIR

class StorytellingFeaturePreprocessor:
    """Preprocessor for storytelling scoring features."""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_stats = {}
        self.feature_names = None
        
    def analyze_feature_distributions(self, df):
        """Analyze each feature's distribution for optimal preprocessing."""
        analysis = {}
        
        for feature in STORYTELLING_FEATURES:
            if feature not in df.columns:
                warnings.warn(f"Feature {feature} not found in data")
                continue
                
            data = df[feature].dropna()
            
            if len(data) == 0:
                warnings.warn(f"Feature {feature} has no valid data")
                continue
            
            analysis[feature] = {
                'count': len(data),
                'missing_pct': df[feature].isnull().mean(),
                'min': data.min(),
                'max': data.max(),
                'mean': data.mean(),
                'std': data.std(),
                'skewness': stats.skew(data) if len(data) > 1 else 0,
                'kurtosis': stats.kurtosis(data) if len(data) > 1 else 0,
                'zeros_pct': (data == 0).mean(),
                'ones_pct': (data == 1).mean() if data.max() <= 1 else 0
            }
            
        return analysis
    
    def create_preprocessing_pipeline(self, feature_analysis):
        """Create preprocessing pipeline based on feature analysis."""
        
        # Categorize features by preprocessing needs
        standard_features = []
        log_transform_features = []
        proportion_features = []
        robust_features = []
        
        for feature in STORYTELLING_FEATURES:
            if feature not in feature_analysis:
                continue
                
            stats_dict = feature_analysis[feature]
            
            # Count data (high skewness, starts from 0)
            if (stats_dict['skewness'] > 2 and 
                stats_dict['min'] >= 0 and 
                'counts' in feature.lower()):
                log_transform_features.append(feature)
                
            # Proportion data (bounded 0-1)
            elif (stats_dict['max'] <= 1.01 and  # Allow small floating point errors
                  stats_dict['min'] >= -0.01 and 
                  stats_dict['ones_pct'] < 0.8):  # Not mostly 1s
                proportion_features.append(feature)
                
            # Features with outliers (high kurtosis)
            elif stats_dict['kurtosis'] > 3:
                robust_features.append(feature)
                
            # Standard continuous features
            else:
                standard_features.append(feature)
        
        # Create transformers
        transformers = []
        
        if standard_features:
            transformers.append(
                ('standard', StandardScaler(), standard_features)
            )
            
        if log_transform_features:
            transformers.append(
                ('log_standard', 
                 Pipeline([
                     ('log', PowerTransformer(method='yeo-johnson')),
                     ('scale', StandardScaler())
                 ]), 
                 log_transform_features)
            )
            
        if proportion_features:
            transformers.append(
                ('proportion', StandardScaler(), proportion_features)
            )
            
        if robust_features:
            transformers.append(
                ('robust', RobustScaler(), robust_features)
            )
        
        # Store feature categorization for later reference
        self.feature_categories = {
            'standard': standard_features,
            'log_transform': log_transform_features,
            'proportion': proportion_features,
            'robust': robust_features
        }
        
        return ColumnTransformer(transformers, remainder='drop')
    
    def handle_missing_values(self, df):
        """Handle missing values based on feature characteristics."""
        df_processed = df.copy()
        
        for feature in STORYTELLING_FEATURES:
            if feature not in df.columns:
                continue
                
            missing_pct = df[feature].isnull().mean()
            
            if missing_pct > 0:
                if missing_pct > 0.5:
                    warnings.warn(f"{feature} has {missing_pct:.1%} missing values")
                
                # For count features, missing likely means 0
                if 'counts' in feature.lower():
                    df_processed[feature].fillna(0, inplace=True)
                
                # For proportion features, use median
                elif df[feature].max() <= 1 and df[feature].min() >= 0:
                    df_processed[feature].fillna(df[feature].median(), inplace=True)
                
                # For other continuous features, use median
                else:
                    df_processed[feature].fillna(df[feature].median(), inplace=True)
        
        return df_processed
    
    def create_interaction_features(self, df):
        """Create meaningful interaction features for storytelling."""
        df_enhanced = df.copy()
        
        # Check if required features exist
        required_features = ['avg_captioned_reels', 'gemini_has_humour', 
                           'mean_face_density', 'outlier_2sigma_ratio']
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            warnings.warn(f"Missing features for interactions: {missing_features}")
            return df_enhanced
        
        # Narrative engagement: captions × humor
        df_enhanced['narrative_engagement'] = (
            df['avg_captioned_reels'] * df['gemini_has_humour']
        )
        
        # Visual storytelling: face density × captions
        df_enhanced['visual_storytelling'] = (
            df['mean_face_density'] * df['avg_captioned_reels']
        )
        
        # Content variety: outlier ratio × humor (variety in storytelling)
        df_enhanced['content_variety'] = (
            df['outlier_2sigma_ratio'] * df['gemini_has_humour']
        )
        
        # Audience connection: agreeing comments × face density
        if 'gemini_comment_sentiment_counts.agreeing' in df.columns:
            if df['gemini_comment_sentiment_counts.agreeing'].max() > 0:
                # Normalize agreeing comments first
                max_agreeing = df['gemini_comment_sentiment_counts.agreeing'].max()
                normalized_agreeing = df['gemini_comment_sentiment_counts.agreeing'] / max_agreeing
                df_enhanced['audience_connection'] = (
                    normalized_agreeing * df['mean_face_density']
                )
            else:
                df_enhanced['audience_connection'] = 0
        
        return df_enhanced
    
    def fit_transform(self, df, target_column='storytelling_score'):
        """Complete preprocessing pipeline."""
        
        # 1. Check if target exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # 2. Separate features and target
        available_features = [f for f in STORYTELLING_FEATURES if f in df.columns]
        if not available_features:
            raise ValueError("No storytelling features found in data")
        
        X = df[available_features].copy()
        y = df[target_column].copy()
        
        # 3. Handle missing values
        X_clean = self.handle_missing_values(X)
        
        # 4. Create interaction features
        X_enhanced = self.create_interaction_features(X_clean)
        
        # 5. Analyze distributions
        self.feature_stats = self.analyze_feature_distributions(X_enhanced)
        
        # 6. Create and fit preprocessing pipeline
        self.preprocessor = self.create_preprocessing_pipeline(self.feature_stats)
        X_processed = self.preprocessor.fit_transform(X_enhanced)
        
        # 7. Store feature names
        self.feature_names = self._get_feature_names_after_preprocessing(X_enhanced)
        
        return X_processed, y
    
    def transform(self, df):
        """Transform new data using fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Apply same preprocessing steps
        available_features = [f for f in STORYTELLING_FEATURES if f in df.columns]
        X = df[available_features].copy()
        
        X_clean = self.handle_missing_values(X)
        X_enhanced = self.create_interaction_features(X_clean)
        X_processed = self.preprocessor.transform(X_enhanced)
        
        return X_processed
    
    def _get_feature_names_after_preprocessing(self, df_enhanced):
        """Get feature names after preprocessing."""
        base_features = [f for f in STORYTELLING_FEATURES if f in df_enhanced.columns]
        interaction_features = [col for col in df_enhanced.columns 
                              if col not in STORYTELLING_FEATURES]
        
        return base_features + interaction_features
    
    def save(self, filepath=None):
        """Save the fitted preprocessor."""
        if filepath is None:
            filepath = MODELS_DIR / "storytelling_preprocessor.pkl"
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Cannot save.")
        
        save_data = {
            'preprocessor': self.preprocessor,
            'feature_stats': self.feature_stats,
            'feature_names': self.feature_names,
            'feature_categories': getattr(self, 'feature_categories', {})
        }
        
        joblib.dump(save_data, filepath)
        print(f"Storytelling preprocessor saved to {filepath}")
    
    def load(self, filepath=None):
        """Load a fitted preprocessor."""
        if filepath is None:
            filepath = MODELS_DIR / "storytelling_preprocessor.pkl"
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        save_data = joblib.load(filepath)
        self.preprocessor = save_data['preprocessor']
        self.feature_stats = save_data['feature_stats']
        self.feature_names = save_data['feature_names']
        self.feature_categories = save_data.get('feature_categories', {})
        
        print(f"Storytelling preprocessor loaded from {filepath}")
    
    def get_feature_importance_info(self):
        """Get information about features for model interpretation."""
        if not self.feature_stats:
            return None
        
        info = {
            'original_features': STORYTELLING_FEATURES,
            'feature_stats': self.feature_stats,
            'feature_categories': getattr(self, 'feature_categories', {}),
            'final_feature_names': self.feature_names
        }
        
        return info