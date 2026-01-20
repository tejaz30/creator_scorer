"""
Communication Model Preprocessor

This module handles data preprocessing specifically for the communication scoring model.
It includes feature engineering, validation, and data quality checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CommunicationPreprocessor:
    """Preprocessor for communication scoring model data."""
    
    def __init__(self):
        self.required_features = [
            'eye_contact_avg_score_0_10',
            'avg_captioned_reels',
            'gemini_is_marketing', 
            'gemini_has_humour',
            'mean_face_density'
        ]
        
        self.feature_ranges = {
            'eye_contact_avg_score_0_10': (0, 10),
            'avg_captioned_reels': (0, 1),
            'gemini_is_marketing': (0, 1),
            'gemini_has_humour': (0, 1),
            'mean_face_density': (0, 10)  # Assuming reasonable upper bound
        }
    
    def validate_data(self, df):
        """Validate input data quality and completeness."""
        print("Validating data for communication model...")
        
        issues = []
        
        # Check required columns
        missing_cols = [col for col in self.required_features if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check data types and ranges
        for feature in self.required_features:
            if feature in df.columns:
                # Check for non-numeric data
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    issues.append(f"Non-numeric data in {feature}")
                
                # Check value ranges
                if feature in self.feature_ranges:
                    min_val, max_val = self.feature_ranges[feature]
                    out_of_range = df[(df[feature] < min_val) | (df[feature] > max_val)][feature]
                    if len(out_of_range) > 0:
                        issues.append(f"{feature} has {len(out_of_range)} values outside range [{min_val}, {max_val}]")
        
        # Check for excessive missing values
        for feature in self.required_features:
            if feature in df.columns:
                missing_pct = df[feature].isnull().sum() / len(df) * 100
                if missing_pct > 50:
                    issues.append(f"{feature} has {missing_pct:.1f}% missing values")
        
        if issues:
            print("Data validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False, issues
        else:
            print("✓ Data validation passed")
            return True, []
    
    def clean_data(self, df):
        """Clean and prepare data for the communication model."""
        df_clean = df.copy()
        
        print("Cleaning data for communication model...")
        
        # Handle missing values with communication-specific logic
        for feature in self.required_features:
            if feature in df_clean.columns:
                missing_count = df_clean[feature].isnull().sum()
                if missing_count > 0:
                    print(f"  Handling {missing_count} missing values in {feature}")
                    
                    if feature == 'eye_contact_avg_score_0_10':
                        # Default to moderate eye contact
                        df_clean[feature] = df_clean[feature].fillna(5.0)
                    elif feature == 'avg_captioned_reels':
                        # Default to no captions
                        df_clean[feature] = df_clean[feature].fillna(0.0)
                    elif feature in ['gemini_is_marketing', 'gemini_has_humour']:
                        # Default to no marketing/humor
                        df_clean[feature] = df_clean[feature].fillna(0.0)
                    elif feature == 'mean_face_density':
                        # Default to moderate face presence
                        df_clean[feature] = df_clean[feature].fillna(2.0)
            else:
                print(f"  Creating missing feature {feature} with default values")
                if feature == 'eye_contact_avg_score_0_10':
                    df_clean[feature] = 5.0
                elif feature == 'avg_captioned_reels':
                    df_clean[feature] = 0.0
                elif feature in ['gemini_is_marketing', 'gemini_has_humour']:
                    df_clean[feature] = 0.0
                elif feature == 'mean_face_density':
                    df_clean[feature] = 2.0
        
        # Clip values to valid ranges
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in df_clean.columns:
                original_range = (df_clean[feature].min(), df_clean[feature].max())
                df_clean[feature] = np.clip(df_clean[feature], min_val, max_val)
                new_range = (df_clean[feature].min(), df_clean[feature].max())
                if original_range != new_range:
                    print(f"  Clipped {feature} from {original_range} to {new_range}")
        
        return df_clean
    
    def engineer_features(self, df):
        """Create communication-specific engineered features."""
        df_enhanced = df.copy()
        
        print("Engineering features for communication model...")
        
        # Check if required features exist
        required_features = ['eye_contact_avg_score_0_10', 'avg_captioned_reels', 
                           'gemini_is_marketing', 'gemini_has_humour', 'mean_face_density']
        
        missing_features = [f for f in required_features if f not in df_enhanced.columns]
        if missing_features:
            print(f"Warning: Missing features for engineering: {missing_features}")
            return df_enhanced
        
        # Direct communication score: eye contact + face presence
        df_enhanced['direct_communication_score'] = (
            df_enhanced['eye_contact_avg_score_0_10'] / 10.0 * 0.7 +
            np.clip(df_enhanced['mean_face_density'] / 5.0, 0, 1) * 0.3
        )
        
        # Accessibility communication: caption usage
        df_enhanced['accessibility_communication'] = df_enhanced['avg_captioned_reels']
        
        # Authentic communication: non-marketing content
        df_enhanced['authentic_communication'] = 1 - df_enhanced['gemini_is_marketing']
        
        # Engaging communication: humor usage
        df_enhanced['engaging_communication'] = df_enhanced['gemini_has_humour']
        
        # Overall communication potential
        df_enhanced['communication_potential'] = (
            df_enhanced['direct_communication_score'] * 0.3 +
            df_enhanced['accessibility_communication'] * 0.2 +
            df_enhanced['authentic_communication'] * 0.25 +
            df_enhanced['engaging_communication'] * 0.25
        )
        
        # Communication consistency (low variance indicates consistent communication style)
        communication_features = [
            'direct_communication_score',
            'accessibility_communication', 
            'authentic_communication',
            'engaging_communication'
        ]
        
        df_enhanced['communication_consistency'] = 1 - df_enhanced[communication_features].std(axis=1)
        df_enhanced['communication_consistency'] = df_enhanced['communication_consistency'].fillna(1.0)
        
        print(f"  Created {len(communication_features) + 2} engineered features")
        
        return df_enhanced
    
    def prepare_for_training(self, df, target_column='communication'):
        """Complete preprocessing pipeline for training data."""
        print(f"\nPreparing data for communication model training...")
        print(f"Input shape: {df.shape}")
        
        # Validate data
        is_valid, issues = self.validate_data(df)
        if not is_valid:
            print("Warning: Data validation failed, proceeding with cleaning...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Engineer features
        df_enhanced = self.engineer_features(df_clean)
        
        # Check target variable
        if target_column in df_enhanced.columns:
            target_stats = df_enhanced[target_column].describe()
            print(f"\nTarget variable ({target_column}) statistics:")
            print(target_stats)
            
            # Check for valid target range (assuming 1-10 scale)
            if target_stats['min'] < 1 or target_stats['max'] > 10:
                print(f"Warning: Target values outside expected range [1, 10]")
        else:
            print(f"Warning: Target column '{target_column}' not found")
        
        print(f"Final shape: {df_enhanced.shape}")
        print("✓ Data preparation complete")
        
        return df_enhanced
    
    def prepare_for_prediction(self, df):
        """Preprocessing pipeline for prediction data (no target variable)."""
        print(f"\nPreparing data for communication prediction...")
        print(f"Input shape: {df.shape}")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Engineer features  
        df_enhanced = self.engineer_features(df_clean)
        
        print(f"Final shape: {df_enhanced.shape}")
        print("✓ Prediction data preparation complete")
        
        return df_enhanced
    
    def get_feature_summary(self, df):
        """Generate summary statistics for communication features."""
        summary = {}
        
        for feature in self.required_features:
            if feature in df.columns:
                stats = df[feature].describe()
                summary[feature] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'missing_count': df[feature].isnull().sum(),
                    'missing_pct': df[feature].isnull().sum() / len(df) * 100
                }
        
        return summary


def main():
    """Example usage of the Communication Preprocessor."""
    print("Communication Model Preprocessor Demo")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = CommunicationPreprocessor()
    
    # Example usage would be:
    # df = pd.read_csv('raw_data.csv')
    # df_processed = preprocessor.prepare_for_training(df)
    # summary = preprocessor.get_feature_summary(df_processed)
    
    print("Preprocessor initialized successfully!")
    print("Required features:", preprocessor.required_features)


if __name__ == "__main__":
    main()