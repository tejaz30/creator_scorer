#!/usr/bin/env python3
"""
Training script for storytelling scoring model.
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import from main project
sys.path.append(str(Path(__file__).parent.parent))

from metric_score_models.storytelling_model import StorytellingScorer
from metric_score_models.config import (
    DATA_DIR, INPUT_CSV, LABELED_DATA_CSV, RESULTS_DIR,
    STORYTELLING_FEATURES
)

def load_training_data(input_path=None, use_sample_labels=False):
    """Load training data for storytelling model."""
    print("📂 Loading storytelling training data...")
    
    # Try to load existing labeled data first
    labeled_path = DATA_DIR / "storytelling_labeled_data.csv"
    
    if input_path:
        print(f"   Loading data from {input_path}")
        df = pd.read_csv(input_path)
    elif labeled_path.exists() and not use_sample_labels:
        print(f"   Loading labeled data from {labeled_path}")
        df = pd.read_csv(labeled_path)
    else:
        # Use default input file
        input_path = DATA_DIR / INPUT_CSV
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        print(f"   Loading feature data from {input_path}")
        df = pd.read_csv(input_path)
    
    print(f"   Data shape: {df.shape}")
    
    # Check available features
    available_features = [f for f in STORYTELLING_FEATURES if f in df.columns]
    missing_features = [f for f in STORYTELLING_FEATURES if f not in df.columns]
    
    print(f"   Available storytelling features: {len(available_features)}/{len(STORYTELLING_FEATURES)}")
    
    if missing_features:
        print(f"   ⚠️  Missing storytelling features: {missing_features}")
        print("   Model will use available features and set missing ones to 0")
    
    return df

def create_sample_storytelling_labels(df):
    """Create sample storytelling labels for demonstration."""
    print("📝 Creating sample storytelling labels...")
    print("   Note: In production, use human-labeled storytelling scores!")
    
    df_labeled = df.copy()
    
    # Create storytelling scores based on available features
    storytelling_score = 5.0  # Base score
    
    # Captions enhance storytelling accessibility
    if 'avg_captioned_reels' in df.columns:
        storytelling_score += df['avg_captioned_reels'] * 2.0
    
    # Humor makes stories more engaging
    if 'gemini_has_humour' in df.columns:
        storytelling_score += df['gemini_has_humour'] * 1.8
    
    # Face presence creates personal connection in stories
    if 'mean_face_density' in df.columns:
        storytelling_score += np.clip(df['mean_face_density'] / 4, 0, 1.2)
    
    # Content variety indicates rich storytelling
    if 'outlier_2sigma_ratio' in df.columns:
        storytelling_score += df['outlier_2sigma_ratio'] * 0.8
    
    # Positive audience engagement with stories
    if 'gemini_comment_sentiment_counts.agreeing' in df.columns:
        max_agreeing = df['gemini_comment_sentiment_counts.agreeing'].max()
        if max_agreeing > 0:
            storytelling_score += (df['gemini_comment_sentiment_counts.agreeing'] / max_agreeing) * 1.2
    
    # Neutral comments might indicate thoughtful storytelling
    if 'gemini_comment_sentiment_counts.neutral' in df.columns:
        max_neutral = df['gemini_comment_sentiment_counts.neutral'].max()
        if max_neutral > 0:
            storytelling_score += (df['gemini_comment_sentiment_counts.neutral'] / max_neutral) * 0.6
    
    # Add realistic noise and clip to valid range
    np.random.seed(43)
    noise = np.random.normal(0, 0.6, len(df))
    storytelling_score += noise
    df_labeled['storytelling_score'] = np.clip(storytelling_score, 1, 10)
    
    print(f"   Generated storytelling scores: {df_labeled['storytelling_score'].min():.2f} - {df_labeled['storytelling_score'].max():.2f}")
    print(f"   Mean storytelling score: {df_labeled['storytelling_score'].mean():.2f}")
    
    return df_labeled

def main():
    parser = argparse.ArgumentParser(description='Train storytelling scoring model')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--sample-labels', action='store_true', 
                       help='Create sample labels (for demo purposes)')
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--save-labeled-data', action='store_true',
                       help='Save labeled data for future use')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_training_data(args.input, args.sample_labels)
        
        # Initialize storytelling scorer
        scorer = StorytellingScorer()
        
        # Create sample labels if needed
        if 'storytelling_score' not in df.columns or args.sample_labels:
            df = create_sample_storytelling_labels(df)
            
            # Save labeled data if requested
            if args.save_labeled_data:
                labeled_path = DATA_DIR / "storytelling_labeled_data.csv"
                df.to_csv(labeled_path, index=False)
                print(f"   💾 Labeled data saved to {labeled_path}")
        
        # Hyperparameter tuning if requested
        if args.tune_hyperparams:
            print("🔧 Performing hyperparameter tuning...")
            try:
                best_params = scorer.hyperparameter_tuning(df, 'storytelling_score')
                print(f"   Best parameters: {best_params}")
            except AttributeError:
                print("   ⚠️  Hyperparameter tuning not implemented for this model")
        
        # Train model
        print("🚀 Training storytelling model...")
        results = scorer.train(df, 'storytelling_score', args.test_size)
        
        # Save model
        scorer.save()
        print("   ✅ Model saved successfully")
        
        # Generate feature importance plot if available
        plots_dir = RESULTS_DIR / "training_plots"
        plots_dir.mkdir(exist_ok=True)
        
        try:
            plot_path = plots_dir / "storytelling_feature_importance.png"
            scorer.plot_feature_importance(save_path=plot_path)
            print(f"   📊 Feature importance plot saved to {plot_path}")
        except AttributeError:
            print("   Note: Feature importance plotting not available")
        
        # Print detailed results
        print(f"\n📊 Storytelling Model Training Results:")
        print(f"   Training RMSE: {results['train_rmse']:.3f}")
        print(f"   Test RMSE: {results['test_rmse']:.3f}")
        print(f"   Training R²: {results['train_r2']:.3f}")
        print(f"   Test R²: {results['test_r2']:.3f}")
        print(f"   CV R² (mean ± std): {results['cv_r2_mean']:.3f} ± {results['cv_r2_std']:.3f}")
        
        # Show top features
        if scorer.feature_importance is not None:
            print(f"\n🎯 Top 5 Most Important Features:")
            for _, row in scorer.feature_importance.head().iterrows():
                print(f"     {row['feature']}: {row['importance']:.3f}")
        
        print(f"\n✅ Storytelling model training completed successfully!")
        print(f"   Use predict_evaluate.py --models storytelling to make predictions")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()