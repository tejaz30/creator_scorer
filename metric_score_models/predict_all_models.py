"""
Unified prediction script for all 6 personality models.
Generates predictions for: Credibility, Storytelling, Communication, Aspirational, Cool, Relatable
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from combined_scorer_v2 import CombinedScorerV2

def predict_all_models(input_csv="../test_set.csv", output_csv="all_personality_predictions.csv"):
    """Generate predictions for all 6 personality models."""
    
    print("🚀 Starting All Models Prediction Pipeline")
    print("="*60)
    
    # Load data
    input_path = Path(input_csv)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    print(f"📊 Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} creators with {len(df.columns)} features")
    
    # Initialize combined scorer
    print(f"\n🔄 Initializing combined scorer...")
    combined_scorer = CombinedScorerV2()
    
    # Load all models
    print(f"🔄 Loading trained models...")
    loaded_models = combined_scorer.load_models()
    
    if not loaded_models:
        print("❌ No models were loaded successfully!")
        print("   Make sure to train the models first using the individual training scripts.")
        return
    
    print(f"✅ Successfully loaded {len(loaded_models)} models: {', '.join(loaded_models)}")
    
    # Print feature summary
    combined_scorer.print_feature_summary()
    
    # Generate predictions
    print(f"\n🎯 Generating predictions for all models...")
    results_df = combined_scorer.predict_all_scores(df)
    
    # Add categories for each score
    score_columns = [col for col in results_df.columns if col.endswith('_score')]
    
    for score_col in score_columns:
        if score_col == 'overall_personality_score':
            category_col = 'overall_personality_category'
            labels = ['Poor', 'Below Average', 'Average', 'Above Average', 'Excellent']
        else:
            category_col = score_col.replace('_score', '_category')
            labels = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        
        results_df[category_col] = pd.cut(
            results_df[score_col],
            bins=[0, 2, 4, 6, 8, 10],
            labels=labels,
            include_lowest=True
        )
    
    # Sort by overall personality score
    if 'overall_personality_score' in results_df.columns:
        results_df = results_df.sort_values('overall_personality_score', ascending=False)
    
    # Save results
    output_path = Path(output_csv)
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to {output_path}")
    
    # Print summary
    print_prediction_summary(results_df)
    
    return results_df

def print_prediction_summary(results_df):
    """Print a summary of the predictions."""
    print("\n📊 Prediction Summary:")
    print("="*60)
    
    score_columns = [col for col in results_df.columns if col.endswith('_score')]
    
    for score_col in score_columns:
        model_name = score_col.replace('_score', '').replace('_', ' ').title()
        scores = results_df[score_col]
        
        print(f"\n🎯 {model_name}:")
        print(f"   Mean: {scores.mean():.2f}")
        print(f"   Range: {scores.min():.2f} - {scores.max():.2f}")
        print(f"   Std Dev: {scores.std():.2f}")
        
        # Show category distribution if available
        category_col = score_col.replace('_score', '_category')
        if category_col in results_df.columns:
            category_counts = results_df[category_col].value_counts()
            print(f"   Distribution:")
            for category, count in category_counts.items():
                percentage = (count / len(results_df)) * 100
                print(f"     {category}: {count} ({percentage:.1f}%)")
    
    # Show top performers
    if 'overall_personality_score' in results_df.columns:
        print(f"\n🏆 Top 5 Overall Performers:")
        top_5 = results_df.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            creator = row.get('creator', f'Creator_{i}')
            overall = row.get('overall_personality_score', 0)
            print(f"   {i}. {creator}: {overall:.2f}")
            
            # Show individual scores
            individual_scores = []
            for model in ['credibility', 'storytelling', 'communication', 'aspirational', 'cool', 'relatable']:
                score_col = f'{model}_score'
                if score_col in row:
                    individual_scores.append(f"{model.title()}: {row[score_col]:.2f}")
            
            if individual_scores:
                print(f"      {' | '.join(individual_scores)}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate predictions for all personality models")
    parser.add_argument("--input", default="../test_set.csv",
                       help="Input CSV file with creator features (default: ../test_set.csv)")
    parser.add_argument("--output", default="all_personality_predictions.csv",
                       help="Output CSV file for predictions (default: all_personality_predictions.csv)")
    
    args = parser.parse_args()
    
    predict_all_models(args.input, args.output)

if __name__ == "__main__":
    main()