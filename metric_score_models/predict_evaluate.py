#!/usr/bin/env python3
"""
Unified prediction and evaluation script for all scoring models.
Supports credibility, storytelling, communication, and future models.
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys
import json

# Add parent directory to path to import from main project
sys.path.append(str(Path(__file__).parent.parent))

from metric_score_models.combined_scorer import CombinedScorer
from metric_score_models.config import (
    DATA_DIR, INPUT_CSV, RESULTS_DIR,
    CREDIBILITY_FEATURES, STORYTELLING_FEATURES, COMMUNICATION_FEATURES
)

def load_prediction_data(input_path=None):
    """Load data for prediction."""
    print("📂 Loading prediction data...")
    
    if input_path:
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        df = pd.read_csv(input_path)
    else:
        # Use default input file
        default_path = DATA_DIR / INPUT_CSV
        if not default_path.exists():
            raise FileNotFoundError(f"Default input file not found: {default_path}")
        df = pd.read_csv(default_path)
    
    print(f"   Data shape: {df.shape}")
    
    # Check for required features for each model
    feature_sets = {
        'credibility': CREDIBILITY_FEATURES,
        'storytelling': STORYTELLING_FEATURES,
        'communication': COMMUNICATION_FEATURES
    }
    
    available_features = {}
    missing_features = {}
    
    for model_name, features in feature_sets.items():
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        
        available_features[model_name] = available
        missing_features[model_name] = missing
        
        print(f"   Available {model_name} features: {len(available)}/{len(features)}")
        if missing:
            print(f"   ⚠️  Missing {model_name} features: {missing}")
    
    return df, available_features, missing_features

def evaluate_model_performance(scorer, df, model_name, target_column):
    """Evaluate a single model's performance if ground truth is available."""
    if target_column not in df.columns:
        print(f"   ⚠️  No ground truth available for {model_name} ({target_column})")
        return None
    
    try:
        if model_name == 'credibility' and scorer.credibility_scorer.is_fitted:
            metrics = scorer.credibility_scorer.evaluate_predictions(df, target_column)
        elif model_name == 'storytelling' and scorer.storytelling_scorer.is_fitted:
            metrics = scorer.storytelling_scorer.evaluate_predictions(df, target_column)
        elif model_name == 'communication' and scorer.communication_scorer.is_fitted:
            metrics = scorer.communication_scorer.evaluate_predictions(df, target_column)
        else:
            return None
        
        print(f"   📊 {model_name.title()} Model Performance:")
        print(f"     RMSE: {metrics['rmse']:.3f}")
        print(f"     MAE: {metrics['mae']:.3f}")
        print(f"     R²: {metrics['r2']:.3f}")
        print(f"     Samples: {metrics['n_samples']}")
        
        return metrics
    except Exception as e:
        print(f"   ❌ {model_name.title()} evaluation failed: {e}")
        return None

def show_feature_importance(scorer):
    """Display feature importance for all fitted models."""
    print("\n🎯 Feature Importance Analysis:")
    
    models_info = [
        ('credibility', scorer.credibility_scorer),
        ('storytelling', scorer.storytelling_scorer),
        ('communication', scorer.communication_scorer)
    ]
    
    for model_name, model in models_info:
        if model.is_fitted:
            print(f"\n   📈 Top {model_name.title()} Features:")
            importance = model.get_feature_importance()
            
            if isinstance(importance, dict):
                # Sort by importance value
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, imp in sorted_features[:8]:
                    print(f"     {feature}: {imp:.3f}")
            else:
                # DataFrame format
                for _, row in importance.head(8).iterrows():
                    print(f"     {row['feature']}: {row['importance']:.3f}")

def generate_insights_report(results_df, output_dir):
    """Generate detailed insights report."""
    insights_path = output_dir / "model_insights.json"
    
    insights = {
        'summary': {
            'total_creators': len(results_df),
            'timestamp': pd.Timestamp.now().isoformat()
        },
        'score_distributions': {},
        'top_performers': {},
        'correlations': {}
    }
    
    # Score distributions
    score_columns = ['credibility_score', 'storytelling_score', 'communication_score', 'overall_score']
    
    for col in score_columns:
        if col in results_df.columns:
            insights['score_distributions'][col] = {
                'mean': float(results_df[col].mean()),
                'std': float(results_df[col].std()),
                'min': float(results_df[col].min()),
                'max': float(results_df[col].max()),
                'median': float(results_df[col].median())
            }
    
    # Top performers
    if 'overall_score' in results_df.columns:
        top_5 = results_df.nlargest(5, 'overall_score')
        insights['top_performers']['overall'] = []
        
        for _, row in top_5.iterrows():
            performer = {
                'creator': str(row.get('creator', 'Unknown')),
                'overall_score': float(row.get('overall_score', 0))
            }
            
            # Add individual scores if available
            for score_col in ['credibility_score', 'storytelling_score', 'communication_score']:
                if score_col in row:
                    performer[score_col] = float(row[score_col])
            
            insights['top_performers']['overall'].append(performer)
    
    # Score correlations
    numeric_cols = [col for col in score_columns if col in results_df.columns]
    if len(numeric_cols) > 1:
        corr_matrix = results_df[numeric_cols].corr()
        insights['correlations'] = corr_matrix.to_dict()
    
    # Save insights
    with open(insights_path, 'w') as f:
        json.dump(insights, f, indent=2)
    
    print(f"   📋 Detailed insights saved to {insights_path}")
    return insights

def main():
    parser = argparse.ArgumentParser(description='Predict and evaluate all scoring models')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--output-dir', type=str, help='Output directory for results and reports')
    
    # Prediction options
    parser.add_argument('--no-confidence', action='store_true',
                       help='Skip confidence interval calculation')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['credibility', 'storytelling', 'communication', 'all'],
                       default=['all'], help='Which models to use for prediction')
    
    # Analysis options
    parser.add_argument('--show-features', action='store_true',
                       help='Show feature importance from trained models')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model performance (requires ground truth labels)')
    parser.add_argument('--generate-insights', action='store_true',
                       help='Generate detailed insights report')
    
    # Output format options
    parser.add_argument('--format', choices=['csv', 'json', 'both'], default='csv',
                       help='Output format for predictions')
    
    args = parser.parse_args()
    
    try:
        # Setup output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = RESULTS_DIR
        output_dir.mkdir(exist_ok=True)
        
        # Initialize combined scorer
        scorer = CombinedScorer()
        
        # Load models
        scorer.load_models()
        
        # Show feature importance if requested
        if args.show_features:
            show_feature_importance(scorer)
        
        # Load data
        df, available_features, missing_features = load_prediction_data(args.input)
        
        # Determine which models to use
        if 'all' in args.models:
            use_models = ['credibility', 'storytelling', 'communication']
        else:
            use_models = args.models
        
        # Check model availability
        model_status = {
            'credibility': scorer.credibility_scorer.is_fitted,
            'storytelling': scorer.storytelling_scorer.is_fitted,
            'communication': scorer.communication_scorer.is_fitted
        }
        
        available_models = [m for m in use_models if model_status.get(m, False)]
        unavailable_models = [m for m in use_models if not model_status.get(m, False)]
        
        if unavailable_models:
            print(f"⚠️  Models not trained: {unavailable_models}")
            print("   Train them first using individual training scripts")
        
        if not available_models:
            raise ValueError("No trained models available for prediction")
        
        print(f"🎯 Using models: {available_models}")
        
        # Make predictions
        if len(available_models) == 1:
            # Single model prediction
            model_name = available_models[0]
            print(f"🎯 Making {model_name} predictions...")
            
            if model_name == 'credibility':
                model = scorer.credibility_scorer
                score_col = 'credibility_score'
                category_col = 'credibility_category'
                categories = ['Low', 'Medium', 'High', 'Very High']
            elif model_name == 'storytelling':
                model = scorer.storytelling_scorer
                score_col = 'storytelling_score'
                category_col = 'storytelling_category'
                categories = ['Poor', 'Fair', 'Good', 'Excellent']
            elif model_name == 'communication':
                model = scorer.communication_scorer
                score_col = 'communication_score'
                category_col = 'communication_category'
                categories = ['Weak', 'Moderate', 'Strong', 'Excellent']
            
            results_df = df.copy()
            
            if args.no_confidence:
                predictions = model.predict(df)
                results_df[score_col] = predictions
            else:
                pred_results = model.predict_with_confidence(df)
                results_df[score_col] = pred_results['predictions']
                results_df[f'{model_name}_lower_bound'] = pred_results['lower_bound']
                results_df[f'{model_name}_upper_bound'] = pred_results['upper_bound']
                results_df[f'{model_name}_uncertainty'] = pred_results['uncertainty']
            
            # Add category
            results_df[category_col] = pd.cut(
                results_df[score_col],
                bins=[0, 3, 6, 8, 10],
                labels=categories,
                include_lowest=True
            )
            
            # Sort by score
            results_df = results_df.sort_values(score_col, ascending=False)
            
        else:
            # Combined prediction
            print("🎯 Making combined predictions...")
            results_df = scorer.save_combined_predictions(df, None)
        
        # Evaluate models if requested and ground truth available
        if args.evaluate:
            print("\n📊 Model Evaluation:")
            evaluation_results = {}
            
            ground_truth_cols = {
                'credibility': 'credibility_score',
                'storytelling': 'storytelling_score', 
                'communication': 'communication_score'
            }
            
            for model_name in available_models:
                target_col = ground_truth_cols.get(model_name)
                if target_col:
                    metrics = evaluate_model_performance(scorer, df, model_name, target_col)
                    if metrics:
                        evaluation_results[model_name] = metrics
        
        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        if args.output:
            output_path = Path(args.output)
        else:
            if len(available_models) == 1:
                output_path = output_dir / f"{available_models[0]}_predictions_{timestamp}.csv"
            else:
                output_path = output_dir / f"combined_predictions_{timestamp}.csv"
        
        # Save in requested format(s)
        if args.format in ['csv', 'both']:
            results_df.to_csv(output_path, index=False)
            print(f"💾 Predictions saved to {output_path}")
        
        if args.format in ['json', 'both']:
            json_path = output_path.with_suffix('.json')
            results_df.to_json(json_path, orient='records', indent=2)
            print(f"💾 Predictions saved to {json_path}")
        
        # Generate insights report if requested
        if args.generate_insights:
            print("\n📋 Generating insights report...")
            insights = generate_insights_report(results_df, output_dir)
        
        # Print summary
        if len(available_models) > 1:
            scorer.print_combined_summary(results_df)
        else:
            model_name = available_models[0]
            score_col = f"{model_name}_score"
            
            print(f"\n📊 {model_name.title()} Prediction Summary:")
            print(f"   Total creators analyzed: {len(results_df)}")
            print(f"   Score range: {results_df[score_col].min():.2f} - {results_df[score_col].max():.2f}")
            print(f"   Mean score: {results_df[score_col].mean():.2f}")
            
            # Category distribution
            category_col = f"{model_name}_category"
            if category_col in results_df.columns:
                category_counts = results_df[category_col].value_counts()
                print(f"   Score distribution:")
                for category, count in category_counts.items():
                    percentage = (count / len(results_df)) * 100
                    print(f"     {category}: {count} ({percentage:.1f}%)")
        
        print(f"\n✅ Prediction and evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()