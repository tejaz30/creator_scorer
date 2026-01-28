"""
Simple Training Script for All Personality Models
- 80/20 train/validation split
- Organized model saving in folders
- Enhanced evaluation metrics beyond R² and RMSE
- Text evaluation report without emojis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import all models
from aspirational_model import aspirational_model
from cool_model import cool_model
from relatable_model import relatable_model
from credibility_model import CredibilityScorer
from storytelling_model import StorytellingScorer
from simple_communication_model import simple_communication_model

# Import enhanced metrics
from enhanced_metrics import calculate_enhanced_metrics, print_enhanced_metrics

def create_directories():
    """Create organized directory structure."""
    base_dir = Path("trained_models_organized")
    
    directories = [
        base_dir,
        base_dir / "models",
        base_dir / "preprocessors", 
        base_dir / "evaluation_reports"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        
    return base_dir

def save_evaluation_report(results, base_dir):
    """Save comprehensive evaluation report to text file with enhanced metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = base_dir / "evaluation_reports" / f"training_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PERSONALITY MODELS TRAINING EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Data Split: 80% Training / 20% Validation\n")
        f.write(f"Total Models Trained: {len(results)}\n\n")
        
        # Enhanced Summary table
        f.write("ENHANCED METRICS SUMMARY TABLE\n")
        f.write("-" * 140 + "\n")
        f.write(f"{'Model':<15} {'Val R²':<8} {'Val RMSE':<10} {'Val MAE':<8} {'Spearman':<10} {'Kendall':<8} {'Acc±0.5':<8} {'Acc±1.0':<8} {'Top3':<8} {'C-Index':<8}\n")
        f.write("-" * 140 + "\n")
        
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                val_r2 = metrics.get('test_r2', metrics.get('val_r2', 0))
                val_rmse = metrics.get('test_rmse', metrics.get('val_rmse', 0))
                val_mae = metrics.get('val_mae', metrics.get('test_mae', 0))
                
                # Enhanced metrics
                spearman = metrics.get('enhanced_metrics', {}).get('spearman_corr', 0)
                kendall = metrics.get('enhanced_metrics', {}).get('kendall_tau', 0)
                acc_05 = metrics.get('enhanced_metrics', {}).get('accuracy_05', 0)
                acc_10 = metrics.get('enhanced_metrics', {}).get('accuracy_10', 0)
                top3 = metrics.get('enhanced_metrics', {}).get('top3_accuracy', 0)
                c_index = metrics.get('enhanced_metrics', {}).get('concordance_index', 0)
                
                f.write(f"{model_name:<15} {val_r2:<8.3f} {val_rmse:<10.3f} {val_mae:<8.3f} {spearman:<10.3f} {kendall:<8.3f} {acc_05:<8.1f}% {acc_10:<8.1f}% {top3:<8.1f}% {c_index:<8.3f}\n")
            else:
                f.write(f"{model_name:<15} {'FAILED':<8} {'FAILED':<10} {'FAILED':<8} {'FAILED':<10} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED MODEL RESULTS WITH ENHANCED METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        # Detailed results for each model
        for model_name, metrics in results.items():
            f.write(f"{model_name.upper()} MODEL\n")
            f.write("-" * 40 + "\n")
            
            if 'error' in metrics:
                f.write(f"TRAINING FAILED: {metrics['error']}\n\n")
                continue
            
            # Traditional Performance metrics
            f.write("Traditional Performance Metrics:\n")
            train_r2 = metrics.get('train_r2', 0)
            val_r2 = metrics.get('test_r2', metrics.get('val_r2', 0))
            train_rmse = metrics.get('train_rmse', 0)
            val_rmse = metrics.get('test_rmse', metrics.get('val_rmse', 0))
            train_mae = metrics.get('train_mae', 0)
            val_mae = metrics.get('val_mae', metrics.get('test_mae', 0))
            
            f.write(f"  Training R2:     {train_r2:.4f}\n")
            f.write(f"  Validation R2:   {val_r2:.4f}\n")
            f.write(f"  Training RMSE:   {train_rmse:.4f}\n")
            f.write(f"  Validation RMSE: {val_rmse:.4f}\n")
            f.write(f"  Training MAE:    {train_mae:.4f}\n")
            f.write(f"  Validation MAE:  {val_mae:.4f}\n")
            
            # Enhanced metrics section
            if 'enhanced_metrics' in metrics:
                enhanced = metrics['enhanced_metrics']
                f.write("\nEnhanced Evaluation Metrics:\n")
                
                # Correlation metrics
                f.write("  Correlation Analysis:\n")
                f.write(f"    Spearman rho:     {enhanced.get('spearman_corr', 0):.4f} (p={enhanced.get('spearman_p_value', 1):.3f})\n")
                f.write(f"    Kendall tau:      {enhanced.get('kendall_tau', 0):.4f} (p={enhanced.get('kendall_p_value', 1):.3f})\n")
                f.write(f"    Pearson r:        {enhanced.get('pearson_corr', 0):.4f} (p={enhanced.get('pearson_p_value', 1):.3f})\n")
                
                # Accuracy within tolerance
                f.write("  Accuracy Within Tolerance:\n")
                f.write(f"    Within ±0.5:      {enhanced.get('accuracy_05', 0):.1f}%\n")
                f.write(f"    Within ±1.0:      {enhanced.get('accuracy_10', 0):.1f}%\n")
                f.write(f"    Within ±1.5:      {enhanced.get('accuracy_15', 0):.1f}%\n")
                
                # Ranking quality
                f.write("  Ranking Quality:\n")
                f.write(f"    Top-3 Accuracy:   {enhanced.get('top3_accuracy', 0):.1f}%\n")
                f.write(f"    Top-5 Accuracy:   {enhanced.get('top5_accuracy', 0):.1f}%\n")
                f.write(f"    Concordance Index: {enhanced.get('concordance_index', 0):.4f}\n")
                
                # Distribution similarity
                f.write("  Distribution Analysis:\n")
                f.write(f"    Wasserstein Dist: {enhanced.get('wasserstein_distance', 0):.4f}\n")
                f.write(f"    KS Statistic:     {enhanced.get('ks_statistic', 0):.4f} (p={enhanced.get('ks_p_value', 1):.3f})\n")
                
                # Robust metrics
                f.write("  Robust Metrics:\n")
                f.write(f"    Median AE:        {enhanced.get('median_ae', 0):.4f}\n")
                f.write(f"    Error IQR:        {enhanced.get('error_iqr', 0):.4f}\n")
                f.write(f"    Mean Bias:        {enhanced.get('mean_error', 0):.4f}\n")
            
            # Traditional accuracy-like metrics
            if 'val_mape' in metrics:
                f.write(f"  Validation MAPE: {metrics['val_mape']:.2f}%\n")
            elif 'test_mape' in metrics:
                f.write(f"  Validation MAPE: {metrics['test_mape']:.2f}%\n")
            
            if 'cv_r2_mean' in metrics:
                f.write(f"  CV R2 (mean):    {metrics['cv_r2_mean']:.4f}\n")
                f.write(f"  CV R2 (std):     {metrics['cv_r2_std']:.4f}\n")
            
            # Model interpretation with enhanced metrics
            if 'enhanced_metrics' in metrics:
                enhanced = metrics['enhanced_metrics']
                spearman = enhanced.get('spearman_corr', 0)
                concordance = enhanced.get('concordance_index', 0)
                
                f.write("\nModel Quality Assessment:\n")
                if val_r2 > 0.7 and spearman > 0.7:
                    f.write("  Overall Quality: EXCELLENT (Strong linear and monotonic relationships)\n")
                elif val_r2 > 0.5 and spearman > 0.6:
                    f.write("  Overall Quality: GOOD (Moderate relationships)\n")
                elif val_r2 > 0.3 or spearman > 0.5:
                    f.write("  Overall Quality: FAIR (Some predictive power)\n")
                elif val_r2 > 0 or spearman > 0.3:
                    f.write("  Overall Quality: POOR (Limited predictive power)\n")
                else:
                    f.write("  Overall Quality: VERY POOR (No meaningful relationships)\n")
                
                # Ranking quality assessment
                if concordance > 0.7:
                    f.write("  Ranking Quality: EXCELLENT (Strong ranking preservation)\n")
                elif concordance > 0.6:
                    f.write("  Ranking Quality: GOOD (Moderate ranking preservation)\n")
                elif concordance > 0.5:
                    f.write("  Ranking Quality: FAIR (Some ranking preservation)\n")
                else:
                    f.write("  Ranking Quality: POOR (Poor ranking preservation)\n")
            else:
                # Fallback to traditional assessment
                if val_r2 > 0.7:
                    f.write("  Model Quality: EXCELLENT (R2 > 0.7)\n")
                elif val_r2 > 0.5:
                    f.write("  Model Quality: GOOD (R2 > 0.5)\n")
                elif val_r2 > 0.3:
                    f.write("  Model Quality: FAIR (R2 > 0.3)\n")
                elif val_r2 > 0:
                    f.write("  Model Quality: POOR (R2 > 0)\n")
                else:
                    f.write("  Model Quality: VERY POOR (R2 <= 0)\n")
            
            # Overfitting check
            r2_diff = train_r2 - val_r2
            if r2_diff > 0.3:
                f.write("  WARNING: Possible overfitting (large train/val gap)\n")
            elif r2_diff > 0.1:
                f.write("  CAUTION: Some overfitting detected\n")
            else:
                f.write("  OK: Good generalization (low train/val gap)\n")
            
            # Feature importance
            if 'feature_importance' in metrics and metrics['feature_importance']:
                f.write("\n  Top 5 Most Important Features:\n")
                if isinstance(metrics['feature_importance'], dict):
                    sorted_features = sorted(metrics['feature_importance'].items(), 
                                           key=lambda x: x[1], reverse=True)[:5]
                    for i, (feature, importance) in enumerate(sorted_features, 1):
                        f.write(f"    {i}. {feature}: {importance:.4f}\n")
            
            f.write("\n")
        
        # Enhanced training recommendations
        f.write("=" * 80 + "\n")
        f.write("ENHANCED TRAINING RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n")
        
        successful_models = [name for name, metrics in results.items() if 'error' not in metrics]
        failed_models = [name for name, metrics in results.items() if 'error' in metrics]
        
        if successful_models:
            f.write(f"Successfully trained models: {', '.join(successful_models)}\n")
            
            # Find best performing model by multiple criteria
            best_r2_model = max(successful_models, 
                               key=lambda x: results[x].get('test_r2', results[x].get('val_r2', 0)))
            best_r2_score = results[best_r2_model].get('test_r2', results[best_r2_model].get('val_r2', 0))
            
            # Best by Spearman correlation (better for non-linear relationships)
            best_spearman_model = None
            best_spearman_score = 0
            for model in successful_models:
                spearman = results[model].get('enhanced_metrics', {}).get('spearman_corr', 0)
                if spearman > best_spearman_score:
                    best_spearman_score = spearman
                    best_spearman_model = model
            
            f.write(f"Best R² model: {best_r2_model} (R² = {best_r2_score:.3f})\n")
            if best_spearman_model:
                f.write(f"Best Spearman model: {best_spearman_model} (ρ = {best_spearman_score:.3f})\n")
            
            # Recommendations based on enhanced metrics
            f.write("\nModel Selection Recommendations:\n")
            f.write("- For linear relationships: Use R² as primary metric\n")
            f.write("- For monotonic relationships: Use Spearman correlation\n")
            f.write("- For ranking tasks: Focus on Concordance Index and Top-K accuracy\n")
            f.write("- For business applications: Consider accuracy within tolerance (±1.0)\n")
        
        if failed_models:
            f.write(f"Failed models: {', '.join(failed_models)}\n")
            f.write("Recommendation: Check feature compatibility and data quality\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF ENHANCED EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Enhanced evaluation report saved to: {report_path}")
    return report_path

def train_all_models_simple(input_csv="../final_creator_scores_with_targets.csv"):
    """Train all models with organized structure and detailed evaluation."""
    
    print("Starting Organized Model Training Pipeline")
    print("="*70)
    
    # Create directory structure
    base_dir = create_directories()
    print(f"Created organized directory structure in: {base_dir}")
    
    # Load data
    input_path = Path(input_csv)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return
    
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} creators with {len(df.columns)} features")
    
    # Identify target columns (last 6 columns)
    target_columns = df.columns[-6:].tolist()
    print(f"Target variables identified: {target_columns}")
    
    # Model training configuration
    models_config = [
        {
            'name': 'Aspirational',
            'model': aspirational_model,
            'target': 'aspirational',
            'model_file': 'aspirational_model.joblib',
            'preprocessor_file': 'aspirational_preprocessor.joblib'
        },
        {
            'name': 'Cool', 
            'model': cool_model,
            'target': 'cool',
            'model_file': 'cool_model.joblib',
            'preprocessor_file': 'cool_preprocessor.joblib'
        },
        {
            'name': 'Relatable',
            'model': relatable_model, 
            'target': 'relatable',
            'model_file': 'relatable_model.joblib',
            'preprocessor_file': 'relatable_preprocessor.joblib'
        },
        {
            'name': 'Credibility',
            'model': CredibilityScorer(),
            'target': 'credible',
            'model_file': 'credibility_model.joblib',
            'preprocessor_file': 'credibility_preprocessor.joblib'
        },
        {
            'name': 'Communication',
            'model': simple_communication_model,
            'target': 'communication', 
            'model_file': 'communication_model.joblib',
            'preprocessor_file': 'communication_preprocessor.joblib'
        },
        {
            'name': 'Storytelling',
            'model': StorytellingScorer(),
            'target': 'story_telling',
            'model_file': 'storytelling_model.joblib',
            'preprocessor_file': 'storytelling_preprocessor.joblib'
        }
    ]
    
    # Train all models
    training_results = {}
    
    for config in models_config:
        model_name = config['name']
        model = config['model']
        target_col = config['target']
        model_file = config['model_file']
        
        print(f"\n{'='*50}")
        print(f"Training {model_name} Model")
        print(f"{'='*50}")
        
        # Check if target column exists
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found, skipping {model_name}")
            training_results[model_name] = {'error': f"Target column '{target_col}' not found"}
            continue
        
        # Check for missing values in target
        missing_targets = df[target_col].isna().sum()
        if missing_targets > 0:
            print(f"Found {missing_targets} missing values in target, filling with median")
            df[target_col] = df[target_col].fillna(df[target_col].median())
        
        try:
            # Train the model with 80/20 split
            print(f"Training {model_name} model with 80/20 split...")
            
            # Handle different model types
            if model_name in ['Credibility', 'Storytelling']:
                # These are class instances with train method
                train_results = model.train(df, target_col, test_size=0.2)
                
                # Get metrics from the model's training_metrics
                results = model.training_metrics.copy()
                
                # Save model and preprocessor to organized folders
                model_path = base_dir / "models" / model_file
                model.save(model_path)
                
            elif model_name == 'Communication':
                # Simple communication model
                results = model.train(df, target_col, test_size=0.2)
                
                # Save to organized folder
                model_path = base_dir / "models" / model_file
                model.save(model_path)
                
            else:
                # These are global model instances with fit method (Aspirational, Cool, Relatable)
                results = model.fit(df, target_col, test_size=0.2)
                
                # Save model to organized folders
                model_path = base_dir / "models" / model_file
                model.save(model_path)
            
            print(f"Model saved to: {base_dir / 'models' / model_file}")
            
            # Store results
            training_results[model_name] = results
            
            # Print results
            print(f"\n{model_name} Training Results:")
            if 'train_r2' in results:
                train_r2 = results['train_r2']
                val_r2 = results.get('test_r2', results.get('val_r2', 0))
                train_rmse = results['train_rmse']
                val_rmse = results.get('test_rmse', results.get('val_rmse', 0))
                
                print(f"   Training R2:     {train_r2:.3f}")
                print(f"   Validation R2:   {val_r2:.3f}")
                print(f"   Training RMSE:   {train_rmse:.3f}")
                print(f"   Validation RMSE: {val_rmse:.3f}")
                
                if 'cv_r2_mean' in results:
                    print(f"   CV R2 (mean ± std): {results['cv_r2_mean']:.3f} ± {results['cv_r2_std']:.3f}")
            
        except Exception as e:
            print(f"Failed to train {model_name} model: {e}")
            training_results[model_name] = {'error': str(e)}
    
    # Save evaluation report
    report_path = save_evaluation_report(training_results, base_dir)
    
    # Print overall summary
    print(f"\n{'='*70}")
    print(f"Training Summary")
    print(f"{'='*70}")
    
    successful_models = []
    failed_models = []
    
    for model_name, results in training_results.items():
        if 'error' in results:
            failed_models.append(model_name)
            print(f"{model_name}: FAILED - {results['error']}")
        else:
            successful_models.append(model_name)
            val_r2 = results.get('test_r2', results.get('val_r2', 0))
            print(f"{model_name}: Validation R2 = {val_r2:.3f}")
    
    print(f"\nFinal Results:")
    print(f"   Successfully trained: {len(successful_models)} models")
    print(f"   Failed: {len(failed_models)} models")
    
    if successful_models:
        print(f"   Successful models: {', '.join(successful_models)}")
        
        # Show best performing model
        best_model = max(successful_models, 
                        key=lambda x: training_results[x].get('test_r2', training_results[x].get('val_r2', 0)))
        best_score = training_results[best_model].get('test_r2', training_results[best_model].get('val_r2', 0))
        print(f"   Best performing: {best_model} (Validation R2 = {best_score:.3f})")
    
    if failed_models:
        print(f"   Failed models: {', '.join(failed_models)}")
    
    print(f"\nNext Steps:")
    print(f"   1. Check evaluation report: {report_path}")
    print(f"   2. Models saved in: {base_dir / 'models'}")
    print(f"   3. Run predictions using organized models")
    
    return training_results, base_dir

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all personality models with organized structure")
    parser.add_argument("--input", default="../final_creator_scores_with_targets.csv", 
                       help="Input CSV file with features and target variables")
    
    args = parser.parse_args()
    
    train_all_models_simple(args.input)

if __name__ == "__main__":
    main()