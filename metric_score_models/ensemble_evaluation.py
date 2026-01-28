"""
Comprehensive Ensemble Model Evaluation
Compares ensemble models against individual models and provides detailed analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import ensemble models
from ensemble_models.aspirational_ensemble import aspirational_ensemble
from ensemble_models.cool_ensemble import cool_ensemble
from ensemble_models.relatable_ensemble import relatable_ensemble
from ensemble_models.credibility_ensemble import credibility_ensemble
from ensemble_models.storytelling_ensemble import storytelling_ensemble
from ensemble_models.communication_ensemble import communication_ensemble

# Import individual models for comparison
from aspirational_model import aspirational_model
from cool_model import cool_model
from relatable_model import relatable_model
from credibility_model import CredibilityScorer
from storytelling_model import StorytellingScorer
from simple_communication_model import simple_communication_model

# Import enhanced metrics
from enhanced_metrics import calculate_enhanced_metrics, print_enhanced_metrics

def load_ensemble_models(base_dir="trained_models_organized"):
    """Load all trained ensemble models."""
    base_path = Path(base_dir)
    ensemble_path = base_path / "ensemble_models"
    
    models = {}
    
    ensemble_files = [
        ('Aspirational', aspirational_ensemble, 'aspirational_ensemble.joblib'),
        ('Cool', cool_ensemble, 'cool_ensemble.joblib'),
        ('Relatable', relatable_ensemble, 'relatable_ensemble.joblib'),
        ('Credibility', credibility_ensemble, 'credibility_ensemble.joblib'),
        ('Communication', communication_ensemble, 'communication_ensemble.joblib'),
        ('Storytelling', storytelling_ensemble, 'storytelling_ensemble.joblib')
    ]
    
    for name, model, filename in ensemble_files:
        filepath = ensemble_path / filename
        if filepath.exists():
            try:
                model.load(filepath)
                models[name] = model
                print(f"✅ Loaded {name} ensemble model")
            except Exception as e:
                print(f"❌ Failed to load {name} ensemble model: {e}")
        else:
            print(f"⚠️ {name} ensemble model file not found: {filepath}")
    
    return models

def load_individual_models(base_dir="trained_models_organized"):
    """Load all trained individual models for comparison."""
    base_path = Path(base_dir)
    models_path = base_path / "models"
    
    models = {}
    
    individual_files = [
        ('Aspirational', aspirational_model, 'aspirational_model.joblib'),
        ('Cool', cool_model, 'cool_model.joblib'),
        ('Relatable', relatable_model, 'relatable_model.joblib'),
        ('Credibility', CredibilityScorer(), 'credibility_model.joblib'),
        ('Communication', simple_communication_model, 'communication_model.joblib'),
        ('Storytelling', StorytellingScorer(), 'storytelling_model.joblib')
    ]
    
    for name, model, filename in individual_files:
        filepath = models_path / filename
        if filepath.exists():
            try:
                model.load(filepath)
                models[name] = model
                print(f"✅ Loaded {name} individual model")
            except Exception as e:
                print(f"❌ Failed to load {name} individual model: {e}")
        else:
            print(f"⚠️ {name} individual model file not found: {filepath}")
    
    return models

def evaluate_model_predictions(model, df, target_col, model_name):
    """Evaluate a model's predictions with enhanced metrics."""
    try:
        # Make predictions
        predictions = model.predict(df)
        
        # Get actual values
        actual = df[target_col].dropna()
        
        # Align predictions and actual values
        min_len = min(len(predictions), len(actual))
        predictions = predictions[:min_len]
        actual = actual.iloc[:min_len]
        
        # Calculate enhanced metrics
        enhanced_metrics = calculate_enhanced_metrics(actual, predictions, model_name)
        
        return {
            'predictions': predictions,
            'actual': actual,
            'enhanced_metrics': enhanced_metrics
        }
    except Exception as e:
        print(f"❌ Failed to evaluate {model_name}: {e}")
        return None

def compare_ensemble_vs_individual(ensemble_models, individual_models, df):
    """Compare ensemble models against individual models."""
    
    target_mapping = {
        'Aspirational': 'aspirational',
        'Cool': 'cool',
        'Relatable': 'relatable',
        'Credibility': 'credible',
        'Communication': 'communication',
        'Storytelling': 'story_telling'
    }
    
    comparison_results = {}
    
    for model_name in ensemble_models.keys():
        if model_name in individual_models:
            target_col = target_mapping[model_name]
            
            if target_col not in df.columns:
                print(f"⚠️ Target column '{target_col}' not found for {model_name}")
                continue
            
            print(f"\n🔍 Evaluating {model_name} models...")
            
            # Evaluate ensemble model
            ensemble_results = evaluate_model_predictions(
                ensemble_models[model_name], df, target_col, f"{model_name} Ensemble"
            )
            
            # Evaluate individual model
            individual_results = evaluate_model_predictions(
                individual_models[model_name], df, target_col, f"{model_name} Individual"
            )
            
            if ensemble_results and individual_results:
                comparison_results[model_name] = {
                    'ensemble': ensemble_results,
                    'individual': individual_results,
                    'target_col': target_col
                }
    
    return comparison_results

def create_comparison_plots(comparison_results, save_dir="trained_models_organized/ensemble_evaluation_reports"):
    """Create visualization plots comparing ensemble vs individual models."""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ensemble vs Individual Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = list(comparison_results.keys())
    metrics_to_plot = ['r2', 'spearman_corr', 'accuracy_10', 'concordance_index']
    metric_names = ['R² Score', 'Spearman Correlation', 'Accuracy ±1.0 (%)', 'Concordance Index']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        ensemble_scores = []
        individual_scores = []
        
        for model in models:
            if model in comparison_results:
                ensemble_metric = comparison_results[model]['ensemble']['enhanced_metrics'].get(metric, 0)
                individual_metric = comparison_results[model]['individual']['enhanced_metrics'].get(metric, 0)
                
                # Convert percentage metrics
                if metric == 'accuracy_10':
                    ensemble_metric = ensemble_metric  # Already in percentage
                    individual_metric = individual_metric  # Already in percentage
                
                ensemble_scores.append(ensemble_metric)
                individual_scores.append(individual_metric)
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ensemble_scores, width, label='Ensemble', alpha=0.8)
        bars2 = ax.bar(x + width/2, individual_scores, width, label='Individual', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path / 'ensemble_vs_individual_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Improvement Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    improvement_data = []
    for model in models:
        if model in comparison_results:
            row = []
            for metric in metrics_to_plot:
                ensemble_score = comparison_results[model]['ensemble']['enhanced_metrics'].get(metric, 0)
                individual_score = comparison_results[model]['individual']['enhanced_metrics'].get(metric, 0)
                improvement = ensemble_score - individual_score
                row.append(improvement)
            improvement_data.append(row)
    
    improvement_df = pd.DataFrame(improvement_data, index=models, columns=metric_names)
    
    sns.heatmap(improvement_df, annot=True, cmap='RdYlGn', center=0, 
                fmt='.3f', ax=ax, cbar_kws={'label': 'Improvement (Ensemble - Individual)'})
    ax.set_title('Ensemble Model Improvement Over Individual Models', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Models')
    
    plt.tight_layout()
    plt.savefig(save_path / 'ensemble_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Prediction Scatter Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Actual vs Predicted Values: Ensemble vs Individual Models', fontsize=16, fontweight='bold')
    
    for idx, model in enumerate(models[:6]):  # Limit to 6 models
        if model in comparison_results:
            ax = axes[idx // 3, idx % 3]
            
            ensemble_data = comparison_results[model]['ensemble']
            individual_data = comparison_results[model]['individual']
            
            # Plot ensemble predictions
            ax.scatter(ensemble_data['actual'], ensemble_data['predictions'], 
                      alpha=0.6, label='Ensemble', s=30)
            
            # Plot individual predictions
            ax.scatter(individual_data['actual'], individual_data['predictions'], 
                      alpha=0.6, label='Individual', s=30)
            
            # Add perfect prediction line
            min_val = min(ensemble_data['actual'].min(), individual_data['actual'].min())
            max_val = max(ensemble_data['actual'].max(), individual_data['actual'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{model} Model Predictions')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'prediction_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization plots saved to: {save_path}")

def generate_comprehensive_report(comparison_results, save_dir="trained_models_organized/ensemble_evaluation_reports"):
    """Generate a comprehensive comparison report."""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = save_path / f"ensemble_vs_individual_comparison_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE ENSEMBLE VS INDIVIDUAL MODEL COMPARISON REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models Compared: {len(comparison_results)}\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 50 + "\n")
        
        total_improvements = 0
        significant_improvements = 0
        
        for model_name, results in comparison_results.items():
            ensemble_r2 = results['ensemble']['enhanced_metrics'].get('r2', 0)
            individual_r2 = results['individual']['enhanced_metrics'].get('r2', 0)
            improvement = ensemble_r2 - individual_r2
            
            total_improvements += improvement
            if improvement > 0.05:  # Significant improvement threshold
                significant_improvements += 1
        
        avg_improvement = total_improvements / len(comparison_results)
        f.write(f"Average R² Improvement: {avg_improvement:+.4f}\n")
        f.write(f"Models with Significant Improvement (>0.05): {significant_improvements}/{len(comparison_results)}\n")
        
        if avg_improvement > 0.02:
            f.write("Overall Assessment: ENSEMBLE MODELS SHOW CLEAR BENEFIT\n")
        elif avg_improvement > 0:
            f.write("Overall Assessment: ENSEMBLE MODELS SHOW MODEST BENEFIT\n")
        else:
            f.write("Overall Assessment: ENSEMBLE MODELS SHOW NO CLEAR BENEFIT\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("DETAILED MODEL COMPARISONS\n")
        f.write("=" * 100 + "\n\n")
        
        # Detailed comparison for each model
        for model_name, results in comparison_results.items():
            f.write(f"{model_name.upper()} MODEL COMPARISON\n")
            f.write("-" * 60 + "\n")
            
            ensemble_metrics = results['ensemble']['enhanced_metrics']
            individual_metrics = results['individual']['enhanced_metrics']
            
            # Key metrics comparison
            key_metrics = [
                ('R² Score', 'r2'),
                ('RMSE', 'rmse'),
                ('Spearman Correlation', 'spearman_corr'),
                ('Kendall Tau', 'kendall_tau'),
                ('Accuracy ±1.0', 'accuracy_10'),
                ('Concordance Index', 'concordance_index'),
                ('Top-3 Accuracy', 'top3_accuracy')
            ]
            
            f.write("Performance Metrics Comparison:\n")
            f.write(f"{'Metric':<20} {'Ensemble':<12} {'Individual':<12} {'Improvement':<12} {'Status':<15}\n")
            f.write("-" * 75 + "\n")
            
            for metric_name, metric_key in key_metrics:
                ensemble_val = ensemble_metrics.get(metric_key, 0)
                individual_val = individual_metrics.get(metric_key, 0)
                improvement = ensemble_val - individual_val
                
                # Determine status
                if metric_key == 'rmse':  # Lower is better for RMSE
                    improvement = -improvement  # Flip sign for RMSE
                    if improvement > 0.05:
                        status = "BETTER"
                    elif improvement > 0:
                        status = "SLIGHTLY BETTER"
                    elif improvement > -0.05:
                        status = "SIMILAR"
                    else:
                        status = "WORSE"
                else:  # Higher is better for other metrics
                    if improvement > 0.05:
                        status = "BETTER"
                    elif improvement > 0:
                        status = "SLIGHTLY BETTER"
                    elif improvement > -0.05:
                        status = "SIMILAR"
                    else:
                        status = "WORSE"
                
                f.write(f"{metric_name:<20} {ensemble_val:<12.4f} {individual_val:<12.4f} {improvement:<12.4f} {status:<15}\n")
            
            # Overall assessment for this model
            ensemble_r2 = ensemble_metrics.get('r2', 0)
            individual_r2 = individual_metrics.get('r2', 0)
            r2_improvement = ensemble_r2 - individual_r2
            
            f.write(f"\nOverall Assessment for {model_name}:\n")
            if r2_improvement > 0.1:
                f.write("  EXCELLENT: Ensemble shows significant improvement\n")
            elif r2_improvement > 0.05:
                f.write("  GOOD: Ensemble shows meaningful improvement\n")
            elif r2_improvement > 0:
                f.write("  FAIR: Ensemble shows modest improvement\n")
            elif r2_improvement > -0.05:
                f.write("  NEUTRAL: Ensemble performs similarly to individual model\n")
            else:
                f.write("  POOR: Individual model outperforms ensemble\n")
            
            f.write(f"  R² Improvement: {r2_improvement:+.4f}\n")
            
            # Feature importance comparison (if available)
            if hasattr(results['ensemble'], 'feature_importance_'):
                f.write("\nTop 5 Most Important Features (Ensemble):\n")
                ensemble_model = results['ensemble']
                if ensemble_model.feature_importance_:
                    sorted_features = sorted(ensemble_model.feature_importance_.items(), 
                                           key=lambda x: x[1], reverse=True)[:5]
                    for i, (feature, importance) in enumerate(sorted_features, 1):
                        f.write(f"  {i}. {feature}: {importance:.4f}\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("=" * 100 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 100 + "\n")
        
        # Model-specific recommendations
        f.write("Model-Specific Recommendations:\n")
        for model_name, results in comparison_results.items():
            ensemble_r2 = results['ensemble']['enhanced_metrics'].get('r2', 0)
            individual_r2 = results['individual']['enhanced_metrics'].get('r2', 0)
            improvement = ensemble_r2 - individual_r2
            
            if improvement > 0.05:
                f.write(f"  {model_name}: USE ENSEMBLE MODEL (significant improvement)\n")
            elif improvement > 0:
                f.write(f"  {model_name}: CONSIDER ENSEMBLE MODEL (modest improvement)\n")
            else:
                f.write(f"  {model_name}: USE INDIVIDUAL MODEL (better performance)\n")
        
        f.write("\nGeneral Recommendations:\n")
        f.write("1. Focus ensemble development on models showing clear improvement\n")
        f.write("2. Investigate why some ensembles underperform individual models\n")
        f.write("3. Consider hyperparameter tuning for underperforming ensembles\n")
        f.write("4. Evaluate computational cost vs performance benefit trade-offs\n")
        f.write("5. Test ensemble models on completely new data for validation\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF COMPARISON REPORT\n")
        f.write("=" * 100 + "\n")
    
    print(f"📋 Comprehensive comparison report saved to: {report_path}")
    return report_path

def main():
    """Main evaluation function."""
    print("🔍 Starting Comprehensive Ensemble Model Evaluation")
    print("="*70)
    
    # Load data
    input_csv = "../final_creator_scores_with_targets.csv"
    input_path = Path(input_csv)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    print(f"📊 Loading evaluation data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} creators for evaluation")
    
    # Load models
    print("\n📁 Loading ensemble models...")
    ensemble_models = load_ensemble_models()
    
    print("\n📁 Loading individual models...")
    individual_models = load_individual_models()
    
    # Find common models
    common_models = set(ensemble_models.keys()) & set(individual_models.keys())
    print(f"\n🎯 Found {len(common_models)} models for comparison: {', '.join(common_models)}")
    
    if not common_models:
        print("❌ No common models found for comparison")
        return
    
    # Perform comparison
    print("\n🔍 Performing model comparison...")
    comparison_results = compare_ensemble_vs_individual(ensemble_models, individual_models, df)
    
    if not comparison_results:
        print("❌ No comparison results generated")
        return
    
    # Create visualizations
    print("\n📊 Creating visualization plots...")
    create_comparison_plots(comparison_results)
    
    # Generate comprehensive report
    print("\n📋 Generating comprehensive comparison report...")
    report_path = generate_comprehensive_report(comparison_results)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"🎯 Evaluation Summary")
    print(f"{'='*70}")
    
    total_improvement = 0
    better_count = 0
    
    for model_name, results in comparison_results.items():
        ensemble_r2 = results['ensemble']['enhanced_metrics'].get('r2', 0)
        individual_r2 = results['individual']['enhanced_metrics'].get('r2', 0)
        improvement = ensemble_r2 - individual_r2
        total_improvement += improvement
        
        if improvement > 0:
            better_count += 1
        
        status = "✅ BETTER" if improvement > 0.05 else "🔄 SIMILAR" if improvement > -0.05 else "❌ WORSE"
        print(f"   {model_name}: {status} (R² improvement: {improvement:+.4f})")
    
    avg_improvement = total_improvement / len(comparison_results)
    print(f"\n📊 Overall Results:")
    print(f"   Average R² improvement: {avg_improvement:+.4f}")
    print(f"   Models improved by ensemble: {better_count}/{len(comparison_results)}")
    print(f"   Evaluation report: {report_path}")

if __name__ == "__main__":
    main()