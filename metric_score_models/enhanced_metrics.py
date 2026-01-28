"""
Enhanced evaluation metrics for personality scoring models.
Goes beyond R² and RMSE to provide more meaningful evaluation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_enhanced_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate comprehensive evaluation metrics for personality scoring.
    
    Args:
        y_true: Actual scores
        y_pred: Predicted scores
        model_name: Name of the model for reporting
    
    Returns:
        dict: Dictionary of all metrics
    """
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"error": "No valid data points"}
    
    metrics = {}
    
    # === TRADITIONAL METRICS ===
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # MAPE (handle division by zero)
    try:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['mape'] = mape if np.isfinite(mape) else 999.0
    except:
        metrics['mape'] = 999.0
    
    # === CORRELATION METRICS ===
    # Spearman (monotonic relationships)
    spearman_corr, spearman_p = stats.spearmanr(y_true, y_pred)
    metrics['spearman_corr'] = spearman_corr if np.isfinite(spearman_corr) else 0.0
    metrics['spearman_p_value'] = spearman_p if np.isfinite(spearman_p) else 1.0
    
    # Kendall's Tau (robust ranking)
    kendall_corr, kendall_p = stats.kendalltau(y_true, y_pred)
    metrics['kendall_tau'] = kendall_corr if np.isfinite(kendall_corr) else 0.0
    metrics['kendall_p_value'] = kendall_p if np.isfinite(kendall_p) else 1.0
    
    # Pearson (linear relationships)
    pearson_corr, pearson_p = stats.pearsonr(y_true, y_pred)
    metrics['pearson_corr'] = pearson_corr if np.isfinite(pearson_corr) else 0.0
    metrics['pearson_p_value'] = pearson_p if np.isfinite(pearson_p) else 1.0
    
    # === ACCURACY WITHIN TOLERANCE ===
    metrics['accuracy_05'] = np.mean(np.abs(y_true - y_pred) <= 0.5) * 100
    metrics['accuracy_10'] = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
    metrics['accuracy_15'] = np.mean(np.abs(y_true - y_pred) <= 1.5) * 100
    
    # === ROBUST METRICS ===
    # Median Absolute Error (less sensitive to outliers)
    metrics['median_ae'] = np.median(np.abs(y_true - y_pred))
    
    # Interquartile Range of errors
    errors = np.abs(y_true - y_pred)
    metrics['error_iqr'] = np.percentile(errors, 75) - np.percentile(errors, 25)
    
    # === DISTRIBUTION METRICS ===
    # Wasserstein Distance (Earth Mover's Distance)
    try:
        from scipy.stats import wasserstein_distance
        metrics['wasserstein_distance'] = wasserstein_distance(y_true, y_pred)
    except ImportError:
        # Fallback if wasserstein_distance not available
        metrics['wasserstein_distance'] = 999.0
    except Exception:
        metrics['wasserstein_distance'] = 999.0
    
    # Kolmogorov-Smirnov test
    try:
        ks_stat, ks_p = stats.ks_2samp(y_true, y_pred)
        metrics['ks_statistic'] = ks_stat
        metrics['ks_p_value'] = ks_p
    except:
        metrics['ks_statistic'] = 1.0
        metrics['ks_p_value'] = 0.0
    
    # === RANKING METRICS ===
    # Top-K accuracy (how often top performers are correctly identified)
    def top_k_accuracy(y_true, y_pred, k=3):
        if len(y_true) < k:
            k = len(y_true)
        
        # Get indices of top-k actual and predicted
        top_k_true = set(np.argsort(y_true)[-k:])
        top_k_pred = set(np.argsort(y_pred)[-k:])
        
        # Calculate overlap
        overlap = len(top_k_true.intersection(top_k_pred))
        return overlap / k * 100
    
    metrics['top3_accuracy'] = top_k_accuracy(y_true, y_pred, k=3)
    metrics['top5_accuracy'] = top_k_accuracy(y_true, y_pred, k=5)
    
    # Concordance Index (C-Index) - measures ranking quality
    def concordance_index(y_true, y_pred):
        n = len(y_true)
        concordant = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i+1, n):
                if y_true[i] != y_true[j]:  # Only consider pairs with different true values
                    total_pairs += 1
                    # Check if prediction order matches true order
                    if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                       (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                        concordant += 1
        
        return concordant / total_pairs if total_pairs > 0 else 0.5
    
    metrics['concordance_index'] = concordance_index(y_true, y_pred)
    
    # === BUSINESS METRICS ===
    # Mean error (bias detection)
    metrics['mean_error'] = np.mean(y_pred - y_true)
    
    # Standard deviation of errors
    metrics['error_std'] = np.std(y_pred - y_true)
    
    # Percentage of predictions within business-acceptable range
    metrics['business_acceptable'] = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
    
    return metrics

def print_enhanced_metrics(metrics, model_name="Model"):
    """Print metrics in a formatted way."""
    
    if 'error' in metrics:
        print(f"❌ {model_name}: {metrics['error']}")
        return
    
    print(f"\n📊 Enhanced Metrics for {model_name}")
    print("=" * 60)
    
    # Traditional Metrics
    print("🔢 Traditional Metrics:")
    print(f"   MAE:  {metrics['mae']:.3f}")
    print(f"   RMSE: {metrics['rmse']:.3f}")
    print(f"   R²:   {metrics['r2']:.3f}")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    
    # Correlation Metrics
    print("\n🔗 Correlation Metrics:")
    print(f"   Spearman ρ: {metrics['spearman_corr']:.3f} (p={metrics['spearman_p_value']:.3f})")
    print(f"   Kendall τ:  {metrics['kendall_tau']:.3f} (p={metrics['kendall_p_value']:.3f})")
    print(f"   Pearson r:  {metrics['pearson_corr']:.3f} (p={metrics['pearson_p_value']:.3f})")
    
    # Accuracy Metrics
    print("\n🎯 Accuracy Within Tolerance:")
    print(f"   ±0.5: {metrics['accuracy_05']:.1f}%")
    print(f"   ±1.0: {metrics['accuracy_10']:.1f}%")
    print(f"   ±1.5: {metrics['accuracy_15']:.1f}%")
    
    # Ranking Metrics
    print("\n🏆 Ranking Quality:")
    print(f"   Top-3 Accuracy: {metrics['top3_accuracy']:.1f}%")
    print(f"   Top-5 Accuracy: {metrics['top5_accuracy']:.1f}%")
    print(f"   Concordance Index: {metrics['concordance_index']:.3f}")
    
    # Distribution Metrics
    print("\n📈 Distribution Similarity:")
    print(f"   Wasserstein Distance: {metrics['wasserstein_distance']:.3f}")
    print(f"   KS Statistic: {metrics['ks_statistic']:.3f} (p={metrics['ks_p_value']:.3f})")
    
    # Robust Metrics
    print("\n🛡️ Robust Metrics:")
    print(f"   Median AE: {metrics['median_ae']:.3f}")
    print(f"   Error IQR: {metrics['error_iqr']:.3f}")
    print(f"   Mean Bias: {metrics['mean_error']:.3f}")
    
    # Business Metrics
    print("\n💼 Business Relevance:")
    print(f"   Business Acceptable (±1.0): {metrics['business_acceptable']:.1f}%")

def get_best_model_by_metric(results_dict, metric='spearman_corr'):
    """
    Find the best model based on a specific metric.
    
    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
        metric: Metric to use for comparison
    
    Returns:
        tuple: (best_model_name, best_score)
    """
    
    valid_models = {name: metrics for name, metrics in results_dict.items() 
                   if 'error' not in metrics and metric in metrics}
    
    if not valid_models:
        return None, None
    
    # For most metrics, higher is better
    higher_is_better = [
        'r2', 'spearman_corr', 'kendall_tau', 'pearson_corr',
        'accuracy_05', 'accuracy_10', 'accuracy_15',
        'top3_accuracy', 'top5_accuracy', 'concordance_index',
        'business_acceptable'
    ]
    
    if metric in higher_is_better:
        best_model = max(valid_models.keys(), key=lambda x: valid_models[x][metric])
    else:
        # Lower is better for error metrics
        best_model = min(valid_models.keys(), key=lambda x: valid_models[x][metric])
    
    return best_model, valid_models[best_model][metric]

# Example usage
if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    y_true = np.random.uniform(1, 10, 50)
    y_pred = y_true + np.random.normal(0, 0.5, 50)  # Add some noise
    
    metrics = calculate_enhanced_metrics(y_true, y_pred, "Test Model")
    print_enhanced_metrics(metrics, "Test Model")