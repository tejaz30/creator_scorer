"""
Training Script for All Ensemble Models
Trains XGBoost + Random Forest ensemble models for all 6 personality traits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
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

# Import enhanced metrics
from enhanced_metrics import calculate_enhanced_metrics, print_enhanced_metrics

def create_ensemble_directories():
    """Create organized directory structure for ensemble models."""
    base_dir = Path("trained_models_organized")
    
    directories = [
        base_dir,
        base_dir / "ensemble_models",
        base_dir / "ensemble_evaluation_reports"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        
    return base_dir

def save_ensemble_evaluation_report(results, base_dir):
    """Save comprehensive ensemble evaluation report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = base_dir / "ensemble_evaluation_reports" / f"ensemble_training_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ENSEMBLE PERSONALITY MODELS TRAINING EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Architecture: XGBoost (60%) + Random Forest (40%) with Meta-Learning\n")
        f.write(f"Training Data Split: 80% Training / 20% Validation\n")
        f.write(f"Total Ensemble Models Trained: {len(results)}\n\n")
        
        # Enhanced Summary table
        f.write("ENSEMBLE PERFORMANCE SUMMARY TABLE\n")
        f.write("-" * 150 + "\n")
        f.write(f"{'Model':<15} {'Val R²':<8} {'Val RMSE':<10} {'Spearman':<10} {'Kendall':<8} {'Acc±1.0':<8} {'C-Index':<8} {'XGB R²':<8} {'RF R²':<8} {'Ridge R²':<9} {'KNN R²':<8} {'Disagreement':<12}\n")
        f.write("-" * 170 + "\n")
        
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                val_r2 = metrics.get('val_r2', 0)
                val_rmse = metrics.get('val_rmse', 0)
                
                # Enhanced metrics
                spearman = metrics.get('enhanced_metrics', {}).get('spearman_corr', 0)
                kendall = metrics.get('enhanced_metrics', {}).get('kendall_tau', 0)
                acc_10 = metrics.get('enhanced_metrics', {}).get('accuracy_10', 0)
                c_index = metrics.get('enhanced_metrics', {}).get('concordance_index', 0)
                
                # Individual model performance
                xgb_r2 = metrics.get('xgb_val_r2', 0)
                rf_r2 = metrics.get('rf_val_r2', 0)
                ridge_r2 = metrics.get('ridge_val_r2', 0)
                knn_r2 = metrics.get('knn_val_r2', 0)
                disagreement = metrics.get('model_disagreement', 0)
                
                f.write(f"{model_name:<15} {val_r2:<8.3f} {val_rmse:<10.3f} {spearman:<10.3f} {kendall:<8.3f} {acc_10:<8.1f}% {c_index:<8.3f} {xgb_r2:<8.3f} {rf_r2:<8.3f} {ridge_r2:<9.3f} {knn_r2:<8.3f} {disagreement:<12.3f}\n")
            else:
                f.write(f"{model_name:<15} {'FAILED':<8} {'FAILED':<10} {'FAILED':<10} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<9} {'FAILED':<8} {'FAILED':<12}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED ENSEMBLE MODEL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Detailed results for each model
        for model_name, metrics in results.items():
            f.write(f"{model_name.upper()} ENSEMBLE MODEL\n")
            f.write("-" * 50 + "\n")
            
            if 'error' in metrics:
                f.write(f"TRAINING FAILED: {metrics['error']}\n\n")
                continue
            
            # Ensemble Performance
            f.write("Ensemble Performance:\n")
            f.write(f"  Training R²:      {metrics.get('train_r2', 0):.4f}\n")
            f.write(f"  Validation R²:    {metrics.get('val_r2', 0):.4f}\n")
            f.write(f"  Training RMSE:    {metrics.get('train_rmse', 0):.4f}\n")
            f.write(f"  Validation RMSE:  {metrics.get('val_rmse', 0):.4f}\n")
            f.write(f"  Validation MAE:   {metrics.get('val_mae', 0):.4f}\n")
            
            # Individual Model Comparison
            f.write("\nIndividual Model Comparison:\n")
            f.write(f"  XGBoost Val R²:       {metrics.get('xgb_val_r2', 0):.4f}\n")
            f.write(f"  Random Forest Val R²: {metrics.get('rf_val_r2', 0):.4f}\n")
            f.write(f"  Ridge Regression Val R²: {metrics.get('ridge_val_r2', 0):.4f}\n")
            f.write(f"  KNN Val R²:           {metrics.get('knn_val_r2', 0):.4f}\n")
            f.write(f"  Model Disagreement:   {metrics.get('model_disagreement', 0):.4f}\n")
            
            # Ensemble Weights
            weights = metrics.get('ensemble_weights', {})
            f.write(f"\nEnsemble Weights:\n")
            f.write(f"  XGBoost Weight:       {weights.get('xgboost', 0):.3f}\n")
            f.write(f"  Random Forest Weight: {weights.get('random_forest', 0):.3f}\n")
            f.write(f"  Ridge Weight:         {weights.get('ridge', 0):.3f}\n")
            f.write(f"  KNN Weight:           {weights.get('knn', 0):.3f}\n")
            
            # Enhanced metrics section
            if 'enhanced_metrics' in metrics:
                enhanced = metrics['enhanced_metrics']
                f.write("\nEnhanced Evaluation Metrics:\n")
                
                # Correlation metrics
                f.write("  Correlation Analysis:\n")
                f.write(f"    Spearman ρ:       {enhanced.get('spearman_corr', 0):.4f} (p={enhanced.get('spearman_p_value', 1):.3f})\n")
                f.write(f"    Kendall τ:        {enhanced.get('kendall_tau', 0):.4f} (p={enhanced.get('kendall_p_value', 1):.3f})\n")
                f.write(f"    Pearson r:        {enhanced.get('pearson_corr', 0):.4f} (p={enhanced.get('pearson_p_value', 1):.3f})\n")
                
                # Accuracy within tolerance
                f.write("  Accuracy Within Tolerance:\n")
                f.write(f"    Within ±0.5:      {enhanced.get('accuracy_05', 0):.1f}%\n")
                f.write(f"    Within ±1.0:      {enhanced.get('accuracy_10', 0):.1f}%\n")
                f.write(f"    Within ±1.5:      {enhanced.get('accuracy_15', 0):.1f}%\n")
                
                # Ranking quality
                f.write("  Ranking Quality:\n")
                f.write(f"    Top-3 Accuracy:   {enhanced.get('top3_accuracy', 0):.1f}%\n")
                f.write(f"    Concordance Index: {enhanced.get('concordance_index', 0):.4f}\n")
                
                # Business metrics
                f.write("  Business Metrics:\n")
                f.write(f"    Business Acceptable: {enhanced.get('business_acceptable', 0):.1f}%\n")
                f.write(f"    Mean Bias:        {enhanced.get('mean_error', 0):.4f}\n")
            
            # Ensemble Quality Assessment
            val_r2 = metrics.get('val_r2', 0)
            spearman = metrics.get('enhanced_metrics', {}).get('spearman_corr', 0)
            xgb_r2 = metrics.get('xgb_val_r2', 0)
            rf_r2 = metrics.get('rf_val_r2', 0)
            
            f.write("\nEnsemble Quality Assessment:\n")
            
            # Overall performance
            if val_r2 > max(xgb_r2, rf_r2):
                f.write("  Ensemble Benefit: POSITIVE (Outperforms individual models)\n")
            elif val_r2 > min(xgb_r2, rf_r2):
                f.write("  Ensemble Benefit: MODERATE (Between individual models)\n")
            else:
                f.write("  Ensemble Benefit: NEGATIVE (Underperforms individual models)\n")
            
            # Performance level
            if val_r2 > 0.5 and spearman > 0.7:
                f.write("  Overall Quality: EXCELLENT (Strong predictive power)\n")
            elif val_r2 > 0.3 and spearman > 0.6:
                f.write("  Overall Quality: GOOD (Moderate predictive power)\n")
            elif val_r2 > 0.1 or spearman > 0.4:
                f.write("  Overall Quality: FAIR (Some predictive power)\n")
            else:
                f.write("  Overall Quality: POOR (Limited predictive power)\n")
            
            # Overfitting check
            r2_diff = metrics.get('train_r2', 0) - val_r2
            if r2_diff > 0.3:
                f.write("  WARNING: Significant overfitting detected\n")
            elif r2_diff > 0.1:
                f.write("  CAUTION: Some overfitting present\n")
            else:
                f.write("  OK: Good generalization\n")
            
            # Top 5 Feature Importance Analysis
            f.write("\nTop 5 Most Important Features:\n")
            try:
                if 'feature_importance' in metrics and metrics['feature_importance']:
                    # Get ensemble feature importance
                    ensemble_importance = metrics['feature_importance']
                    sorted_features = sorted(ensemble_importance.items(), 
                                           key=lambda x: x[1], reverse=True)[:5]
                    
                    f.write("  Ensemble (Combined) Importance:\n")
                    for i, (feature, importance) in enumerate(sorted_features, 1):
                        f.write(f"    {i}. {feature}: {importance:.4f}\n")
                    
                    # Show individual model comparisons if available
                    if ('xgb_feature_importance' in metrics and 
                        'rf_feature_importance' in metrics and
                        'ridge_feature_importance' in metrics and
                        'knn_feature_importance' in metrics):
                        
                        f.write("\n  Individual Model Comparison (Top 5 Features):\n")
                        f.write(f"    {'Feature':<25} {'Ensemble':<10} {'XGBoost':<10} {'RF':<10} {'Ridge':<10} {'KNN':<10}\n")
                        f.write(f"    {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}\n")
                        
                        xgb_importance = metrics['xgb_feature_importance']
                        rf_importance = metrics['rf_feature_importance']
                        ridge_importance = metrics['ridge_feature_importance']
                        knn_importance = metrics['knn_feature_importance']
                        
                        for i, (feature, ens_imp) in enumerate(sorted_features, 1):
                            xgb_imp = xgb_importance.get(feature, 0.0)
                            rf_imp = rf_importance.get(feature, 0.0)
                            ridge_imp = ridge_importance.get(feature, 0.0)
                            knn_imp = knn_importance.get(feature, 0.0)
                            f.write(f"    {feature:<25} {ens_imp:<10.4f} {xgb_imp:<10.4f} {rf_imp:<10.4f} {ridge_imp:<10.4f} {knn_imp:<10.4f}\n")
                        
                        # Show which model contributes most to each top feature
                        f.write("\n  Feature Contribution Analysis:\n")
                        for i, (feature, ens_imp) in enumerate(sorted_features, 1):
                            xgb_imp = xgb_importance.get(feature, 0.0)
                            rf_imp = rf_importance.get(feature, 0.0)
                            ridge_imp = ridge_importance.get(feature, 0.0)
                            knn_imp = knn_importance.get(feature, 0.0)
                            
                            # Find dominant model
                            model_imps = {'XGBoost': xgb_imp, 'Random Forest': rf_imp, 
                                        'Ridge': ridge_imp, 'KNN': knn_imp}
                            dominant_model = max(model_imps, key=model_imps.get)
                            dominant_value = model_imps[dominant_model]
                            
                            # Find second highest
                            remaining_models = {k: v for k, v in model_imps.items() if k != dominant_model}
                            if remaining_models:
                                second_value = max(remaining_models.values())
                                if second_value > 0:
                                    ratio = dominant_value / second_value
                                    f.write(f"    {i}. {feature}: {dominant_model} leads by {ratio:.2f}x\n")
                                else:
                                    f.write(f"    {i}. {feature}: Dominated by {dominant_model}\n")
                            else:
                                f.write(f"    {i}. {feature}: Dominated by {dominant_model}\n")
                else:
                    f.write("  Feature importance data not available\n")
            except Exception as e:
                f.write(f"  Error retrieving feature importance: {str(e)}\n")
            
            f.write("\n")
        
        # ENSEMBLE ARCHITECTURE EXPLANATION
        f.write("=" * 80 + "\n")
        f.write("ENSEMBLE ARCHITECTURE EXPLANATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. ENSEMBLE WEIGHT DETERMINATION:\n")
        f.write("-" * 40 + "\n")
        f.write("The ensemble combines 4 diverse models using two methods:\n\n")
        
        f.write("A) Default Fixed Weights (when meta-learning is disabled):\n")
        f.write("   - XGBoost: 35% weight (0.35) - Gradient boosting performance\n")
        f.write("   - Random Forest: 25% weight (0.25) - Bagging stability\n")
        f.write("   - Ridge Regression: 25% weight (0.25) - Linear relationships\n")
        f.write("   - KNN: 15% weight (0.15) - Local pattern recognition\n")
        f.write("   - Rationale: Balanced combination of different learning paradigms\n")
        f.write("     * Tree-based models (60%): XGBoost + Random Forest for non-linear patterns\n")
        f.write("     * Linear model (25%): Ridge for global linear relationships\n")
        f.write("     * Instance-based (15%): KNN for local similarity patterns\n")
        f.write("   - Fixed weights ensure consistent behavior across different datasets\n\n")
        
        f.write("B) Meta-Learning Adaptive Weights (when enabled - current setting):\n")
        f.write("   - Uses 5-fold cross-validation to train a linear meta-learner\n")
        f.write("   - Meta-learner learns optimal weights based on all 4 model predictions\n")
        f.write("   - Weights are automatically adjusted for each personality trait\n")
        f.write("   - Process: Train base models → Get CV predictions → Train meta-learner\n")
        f.write("   - Meta-learner coefficients determine final ensemble weights\n")
        f.write("   - Final weights shown in 'Ensemble Weights' section above\n")
        f.write("   - Advantage: Adapts to each trait's specific prediction patterns\n")
        f.write("   - Captures complementary strengths of different model types\n\n")
        
        f.write("2. MODEL ARCHITECTURE DETAILS:\n")
        f.write("-" * 40 + "\n")
        
        # Import config to get actual parameters
        try:
            from config import XGBOOST_PARAMS, RANDOM_FOREST_PARAMS, RIDGE_PARAMS, KNN_PARAMS
            
            f.write("A) XGBoost Configuration (Gradient Boosting):\n")
            f.write(f"   - Number of Trees (n_estimators): {XGBOOST_PARAMS.get('n_estimators', 'N/A')}\n")
            f.write(f"   - Maximum Tree Depth: {XGBOOST_PARAMS.get('max_depth', 'N/A')} levels\n")
            f.write(f"   - Learning Rate (eta): {XGBOOST_PARAMS.get('learning_rate', 'N/A')}\n")
            f.write(f"   - Subsample Ratio: {XGBOOST_PARAMS.get('subsample', 'N/A')} (fraction of data per tree)\n")
            f.write(f"   - Feature Sampling (colsample_bytree): {XGBOOST_PARAMS.get('colsample_bytree', 'N/A')}\n")
            f.write(f"   - Min Child Weight: {XGBOOST_PARAMS.get('min_child_weight', 'N/A')}\n")
            f.write(f"   - Regularization Alpha (L1): {XGBOOST_PARAMS.get('reg_alpha', 'N/A')}\n")
            f.write(f"   - Regularization Lambda (L2): {XGBOOST_PARAMS.get('reg_lambda', 'N/A')}\n")
            f.write("   - Architecture: Sequential gradient boosting\n")
            f.write("     * Each tree corrects errors from previous trees\n")
            f.write("     * Trees are built in sequence, not parallel\n")
            f.write("     * Final prediction = sum of all tree predictions\n")
            f.write("     * Excellent for capturing complex patterns and interactions\n\n")
            
            f.write("B) Random Forest Configuration (Bootstrap Aggregating):\n")
            f.write(f"   - Number of Trees (n_estimators): {RANDOM_FOREST_PARAMS.get('n_estimators', 'N/A')}\n")
            f.write(f"   - Maximum Tree Depth: {RANDOM_FOREST_PARAMS.get('max_depth', 'N/A')} levels\n")
            f.write(f"   - Min Samples to Split: {RANDOM_FOREST_PARAMS.get('min_samples_split', 'N/A')}\n")
            f.write(f"   - Min Samples per Leaf: {RANDOM_FOREST_PARAMS.get('min_samples_leaf', 'N/A')}\n")
            f.write(f"   - Random State: {RANDOM_FOREST_PARAMS.get('random_state', 'N/A')}\n")
            f.write(f"   - Parallel Jobs: {RANDOM_FOREST_PARAMS.get('n_jobs', 'N/A')} (-1 = all cores)\n")
            f.write("   - Architecture: Parallel independent trees with bagging\n")
            f.write("     * Each tree trained on random bootstrap sample of data\n")
            f.write("     * Each split considers random subset of features\n")
            f.write("     * Trees are built independently in parallel\n")
            f.write("     * Final prediction = average of all tree predictions\n")
            f.write("     * Excellent for reducing overfitting and providing stability\n\n")
            
            f.write("C) Ridge Regression Configuration (Regularized Linear):\n")
            f.write(f"   - Regularization Alpha: {RIDGE_PARAMS.get('alpha', 'N/A')}\n")
            f.write(f"   - Fit Intercept: {RIDGE_PARAMS.get('fit_intercept', 'N/A')}\n")
            f.write(f"   - Solver: {RIDGE_PARAMS.get('solver', 'N/A')}\n")
            f.write(f"   - Random State: {RIDGE_PARAMS.get('random_state', 'N/A')}\n")
            f.write("   - Architecture: Linear regression with L2 regularization\n")
            f.write("     * Minimizes: ||y - Xw||² + α||w||²\n")
            f.write("     * Shrinks coefficients toward zero to prevent overfitting\n")
            f.write("     * Handles multicollinearity well\n")
            f.write("     * Provides global linear relationships\n")
            f.write("     * Fast training and prediction\n")
            f.write("     * Requires feature scaling (automatically applied)\n\n")
            
            f.write("D) K-Nearest Neighbors Configuration (Instance-Based):\n")
            f.write(f"   - Number of Neighbors (k): {KNN_PARAMS.get('n_neighbors', 'N/A')}\n")
            f.write(f"   - Weighting: {KNN_PARAMS.get('weights', 'N/A')}\n")
            f.write(f"   - Algorithm: {KNN_PARAMS.get('algorithm', 'N/A')}\n")
            f.write(f"   - Distance Metric: {KNN_PARAMS.get('metric', 'N/A')}\n")
            f.write(f"   - Minkowski Parameter (p): {KNN_PARAMS.get('p', 'N/A')} (2 = Euclidean)\n")
            f.write(f"   - Leaf Size: {KNN_PARAMS.get('leaf_size', 'N/A')}\n")
            f.write("   - Architecture: Lazy learning with local similarity\n")
            f.write("     * No explicit training phase (stores all data)\n")
            f.write("     * Prediction based on k nearest neighbors\n")
            f.write("     * Distance-weighted averaging of neighbor targets\n")
            f.write("     * Captures local patterns and non-linear relationships\n")
            f.write("     * Adapts to local data density\n")
            f.write("     * Requires feature scaling (automatically applied)\n\n")
            
            # Calculate total model complexity
            total_xgb_trees = XGBOOST_PARAMS.get('n_estimators', 0)
            total_rf_trees = RANDOM_FOREST_PARAMS.get('n_estimators', 0)
            total_trees = total_xgb_trees + total_rf_trees
            
            f.write("E) Combined Ensemble Complexity:\n")
            f.write(f"   - Total Trees in Ensemble: {total_trees} ({total_xgb_trees} XGB + {total_rf_trees} RF)\n")
            f.write(f"   - Tree-based Models: {total_trees} trees total\n")
            f.write(f"   - Linear Models: 1 Ridge regression\n")
            f.write(f"   - Instance-based Models: 1 KNN (stores all training data)\n")
            f.write("   - Model Diversity: 4 different learning paradigms\n")
            f.write("   - Prediction Complexity: O(trees × depth) + O(features) + O(k × samples)\n")
            f.write("   - Memory Usage: ~4x individual model (stores all 4 models)\n")
            f.write("   - Training Time: ~4x individual model (trains all 4 models)\n\n")
            
        except ImportError:
            f.write("   Configuration details not available (config import failed)\n\n")
        
        f.write("3. ENSEMBLE PREDICTION PROCESS:\n")
        f.write("-" * 40 + "\n")
        f.write("Step 1: Feature Preprocessing\n")
        f.write("        - Handle missing values (median/mode imputation)\n")
        f.write("        - Ensure all required features are present\n")
        f.write("        - Scale features for Ridge and KNN (StandardScaler)\n")
        f.write("        - Keep original features for XGBoost and Random Forest\n\n")
        
        f.write("Step 2: Individual Model Predictions\n")
        f.write("        - XGBoost: Sequential tree predictions summed (raw features)\n")
        f.write("        - Random Forest: Parallel tree predictions averaged (raw features)\n")
        f.write("        - Ridge: Linear combination of scaled features\n")
        f.write("        - KNN: Distance-weighted average of k neighbors (scaled features)\n")
        f.write("        - All models predict independently\n\n")
        
        f.write("Step 3: Ensemble Combination\n")
        f.write("        If meta-learning enabled (current setting):\n")
        f.write("        - Stack all 4 predictions as meta-features\n")
        f.write("        - Apply learned scaling transformation\n")
        f.write("        - Meta-learner combines using learned weights\n")
        f.write("        If meta-learning disabled:\n")
        f.write("        - Weighted average: 0.35×XGB + 0.25×RF + 0.25×Ridge + 0.15×KNN\n\n")
        
        f.write("Step 4: Post-processing\n")
        f.write("        - Clip predictions to valid score range (1-10)\n")
        f.write("        - Calculate uncertainty from model disagreement (std dev)\n")
        f.write("        - Return final prediction with confidence bounds\n\n")
        
        f.write("Mathematical Formulas:\n")
        f.write("        Meta-learning: y_ensemble = meta_learner([y_xgb, y_rf, y_ridge, y_knn])\n")
        f.write("        Fixed weights: y_ensemble = 0.35×y_xgb + 0.25×y_rf + 0.25×y_ridge + 0.15×y_knn\n")
        f.write("        Uncertainty: σ = std([y_xgb, y_rf, y_ridge, y_knn]) (model disagreement)\n")
        f.write("        Ridge: y_ridge = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ (linear combination)\n")
        f.write("        KNN: y_knn = Σ(wᵢ × yᵢ) where wᵢ = 1/distance(x, xᵢ)\n\n")
        
        f.write("4. ENSEMBLE BENEFITS & RATIONALE:\n")
        f.write("-" * 40 + "\n")
        f.write("A) Performance Benefits:\n")
        f.write("   - Reduced Overfitting: Multiple diverse models regularize each other\n")
        f.write("   - Improved Robustness: 4 different algorithms reduce single-model bias\n")
        f.write("   - Better Generalization: Combines different learning paradigms\n")
        f.write("   - Error Compensation: Models make different types of errors\n")
        f.write("   - Complementary Strengths: Tree-based + Linear + Instance-based learning\n\n")
        
        f.write("B) Model Diversity Benefits:\n")
        f.write("   - XGBoost: Captures complex non-linear interactions\n")
        f.write("   - Random Forest: Provides stability and handles outliers\n")
        f.write("   - Ridge: Captures global linear trends and relationships\n")
        f.write("   - KNN: Adapts to local patterns and data density\n")
        f.write("   - Different Assumptions: Each model has different inductive biases\n\n")
        
        f.write("C) Interpretability Benefits:\n")
        f.write("   - Feature Importance: Weighted combination of tree-based importances\n")
        f.write("   - Model Comparison: Can analyze which model contributes more per feature\n")
        f.write("   - Uncertainty Quantification: Model disagreement indicates confidence\n")
        f.write("   - Prediction Explanation: Can trace back to individual model contributions\n")
        f.write("   - Linear Component: Ridge provides interpretable linear relationships\n\n")
        
        f.write("D) Practical Benefits:\n")
        f.write("   - Stability: Less sensitive to hyperparameter choices\n")
        f.write("   - Reliability: Multiple backup models if one fails or performs poorly\n")
        f.write("   - Flexibility: Can adjust weights based on performance\n")
        f.write("   - Scalability: Tree models can be trained in parallel\n")
        f.write("   - Versatility: Handles both linear and non-linear relationships\n\n")
        
        f.write("E) Why This 4-Model Combination?\n")
        f.write("   - Complementary Strengths: Each model excels in different scenarios\n")
        f.write("     * XGBoost: Complex patterns, feature interactions\n")
        f.write("     * Random Forest: Stability, outlier resistance\n")
        f.write("     * Ridge: Linear relationships, regularization\n")
        f.write("     * KNN: Local patterns, adaptive boundaries\n")
        f.write("   - Different Biases: Sequential vs parallel vs linear vs instance-based\n")
        f.write("   - Proven Approaches: All are well-established, reliable algorithms\n")
        f.write("   - Computational Efficiency: Balance between diversity and speed\n")
        f.write("   - Risk Mitigation: If one model type fails, others compensate\n\n")
        
        # Ensemble Training Recommendations
        f.write("=" * 80 + "\n")
        f.write("ENSEMBLE TRAINING RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n")
        
        successful_models = [name for name, metrics in results.items() if 'error' not in metrics]
        failed_models = [name for name, metrics in results.items() if 'error' in metrics]
        
        if successful_models:
            f.write(f"Successfully trained ensemble models: {', '.join(successful_models)}\n")
            
            # Find best performing ensemble
            best_ensemble = max(successful_models, 
                               key=lambda x: results[x].get('val_r2', 0))
            best_score = results[best_ensemble].get('val_r2', 0)
            f.write(f"Best ensemble model: {best_ensemble} (Val R² = {best_score:.3f})\n")
            
            # Ensemble effectiveness analysis
            f.write("\nEnsemble Effectiveness Analysis:\n")
            for model in successful_models:
                metrics = results[model]
                ensemble_r2 = metrics.get('val_r2', 0)
                xgb_r2 = metrics.get('xgb_val_r2', 0)
                rf_r2 = metrics.get('rf_val_r2', 0)
                
                improvement = ensemble_r2 - max(xgb_r2, rf_r2)
                f.write(f"  {model}: Ensemble R² = {ensemble_r2:.3f}, ")
                f.write(f"Best Individual = {max(xgb_r2, rf_r2):.3f}, ")
                f.write(f"Improvement = {improvement:+.3f}\n")
        
        if failed_models:
            f.write(f"\nFailed ensemble models: {', '.join(failed_models)}\n")
        
        f.write("\nNext Steps:\n")
        f.write("1. Compare ensemble vs individual model performance\n")
        f.write("2. Analyze feature importance from both XGBoost and Random Forest\n")
        f.write("3. Consider hyperparameter tuning for underperforming ensembles\n")
        f.write("4. Evaluate ensemble predictions on new data\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF ENSEMBLE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"📊 Ensemble evaluation report saved to: {report_path}")
    return report_path

def train_all_ensemble_models(input_csv="../final_creator_scores_with_targets.csv"):
    """Train all ensemble models with comprehensive evaluation."""
    
    print("🚀 Starting Ensemble Model Training Pipeline")
    print("="*70)
    print("Architecture: XGBoost (60%) + Random Forest (40%) with Meta-Learning")
    
    # Create directory structure
    base_dir = create_ensemble_directories()
    print(f"📁 Created ensemble directory structure in: {base_dir}")
    
    # Load data
    input_path = Path(input_csv)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    print(f"📊 Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} creators with {len(df.columns)} features")
    
    # Ensemble models configuration
    ensemble_config = [
        {
            'name': 'Aspirational',
            'model': aspirational_ensemble,
            'target': 'aspirational',
            'model_file': 'aspirational_ensemble.joblib'
        },
        {
            'name': 'Cool', 
            'model': cool_ensemble,
            'target': 'cool',
            'model_file': 'cool_ensemble.joblib'
        },
        {
            'name': 'Relatable',
            'model': relatable_ensemble, 
            'target': 'relatable',
            'model_file': 'relatable_ensemble.joblib'
        },
        {
            'name': 'Credibility',
            'model': credibility_ensemble,
            'target': 'credible',
            'model_file': 'credibility_ensemble.joblib'
        },
        {
            'name': 'Communication',
            'model': communication_ensemble,
            'target': 'communication', 
            'model_file': 'communication_ensemble.joblib'
        },
        {
            'name': 'Storytelling',
            'model': storytelling_ensemble,
            'target': 'story_telling',
            'model_file': 'storytelling_ensemble.joblib'
        }
    ]
    
    # Train all ensemble models
    training_results = {}
    
    for config in ensemble_config:
        model_name = config['name']
        model = config['model']
        target_col = config['target']
        model_file = config['model_file']
        
        print(f"\n{'='*60}")
        print(f"🎯 Training {model_name} Ensemble Model")
        print(f"{'='*60}")
        
        # Check if target column exists
        if target_col not in df.columns:
            print(f"❌ Target column '{target_col}' not found, skipping {model_name}")
            training_results[model_name] = {'error': f"Target column '{target_col}' not found"}
            continue
        
        # Check for missing values in target
        missing_targets = df[target_col].isna().sum()
        if missing_targets > 0:
            print(f"⚠️ Found {missing_targets} missing values in target, filling with median")
            df[target_col] = df[target_col].fillna(df[target_col].median())
        
        try:
            # Train the ensemble model
            print(f"🔧 Training {model_name} ensemble with 80/20 split...")
            
            # Train ensemble
            results = model.fit(df, target_col, test_size=0.2, random_state=42)
            
            # Calculate enhanced metrics
            if hasattr(model, 'y_test') and hasattr(model, 'y_pred'):
                enhanced_metrics = calculate_enhanced_metrics(
                    model.y_test, model.y_pred, f"{model_name} Ensemble"
                )
                results['enhanced_metrics'] = enhanced_metrics
                
                # Print enhanced metrics
                print_enhanced_metrics(enhanced_metrics, f"{model_name} Ensemble")
            
            # Save ensemble model
            model_path = base_dir / "ensemble_models" / model_file
            model.save(model_path)
            
            # Capture feature importance for reporting
            try:
                if hasattr(model, 'feature_importance_') and model.feature_importance_:
                    results['feature_importance'] = model.feature_importance_.copy()
                    print(f"   📊 Captured ensemble feature importance: {len(results['feature_importance'])} features")
                    
                    # Also capture individual model importances for comparison
                    if hasattr(model, 'xgb_feature_importance_'):
                        results['xgb_feature_importance'] = model.xgb_feature_importance_.copy()
                        print(f"   📊 Captured XGBoost feature importance: {len(results['xgb_feature_importance'])} features")
                    if hasattr(model, 'rf_feature_importance_'):
                        results['rf_feature_importance'] = model.rf_feature_importance_.copy()
                        print(f"   📊 Captured Random Forest feature importance: {len(results['rf_feature_importance'])} features")
                    if hasattr(model, 'ridge_feature_importance_'):
                        results['ridge_feature_importance'] = model.ridge_feature_importance_.copy()
                        print(f"   📊 Captured Ridge feature importance: {len(results['ridge_feature_importance'])} features")
                    if hasattr(model, 'knn_feature_importance_'):
                        results['knn_feature_importance'] = model.knn_feature_importance_.copy()
                        print(f"   📊 Captured KNN feature importance: {len(results['knn_feature_importance'])} features")
                else:
                    print(f"   ⚠️ No feature importance found for {model_name}")
                    print(f"      has feature_importance_: {hasattr(model, 'feature_importance_')}")
                    if hasattr(model, 'feature_importance_'):
                        print(f"      feature_importance_ content: {model.feature_importance_}")
                    results['feature_importance'] = {}
            except Exception as e:
                print(f"⚠️ Could not capture feature importance for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                results['feature_importance'] = {}
            
            # Store results
            training_results[model_name] = results
            
        except Exception as e:
            print(f"❌ Failed to train {model_name} ensemble model: {e}")
            import traceback
            traceback.print_exc()
            training_results[model_name] = {'error': str(e)}
    
    # Save evaluation report
    report_path = save_ensemble_evaluation_report(training_results, base_dir)
    
    # Print overall summary
    print(f"\n{'='*70}")
    print(f"🎯 Ensemble Training Summary")
    print(f"{'='*70}")
    
    successful_models = []
    failed_models = []
    
    for model_name, results in training_results.items():
        if 'error' in results:
            failed_models.append(model_name)
            print(f"❌ {model_name}: FAILED - {results['error']}")
        else:
            successful_models.append(model_name)
            val_r2 = results.get('val_r2', 0)
            xgb_r2 = results.get('xgb_val_r2', 0)
            rf_r2 = results.get('rf_val_r2', 0)
            improvement = val_r2 - max(xgb_r2, rf_r2)
            
            print(f"✅ {model_name}: Ensemble R² = {val_r2:.3f} (Improvement: {improvement:+.3f})")
    
    print(f"\n📊 Final Results:")
    print(f"   Successfully trained: {len(successful_models)} ensemble models")
    print(f"   Failed: {len(failed_models)} ensemble models")
    
    if successful_models:
        print(f"   Successful models: {', '.join(successful_models)}")
        
        # Show best performing ensemble
        best_model = max(successful_models, 
                        key=lambda x: training_results[x].get('val_r2', 0))
        best_score = training_results[best_model].get('val_r2', 0)
        print(f"   🏆 Best performing: {best_model} (Validation R² = {best_score:.3f})")
        
        # Show ensemble effectiveness
        total_improvement = 0
        for model in successful_models:
            results = training_results[model]
            ensemble_r2 = results.get('val_r2', 0)
            xgb_r2 = results.get('xgb_val_r2', 0)
            rf_r2 = results.get('rf_val_r2', 0)
            improvement = ensemble_r2 - max(xgb_r2, rf_r2)
            total_improvement += improvement
        
        avg_improvement = total_improvement / len(successful_models)
        print(f"   📈 Average ensemble improvement: {avg_improvement:+.3f} R²")
    
    if failed_models:
        print(f"   ❌ Failed models: {', '.join(failed_models)}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review evaluation report: {report_path}")
    print(f"   2. Ensemble models saved in: {base_dir / 'ensemble_models'}")
    print(f"   3. Compare with individual model performance")
    print(f"   4. Run ensemble predictions on new data")
    
    return training_results, base_dir

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all ensemble personality models")
    parser.add_argument("--input", default="../final_creator_scores_with_targets.csv", 
                       help="Input CSV file with features and target variables")
    
    args = parser.parse_args()
    
    train_all_ensemble_models(args.input)

if __name__ == "__main__":
    main()