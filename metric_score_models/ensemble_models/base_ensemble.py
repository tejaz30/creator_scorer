"""
Base Ensemble Model combining XGBoost and Random Forest.

Provides the foundation for all personality trait ensemble models.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    XGBOOST_PARAMS, RANDOM_FOREST_PARAMS, RIDGE_PARAMS, KNN_PARAMS, ENSEMBLE_WEIGHTS,
    ENSEMBLE_META_LEARNING, ENSEMBLE_CV_FOLDS, PERSONALITY_SCORE_RANGES
)

class BaseEnsembleModel:
    """
    Base ensemble model combining XGBoost, Random Forest, Ridge Regression, and KNN.
    
    Features:
    - 4-model ensemble: XGBoost (35%) + Random Forest (25%) + Ridge (25%) + KNN (15%)
    - Optional meta-learning for optimal weight discovery
    - Combined feature importance from tree-based models (XGBoost + Random Forest)
    - Uncertainty quantification through model disagreement
    - Diverse learning approaches: gradient boosting, bagging, linear, instance-based
    """
    
    def __init__(self, trait_name, features, weights=None, use_meta_learning=None):
        """
        Initialize the ensemble model.
        
        Args:
            trait_name: Name of personality trait (e.g., 'aspirational')
            features: List of feature names to use
            weights: Custom weights dict, defaults to config
            use_meta_learning: Whether to use meta-learner, defaults to config
        """
        self.trait_name = trait_name
        self.features = features
        self.weights = weights or ENSEMBLE_WEIGHTS.copy()
        self.use_meta_learning = use_meta_learning if use_meta_learning is not None else ENSEMBLE_META_LEARNING
        
        # Initialize models
        self.xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        self.rf_model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
        
        # Handle Ridge regression normalize parameter (deprecated in newer sklearn)
        ridge_params = RIDGE_PARAMS.copy()
        if 'normalize' in ridge_params:
            ridge_params.pop('normalize')  # Remove deprecated parameter
        self.ridge_model = Ridge(**ridge_params)
        
        self.knn_model = KNeighborsRegressor(**KNN_PARAMS)
        
        # Scaler for Ridge and KNN (they need scaled features)
        self.feature_scaler = StandardScaler()
        
        # Meta-learner for optimal weight discovery
        self.meta_learner = LinearRegression() if self.use_meta_learning else None
        self.meta_scaler = StandardScaler()  # For meta-learner features
        
        # Training state
        self.is_fitted = False
        self.training_metrics = {}
        self.feature_importance_ = {}
        
        # Score range for this trait
        self.score_range = PERSONALITY_SCORE_RANGES.get(trait_name, (1, 10))
        
    def _prepare_features(self, df):
        """Prepare and validate features."""
        df_processed = df.copy()
        
        # Ensure all features exist
        for feature in self.features:
            if feature not in df_processed.columns:
                print(f"  ⚠️ Missing feature {feature}, filling with default")
                df_processed[feature] = 0.0
        
        # Fill missing values with feature-specific defaults
        for feature in self.features:
            if df_processed[feature].isnull().any():
                if 'score' in feature.lower() or 'density' in feature.lower():
                    default_val = df_processed[feature].median()
                else:
                    default_val = 0.0
                df_processed[feature] = df_processed[feature].fillna(default_val)
        
        return df_processed[self.features]
    
    def fit(self, df, target_column, test_size=0.2, random_state=42):
        """
        Train the ensemble model.
        
        Args:
            df: Training dataframe
            target_column: Name of target column
            test_size: Validation split size
            random_state: Random seed
            
        Returns:
            dict: Training metrics
        """
        print(f"🚀 Training {self.trait_name.title()} Ensemble Model...")
        
        # Prepare data
        X = self._prepare_features(df)
        y = df[target_column].copy()
        
        # Remove missing targets
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        print(f"   Training on {len(X)} samples with {len(self.features)} features")
        print(f"   Features: {self.features[:3]}{'...' if len(self.features) > 3 else ''}")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Train individual models
        print("   🔧 Training XGBoost...")
        self.xgb_model.fit(X_train, y_train)
        
        print("   🔧 Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        
        # Scale features for Ridge and KNN
        print("   🔧 Scaling features for Ridge and KNN...")
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        print("   🔧 Training Ridge Regression...")
        self.ridge_model.fit(X_train_scaled, y_train)
        
        print("   🔧 Training KNN...")
        self.knn_model.fit(X_train_scaled, y_train)
        
        # Get individual predictions
        xgb_pred_train = self.xgb_model.predict(X_train)
        xgb_pred_val = self.xgb_model.predict(X_val)
        rf_pred_train = self.rf_model.predict(X_train)
        rf_pred_val = self.rf_model.predict(X_val)
        ridge_pred_train = self.ridge_model.predict(X_train_scaled)
        ridge_pred_val = self.ridge_model.predict(X_val_scaled)
        knn_pred_train = self.knn_model.predict(X_train_scaled)
        knn_pred_val = self.knn_model.predict(X_val_scaled)
        
        # Train meta-learner if enabled
        if self.use_meta_learning:
            print("   🧠 Training meta-learner...")
            # Use cross-validation to get unbiased predictions for meta-learning
            meta_features_train = np.column_stack([
                cross_val_predict(self.xgb_model, X_train, y_train, cv=ENSEMBLE_CV_FOLDS),
                cross_val_predict(self.rf_model, X_train, y_train, cv=ENSEMBLE_CV_FOLDS),
                cross_val_predict(self.ridge_model, X_train_scaled, y_train, cv=ENSEMBLE_CV_FOLDS),
                cross_val_predict(self.knn_model, X_train_scaled, y_train, cv=ENSEMBLE_CV_FOLDS)
            ])
            
            # Scale meta-features
            meta_features_train_scaled = self.meta_scaler.fit_transform(meta_features_train)
            self.meta_learner.fit(meta_features_train_scaled, y_train)
            
            # Get meta-learner predictions
            meta_features_val = np.column_stack([xgb_pred_val, rf_pred_val, ridge_pred_val, knn_pred_val])
            meta_features_val_scaled = self.meta_scaler.transform(meta_features_val)
            ensemble_pred_train = self.meta_learner.predict(self.meta_scaler.transform(
                np.column_stack([xgb_pred_train, rf_pred_train, ridge_pred_train, knn_pred_train])
            ))
            ensemble_pred_val = self.meta_learner.predict(meta_features_val_scaled)
            
            # Update weights based on meta-learner coefficients
            if len(self.meta_learner.coef_) >= 4:
                coef_abs = np.abs(self.meta_learner.coef_)
                total_weight = np.sum(coef_abs)
                if total_weight > 0:
                    self.weights['xgboost'] = coef_abs[0] / total_weight
                    self.weights['random_forest'] = coef_abs[1] / total_weight
                    self.weights['ridge'] = coef_abs[2] / total_weight
                    self.weights['knn'] = coef_abs[3] / total_weight
                    print(f"   📊 Meta-learner weights: XGB={self.weights['xgboost']:.3f}, RF={self.weights['random_forest']:.3f}, Ridge={self.weights['ridge']:.3f}, KNN={self.weights['knn']:.3f}")
        else:
            # Use weighted average
            ensemble_pred_train = (
                self.weights['xgboost'] * xgb_pred_train + 
                self.weights['random_forest'] * rf_pred_train +
                self.weights['ridge'] * ridge_pred_train +
                self.weights['knn'] * knn_pred_train
            )
            ensemble_pred_val = (
                self.weights['xgboost'] * xgb_pred_val + 
                self.weights['random_forest'] * rf_pred_val +
                self.weights['ridge'] * ridge_pred_val +
                self.weights['knn'] * knn_pred_val
            )
        
        # Clip predictions to valid range
        ensemble_pred_train = np.clip(ensemble_pred_train, *self.score_range)
        ensemble_pred_val = np.clip(ensemble_pred_val, *self.score_range)
        
        # Calculate metrics
        self.training_metrics = self._calculate_metrics(
            y_train, y_val, ensemble_pred_train, ensemble_pred_val,
            xgb_pred_train, xgb_pred_val, rf_pred_train, rf_pred_val,
            ridge_pred_train, ridge_pred_val, knn_pred_train, knn_pred_val
        )
        
        # Calculate combined feature importance
        self._calculate_feature_importance()
        
        # Verify feature importance was calculated
        if not hasattr(self, 'feature_importance_') or not self.feature_importance_:
            print(f"   ⚠️ Feature importance calculation failed for {self.trait_name}")
            # Force recalculation
            self.feature_importance_ = {}
            try:
                xgb_importance = dict(zip(self.features, self.xgb_model.feature_importances_))
                rf_importance = dict(zip(self.features, self.rf_model.feature_importances_))
                
                for feature in self.features:
                    combined_importance = (
                        self.weights['xgboost'] * xgb_importance[feature] +
                        self.weights['random_forest'] * rf_importance[feature]
                    )
                    self.feature_importance_[feature] = combined_importance
                
                self.xgb_feature_importance_ = xgb_importance
                self.rf_feature_importance_ = rf_importance
                print(f"   ✅ Feature importance recalculated: {len(self.feature_importance_)} features")
            except Exception as e:
                print(f"   ❌ Feature importance recalculation failed: {e}")
        else:
            print(f"   ✅ Feature importance calculated: {len(self.feature_importance_)} features")
        
        self.is_fitted = True
        
        # Store predictions for enhanced metrics
        self.y_test = y_val
        self.y_pred = ensemble_pred_val
        
        # Print results
        self._print_training_results()
        
        return self.training_metrics
    
    def _calculate_metrics(self, y_train, y_val, ens_pred_train, ens_pred_val,
                          xgb_pred_train, xgb_pred_val, rf_pred_train, rf_pred_val,
                          ridge_pred_train, ridge_pred_val, knn_pred_train, knn_pred_val):
        """Calculate comprehensive training metrics."""
        
        # Ensemble metrics
        train_r2 = r2_score(y_train, ens_pred_train)
        val_r2 = r2_score(y_val, ens_pred_val)
        train_rmse = np.sqrt(mean_squared_error(y_train, ens_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, ens_pred_val))
        train_mae = mean_absolute_error(y_train, ens_pred_train)
        val_mae = mean_absolute_error(y_val, ens_pred_val)
        
        # Individual model metrics for comparison
        xgb_val_r2 = r2_score(y_val, xgb_pred_val)
        rf_val_r2 = r2_score(y_val, rf_pred_val)
        ridge_val_r2 = r2_score(y_val, ridge_pred_val)
        knn_val_r2 = r2_score(y_val, knn_pred_val)
        
        # Accuracy within tolerance
        val_acc_05 = np.mean(np.abs(y_val - ens_pred_val) <= 0.5) * 100
        val_acc_10 = np.mean(np.abs(y_val - ens_pred_val) <= 1.0) * 100
        
        # Model disagreement (uncertainty measure) - use standard deviation of all predictions
        all_predictions = np.column_stack([xgb_pred_val, rf_pred_val, ridge_pred_val, knn_pred_val])
        model_disagreement = np.mean(np.std(all_predictions, axis=1))
        
        # MAPE
        try:
            from sklearn.metrics import mean_absolute_percentage_error
            val_mape = mean_absolute_percentage_error(y_val, ens_pred_val) * 100
        except:
            val_mape = 0.0
        
        return {
            # Ensemble metrics
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': val_r2,  # For compatibility
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'test_rmse': val_rmse,  # For compatibility
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': val_mae,  # For compatibility
            'val_mape': val_mape,
            'test_mape': val_mape,  # For compatibility
            'val_acc_05': val_acc_05,
            'test_acc_05': val_acc_05,  # For compatibility
            'val_acc_10': val_acc_10,
            'test_acc_10': val_acc_10,  # For compatibility
            
            # Individual model comparison
            'xgb_val_r2': xgb_val_r2,
            'rf_val_r2': rf_val_r2,
            'ridge_val_r2': ridge_val_r2,
            'knn_val_r2': knn_val_r2,
            'model_disagreement': model_disagreement,
            
            # Sample counts
            'n_train_samples': len(y_train),
            'n_val_samples': len(y_val),
            
            # Weights used
            'ensemble_weights': self.weights.copy()
        }
    
    def _calculate_feature_importance(self):
        """Calculate combined feature importance from tree-based models (XGBoost + Random Forest)."""
        try:
            # Get individual importances from tree-based models only
            # Ridge and KNN don't have feature_importances_ attribute
            xgb_importance = dict(zip(self.features, self.xgb_model.feature_importances_))
            rf_importance = dict(zip(self.features, self.rf_model.feature_importances_))
            
            # For Ridge, we can use coefficient magnitudes as importance (if needed)
            ridge_importance = {}
            try:
                ridge_coef = np.abs(self.ridge_model.coef_)
                # Normalize coefficients to sum to 1 (like feature importance)
                ridge_coef_norm = ridge_coef / np.sum(ridge_coef) if np.sum(ridge_coef) > 0 else ridge_coef
                ridge_importance = dict(zip(self.features, ridge_coef_norm))
            except:
                ridge_importance = {feature: 0.0 for feature in self.features}
            
            # KNN doesn't have interpretable feature importance, set to uniform
            knn_importance = {feature: 1.0/len(self.features) for feature in self.features}
            
            # Combine using ensemble weights (focus on tree-based models for interpretability)
            # Weight tree-based models more heavily for feature importance
            tree_weight = self.weights['xgboost'] + self.weights['random_forest']
            linear_weight = self.weights['ridge']
            
            self.feature_importance_ = {}
            for feature in self.features:
                # Combine tree-based importances (primary) with Ridge coefficients (secondary)
                if tree_weight > 0:
                    tree_combined = (
                        (self.weights['xgboost'] / tree_weight) * xgb_importance[feature] +
                        (self.weights['random_forest'] / tree_weight) * rf_importance[feature]
                    )
                    combined_importance = (
                        0.8 * tree_combined +  # 80% from tree models
                        0.2 * ridge_importance[feature]  # 20% from Ridge coefficients
                    )
                else:
                    combined_importance = ridge_importance[feature]
                
                self.feature_importance_[feature] = combined_importance
            
            # Store individual importances for analysis
            self.xgb_feature_importance_ = xgb_importance
            self.rf_feature_importance_ = rf_importance
            self.ridge_feature_importance_ = ridge_importance
            self.knn_feature_importance_ = knn_importance
            
        except Exception as e:
            print(f"   ❌ Feature importance calculation error: {e}")
            self.feature_importance_ = {}
            self.xgb_feature_importance_ = {}
            self.rf_feature_importance_ = {}
            self.ridge_feature_importance_ = {}
            self.knn_feature_importance_ = {}
    
    def predict(self, df):
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self._prepare_features(df)
        X_scaled = self.feature_scaler.transform(X)
        
        # Get individual predictions
        xgb_pred = self.xgb_model.predict(X)
        rf_pred = self.rf_model.predict(X)
        ridge_pred = self.ridge_model.predict(X_scaled)
        knn_pred = self.knn_model.predict(X_scaled)
        
        if self.use_meta_learning and self.meta_learner is not None:
            # Use meta-learner
            meta_features = np.column_stack([xgb_pred, rf_pred, ridge_pred, knn_pred])
            meta_features_scaled = self.meta_scaler.transform(meta_features)
            ensemble_pred = self.meta_learner.predict(meta_features_scaled)
        else:
            # Use weighted average
            ensemble_pred = (
                self.weights['xgboost'] * xgb_pred + 
                self.weights['random_forest'] * rf_pred +
                self.weights['ridge'] * ridge_pred +
                self.weights['knn'] * knn_pred
            )
        
        # Clip to valid range
        ensemble_pred = np.clip(ensemble_pred, *self.score_range)
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, df):
        """Predict with uncertainty estimation based on model disagreement."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self._prepare_features(df)
        X_scaled = self.feature_scaler.transform(X)
        
        # Get individual predictions
        xgb_pred = self.xgb_model.predict(X)
        rf_pred = self.rf_model.predict(X)
        ridge_pred = self.ridge_model.predict(X_scaled)
        knn_pred = self.knn_model.predict(X_scaled)
        
        # Calculate ensemble prediction
        ensemble_pred = self.predict(df)
        
        # Uncertainty based on model disagreement (standard deviation of all predictions)
        all_predictions = np.column_stack([xgb_pred, rf_pred, ridge_pred, knn_pred])
        uncertainty = np.std(all_predictions, axis=1)
        
        # Confidence intervals (simple approach)
        confidence_interval = uncertainty * 1.96  # ~95% confidence
        lower_bound = np.clip(ensemble_pred - confidence_interval, *self.score_range)
        upper_bound = np.clip(ensemble_pred + confidence_interval, *self.score_range)
        
        return {
            'predictions': ensemble_pred,
            'uncertainty': uncertainty,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'xgb_predictions': xgb_pred,
            'rf_predictions': rf_pred,
            'ridge_predictions': ridge_pred,
            'knn_predictions': knn_pred
        }
    
    def get_feature_importance(self, top_n=None):
        """Get combined feature importance rankings."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance, 'model': 'ensemble'}
            for feature, importance in self.feature_importance_.items()
        ])
        
        # Add individual model importances for comparison
        for feature in self.features:
            importance_df = pd.concat([
                importance_df,
                pd.DataFrame([{
                    'feature': feature,
                    'importance': self.xgb_feature_importance_[feature],
                    'model': 'xgboost'
                }]),
                pd.DataFrame([{
                    'feature': feature,
                    'importance': self.rf_feature_importance_[feature],
                    'model': 'random_forest'
                }])
            ], ignore_index=True)
        
        # Sort ensemble importance
        ensemble_importance = importance_df[importance_df['model'] == 'ensemble'].sort_values(
            'importance', ascending=False
        )
        
        if top_n:
            ensemble_importance = ensemble_importance.head(top_n)
        
        return ensemble_importance, importance_df
    
    def save(self, filepath):
        """Save the ensemble model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'trait_name': self.trait_name,
            'features': self.features,
            'weights': self.weights,
            'use_meta_learning': self.use_meta_learning,
            'xgb_model': self.xgb_model,
            'rf_model': self.rf_model,
            'ridge_model': self.ridge_model,
            'knn_model': self.knn_model,
            'feature_scaler': self.feature_scaler,
            'meta_learner': self.meta_learner,
            'meta_scaler': self.meta_scaler,
            'feature_importance_': self.feature_importance_,
            'training_metrics': self.training_metrics,
            'score_range': self.score_range,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"   ✅ {self.trait_name.title()} ensemble model saved to {filepath}")
    
    def load(self, filepath):
        """Load a trained ensemble model."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.trait_name = model_data['trait_name']
        self.features = model_data['features']
        self.weights = model_data['weights']
        self.use_meta_learning = model_data['use_meta_learning']
        self.xgb_model = model_data['xgb_model']
        self.rf_model = model_data['rf_model']
        self.ridge_model = model_data['ridge_model']
        self.knn_model = model_data['knn_model']
        self.feature_scaler = model_data['feature_scaler']
        self.meta_learner = model_data['meta_learner']
        self.meta_scaler = model_data['meta_scaler']
        self.feature_importance_ = model_data['feature_importance_']
        self.training_metrics = model_data['training_metrics']
        self.score_range = model_data['score_range']
        self.is_fitted = model_data['is_fitted']
        
        print(f"   ✅ {self.trait_name.title()} ensemble model loaded from {filepath}")
    
    def _print_training_results(self):
        """Print training results."""
        metrics = self.training_metrics
        
        print(f"\n📊 {self.trait_name.title()} Ensemble Results:")
        print(f"   Ensemble Train R²:  {metrics['train_r2']:.3f}")
        print(f"   Ensemble Val R²:    {metrics['val_r2']:.3f}")
        print(f"   Ensemble Val RMSE:  {metrics['val_rmse']:.3f}")
        print(f"   Ensemble Val MAE:   {metrics['val_mae']:.3f}")
        print(f"   Val Accuracy ±1.0:  {metrics['val_acc_10']:.1f}%")
        
        print(f"\n🔍 Individual Model Comparison:")
        print(f"   XGBoost Val R²:     {metrics['xgb_val_r2']:.3f}")
        print(f"   Random Forest Val R²: {metrics['rf_val_r2']:.3f}")
        print(f"   Ridge Regression Val R²: {metrics['ridge_val_r2']:.3f}")
        print(f"   KNN Val R²:         {metrics['knn_val_r2']:.3f}")
        print(f"   Model Disagreement: {metrics['model_disagreement']:.3f}")
        
        print(f"\n⚖️ Final Ensemble Weights:")
        print(f"   XGBoost:      {self.weights['xgboost']:.3f}")
        print(f"   Random Forest: {self.weights['random_forest']:.3f}")
        print(f"   Ridge:        {self.weights['ridge']:.3f}")
        print(f"   KNN:          {self.weights['knn']:.3f}")
        
        # Show top 3 most important features
        if self.feature_importance_:
            top_features = sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"\n🎯 Top 3 Features:")
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"   {i}. {feature}: {importance:.3f}")