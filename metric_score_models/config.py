"""
Configuration for credibility scoring models.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent  # Parent directory where CSV files are located
MODELS_DIR = BASE_DIR / "trained_models_organized"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Input/Output files
INPUT_CSV = "final_creator_scores.csv"
LABELED_DATA_CSV = "credibility_labeled_data.csv"
PREDICTIONS_CSV = "credibility_predictions.csv"

# Model configuration
CREDIBILITY_FEATURES = [
    'eye_contact_avg_score_0_10',
    'mean_hist_score', 
    'mean_scene_score',
    'mean_face_density',
    'gemini_is_marketing',
    'gemini_comment_sentiment_counts.agreeing',
    'series_reel_mean'
]

STORYTELLING_FEATURES = [
    'avg_captioned_reels',
    'gemini_has_humour',
    'gemini_comment_sentiment_counts.agreeing',
    'gemini_comment_sentiment_counts.neutral',
    'mean_face_density',
    'outlier_2sigma_ratio'
]

COMMUNICATION_FEATURES = [
    'eye_contact_avg_score_0_10',
    'avg_captioned_reels',
    'gemini_is_marketing',
    'gemini_has_humour',
    'mean_face_density'
]

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1
}

# Cross-validation settings
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Credibility score range
MIN_CREDIBILITY_SCORE = 1
MAX_CREDIBILITY_SCORE = 10

# Storytelling score range
MIN_STORYTELLING_SCORE = 1
MAX_STORYTELLING_SCORE = 10

# Communication score range
MIN_COMMUNICATION_SCORE = 1
MAX_COMMUNICATION_SCORE = 10

# ============================================================================
# ENSEMBLE MODEL CONFIGURATION
# ============================================================================

# Ensemble weights (XGBoost + Random Forest + Ridge + KNN)
ENSEMBLE_WEIGHTS = {
    'xgboost': 0.35,        # Performance-focused (gradient boosting)
    'random_forest': 0.25,  # Stability-focused (bagging)
    'ridge': 0.25,          # Linear relationships (regularized linear)
    'knn': 0.15             # Local patterns (instance-based)
}

# Meta-learning configuration
ENSEMBLE_META_LEARNING = True  # Use meta-learner for optimal weights
ENSEMBLE_CV_FOLDS = 5

# Random Forest parameters for ensemble
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# Ridge Regression parameters for ensemble
RIDGE_PARAMS = {
    'alpha': 1.0,           # Regularization strength
    'fit_intercept': True,
    'normalize': False,     # Deprecated in newer sklearn, will handle in code
    'random_state': 42,
    'solver': 'auto'
}

# KNN parameters for ensemble
KNN_PARAMS = {
    'n_neighbors': 5,       # Number of neighbors
    'weights': 'distance',  # Weight by inverse distance
    'algorithm': 'auto',    # Let sklearn choose best algorithm
    'leaf_size': 30,
    'p': 2,                 # Minkowski distance parameter (2 = Euclidean)
    'metric': 'minkowski',
    'n_jobs': -1
}

# Personality trait features (from existing models)
ASPIRATIONAL_FEATURES = [
    'mean_hist_score',
    'gemini_genz_word_count', 
    'gemini_comment_sentiment_counts.appreciating',
    'series_reel_mean',
    'avg_captioned_reels',
    'mean_scene_score',
    'mean_face_density',
    'outlier_2sigma_ratio'
]

COOL_FEATURES = [
    'mean_hist_score',
    'gemini_comment_sentiment_counts.appreciating',
    'avg_english_pct_non_music',
    'gemini_has_humour',
    'series_reel_mean',
    'gemini_is_educational',
    'mean_scene_score',
    'mean_face_density'
]

RELATABLE_FEATURES = [
    'eye_contact_avg_score_0_10',
    'avg_captioned_reels',
    'gemini_comment_sentiment_counts.neutral',
    'gemini_has_humour',
    'gemini_is_educational',
    'gemini_comment_sentiment_counts.appreciating',
    'mean_scene_score',
    'mean_face_density'
]

# Score ranges for all personality traits
PERSONALITY_SCORE_RANGES = {
    'aspirational': (1, 10),
    'cool': (1, 10),
    'relatable': (1, 10),
    'credible': (1, 10),
    'communication': (1, 10),
    'story_telling': (1, 10)
}