"""
Configuration for credibility scoring models.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent  # Parent directory where CSV files are located
MODELS_DIR = BASE_DIR / "trained_models"
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