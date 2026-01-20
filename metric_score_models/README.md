# Creator Scoring Models

This directory contains machine learning models for scoring Instagram creators across multiple dimensions.

## 📁 Folder Organization

### Core Models
- `model.py` - Credibility scoring model (XGBoost)
- `storytelling_model.py` - Storytelling scoring model (XGBoost)
- `communication_model.py` - Communication scoring model (XGBoost)
- `combined_scorer.py` - Unified interface for all models

### Preprocessors
- `preprocessor.py` - Credibility feature preprocessing
- `storytelling_preprocessor.py` - Storytelling feature preprocessing
- `communication_preprocessor.py` - Communication feature preprocessing

### Training Scripts (Individual)
- `train_credibility.py` - Train credibility model only
- `train_storytelling.py` - Train storytelling model only
- `train_communication.py` - Train communication model only

### Prediction & Evaluation (Unified)
- `predict_evaluate.py` - **Main script** for predictions and evaluation of all models

### Configuration
- `config.py` - Model configuration, features, and paths

### Legacy Files (Deprecated)
- `train_combined.py` - Old combined training script
- `predict_combined.py` - Old combined prediction script
- `predict.py`, `evaluate.py`, `train.py` - Old individual scripts
- `predict_communication.py`, `train_communication.py` - Old communication scripts

## 🚀 Quick Start

### 1. Train Models (Individual)

Each model should be trained separately as they use different features and may require different approaches:

```bash
# Train credibility model
python metric_score_models/train_credibility.py --sample-labels --save-labeled-data

# Train storytelling model
python metric_score_models/train_storytelling.py --sample-labels --save-labeled-data

# Train communication model
python metric_score_models/train_communication.py --sample-labels --save-labeled-data
```

### 2. Make Predictions (Unified)

Use the single prediction script for all models:

```bash
# Predict with all available models
python metric_score_models/predict_evaluate.py

# Predict with specific models
python metric_score_models/predict_evaluate.py --models credibility storytelling

# Single model prediction
python metric_score_models/predict_evaluate.py --models communication

# With evaluation and insights
python metric_score_models/predict_evaluate.py --evaluate --generate-insights --show-features
```

## 📊 Model Details

### Credibility Model
**Purpose**: Measures trustworthiness and reliability of creators
**Features**: 
- `eye_contact_avg_score_0_10` - Direct eye contact with audience
- `mean_hist_score` - Visual quality consistency
- `mean_scene_score` - Scene composition quality
- `mean_face_density` - Face presence in content
- `gemini_is_marketing` - Marketing content ratio (inverse)
- `gemini_comment_sentiment_counts.agreeing` - Positive audience response
- `series_reel_mean` - Content series consistency

**Categories**: Low, Medium, High, Very High

### Storytelling Model
**Purpose**: Evaluates narrative and engagement quality
**Features**:
- `avg_captioned_reels` - Caption usage for accessibility
- `gemini_has_humour` - Humor in content
- `gemini_comment_sentiment_counts.agreeing` - Positive engagement
- `gemini_comment_sentiment_counts.neutral` - Thoughtful responses
- `mean_face_density` - Personal connection
- `outlier_2sigma_ratio` - Content variety

**Categories**: Poor, Fair, Good, Excellent

### Communication Model
**Purpose**: Assesses audience connection and interaction quality
**Features**:
- `eye_contact_avg_score_0_10` - Direct audience connection
- `avg_captioned_reels` - Accessibility and clarity
- `gemini_is_marketing` - Personal vs promotional balance
- `gemini_has_humour` - Engagement through humor
- `mean_face_density` - Personal presence

**Categories**: Weak, Moderate, Strong, Excellent

## 🎯 Overall Scoring

The combined system provides an overall score using weighted averages:
- **Credibility**: 40% (trust and reliability)
- **Storytelling**: 30% (content engagement)
- **Communication**: 30% (audience connection)

## 📈 Output Formats

### CSV Output
- Individual model scores (1-10 scale)
- Confidence intervals (if enabled)
- Category classifications
- Overall weighted score
- All original features

### JSON Output
- Same data in JSON format for API integration

### Insights Report
- Score distributions and statistics
- Top performers analysis
- Feature correlations
- Model performance metrics

## 🔧 Advanced Usage

### Custom Input Data
```bash
python metric_score_models/predict_evaluate.py --input custom_data.csv --output results.csv
```

### Evaluation with Ground Truth
```bash
python metric_score_models/predict_evaluate.py --evaluate --input labeled_data.csv
```

### Feature Importance Analysis
```bash
python metric_score_models/predict_evaluate.py --show-features
```

### Multiple Output Formats
```bash
python metric_score_models/predict_evaluate.py --format both --generate-insights
```

## 📝 Adding New Models

To add a new scoring model:

1. Create `new_model.py` following the pattern of existing models
2. Add features to `config.py`
3. Create `train_new_model.py` training script
4. Update `combined_scorer.py` to include the new model
5. Update `predict_evaluate.py` to handle the new model

## 🗂️ File Structure

```
metric_score_models/
├── README.md                    # This file
├── config.py                    # Configuration and features
├── combined_scorer.py           # Unified model interface
│
├── model.py                     # Credibility model
├── storytelling_model.py        # Storytelling model  
├── communication_model.py       # Communication model
│
├── preprocessor.py              # Credibility preprocessing
├── storytelling_preprocessor.py # Storytelling preprocessing
├── communication_preprocessor.py# Communication preprocessing
│
├── train_credibility.py         # Train credibility model
├── train_storytelling.py        # Train storytelling model
├── train_communication.py       # Train communication model
│
├── predict_evaluate.py          # Main prediction & evaluation script
│
├── trained_models/              # Saved model files
├── results/                     # Prediction outputs and reports
└── training_plots/              # Feature importance plots
```

## 🚨 Migration from Old Scripts

If you were using the old scripts:
- Replace `train_combined.py` → Use individual `train_*.py` scripts
- Replace `predict_combined.py` → Use `predict_evaluate.py`
- Replace individual predict scripts → Use `predict_evaluate.py --models <model_name>`

The new organization provides better separation of concerns and easier maintenance for adding future models.