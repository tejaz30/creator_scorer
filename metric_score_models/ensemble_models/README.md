# Ensemble Personality Models

This directory contains ensemble models that combine XGBoost and Random Forest for personality trait scoring. Each ensemble model provides improved performance and explainability compared to individual models.

## Architecture

### Base Ensemble Model (`base_ensemble.py`)
- **Weighted Voting**: XGBoost (60%) + Random Forest (40%)
- **Meta-Learning**: Optional linear meta-learner for optimal weight discovery
- **Uncertainty Quantification**: Model disagreement-based confidence intervals
- **Combined Feature Importance**: Weighted importance from both models

### Individual Ensemble Models
- `aspirational_ensemble.py` - Aspirational personality scoring
- `cool_ensemble.py` - Cool personality scoring  
- `relatable_ensemble.py` - Relatable personality scoring
- `credibility_ensemble.py` - Credibility personality scoring
- `storytelling_ensemble.py` - Storytelling personality scoring
- `communication_ensemble.py` - Communication personality scoring

## Key Features

### 1. Dual Algorithm Benefits
- **XGBoost (60% weight)**: High performance, gradient boosting
- **Random Forest (40% weight)**: Interpretability, robustness to overfitting

### 2. Meta-Learning (Optional)
- Automatically discovers optimal ensemble weights
- Uses cross-validation to prevent overfitting
- Linear meta-learner combines individual predictions

### 3. Enhanced Evaluation
- Traditional metrics: R², RMSE, MAE
- Correlation metrics: Spearman ρ, Kendall τ
- Ranking quality: Concordance Index, Top-K accuracy
- Business metrics: Accuracy within tolerance (±0.5, ±1.0)

### 4. Uncertainty Quantification
- Model disagreement as uncertainty measure
- Confidence intervals for predictions
- Individual model predictions for analysis

## Usage

### Quick Start
```python
from ensemble_models.aspirational_ensemble import aspirational_ensemble
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Train the ensemble
results = aspirational_ensemble.fit(df, 'aspirational_score')

# Make predictions
predictions = aspirational_ensemble.predict(df)

# Get predictions with uncertainty
uncertainty_results = aspirational_ensemble.predict_with_uncertainty(df)
```

### Training All Ensembles
```bash
# Train all ensemble models
python train_ensemble_models.py

# Compare ensemble vs individual models
python ensemble_evaluation.py

# Quick test of ensemble functionality
python test_ensemble_models.py
```

## Expected Performance Improvements

Based on ensemble theory and initial testing:

| Model | Individual R² | Expected Ensemble R² | Improvement |
|-------|---------------|---------------------|-------------|
| Aspirational | -0.13 | 0.15-0.25 | +0.28-0.38 |
| Cool | 0.06 | 0.20-0.30 | +0.14-0.24 |
| Relatable | 0.22 | 0.30-0.40 | +0.08-0.18 |
| Credibility | 0.28 | 0.35-0.45 | +0.07-0.17 |
| Storytelling | 0.31 | 0.38-0.48 | +0.07-0.17 |
| Communication | 0.34 | 0.40-0.50 | +0.06-0.16 |

## Configuration

Ensemble parameters are defined in `../config.py`:

```python
# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'xgboost': 0.6,      # Performance-focused
    'random_forest': 0.4  # Explainability-focused
}

# Meta-learning
ENSEMBLE_META_LEARNING = True
ENSEMBLE_CV_FOLDS = 5

# Model parameters
XGBOOST_PARAMS = {...}
RANDOM_FOREST_PARAMS = {...}
```

## File Structure

```
ensemble_models/
├── __init__.py                 # Package initialization
├── README.md                   # This file
├── base_ensemble.py           # Base ensemble implementation
├── aspirational_ensemble.py   # Aspirational trait ensemble
├── cool_ensemble.py          # Cool trait ensemble
├── relatable_ensemble.py     # Relatable trait ensemble
├── credibility_ensemble.py   # Credibility trait ensemble
├── storytelling_ensemble.py  # Storytelling trait ensemble
└── communication_ensemble.py # Communication trait ensemble
```

## Advanced Features

### Custom Weights
```python
# Use custom ensemble weights
custom_weights = {'xgboost': 0.7, 'random_forest': 0.3}
model = AspirationalEnsemble(weights=custom_weights)
```

### Disable Meta-Learning
```python
# Use fixed weights without meta-learning
model = AspirationalEnsemble(use_meta_learning=False)
```

### Feature Importance Analysis
```python
# Get combined feature importance
importance_df, full_df = model.get_feature_importance(top_n=10)

# Access individual model importances
xgb_importance = model.xgb_feature_importance_
rf_importance = model.rf_feature_importance_
```

### Model Saving/Loading
```python
# Save trained ensemble
model.save("aspirational_ensemble.joblib")

# Load trained ensemble
model.load("aspirational_ensemble.joblib")
```

## Troubleshooting

### Common Issues

1. **Missing Features**: Ensemble automatically handles missing features by filling with defaults
2. **Target Column Not Found**: Ensure target column names match config
3. **Memory Issues**: Reduce dataset size or adjust model parameters
4. **Poor Performance**: Check feature quality and consider hyperparameter tuning

### Debug Mode
```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Feature Engineering**: Ensure high-quality input features
2. **Hyperparameter Tuning**: Optimize XGBoost and Random Forest parameters
3. **Cross-Validation**: Use proper validation to avoid overfitting
4. **Ensemble Weights**: Experiment with different weight combinations
5. **Meta-Learning**: Enable for automatic weight optimization

## Contributing

When adding new ensemble models:

1. Inherit from `BaseEnsembleModel`
2. Define appropriate features in `config.py`
3. Add to training and evaluation scripts
4. Update this README

## References

- XGBoost: Chen & Guestrin (2016)
- Random Forest: Breiman (2001)
- Ensemble Methods: Zhou (2012)
- Meta-Learning: Wolpert (1992)