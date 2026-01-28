"""
Credibility Ensemble Model
Combines XGBoost and Random Forest for credibility personality scoring.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .base_ensemble import BaseEnsembleModel
from config import CREDIBILITY_FEATURES

class CredibilityEnsemble(BaseEnsembleModel):
    """Ensemble model for credibility personality scoring."""
    
    def __init__(self, weights=None, use_meta_learning=None):
        """Initialize credibility ensemble model."""
        super().__init__(
            trait_name='credible',  # Note: using 'credible' to match target column
            features=CREDIBILITY_FEATURES,
            weights=weights,
            use_meta_learning=use_meta_learning
        )

# Global instance for compatibility
credibility_ensemble = CredibilityEnsemble()