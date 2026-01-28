"""
Aspirational Ensemble Model
Combines XGBoost and Random Forest for aspirational personality scoring.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .base_ensemble import BaseEnsembleModel
from config import ASPIRATIONAL_FEATURES

class AspirationalEnsemble(BaseEnsembleModel):
    """Ensemble model for aspirational personality scoring."""
    
    def __init__(self, weights=None, use_meta_learning=None):
        """Initialize aspirational ensemble model."""
        super().__init__(
            trait_name='aspirational',
            features=ASPIRATIONAL_FEATURES,
            weights=weights,
            use_meta_learning=use_meta_learning
        )

# Global instance for compatibility
aspirational_ensemble = AspirationalEnsemble()