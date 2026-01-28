"""
Cool Ensemble Model
Combines XGBoost and Random Forest for cool personality scoring.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .base_ensemble import BaseEnsembleModel
from config import COOL_FEATURES

class CoolEnsemble(BaseEnsembleModel):
    """Ensemble model for cool personality scoring."""
    
    def __init__(self, weights=None, use_meta_learning=None):
        """Initialize cool ensemble model."""
        super().__init__(
            trait_name='cool',
            features=COOL_FEATURES,
            weights=weights,
            use_meta_learning=use_meta_learning
        )

# Global instance for compatibility
cool_ensemble = CoolEnsemble()