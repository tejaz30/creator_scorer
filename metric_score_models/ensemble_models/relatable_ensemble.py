"""
Relatable Ensemble Model
Combines XGBoost and Random Forest for relatable personality scoring.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .base_ensemble import BaseEnsembleModel
from config import RELATABLE_FEATURES

class RelatableEnsemble(BaseEnsembleModel):
    """Ensemble model for relatable personality scoring."""
    
    def __init__(self, weights=None, use_meta_learning=None):
        """Initialize relatable ensemble model."""
        super().__init__(
            trait_name='relatable',
            features=RELATABLE_FEATURES,
            weights=weights,
            use_meta_learning=use_meta_learning
        )

# Global instance for compatibility
relatable_ensemble = RelatableEnsemble()