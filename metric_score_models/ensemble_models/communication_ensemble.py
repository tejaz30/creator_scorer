"""
Communication Ensemble Model
Combines XGBoost and Random Forest for communication personality scoring.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .base_ensemble import BaseEnsembleModel
from config import COMMUNICATION_FEATURES

class CommunicationEnsemble(BaseEnsembleModel):
    """Ensemble model for communication personality scoring."""
    
    def __init__(self, weights=None, use_meta_learning=None):
        """Initialize communication ensemble model."""
        super().__init__(
            trait_name='communication',
            features=COMMUNICATION_FEATURES,
            weights=weights,
            use_meta_learning=use_meta_learning
        )

# Global instance for compatibility
communication_ensemble = CommunicationEnsemble()