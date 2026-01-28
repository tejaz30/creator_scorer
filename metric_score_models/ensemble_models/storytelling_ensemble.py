"""
Storytelling Ensemble Model
Combines XGBoost and Random Forest for storytelling personality scoring.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .base_ensemble import BaseEnsembleModel
from config import STORYTELLING_FEATURES

class StorytellingEnsemble(BaseEnsembleModel):
    """Ensemble model for storytelling personality scoring."""
    
    def __init__(self, weights=None, use_meta_learning=None):
        """Initialize storytelling ensemble model."""
        super().__init__(
            trait_name='story_telling',  # Note: using 'story_telling' to match target column
            features=STORYTELLING_FEATURES,
            weights=weights,
            use_meta_learning=use_meta_learning
        )

# Global instance for compatibility
storytelling_ensemble = StorytellingEnsemble()