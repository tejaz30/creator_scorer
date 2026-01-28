"""
Ensemble Models Package for Personality Scoring.

Combines XGBoost (performance) with Random Forest (explainability)
for robust personality trait prediction.
"""

from .base_ensemble import BaseEnsembleModel
from .aspirational_ensemble import AspirationalEnsemble
from .cool_ensemble import CoolEnsemble
from .relatable_ensemble import RelatableEnsemble
from .credibility_ensemble import CredibilityEnsemble
from .storytelling_ensemble import StorytellingEnsemble
from .communication_ensemble import CommunicationEnsemble

__all__ = [
    'BaseEnsembleModel',
    'AspirationalEnsemble',
    'CoolEnsemble', 
    'RelatableEnsemble',
    'CredibilityEnsemble',
    'StorytellingEnsemble',
    'CommunicationEnsemble'
]