"""
Premier League 2025-26 Prediction System

Advanced machine learning system that combines Random Forest algorithms
with Elo ratings to predict the 2025-26 Premier League season.

Core Components:
- EnhancedPredictor: Main ML+Elo hybrid prediction system
- MLPredictor: Random Forest models for position and points prediction
- HistoricalDataLoader: Processing historical Premier League data
- EloSystem: Football-specific Elo rating calculations
"""

__version__ = "2.0.0"
__author__ = "Premier League Predictor Team"

# Import core classes when needed to avoid circular dependencies
# Usage: from src.enhanced_predictor import EnhancedPredictor

__all__ = [
    "EnhancedPredictor",
    "MLPredictor", 
    "HistoricalDataLoader",
    "EloSystem"
]