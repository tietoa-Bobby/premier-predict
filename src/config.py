"""
Configuration settings for the Premier League predictor.

This module contains all configuration parameters for the Premier League
prediction system including team lists, model parameters, and system settings.
"""

from typing import List

# ================================
# ELO SYSTEM PARAMETERS
# ================================
BASE_ELO_RATING: int = 1500  # Starting rating for all teams
K_FACTOR: int = 20  # How much ratings change after each match (higher = more volatile)
HOME_ADVANTAGE: int = 100  # Home team gets this many extra Elo points
GOAL_DIFFERENCE_WEIGHT: int = 5  # Multiplier for goal difference impact on rating changes

# ================================
# MACHINE LEARNING PARAMETERS  
# ================================
# Random Forest Classifier Settings
N_ESTIMATORS: int = 150  # Number of trees in random forest
MAX_DEPTH: int = 10      # Maximum depth of trees (prevents overfitting)
MIN_SAMPLES_SPLIT: int = 5  # Minimum samples required to split an internal node
MIN_SAMPLES_LEAF: int = 2   # Minimum samples required to be at a leaf node
RANDOM_STATE: int = 42      # For reproducible results

# Feature Engineering Settings
FEATURE_SEASONS: int = 3  # Number of previous seasons to use as features

# ================================
# SIMULATION PARAMETERS
# ================================
NUM_SIMULATIONS: int = 10000  # Number of Monte Carlo simulations for predictions
RANDOM_SEED: int = 42         # Seed for random number generation

# ================================
# PREMIER LEAGUE TEAMS (2025-26)
# ================================
# 17 teams that stayed up from 2024-25 + 3 promoted teams
PREMIER_LEAGUE_TEAMS_2025_26: List[str] = [
    # Teams that stayed up from 2024-25
    'Arsenal', 'Aston Villa', 'Brighton', 'Chelsea', 'Crystal Palace',
    'Everton', 'Fulham', 'Liverpool', 'Manchester City', 
    'Manchester United', 'Newcastle', 'Nottingham Forest',
    'Tottenham', 'West Ham', 'Wolves', 'Bournemouth', 'Brentford',
    
    # Promoted teams for 2025-26 (Championship winners + playoff winners)
    'Leeds United', 'Burnley', 'Sunderland'
]

# ================================
# HISTORICAL DATA CONFIGURATION
# ================================
# Historical seasons available for training (most recent 7 seasons)
HISTORICAL_SEASONS: List[str] = [
    '2018-19', '2019-20', '2020-21', '2021-22', 
    '2022-23', '2023-24', '2024-25'
]

# Season we want to predict
TARGET_SEASON: str = '2025-26'

# ================================
# POINTS SYSTEM
# ================================
POINTS_WIN: int = 3    # Points awarded for a win
POINTS_DRAW: int = 1   # Points awarded for a draw  
POINTS_LOSS: int = 0   # Points awarded for a loss

# ================================
# DISPLAY AND OUTPUT SETTINGS
# ================================
CHART_STYLE: str = 'seaborn-v0_8'  # Matplotlib style for charts
FIGURE_SIZE: tuple = (12, 8)       # Default figure size for plots
DPI: int = 300                     # Resolution for saved figures

# File naming patterns
CSV_FILENAME_PATTERN: str = "{season}.csv"  # e.g., "2023-24.csv"
OUTPUT_DIR: str = "outputs"                 # Directory for prediction outputs
