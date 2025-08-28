"""
Elo Rating System for Football

This module implements a customized Elo rating system specifically designed for football,
including home advantage, goal difference weighting, and match outcome probabilities.
"""

import math
from typing import Dict, Tuple, Optional
from src.config import BASE_ELO_RATING, K_FACTOR, HOME_ADVANTAGE, GOAL_DIFFERENCE_WEIGHT


class EloSystem:
    """
    Elo rating system for football matches.
    
    Features:
    - Home advantage adjustment
    - Goal difference weighting
    - Draw probability calculation
    - Rating updates based on match results
    """
    
    def __init__(self, base_rating: int = BASE_ELO_RATING, k_factor: int = K_FACTOR,
                 home_advantage: int = HOME_ADVANTAGE, gd_weight: int = GOAL_DIFFERENCE_WEIGHT) -> None:
        """
        Initialize the Elo system with configurable parameters.
        
        Args:
            base_rating: Starting rating for all teams (default: 1500)
            k_factor: Maximum rating change per match (default: 20)
            home_advantage: Rating boost for home team (default: 100)
            gd_weight: Goal difference impact multiplier (default: 5)
        """
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.gd_weight = gd_weight
        self.ratings: Dict[str, float] = {}
    
    def initialize_team(self, team: str, rating: Optional[float] = None) -> None:
        """Initialize a team with a starting rating."""
        if rating is None:
            rating = self.base_rating
        self.ratings[team] = rating
    
    def get_rating(self, team: str) -> float:
        """Get current rating for a team."""
        if team not in self.ratings:
            self.initialize_team(team)
        return self.ratings[team]
    
    def calculate_expected_score(self, home_rating: float, away_rating: float) -> Tuple[float, float]:
        """
        Calculate expected scores for both teams.
        
        Args:
            home_rating: Home team's Elo rating
            away_rating: Away team's Elo rating
            
        Returns:
            Tuple of (home_expected, away_expected) scores
        """
        # Apply home advantage
        adjusted_home_rating = home_rating + self.home_advantage
        
        # Calculate expected scores using Elo formula
        rating_diff = adjusted_home_rating - away_rating
        home_expected = 1 / (1 + 10 ** (-rating_diff / 400))
        away_expected = 1 - home_expected
        
        return home_expected, away_expected
    
    def calculate_match_probabilities(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Calculate win/draw/loss probabilities for a match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dictionary with 'home_win', 'draw', 'away_win' probabilities
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        home_expected, away_expected = self.calculate_expected_score(home_rating, away_rating)
        
        # Convert to win/draw/loss probabilities
        # This is a simplified model - more sophisticated approaches exist
        rating_diff = (home_rating + self.home_advantage) - away_rating
        
        # Use logistic function to convert rating difference to probabilities
        if rating_diff > 0:
            home_win_prob = 0.5 + (rating_diff / 800)
        else:
            home_win_prob = 0.5 * (1 + rating_diff / 400)
        
        # Clamp probabilities
        home_win_prob = max(0.1, min(0.8, home_win_prob))
        away_win_prob = max(0.1, min(0.8, 1 - home_win_prob - 0.25))
        draw_prob = 1 - home_win_prob - away_win_prob
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob
        }
    
    def update_ratings(self, home_team: str, away_team: str, 
                      home_goals: int, away_goals: int) -> Tuple[float, float]:
        """
        Update Elo ratings after a match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            
        Returns:
            Tuple of (new_home_rating, new_away_rating)
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Calculate expected scores
        home_expected, away_expected = self.calculate_expected_score(home_rating, away_rating)
        
        # Determine actual scores (1 for win, 0.5 for draw, 0 for loss)
        if home_goals > away_goals:
            home_actual, away_actual = 1.0, 0.0
        elif home_goals < away_goals:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5
        
        # Calculate goal difference multiplier
        goal_diff = abs(home_goals - away_goals)
        if goal_diff <= 1:
            multiplier = 1.0
        elif goal_diff == 2:
            multiplier = 1.5
        else:
            multiplier = (11 + goal_diff) / 8
        
        # Update ratings
        home_rating_change = self.k_factor * multiplier * (home_actual - home_expected)
        away_rating_change = self.k_factor * multiplier * (away_actual - away_expected)
        
        new_home_rating = home_rating + home_rating_change
        new_away_rating = away_rating + away_rating_change
        
        # Update stored ratings
        self.ratings[home_team] = new_home_rating
        self.ratings[away_team] = new_away_rating
        
        return new_home_rating, new_away_rating
    
    def get_all_ratings(self) -> Dict[str, float]:
        """Get all current team ratings."""
        return self.ratings.copy()
    
    def reset_ratings(self) -> None:
        """Reset all ratings to base rating."""
        for team in self.ratings:
            self.ratings[team] = self.base_rating
