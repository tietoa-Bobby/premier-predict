"""
Historical Data Loader for Premier League Prediction

Loads and processes multiple seasons of Premier League data for training
machine learning models to predict future seasons.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
from src.config import HISTORICAL_SEASONS, PREMIER_LEAGUE_TEAMS_2025_26, POINTS_WIN, POINTS_DRAW


class HistoricalDataLoader:
    """
    Loads and processes historical Premier League seasons for prediction modeling.
    """
    
    def __init__(self) -> None:
        """Initialize the historical data loader with empty containers."""
        self.seasons_data: Dict[str, pd.DataFrame] = {}  # Dict[season, DataFrame]
        self.team_statistics: Dict[str, Dict[str, Dict]] = {}  # Dict[season, Dict[team, stats]]
        self.final_tables: Dict[str, pd.DataFrame] = {}  # Dict[season, DataFrame]
        
    def load_season_data(self, filepath: str, season: str) -> bool:
        """
        Load data for a single season.
        
        Args:
            filepath: Path to CSV file (e.g., 'data/2023-24.csv')
            season: Season identifier (e.g., '2023-24')
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load CSV data
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['Date', 'Team 1', 'FT', 'Team 2']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Missing required columns in {filepath}")
                return False
            
            # Clean and standardize team names
            df['Team 1'] = df['Team 1'].apply(self._standardize_team_name)
            df['Team 2'] = df['Team 2'].apply(self._standardize_team_name)
            
            # Parse scores
            df[['Home_Goals', 'Away_Goals']] = df['FT'].apply(self._parse_score).tolist()
            
            # Filter out invalid scores
            df = df.dropna(subset=['Home_Goals', 'Away_Goals'])
            
            # Store season data
            self.seasons_data[season] = df
            
            # Calculate season statistics
            self._calculate_season_statistics(season)
            
            print(f"Loaded {len(df)} matches for season {season}")
            return True
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return False
    
    def load_multiple_seasons(self, data_dir: str, seasons: List[str] = None) -> Dict[str, bool]:
        """
        Load multiple seasons from a directory.
        
        Args:
            data_dir: Directory containing season CSV files
            seasons: List of seasons to load (default: all historical seasons)
            
        Returns:
            Dictionary mapping season to success status
        """
        if seasons is None:
            seasons = HISTORICAL_SEASONS
            
        results = {}
        data_path = Path(data_dir)
        
        for season in seasons:
            # Try different filename patterns
            possible_files = [
                f"{season}.csv",
                f"premier_league_{season}.csv"
            ]
            
            loaded = False
            for filename in possible_files:
                filepath = data_path / filename
                if filepath.exists():
                    results[season] = self.load_season_data(str(filepath), season)
                    loaded = True
                    break
            
            if not loaded:
                print(f"Warning: Could not find data file for season {season}")
                results[season] = False
        
        return results
    
    def _standardize_team_name(self, team_name: str) -> str:
        """
        Standardize team names to match current naming conventions.
        
        Args:
            team_name: Raw team name from CSV
            
        Returns:
            Standardized team name
        """
        # Team name mappings for consistency
        name_mappings = {
            'Man City': 'Manchester City',
            'Man United': 'Manchester United',
            'Tottenham': 'Tottenham',
            'Spurs': 'Tottenham',
            'Brighton': 'Brighton',
            'Brighton & Hove Albion': 'Brighton',
            'Newcastle': 'Newcastle',
            'Newcastle United': 'Newcastle',
            'Nottm Forest': 'Nottingham Forest',
            'Sheffield United': 'Sheffield United',
            'Luton': 'Luton Town',
            'Luton Town': 'Luton Town',
            'West Ham': 'West Ham',
            'Crystal Palace': 'Crystal Palace',
            'Aston Villa': 'Aston Villa',
            'Wolves': 'Wolves',
            'Wolverhampton': 'Wolves',
            'Leicester': 'Leicester City',
            'Leeds': 'Leeds United',
            'Leeds United': 'Leeds United',
            'Burnley': 'Burnley',
            'Sunderland': 'Sunderland'
        }
        
        return name_mappings.get(team_name, team_name)
    
    def _parse_score(self, score_str: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse a score string like '2-1' into home and away goals.
        
        Args:
            score_str: Score string (e.g., '2-1', '0-0')
            
        Returns:
            Tuple of (home_goals, away_goals) or (None, None) if invalid
        """
        try:
            if pd.isna(score_str) or score_str == '':
                return None, None
                
            # Handle different score formats
            score_str = str(score_str).strip()
            
            # Match patterns like '2-1', '0-0', etc.
            match = re.match(r'(\d+)-(\d+)', score_str)
            if match:
                return int(match.group(1)), int(match.group(2))
            
            return None, None
            
        except Exception:
            return None, None
    
    def _calculate_season_statistics(self, season: str) -> None:
        """
        Calculate comprehensive statistics for all teams in a season.
        
        Args:
            season: Season identifier
        """
        df = self.seasons_data[season]
        teams = list(set(df['Team 1'].tolist() + df['Team 2'].tolist()))
        
        stats = {}
        
        for team in teams:
            # Home matches
            home_matches = df[df['Team 1'] == team]
            # Away matches  
            away_matches = df[df['Team 2'] == team]
            
            # Initialize counters
            wins = draws = losses = 0
            goals_for = goals_against = 0
            home_wins = away_wins = 0
            
            # Process home matches
            for _, match in home_matches.iterrows():
                home_goals = match['Home_Goals']
                away_goals = match['Away_Goals']
                
                goals_for += home_goals
                goals_against += away_goals
                
                if home_goals > away_goals:
                    wins += 1
                    home_wins += 1
                elif home_goals == away_goals:
                    draws += 1
                else:
                    losses += 1
            
            # Process away matches
            for _, match in away_matches.iterrows():
                home_goals = match['Home_Goals']
                away_goals = match['Away_Goals']
                
                goals_for += away_goals
                goals_against += home_goals
                
                if away_goals > home_goals:
                    wins += 1
                    away_wins += 1
                elif away_goals == home_goals:
                    draws += 1
                else:
                    losses += 1
            
            # Calculate derived statistics
            games_played = wins + draws + losses
            points = wins * POINTS_WIN + draws * POINTS_DRAW
            goal_difference = goals_for - goals_against
            
            # Rates and averages
            if games_played > 0:
                win_rate = wins / games_played
                points_per_game = points / games_played
                goals_per_game = goals_for / games_played
                goals_against_per_game = goals_against / games_played
            else:
                win_rate = points_per_game = goals_per_game = goals_against_per_game = 0
            
            stats[team] = {
                'games_played': games_played,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_difference': goal_difference,
                'points': points,
                'win_rate': win_rate,
                'points_per_game': points_per_game,
                'goals_per_game': goals_per_game,
                'goals_against_per_game': goals_against_per_game,
                'home_wins': home_wins,
                'away_wins': away_wins,
                'home_win_rate': home_wins / max(1, len(home_matches)),
                'away_win_rate': away_wins / max(1, len(away_matches))
            }
        
        self.team_statistics[season] = stats
        
        # Create final table
        self._create_final_table(season)
    
    def _create_final_table(self, season: str) -> None:
        """
        Create the final league table for a season.
        
        Args:
            season: Season identifier
        """
        stats = self.team_statistics[season]
        
        # Convert to DataFrame
        table_data = []
        for team, team_stats in stats.items():
            table_data.append({
                'Team': team,
                'Played': team_stats['games_played'],
                'Won': team_stats['wins'],
                'Drawn': team_stats['draws'],
                'Lost': team_stats['losses'],
                'Goals_For': team_stats['goals_for'],
                'Goals_Against': team_stats['goals_against'],
                'Goal_Difference': team_stats['goal_difference'],
                'Points': team_stats['points']
            })
        
        df = pd.DataFrame(table_data)
        
        # Sort by points, then goal difference, then goals for
        df = df.sort_values(['Points', 'Goal_Difference', 'Goals_For'], 
                           ascending=[False, False, False])
        df = df.reset_index(drop=True)
        df['Position'] = df.index + 1
        
        self.final_tables[season] = df
    
    def get_team_features(self, team: str, season: str) -> Dict[str, float]:
        """
        Get feature vector for a team in a specific season.
        
        Args:
            team: Team name
            season: Season identifier
            
        Returns:
            Dictionary of features for machine learning
        """
        if season not in self.team_statistics:
            return {}
        
        if team not in self.team_statistics[season]:
            # For teams not in the season, return average of bottom 3 teams
            return self._get_average_bottom_three_features(season)
        
        stats = self.team_statistics[season][team]
        
        # Feature engineering
        features = {
            'points_per_game': stats['points_per_game'],
            'win_rate': stats['win_rate'],
            'goals_per_game': stats['goals_per_game'],
            'goals_against_per_game': stats['goals_against_per_game'],
            'goal_difference_per_game': stats['goal_difference'] / max(1, stats['games_played']),
            'home_win_rate': stats['home_win_rate'],
            'away_win_rate': stats['away_win_rate'],
            'points': stats['points'],
            'goal_difference': stats['goal_difference']
        }
        
        return features
    
    def _get_average_bottom_three_features(self, season: str) -> Dict[str, float]:
        """
        Get average features of bottom 3 teams for promoted teams.
        
        Args:
            season: Season identifier
            
        Returns:
            Dictionary of average features
        """
        if season not in self.final_tables:
            return {}
        
        table = self.final_tables[season]
        bottom_three = table.tail(3)
        
        avg_features = {}
        for team in bottom_three['Team']:
            features = self.get_team_features(team, season)
            for key, value in features.items():
                if key not in avg_features:
                    avg_features[key] = []
                avg_features[key].append(value)
        
        # Calculate averages
        return {key: np.mean(values) for key, values in avg_features.items()}
    
    def create_training_dataset(self, feature_seasons: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create training dataset where season n features predict season n+1 position.
        
        Args:
            feature_seasons: Number of previous seasons to use as features
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        training_data = []
        
        seasons = sorted(self.seasons_data.keys())
        
        for i in range(feature_seasons, len(seasons)):
            target_season = seasons[i]
            feature_season_list = seasons[i-feature_seasons:i]
            
            if target_season not in self.final_tables:
                continue
            
            target_table = self.final_tables[target_season]
            
            for _, row in target_table.iterrows():
                team = row['Team']
                target_position = row['Position']
                
                # Collect features from previous seasons
                team_features = {'team': team, 'target_season': target_season}
                
                for j, feat_season in enumerate(feature_season_list):
                    season_features = self.get_team_features(team, feat_season)
                    
                    # Add season suffix to feature names
                    for feat_name, feat_value in season_features.items():
                        team_features[f"{feat_name}_season_{j+1}"] = feat_value
                
                team_features['target_position'] = target_position
                training_data.append(team_features)
        
        df = pd.DataFrame(training_data)
        
        # Separate features and targets
        feature_cols = [col for col in df.columns if col not in ['team', 'target_season', 'target_position']]
        features_df = df[feature_cols].fillna(0)  # Fill NaN with 0 for missing data
        targets_df = df[['team', 'target_season', 'target_position']]
        
        return features_df, targets_df
    
    def get_prediction_features(self, team: str, recent_seasons: List[str]) -> Dict[str, float]:
        """
        Get features for a team to make predictions on future season.
        
        Args:
            team: Team name
            recent_seasons: List of recent seasons to use as features
            
        Returns:
            Dictionary of features
        """
        team_features = {}
        
        for j, season in enumerate(recent_seasons):
            season_features = self.get_team_features(team, season)
            
            # Add season suffix to feature names
            for feat_name, feat_value in season_features.items():
                team_features[f"{feat_name}_season_{j+1}"] = feat_value
        
        return team_features
    
    def print_season_summary(self, season: str) -> None:
        """
        Print summary of a loaded season.
        
        Args:
            season: Season identifier
        """
        if season not in self.final_tables:
            print(f"Season {season} not loaded")
            return
        
        table = self.final_tables[season]
        print(f"\n{season} FINAL TABLE:")
        print("=" * 60)
        
        for _, row in table.iterrows():
            print(f"{row['Position']:2d}. {row['Team']:<20} {row['Points']:>3d} pts "
                  f"({row['Won']:>2d}W {row['Drawn']:>2d}D {row['Lost']:>2d}L) "
                  f"GD: {row['Goal_Difference']:+3d}")
    
    def get_available_seasons(self) -> List[str]:
        """Get list of successfully loaded seasons."""
        return sorted(self.seasons_data.keys())
