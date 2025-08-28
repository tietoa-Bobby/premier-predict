"""
Enhanced Premier League Predictor for 2025-26 Season

Combines historical data analysis, machine learning, and Elo ratings
to predict the 2025-26 Premier League season.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
from pathlib import Path

from src.ml_predictor import MLPredictor
from src.elo_system import EloSystem
from src.historical_data_loader import HistoricalDataLoader
from src.config import (
    PREMIER_LEAGUE_TEAMS_2025_26, HISTORICAL_SEASONS, 
    TARGET_SEASON, FEATURE_SEASONS
)


class EnhancedPredictor:
    """
    Enhanced predictor combining ML and Elo ratings for 2025-26 season prediction.
    """
    
    def __init__(self) -> None:
        """Initialize the enhanced predictor with all required components."""
        self.ml_predictor = MLPredictor()
        self.elo_system = EloSystem()
        self.data_loader = HistoricalDataLoader()
        self.is_trained = False
        
    def load_and_train(self, data_dir: str, seasons: Optional[List[str]] = None) -> Dict:
        """
        Load historical data and train all models.
        
        Args:
            data_dir: Directory containing season CSV files
            seasons: List of seasons to load (default: all historical seasons)
            
        Returns:
            Dictionary with training results and metrics
            
        Raises:
            ValueError: If data loading or training fails
        """
        if seasons is None:
            seasons = HISTORICAL_SEASONS
            
        print("ENHANCED PREMIER LEAGUE PREDICTOR")
        print("=" * 50)
        print(f"Target Season: {TARGET_SEASON}")
        print(f"Training on seasons: {seasons}")
        print(f"Teams for {TARGET_SEASON}: {len(PREMIER_LEAGUE_TEAMS_2025_26)}")
        
        # Load historical data with error handling
        try:
            success = self.ml_predictor.load_historical_data(data_dir, seasons)
            if not success:
                raise ValueError("Failed to load historical data - check file formats and paths")
        except Exception as e:
            raise ValueError(f"Error loading historical data: {str(e)}")
        
        # Train machine learning models
        print("\n🤖 TRAINING MACHINE LEARNING MODELS")
        print("-" * 40)
        try:
            ml_metrics = self.ml_predictor.train_models()
        except Exception as e:
            raise ValueError(f"Error training ML models: {str(e)}")
        
        # Initialize Elo ratings from most recent season
        self._initialize_elo_from_recent_season()
        
        self.is_trained = True
        
        results = {
            'ml_metrics': ml_metrics,
            'loaded_seasons': self.ml_predictor.data_loader.get_available_seasons(),
            'feature_importance': self.ml_predictor.analyze_feature_importance()
        }
        
        return results
    
    def _initialize_elo_from_recent_season(self):
        """Initialize Elo ratings based on most recent season performance."""
        available_seasons = sorted(self.ml_predictor.data_loader.get_available_seasons())
        if not available_seasons:
            return
        
        recent_season = available_seasons[-1]
        recent_table = self.ml_predictor.data_loader.final_tables[recent_season]
        
        print(f"\nINITIALIZING ELO RATINGS FROM {recent_season}")
        print("-" * 40)
        
        # Initialize teams with ratings based on final position
        for _, row in recent_table.iterrows():
            team = row['Team']
            position = row['Position']
            
            # Calculate initial Elo rating based on position
            # Top teams get higher ratings, relegated teams get lower
            if position <= 4:  # Top 4
                initial_rating = 1700 + (5 - position) * 50
            elif position <= 10:  # Mid-table
                initial_rating = 1500 + (11 - position) * 20
            else:  # Bottom half
                initial_rating = 1300 + (21 - position) * 15
            
            self.elo_system.initialize_team(team, initial_rating)
        
        # Initialize new teams (promoted) with average relegated team rating
        relegated_teams_rating = np.mean([
            self.elo_system.get_rating(team) 
            for team in recent_table.tail(3)['Team']
            if team in self.elo_system.ratings
        ])
        
        for team in PREMIER_LEAGUE_TEAMS_2025_26:
            if team not in self.elo_system.ratings:
                self.elo_system.initialize_team(team, relegated_teams_rating)
                print(f"  {team}: {relegated_teams_rating:.0f} (promoted)")
    
    def predict_2025_26_season(self) -> Dict:
        """
        Generate comprehensive predictions for the 2025-26 season.
        
        Returns:
            Dictionary containing all predictions and analysis
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call load_and_train() first.")
        
        print(f"\n🔮 PREDICTING {TARGET_SEASON} SEASON")
        print("=" * 50)
        
        # Get recent seasons for ML features
        available_seasons = sorted(self.ml_predictor.data_loader.get_available_seasons())
        recent_seasons = available_seasons[-FEATURE_SEASONS:]
        
        # ML Prediction
        print("\n🤖 Machine Learning Prediction:")
        ml_prediction = self.ml_predictor.predict_season(
            PREMIER_LEAGUE_TEAMS_2025_26, recent_seasons
        )
        
        # Elo-based prediction
        print("\nElo Rating Prediction:")
        elo_prediction = self._predict_with_elo()
        
        # Combined prediction
        print("\nCombined Prediction:")
        combined_prediction = self._combine_predictions(ml_prediction, elo_prediction)
        
        # Analysis
        analysis = self._analyze_predictions(ml_prediction, elo_prediction, combined_prediction)
        
        return {
            'ml_prediction': ml_prediction,
            'elo_prediction': elo_prediction,
            'combined_prediction': combined_prediction,
            'analysis': analysis,
            'feature_importance': self.ml_predictor.analyze_feature_importance(),
            'elo_ratings': {team: self.elo_system.get_rating(team) 
                           for team in PREMIER_LEAGUE_TEAMS_2025_26}
        }
    
    def _predict_with_elo(self) -> pd.DataFrame:
        """
        Create prediction based on current Elo ratings.
        
        Returns:
            DataFrame with Elo-based predictions
        """
        elo_predictions = []
        
        for team in PREMIER_LEAGUE_TEAMS_2025_26:
            rating = self.elo_system.get_rating(team)
            
            # Convert Elo rating to approximate position and points
            # Higher rating = better position (lower number)
            all_ratings = [self.elo_system.get_rating(t) for t in PREMIER_LEAGUE_TEAMS_2025_26]
            position = sorted(all_ratings, reverse=True).index(rating) + 1
            
            # Estimate points based on Elo rating
            # Top teams (~1800+): 75-90 points
            # Mid teams (~1500): 45-60 points  
            # Bottom teams (~1300): 25-40 points
            if rating >= 1700:
                estimated_points = 75 + (rating - 1700) / 10
            elif rating >= 1500:
                estimated_points = 45 + (rating - 1500) / 10
            else:
                estimated_points = 25 + (rating - 1300) / 10
            
            elo_predictions.append({
                'Team': team,
                'Elo_Rating': rating,
                'Predicted_Position': position,
                'Predicted_Points': round(estimated_points, 1)
            })
        
        df = pd.DataFrame(elo_predictions)
        df = df.sort_values('Elo_Rating', ascending=False)
        df = df.reset_index(drop=True)
        df['Final_Position'] = df.index + 1
        
        return df
    
    def _combine_predictions(self, ml_pred: pd.DataFrame, 
                           elo_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Combine ML and Elo predictions using weighted average.
        
        Args:
            ml_pred: ML prediction DataFrame
            elo_pred: Elo prediction DataFrame
            
        Returns:
            Combined prediction DataFrame
        """
        # Merge predictions
        ml_clean = ml_pred[['Team', 'Predicted_Position', 'Predicted_Points']].copy()
        ml_clean.columns = ['Team', 'ML_Position', 'ML_Points']
        
        elo_clean = elo_pred[['Team', 'Final_Position', 'Predicted_Points', 'Elo_Rating']].copy()
        elo_clean.columns = ['Team', 'Elo_Position', 'Elo_Points', 'Elo_Rating']
        
        combined = ml_clean.merge(elo_clean, on='Team')
        
        # Weighted combination (60% ML, 40% Elo)
        ml_weight = 0.6
        elo_weight = 0.4
        
        combined['Combined_Position'] = (
            combined['ML_Position'] * ml_weight + 
            combined['Elo_Position'] * elo_weight
        )
        
        combined['Combined_Points'] = (
            combined['ML_Points'] * ml_weight + 
            combined['Elo_Points'] * elo_weight
        )
        
        # Sort by combined position
        combined = combined.sort_values('Combined_Position')
        combined = combined.reset_index(drop=True)
        combined['Final_Position'] = combined.index + 1
        
        # Round values
        combined['Combined_Position'] = combined['Combined_Position'].round(1)
        combined['Combined_Points'] = combined['Combined_Points'].round(1)
        
        return combined
    
    def _analyze_predictions(self, ml_pred: pd.DataFrame, elo_pred: pd.DataFrame, 
                           combined_pred: pd.DataFrame) -> Dict:
        """
        Analyze and compare different predictions.
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Title contenders (top 4)
        ml_top4 = set(ml_pred.head(4)['Team'])
        elo_top4 = set(elo_pred.head(4)['Team']) 
        combined_top4 = set(combined_pred.head(4)['Team'])
        
        analysis['title_contenders'] = {
            'ml': list(ml_top4),
            'elo': list(elo_top4),
            'combined': list(combined_top4),
            'consensus': list(ml_top4 & elo_top4)
        }
        
        # Relegation candidates (bottom 3)
        ml_bottom3 = set(ml_pred.tail(3)['Team'])
        elo_bottom3 = set(elo_pred.tail(3)['Team'])
        combined_bottom3 = set(combined_pred.tail(3)['Team'])
        
        analysis['relegation_candidates'] = {
            'ml': list(ml_bottom3),
            'elo': list(elo_bottom3),
            'combined': list(combined_bottom3),
            'consensus': list(ml_bottom3 & elo_bottom3)
        }
        
        # Promoted teams performance
        promoted_teams = ['Leeds United', 'Burnley', 'Sunderland']
        analysis['promoted_teams'] = {}
        
        for team in promoted_teams:
            ml_pos = ml_pred[ml_pred['Team'] == team]['Final_Position'].iloc[0]
            elo_pos = elo_pred[elo_pred['Team'] == team]['Final_Position'].iloc[0]
            combined_pos = combined_pred[combined_pred['Team'] == team]['Final_Position'].iloc[0]
            
            analysis['promoted_teams'][team] = {
                'ml_position': ml_pos,
                'elo_position': elo_pos,
                'combined_position': combined_pos
            }
        
        # Biggest disagreements
        position_diffs = []
        for _, row in combined_pred.iterrows():
            team = row['Team']
            ml_pos = ml_pred[ml_pred['Team'] == team]['Final_Position'].iloc[0]
            elo_pos = elo_pred[elo_pred['Team'] == team]['Final_Position'].iloc[0]
            diff = abs(ml_pos - elo_pos)
            position_diffs.append({'team': team, 'difference': diff})
        
        position_diffs.sort(key=lambda x: x['difference'], reverse=True)
        analysis['biggest_disagreements'] = position_diffs[:5]
        
        return analysis
    
    def print_comprehensive_prediction(self, results: Dict) -> None:
        """
        Print comprehensive prediction results.
        
        Args:
            results: Results from predict_2025_26_season()
        """
        print(f"\nPREMIER LEAGUE {TARGET_SEASON} PREDICTION")
        print("=" * 60)
        
        combined = results['combined_prediction']
        
        print(f"\nPREDICTED FINAL TABLE:")
        print("-" * 60)
        for _, row in combined.iterrows():
            pos = int(row['Final_Position'])
            team = row['Team']
            points = row['Combined_Points']
            
            # Add indicators for different positions
            if pos <= 4:
                indicator = "[CL]"  # Champions League
            elif pos <= 7:
                indicator = "[EU]"  # Europa League
            elif pos >= 18:
                indicator = "[REL]" # Relegation
            else:
                indicator = "     "
            
            print(f"{pos:2d}. {team:<20} {points:>5.1f} pts {indicator}")
        
        # Analysis
        analysis = results['analysis']
        
        print(f"\nTITLE CONTENDERS:")
        for team in analysis['title_contenders']['combined']:
            pos = combined[combined['Team'] == team]['Final_Position'].iloc[0]
            print(f"  {pos}. {team}")
        
        print(f"\nRELEGATION BATTLE:")
        for team in analysis['relegation_candidates']['combined']:
            pos = combined[combined['Team'] == team]['Final_Position'].iloc[0]
            print(f"  {pos}. {team}")
        
        print(f"\nPROMOTED TEAMS OUTLOOK:")
        for team, positions in analysis['promoted_teams'].items():
            pos = positions['combined_position']
            if pos >= 18:
                outlook = "Relegation fight"
            elif pos >= 15:
                outlook = "Survival battle"
            elif pos >= 10:
                outlook = "Mid-table safety"
            else:
                outlook = "Exceeding expectations"
            
            print(f"  {pos:2.0f}. {team:<15} - {outlook}")
        
        print(f"\nMODEL INSIGHTS:")
        importance = results['feature_importance'].head()
        print("  Top predictive factors:")
        for _, row in importance.iterrows():
            print(f"    • {row['Feature']}: {row['Importance']:.3f}")
    
    def validate_historical_predictions(self, validation_season: str = None) -> Dict:
        """
        Validate model accuracy against a historical season.
        
        Args:
            validation_season: Season to validate against
            
        Returns:
            Validation results
        """
        available_seasons = self.ml_predictor.data_loader.get_available_seasons()
        
        if validation_season is None:
            # Use second most recent season for validation
            validation_season = sorted(available_seasons)[-2]
        
        if validation_season not in available_seasons:
            raise ValueError(f"Season {validation_season} not available")
        
        print(f"\nVALIDATING AGAINST {validation_season}")
        print("-" * 40)
        
        comparison = self.ml_predictor.compare_predictions_with_actual(validation_season)
        
        print(f"\nPrediction vs Actual for {validation_season}:")
        print("Team                 Actual  Predicted  Error")
        print("-" * 50)
        
        for _, row in comparison.iterrows():
            print(f"{row['Team']:<20} {row['Position']:>6} {row['Final_Position']:>9} "
                  f"{row['Position_Error']:>6.0f}")
        
        avg_error = comparison['Position_Error'].mean()
        print(f"\nAverage Position Error: {avg_error:.1f} positions")
        
        return comparison
    
    def export_predictions(self, results: Dict, output_dir: str = "outputs") -> None:
        """
        Export prediction results to files.
        
        Args:
            results: Results from predict_2025_26_season()
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to Excel
        excel_path = f"{output_dir}/premier_league_2025_26_predictions.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            results['combined_prediction'].to_excel(
                writer, sheet_name='Combined_Prediction', index=False
            )
            results['ml_prediction'].to_excel(
                writer, sheet_name='ML_Prediction', index=False
            )
            results['elo_prediction'].to_excel(
                writer, sheet_name='Elo_Prediction', index=False
            )
            results['feature_importance'].to_excel(
                writer, sheet_name='Feature_Importance', index=False
            )
        
        print(f"\nResults exported to {excel_path}")
        
        # Save model
        model_path = f"{output_dir}/trained_model_2025_26.joblib"
        self.ml_predictor.save_model(model_path)
        print(f"Model saved to {model_path}")
