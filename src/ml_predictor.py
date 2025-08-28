"""
Machine Learning Predictor for Premier League

Uses Random Forest classifier to predict league positions based on
historical performance data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from src.historical_data_loader import HistoricalDataLoader
from src.config import (
    N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF, 
    RANDOM_STATE, PREMIER_LEAGUE_TEAMS_2025_26, FEATURE_SEASONS
)


class MLPredictor:
    """
    Machine Learning predictor for Premier League positions.
    """
    
    def __init__(self) -> None:
        """Initialize the ML predictor with default settings."""
        self.data_loader = HistoricalDataLoader()
        self.classifier: Optional[RandomForestClassifier] = None
        self.regressor: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        
    def load_historical_data(self, data_dir: str, seasons: List[str] = None) -> bool:
        """
        Load historical data for training.
        
        Args:
            data_dir: Directory containing season CSV files
            seasons: List of seasons to load
            
        Returns:
            True if data loaded successfully
        """
        print("Loading historical Premier League data...")
        results = self.data_loader.load_multiple_seasons(data_dir, seasons)
        
        successful_seasons = [season for season, success in results.items() if success]
        failed_seasons = [season for season, success in results.items() if not success]
        
        if failed_seasons:
            print(f"Warning: Failed to load seasons: {failed_seasons}")
        
        if len(successful_seasons) < 2:
            print("Error: Need at least 2 seasons for training")
            return False
        
        print(f"Successfully loaded {len(successful_seasons)} seasons: {successful_seasons}")
        return True
    
    def train_models(self, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train Random Forest models for position prediction.
        
        Args:
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary of model performance metrics
        """
        print("\nPreparing training dataset...")
        features_df, targets_df = self.data_loader.create_training_dataset(FEATURE_SEASONS)
        
        if len(features_df) == 0:
            raise ValueError("No training data available")
        
        print(f"Training dataset: {len(features_df)} samples, {len(features_df.columns)} features")
        
        # Store feature names
        self.feature_names = list(features_df.columns)
        
        # Prepare targets
        y_position = targets_df['target_position'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_position, test_size=test_size, random_state=RANDOM_STATE
        )
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train classification model (for position prediction)
        print("\nTraining Random Forest Classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Train regression model (for points prediction)
        print("Training Random Forest Regressor...")
        
        # Create points targets from positions (approximate)
        y_points_train = self._position_to_points(y_train)
        y_points_test = self._position_to_points(y_test)
        
        self.regressor = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE
        )
        
        self.regressor.fit(X_train, y_points_train)
        
        # Evaluate models
        metrics = self._evaluate_models(X_test, y_test, y_points_test)
        
        self.is_trained = True
        print("\nModel training completed!")
        
        return metrics
    
    def _position_to_points(self, positions: np.ndarray) -> np.ndarray:
        """
        Convert league positions to approximate points (for regression target).
        
        Args:
            positions: Array of league positions
            
        Returns:
            Array of approximate points
        """
        # Rough approximation: top teams get ~80-90 points, bottom teams get ~30-40
        # Linear interpolation between position and typical points
        points = 100 - (positions - 1) * 3.5
        return np.clip(points, 20, 100)  # Reasonable bounds
    
    def _evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                        y_points_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate trained models.
        
        Args:
            X_test: Test features
            y_test: Test position targets
            y_points_test: Test points targets
            
        Returns:
            Dictionary of metrics
        """
        # Classification metrics
        y_pred_class = self.classifier.predict(X_test)
        class_accuracy = accuracy_score(y_test, y_pred_class)
        position_mae = mean_absolute_error(y_test, y_pred_class)
        
        # Regression metrics
        y_pred_reg = self.regressor.predict(X_test)
        points_mae = mean_absolute_error(y_points_test, y_pred_reg)
        
        # Simple validation for small datasets (avoid stratified CV issues)
        try:
            cv_folds = min(3, len(X_test) // 2, len(np.unique(y_test)))
            if cv_folds < 2 or len(np.unique(y_test)) < cv_folds:
                # Use simple holdout validation instead of cross-validation
                cv_mean = class_accuracy
                cv_std = 0.0
            else:
                cv_scores = cross_val_score(self.classifier, X_test, y_test, cv=cv_folds)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
        except ValueError:
            # Fall back to simple accuracy if CV fails
            cv_mean = class_accuracy
            cv_std = 0.0
        
        metrics = {
            'position_accuracy': class_accuracy,
            'position_mae': position_mae,
            'points_mae': points_mae,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        print(f"\nModel Performance:")
        print(f"Position Accuracy: {class_accuracy:.3f}")
        print(f"Position MAE: {position_mae:.2f}")
        print(f"Points MAE: {points_mae:.1f}")
        print(f"CV Score: {cv_mean:.3f} ± {cv_std:.3f}")
        
        return metrics
    
    def predict_season(self, target_teams: List[str], 
                      recent_seasons: List[str]) -> pd.DataFrame:
        """
        Predict league table for a future season.
        
        Args:
            target_teams: List of teams in the target season
            recent_seasons: List of recent seasons to use as features
            
        Returns:
            Predicted league table DataFrame
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_models() first.")
        
        print(f"\nPredicting season for {len(target_teams)} teams...")
        print(f"Using features from seasons: {recent_seasons}")
        
        predictions = []
        
        for team in target_teams:
            # Get features for the team
            features = self.data_loader.get_prediction_features(team, recent_seasons)
            
            # Ensure all features are present
            feature_vector = []
            for feat_name in self.feature_names:
                feature_vector.append(features.get(feat_name, 0))
            
            # Scale features
            X = self.scaler.transform([feature_vector])
            
            # Make predictions
            predicted_position = self.classifier.predict(X)[0]
            predicted_points = self.regressor.predict(X)[0]
            
            # Get prediction probabilities for position
            position_probs = self.classifier.predict_proba(X)[0]
            
            predictions.append({
                'Team': team,
                'Predicted_Position': predicted_position,
                'Predicted_Points': round(predicted_points, 1),
                'Position_Confidence': max(position_probs)
            })
        
        # Create DataFrame and sort by predicted position
        df = pd.DataFrame(predictions)
        
        # Fix duplicate positions by ranking based on predicted points
        # Sort by predicted points (highest first) to assign proper positions
        df = df.sort_values('Predicted_Points', ascending=False)
        df = df.reset_index(drop=True)
        
        # Assign sequential positions 1-20
        df['Final_Position'] = df.index + 1
        
        # Update the predicted position to match the final ranking
        df['Predicted_Position'] = df['Final_Position']
        
        # Sort by final position for display
        df = df.sort_values('Final_Position')
        df = df.reset_index(drop=True)
        
        return df
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze feature importance from the trained model.
        
        Returns:
            DataFrame with feature importance rankings
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        importance = self.classifier.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        model_data = {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.classifier = model_data['classifier']
        self.regressor = model_data['regressor']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def compare_predictions_with_actual(self, season: str) -> pd.DataFrame:
        """
        Compare predictions with actual results for a completed season.
        
        Args:
            season: Season to compare
            
        Returns:
            DataFrame comparing predictions vs actual
        """
        if season not in self.data_loader.final_tables:
            raise ValueError(f"Season {season} not loaded")
        
        actual_table = self.data_loader.final_tables[season]
        
        # Get previous seasons for features
        available_seasons = sorted(self.data_loader.get_available_seasons())
        season_idx = available_seasons.index(season)
        
        if season_idx < FEATURE_SEASONS:
            raise ValueError(f"Not enough previous seasons for {season}")
        
        previous_seasons = available_seasons[season_idx-FEATURE_SEASONS:season_idx]
        
        # Make predictions
        teams = actual_table['Team'].tolist()
        predicted_table = self.predict_season(teams, previous_seasons)
        
        # Merge actual and predicted
        comparison = actual_table[['Team', 'Position', 'Points']].merge(
            predicted_table[['Team', 'Final_Position', 'Predicted_Points']],
            on='Team'
        )
        
        comparison['Position_Error'] = abs(comparison['Position'] - comparison['Final_Position'])
        comparison['Points_Error'] = abs(comparison['Points'] - comparison['Predicted_Points'])
        
        return comparison.sort_values('Position')
    
    def print_season_summary(self, season: str) -> None:
        """Print summary for a loaded season."""
        self.data_loader.print_season_summary(season)
