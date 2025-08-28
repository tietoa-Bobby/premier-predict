#!/usr/bin/env python3
"""
Premier League 2025-26 Season Predictor

Advanced prediction system that combines machine learning and Elo ratings
to predict the 2025-26 Premier League season using historical data.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.enhanced_predictor import EnhancedPredictor
from src.config import HISTORICAL_SEASONS, TARGET_SEASON


def main() -> int:
    """Main function to run the enhanced predictor."""
    
    print("PREMIER LEAGUE 2025-26 SEASON PREDICTOR")
    print("=" * 60)
    print("Advanced ML + Elo Rating System")
    print("Predicts final table for 2025-26 season")
    print("Trains on historical Premier League data")
    print()
    
    # Check for data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("ERROR: Data directory not found.")
        print("\nTO GET STARTED:")
        print("1. Create a 'data' directory")
        print("2. Add historical CSV files: 2018-19.csv, 2019-20.csv, etc.")
        print("3. Re-run this script")
        return 1
    
    # Initialize enhanced predictor
    predictor = EnhancedPredictor()
    
    try:
        return run_prediction_analysis(predictor, str(data_dir))
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTROUBLESHOOTING:")
        print("  • Ensure CSV files are in correct format")
        print("  • Check team name spelling matches expected format")
        print("  • Verify Date,Team 1,FT,HT,Team 2 column structure")
        print("  • Make sure FT column has scores like '2-1', '0-0'")
        return 1


def run_prediction_analysis(predictor: EnhancedPredictor, data_dir: str) -> int:
    """Run the complete prediction analysis."""
    # Load data and train models
    print(f"\nLOADING HISTORICAL DATA FROM '{data_dir}' DIRECTORY")
    training_results = predictor.load_and_train(data_dir)
    
    print(f"\nTRAINING COMPLETED!")
    print(f"Loaded seasons: {training_results['loaded_seasons']}")
    print(f"ML Position Accuracy: {training_results['ml_metrics']['position_accuracy']:.1%}")
    print(f"Average Position Error: {training_results['ml_metrics']['position_mae']:.1f}")
    
    # Generate 2025-26 predictions
    print(f"\nGENERATING {TARGET_SEASON} PREDICTIONS...")
    predictions = predictor.predict_2025_26_season()
    
    # Print comprehensive results
    predictor.print_comprehensive_prediction(predictions)
    
    # Validate against historical data
    if len(training_results['loaded_seasons']) >= 2:
        print(f"\nVALIDATING MODEL ACCURACY...")
        validation = predictor.validate_historical_predictions()
    
    # Export results
    print(f"\nEXPORTING RESULTS...")
    predictor.export_predictions(predictions)
    
    print(f"\nANALYSIS COMPLETE!")
    print("=" * 60)
    print("Check 'outputs' directory for:")
    print("  • Complete Excel report with all predictions")
    print("  • Trained machine learning model")
    print("  • Feature importance analysis")
    
    print(f"\nMETHODOLOGY SUMMARY:")
    print("  • Random Forest ML model trained on historical data")
    print("  • Elo rating system for team strength assessment")
    print("  • Combined prediction using weighted averaging")
    print("  • Features from 3 previous seasons")
    print("  • Accounts for promoted teams with averaged stats")
    
    print(f"\nKEY INSIGHTS FOR {TARGET_SEASON}:")
    analysis = predictions['analysis']
    
    title_teams = analysis['title_contenders']['combined'][:2]
    print(f"  Title favorites: {', '.join(title_teams)}")
    
    rel_teams = analysis['relegation_candidates']['combined']
    print(f"  Relegation risks: {', '.join(rel_teams)}")
    
    # Show top feature
    top_feature = predictions['feature_importance'].iloc[0]
    print(f"  Most predictive factor: {top_feature['Feature']}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
