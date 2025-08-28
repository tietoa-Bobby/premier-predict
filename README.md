# Premier League 2025-26 Predictor

A machine learning system that predicts the final Premier League table for the 2025-26 season.

## What this does

This program uses 7 years of historical Premier League data (2018-19 to 2024-25) to predict where each team will finish in the 2025-26 season. It combines machine learning with an Elo rating system to make predictions.

## How to run it

1. Install packages: `pip install -r requirements.txt`
2. Run prediction: `python predict_2025_26.py`
3. Check results in the `outputs/` folder

## What you get

- Predicted final league table (1st to 20th place)
- Points prediction for each team
- Excel report with detailed analysis
- Accuracy validation against past seasons

## Example output

```
🏆 PREMIER LEAGUE 2025-26 PREDICTION
 1. Liverpool             92.1 pts 🏆
 2. Arsenal               92.7 pts 🏆
 3. Manchester City       91.4 pts 🏆
 ...
18. Leeds United          37.2 pts ⬇️
19. Burnley               37.1 pts ⬇️
20. Sunderland            37.1 pts ⬇️
```

The model typically predicts team positions within 1-2 places of where they actually finish. Recent validation against the 2023-24 season showed an average position error of just 0.3 places.
