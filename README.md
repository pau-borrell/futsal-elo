# Futsal Match Prediction with Elo Ratings

Python implementation of an Elo-based prediction model for FCF Futsal Barcelona League.
Features:
- Goal-difference and home-advantage adjusted Elo.
- Logistic regression for data-driven draw probability.
- Backtesting with Brier Score and Log Loss.
- Optional bookmaker-style odds generation.

## Usage
pip install -r requirements.txt
python src/run.py --data data/FCF_Futsal_BCN_Gr10_2025_26_DB.xlsx --out out --commission 0.20

Outputs:
- fixture_predictions.csv with probabilities
- printed evaluation metrics

## Example
{'brier_home': 0.19, 'brier_draw': 0.24, 'brier_away': 0.21, 'logloss': 1.07}
{'home': 'Team A', 'away': 'Team B', 'probs': (0.45, 0.25, 0.30), 'odds': (1.85, 3.8, 2.9)}

## Requirements
Python ≥3.10
Dependencies in `requirements.txt`

## License
MIT License © 2025 Pau Borrell Bullich
