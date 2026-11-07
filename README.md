# Futsal Elo Lab

An explainable match prediction project for amateur futsal: turn FCF match results into team ratings, three-way probabilities and an auditable evaluation.

Built around Barcelona Group 10, with a personal connection to POLARIS FORT PIENC B. The project asks whether a lightweight rating system can capture useful signal when detailed player and event data are unavailable.

## Try it

Python 3.10–3.12. Run commands from the repository root.

```bash
python -m venv .venv
# Activate: source .venv/bin/activate (macOS/Linux)
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/run.py --data data/FCF_Futsal_BCN_Gr10_2025_26_DB.xlsx --out out --plot-calibration
```

For the interactive matchup explorer:

```bash
python -m pip install -r requirements-demo.txt
streamlit run app.py
```

Select two teams to explore win/draw/loss probabilities, rankings, rating trajectories and evaluation metrics. The demo runs locally using the bundled snapshot.

## What this demonstrates

- **Modeling:** home-adjusted Elo, goal-margin scaling, season reversion and logistic draw estimation.
- **Evaluation:** expanding-window predictions, date-batched updates and baselines using only earlier results.
- **Engineering:** validated Excel/CSV ingestion, reproducible CLI outputs, regression tests and GitHub Actions.
- **Communication:** a local interactive demo, documented assumptions and explicit limitations.

## Method

Every team starts at 1500. Home advantage adds 100 points when computing the home expected score. A K-factor of 32 scales rating changes, with a logarithmic goal-margin multiplier; draws use an observed score of 0.5. A bounded correction prevents singular or negative margin multipliers. At a season change, ratings move 15% toward 1500.

The draw model uses the absolute pre-match rating difference. During evaluation it is refitted at each new date using only completed earlier dates, after at least 20 observations containing both draws and non-draws. Before then, the fixed draw prior is 15%. The remaining probability mass is divided using the home-adjusted Elo logistic function. This is a pragmatic three-way probability model, not a joint goal-scoring model.

All matches with the same Date value are forecast before incorporating any of their results. Supply dates without kickoff times for conservative day-level evaluation. Final ratings and a draw model fitted to all available history are then used for future forecasts.

## Evaluation and results

The CLI prints and saves actual computed results; this README deliberately contains no unverified performance numbers. Earlier versions fitted the draw estimator on the full dataset before backtesting. Those historical scores should be recomputed with this version before being quoted.

| Metric | Interpretation |
|---|---|
| logloss | Multiclass negative log likelihood; lower is better |
| baseline_logloss | Expanding historical outcome frequencies with add-one smoothing |
| uniform_logloss | Equal three-way probabilities, approximately 1.099 |
| brier_home/draw/away | Binary squared probability error for each outcome |
| brier_multiclass | Sum of the three Brier scores; range 0–2 |
| accuracy | Fraction of most-probable outcomes that are correct |

The baseline and model are scored on exactly the same matches, including the cold start. Inspect calibration plots alongside aggregate metrics. CI runs the bundled-data pipeline and attaches its output artifacts to each successful run.

## Reproducible outputs

| File | Contents |
|---|---|
| fixture_predictions.csv | Date, teams, pre-match ratings, probabilities and observed outcome |
| rankings.csv | Final team ratings, descending |
| elo_history.csv / elo_history.png | Rating trajectories |
| calibration.png | Optional predicted probability versus observed frequency |
| metrics.json | Model and baseline metrics |
| run_manifest.json | Input SHA-256, date coverage, seed, parameters and Python version |
| forecast.json | Optional named matchup probabilities and illustrative decimal odds |

To forecast a real matchup, append `--home "EXACT TEAM NAME" --away "EXACT TEAM NAME"` to the CLI command; copy names from rankings.csv. Unknown teams are rejected. `--commission 0` gives fair decimal odds; the default 0.20 applies proportional overround for illustration.

## Data contract

See [data documentation](docs/DATA.md) for source context and schema. CSV and Excel files use the first sheet by default. Completed fixtures require valid dates, distinct nonempty teams and integer non-negative scores. Rows with both scores absent are excluded; partial scores, duplicates and malformed completed fixtures are rejected.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover temporal leakage, same-date isolation, probability normalization, rating conservation, repeatable evaluation, sparse draw data, invalid inputs and the bundled workbook. GitHub Actions tests Python 3.10 and 3.12 and runs the full export pipeline.

## Limitations and next steps

This is a historical amateur-league snapshot, not a live data feed. Small samples, postponed fixtures, changing squads and division strength limit generalization. Hyperparameters are defaults, not claimed to be optimal. The Elo expected score is used as a conditional non-draw win share, which needs empirical calibration.

Useful next experiments are held-out-season evaluation, nested chronological tuning, bootstrap uncertainty with temporal blocks, a Poisson goal-model baseline and sensitivity analysis for home advantage and draw priors. Demonstrating improvement over the historical-frequency baseline requires rerunning and inspecting the generated metrics.

MIT license for code. Match-data rights remain with their original source; see the data documentation.
