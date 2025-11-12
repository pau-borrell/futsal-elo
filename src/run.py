import argparse
import os
import json
from elo_system import EloSystem


def main():
    # CLI for reproducible runs
    ap = argparse.ArgumentParser(description="Run Elo-based futsal prediction pipeline.")
    ap.add_argument("--data", required=True, help="Path to Excel with matches.")
    ap.add_argument("--out", default="out", help="Output folder for CSVs/plots.")
    ap.add_argument("--commission", type=float, default=0.20, help="Overround for odds, e.g. 0.20 = 20%.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--plot-calibration", action="store_true", help="Save calibration plots to out/calibration.png")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Instantiate system
    elo = EloSystem(
        initial_elo=1500.0,
        k_factor=32.0,
        home_advantage=100.0,
        reversion_percentage=0.15,
        random_state=args.seed,
    )

    # Load data and set up teams
    df = elo.load_and_prepare_data(args.data)
    elo.initialize_teams(df)

    # Train draw model, compute final ratings, backtest
    elo.train_draw_model()
    elo.process_historical_games()

    cal_plot_path = os.path.join(args.out, "calibration.png") if args.plot_calibration else None
    elo.run_backtesting_and_evaluation(out_dir="out")

    # Save per-fixture predictions from backtest
    pred_csv = os.path.join(args.out, "fixture_predictions.csv")
    elo.backtest_results_df.to_csv(pred_csv, index=False)

    # Save metrics JSON
    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(elo.evaluation_metrics, f, indent=2)

    # Example odds for a hypothetical match; replace with real team names as needed
    example = elo.predict_match_probabilities_and_odds("Team A", "Team B", overround=args.commission)
    example_path = os.path.join(args.out, "example_odds.json")
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)

    # Minimal console output
    print("Saved:", pred_csv)
    print("Saved:", metrics_path)
    if cal_plot_path:
        print("Saved:", cal_plot_path)
    print("Saved:", example_path)


if __name__ == "__main__":
    main()
