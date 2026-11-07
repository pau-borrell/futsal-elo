"""Reproducible command-line entry point."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import pandas as pd
from elo_system import EloSystem


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="Excel or CSV match history")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--commission", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-calibration", action="store_true")
    ap.add_argument("--home", help="Optional known home team")
    ap.add_argument("--away", help="Optional known away team")
    args = ap.parse_args()
    if bool(args.home) != bool(args.away):
        ap.error("--home and --away must be supplied together")
    elo = EloSystem(random_state=args.seed)
    try:
        df = elo.load_and_prepare_data(args.data)
        elo.initialize_teams(df)
        args.out.mkdir(parents=True, exist_ok=True)
        elo.run_backtesting_and_evaluation(
            save_calibration_plots_to=str(args.out / "calibration.png") if args.plot_calibration else None,
            save_elo_plot_to=str(args.out / "elo_history.png"),
        )
        elo.train_draw_model()
        elo.backtest_results_df.to_csv(args.out / "fixture_predictions.csv", index=False)
        elo.elo_history_df.to_csv(args.out / "elo_history.csv", index=False)
        rankings = pd.DataFrame(sorted(elo.elo_ratings.items(), key=lambda x: -x[1]),
                                columns=["Team", "Elo"])
        rankings.to_csv(args.out / "rankings.csv", index=False)
        manifest = {
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "python": platform.python_version(), "seed": args.seed,
            "first_date": str(df.Date.min()), "last_date": str(df.Date.max()),
            "teams": len(rankings), "evaluation": "expanding-window, date-batched",
            "parameters": {"initial_elo": elo.initial_elo, "k_factor": elo.k_factor,
                           "home_advantage": elo.home_advantage,
                           "reversion_percentage": elo.reversion_percentage},
        }
        for name, value in [("metrics", elo.evaluation_metrics), ("run_manifest", manifest)]:
            (args.out / f"{name}.json").write_text(
                json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        if args.home:
            forecast = elo.predict_match_probabilities_and_odds(
                args.home, args.away, overround=args.commission)
            (args.out / "forecast.json").write_text(json.dumps(forecast, indent=2), encoding="utf-8")
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    print(json.dumps(elo.evaluation_metrics, indent=2))
    print(f"Artifacts saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
