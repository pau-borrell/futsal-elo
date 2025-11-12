import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


class EloSystem:
    """
    Elo rating and prediction system with:
      - Goal difference scaling
      - Home advantage adjustment
      - End-of-season reversion
      - Data-driven draw probability via logistic regression
      - Backtesting and evaluation with Brier and LogLoss
      - Calibration and Elo evolution plots
    """

    def __init__(
        self,
        initial_elo: float = 1500.0,
        k_factor: float = 32.0,
        home_advantage: float = 100.0,
        reversion_percentage: float = 0.15,
        random_state: int = 42,
    ):
        # Core hyperparameters
        self.initial_elo = float(initial_elo)
        self.k_factor = float(k_factor)
        self.home_advantage = float(home_advantage)
        self.reversion_percentage = float(reversion_percentage)
        self.random_state = int(random_state)

        # Runtime variables
        self.elo_ratings: Dict[str, float] = {}
        self.played_games_df: pd.DataFrame | None = None
        self.draw_model: LogisticRegression | None = None
        self.backtest_results_df: pd.DataFrame | None = None
        self.evaluation_metrics: Dict[str, float] = {}
        np.random.seed(self.random_state)

    # ---------------------------
    # Data loading
    # ---------------------------
    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """
        Load Excel file, clean and sort chronologically.
        Must include HomeTeam, AwayTeam, HomeGoals, AwayGoals.
        """
        df = pd.read_excel(file_path)
        df = df.dropna(subset=["HomeGoals", "AwayGoals"]).copy()
        df["HomeGoals"] = df["HomeGoals"].astype(int)
        df["AwayGoals"] = df["AwayGoals"].astype(int)

        sort_cols = [c for c in ["Season", "Date", "Jornada"] if c in df.columns]
        df = df.sort_values(by=sort_cols if sort_cols else ["Date"]).reset_index(drop=True)
        self.played_games_df = df
        return df

    def initialize_teams(self, df: pd.DataFrame) -> None:
        """Initialize Elo ratings for all teams."""
        teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
        self.elo_ratings = {t: self.initial_elo for t in teams}

    # ---------------------------
    # Core Elo mechanics
    # ---------------------------
    def _expected_home_win(self, home_elo: float, away_elo: float) -> float:
        """Compute expected home win probability with home advantage."""
        diff = (home_elo + self.home_advantage) - away_elo
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update_elo(self, home_elo: float, away_elo: float, home_goals: int, away_goals: int) -> Tuple[float, float]:
        """
        Update Elo ratings for one match.
        Includes goal-difference scaling.
        """
        expected = self._expected_home_win(home_elo, away_elo)

        # Determine outcome
        if home_goals > away_goals:
            outcome = 1.0
            factor = np.log(abs(home_goals - away_goals) + 1.0) * (
                2.2 / ((((home_elo + self.home_advantage) - away_elo) * 0.001) + 2.2)
            )
        elif away_goals > home_goals:
            outcome = 0.0
            factor = np.log(abs(home_goals - away_goals) + 1.0) * (
                2.2 / ((((away_elo) - (home_elo + self.home_advantage)) * 0.001) + 2.2)
            )
        else:
            outcome = 0.5
            factor = 1.0

        change = self.k_factor * (outcome - expected) * factor
        return home_elo + change, away_elo - change

    # ---------------------------
    # Draw model
    # ---------------------------
    def train_draw_model(self) -> None:
        """Train logistic regression to predict draws using |Elo_diff|."""
        if self.played_games_df is None or self.played_games_df.empty:
            raise ValueError("No data loaded.")

        if not self.elo_ratings:
            self.initialize_teams(self.played_games_df)

        temp = {t: self.initial_elo for t in self.elo_ratings}
        diffs = []
        prev_season = None

        for _, r in self.played_games_df.iterrows():
            # Apply seasonal reversion
            if prev_season is not None and ("Season" in r) and (r["Season"] != prev_season):
                for t in temp:
                    temp[t] = (1 - self.reversion_percentage) * temp[t] + self.reversion_percentage * self.initial_elo

            h, a = r["HomeTeam"], r["AwayTeam"]
            h_elo, a_elo = temp[h], temp[a]
            diffs.append(abs(h_elo - a_elo))

            uh, ua = self.update_elo(h_elo, a_elo, int(r["HomeGoals"]), int(r["AwayGoals"]))
            temp[h], temp[a] = uh, ua
            prev_season = r["Season"] if "Season" in r else prev_season

        X = np.array(diffs, dtype=float).reshape(-1, 1)
        y = (self.played_games_df["HomeGoals"] == self.played_games_df["AwayGoals"]).astype(int).to_numpy()

        self.draw_model = LogisticRegression(solver="liblinear", random_state=self.random_state)
        self.draw_model.fit(X, y)

    def _predict_draw_probability(self, elo_diff_abs: float) -> float:
        """Predict draw probability from |Elo_diff|."""
        if self.draw_model is None:
            return 0.15
        return float(self.draw_model.predict_proba(np.array([[elo_diff_abs]], dtype=float))[:, 1][0])

    # ---------------------------
    # Prediction and odds
    # ---------------------------
    def calculate_outcome_probabilities(self, home_elo: float, away_elo: float) -> Tuple[float, float, float]:
        """Return normalized (P_home, P_draw, P_away)."""
        elo_diff_abs = abs(home_elo - away_elo)
        p_draw = self._predict_draw_probability(elo_diff_abs)
        p_home_raw = self._expected_home_win(home_elo, away_elo)
        remainder = max(0.0, 1.0 - p_draw)
        p_home = p_home_raw * remainder
        p_away = (1.0 - p_home_raw) * remainder
        s = p_home + p_draw + p_away
        if s <= 0:
            return 1 / 3, 1 / 3, 1 / 3
        return p_home / s, p_draw / s, p_away / s

    def calculate_odds(self, p: float, overround: float = 0.20) -> float | None:
        """Convert probabilities to bookmaker odds."""
        if p <= 0.0 or p >= 1.0:
            return None
        return 1.0 / (p * (1.0 + overround))

    # ---------------------------
    # Historical processing and plotting
    # ---------------------------
    def process_historical_games(self, save_plot_to: str | None = None) -> pd.DataFrame:
        """
        Replay historical matches to compute Elo evolution.
        Returns a DataFrame with Elo over time and optionally saves a plot.
        """
        if self.played_games_df is None or self.played_games_df.empty:
            raise ValueError("No data loaded.")
        if not self.elo_ratings:
            self.initialize_teams(self.played_games_df)

        prev_season = None
        history = []

        for i, r in self.played_games_df.iterrows():
            if prev_season is not None and ("Season" in r) and (r["Season"] != prev_season):
                for t in self.elo_ratings:
                    self.elo_ratings[t] = (1 - self.reversion_percentage) * self.elo_ratings[t] + self.reversion_percentage * self.initial_elo

            h, a = r["HomeTeam"], r["AwayTeam"]
            h_elo_before, a_elo_before = self.elo_ratings[h], self.elo_ratings[a]
            uh, ua = self.update_elo(h_elo_before, a_elo_before, int(r["HomeGoals"]), int(r["AwayGoals"]))
            self.elo_ratings[h], self.elo_ratings[a] = uh, ua
            prev_season = r["Season"] if "Season" in r else prev_season

            # Record the Elo evolution for both teams
            history.append({"Match": i, "Team": h, "Elo": uh})
            history.append({"Match": i, "Team": a, "Elo": ua})

        history_df = pd.DataFrame(history)

        if save_plot_to:
            plt.figure(figsize=(10, 6))
            for team, subdf in history_df.groupby("Team"):
                plt.plot(subdf["Match"], subdf["Elo"], label=team)
            plt.xlabel("Match index")
            plt.ylabel("Elo rating")
            plt.title("Elo Evolution Over Time")
            plt.legend(fontsize="small")
            plt.grid(alpha=0.4)
            plt.tight_layout()
            plt.savefig(save_plot_to, dpi=160)
            plt.close()

        return history_df

    # ---------------------------
    # Backtesting
    # ---------------------------
    def run_backtesting_and_evaluation(self, out_dir: str = "out", n_bins: int = 10) -> None:
        """Simulate matches in order, compute predictions and metrics."""
        if self.played_games_df is None or self.played_games_df.empty:
            raise ValueError("No data loaded.")
        if not self.elo_ratings:
            raise ValueError("Teams not initialized.")
        if self.draw_model is None:
            self.train_draw_model()

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        temp = {t: self.initial_elo for t in self.elo_ratings}
        prev_season = None
        rows = []

        for _, r in self.played_games_df.iterrows():
            if prev_season is not None and ("Season" in r) and (r["Season"] != prev_season):
                for t in temp:
                    temp[t] = (1 - self.reversion_percentage) * temp[t] + self.reversion_percentage * self.initial_elo

            h, a = r["HomeTeam"], r["AwayTeam"]
            h_elo, a_elo = temp[h], temp[a]
            ph, p_draw, pa = self.calculate_outcome_probabilities(h_elo, a_elo)

            # Actual result encoding
            if r["HomeGoals"] > r["AwayGoals"]:
                actual = 1.0
            elif r["AwayGoals"] > r["HomeGoals"]:
                actual = 0.0
            else:
                actual = 0.5

            rows.append(
                {
                    "HomeTeam": h,
                    "AwayTeam": a,
                    "HomeEloBefore": h_elo,
                    "AwayEloBefore": a_elo,
                    "HomeWinProb": ph,
                    "DrawProb": p_draw,
                    "AwayWinProb": pa,
                    "Actual": actual,
                }
            )

            # Update
            uh, ua = self.update_elo(h_elo, a_elo, int(r["HomeGoals"]), int(r["AwayGoals"]))
            temp[h], temp[a] = uh, ua
            prev_season = r["Season"] if "Season" in r else prev_season

        df = pd.DataFrame(rows)
        self.backtest_results_df = df

        # Evaluation metrics
        df["ActualHome"] = (df["Actual"] == 1.0).astype(int)
        df["ActualDraw"] = (df["Actual"] == 0.5).astype(int)
        df["ActualAway"] = (df["Actual"] == 0.0).astype(int)

        brier_home = brier_score_loss(df["ActualHome"], df["HomeWinProb"])
        brier_draw = brier_score_loss(df["ActualDraw"], df["DrawProb"])
        brier_away = brier_score_loss(df["ActualAway"], df["AwayWinProb"])

        y_true = df["Actual"].map({0.0: 0, 0.5: 1, 1.0: 2}).to_numpy()
        y_pred = df[["AwayWinProb", "DrawProb", "HomeWinProb"]].to_numpy()
        y_pred = np.clip(y_pred, 1e-15, 1.0)
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        ll = log_loss(y_true, y_pred, labels=[0, 1, 2])

        self.evaluation_metrics = {
            "brier_home": float(brier_home),
            "brier_draw": float(brier_draw),
            "brier_away": float(brier_away),
            "logloss": float(ll),
        }

        # Save files
        preds_path = Path(out_dir) / "fixture_predictions.csv"
        metrics_path = Path(out_dir) / "metrics.json"
        elo_plot_path = Path(out_dir) / "elo_evolution.png"

        df.to_csv(preds_path, index=False)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_metrics, f, indent=2)

        # Plot calibration
        cal_path = Path(out_dir) / "calibration.png"
        plt.figure(figsize=(14, 4))
        def _cal(ax, y_true_bin, y_prob, title):
            try:
                prob_pred, frac_pos = calibration_curve(y_true_bin, y_prob, n_bins=n_bins, strategy="uniform")
            except Exception:
                prob_pred, frac_pos = np.array([0.0, 1.0]), np.array([0.0, 1.0])
            ax.plot([0, 1], [0, 1], "k:", label="perfect")
            ax.plot(prob_pred, frac_pos, "o-", label=title)
            ax.set_title(title)
            ax.set_xlabel("Mean predicted prob")
            ax.set_ylabel("Observed frequency")
            ax.legend(loc="lower right")
            ax.grid(alpha=0.4)

        _cal(plt.subplot(1, 3, 1), df["ActualHome"], df["HomeWinProb"], "Home")
        _cal(plt.subplot(1, 3, 2), df["ActualDraw"], df["DrawProb"], "Draw")
        _cal(plt.subplot(1, 3, 3), df["ActualAway"], df["AwayWinProb"], "Away")

        plt.tight_layout()
        plt.savefig(cal_path, dpi=160)
        plt.close()

        # Generate Elo evolution plot
        self.process_historical_games(save_plot_to=elo_plot_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, required=True)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--k", type=float, default=32.0)
    parser.add_argument("--ha", type=float, default=100.0)
    parser.add_argument("--rev", type=float, default=0.15)
    args = parser.parse_args()

    es = EloSystem(k_factor=args.k, home_advantage=args.ha, reversion_percentage=args.rev)
    df = es.load_and_prepare_data(args.xlsx)
    es.initialize_teams(df)
    es.train_draw_model()
    es.run_backtesting_and_evaluation(out_dir=args.out)
    print("Outputs saved to:", args.out)
