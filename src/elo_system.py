import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class EloSystem:
    """
    Elo-based rating and prediction system with:
    - Goal-difference scaling
    - Home advantage
    - End-of-season reversion
    - Data-driven draw probability via logistic regression
    - Backtesting with Brier scores and multiclass LogLoss
    - Optional calibration plots
    - Elo history plot
    """

    def __init__(
        self,
        initial_elo: float = 1500.0,
        k_factor: float = 32.0,
        home_advantage: float = 100.0,
        reversion_percentage: float = 0.15,
        random_state: int = 42,
    ):
        self.initial_elo = float(initial_elo)
        self.k_factor = float(k_factor)
        self.home_advantage = float(home_advantage)
        self.reversion_percentage = float(reversion_percentage)
        self.elo_ratings: Dict[str, float] = {}
        self.played_games_df: pd.DataFrame | None = None
        self.draw_model: LogisticRegression | None = None
        self.backtest_results_df: pd.DataFrame | None = None
        self.evaluation_metrics: Dict[str, float] = {}
        self.elo_history_df: pd.DataFrame | None = None
        self.random_state = int(random_state)
        if not 0 <= self.reversion_percentage <= 1:
            raise ValueError("reversion_percentage must be between 0 and 1.")
        if self.k_factor <= 0 or not np.isfinite([
            self.initial_elo, self.k_factor, self.home_advantage
        ]).all():
            raise ValueError("Elo parameters must be finite and k_factor positive.")

    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        df = (pd.read_csv(file_path) if str(file_path).lower().endswith(".csv")
              else pd.read_excel(file_path))
        required = {"HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Date"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        # Unplayed fixtures have both scores missing; partial scores are errors.
        partial = df[["HomeGoals", "AwayGoals"]].isna().sum(axis=1) == 1
        if partial.any():
            raise ValueError("Fixtures must have both scores or neither.")
        df = df.dropna(subset=["HomeGoals", "AwayGoals"]).copy()
        if df.empty:
            raise ValueError("No completed matches found.")
        for col in ["HomeTeam", "AwayTeam"]:
            if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
                raise ValueError("Team names cannot be empty.")
            df[col] = df[col].astype(str).str.strip()
        if df["HomeTeam"].eq(df["AwayTeam"]).any():
            raise ValueError("A team cannot play itself.")
        for col in ["HomeGoals", "AwayGoals"]:
            values = pd.to_numeric(df[col], errors="raise")
            if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
                raise ValueError("Goals must be finite non-negative integers.")
            df[col] = values.astype(int)
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")
        if df["Date"].isna().any():
            raise ValueError("Completed matches require a date.")
        if "Season" in df and df["Season"].isna().any():
            raise ValueError("Season cannot be missing when supplied.")
        if df.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
            raise ValueError("Duplicate fixture found.")
        # Stable ordering; evaluation freezes all information within each date.
        df = df.sort_values("Date", kind="stable").reset_index(drop=True)
        self.played_games_df = df
        return df

    def initialize_teams(self, df: pd.DataFrame) -> None:
        teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
        self.elo_ratings = {t: self.initial_elo for t in teams}

    def _expected_home_win(self, home_elo: float, away_elo: float) -> float:
        adj_home = home_elo + self.home_advantage
        adj_away = away_elo
        diff = adj_home - adj_away
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update_elo(
        self,
        home_elo: float,
        away_elo: float,
        home_goals: int,
        away_goals: int,
    ) -> Tuple[float, float]:
        expected = self._expected_home_win(home_elo, away_elo)
        if home_goals > away_goals:
            outcome = 1.0
            factor = np.log(abs(home_goals - away_goals) + 1.0) * (
                2.2 / max(0.5, ((home_elo + self.home_advantage) - away_elo) * 0.001 + 2.2)
            )
        elif away_goals > home_goals:
            outcome = 0.0
            factor = np.log(abs(home_goals - away_goals) + 1.0) * (
                2.2 / max(0.5, (away_elo - (home_elo + self.home_advantage)) * 0.001 + 2.2)
            )
        else:
            outcome = 0.5
            factor = 1.0
        change = self.k_factor * (outcome - expected) * factor
        return home_elo + change, away_elo - change

    def train_draw_model(self) -> None:
        if self.played_games_df is None or self.played_games_df.empty:
            raise ValueError("No data loaded. Call load_and_prepare_data first.")
        temp_ratings = {t: self.initial_elo for t in self.elo_ratings}
        diffs: list[float] = []
        self.initialize_teams(self.played_games_df)
        prev_season = None
        for _, r in self.played_games_df.iterrows():
            if prev_season is not None and "Season" in r and r["Season"] != prev_season:
                for t in temp_ratings:
                    temp_ratings[t] = (1 - self.reversion_percentage) * temp_ratings[t] + self.reversion_percentage * self.initial_elo
            h, a = r["HomeTeam"], r["AwayTeam"]
            h_elo, a_elo = temp_ratings[h], temp_ratings[a]
            diffs.append(abs(h_elo - a_elo))
            uh, ua = self.update_elo(h_elo, a_elo, int(r["HomeGoals"]), int(r["AwayGoals"]))
            temp_ratings[h], temp_ratings[a] = uh, ua
            prev_season = r["Season"] if "Season" in r else prev_season
        if self.backtest_results_df is not None:
            # Match date-batched pre-match features used during evaluation.
            diffs = (self.backtest_results_df["HomeEloBefore"]
                     - self.backtest_results_df["AwayEloBefore"]).abs().tolist()
        X = np.array(diffs).reshape(-1, 1)
        y = (self.played_games_df["HomeGoals"] == self.played_games_df["AwayGoals"]).astype(int).to_numpy()
        self.draw_model = LogisticRegression(solver="liblinear", random_state=self.random_state)
        if len(np.unique(y)) < 2:
            self.draw_model = None
        else:
            self.draw_model.fit(X, y)

    def _predict_draw_probability(self, elo_diff: float) -> float:
        if self.draw_model is None:
            return 0.15
        return float(self.draw_model.predict_proba(np.array([[abs(elo_diff)]]))[:, 1][0])

    def calculate_outcome_probabilities(self, home_elo: float, away_elo: float) -> Tuple[float, float, float]:
        diff = home_elo - away_elo
        p_draw = self._predict_draw_probability(diff)
        p_home_raw = self._expected_home_win(home_elo, away_elo)
        remainder = max(0.0, 1.0 - p_draw)
        p_home = p_home_raw * remainder
        p_away = (1.0 - p_home_raw) * remainder
        s = p_home + p_draw + p_away
        if s <= 0:
            return 1 / 3, 1 / 3, 1 / 3
        return p_home / s, p_draw / s, p_away / s

    def calculate_odds(self, p: float, overround: float = 0.20) -> float | None:
        if not np.isfinite(overround) or overround < 0:
            raise ValueError("overround must be finite and non-negative.")
        if p <= 0.0 or p >= 1.0:
            return None
        return 1.0 / (p * (1.0 + overround))

    def predict_match_probabilities_and_odds(
        self,
        home_team: str,
        away_team: str,
        overround: float = 0.20
    ) -> Dict[str, object]:
        if home_team == away_team:
            raise ValueError("Choose two different teams.")
        unknown = {home_team, away_team} - self.elo_ratings.keys()
        if unknown:
            raise ValueError(f"Unknown teams: {sorted(unknown)}")
        h_elo = self.elo_ratings[home_team]
        a_elo = self.elo_ratings.get(away_team, self.initial_elo)
        ph, p_draw, pa = self.calculate_outcome_probabilities(h_elo, a_elo)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "probs": (ph, p_draw, pa),
            "odds": (
                self.calculate_odds(ph, overround),
                self.calculate_odds(p_draw, overround),
                self.calculate_odds(pa, overround),
            ),
        }

    def process_historical_games(self) -> None:
        if self.played_games_df is None or self.played_games_df.empty:
            raise ValueError("No data loaded. Call load_and_prepare_data first.")
        if not self.elo_ratings:
            raise ValueError("Teams not initialized. Call initialize_teams first.")
        self.initialize_teams(self.played_games_df)
        prev_season = None
        for _, r in self.played_games_df.iterrows():
            if prev_season is not None and "Season" in r and r["Season"] != prev_season:
                for t in self.elo_ratings:
                    self.elo_ratings[t] = (1 - self.reversion_percentage) * self.elo_ratings[t] + self.reversion_percentage * self.initial_elo
            h, a = r["HomeTeam"], r["AwayTeam"]
            uh, ua = self.update_elo(self.elo_ratings[h], self.elo_ratings[a], int(r["HomeGoals"]), int(r["AwayGoals"]))
            self.elo_ratings[h], self.elo_ratings[a] = uh, ua
            prev_season = r["Season"] if "Season" in r else prev_season

    def run_backtesting_and_evaluation(
        self,
        save_calibration_plots_to: str | None = None,
        n_bins: int = 10,
        save_elo_plot_to: str | None = None,
    ) -> None:
        if self.played_games_df is None or self.played_games_df.empty:
            raise ValueError("No data loaded. Call load_and_prepare_data first.")
        if not self.elo_ratings:
            raise ValueError("Teams not initialized. Call initialize_teams first.")
        # Never reuse a draw model trained on the full dataset in evaluation.
        self.draw_model = None
        past_features, past_draws = [], []
        outcome_counts = np.ones(3)
        active_date = None
        pending = []
        baseline_predictions = []

        def learn_pending():
            for h, a, hg, ag, feature, outcome in pending:
                temp[h], temp[a] = self.update_elo(temp[h], temp[a], hg, ag)
                past_features.append([feature])
                past_draws.append(int(hg == ag))
                outcome_counts[outcome] += 1
            pending.clear()
            self.draw_model = None
            if len(past_draws) >= 20 and len(set(past_draws)) == 2:
                self.draw_model = LogisticRegression(
                    solver="liblinear", random_state=self.random_state
                ).fit(past_features, past_draws)
        temp = {t: self.initial_elo for t in self.elo_ratings}
        prev_season = None
        rows = []
        hist = []
        step = 0
        for t, v in temp.items():
            hist.append({"Step": step, "Team": t, "Elo": v})
        for _, r in self.played_games_df.iterrows():
            if r["Date"] != active_date:
                learn_pending()
                active_date = r["Date"]
            if prev_season is not None and "Season" in r and r["Season"] != prev_season:
                for t in temp:
                    temp[t] = (1 - self.reversion_percentage) * temp[t] + self.reversion_percentage * self.initial_elo
            h, a = r["HomeTeam"], r["AwayTeam"]
            h_elo, a_elo = temp[h], temp[a]
            ph, p_draw, pa = self.calculate_outcome_probabilities(h_elo, a_elo)
            if r["HomeGoals"] > r["AwayGoals"]:
                actual = 1.0
            elif r["AwayGoals"] > r["HomeGoals"]:
                actual = 0.0
            else:
                actual = 0.5
            rows.append(
                {
                    "Date": r["Date"].isoformat(),
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
            baseline_predictions.append(outcome_counts / outcome_counts.sum())
            pending.append((h, a, int(r["HomeGoals"]), int(r["AwayGoals"]),
                            abs(h_elo - a_elo), {0.0: 0, 0.5: 1, 1.0: 2}[actual]))
            uh, ua = self.update_elo(h_elo, a_elo, int(r["HomeGoals"]), int(r["AwayGoals"]))
            step += 1
            hist.append({"Step": step, "Team": h, "Elo": uh})
            hist.append({"Step": step, "Team": a, "Elo": ua})
            prev_season = r["Season"] if "Season" in r else prev_season
        learn_pending()
        self.elo_ratings = temp.copy()
        df = pd.DataFrame(rows)
        self.backtest_results_df = df
        self.elo_history_df = pd.DataFrame(hist)
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
            "n_matches": len(df),
            "baseline_logloss": float(log_loss(y_true, baseline_predictions, labels=[0, 1, 2])),
            "uniform_logloss": float(np.log(3)),
            "brier_multiclass": float(brier_home + brier_draw + brier_away),
            "accuracy": float(np.mean(y_pred.argmax(axis=1) == y_true)),
            "brier_home": float(brier_home),
            "brier_draw": float(brier_draw),
            "brier_away": float(brier_away),
            "logloss": float(ll),
        }
        if save_calibration_plots_to is not None:
            plt.figure(figsize=(14, 4))
            def _cal(ax, y_true_bin, y_prob, title):
                frac_pos, prob_pred = calibration_curve(
                    y_true_bin, y_prob, n_bins=n_bins, strategy="uniform"
                )
                ax.plot([0, 1], [0, 1], "k:", label="perfect")
                ax.plot(prob_pred, frac_pos, "o-", label=title)
                ax.set_title(title)
                ax.set_xlabel("Mean predicted prob")
                ax.set_ylabel("Observed frequency")
                ax.legend(loc="lower right")
                ax.grid(alpha=0.4)
            ax1 = plt.subplot(1, 3, 1)
            _cal(ax1, df["ActualHome"].to_numpy(), df["HomeWinProb"].to_numpy(), "Home")
            ax2 = plt.subplot(1, 3, 2)
            _cal(ax2, df["ActualDraw"].to_numpy(), df["DrawProb"].to_numpy(), "Draw")
            ax3 = plt.subplot(1, 3, 3)
            _cal(ax3, df["ActualAway"].to_numpy(), df["AwayWinProb"].to_numpy(), "Away")
            plt.tight_layout()
            plt.savefig(save_calibration_plots_to, dpi=160)
            plt.close()
        if save_elo_plot_to is not None and self.elo_history_df is not None and not self.elo_history_df.empty:
            plt.figure(figsize=(12, 6))
            for team, g in self.elo_history_df.groupby("Team"):
                g = g.sort_values("Step")
                plt.plot(g["Step"].to_numpy(), g["Elo"].to_numpy(), linewidth=1, alpha=0.7)
            plt.xlabel("Step")
            plt.ylabel("Elo")
            plt.title("Elo ratings over time")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_elo_plot_to, dpi=160)
            plt.close()
