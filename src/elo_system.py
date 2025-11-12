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

        # Runtime members
        self.elo_ratings: Dict[str, float] = {}
        self.played_games_df: pd.DataFrame | None = None
        self.draw_model: LogisticRegression | None = None
        self.backtest_results_df: pd.DataFrame | None = None
        self.evaluation_metrics: Dict[str, float] = {}
        self.random_state = int(random_state)
        np.random.seed(self.random_state)

    # ---------------------------
    # Data loading
    # ---------------------------
    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """
        Load Excel, drop unplayed matches, cast goals to int, and sort chronologically.
        Required columns: HomeTeam, AwayTeam, HomeGoals, AwayGoals.
        Optional: Season, Date, Jornada.
        """
        df = pd.read_excel(file_path)

        # Keep only matches with both scores present
        df = df.dropna(subset=["HomeGoals", "AwayGoals"]).copy()
        df["HomeGoals"] = df["HomeGoals"].astype(int)
        df["AwayGoals"] = df["AwayGoals"].astype(int)

        # Sort by Season → Date → Jornada if present, else Date
        sort_cols = [c for c in ["Season", "Date", "Jornada"] if c in df.columns]
        df = df.sort_values(by=sort_cols if sort_cols else ["Date"]).reset_index(drop=True)

        self.played_games_df = df
        return df

    def initialize_teams(self, df: pd.DataFrame) -> None:
        """
        Initialize all seen teams to initial Elo.
        """
        teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
        self.elo_ratings = {t: self.initial_elo for t in teams}

    # ---------------------------
    # Core Elo update
    # ---------------------------
    def _expected_home_win(self, home_elo: float, away_elo: float) -> float:
        """
        Standard Elo expectation with home-advantage shift applied to the home team.
        """
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
        """
        Update both teams’ ratings for one match.
        Includes goal-difference scaling (common Elo extension).
        """
        expected = self._expected_home_win(home_elo, away_elo)

        # Match outcome as Elo target
        if home_goals > away_goals:
            outcome = 1.0
            # Goal-diff factor increases magnitude for bigger wins, damped by rating gap
            factor = np.log(abs(home_goals - away_goals) + 1.0) * (
                2.2 / (((home_elo + self.home_advantage) - away_elo) * 0.001 + 2.2)
            )
        elif away_goals > home_goals:
            outcome = 0.0
            factor = np.log(abs(home_goals - away_goals) + 1.0) * (
                2.2 / (((away_elo) - (home_elo + self.home_advantage)) * 0.001 + 2.2)
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
        """
        Train a logistic regression that maps pre-match Elo difference (home - away)
        to probability of a draw. Uses a single global model fit on all historic games.
        """
        if self.played_games_df is None or self.played_games_df.empty:
            raise ValueError("No data loaded. Call load_and_prepare_data first.")

        # Build pre-match Elo differences by simulating through history with current params
        temp_ratings = {t: self.initial_elo for t in self.elo_ratings}
        diffs: list[float] = []
        prev_season = None

        for _, r in self.played_games_df.iterrows():
            # Season reversion at boundary
            if prev_season is not None and "Season" in r and r["Season"] != prev_season:
                for t in temp_ratings:
                    temp_ratings[t] = (1 - self.reversion_percentage) * temp_ratings[t] + self.reversion_percentage * self.initial_elo

            h, a = r["HomeTeam"], r["AwayTeam"]
            h_elo, a_elo = temp_ratings[h], temp_ratings[a]
            diffs.append(h_elo - a_elo)

            uh, ua = self.update_elo(h_elo, a_elo, int(r["HomeGoals"]), int(r["AwayGoals"]))
            temp_ratings[h], temp_ratings[a] = uh, ua
            prev_season = r["Season"] if "Season" in r else prev_season

        X = np.array(diffs).reshape(-1, 1)
        y = (self.played_games_df["HomeGoals"] == self.played_games_df["AwayGoals"]).astype(int).to_numpy()

        self.draw_model = LogisticRegression(solver="liblinear", random_state=self.random_state)
        self.draw_model.fit(X, y)

    def _predict_draw_probability(self, elo_diff: float) -> float:
        """
        Predict P(draw) from Elo difference. Fallback to 0.15 if model not trained.
        """
        if self.draw_model is None:
            return 0.15
        return float(self.draw_model.predict_proba(np.array([[elo_diff]]))[:, 1][0])

    # ---------------------------
    # Probabilities and odds
    # ---------------------------
    def calculate_outcome_probabilities(self, home_elo: float, away_elo: float) -> Tuple[float, float, float]:
        """
        Compute (P_home, P_draw, P_away) that sum to 1.
        We take standard Elo home-win expectation, reserve P_draw from the model,
        and split the remaining mass between home/away in proportion to expectation.
        """
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
        """
        Convert true probability to bookmaker-style odds with a simple overround.
        If p in {0,1}, return None since odds would be undefined or infinite.
        """
        if p <= 0.0 or p >= 1.0:
            return None
        return 1.0 / (p * (1.0 + overround))

    def predict_match_probabilities_and_odds(
        self, home_team: str, away_team: str, overround: float = 0.20
    ) -> Dict[str, object]:
        """
        Convenience API: get probabilities and odds for an arbitrary matchup.
        Uses current Elo ratings. If a team is new, uses initial Elo.
        """
        h_elo = self.elo_ratings.get(home_team, self.initial_elo)
        a_elo = self.elo_ratings.get(away_team, self.initial_elo)
        ph, pd, pa = self.calculate_outcome_probabilities(h_elo, a_elo)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "probs": (ph, pd, pa),
            "odds": (
                self.calculate_odds(ph, overround),
                self.calculate_odds(pd, overround),
                self.calculate_odds(pa, overround),
            ),
        }

    # ---------------------------
    # Historical processing
    # ---------------------------
    def process_historical_games(self) -> None:
        """
        Walk through history once to produce final current Elo ratings.
        """
        if self.played_games_df is None or self.played_games_df.empty:
            raise ValueError("No data loaded. Call load_and_prepare_data first.")
        if not self.elo_ratings:
            raise ValueError("Teams not initialized. Call initialize_teams first.")

        prev_season = None
        for _, r in self.played_games_df.iterrows():
            # Reversion on season boundary
            if prev_season is not None and "Season" in r and r["Season"] != prev_season:
                for t in self.elo_ratings:
                    self.elo_ratings[t] = (1 - self.reversion_percentage) * self.elo_ratings[t] + self.reversion_percentage * self.initial_elo

            h, a = r["HomeTeam"], r["AwayTeam"]
            uh, ua = self.update_elo(self.elo_ratings[h], self.elo_ratings[a], int(r["HomeGoals"]), int(r["AwayGoals"]))
            self.elo_ratings[h], self.elo_ratings[a] = uh, ua
            prev_season = r["Season"] if "Season" in r else prev_season

    # ---------------------------
    # Backtesting and evaluation
    # ---------------------------
    def run_backtesting_and_evaluation(
        self,
        save_calibration_plots_to: str | None = None,
        n_bins: int = 10,
    ) -> None:
        """
        One-pass chronological backtest.
        At each step, use the current Elo ratings and a global draw model to predict,
        then update with the observed result. Stores predictions and metrics.
        """
        if self.played_games_df is None or self.played_games_df.empty:
            raise ValueError("No data loaded. Call load_and_prepare_data first.")
        if not self.elo_ratings:
            raise ValueError("Teams not initialized. Call initialize_teams first.")
        if self.draw_model is None:
            # Train once on whole history. Simple, fast, and good for a first public version.
            self.train_draw_model()

        # Fresh temp ratings for the simulation
        temp = {t: self.initial_elo for t in self.elo_ratings}
        prev_season = None
        rows = []

        for _, r in self.played_games_df.iterrows():
            # Reversion on season boundary
            if prev_season is not None and "Season" in r and r["Season"] != prev_season:
                for t in temp:
                    temp[t] = (1 - self.reversion_percentage) * temp[t] + self.reversion_percentage * self.initial_elo

            h, a = r["HomeTeam"], r["AwayTeam"]
            h_elo, a_elo = temp[h], temp[a]

            # Predict before seeing the result
            ph, pd, pa = self.calculate_outcome_probabilities(h_elo, a_elo)

            # Encode actual result
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
                    "DrawProb": pd,
                    "AwayWinProb": pa,
                    "Actual": actual,
                }
            )

            # Update ratings with the observed outcome
            uh, ua = self.update_elo(h_elo, a_elo, int(r["HomeGoals"]), int(r["AwayGoals"]))
            temp[h], temp[a] = uh, ua
            prev_season = r["Season"] if "Season" in r else prev_season

        df = pd.DataFrame(rows)
        self.backtest_results_df = df

        # Brier scores for each outcome
        df["ActualHome"] = (df["Actual"] == 1.0).astype(int)
        df["ActualDraw"] = (df["Actual"] == 0.5).astype(int)
        df["ActualAway"] = (df["Actual"] == 0.0).astype(int)
        brier_home = brier_score_loss(df["ActualHome"], df["HomeWinProb"])
        brier_draw = brier_score_loss(df["ActualDraw"], df["DrawProb"])
        brier_away = brier_score_loss(df["ActualAway"], df["AwayWinProb"])

        # Multiclass LogLoss: order [Away, Draw, Home]
        y_true = df["Actual"].map({0.0: 0, 0.5: 1, 1.0: 2}).to_numpy()
        y_pred = df[["AwayWinProb", "DrawProb", "HomeWinProb"]].to_numpy()
        # Safety clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1.0)
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        ll = log_loss(y_true, y_pred, labels=[0, 1, 2])

        self.evaluation_metrics = {
            "brier_home": float(brier_home),
            "brier_draw": float(brier_draw),
            "brier_away": float(brier_away),
            "logloss": float(ll),
        }

        # Optional calibration plots per class
        if save_calibration_plots_to is not None:
            plt.figure(figsize=(14, 4))

            def _cal(ax, y_true_bin, y_prob, title):
                try:
                    prob_pred, frac_pos = calibration_curve(y_true_bin, y_prob, n_bins=n_bins, strategy="uniform")
                except Exception:
                    # Degenerate cases: flat lines
                    prob_pred, frac_pos = np.array([0.0, 1.0]), np.array([0.0, 1.0])
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
