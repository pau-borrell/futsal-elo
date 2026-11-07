import numpy as np
import pandas as pd
import pytest
from elo_system import EloSystem


def games(n=32):
    return pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=n),
        "Season": ["2025"] * n,
        "HomeTeam": ["A", "B"] * (n // 2),
        "AwayTeam": ["B", "A"] * (n // 2),
        "HomeGoals": [2, 1, 0, 3] * (n // 4),
        "AwayGoals": [1, 1, 2, 0] * (n // 4),
    })


def evaluate(df):
    m = EloSystem()
    m.played_games_df = df.copy()
    m.initialize_teams(df)
    m.run_backtesting_and_evaluation()
    return m


def test_future_results_do_not_change_earlier_predictions():
    original = games()
    altered = original.copy()
    altered.loc[24:, "HomeGoals"] = 20
    a, b = evaluate(original), evaluate(altered)
    cols = ["HomeWinProb", "DrawProb", "AwayWinProb"]
    np.testing.assert_allclose(a.backtest_results_df[cols].iloc[:24],
                               b.backtest_results_df[cols].iloc[:24])


def test_same_date_does_not_leak_results():
    df = games()
    df.loc[1, "Date"] = df.loc[0, "Date"]
    other = df.copy()
    other.loc[0, "HomeGoals"] = 50
    cols = ["HomeWinProb", "DrawProb", "AwayWinProb"]
    np.testing.assert_allclose(evaluate(df).backtest_results_df[cols].iloc[:2],
                               evaluate(other).backtest_results_df[cols].iloc[:2])


def test_probabilities_and_conservation():
    m = evaluate(games())
    p = m.backtest_results_df[["HomeWinProb", "DrawProb", "AwayWinProb"]].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert sum(m.elo_ratings.values()) == pytest.approx(3000)
    assert np.isfinite(list(m.evaluation_metrics.values())).all()


def test_single_class_and_repeatability():
    df = games()
    df["HomeGoals"], df["AwayGoals"] = 1, 0
    m = evaluate(df)
    first = m.backtest_results_df.copy()
    m.train_draw_model()
    m.run_backtesting_and_evaluation()
    pd.testing.assert_frame_equal(first, m.backtest_results_df)


def test_rating_updates_and_odds():
    m = EloSystem()
    home, away = m.update_elo(1500, 1500, 3, 0)
    assert home > 1500 > away
    assert home + away == pytest.approx(3000)
    assert m.calculate_odds(.5, 0) == 2
    assert all(np.isfinite(m.update_elo(5000, 1000, 0, 1)))
    with pytest.raises(ValueError):
        m.calculate_odds(.5, -1)
    with pytest.raises(ValueError):
        m.predict_match_probabilities_and_odds("unknown", "other")


@pytest.mark.parametrize("column,value", [
    ("HomeGoals", -1), ("HomeGoals", 1.5), ("HomeTeam", ""),
    ("Date", "invalid"), ("AwayGoals", None),
])
def test_invalid_data(tmp_path, column, value):
    df = games()
    df.loc[0, column] = value
    path = tmp_path / "matches.csv"
    df.to_csv(path, index=False)
    with pytest.raises((ValueError, TypeError)):
        EloSystem().load_and_prepare_data(path)


def test_unplayed_and_duplicate_fixtures(tmp_path):
    df = games()
    df.loc[0, ["HomeGoals", "AwayGoals"]] = np.nan
    path = tmp_path / "matches.csv"
    df.to_csv(path, index=False)
    assert len(EloSystem().load_and_prepare_data(path)) == 31
    pd.concat([df, df.iloc[[1]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        EloSystem().load_and_prepare_data(path)


def test_bundled_workbook():
    from pathlib import Path
    path = Path(__file__).parents[1] / "data/FCF_Futsal_BCN_Gr10_2025_26_DB.xlsx"
    m = EloSystem()
    df = m.load_and_prepare_data(path)
    assert len(df) > 0
    assert df.Date.is_monotonic_increasing
    result = evaluate(df)
    assert result.evaluation_metrics["n_matches"] == len(df)
