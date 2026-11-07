"""Local interactive demo: streamlit run app.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
import pandas as pd
import streamlit as st
from elo_system import EloSystem

st.set_page_config(page_title="Futsal Elo Lab", page_icon="⚽", layout="wide")
st.title("Futsal Elo Lab")
st.caption("FCF Barcelona Group 10 · chronological evaluation · explainable ratings")

@st.cache_resource
def build():
    model = EloSystem()
    frame = model.load_and_prepare_data(
        Path(__file__).parent / "data/FCF_Futsal_BCN_Gr10_2025_26_DB.xlsx")
    model.initialize_teams(frame)
    model.run_backtesting_and_evaluation()
    model.train_draw_model()
    return model

model = build()
st.info("Historical snapshot. Forecasts do not include injuries, lineups or live results.")
left, right = st.columns(2)
teams = sorted(model.elo_ratings)
home = left.selectbox("Home team", teams)
away = right.selectbox("Away team", [t for t in teams if t != home])
forecast = model.predict_match_probabilities_and_odds(home, away, overround=0)
for col, label, probability in zip(st.columns(3), ["Home win", "Draw", "Away win"], forecast["probs"]):
    col.metric(label, f"{probability:.1%}")
st.subheader("Final ratings")
st.dataframe(pd.DataFrame(sorted(model.elo_ratings.items(), key=lambda x: -x[1]),
                         columns=["Team", "Elo"]), hide_index=True, use_container_width=True)
st.subheader("Walk-forward evaluation")
st.json(model.evaluation_metrics)
st.caption("Lower log loss and Brier scores are better. Baselines use only earlier dates.")
st.line_chart(model.elo_history_df.pivot(index="Step", columns="Team", values="Elo").ffill())
st.download_button("Download backtest", model.backtest_results_df.to_csv(index=False),
                   "fixture_predictions.csv", "text/csv")
