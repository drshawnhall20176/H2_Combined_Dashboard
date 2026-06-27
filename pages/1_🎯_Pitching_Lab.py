"""
Pitching Lab — consolidates the old "Master Matchup Engine" and "Pitching Test V5"
pages into one. Replaces all mock data with live probable-starter stats and computes
real ERA-vs-FIP regression signals across the whole slate.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

import mlb_engine as E

st.set_page_config(page_title="Pitching Lab", page_icon="🎯", layout="wide")
st.title("🎯 Pitching Lab")
st.caption("Live ERA vs FIP regression across today's probable starters")


@st.cache_data(ttl=600, show_spinner=False)
def load_pitchers(date_str: str, fip_constant: float):
    return E.build_pitching_slate(date_str, fip_constant)


col_a, col_b = st.columns([2, 1])
with col_a:
    target_date = st.date_input("Analysis Date", datetime.now())
with col_b:
    fip_constant = st.number_input("FIP constant", value=E.FIP_CONSTANT_DEFAULT,
                                   step=0.01, help="Season-specific; ~3.1-3.2.")

date_str = target_date.strftime("%Y-%m-%d")

with st.spinner("Loading probable starters..."):
    rows = load_pitchers(date_str, fip_constant)

if not rows:
    st.info("No probable starters found for this date. Pick a date with scheduled games "
            "(probables are usually posted within a day or two of game time).")
    st.stop()

df = pd.DataFrame(rows)

# --- Regression signals -----------------------------------------------------
# Delta = ERA - FIP. Positive -> peripherals better than results (positive regression).
buys = df[df["Delta"] >= 0.50].sort_values("Delta", ascending=False)
fades = df[df["Delta"] <= -0.50].sort_values("Delta")

m1, m2, m3 = st.columns(3)
m1.metric("Probable starters", len(df))
m2.metric("Positive-regression (buy)", len(buys))
m3.metric("Negative-regression (fade)", len(fades))

st.subheader("All probable starters")
styled = (
    df.sort_values("Delta", ascending=False)
    .style.format({"ERA": "{:.2f}", "FIP": "{:.2f}", "Delta": "{:+.2f}",
                   "K/9": "{:.1f}", "WHIP": "{:.2f}", "HR/9": "{:.2f}", "OBA": "{:.3f}"})
    .background_gradient(cmap="RdYlGn", subset=["Delta", "K/9"])
    .background_gradient(cmap="RdYlGn_r", subset=["ERA", "FIP", "WHIP", "HR/9"])
)
st.dataframe(styled, use_container_width=True, hide_index=True)

# --- Discussion hooks -------------------------------------------------------
st.divider()
st.subheader("🤳 Discussion hooks (auto-generated)")
st.caption("Talking points where the underlying metrics diverge from the surface results.")
if buys.empty:
    st.write("No strong positive-regression candidates on this slate.")
for _, r in buys.head(5).iterrows():
    st.code(
        f"{r['Pitcher']} ({r['Team']}) carries a {r['ERA']:.2f} ERA but a {r['FIP']:.2f} FIP "
        f"— a {r['Delta']:+.2f} gap. The peripherals (K/9 {r['K/9']:.1f}, WHIP {r['WHIP']:.2f}) "
        f"suggest he's pitching better than the line shows. #MLB",
        language=None,
    )

st.caption("Trends, not guarantees. FIP normalizes for defense/luck but ignores park, "
           "opponent, and pitch-level data.")
