"""
Pitching Lab — ERA-vs-FIP regression PLUS matchup-aware starter projections.

The FIP table flags positive/negative regression candidates. The projections table shows
each starter's expected IP/K/BB/outs computed against the OPPOSING LINEUP (odds-ratio
matchup), so a strikeout arm vs a whiff-prone lineup projects higher than vs a contact team.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import pytz

import mlb_engine as E
import projections as P

st.set_page_config(page_title="Pitching Lab", page_icon="🎯", layout="wide")
st.title("🎯 Pitching Lab")
st.caption("ERA vs FIP regression and matchup-aware strikeout/innings projections")

eastern = pytz.timezone("US/Eastern")


def game_time_et(iso_utc):
    """ISO-UTC start -> '7:10 PM ET', or 'TBD' if missing."""
    if not iso_utc:
        return "TBD"
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(eastern)
        return dt.strftime("%I:%M %p").lstrip("0") + " ET"
    except (ValueError, TypeError):
        return "TBD"


@st.cache_data(ttl=600, show_spinner=False)
def load(date_str: str, fip_constant: float):
    rows, meta = E.build_slate(date_str, fip_constant)
    projections = P.build_pitcher_projection_rows(rows, meta, seed=11)
    # FIP regression table, rebuilt from the probable starters in meta.
    fip_rows = []
    for m in meta:
        for pm, team, opp in ((m["home_pm"], m["home_name"], m["away_name"]),
                              (m["away_pm"], m["away_name"], m["home_name"])):
            if pm.id is None or pm.era == 0:
                continue
            fip_rows.append({
                "Pitcher": pm.name, "Team": team, "Opponent": opp, "Hand": pm.hand,
                "ERA": round(pm.era, 2), "FIP": pm.fip, "Delta": round(pm.era - pm.fip, 2),
                "K/9": round(pm.k9, 1), "WHIP": round(pm.whip, 2), "HR/9": round(pm.hr9, 2), "OBA": pm.oba,
                "_game_date": m.get("game_date"),
            })
    return fip_rows, projections


col_a, col_b = st.columns([2, 1])
with col_a:
    target_date = st.date_input("Analysis Date", datetime.now())
with col_b:
    fip_constant = st.number_input("FIP constant", value=E.FIP_CONSTANT_DEFAULT,
                                   step=0.01, help="Season-specific; ~3.1-3.2.")

date_str = target_date.strftime("%Y-%m-%d")

with st.spinner("Loading starters and opposing lineups..."):
    fip_rows, proj_rows = load(date_str, fip_constant)

if not fip_rows:
    st.info("No probable starters found for this date. Pick a date with scheduled games.")
    st.stop()

df = pd.DataFrame(fip_rows)
df["Time"] = df["_game_date"].apply(game_time_et)

# === Matchup-aware projections =============================================
st.subheader("⚡ Matchup-aware starter projections")
st.caption("Expected line vs the opposing lineup. Proj K already accounts for how much that "
           "specific lineup strikes out — the same odds-ratio matchup used on the hitter side.")
if proj_rows:
    pdf = pd.DataFrame(proj_rows)
    pdf["Time"] = pdf["_game_date"].apply(game_time_et)
    sort_mode = st.radio("Sort", ["Chronological", "Projected K"], horizontal=True, key="proj_sort")
    if sort_mode == "Chronological":
        pdf = pdf.sort_values("_game_date", kind="stable", na_position="last")
    else:
        pdf = pdf.sort_values("Proj K", ascending=False, kind="stable")
    show = pdf.rename(columns={"K over%": "SO o5.5", "K fair": "SO fair"})
    cols = ["Time", "Pitcher", "Team", "Opp", "Hand", "Proj IP", "Proj K", "SO o5.5", "SO fair",
            "Proj BB", "Proj Outs", "ERA", "FIP"]
    show = show[[c for c in cols if c in show.columns]]
    st.dataframe(
        show.style.format({"SO o5.5": "{:.1%}", "Proj IP": "{:.1f}", "Proj K": "{:.1f}",
                           "Proj BB": "{:.1f}", "Proj Outs": "{:.1f}", "ERA": "{:.2f}", "FIP": "{:.2f}"})
        .background_gradient(cmap="RdYlGn", subset=["Proj K", "SO o5.5"]),
        use_container_width=True, hide_index=True, height=420)
else:
    st.write("No projectable starters (need 3+ starts of data).")

# === FIP regression ========================================================
st.divider()
st.subheader("📉 ERA vs FIP regression")
buys = df[df["Delta"] >= 0.50].sort_values("Delta", ascending=False)
fades = df[df["Delta"] <= -0.50].sort_values("Delta")
m1, m2, m3 = st.columns(3)
m1.metric("Probable starters", len(df))
m2.metric("Positive-regression (buy)", len(buys))
m3.metric("Negative-regression (fade)", len(fades))

fip_cols = ["Time", "Pitcher", "Team", "Opponent", "Hand", "ERA", "FIP", "Delta",
            "K/9", "WHIP", "HR/9", "OBA"]
styled = (
    df.sort_values("Delta", ascending=False)[fip_cols]
    .style.format({"ERA": "{:.2f}", "FIP": "{:.2f}", "Delta": "{:+.2f}",
                   "K/9": "{:.1f}", "WHIP": "{:.2f}", "HR/9": "{:.2f}", "OBA": "{:.3f}"})
    .background_gradient(cmap="RdYlGn", subset=["Delta", "K/9"])
    .background_gradient(cmap="RdYlGn_r", subset=["ERA", "FIP", "WHIP", "HR/9"])
)
st.dataframe(styled, use_container_width=True, hide_index=True)

# === Discussion hooks ======================================================
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

st.caption("Trends, not guarantees. FIP normalizes for defense/luck; projections assume the "
           "starter goes his typical length and the opposing lineup is roughly as posted.")
