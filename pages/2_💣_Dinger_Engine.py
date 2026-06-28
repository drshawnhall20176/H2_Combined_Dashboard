"""
Dinger Engine — refactored from the original page 3.

Same idea (every projected hitter on the slate, platoon edges, matchup leaderboards),
but it runs on the shared concurrent backend: one hydrated request per hitter, per-team
lineup detection, and a real Confirmed/Projected badge. Loads a full slate in seconds.
"""

import streamlit as st
import pandas as pd

import mlb_engine as E
import projections as P
import statcast_data as SC
from datetime import datetime
import pytz

st.set_page_config(page_title="Dinger Engine", page_icon="💣", layout="wide")
st.title("💣 H2 Sports — Dinger Engine")
st.caption("Live hitter matchups, platoon edges, and power leaderboards")


@st.cache_data(ttl=3600, show_spinner=False)
def load_statcast():
    return SC.load()  # (lookup_by_player_id, calibration_k); ({}, None) if no cache file


@st.cache_data(ttl=300, show_spinner=False)
def load_slate(date_str: str, fip_constant: float):
    rows, meta = E.build_slate(date_str, fip_constant)
    sc, k = load_statcast()
    P.enrich_hitter_rows(rows, seed=7, statcast=sc, statcast_k=k)  # matchup/platoon/Statcast
    return rows, meta, (len(sc) if sc else 0)


eastern = pytz.timezone("US/Eastern")
default_date = datetime.now(eastern)

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    target_date = st.date_input("Slate date", default_date)
with c2:
    fip_constant = st.number_input("FIP constant", value=E.FIP_CONSTANT_DEFAULT, step=0.01)
with c3:
    st.write("")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

date_str = target_date.strftime("%Y-%m-%d")

with st.spinner("Compiling telemetry..."):
    rows, meta, n_statcast = load_slate(date_str, fip_constant)

if not rows:
    st.info("No hitters compiled for this date. Pick a date with scheduled MLB games.")
    st.stop()

df = pd.DataFrame(rows)

confirmed = (df["Lineup"] == "Confirmed").sum()
st.caption(f"{len(meta)} games · {len(df)} hitters · "
           f"{confirmed} from confirmed lineups, {len(df) - confirmed} projected from active rosters")
if n_statcast:
    st.caption(f"🟢 Statcast power model active ({n_statcast} batters) — HR regresses toward "
               f"barrel-implied expected rate.")
else:
    st.caption("⚪ Statcast model off — run `python refresh_statcast.py` to enable barrel-based "
               "HR regression and the 'Due to Homer' board.")


# --- Styling ----------------------------------------------------------------
DISPLAY_COLS = ["Hitter", "Team", "Hand", "Opp Pitcher", "Opp Hand", "Advantage", "Lineup",
                "HR%", "Hit%", "TB1.5%", "SO Prob", "Barrel%", "xHR/PA", "K%", "HR", "TB", "SLG", "OPS", "ISO", "PowerIndex"]


def style_hitters(data: pd.DataFrame):
    cols = [c for c in DISPLAY_COLS if c in data.columns]
    view = data[cols]
    pct = [c for c in ("HR%", "Hit%", "TB1.5%", "SO Prob", "K%", "Barrel%", "xHR/PA") if c in view.columns]
    fmt = {"HR": "{:.0f}", "TB": "{:.0f}", "SLG": "{:.3f}", "OPS": "{:.3f}",
           "ISO": "{:.3f}", "PowerIndex": "{:.1f}"}
    fmt.update({c: "{:.1%}" for c in pct})
    styler = view.style.format(fmt)
    grad_up = [c for c in ("HR%", "Hit%", "TB1.5%", "HR", "TB", "SLG", "OPS", "ISO", "PowerIndex") if c in view.columns]
    if grad_up:
        styler = styler.background_gradient(cmap="RdYlGn", subset=grad_up)
    # Strikeouts are bad for a hitter, so high = red on both the game prob and the season rate.
    k_cols = [c for c in ("SO Prob", "K%") if c in view.columns]
    if k_cols:
        styler = styler.background_gradient(cmap="RdYlGn_r", subset=k_cols)
    return styler


# --- Leaderboards -----------------------------------------------------------
st.subheader("Slate leaderboards")
lc1, lc2, lc3 = st.columns(3)
with lc1:
    st.markdown("**🎯 Top HR probability** (matchup-aware)")
    if "HR%" in df.columns:
        top_hr = df.nlargest(8, "HR%")[["Hitter", "Team", "Opp Pitcher", "HR%"]]
        st.dataframe(top_hr.style.format({"HR%": "{:.1%}"}), hide_index=True, use_container_width=True)
    else:
        st.dataframe(df.nlargest(8, "PowerIndex")[["Hitter", "Team", "Opp Pitcher", "PowerIndex"]],
                     hide_index=True, use_container_width=True)
with lc2:
    st.markdown("**Best total-bases plays**")
    if "TB1.5%" in df.columns:
        top_tb = df.nlargest(8, "TB1.5%")[["Hitter", "Team", "Opp Pitcher", "TB1.5%"]]
        st.dataframe(top_tb.style.format({"TB1.5%": "{:.1%}"}), hide_index=True, use_container_width=True)
with lc3:
    st.markdown("**Platoon-advantage bats**")
    sort_key = "HR%" if "HR%" in df.columns else "PowerIndex"
    adv = df[df["Advantage"] == "Advantage"].nlargest(8, sort_key)
    fmtcol = {sort_key: "{:.1%}"} if sort_key == "HR%" else {}
    st.dataframe(adv[["Hitter", "Team", "Hand", "Opp Hand", sort_key]].style.format(fmtcol),
                 hide_index=True, use_container_width=True)

# --- Statcast: due-to-homer regression candidates --------------------------
if "Due" in df.columns:
    st.markdown("**🔥 Due to homer** — biggest gap between barrel-implied power and actual HR results "
                "(positive = hitting the ball harder than the HR count shows)")
    due = df[df["Due"] > 0].nlargest(10, "Due")[
        ["Hitter", "Team", "Opp Pitcher", "Barrel%", "xHR/PA", "HR%", "Due"]]
    st.dataframe(
        due.style.format({"Barrel%": "{:.1%}", "xHR/PA": "{:.1%}", "HR%": "{:.1%}", "Due": "{:+.1%}"})
        .background_gradient(cmap="Oranges", subset=["Due"]),
        hide_index=True, use_container_width=True)

# --- Per-game detail --------------------------------------------------------
st.divider()
st.subheader("Game-by-game")


def game_time_et(iso_utc):
    """Format an ISO-UTC start time as local Eastern, e.g. '7:10 PM ET'. 'TBD' if missing."""
    if not iso_utc:
        return "TBD"
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(eastern)
        return dt.strftime("%I:%M %p").lstrip("0") + " ET"   # lstrip keeps it Windows-safe
    except (ValueError, TypeError):
        return "TBD"


# Chronological order: ISO-UTC strings sort by start time; games without a time go last.
meta_sorted = sorted(meta, key=lambda m: m.get("game_date") or "9999")

for m in meta_sorted:
    hp, ap = m["home_pm"], m["away_pm"]
    when = game_time_et(m.get("game_date"))
    badge = "" if (df[df["GameLabel"].str.startswith(m["label"].split(" (Game")[0])]["Lineup"] == "Confirmed").any() else " · projected lineups"
    with st.expander(f"🕒 {when}  ·  {m['label']}  —  {m['venue']}  ({m['status']}){badge}"):
        st.markdown(
            f"✈️ **{m['away_name']}** SP {ap.name}: K/9 {ap.k9:.1f} · ERA {ap.era:.2f} · "
            f"FIP {ap.fip:.2f} · WHIP {ap.whip:.2f}"
        )
        st.markdown(
            f"🏠 **{m['home_name']}** SP {hp.name}: K/9 {hp.k9:.1f} · ERA {hp.era:.2f} · "
            f"FIP {hp.fip:.2f} · WHIP {hp.whip:.2f}"
        )
        t_away, t_home = st.tabs([f"✈️ {m['away_name']} bats", f"🏠 {m['home_name']} bats"])
        game_df = df[df["GameLabel"] == m["label"]]
        sort_col = "HR%" if "HR%" in game_df.columns else "PowerIndex"
        with t_away:
            sub = game_df[game_df["Team"] == m["away_name"]].sort_values(sort_col, ascending=False)
            st.dataframe(style_hitters(sub), use_container_width=True, hide_index=True)
        with t_home:
            sub = game_df[game_df["Team"] == m["home_name"]].sort_values(sort_col, ascending=False)
            st.dataframe(style_hitters(sub), use_container_width=True, hide_index=True)

st.caption("HR% / Hit% / TB1.5% / SO Prob are matchup-aware model probabilities for TODAY's game: "
           "each hitter's stabilized rates are combined with the opposing pitcher's allowed rates "
           "(odds-ratio method) and his platoon split, then park-adjusted. K% is the hitter's SEASON "
           "strikeout rate (a skill stat) for reference. PowerIndex is the legacy heuristic.")
