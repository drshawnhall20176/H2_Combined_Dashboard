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
import weather as WX
from datetime import datetime
import pytz
 
st.set_page_config(page_title="Dinger Engine", page_icon="💣", layout="wide")
st.title("💣 H2 Sports — Dinger Engine")
st.caption("Live hitter matchups, platoon edges, and power leaderboards")
 
 
@st.cache_data(ttl=3600, show_spinner=False)
def load_statcast():
    return SC.load()  # (lookup_by_player_id, calibration_k); ({}, None) if no cache file
 
 
@st.cache_data(ttl=1800, show_spinner=False)
def load_weather(meta_keys: tuple):
    """meta_keys: tuple of (venue_id, game_date, venue_name). Returns {venue_id: weather|None}."""
    out = {}
    for vid, gdate, vname in meta_keys:
        if vid is not None and vid not in out:
            try:
                out[vid] = WX.get_game_weather(vid, gdate, vname)
            except Exception:
                out[vid] = None
    return out
 
 
@st.cache_data(ttl=300, show_spinner=False)
def load_slate(date_str: str, fip_constant: float):
    rows, meta = E.build_slate(date_str, fip_constant)
    sc, k = load_statcast()
    wx_by_venue = load_weather(tuple((m.get("venue_id"), m.get("game_date"), m.get("venue")) for m in meta))
    for r in rows:
        wx = wx_by_venue.get(r.get("_venue_id"))
        r["_weather_hr"] = wx["hr_factor"] if wx else 1.0
    P.enrich_hitter_rows(rows, seed=7, statcast=sc, statcast_k=k)  # matchup/platoon/Statcast/weather
    return rows, meta, (len(sc) if sc else 0), wx_by_venue
 
 
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
    rows, meta, n_statcast, wx_by_venue = load_slate(date_str, fip_constant)
 
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
                "Opp HR/9", "HR%", "Hit%", "TB1.5%", "SO Prob", "Barrel%", "xHR/PA", "K%", "HR", "TB", "SLG", "OPS", "ISO", "PowerIndex"]
 
 
def hr9_band(v):
    """Fixed-threshold coloring for pitcher HR/9 (absolute, not slate-relative).
    <0.8 excellent · 0.8-1.1 solid · 1.1-1.3 average · 1.3-1.5 below avg · >1.5 homer-prone."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if x < 0.8:
        return "background-color:#1a9850;color:white"   # excellent (elite arm)
    if x < 1.1:
        return "background-color:#a6d96a"               # above average to solid
    if x < 1.3:
        return "background-color:#fee08b"               # average
    if x < 1.5:
        return "background-color:#fdae61"               # below average
    return "background-color:#d73027;color:white"       # bad / home-run prone
 
 
def style_hitters(data: pd.DataFrame):
    cols = [c for c in DISPLAY_COLS if c in data.columns]
    view = data[cols]
    pct = [c for c in ("HR%", "Hit%", "TB1.5%", "SO Prob", "K%", "Barrel%", "xHR/PA") if c in view.columns]
    fmt = {"HR": "{:.0f}", "TB": "{:.0f}", "SLG": "{:.3f}", "OPS": "{:.3f}",
           "ISO": "{:.3f}", "PowerIndex": "{:.1f}", "Opp HR/9": "{:.2f}"}
    fmt.update({c: "{:.1%}" for c in pct})
    styler = view.style.format(fmt)
    grad_up = [c for c in ("HR%", "Hit%", "TB1.5%", "HR", "TB", "SLG", "OPS", "ISO", "PowerIndex") if c in view.columns]
    if grad_up:
        styler = styler.background_gradient(cmap="RdYlGn", subset=grad_up)
    # Strikeouts hurt the hitter, so high = red on both the game prob and the season rate.
    red_high = [c for c in ("SO Prob", "K%") if c in view.columns]
    if red_high:
        styler = styler.background_gradient(cmap="RdYlGn_r", subset=red_high)
    # Opp HR/9 uses fixed bands (elite arm green -> homer-prone red), not a slate-relative gradient.
    if "Opp HR/9" in view.columns:
        styler = styler.apply(lambda s: [hr9_band(v) for v in s], subset=["Opp HR/9"])
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
        wx = wx_by_venue.get(m.get("venue_id"))
        if wx:
            if wx.get("dome"):
                st.markdown("🏟️ **Indoors** (fixed roof) — weather neutral")
            else:
                f = wx["hr_factor"]
                tag = f"🟢 +{(f - 1) * 100:.0f}% HR" if f > 1.02 else (
                    f"🔴 {(f - 1) * 100:.0f}% HR" if f < 0.98 else "⚪ neutral")
                approx = " · _wind orientation approximate_" if wx.get("approx_wind") else ""
                st.markdown(f"🌤️ **{wx['summary']}** → {tag}{approx}")
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
st.caption("Opp HR/9 = the opposing starter's home runs allowed per 9 innings, colored on fixed bands "
           "(not slate-relative): 🟢 under 0.80 excellent · 🟩 0.80–1.10 solid · 🟡 1.10–1.30 average · "
           "🟠 1.30–1.50 below average · 🔴 over 1.50 homer-prone. A redder arm is a better power spot "
           "for the hitter.")
