"""
Edge Board — the predictive + edge layer.

Two views from a single Monte Carlo pass:
  1. Model board: probabilities and fair prices for every prop (no odds needed).
  2. Live edges: when you fetch odds, the model is re-evaluated AT THE BOOK'S LINE,
     the price is de-vigged, and plays are ranked by EV%.

The API key is read from st.secrets / env — never hardcoded. Player props are quota-
expensive, so the live fetch is behind a button and cached.
"""

import os
from datetime import datetime

import pandas as pd
import pytz
import streamlit as st

import mlb_engine as E
import projections as P
import odds_api as O
import statcast_data as SC
import weather as WX

st.set_page_config(page_title="Edge Board", page_icon="📈", layout="wide")
st.title("📈 Edge Board")
st.caption("Model probabilities, fair prices, and live edges for every prop on the slate")

eastern = pytz.timezone("US/Eastern")

MARKET_LABEL = {
    "batter_home_runs": "Batter HR", "batter_total_bases": "Batter Total Bases",
    "batter_hits": "Batter Total Hits", "batter_strikeouts": "Batter Strikeouts",
    "pitcher_strikeouts": "Pitcher Strikeouts", "pitcher_outs": "Pitcher Outs",
    "pitcher_walks": "Pitcher Walks",
}


def get_api_key():
    try:
        return st.secrets["ODDS_API_KEY"]
    except Exception:
        return os.environ.get("ODDS_API_KEY")


@st.cache_data(ttl=3600, show_spinner=False)
def load_statcast():
    return SC.load()  # (lookup, k); ({}, None) if no cache file


@st.cache_data(ttl=1800, show_spinner=False)
def load_weather(meta_keys: tuple):
    out = {}
    for vid, gdate in meta_keys:
        if vid is not None and vid not in out:
            try:
                out[vid] = WX.get_game_weather(vid, gdate)
            except Exception:
                out[vid] = None
    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_index(date_str: str, fip_constant: float, sims: int, seed: int):
    rows, meta = E.build_slate(date_str, fip_constant)
    sc, k = load_statcast()
    wx = load_weather(tuple((m.get("venue_id"), m.get("game_date")) for m in meta))
    for r in rows:
        w = wx.get(r.get("_venue_id"))
        r["_weather_hr"] = w["hr_factor"] if w else 1.0   # temp + wind on HR, matches Dinger Engine
    # Statcast + weather attached -> HR probabilities here are consistent with the Dinger Engine.
    return P.build_projection_index(rows, meta, sims=sims, seed=seed, statcast=sc, statcast_k=k)


@st.cache_data(ttl=300, show_spinner=False)
def load_edges(date_str: str, markets_tuple: tuple, _index: dict, _api_key: str):
    offers, info = O.fetch_slate_props(date_str, _api_key, list(markets_tuple))
    edges, stats = O.compute_edges(_index, offers)
    return edges, {**info, **stats}


# --- controls ---------------------------------------------------------------
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    target_date = st.date_input("Slate date", datetime.now(eastern))
with c2:
    min_prob = st.slider("Min model prob (model board)", 0.50, 0.95, 0.60, 0.01)
with c3:
    st.write("")
    if st.button("🔄 Refresh slate"):
        st.cache_data.clear()
        st.rerun()

date_str = target_date.strftime("%Y-%m-%d")

with st.spinner("Projecting the slate..."):
    index = load_index(date_str, E.FIP_CONSTANT_DEFAULT, P.DEFAULT_SIMS, seed=7)

if not index:
    st.info("No projectable props for this date. Pick a date with scheduled MLB games.")
    st.stop()

board = pd.DataFrame(P.default_board_from_index(index))

# ============================================================================
# LIVE EDGES
# ============================================================================
st.subheader("💵 Live edges")
api_key = get_api_key()

if not api_key:
    st.warning(
        "No API key found. Create `.streamlit/secrets.toml` with "
        "`ODDS_API_KEY = \"your_key\"` (and add it to .gitignore), or set the "
        "`ODDS_API_KEY` environment variable. Then reload.",
        icon="🔑",
    )
else:
    ec1, ec2 = st.columns([3, 1])
    with ec1:
        chosen = st.multiselect(
            "Markets to price (each market × each game = 1 quota unit)",
            O.SUPPORTED_MARKETS, default=O.SUPPORTED_MARKETS,
            format_func=lambda k: MARKET_LABEL.get(k, k),
        )
    with ec2:
        min_ev = st.slider("Min EV%", -10.0, 30.0, 0.0, 0.5)

    n_games = len({v["ctx"]["game"] for v in index.values()})
    est_cost = len(chosen) * max(n_games, 1)
    st.caption(f"Estimated quota cost of a fetch: ~{est_cost} units "
               f"({len(chosen)} markets × ~{n_games} games). Cached for 5 min after fetching.")

    st.markdown("**Stake sizing (fractional Kelly)**")
    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        bankroll = st.number_input("Bankroll ($)", min_value=1.0, value=50.0, step=10.0)
    with kc2:
        frac_label = st.select_slider("Kelly fraction", options=["Quarter", "Half", "Full"],
                                      value="Quarter",
                                      help="Quarter-Kelly is the safe default — model probabilities "
                                           "are noisy, and full Kelly overbets when an edge is off.")
        kelly_frac = {"Quarter": 0.25, "Half": 0.5, "Full": 1.0}[frac_label]
    with kc3:
        cap_pct = st.slider("Max bet (% of bankroll)", 1, 25, 5,
                            help="Hard ceiling per bet — protects against a mis-estimated edge "
                                 "recommending a huge stake.") / 100.0

    if st.button("📡 Fetch live odds & compute edges", type="primary", disabled=not chosen):
        st.session_state["do_fetch"] = True

    if st.session_state.get("do_fetch"):
        try:
            with st.spinner("Fetching odds and computing edges..."):
                edges, info = load_edges(date_str, tuple(sorted(chosen)), index, api_key)
        except O.OddsAPIError as e:
            st.error(f"Odds API error: {e}")
            edges, info = [], {}

        if info:
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Quota remaining", info.get("remaining", "—"))
            q2.metric("Games priced", info.get("events_fetched", "—"))
            q3.metric("Props matched", info.get("matched", "—"))
            q4.metric("Unmatched (name/line)", info.get("unmatched", "—"))

        if edges:
            edf = pd.DataFrame(edges)
            edf = edf[edf["EV%"] >= min_ev].copy()
            edf["Market"] = edf["Market"].map(lambda k: MARKET_LABEL.get(k, k))
            # Recommended stake per bet (fractional Kelly, capped). Recomputes instantly when
            # you move the bankroll / fraction / cap controls — no re-fetch.
            edf["Stake $"] = edf.apply(
                lambda r: O.kelly_stake(r["ModelProb"], r["Price"], bankroll, kelly_frac, cap_pct), axis=1)
            edf["Stake %"] = edf["Stake $"] / bankroll

            total_stake = edf["Stake $"].sum()
            bets = int((edf["Stake $"] > 0).sum())
            s1, s2, s3 = st.columns(3)
            s1.metric("Recommended bets", bets)
            s2.metric("Total exposure", f"${total_stake:,.2f}")
            s3.metric("of bankroll", f"{(total_stake / bankroll * 100) if bankroll else 0:.0f}%")

            show = edf.rename(columns={"ModelProb": "Model %", "ImpliedBest": "Impl %",
                                       "NoVigMkt": "NoVig %", "EdgeVsMkt": "Edge", "Price": "Odds"})
            cols = ["Player", "Team", "Market", "Side", "Line", "Proj", "Model %",
                    "Book", "Odds", "EV%", "Stake $", "Stake %", "Game"]
            show = show[[c for c in cols if c in show.columns]]
            styler = (
                show.style
                .format({"Model %": "{:.1%}", "Proj": "{:.2f}", "Line": "{:.1f}",
                         "EV%": "{:+.1f}", "Stake $": "${:.2f}", "Stake %": "{:.1%}"})
                .background_gradient(cmap="RdYlGn", subset=["EV%"])
                .background_gradient(cmap="Blues", subset=["Stake $"])
            )
            st.dataframe(styler, use_container_width=True, hide_index=True, height=520)
            st.caption("Ranked by EV% at the best available price. Stake = fractional Kelly on your "
                       "bankroll, capped. EV% = model_prob × decimal payout − 1.")
        else:
            st.info("No edges to show (no props matched, or all below the EV filter).")

# ============================================================================
# MODEL BOARD (no odds needed)
# ============================================================================
st.divider()
st.subheader("🧮 Model board (no odds)")
st.caption("Model probabilities and fair prices at default lines. Use this to eyeball value "
           "manually, or before spending quota.")

view = board[board["ModelProb"] >= min_prob].sort_values("ModelProb", ascending=False)
disp = view.rename(columns={"ModelProb": "Model %", "Projection": "Proj",
                            "FairDec": "Fair (dec)", "FairAm": "Fair (am)"})
cols = ["Player", "Team", "Market", "Side", "Line", "Proj", "Model %", "Fair (dec)", "Fair (am)", "Opp", "Game"]
disp = disp[[c for c in cols if c in disp.columns]]
styler2 = (
    disp.style
    .format({"Model %": "{:.1%}", "Proj": "{:.2f}", "Line": "{:.1f}", "Fair (dec)": "{:.2f}"})
    .background_gradient(cmap="Greens", subset=["Model %"])
)
st.dataframe(styler2, use_container_width=True, hide_index=True, height=420)

with st.expander("How edge is computed (read me)"):
    st.markdown(
        """
1. **Model %** comes from a per-PA Monte Carlo (batters) or innings/Poisson model
   (pitchers), evaluated **at the book's actual line** — not a default — so it's
   comparable to the price.
2. **De-vig:** a book's Over and Under both carry juice. We convert each to an implied
   probability and normalize so they sum to 100% → the **NoVig %** (fair market prob).
3. **EV%** uses the *best* available price across books: `model_prob × decimal − 1`.
   Positive EV% means the price beats your fair value — that's the bet a trader takes.
4. **Edge vs market** = Model % − NoVig %. If this is large, you're disagreeing with the
   market — sometimes that's an edge, often it means the model is missing something
   (injury, weather, role change). Trust it only once calibration backs it up.
5. **Stake $** = fractional Kelly: `f* = (p·d − 1)/(d − 1)`, scaled by your chosen fraction
   and capped. Kelly is the bet size that maximizes long-run growth — but only if your
   probability is exact. Since it isn't, **quarter-Kelly with a hard cap** is the disciplined
   default: it captures most of the growth with far less risk of ruin when an edge is
   mis-estimated. Negative-EV bets get $0.

Line shopping matters: always bet the **best** price (the Book column), since EV swings
fast with the number. And remember: from a small bankroll, correct sizing means *small*
bets and slow, bumpy growth — that's the math, not a flaw.
"""
    )
