import streamlit as st

st.set_page_config(page_title="H2 Sports MLB Dashboard", page_icon="⚾", layout="wide")

st.title("⚾ H2 Sports — MLB Model Dashboard")
st.caption("Live matchup analytics powered by the public MLB Stats API")

st.markdown(
    """
This dashboard is built on a single shared backend (`mlb_engine.py`) so every page
pulls the same live data and stays consistent.

**Pages**
- **🎯 Pitching Lab** — probable starters across today's slate, ERA vs FIP regression,
  and auto-generated discussion hooks.
- **💣 Dinger Engine** — every projected hitter on the slate with platoon edges, ISO/OPS,
  and matchup leaderboards. Uses posted lineups when available, active rosters otherwise.

Select a page from the sidebar to begin. Pick a date, and the engine fetches the slate
concurrently — a full day usually loads in a few seconds.
"""
)

st.info(
    "Analytics here describe likelihoods and trends, not certainties. If you publish picks "
    "to an audience, note that gambling-content promotion is regulated in many regions "
    "(affiliate/advertising rules vary by jurisdiction)."
)
