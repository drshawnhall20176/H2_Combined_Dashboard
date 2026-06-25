import streamlit as st

st.set_page_config(
    page_title="H2 Sports Analytics Portal",
    page_icon="⚾",
    layout="wide"
)

st.title("H2 Sports Analytics Trading Station")
st.markdown("---")
st.subheader("Welcome to the Command Center")

st.markdown("""
Use the sidebar on the left to navigate between your tools:
* **Master Matchup Engine:** Live EV trading, regression profiles, and game physics.
* **Pitching Test V5:** Pitcher evaluation sandbox.
* **Inspect Columns:** Raw data validation and parsing.
""")