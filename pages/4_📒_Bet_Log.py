"""
Bet Log — the proof layer.

Log every bet, capture closing lines, settle results, and see whether the model actually
works: ROI, CLV (did you beat the closing line?), and a calibration curve (do your 60%s
hit 60%?). This is the evidence a subscriber pays for and a pick-seller can't fake.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import betlog as B

st.set_page_config(page_title="Bet Log", page_icon="📒", layout="wide")
st.title("📒 Bet Log — proof layer")
st.caption("Track CLV, ROI, and calibration. The record that proves the model works.")

st.info("Bets are stored locally in `data/bets.db` (SQLite). This persists on your machine. "
        "On Streamlit Community Cloud the file resets on reboot — swap to Postgres/Supabase "
        "for durable cloud storage when you're ready (all SQL is isolated in betlog.py).",
        icon="💾")

MARKETS = ["Batter HR", "Batter Total Bases", "Batter Total Hits", "Batter Strikeouts",
           "Pitcher Strikeouts", "Pitcher Outs", "Pitcher Walks", "Other"]

# --- Log a bet --------------------------------------------------------------
with st.expander("➕ Log a bet", expanded=False):
    with st.form("log_bet", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            d = st.date_input("Slate date", datetime.now())
            game = st.text_input("Game", placeholder="HOU @ DET")
            player = st.text_input("Player", placeholder="Jose Altuve")
        with c2:
            market = st.selectbox("Market", MARKETS)
            side = st.selectbox("Side", ["Over", "Under", "Yes"])
            line = st.number_input("Line", value=1.5, step=0.5)
        with c3:
            entry_odds = st.number_input("Entry odds (American)", value=-110, step=5)
            model_prob = st.number_input("Model prob", min_value=0.0, max_value=1.0, value=0.55, step=0.01,
                                         help="The model's probability for this side, from the Edge Board.")
            stake = st.number_input("Stake ($)", min_value=0.0, value=2.50, step=0.5)
        book = st.text_input("Book", placeholder="fanduel")
        notes = st.text_input("Notes", placeholder="optional")
        if st.form_submit_button("Log bet", type="primary"):
            if player and game:
                B.add_bet(slate_date=d.isoformat(), game=game, player=player, market=market,
                          side=side, line=line, entry_odds=int(entry_odds), model_prob=model_prob,
                          stake=stake, book=book, notes=notes)
                st.success(f"Logged: {player} {market} {side} {line}")
            else:
                st.warning("Player and game are required.")

bets = B.list_bets()
if not bets:
    st.info("No bets logged yet. Use **Log a bet** above to start your record.")
    st.stop()

s = B.summary(bets)

# --- Summary ----------------------------------------------------------------
st.subheader("Performance")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Record", f"{s['wins']}-{s['losses']}", help=f"{s['open']} open, {s['settled']} settled")
m2.metric("Profit", f"${s['profit']:,.2f}")
m3.metric("ROI", f"{s['roi']:+.1f}%" if s["roi"] is not None else "—")
m4.metric("Avg CLV", f"{s['avg_clv']:+.2f}%" if s["avg_clv"] is not None else "—",
          help=f"Over {s['clv_n']} bets with a closing line recorded.")
m5.metric("Beat-close rate", f"{s['beat_close_rate']:.0f}%" if s["beat_close_rate"] is not None else "—",
          help="Share of bets where you got a better price than the close. >50% is the goal.")

if s["clv_n"] == 0:
    st.caption("💡 Enter **closing odds** when you settle bets below to unlock CLV — it's the "
               "fastest signal that you're beating the market, long before ROI stabilizes.")

# --- Open bets: settle ------------------------------------------------------
open_bets = [b for b in bets if not b.get("result")]
if open_bets:
    st.subheader(f"Open bets ({len(open_bets)}) — enter closing odds & result, then save")
    odf = pd.DataFrame(open_bets)[["id", "player", "market", "side", "line", "entry_odds",
                                   "model_prob", "stake", "close_odds", "result"]]
    edited = st.data_editor(
        odf, hide_index=True, use_container_width=True, key="settle_editor",
        disabled=["id", "player", "market", "side", "line", "entry_odds", "model_prob", "stake"],
        column_config={
            "close_odds": st.column_config.NumberColumn("Closing odds", help="The price at game time / close."),
            "result": st.column_config.SelectboxColumn("Result", options=["", "win", "loss", "push", "void"]),
            "model_prob": st.column_config.NumberColumn("Model %", format="%.2f"),
        },
    )
    if st.button("💾 Save settlements", type="primary"):
        n = 0
        for _, r in edited.iterrows():
            co = None if pd.isna(r["close_odds"]) else int(r["close_odds"])
            res = r["result"] or None
            B.update_bet(int(r["id"]), close_odds=co, result=res)
            n += 1
        st.success(f"Saved {n} bet(s).")
        st.rerun()

# --- Calibration ------------------------------------------------------------
st.subheader("Calibration — do your probabilities tell the truth?")
cal = B.calibration(bets, n_bins=5)
settled_n = s["settled"]
if settled_n < 20:
    st.caption(f"Only {settled_n} settled bets so far. Calibration needs volume to mean anything "
               "— aim for 50+ before reading much into the curve.")
if cal:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    xs = [c["predicted"] for c in cal]
    ys = [c["actual"] for c in cal]
    ns = [c["n"] for c in cal]
    ax.scatter(xs, ys, s=[max(40, n * 12) for n in ns], color="#2563eb", alpha=0.75, zorder=3)
    for c in cal:
        ax.annotate(f"n={c['n']}", (c["predicted"], c["actual"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlabel("Model predicted probability")
    ax.set_ylabel("Actual win rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Reliability curve")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    st.caption("Points on the dashed line = well-calibrated. Points BELOW = overconfident "
               "(your 70%s aren't hitting 70%). Points ABOVE = underconfident. This is the chart "
               "that catches a model lying to you — like a phantom 90% that never cashes.")
else:
    st.caption("No settled bets yet — settle some above to build the calibration curve.")

# --- Full ledger ------------------------------------------------------------
st.divider()
st.subheader("Ledger")
df = pd.DataFrame(bets)
df["CLV%"] = df.apply(lambda r: B.clv_pct(r.get("entry_odds"), r.get("close_odds")), axis=1)
df["P&L"] = df.apply(lambda r: B.bet_pnl(r), axis=1)
cols = ["slate_date", "game", "player", "market", "side", "line", "entry_odds", "model_prob",
        "stake", "book", "close_odds", "CLV%", "result", "P&L"]
show = df[[c for c in cols if c in df.columns]]
st.dataframe(
    show.style.format({"model_prob": "{:.2f}", "CLV%": "{:+.1f}", "P&L": "${:+.2f}"}, na_rep="—")
    .background_gradient(cmap="RdYlGn", subset=["CLV%"]),
    use_container_width=True, hide_index=True)

with st.expander("Why CLV is the metric that matters"):
    st.markdown(
        """
**Closing Line Value (CLV)** is how much better your price was than the line's *close*. If
you bet a prop at +120 and it closes at +100, you beat the close — positive CLV.

It matters because the **closing line is the market's most accurate estimate**, sharpened by
all the money bet right up to game time. Consistently beating it is the clearest evidence you
have a real edge — and unlike ROI, which is buried in variance and takes a full season to
trust, CLV shows up in **weeks**. A bettor with positive long-run CLV is almost always a
long-run winner, even through cold streaks.

So the order of proof is: **beat-close rate > 50% and positive avg CLV first** (you're getting
good numbers), then **calibration** (your probabilities are honest), then **ROI** (the money
follows). Track CLV from bet #1.
"""
    )
