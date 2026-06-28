"""
betlog.py — the proof layer's data store and analytics.

Records every bet, tracks Closing Line Value (CLV), settles results, and computes
calibration. This is what turns "I have a model" into "here's documented evidence it
beats the market."

STORAGE: SQLite at data/bets.db. Persists locally (where you'll log bets). On Streamlit
Community Cloud the filesystem is ephemeral, so for durable cloud storage swap the four
functions below the SCHEMA for a Postgres/Supabase backend — all SQL is isolated here,
so nothing else in the app changes.

CLV is the headline metric: did you get a better price than the line's CLOSE? It's the
best-known early predictor of long-term winning and shows up in weeks, not seasons.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

from odds_api import american_to_decimal

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "bets.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_placed  TEXT NOT NULL,
    slate_date TEXT,
    game       TEXT,
    player     TEXT,
    market     TEXT,
    side       TEXT,
    line       REAL,
    entry_odds INTEGER,
    model_prob REAL,
    stake      REAL,
    book       TEXT,
    close_odds INTEGER,
    result     TEXT,
    notes      TEXT
);
"""

_FIELDS = ["ts_placed", "slate_date", "game", "player", "market", "side", "line",
           "entry_odds", "model_prob", "stake", "book", "close_odds", "result", "notes"]


# ===========================================================================
# STORAGE (the only place that touches SQL — swap this block for Postgres later)
# ===========================================================================
@contextmanager
def _conn(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(_SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def add_bet(db_path: str = DB_PATH, **fields) -> int:
    fields.setdefault("ts_placed", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    vals = [fields.get(c) for c in _FIELDS]
    with _conn(db_path) as con:
        cur = con.execute(
            f"INSERT INTO bets ({','.join(_FIELDS)}) VALUES ({','.join('?' * len(_FIELDS))})", vals)
        return cur.lastrowid


def list_bets(db_path: str = DB_PATH, settled: Optional[bool] = None) -> List[Dict]:
    with _conn(db_path) as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM bets ORDER BY id DESC").fetchall()]
    if settled is True:
        rows = [b for b in rows if b.get("result")]
    elif settled is False:
        rows = [b for b in rows if not b.get("result")]
    return rows


def update_bet(bet_id: int, db_path: str = DB_PATH, **fields) -> None:
    fields = {k: v for k, v in fields.items() if k in _FIELDS}
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with _conn(db_path) as con:
        con.execute(f"UPDATE bets SET {sets} WHERE id=?", [*fields.values(), bet_id])


def delete_bet(bet_id: int, db_path: str = DB_PATH) -> None:
    with _conn(db_path) as con:
        con.execute("DELETE FROM bets WHERE id=?", [bet_id])


# ===========================================================================
# ANALYTICS (pure functions over lists of bet dicts — fully testable offline)
# ===========================================================================
def clv_pct(entry_odds, close_odds) -> Optional[float]:
    """Closing Line Value as a percent: how much better your price was than the close.

    (decimal_entry / decimal_close − 1) × 100. Positive = you beat the close."""
    if entry_odds is None or close_odds is None:
        return None
    return round((american_to_decimal(entry_odds) / american_to_decimal(close_odds) - 1) * 100, 2)


def bet_pnl(bet: Dict) -> Optional[float]:
    """Profit for a settled bet. Win pays net odds × stake; loss = −stake; push/void = 0."""
    result = (bet.get("result") or "").lower()
    stake = bet.get("stake") or 0.0
    odds = bet.get("entry_odds")
    if result == "win" and odds is not None:
        return round(stake * (american_to_decimal(odds) - 1), 2)
    if result == "loss":
        return round(-stake, 2)
    if result in ("push", "void"):
        return 0.0
    return None  # unsettled


def summary(bets: List[Dict]) -> Dict:
    settled = [b for b in bets if (b.get("result") or "").lower() in ("win", "loss")]
    wins = sum(1 for b in settled if (b["result"] or "").lower() == "win")
    losses = sum(1 for b in settled if (b["result"] or "").lower() == "loss")
    staked = sum(b.get("stake") or 0.0 for b in settled)
    profit = sum(bet_pnl(b) or 0.0 for b in settled)
    roi = (profit / staked * 100) if staked > 0 else None

    clvs = [clv_pct(b.get("entry_odds"), b.get("close_odds")) for b in bets]
    clvs = [c for c in clvs if c is not None]
    avg_clv = (sum(clvs) / len(clvs)) if clvs else None
    beat = (sum(1 for c in clvs if c > 0) / len(clvs) * 100) if clvs else None

    return {
        "n": len(bets), "settled": len(settled), "wins": wins, "losses": losses,
        "open": len(bets) - len(settled),
        "staked": round(staked, 2), "profit": round(profit, 2),
        "roi": round(roi, 2) if roi is not None else None,
        "clv_n": len(clvs),
        "avg_clv": round(avg_clv, 2) if avg_clv is not None else None,
        "beat_close_rate": round(beat, 1) if beat is not None else None,
    }


def calibration(bets: List[Dict], n_bins: int = 5) -> List[Dict]:
    """Bucket settled bets by model probability and compare predicted vs actual hit rate.

    Returns non-empty buckets: {lo, hi, predicted (mean model prob), actual (win rate), n}.
    Well-calibrated -> predicted ≈ actual in every bucket."""
    settled = [b for b in bets
               if (b.get("result") or "").lower() in ("win", "loss") and b.get("model_prob") is not None]
    if not settled:
        return []
    width = 1.0 / n_bins
    out = []
    for i in range(n_bins):
        lo, hi = i * width, (i + 1) * width
        # include 1.0 in the top bucket
        grp = [b for b in settled if (lo <= b["model_prob"] < hi) or (i == n_bins - 1 and b["model_prob"] == 1.0)]
        if not grp:
            continue
        predicted = sum(b["model_prob"] for b in grp) / len(grp)
        actual = sum(1 for b in grp if (b["result"] or "").lower() == "win") / len(grp)
        out.append({"lo": round(lo, 2), "hi": round(hi, 2),
                    "predicted": round(predicted, 3), "actual": round(actual, 3), "n": len(grp)})
    return out
