"""
statcast_data.py — Statcast expected-power layer for the Dinger Engine.

WHY: a hitter's actual HR count is noisy and luck-contaminated. His quality of contact
(barrel rate, exit velocity) is far more stable and predicts FUTURE power better. So a
hitter crushing the ball with a cold HR count is a positive-regression dinger bet the
market may be slow to price — the "buy the dip" logic, pointed at bats.

ARCHITECTURE: Savant pulls are slow/heavy, so we cache to disk nightly and the dashboard
reads the file instantly. Run `python refresh_statcast.py` (or statcast_data.refresh())
once a day; the app uses the last good file and never blocks on Savant.

This module imports pybaseball ONLY inside refresh(), so the dashboard can import and run
even if pybaseball isn't installed or Savant is unreachable — load() just returns empty
and projections fall back to the league prior. Nothing breaks without Statcast.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_PATH = os.path.join(DATA_DIR, "statcast_batters.csv")

LG_HR_PA = 0.033          # league HR per PA, the anchor for calibration
MIN_PA_QUALIFIED = 100    # PA floor for computing the league barrel->HR calibration


# --------------------------------------------------------------------- refresh
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def refresh(year: int, out_path: str = DEFAULT_PATH) -> str:
    """Pull Savant leaderboards via pybaseball, merge, and write a compact CSV to disk.

    Returns the path written. Run nightly. Column names occasionally drift between
    pybaseball versions, so this is defensive — verify the printed row count looks right.
    """
    from pybaseball import statcast_batter_exitvelo_barrels, statcast_batter_expected_stats

    ev = _norm_cols(statcast_batter_exitvelo_barrels(year))      # barrels / exit velo
    xs = _norm_cols(statcast_batter_expected_stats(year))        # xBA / xSLG / xwOBA

    ev = ev.rename(columns={"avg_hit_speed": "avg_ev", "ev95percent": "hardhit",
                            "brl_percent": "brl_pct"})
    merged = pd.merge(ev, xs, on="player_id", how="inner", suffixes=("", "_xs"))

    def col(df, name, default=0.0):
        return df[name] if name in df.columns else default

    name = (col(merged, "first_name", "").astype(str).str.strip() + " " +
            col(merged, "last_name", "").astype(str).str.strip()).str.strip()

    out = pd.DataFrame({
        "player_id": merged["player_id"].astype(int),
        "name": name,
        "pa": col(merged, "pa", 0),
        "brl_pa": col(merged, "brl_pa", 0.0),
        "brl_pct": col(merged, "brl_pct", 0.0),
        "hardhit": col(merged, "hardhit", 0.0),
        "avg_ev": col(merged, "avg_ev", 0.0),
        "slg": col(merged, "slg", 0.0),
        "xslg": col(merged, "est_slg", 0.0),
        "xiso": col(merged, "est_slg", 0.0) - col(merged, "est_ba", 0.0),
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} batters to {out_path}")
    return out_path


# ---------------------------------------------------------------------- load
def load(path: str = DEFAULT_PATH) -> Tuple[Dict[int, Dict], Optional[float]]:
    """Read the cached CSV. Returns (lookup_by_player_id, calibration_k).

    Returns ({}, None) if the file is missing — callers must treat Statcast as optional.
    calibration_k maps barrel rate to expected HR rate: xHR/PA = k * brl_pa, with k chosen
    so the league-average barrel rate maps to league-average HR rate."""
    if not os.path.exists(path):
        return {}, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}, None
    if df.empty or "player_id" not in df.columns or "brl_pa" not in df.columns:
        return {}, None

    qualified = df[df.get("pa", 0) >= MIN_PA_QUALIFIED]
    base = qualified if len(qualified) else df
    mean_brl = float(base["brl_pa"].mean()) if len(base) else 0.0
    k = (LG_HR_PA / mean_brl) if mean_brl > 0 else None

    lookup: Dict[int, Dict] = {}
    for r in df.itertuples(index=False):
        d = r._asdict()
        lookup[int(d["player_id"])] = {
            "name": d.get("name"),
            "pa": float(d.get("pa", 0) or 0),
            "brl_pa": float(d.get("brl_pa", 0) or 0),
            "brl_pct": float(d.get("brl_pct", 0) or 0),
            "hardhit": float(d.get("hardhit", 0) or 0),
            "avg_ev": float(d.get("avg_ev", 0) or 0),
            "xiso": float(d.get("xiso", 0) or 0),
            "slg": float(d.get("slg", 0) or 0),
            "xslg": float(d.get("xslg", 0) or 0),
        }
    return lookup, k


def expected_hr_rate(brl_pa: float, k: Optional[float]) -> Optional[float]:
    """Contact-implied HR per PA. Returns None if uncalibrated."""
    if k is None or brl_pa is None:
        return None
    return max(k * brl_pa, 0.0)
