"""
projections.py — turns the season stats the engine already fetches into real
probabilities for the seven prop markets, plus model "fair odds" you can hold up
against a sportsbook price.

Batters: build a per-plate-appearance outcome distribution (K / BB / out-in-play /
1B / 2B / 3B / HR) from season rates, adjust for park, and Monte-Carlo the game's
plate appearances. One simulation yields HR, total bases, hits, and strikeouts.

Pitchers: project expected innings -> batters faced, then K and BB as Poisson and
recorded outs as a clipped normal.

Pure NumPy. No network, no Streamlit. The engine calls build_signals() with data it
already has, so projections add zero API calls.

IMPORTANT — what this is and isn't:
  * These are MODEL probabilities, not market-calibrated truth, and not edges.
  * "Fair odds" = the price implied by the model probability. Edge only exists once
    you compare it to a real book line (next build step). Until then, treat fair odds
    as "the number you'd need to beat to have value."
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Dict, List, Optional

import numpy as np

# ---- per-PA outcome model --------------------------------------------------
OUTCOMES = ["out_play", "k", "bb", "single", "double", "triple", "hr"]
OUT_PLAY, K, BB, SINGLE, DOUBLE, TRIPLE, HR = range(7)
TB_VALUE = np.array([0, 0, 0, 1, 2, 3, 4], dtype=np.int64)
HIT_FLAG = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

# Expected plate appearances by batting-order index (0 = leadoff ... 8 = nine-hole).
LINEUP_SPOT_PA = [4.65, 4.55, 4.45, 4.35, 4.25, 4.10, 4.00, 3.90, 3.80]
DEFAULT_UNKNOWN_PA = 4.25

# Park factors by MLB venue id (hr / hits multipliers). Unlisted -> neutral.
PARK_FACTORS = {
    1: {"hr": 1.18, "hits": 1.04}, 2: {"hr": 0.95, "hits": 1.00}, 3: {"hr": 0.96, "hits": 1.08},
    4: {"hr": 1.10, "hits": 1.02}, 5: {"hr": 1.10, "hits": 1.03}, 7: {"hr": 1.30, "hits": 1.10},
    9: {"hr": 0.92, "hits": 0.96}, 12: {"hr": 1.02, "hits": 1.00}, 14: {"hr": 1.05, "hits": 1.00},
    15: {"hr": 1.08, "hits": 1.02}, 17: {"hr": 1.06, "hits": 1.02}, 19: {"hr": 0.98, "hits": 1.01},
    22: {"hr": 1.07, "hits": 1.01},
}
NEUTRAL_PARK = {"hr": 1.0, "hits": 1.0}

# Default lines (placeholders until a live odds feed supplies the real book line).
DEFAULT_LINES = {
    "Batter Total Bases": 1.5,
    "Batter Total Hits": 0.5,
    "Batter Strikeouts": 0.5,
    "Pitcher Strikeouts": 5.5,
    "Pitcher Outs": 17.5,
    "Pitcher Walks": 1.5,
}

DEFAULT_SIMS = 12000

# Maps our model markets to The Odds API market keys (verify against their docs;
# keys occasionally change). HR is just Over 0.5 on batter_home_runs.
ODDS_MARKET_KEYS = {
    "batter_home_runs": "hr",
    "batter_total_bases": "tb",
    "batter_hits": "hits",
    "batter_strikeouts": "bk",
    "pitcher_strikeouts": "pk",
    "pitcher_outs": "outs",
    "pitcher_walks": "pbb",
}


def _f(stat: Dict, key: str, default: float = 0.0) -> float:
    try:
        return float(stat.get(key, default))
    except (TypeError, ValueError):
        return default


def _parse_ip(v) -> float:
    s = str(v or "0")
    if "." not in s:
        try:
            return float(s)
        except ValueError:
            return 0.0
    whole, frac = s.split(".", 1)
    try:
        return float(whole) + {"0": 0, "1": 1, "2": 2}.get(frac[:1], 0) / 3.0
    except ValueError:
        return 0.0


# ---- odds helpers ----------------------------------------------------------
def prob_to_decimal(p: float) -> Optional[float]:
    return round(1.0 / p, 2) if p > 0 else None


def prob_to_american(p: float) -> Optional[int]:
    if p <= 0 or p >= 1:
        return None
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


# ---- regression to the mean (shrinkage) ------------------------------------
# Small samples lie. We pull every observed rate toward a league baseline by a weight
# tied to how much data backs it, so an 11-inning hot streak doesn't project like a skill.
# Per-PA league rates (approx. 2020s MLB) and per-stat "prior" weights (in PA / BF).
# Prior = the sample size at which observed and league get equal weight; bigger prior =
# more regression. Rates that stabilize slowly (HR, hits) get bigger priors than fast ones (K).
LG_BATTER = {  # rate, prior_pa
    "hr": (0.033, 170), "2b": (0.046, 140), "3b": (0.004, 120),
    "1b": (0.143, 140), "bb": (0.085, 110), "k": (0.225, 90),
}
LG_PITCHER = {  # rate per batter faced, prior_bf
    "k": (0.222, 150), "bb": (0.082, 350),
}


def _shrink(count: float, sample: float, lg_rate: float, prior: float) -> float:
    """Regress an observed rate toward league average. Returns a per-event probability."""
    return (count + lg_rate * prior) / (sample + prior) if (sample + prior) > 0 else lg_rate


# League per-PA rates as a flat lookup (for odds-ratio matchup math).
LG_RATE = {k: v[0] for k, v in LG_BATTER.items()}
LG_NONHR_HIT = LG_RATE["1b"] + LG_RATE["2b"] + LG_RATE["3b"]  # ~0.193

# Platoon splits stabilize slowly, so a vs-hand split is regressed toward the player's
# own (already league-stabilized) season rate using this prior, in PA.
SPLIT_PRIOR_PA = 150


# ---- odds-ratio (log5) matchup math ----------------------------------------
def _odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


def odds_ratio(p_bat: float, p_pit: float, p_lg: float) -> float:
    """Tango's odds-ratio method: combine a batter's rate, the pitcher's rate of ALLOWING
    that event, and league average into a single matchup-specific probability.
    p = OR_bat * OR_pit / OR_lg, converted back from odds."""
    if p_lg <= 0:
        return p_bat
    o = _odds(p_bat) * _odds(p_pit) / _odds(p_lg)
    return o / (1 + o)


# ---- batter model ----------------------------------------------------------
def pitcher_allowed_rates(stat: Optional[Dict]) -> Optional[Dict]:
    """Shrunk per-batter rates of what a pitcher ALLOWS, for the matchup math.
    Returns None for missing/thin pitchers so the batter falls back to neutral."""
    if not stat:
        return None
    bf = _f(stat, "battersFaced")
    if bf < 40:
        return None
    hr = _f(stat, "homeRuns"); so = _f(stat, "strikeOuts"); bb = _f(stat, "baseOnBalls")
    hits = _f(stat, "hits")
    nonhr_hit = max(hits - hr, 0.0)
    return {
        "hr": _shrink(hr, bf, LG_RATE["hr"], 220),
        "k": _shrink(so, bf, LG_RATE["k"], 150),
        "bb": _shrink(bb, bf, LG_RATE["bb"], 350),
        "nonhr_hit": _shrink(nonhr_hit, bf, LG_NONHR_HIT, 180),
    }


def _rates_from_stat(stat: Dict) -> Optional[Dict]:
    """Raw (unshrunk) per-PA component rates from a hitting stat dict."""
    pa = _f(stat, "plateAppearances")
    if pa <= 0:
        return None
    hits = _f(stat, "hits"); doubles = _f(stat, "doubles"); triples = _f(stat, "triples")
    hr = _f(stat, "homeRuns"); bb = _f(stat, "baseOnBalls"); so = _f(stat, "strikeOuts")
    singles = max(hits - doubles - triples - hr, 0.0)
    return {"pa": pa, "hr": hr, "2b": doubles, "3b": triples, "1b": singles, "bb": bb, "k": so}


def batter_base_rates(season_stat: Dict, split_stat: Optional[Dict] = None,
                      xhr_pa: Optional[float] = None) -> Optional[Dict]:
    """Per-PA outcome rates for a hitter: season regressed to league (or, for HR, toward
    the Statcast contact-implied rate when supplied), then the vs-hand split regressed
    toward that stabilized season rate."""
    s = _rates_from_stat(season_stat)
    if s is None or s["pa"] < 20:
        return None
    pa = s["pa"]
    base = {}
    for o in ("hr", "2b", "3b", "1b", "bb", "k"):
        lg_rate, prior = LG_BATTER[o]
        # For HR, regress toward the barrel-implied expected rate if we have it — a far
        # better prior than league average for that specific hitter.
        target = xhr_pa if (o == "hr" and xhr_pa is not None) else lg_rate
        base[o] = _shrink(s[o], pa, target, prior)

    sp = _rates_from_stat(split_stat) if split_stat else None
    if sp and sp["pa"] >= 20:
        spa = sp["pa"]
        for o in ("hr", "2b", "3b", "1b", "bb", "k"):
            base[o] = _shrink(sp[o], spa, base[o], SPLIT_PRIOR_PA)  # regress split toward season
    return base


def batter_pa_probs(season_stat: Dict, park: Dict, opp_allowed: Optional[Dict] = None,
                    split_stat: Optional[Dict] = None, xhr_pa: Optional[float] = None) -> Optional[np.ndarray]:
    """Per-PA outcome distribution: matchup-, platoon-, and (optionally) Statcast-aware.

    Order: stabilized base rates (handedness + barrel-implied HR) -> odds-ratio vs the
    pitcher -> park."""
    base = batter_base_rates(season_stat, split_stat, xhr_pa)
    if base is None:
        return None
    p_hr, p_3b, p_2b, p_1b = base["hr"], base["3b"], base["2b"], base["1b"]
    p_bb, p_k = base["bb"], base["k"]

    # Matchup: combine batter rate with the pitcher's allowed rate via odds-ratio.
    if opp_allowed:
        p_hr = odds_ratio(p_hr, opp_allowed["hr"], LG_RATE["hr"])
        p_k = odds_ratio(p_k, opp_allowed["k"], LG_RATE["k"])
        p_bb = odds_ratio(p_bb, opp_allowed["bb"], LG_RATE["bb"])
        nonhr = p_1b + p_2b + p_3b
        if nonhr > 0:
            adj = odds_ratio(nonhr, opp_allowed["nonhr_hit"], LG_NONHR_HIT)
            scale = adj / nonhr
            p_1b *= scale; p_2b *= scale; p_3b *= scale

    # Park adjustment.
    p_hr *= park.get("hr", 1.0)
    p_3b *= park.get("hits", 1.0); p_2b *= park.get("hits", 1.0); p_1b *= park.get("hits", 1.0)

    probs = np.array([0.0, p_k, p_bb, p_1b, p_2b, p_3b, p_hr], dtype=np.float64)
    if probs.sum() >= 1.0:
        probs = probs / probs.sum()
    probs[OUT_PLAY] = max(1.0 - probs.sum(), 0.0)
    return probs


def simulate_batter(probs: np.ndarray, exp_pa: float, sims: int, rng) -> Dict[str, np.ndarray]:
    base = int(np.floor(exp_pa))
    extra_p = exp_pa - base
    max_pa = base + 1
    draws = rng.choice(len(OUTCOMES), size=(sims, max_pa), p=probs)
    valid = np.ones((sims, max_pa), dtype=bool)
    valid[:, base:] = (rng.random(sims) < extra_p)[:, None]
    tb = np.where(valid, TB_VALUE[draws], 0).sum(axis=1)
    hits = np.where(valid, HIT_FLAG[draws], 0).sum(axis=1)
    hr = np.where(valid, (draws == HR), 0).sum(axis=1)
    k = np.where(valid, (draws == K), 0).sum(axis=1)
    return {"tb": tb, "hits": hits, "hr": hr, "k": k}


# ---- pitcher model ---------------------------------------------------------
def project_pitcher(stat: Dict) -> Optional[Dict]:
    """Project a STARTER's K / outs / walks. Returns None for non-starters or thin samples.

    Three guards against the inflation bug:
      1. Starter gate: needs real starts, else it's a bullpen game/opener -> skip.
      2. Shrinkage: K and BB rates regress toward league average by batters faced.
      3. Clamps: expected counts capped at realistic ceilings as a backstop.
    """
    bf = _f(stat, "battersFaced")
    ip = _parse_ip(stat.get("inningsPitched"))
    gs = _f(stat, "gamesStarted")
    so = _f(stat, "strikeOuts")
    bb = _f(stat, "baseOnBalls")

    # 1. Starter gate. A genuine probable starter has multiple starts and real innings.
    #    A reliever/opener listed as "probable" should not get starter-volume props.
    if gs < 3 or ip < 15 or bf < 60:
        return None

    # Expected innings from this pitcher's own start length, bounded to realistic range.
    exp_ip = float(np.clip(ip / gs, 3.0, 7.0))
    # Batters faced from a league-baseline rate (~4.3/IP), lightly nudged by the pitcher's
    # own baserunner tendency but kept in a sane band so noise can't blow it up.
    bf_per_ip = float(np.clip(bf / ip if ip > 0 else 4.3, 3.9, 4.7))
    exp_bf = exp_ip * bf_per_ip

    # 2. Shrinkage: regress per-batter K and BB rates toward league average.
    k_rate = _shrink(so, bf, *LG_PITCHER["k"])
    bb_rate = _shrink(bb, bf, *LG_PITCHER["bb"])

    # 3. Clamp expected counts to realistic ceilings (backstop against any residual noise).
    exp_k = float(min(k_rate * exp_bf, 0.45 * exp_bf))
    exp_bb = float(min(bb_rate * exp_bf, 0.25 * exp_bf))

    return {
        "exp_ip": exp_ip, "exp_outs": exp_ip * 3.0, "exp_bf": exp_bf,
        "exp_k": exp_k, "exp_bb": exp_bb,
    }


def simulate_pitcher(proj: Dict, sims: int, rng) -> Dict[str, np.ndarray]:
    k = rng.poisson(proj["exp_k"], size=sims)
    bb = rng.poisson(proj["exp_bb"], size=sims)
    sigma = max(3.0, proj["exp_outs"] * 0.22)
    outs = np.clip(np.round(rng.normal(proj["exp_outs"], sigma, size=sims)), 0, 27).astype(np.int64)
    return {"k": k, "bb": bb, "outs": outs}


# ---- signal assembly -------------------------------------------------------
def _signal(player, team, game, market, side, line, prob, projection, **extra) -> Dict:
    prob = float(round(prob, 4))
    sig = {
        "Player": player, "Team": team, "Game": game, "Market": market,
        "Side": side, "Line": line, "ModelProb": prob, "Projection": round(float(projection), 2),
        "FairDec": prob_to_decimal(prob), "FairAm": prob_to_american(prob),
        # placeholders the odds-feed step will fill:
        "BookOdds": None, "Implied": None, "EdgePct": None,
    }
    sig.update(extra)
    return sig


def _favored(samples: np.ndarray, line: float):
    over = float(np.mean(samples > line))
    return ("Over", over) if over >= 0.5 else ("Under", 1.0 - over)


def build_signals(rows: List[Dict], meta: List[Dict], sims: int = DEFAULT_SIMS,
                  seed: Optional[int] = None) -> List[Dict]:
    """Produce one signal per (player, market) from data the engine already fetched.

    `rows` must carry the private fields the engine attaches: _stat, _exp_pa, _venue_id.
    `meta` pitchers come from PitcherMetrics (with a .stat dict)."""
    rng = np.random.default_rng(seed)
    signals: List[Dict] = []

    # Batters
    for r in rows:
        stat = r.get("_stat")
        if not stat:
            continue
        park = PARK_FACTORS.get(r.get("_venue_id"), NEUTRAL_PARK)
        opp_allowed = pitcher_allowed_rates(r.get("_opp_stat"))
        probs = batter_pa_probs(stat, park, opp_allowed, r.get("_split_stat"))
        if probs is None:
            continue
        sim = simulate_batter(probs, r.get("_exp_pa", DEFAULT_UNKNOWN_PA), sims, rng)
        player, team, game = r["Hitter"], r["Team"], r["GameLabel"]

        hr_p = float(np.mean(sim["hr"] >= 1))
        signals.append(_signal(player, team, game, "Batter HR", "Yes", None, hr_p, sim["hr"].mean(),
                               Opp=r.get("Opp Pitcher"), Lineup=r.get("Lineup")))
        for market, arr in (("Batter Total Bases", sim["tb"]), ("Batter Total Hits", sim["hits"]),
                            ("Batter Strikeouts", sim["k"])):
            line = DEFAULT_LINES[market]
            side, p = _favored(arr, line)
            signals.append(_signal(player, team, game, market, side, line, p, arr.mean(),
                                   Opp=r.get("Opp Pitcher"), Lineup=r.get("Lineup")))

    # Pitchers
    for m in meta:
        for pm, team, opp in ((m["home_pm"], m["home_name"], m["away_name"]),
                              (m["away_pm"], m["away_name"], m["home_name"])):
            if pm.id is None or not pm.stat:
                continue
            proj = project_pitcher(pm.stat)
            if not proj:
                continue
            sim = simulate_pitcher(proj, sims, rng)
            for market, arr, key in (("Pitcher Strikeouts", sim["k"], "exp_k"),
                                     ("Pitcher Outs", sim["outs"], "exp_outs"),
                                     ("Pitcher Walks", sim["bb"], "exp_bb")):
                line = DEFAULT_LINES[market]
                side, p = _favored(arr, line)
                signals.append(_signal(pm.name, team, m["label"], market, side, line, p,
                                       proj[key], Opp=opp, Lineup="SP"))
    return signals


# ============================================================================
# Arbitrary-line evaluation for live-odds edge calculation
# ============================================================================
# The default-line board above answers "what does the model think?" To compute
# EDGE we must evaluate the model at the BOOK'S line, whatever it is. These build a
# compact discrete distribution per player+market so any half-line can be scored.

def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes so model names match book names."""
    s = unicodedata.normalize("NFD", str(name))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _dist(samples: np.ndarray) -> np.ndarray:
    """Normalized histogram: index i -> P(outcome == i)."""
    counts = np.bincount(samples.astype(np.int64)).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def prob_over(dist: np.ndarray, line: float) -> float:
    """P(X > line) for a half-line (e.g. 1.5 -> P(X >= 2))."""
    thresh = math.floor(line) + 1
    return float(dist[thresh:].sum()) if thresh < len(dist) else 0.0


def prob_for_side(dist: np.ndarray, line: float, side: str) -> float:
    over = prob_over(dist, line)
    return over if side.lower().startswith("o") else 1.0 - over


def build_projection_index(rows: List[Dict], meta: List[Dict],
                           sims: int = DEFAULT_SIMS, seed: Optional[int] = None,
                           statcast: Optional[Dict] = None, statcast_k: Optional[float] = None) -> Dict:
    """Return {(normalized_name, odds_market_key): {dist, mean, ctx}} for the slate.

    odds_market_key uses The Odds API names (batter_hits, pitcher_strikeouts, ...) so
    the odds matcher can join directly."""
    rng = np.random.default_rng(seed)
    index: Dict = {}

    for r in rows:
        stat = r.get("_stat")
        if not stat:
            continue
        park = PARK_FACTORS.get(r.get("_venue_id"), NEUTRAL_PARK)
        opp_allowed = pitcher_allowed_rates(r.get("_opp_stat"))
        xhr = xhr_from_statcast(r.get("_pid"), statcast, statcast_k)
        probs = batter_pa_probs(stat, park, opp_allowed, r.get("_split_stat"), xhr)
        if probs is None:
            continue
        sim = simulate_batter(probs, r.get("_exp_pa", DEFAULT_UNKNOWN_PA), sims, rng)
        nm = normalize_name(r["Hitter"])
        ctx = {"player": r["Hitter"], "team": r["Team"], "game": r["GameLabel"],
               "opp": r.get("Opp Pitcher"), "lineup": r.get("Lineup")}
        for key, arr in (("batter_home_runs", sim["hr"]), ("batter_total_bases", sim["tb"]),
                         ("batter_hits", sim["hits"]), ("batter_strikeouts", sim["k"])):
            index[(nm, key)] = {"dist": _dist(arr), "mean": float(arr.mean()), "ctx": ctx}

    for m in meta:
        for pm, team, opp in ((m["home_pm"], m["home_name"], m["away_name"]),
                              (m["away_pm"], m["away_name"], m["home_name"])):
            if pm.id is None or not pm.stat:
                continue
            proj = project_pitcher(pm.stat)
            if not proj:
                continue
            sim = simulate_pitcher(proj, sims, rng)
            nm = normalize_name(pm.name)
            ctx = {"player": pm.name, "team": team, "game": m["label"], "opp": opp, "lineup": "SP"}
            for key, arr, mean in (("pitcher_strikeouts", sim["k"], proj["exp_k"]),
                                   ("pitcher_outs", sim["outs"], proj["exp_outs"]),
                                   ("pitcher_walks", sim["bb"], proj["exp_bb"])):
                index[(nm, key)] = {"dist": _dist(arr), "mean": float(mean), "ctx": ctx}
    return index


# Display name + default line per Odds API market, for the model-only board.
_MARKET_DISPLAY = {
    "batter_home_runs": ("Batter HR", 0.5),
    "batter_total_bases": ("Batter Total Bases", 1.5),
    "batter_hits": ("Batter Total Hits", 0.5),
    "batter_strikeouts": ("Batter Strikeouts", 0.5),
    "pitcher_strikeouts": ("Pitcher Strikeouts", 5.5),
    "pitcher_outs": ("Pitcher Outs", 17.5),
    "pitcher_walks": ("Pitcher Walks", 1.5),
}


def default_board_from_index(index: Dict) -> List[Dict]:
    """Build the model-only board (favored side at default lines) from the index,
    so we run the Monte Carlo just once and reuse it for both views."""
    out: List[Dict] = []
    for (nm, mkey), entry in index.items():
        disp, line = _MARKET_DISPLAY.get(mkey, (mkey, 0.5))
        dist, ctx = entry["dist"], entry["ctx"]
        over = prob_over(dist, line)
        if mkey == "batter_home_runs":
            side, prob = "Yes", over
        else:
            side, prob = ("Over", over) if over >= 0.5 else ("Under", 1 - over)
        out.append(_signal(ctx["player"], ctx["team"], ctx["game"], disp, side,
                           None if mkey == "batter_home_runs" else line, prob, entry["mean"],
                           Opp=ctx.get("opp"), Lineup=ctx.get("lineup")))
    return out


def xhr_from_statcast(pid, statcast: Optional[Dict], k: Optional[float]) -> Optional[float]:
    """Contact-implied HR/PA for a player id, or None if unavailable."""
    if not statcast or k is None or pid is None:
        return None
    row = statcast.get(pid)
    if row is None:
        try:
            row = statcast.get(int(pid))
        except (TypeError, ValueError):
            row = None
    if not row:
        return None
    brl = row.get("brl_pa")
    return max(k * brl, 0.0) if brl is not None else None


def enrich_hitter_rows(rows: List[Dict], sims: int = DEFAULT_SIMS, seed: Optional[int] = None,
                       statcast: Optional[Dict] = None, statcast_k: Optional[float] = None) -> List[Dict]:
    """Attach matchup-aware model probabilities to each hitter row in place:
    HR%, Hit% (>=1), TB1.5% (>1.5 total bases), xK% (>=1 strikeout).

    When a Statcast lookup is supplied, HR regresses toward the barrel-implied rate and
    extra columns are added: Barrel%, xHR/PA, and Due (xHR minus actual HR rate = positive-
    regression dinger signal)."""
    rng = np.random.default_rng(seed)
    for r in rows:
        stat = r.get("_stat")
        if not stat:
            continue
        park = PARK_FACTORS.get(r.get("_venue_id"), NEUTRAL_PARK)
        opp_allowed = pitcher_allowed_rates(r.get("_opp_stat"))
        xhr = xhr_from_statcast(r.get("_pid"), statcast, statcast_k)
        probs = batter_pa_probs(stat, park, opp_allowed, r.get("_split_stat"), xhr)
        if probs is None:
            continue
        sim = simulate_batter(probs, r.get("_exp_pa", DEFAULT_UNKNOWN_PA), sims, rng)
        r["HR%"] = float(np.mean(sim["hr"] >= 1))
        r["Hit%"] = float(np.mean(sim["hits"] >= 1))
        r["TB1.5%"] = float(np.mean(sim["tb"] > 1.5))
        r["SO Prob"] = float(np.mean(sim["k"] >= 1))
        if xhr is not None:
            sc = statcast.get(r.get("_pid")) or {}
            actual_hr_pa = _f(stat, "homeRuns") / max(_f(stat, "plateAppearances"), 1)
            r["Barrel%"] = sc.get("brl_pct", 0.0)
            r["xHR/PA"] = xhr
            r["Due"] = xhr - actual_hr_pa   # positive = hitting better than HR results show
    return rows
