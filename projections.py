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
                    split_stat: Optional[Dict] = None, xhr_pa: Optional[float] = None,
                    weather_hr: float = 1.0) -> Optional[np.ndarray]:
    """Per-PA outcome distribution: matchup-, platoon-, Statcast-, and weather-aware.

    Order: stabilized base rates (handedness + barrel-implied HR) -> odds-ratio vs the
    pitcher -> park -> weather (temperature + wind on HR)."""
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

    # Park, then weather (temperature + out-to-CF wind act on home runs).
    p_hr *= park.get("hr", 1.0) * weather_hr
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
def lineup_k_bb_rates(stats_list: list) -> Optional[Dict]:
    """Aggregate a lineup's per-PA strikeout and walk rates (shrunk toward league).

    This is the symmetric input to the batter matchup: how often THIS lineup, as a group,
    strikes out and walks. Used to make pitcher K/BB projections matchup-aware."""
    tot_pa = tot_k = tot_bb = 0.0
    for s in stats_list:
        if not s:
            continue
        tot_pa += _f(s, "plateAppearances")
        tot_k += _f(s, "strikeOuts")
        tot_bb += _f(s, "baseOnBalls")
    if tot_pa < 200:  # not enough lineup data to trust
        return None
    return {
        "k": _shrink(tot_k, tot_pa, LG_RATE["k"], 300),
        "bb": _shrink(tot_bb, tot_pa, LG_RATE["bb"], 300),
    }


def build_lineup_rate_map(rows: list) -> Dict:
    """Map (game_label, team_name) -> that lineup's aggregate K/BB rates, for pitcher matchups."""
    groups: Dict = {}
    for r in rows:
        key = (r.get("GameLabel"), r.get("Team"))
        groups.setdefault(key, []).append(r.get("_stat"))
    return {k: lineup_k_bb_rates(v) for k, v in groups.items()}


def project_pitcher(stat: Dict, opp_lineup: Optional[Dict] = None) -> Optional[Dict]:
    """Project a STARTER's K / outs / walks. Returns None for non-starters or thin samples.

    Guards against the inflation bug:
      1. Starter gate: needs real starts, else it's a bullpen game/opener -> skip.
      2. Shrinkage: K and BB rates regress toward league average by batters faced.
      3. Clamps: expected counts capped at realistic ceilings as a backstop.

    When opp_lineup (the opposing lineup's K/BB rates) is supplied, K and BB are made
    matchup-aware via the odds-ratio method: a strikeout pitcher facing a whiff-prone
    lineup projects for more Ks; facing a contact lineup, fewer.
    """
    bf = _f(stat, "battersFaced")
    ip = _parse_ip(stat.get("inningsPitched"))
    gs = _f(stat, "gamesStarted")
    so = _f(stat, "strikeOuts")
    bb = _f(stat, "baseOnBalls")

    # 1. Starter gate. A genuine probable starter has multiple starts and real innings.
    if gs < 3 or ip < 15 or bf < 60:
        return None

    # Expected innings from this pitcher's own start length, bounded to realistic range.
    exp_ip = float(np.clip(ip / gs, 3.0, 7.0))
    bf_per_ip = float(np.clip(bf / ip if ip > 0 else 4.3, 3.9, 4.7))
    exp_bf = exp_ip * bf_per_ip

    # 2. Shrinkage: regress per-batter K and BB rates toward league average.
    k_rate = _shrink(so, bf, *LG_PITCHER["k"])
    bb_rate = _shrink(bb, bf, *LG_PITCHER["bb"])

    # 2b. Matchup: combine with the opposing lineup's rates via odds-ratio.
    if opp_lineup:
        k_rate = odds_ratio(k_rate, opp_lineup["k"], LG_RATE["k"])
        bb_rate = odds_ratio(bb_rate, opp_lineup["bb"], LG_RATE["bb"])

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

    # Pitchers (matchup-aware: each starter projected vs the opposing lineup's K/BB rates)
    lineup_map = build_lineup_rate_map(rows)
    for m in meta:
        for pm, team, opp in ((m["home_pm"], m["home_name"], m["away_name"]),
                              (m["away_pm"], m["away_name"], m["home_name"])):
            if pm.id is None or not pm.stat:
                continue
            proj = project_pitcher(pm.stat, lineup_map.get((m["label"], opp)))
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
        probs = batter_pa_probs(stat, park, opp_allowed, r.get("_split_stat"), xhr, r.get("_weather_hr", 1.0))
        if probs is None:
            continue
        sim = simulate_batter(probs, r.get("_exp_pa", DEFAULT_UNKNOWN_PA), sims, rng)
        nm = normalize_name(r["Hitter"])
        ctx = {"player": r["Hitter"], "team": r["Team"], "game": r["GameLabel"],
               "opp": r.get("Opp Pitcher"), "lineup": r.get("Lineup")}
        for key, arr in (("batter_home_runs", sim["hr"]), ("batter_total_bases", sim["tb"]),
                         ("batter_hits", sim["hits"]), ("batter_strikeouts", sim["k"])):
            index[(nm, key)] = {"dist": _dist(arr), "mean": float(arr.mean()), "ctx": ctx}

    lineup_map = build_lineup_rate_map(rows)
    for m in meta:
        for pm, team, opp in ((m["home_pm"], m["home_name"], m["away_name"]),
                              (m["away_pm"], m["away_name"], m["home_name"])):
            if pm.id is None or not pm.stat:
                continue
            proj = project_pitcher(pm.stat, lineup_map.get((m["label"], opp)))
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
    HR%, Hit% (>=1), TB1.5% (>1.5 total bases), SO Prob (>=1 strikeout).

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
        probs = batter_pa_probs(stat, park, opp_allowed, r.get("_split_stat"), xhr, r.get("_weather_hr", 1.0))
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


def build_pitcher_projection_rows(rows: List[Dict], meta: List[Dict],
                                  sims: int = DEFAULT_SIMS, seed: Optional[int] = None) -> List[Dict]:
    """Matchup-aware starter projections for the Pitching Lab: expected IP/K/BB/outs plus
    the strikeout-over probability and fair odds at the default line."""
    rng = np.random.default_rng(seed)
    lineup_map = build_lineup_rate_map(rows)
    out: List[Dict] = []
    for m in meta:
        for pm, team, opp in ((m["home_pm"], m["home_name"], m["away_name"]),
                              (m["away_pm"], m["away_name"], m["home_name"])):
            if pm.id is None or not pm.stat:
                continue
            opp_rates = lineup_map.get((m["label"], opp))
            proj = project_pitcher(pm.stat, opp_rates)
            if not proj:
                continue
            sim = simulate_pitcher(proj, sims, rng)
            k_line = DEFAULT_LINES["Pitcher Strikeouts"]
            k_over = float(np.mean(sim["k"] > k_line))
            outs_over = float(np.mean(sim["outs"] > DEFAULT_LINES["Pitcher Outs"]))
            bb_over = float(np.mean(sim["bb"] > DEFAULT_LINES["Pitcher Walks"]))
            out.append({
                "Pitcher": pm.name, "Team": team, "Opp": opp, "Hand": pm.hand,
                "ERA": round(pm.era, 2), "FIP": pm.fip,
                "Proj IP": round(proj["exp_ip"], 1), "Proj K": round(proj["exp_k"], 1),
                "Proj BB": round(proj["exp_bb"], 1), "Proj Outs": round(proj["exp_outs"], 1),
                "K line": k_line, "K over%": round(k_over, 4), "K fair": prob_to_american(k_over),
                "Outs over%": round(outs_over, 4), "BB over%": round(bb_over, 4),
                "_opp_k": (opp_rates or {}).get("k"), "_opp_bb": (opp_rates or {}).get("bb"),
                "_game": m["label"], "_pid": pm.id, "_game_date": m.get("game_date"),
            })
    out.sort(key=lambda r: r["Proj K"], reverse=True)
    return out


# ===========================================================================
# BEST BETS — cross-market synthesis with transparent reasoning
# ===========================================================================
# Typical single-game over-probability at the default line for each market. "Conviction"
# is the model's probability for the favored side divided by this reference, so a play
# scores high only when the model diverges from a typical prop of that type. This is a
# CONVICTION measure, not expected value — true value needs the live price (Edge Board).
BEST_BET_REF = {
    "Batter HR": 0.11, "Batter Total Bases": 0.42, "Batter Total Hits": 0.65,
    "Batter Strikeouts": 0.62, "Pitcher Strikeouts": 0.50, "Pitcher Outs": 0.50,
    "Pitcher Walks": 0.45,
}


def _favored_side(prob_over: float, ref: float):
    """Return (side, prob_of_that_side, ref_for_that_side)."""
    if prob_over >= ref:
        return "Over", prob_over, ref
    return "Under", 1.0 - prob_over, 1.0 - ref


def _hitter_reasons(r: Dict, market: str, side: str) -> List[str]:
    why = []
    offense = market in ("Batter HR", "Batter Total Bases", "Batter Total Hits")
    if side == "Over" and offense and r.get("Advantage") == "Advantage":
        why.append(f"platoon edge ({r.get('Hand')} bat vs {r.get('Opp Hand')}HP)")
    if market in ("Batter HR", "Batter Total Bases") and (r.get("_weather_hr") or 1.0) >= 1.05:
        why.append(f"weather aiding power (+{(r['_weather_hr'] - 1) * 100:.0f}%)")
    if market == "Batter HR" and (r.get("Due") or 0) > 0.01:
        why.append("barrels imply more power than the HR count shows")
    if market == "Batter Strikeouts":
        why.append("elevated whiff risk in this matchup" if side == "Over"
                   else "strong contact profile (rarely strikes out)")
    if not why:
        why.append(f"model leans {side} of a typical line here")
    return why


def _pitcher_reasons(r: Dict, market: str, side: str) -> List[str]:
    why, opp_k, opp_bb = [], r.get("_opp_k"), r.get("_opp_bb")
    if market == "Pitcher Strikeouts":
        if side == "Over":
            if opp_k and opp_k > 0.23:
                why.append(f"{r['Opp']} whiff-prone ({opp_k * 100:.0f}% K rate)")
            why.append(f"projects {r.get('Proj K')} K")
        else:
            if opp_k and opp_k < 0.20:
                why.append(f"{r['Opp']} tough to strike out ({opp_k * 100:.0f}% K rate)")
            why.append(f"projects only {r.get('Proj K')} K")
    elif market == "Pitcher Walks":
        if side == "Over" and opp_bb and opp_bb > 0.09:
            why.append(f"{r['Opp']} patient lineup ({opp_bb * 100:.0f}% walk rate)")
        why.append(f"projects {r.get('Proj BB')} BB")
    elif market == "Pitcher Outs":
        why.append(f"projects {r.get('Proj IP')} IP ({r.get('Proj Outs')} outs)")
    if not why:
        why.append(f"model leans {side} of a typical line here")
    return why


def build_best_bets(hitter_rows: List[Dict], pitcher_rows: List[Dict]) -> List[Dict]:
    """Rank model candidate plays across all markets by conviction (model prob vs the
    market-typical prob for that prop), each with transparent reasoning. No odds required.

    These are the model's strongest LEANS, not guaranteed value — check the live price on
    the Edge Board and let the proof layer (CLV/calibration) be the judge."""
    plays: List[Dict] = []

    batter_specs = [("Batter HR", "HR%", 0.5), ("Batter Total Bases", "TB1.5%", 1.5),
                    ("Batter Total Hits", "Hit%", 0.5), ("Batter Strikeouts", "SO Prob", 0.5)]
    for r in hitter_rows:
        for market, col, line in batter_specs:
            p = r.get(col)
            if p is None:
                continue
            side, sp, ref_s = _favored_side(p, BEST_BET_REF[market])
            if market == "Batter HR" and side == "Under":
                continue  # "won't homer" isn't a real play
            plays.append({
                "Player": r["Hitter"], "PlayerId": r.get("_pid"), "Team": r["Team"], "Game": r["GameLabel"],
                "Market": market, "Side": side, "Line": line,
                "ModelProb": round(sp, 4), "Fair": prob_to_american(sp),
                "Conviction": round(sp / ref_s, 2) if ref_s > 0 else 0.0,
                "Why": "; ".join(_hitter_reasons(r, market, side)),
            })

    pitcher_specs = [("Pitcher Strikeouts", "K over%", DEFAULT_LINES["Pitcher Strikeouts"]),
                     ("Pitcher Outs", "Outs over%", DEFAULT_LINES["Pitcher Outs"]),
                     ("Pitcher Walks", "BB over%", DEFAULT_LINES["Pitcher Walks"])]
    for r in pitcher_rows:
        for market, col, line in pitcher_specs:
            p = r.get(col)
            if p is None:
                continue
            side, sp, ref_s = _favored_side(p, BEST_BET_REF[market])
            plays.append({
                "Player": r["Pitcher"], "PlayerId": r.get("_pid"), "Team": r["Team"], "Game": r.get("_game", ""),
                "Market": market, "Side": side, "Line": line,
                "ModelProb": round(sp, 4), "Fair": prob_to_american(sp),
                "Conviction": round(sp / ref_s, 2) if ref_s > 0 else 0.0,
                "Why": "; ".join(_pitcher_reasons(r, market, side)),
            })

    plays.sort(key=lambda x: x["Conviction"], reverse=True)
    return plays
