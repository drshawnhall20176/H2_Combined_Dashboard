"""
Orchestration + CLI.

Pulls today's slate, builds candidate prop legs for every scheduled game, and
selects a N-leg parlay (default 7) from the highest-confidence legs.

Lineup handling (as specified):
  * If a team's batting order is already posted, use it (batting-spot -> expected PA).
  * If not, fall back to the active roster: take the top-9 position players by season
    plate appearances as projected starters, with a default PA. Re-run later once the
    real lineup posts to refine.
  * Pitchers always come from the listed probable starters.

Usage:
  python parlay.py                       # today's slate, 7 legs
  python parlay.py --date 2026-06-26 --legs 7
  python parlay.py --one-per-game --min-prob 0.6
  python parlay.py --recent-form         # blend last-15-game batter form (slower)
  python parlay.py --output json > picks.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from typing import Dict, List, Optional

import numpy as np

import config
from mlb_data import MLBStatsClient, to_float
import projections as P


# market metadata: (config key, human label, default side handling)
BATTER_MARKETS = [
    ("batter_hr", "Batter HR (anytime)"),
    ("batter_total_bases", "Batter Total Bases"),
    ("batter_hits", "Batter Total Hits"),
    ("batter_strikeouts", "Batter Strikeouts"),
]
PITCHER_MARKETS = [
    ("pitcher_strikeouts", "Pitcher Strikeouts"),
    ("pitcher_outs", "Pitcher Outs"),
    ("pitcher_walks", "Pitcher Walks"),
]


def best_side_leg(samples: np.ndarray, line: float, **meta) -> P.Leg:
    ou = P.over_under(samples, line)
    side = "over" if ou["over"] >= ou["under"] else "under"
    return P.Leg(side=side, prob=ou[side], line=line, **meta)


def build_batter_legs(client, game_str, team_name, batters, park, season, sims, rng, recent_form):
    legs: List[P.Leg] = []
    for b in batters:
        pid, name, exp_pa = b["id"], b["name"], b["pa"]
        if pid is None:
            continue
        stat = client.get_season_hitting(pid, season)
        if not stat:
            continue
        recent = None
        if recent_form:
            recent = _recent_batter_form(client, pid, season)
        probs = P.batter_pa_probabilities(stat, park, recent)
        if probs is None:
            continue
        sim = P.simulate_batter(probs, exp_pa, sims, rng)
        note = f"~{exp_pa:.1f} PA{' (proj)' if b.get('projected') else ''}"

        # HR: anytime (yes side only)
        legs.append(P.Leg(game=game_str, team=team_name, player=name,
                          market="Batter HR (anytime)", line=None, side="yes",
                          prob=P.at_least_one(sim["hr"]), detail=note))
        legs.append(best_side_leg(sim["tb"], config.DEFAULT_LINES["batter_total_bases"],
                                  game=game_str, team=team_name, player=name,
                                  market="Batter Total Bases", detail=note))
        legs.append(best_side_leg(sim["hits"], config.DEFAULT_LINES["batter_hits"],
                                  game=game_str, team=team_name, player=name,
                                  market="Batter Total Hits", detail=note))
        legs.append(best_side_leg(sim["k"], config.DEFAULT_LINES["batter_strikeouts"],
                                  game=game_str, team=team_name, player=name,
                                  market="Batter Strikeouts", detail=note))
    return legs


def build_pitcher_legs(client, game_str, team_name, pitcher, season, sims, rng):
    legs: List[P.Leg] = []
    if not pitcher or pitcher.get("id") is None:
        return legs
    pid, name = pitcher["id"], pitcher["name"]
    stat = client.get_season_pitching(pid, season)
    if not stat:
        return legs
    logs = client.get_game_log(pid, season, "pitching")
    proj = P.project_pitcher(stat, logs)
    if not proj:
        return legs
    sim = P.simulate_pitcher(proj, sims, rng)
    note = f"~{proj['exp_ip']:.1f} IP, {proj['exp_k']:.1f}K proj"

    legs.append(best_side_leg(sim["k"], config.DEFAULT_LINES["pitcher_strikeouts"],
                              game=game_str, team=team_name, player=name,
                              market="Pitcher Strikeouts", detail=note))
    legs.append(best_side_leg(sim["outs"], config.DEFAULT_LINES["pitcher_outs"],
                              game=game_str, team=team_name, player=name,
                              market="Pitcher Outs", detail=note))
    legs.append(best_side_leg(sim["bb"], config.DEFAULT_LINES["pitcher_walks"],
                              game=game_str, team=team_name, player=name,
                              market="Pitcher Walks", detail=note))
    return legs


def _recent_batter_form(client, pid, season) -> Optional[Dict]:
    logs = client.get_game_log(pid, season, "hitting")[: config.RECENT_GAMES_N]
    if not logs:
        return None
    pa = sum(to_float(g.get("plateAppearances")) for g in logs)
    hits = sum(to_float(g.get("hits")) for g in logs)
    hr = sum(to_float(g.get("homeRuns")) for g in logs)
    return {"pa": pa, "hits": hits, "hr": hr}


def resolve_batters(client, side, season) -> List[Dict]:
    """Return projected batters with expected PA, using the posted lineup if available,
    else the active-roster top-9-by-PA heuristic."""
    lineup = side.get("lineup") or []
    if lineup:
        out = []
        for i, p in enumerate(lineup[:9]):
            pa = config.LINEUP_SPOT_PA[i] if i < len(config.LINEUP_SPOT_PA) else config.DEFAULT_UNKNOWN_PA
            out.append({"id": p["id"], "name": p["name"], "pa": pa, "projected": False})
        return out

    # Fallback: active roster, position players only, ranked by season PA.
    team_id = side.get("team_id")
    if team_id is None:
        return []
    roster = client.get_active_roster(team_id)
    candidates = []
    for r in roster:
        if r.get("position") in ("P",):  # exclude pure pitchers (two-way players kept)
            continue
        stat = client.get_season_hitting(r["id"], season)
        pa = to_float(stat.get("plateAppearances")) if stat else 0.0
        candidates.append({"id": r["id"], "name": r["name"], "season_pa": pa})
    candidates.sort(key=lambda x: x["season_pa"], reverse=True)
    top = candidates[: config.PROJECTED_STARTERS_PER_TEAM]
    return [{"id": c["id"], "name": c["name"], "pa": config.DEFAULT_UNKNOWN_PA, "projected": True} for c in top]


def generate_all_legs(client, games, season, sims, rng, recent_form) -> List[P.Leg]:
    legs: List[P.Leg] = []
    for g in games:
        if g.get("gamePk") is None:
            continue
        away, home = g["away"], g["home"]
        game_str = f"{away['team_name']} @ {home['team_name']}"
        park = config.PARK_FACTORS.get(g.get("venue_id"), config.NEUTRAL_PARK)

        for side in (away, home):
            batters = resolve_batters(client, side, season)
            legs += build_batter_legs(client, game_str, side["team_name"], batters,
                                      park, season, sims, rng, recent_form)
            legs += build_pitcher_legs(client, game_str, side["team_name"],
                                       side.get("probable_pitcher"), season, sims, rng)
    return legs


def select_parlay(legs: List[P.Leg], n: int, min_prob: float,
                  one_per_game: bool, max_per_player: int) -> List[P.Leg]:
    legs = [l for l in legs if l.prob >= min_prob]
    legs.sort(key=lambda l: l.prob, reverse=True)
    picked: List[P.Leg] = []
    used_games, player_counts = set(), {}
    for leg in legs:
        if len(picked) >= n:
            break
        if one_per_game and leg.game in used_games:
            continue
        if player_counts.get(leg.player, 0) >= max_per_player:
            continue
        picked.append(leg)
        used_games.add(leg.game)
        player_counts[leg.player] = player_counts.get(leg.player, 0) + 1
    return picked


def format_line(leg: P.Leg) -> str:
    if leg.line is None:
        target = "anytime"
    else:
        target = f"{leg.side} {leg.line}"
    return target


def render_text(parlay: List[P.Leg], date_str: str) -> str:
    if not parlay:
        return "No legs met the criteria. Try lowering --min-prob or check the slate."
    lines = [f"\n7-LEG PARLAY  —  {date_str}", "=" * 64]
    combined = 1.0
    for i, leg in enumerate(parlay, 1):
        combined *= leg.prob
        target = format_line(leg)
        lines.append(
            f"{i}. {leg.player:<22} {leg.market:<22} {target:<12} "
            f"{leg.prob*100:5.1f}%   [{leg.game} | {leg.detail}]"
        )
    lines.append("=" * 64)
    fair_odds = (1.0 / combined) if combined > 0 else float("inf")
    lines.append(f"Model parlay hit probability: {combined*100:.2f}%  "
                 f"(fair decimal odds ~{fair_odds:.1f})")
    lines.append("Legs assumed independent; max one prop per player limits correlation.")
    lines.append("Default lines are placeholders — compare to real book lines for actual value.")
    return "\n".join(lines)


def to_json(parlay: List[P.Leg], date_str: str) -> str:
    combined = float(np.prod([l.prob for l in parlay])) if parlay else 0.0
    payload = {
        "date": date_str,
        "legs": [
            {"player": l.player, "team": l.team, "game": l.game, "market": l.market,
             "line": l.line, "side": l.side, "probability": round(l.prob, 4), "note": l.detail}
            for l in parlay
        ],
        "parlay_probability": round(combined, 6),
        "fair_decimal_odds": round(1.0 / combined, 3) if combined > 0 else None,
    }
    return json.dumps(payload, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate an N-leg MLB prop parlay from MLB Stats API data.")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="YYYY-MM-DD (default: today)")
    ap.add_argument("--legs", type=int, default=7)
    ap.add_argument("--season", type=int, default=config.default_season())
    ap.add_argument("--sims", type=int, default=config.DEFAULT_SIMS)
    ap.add_argument("--min-prob", type=float, default=0.55, help="discard legs below this model probability")
    ap.add_argument("--one-per-game", action="store_true", help="at most one leg per game (more decorrelated)")
    ap.add_argument("--max-per-player", type=int, default=1, help="cap props taken on a single player")
    ap.add_argument("--recent-form", action="store_true", help="blend recent batter form (extra API calls)")
    ap.add_argument("--output", choices=["text", "json"], default="text")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    client = MLBStatsClient()

    try:
        games = client.get_schedule(args.date)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to fetch schedule: {e}", file=sys.stderr)
        return 1

    if not games:
        print(f"No MLB games scheduled on {args.date}.")
        return 0

    print(f"Found {len(games)} game(s) on {args.date}. Building projections...", file=sys.stderr)
    legs = generate_all_legs(client, games, args.season, args.sims, rng, args.recent_form)
    parlay = select_parlay(legs, args.legs, args.min_prob, args.one_per_game, args.max_per_player)

    if args.output == "json":
        print(to_json(parlay, args.date))
    else:
        print(render_text(parlay, args.date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
