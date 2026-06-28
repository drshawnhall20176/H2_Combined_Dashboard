"""
mlb_engine.py — shared data/analytics backend for the dashboard.

Framework-agnostic (no Streamlit import). Pages wrap the expensive calls with
@st.cache_data for caching/TTL. This module fixes the issues from the original pages:

  * one hydrated request per hitter (batSide + season stats together), not two
  * per-team lineup detection (posted batting order -> active roster fallback)
  * concurrent fetching so a full slate loads in a few seconds, not a minute
  * parameterized FIP constant, no dead imports, no bare excepts

Data source: public MLB Stats API (statsapi.mlb.com), no key required.
"""

from __future__ import annotations

import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 10

# League FIP constant. It is season-specific (lgERA minus the FIP numerator over lgIP),
# historically ~3.1-3.2. Override per season if you want exactness; see derive_fip_constant.
FIP_CONSTANT_DEFAULT = 3.17

# Expected plate appearances by batting-order spot (0 = leadoff). Mirrors projections.py.
LINEUP_SPOT_PA = [4.65, 4.55, 4.45, 4.35, 4.25, 4.10, 4.00, 3.90, 3.80]
DEFAULT_UNKNOWN_PA = 4.25

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "h2-mlb-dashboard/1.0"})


# --------------------------------------------------------------------- helpers
def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(text))
                   if unicodedata.category(c) != "Mn")


def fetch_json(url: str, params: Optional[Dict] = None, retries: int = 2) -> Dict[str, Any]:
    """GET JSON with a couple of retries. Returns {} on failure (never raises)."""
    for attempt in range(retries + 1):
        try:
            r = _SESSION.get(url, params=params or {}, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
    return {}


def parse_innings(ip_value: Any) -> float:
    """'85.1' -> 85.333 ('.1' = 1 out, '.2' = 2 outs)."""
    s = str(ip_value or "0")
    if "." not in s:
        return safe_float(s)
    whole, frac = s.split(".", 1)
    return safe_float(whole) + {"0": 0, "1": 1, "2": 2}.get(frac[:1], 0) / 3.0


# --------------------------------------------------------------------- analytics
def calculate_fip(stat: Dict[str, Any], constant: float = FIP_CONSTANT_DEFAULT) -> float:
    """FIP = ((13*HR + 3*(BB+HBP) - 2*K) / IP) + constant."""
    hr = safe_float(stat.get("homeRuns"))
    bb = safe_float(stat.get("baseOnBalls"))
    hbp = safe_float(stat.get("hitByPitch"))
    k = safe_float(stat.get("strikeOuts"))
    ip = parse_innings(stat.get("inningsPitched"))
    if ip <= 0:
        return 0.0
    return round(((13 * hr) + (3 * (bb + hbp)) - (2 * k)) / ip + constant, 2)


def platoon_advantage(bat_hand: str, pit_hand: str) -> str:
    """Switch hitters always hold the platoon edge; otherwise opposite hands = advantage."""
    if bat_hand == "S":
        return "Advantage"
    if not bat_hand or not pit_hand:
        return "Unknown"
    return "Advantage" if bat_hand != pit_hand else "Disadvantage"


def power_index(iso: float, ops: float, advantage: str) -> float:
    """Transparent, sortable heuristic for the matchup leaderboards.

    NOT a probability. It rewards isolated power and overall OPS, with a small platoon
    nudge. Replace with the projection model (per-PA Monte Carlo) when you want real
    prop probabilities instead of a ranking score.
    """
    base = 100.0 * (iso + max(ops - 0.700, -0.3) / 2.0)
    return round(base + (5.0 if advantage == "Advantage" else 0.0), 1)


# --------------------------------------------------------------------- fetchers
@dataclass
class PitcherMetrics:
    id: Optional[int]
    name: str = "TBD"
    hand: str = "R"
    k9: float = 0.0
    hr9: float = 0.0
    era: float = 0.0
    whip: float = 0.0
    oba: float = 0.0
    fip: float = 0.0
    stat: Dict[str, Any] = field(default_factory=dict)


def _aggregate_pitching_splits(splits: list) -> Dict[str, Any]:
    """Sum multiple season splits (e.g. a traded pitcher's two stints) into one stat dict.

    Counting stats are summed; innings are summed as outs and rebuilt in MLB's '.1/.2'
    thirds format so the parsers stay correct; rate fields are recomputed."""
    sums = {k: 0.0 for k in ("strikeOuts", "baseOnBalls", "hitByPitch", "homeRuns",
                             "battersFaced", "gamesStarted", "gamesPlayed", "earnedRuns",
                             "hits", "atBats")}
    total_outs = 0
    for sp in splits:
        s = sp.get("stat", {}) or {}
        for k in sums:
            sums[k] += safe_float(s.get(k))
        total_outs += int(round(parse_innings(s.get("inningsPitched")) * 3))

    ip_float = total_outs / 3.0
    sums["inningsPitched"] = f"{total_outs // 3}.{total_outs % 3}"  # e.g. 85.1
    if ip_float > 0:
        sums["era"] = sums["earnedRuns"] / ip_float * 9
        sums["whip"] = (sums["baseOnBalls"] + sums["hits"]) / ip_float
        sums["homeRunsPer9"] = sums["homeRuns"] / ip_float * 9
    sums["avg"] = (sums["hits"] / sums["atBats"]) if sums["atBats"] > 0 else 0.0
    return sums


def get_pitcher_metrics(pitcher_id: Optional[int],
                        fip_constant: float = FIP_CONSTANT_DEFAULT) -> PitcherMetrics:
    if not pitcher_id:
        return PitcherMetrics(id=None)
    data = fetch_json(f"{BASE}/people/{pitcher_id}",
                      {"hydrate": "stats(group=[pitching],type=[season])"})
    try:
        person = data["people"][0]
        splits = person["stats"][0]["splits"]
        if not splits:
            raise IndexError
        stat = _aggregate_pitching_splits(splits)  # handles 1 or many splits
    except (KeyError, IndexError):
        return PitcherMetrics(id=pitcher_id, name=data.get("people", [{}])[0].get("fullName", "TBD")
                              if data.get("people") else "TBD")

    so = safe_float(stat.get("strikeOuts"))
    ip = parse_innings(stat.get("inningsPitched"))
    return PitcherMetrics(
        id=pitcher_id,
        name=person.get("fullName", "TBD"),
        hand=person.get("pitchHand", {}).get("code", "R"),
        k9=(so / ip * 9) if ip > 0 else 0.0,
        hr9=safe_float(stat.get("homeRunsPer9")),
        era=safe_float(stat.get("era")),
        whip=safe_float(stat.get("whip")),
        oba=safe_float(stat.get("avg")),
        fip=calculate_fip(stat, fip_constant),
        stat=stat,
    )


def get_hitter_raw(player_id: int) -> Optional[Dict[str, Any]]:
    """name, bat hand, season hitting stat, and best-effort vs-LHP/vs-RHP splits.

    Self-contained: one combined call for season + splits, falling back to a season-only
    call so hitters ALWAYS load. Platoon splits are a bonus, never a requirement."""
    def parse(person: Dict[str, Any]):
        season = vs_l = vs_r = None
        for block in person.get("stats", []):
            btype = (block.get("type", {}) or {}).get("displayName", "")
            for sp in block.get("splits", []):
                code = (sp.get("split", {}) or {}).get("code")
                stat = sp.get("stat")
                if btype == "season" and season is None:
                    season = stat
                elif code == "vl":
                    vs_l = stat
                elif code == "vr":
                    vs_r = stat
        return season, vs_l, vs_r

    data = fetch_json(
        f"{BASE}/people/{player_id}",
        {"hydrate": "stats(group=[hitting],type=[season,statSplits],sitCodes=[vl,vr])"},
    )
    person = (data.get("people") or [None])[0]
    season = vs_l = vs_r = None
    if person:
        season, vs_l, vs_r = parse(person)

    if season is None:  # fallback: season-only (the original, reliable hydrate)
        data = fetch_json(f"{BASE}/people/{player_id}",
                          {"hydrate": "stats(group=[hitting],type=[season])"})
        person = (data.get("people") or [None])[0]
        if person:
            season, _, _ = parse(person)

    if person is None or season is None:
        return None
    return {
        "id": player_id,
        "name": person.get("fullName", "Unknown"),
        "bat_hand": person.get("batSide", {}).get("code", "R"),
        "stat": season,
        "vs_l": vs_l,
        "vs_r": vs_r,
    }


def get_schedule(date_str: str) -> List[Dict[str, Any]]:
    """Normalized game list with probable pitchers and venue."""
    sched = fetch_json(f"{BASE}/schedule",
                       {"sportId": 1, "date": date_str, "hydrate": "probablePitcher,venue"})
    games: List[Dict[str, Any]] = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            teams = g.get("teams", {})
            home, away = teams.get("home", {}), teams.get("away", {})
            games.append({
                "gamePk": g.get("gamePk"),
                "gameNumber": g.get("gameNumber", 1),
                "game_date": g.get("gameDate"),   # ISO UTC start, e.g. 2026-06-28T17:10:00Z
                "status": (g.get("status", {}) or {}).get("detailedState", ""),
                "venue_name": (g.get("venue", {}) or {}).get("name", ""),
                "venue_id": (g.get("venue", {}) or {}).get("id"),
                "home_name": home.get("team", {}).get("name", "Home"),
                "away_name": away.get("team", {}).get("name", "Away"),
                "home_id": home.get("team", {}).get("id"),
                "away_id": away.get("team", {}).get("id"),
                "home_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
                "away_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
            })
    return sorted(games, key=lambda x: (x["away_name"], x["gameNumber"]))


def _team_starters(game: Dict, team_key: str, box: Dict) -> Tuple[List[int], bool]:
    """Return (player_ids, projected). Posted batting order if available, else active roster.

    Decided PER TEAM (fixes the original home-only bug)."""
    order = []
    try:
        order = box["teams"][team_key].get("battingOrder", []) or []
    except (KeyError, TypeError):
        order = []
    if order:
        return [int(p) for p in order][:9], False

    team_id = game[f"{team_key}_id"]
    roster = fetch_json(f"{BASE}/teams/{team_id}/roster/Active", {"hydrate": "person"}).get("roster", [])
    pids = [r["person"]["id"] for r in roster
            if r.get("position", {}).get("abbreviation") != "P" and r.get("person", {}).get("id")]
    return pids, True


# --------------------------------------------------------------------- orchestration
def build_slate(date_str: str, fip_constant: float = FIP_CONSTANT_DEFAULT,
                max_workers: int = 8) -> Tuple[List[Dict], List[Dict]]:
    """Fetch and assemble the full slate concurrently.

    Returns (hitter_rows, game_meta):
      hitter_rows : list of flat dicts ready for a DataFrame (one per hitter)
      game_meta   : list of per-game dicts (label, venue, both PitcherMetrics, names)
    """
    games = [g for g in get_schedule(date_str) if g.get("gamePk")]
    if not games:
        return [], []

    # Phase 1 — per-game setup (pitcher metrics + boxscore + starters) in parallel.
    def setup_game(game: Dict) -> Dict:
        gid = game["gamePk"]
        home_pm = get_pitcher_metrics(game["home_pitcher_id"], fip_constant)
        away_pm = get_pitcher_metrics(game["away_pitcher_id"], fip_constant)
        box = fetch_json(f"{BASE}/game/{gid}/boxscore")
        label = f"{game['away_name']} @ {game['home_name']} (Game {game['gameNumber']})"
        # Away hitters face the HOME pitcher, and vice versa.
        sides = []
        for team_key, opp_pm in (("away", home_pm), ("home", away_pm)):
            pids, projected = _team_starters(game, team_key, box)
            sides.append({
                "team_key": team_key,
                "team_name": game[f"{team_key}_name"],
                "opp_pm": opp_pm,
                "pids": pids,
                "projected": projected,
            })
        return {"game": game, "label": label, "home_pm": home_pm, "away_pm": away_pm, "sides": sides}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        setups = list(ex.map(setup_game, games))

    # Phase 2 — fetch every unique hitter ONCE, concurrently.
    unique_pids = {pid for s in setups for side in s["sides"] for pid in side["pids"]}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        raws = list(ex.map(get_hitter_raw, unique_pids))
    raw_by_id = {r["id"]: r for r in raws if r}

    # Phase 3 — assemble rows + meta (pure, no network).
    rows: List[Dict] = []
    meta: List[Dict] = []
    for s in setups:
        meta.append({
            "label": s["label"],
            "venue": s["game"]["venue_name"],
            "venue_id": s["game"].get("venue_id"),
            "status": s["game"]["status"],
            "game_date": s["game"].get("game_date"),
            "away_name": s["game"]["away_name"],
            "home_name": s["game"]["home_name"],
            "home_pm": s["home_pm"],
            "away_pm": s["away_pm"],
        })
        for side in s["sides"]:
            opp = side["opp_pm"]
            for idx, pid in enumerate(side["pids"]):
                raw = raw_by_id.get(pid)
                if not raw:
                    continue
                rows.append(_hitter_row(raw, opp, side["team_name"], s["label"],
                                        side["projected"], idx, s["game"].get("venue_id")))
    return rows, meta


def build_pitching_slate(date_str: str, fip_constant: float = FIP_CONSTANT_DEFAULT,
                         max_workers: int = 8) -> List[Dict]:
    """Lightweight: probable starters across the slate with ERA/FIP/peripherals.

    Does NOT fetch hitters or boxscores, so it is much cheaper than build_slate.
    Returns one row per probable starter, including Delta = ERA - FIP (positive =
    underlying performance better than results = positive-regression candidate)."""
    games = [g for g in get_schedule(date_str) if g.get("gamePk")]
    tasks = []  # (pitcher_id, team_name, opponent, game_label)
    for g in games:
        label = f"{g['away_name']} @ {g['home_name']}"
        tasks.append((g["home_pitcher_id"], g["home_name"], g["away_name"], label))
        tasks.append((g["away_pitcher_id"], g["away_name"], g["home_name"], label))

    def fetch(t):
        pid, team, opp, label = t
        pm = get_pitcher_metrics(pid, fip_constant)
        return pm, team, opp, label

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(fetch, tasks))

    rows = []
    for pm, team, opp, label in results:
        if pm.id is None or pm.era == 0:
            continue
        rows.append({
            "Pitcher": pm.name, "Team": team, "Opponent": opp, "Game": label, "Hand": pm.hand,
            "ERA": round(pm.era, 2), "FIP": pm.fip, "Delta": round(pm.era - pm.fip, 2),
            "K/9": round(pm.k9, 1), "WHIP": round(pm.whip, 2), "HR/9": round(pm.hr9, 2), "OBA": pm.oba,
        })
    return rows


def _hitter_row(raw: Dict, opp: PitcherMetrics, team_name: str,
                game_label: str, projected: bool, lineup_idx: int = 0,
                venue_id: int = None) -> Dict:
    stat = raw["stat"]
    avg = safe_float(stat.get("avg"))
    slg = safe_float(stat.get("slg"))
    ops = safe_float(stat.get("ops"))
    pa = max(safe_float(stat.get("plateAppearances"), 1), 1)
    iso = round(slg - avg, 3)
    adv = platoon_advantage(raw["bat_hand"], opp.hand)
    # Expected PA for the game: batting-order spot if the lineup is posted, else a default.
    exp_pa = (LINEUP_SPOT_PA[lineup_idx] if (not projected and lineup_idx < len(LINEUP_SPOT_PA))
              else DEFAULT_UNKNOWN_PA)
    return {
        "Hitter": raw["name"],
        "Team": team_name,
        "GameLabel": game_label,
        "Hand": raw["bat_hand"],
        "Opp Pitcher": opp.name,
        "Opp Hand": opp.hand,
        "Advantage": adv,
        "Lineup": "Projected" if projected else "Confirmed",
        "HR": safe_float(stat.get("homeRuns")),
        "Hits": safe_float(stat.get("hits")),
        "TB": safe_float(stat.get("totalBases")),
        "AVG": avg,
        "OBP": safe_float(stat.get("obp")),
        "SLG": slg,
        "OPS": ops,
        "ISO": iso,
        "K%": safe_float(stat.get("strikeOuts")) / pa,
        "PowerIndex": power_index(iso, ops, adv),
        # private fields consumed by projections.py (underscore -> not shown in tables)
        "_pid": raw["id"],
        "_stat": stat,
        "_exp_pa": exp_pa,
        "_venue_id": venue_id,
        "_opp_stat": opp.stat,                       # opposing pitcher's season line (matchup)
        "_split_stat": (raw.get("vs_l") if opp.hand == "L" else raw.get("vs_r")),  # platoon split
    }
