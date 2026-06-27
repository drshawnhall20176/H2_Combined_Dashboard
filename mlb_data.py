"""
Thin client over the public MLB Stats API (statsapi.mlb.com).

No API key required. This is the same data source the league's own apps use, so it's
far more reliable than scraping HTML. Everything here is read-only GET requests.

Key endpoints used:
  /schedule        -> today's games, probable pitchers, posted lineups, venue
  /teams/{id}/roster?rosterType=active  -> 26-man active roster (lineup fallback)
  /people/{id}/stats?stats=season       -> season hitting/pitching totals
  /people/{id}/stats?stats=gameLog      -> per-game logs (recent form / IP projection)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

import config


class MLBStatsClient:
    def __init__(self, base: str = config.API_BASE, timeout: int = config.REQUEST_TIMEOUT):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "mlb-parlay-mvp/1.0"})
        # tiny in-process cache so we don't refetch the same player stats repeatedly
        self._cache: Dict[str, Any] = {}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, retries: int = 2) -> Dict[str, Any]:
        url = f"{self.base}/{path.lstrip('/')}"
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # noqa: BLE001 - simple retry on any transient error
                last_err = e
                if attempt < retries:
                    time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"GET {url} failed after retries: {last_err}")

    # ----------------------------------------------------------------- schedule
    def get_schedule(self, game_date: str) -> List[Dict[str, Any]]:
        """Return a list of game dicts for the given YYYY-MM-DD date.

        Each game is normalized to:
          {gamePk, gameDate, status, venue_id, venue_name,
           home: {team_id, team_name, probable_pitcher, lineup},
           away: {...}}
        where lineup is a list of {id, name} in batting order, or [] if not posted.
        """
        data = self._get(
            "schedule",
            params={
                "sportId": 1,
                "date": game_date,
                "hydrate": "probablePitcher,lineups,team,venue",
            },
        )
        games: List[Dict[str, Any]] = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                games.append(self._normalize_game(g))
        return games

    @staticmethod
    def _normalize_game(g: Dict[str, Any]) -> Dict[str, Any]:
        venue = g.get("venue", {}) or {}
        teams = g.get("teams", {}) or {}
        lineups = g.get("lineups", {}) or {}

        def side(key: str, lineup_key: str) -> Dict[str, Any]:
            t = teams.get(key, {}) or {}
            team = t.get("team", {}) or {}
            pp = t.get("probablePitcher") or None
            probable = None
            if pp:
                probable = {"id": pp.get("id"), "name": pp.get("fullName")}
            order = lineups.get(lineup_key, []) or []
            lineup = [{"id": p.get("id"), "name": p.get("fullName")} for p in order if p.get("id")]
            return {
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "probable_pitcher": probable,
                "lineup": lineup,
            }

        return {
            "gamePk": g.get("gamePk"),
            "gameDate": g.get("gameDate"),
            "status": (g.get("status", {}) or {}).get("detailedState"),
            "venue_id": venue.get("id"),
            "venue_name": venue.get("name"),
            "home": side("home", "homePlayers"),
            "away": side("away", "awayPlayers"),
        }

    # -------------------------------------------------------------------- roster
    def get_active_roster(self, team_id: int) -> List[Dict[str, Any]]:
        key = f"roster:{team_id}"
        if key in self._cache:
            return self._cache[key]
        data = self._get(f"teams/{team_id}/roster", params={"rosterType": "active"})
        out = []
        for r in data.get("roster", []):
            person = r.get("person", {}) or {}
            pos = (r.get("position", {}) or {}).get("abbreviation", "")
            out.append({"id": person.get("id"), "name": person.get("fullName"), "position": pos})
        self._cache[key] = out
        return out

    # --------------------------------------------------------------- player stats
    def get_season_hitting(self, person_id: int, season: int) -> Optional[Dict[str, Any]]:
        return self._season_stat(person_id, season, "hitting")

    def get_season_pitching(self, person_id: int, season: int) -> Optional[Dict[str, Any]]:
        return self._season_stat(person_id, season, "pitching")

    def _season_stat(self, person_id: int, season: int, group: str) -> Optional[Dict[str, Any]]:
        key = f"season:{group}:{person_id}:{season}"
        if key in self._cache:
            return self._cache[key]
        data = self._get(
            f"people/{person_id}/stats",
            params={"stats": "season", "group": group, "season": season},
        )
        stat = None
        for block in data.get("stats", []):
            splits = block.get("splits", [])
            if splits:
                stat = splits[0].get("stat")
                break
        self._cache[key] = stat
        return stat

    def get_game_log(self, person_id: int, season: int, group: str) -> List[Dict[str, Any]]:
        key = f"gamelog:{group}:{person_id}:{season}"
        if key in self._cache:
            return self._cache[key]
        data = self._get(
            f"people/{person_id}/stats",
            params={"stats": "gameLog", "group": group, "season": season},
        )
        logs: List[Dict[str, Any]] = []
        for block in data.get("stats", []):
            for sp in block.get("splits", []):
                row = sp.get("stat", {}) or {}
                row["_date"] = sp.get("date")
                logs.append(row)
        self._cache[key] = logs
        return logs


# ----------------------------------------------------------------- stat helpers
def parse_innings(ip_value: Any) -> float:
    """MLB reports innings like '85.1' meaning 85 and 1/3 innings ('.1'=1 out, '.2'=2 outs)."""
    if ip_value in (None, ""):
        return 0.0
    s = str(ip_value)
    if "." not in s:
        try:
            return float(s)
        except ValueError:
            return 0.0
    whole, frac = s.split(".", 1)
    outs = {"0": 0, "1": 1, "2": 2}.get(frac[:1], 0)
    try:
        return int(whole) + outs / 3.0
    except ValueError:
        return 0.0


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
