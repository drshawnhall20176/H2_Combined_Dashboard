"""
odds_api.py — The Odds API client + edge calculation.

Fetches MLB player-prop odds and joins them to the model's projection index to compute
true edge: the model probability evaluated AT THE BOOK'S LINE, compared to a de-vigged
market price.

Key handling: the API key is passed IN as an argument (the page reads it from
st.secrets / env). This module never stores, logs, or hardcodes it.

Quota: player props cost 1 unit per market per event. fetch_slate_props pulls all
requested markets for each slate event in a single request each, and returns the
remaining-quota header so the UI can warn you.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

import projections as P

BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"

# The model's markets, expressed as Odds API market keys.
SUPPORTED_MARKETS = [
    "batter_home_runs", "batter_total_bases", "batter_hits", "batter_strikeouts",
    "pitcher_strikeouts", "pitcher_outs", "pitcher_walks",
]


# ---- odds math -------------------------------------------------------------
def american_to_decimal(american: float) -> float:
    return 1 + (american / 100 if american > 0 else 100 / (-american))


def implied_prob(american: float) -> float:
    return (-american) / ((-american) + 100) if american < 0 else 100 / (american + 100)


def ev_percent(prob: float, american: float) -> float:
    """Expected value per $1 staked, as a percent. +5 means +5% EV."""
    return (prob * american_to_decimal(american) - 1) * 100


def devig_two_way(over_american: float, under_american: float) -> Optional[float]:
    """Return the no-vig (fair) probability of the OVER from a book's two-sided prices."""
    io, iu = implied_prob(over_american), implied_prob(under_american)
    total = io + iu
    return io / total if total > 0 else None


# ---- API client ------------------------------------------------------------
class OddsAPIError(Exception):
    pass


def _get(path: str, params: Dict) -> Tuple[Dict, Dict]:
    try:
        r = requests.get(f"{BASE}/{path}", params=params, timeout=20)
    except requests.RequestException as e:
        raise OddsAPIError(f"network error: {e}") from e
    if r.status_code == 401:
        raise OddsAPIError("401 Unauthorized — check your API key.")
    if r.status_code == 429:
        raise OddsAPIError("429 — out of quota for this period.")
    if r.status_code != 200:
        raise OddsAPIError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json(), {
        "remaining": r.headers.get("x-requests-remaining"),
        "used": r.headers.get("x-requests-used"),
    }


def fetch_events(api_key: str) -> List[Dict]:
    data, _ = _get(f"sports/{SPORT}/events", {"apiKey": api_key, "dateFormat": "iso"})
    return data if isinstance(data, list) else []


def fetch_event_props(event_id: str, api_key: str, markets: List[str],
                      regions: str = "us") -> Tuple[Dict, Dict]:
    return _get(
        f"sports/{SPORT}/events/{event_id}/odds",
        {"apiKey": api_key, "regions": regions, "markets": ",".join(markets),
         "oddsFormat": "american", "dateFormat": "iso"},
    )


# ---- parsing ---------------------------------------------------------------
def parse_event_offers(event_json: Dict) -> List[Dict]:
    """Collapse all bookmakers into per-(market, player, line) offers with both sides.

    Returns list of dicts:
      {market, player, point, over:{book:price}, under:{book:price}}
    """
    offers: Dict[Tuple, Dict] = {}
    for bm in event_json.get("bookmakers", []):
        book = bm.get("key", "?")
        for mk in bm.get("markets", []):
            mkey = mk.get("key")
            if mkey not in SUPPORTED_MARKETS:
                continue
            for oc in mk.get("outcomes", []):
                player = oc.get("description")
                point = oc.get("point")
                side = (oc.get("name") or "").lower()
                price = oc.get("price")
                if player is None or point is None or price is None:
                    continue
                k = (mkey, player, point)
                slot = offers.setdefault(k, {"market": mkey, "player": player,
                                             "point": point, "over": {}, "under": {}})
                if side.startswith("o"):
                    slot["over"][book] = price
                elif side.startswith("u"):
                    slot["under"][book] = price
    return list(offers.values())


# ---- edge computation ------------------------------------------------------
def _best_price(book_prices: Dict[str, float]) -> Optional[Tuple[str, float]]:
    """Best (highest decimal payout) price across books for one side."""
    if not book_prices:
        return None
    book, price = max(book_prices.items(), key=lambda kv: american_to_decimal(kv[1]))
    return book, price


def compute_edges(index: Dict, offers: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Join book offers to the model index and compute EV/edge per playable side.

    Returns (edge_rows, stats). edge_rows are sorted by EV% descending."""
    rows: List[Dict] = []
    matched = unmatched = 0

    for off in offers:
        mkey, point = off["market"], off["point"]
        nm = P.normalize_name(off["player"])
        entry = index.get((nm, mkey))
        if entry is None:
            unmatched += 1
            continue
        matched += 1
        dist = entry["dist"]

        # Consensus no-vig prob of the OVER (averaged across books offering both sides).
        novig_overs = []
        for book in set(off["over"]) & set(off["under"]):
            nv = devig_two_way(off["over"][book], off["under"][book])
            if nv is not None:
                novig_overs.append(nv)
        novig_over = sum(novig_overs) / len(novig_overs) if novig_overs else None

        for side, prices in (("Over", off["over"]), ("Under", off["under"])):
            bp = _best_price(prices)
            if bp is None:
                continue
            book, price = bp
            model_p = P.prob_for_side(dist, point, side)
            novig_side = (novig_over if side == "Over" else (1 - novig_over)) if novig_over is not None else None
            rows.append({
                "Player": entry["ctx"]["player"],
                "Team": entry["ctx"]["team"],
                "Game": entry["ctx"]["game"],
                "Market": mkey,
                "Side": side,
                "Line": point,
                "ModelProb": round(model_p, 4),
                "Proj": round(entry["mean"], 2),
                "Book": book,
                "Price": price,
                "ImpliedBest": round(implied_prob(price), 4),
                "NoVigMkt": round(novig_side, 4) if novig_side is not None else None,
                "EdgeVsMkt": round(model_p - novig_side, 4) if novig_side is not None else None,
                "EV%": round(ev_percent(model_p, price), 2),
            })

    rows.sort(key=lambda r: r["EV%"], reverse=True)
    return rows, {"matched": matched, "unmatched": unmatched}


def fetch_slate_props(date_str: str, api_key: str, markets: List[str]) -> Tuple[List[Dict], Dict]:
    """Pull props for every event on the slate date. Returns (offers, info).

    info includes remaining quota and event counts so the UI can show cost."""
    events = fetch_events(api_key)
    todays = [e for e in events if str(e.get("commence_time", ""))[:10] == date_str]
    offers: List[Dict] = []
    remaining = None
    fetched = 0
    for e in todays:
        try:
            ej, hdr = fetch_event_props(e["id"], api_key, markets)
        except OddsAPIError:
            continue
        remaining = hdr.get("remaining") or remaining
        offers.extend(parse_event_offers(ej))
        fetched += 1
    return offers, {"events_total": len(todays), "events_fetched": fetched,
                    "remaining": remaining}
