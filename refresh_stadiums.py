"""
refresh_stadiums.py — build an authoritative stadium table from MLB's own API.

Solves the verification problem: venue ids, current names, and coordinates come straight
from MLB (so this auto-handles sponsor renames and relocations, e.g. a team playing in a
temporary park). The two fields MLB's API does NOT provide — roof type and the home-plate-
to-center-field bearing for wind — are taken from weather._STATIC_STADIUMS, matched by name.

Writes data/stadiums.json, which weather.py loads and uses to override its static defaults.

    python refresh_stadiums.py            # current season
    python refresh_stadiums.py 2026
"""

import json
import os
import sys
from datetime import date

import mlb_engine as E
import weather as W

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stadiums.json")


def main() -> int:
    year = int(sys.argv[1]) if len(sys.argv) > 1 else date.today().year
    print(f"Fetching MLB venues for {year}...")

    # roof + bearing overlay keyed by normalized park name (our domain knowledge).
    overlay = {W._norm(v["name"]): (v["roof"], v["cf_bearing"])
               for v in W._STATIC_STADIUMS.values()}

    try:
        teams = E.fetch_json(f"{E.BASE}/teams", {"sportId": 1, "season": year})["teams"]
    except Exception as e:  # noqa: BLE001
        print(f"Failed to fetch teams: {e}")
        return 1

    venue_ids = sorted({t["venue"]["id"] for t in teams if t.get("venue", {}).get("id")})
    try:
        vresp = E.fetch_json(f"{E.BASE}/venues",
                             {"venueIds": ",".join(map(str, venue_ids)), "hydrate": "location"})
        venues = vresp["venues"]
    except Exception as e:  # noqa: BLE001
        print(f"Failed to fetch venue coordinates: {e}")
        return 1

    table, missing_coords, missing_overlay = {}, [], []
    for v in venues:
        vid, name = v["id"], v.get("name", "")
        coords = (v.get("location", {}) or {}).get("defaultCoordinates", {}) or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")
        if lat is None or lon is None:
            missing_coords.append(name)
            continue
        key = W._norm(name)
        if key in overlay:
            roof, bearing = overlay[key]
        else:
            roof, bearing = "open", 0
            missing_overlay.append(name)
        table[str(vid)] = {"name": name, "lat": lat, "lon": lon,
                           "roof": roof, "cf_bearing": bearing}

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(table, f, indent=2)

    print(f"Wrote {len(table)} parks to {OUT_PATH}")
    if missing_coords:
        print(f"  ⚠ no coordinates from API for: {', '.join(missing_coords)}")
    if missing_overlay:
        print(f"  ⚠ no roof/bearing match (defaulted to open/0 — add to weather._STATIC_STADIUMS): "
              f"{', '.join(missing_overlay)}")
    print("weather.py will use this file automatically on its next load.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
