"""
weather.py — game-time weather and its effect on home runs.

Two effects are physics, not hunches:
  • Temperature: warm air is less dense, the ball carries farther. ~robust, safe to apply.
  • Wind: the component blowing out to center field adds carry; blowing in kills it.

Plus dome handling: under a fixed roof, weather is irrelevant (factor 1.0).

ARCHITECTURE: Open-Meteo is free and keyless. Fetches are cached per (venue, hour). If a
park isn't in the table or the fetch fails, weather degrades to "n/a" (factor 1.0) and the
projection is simply un-adjusted — nothing breaks.

HONESTY NOTE: temperature and dome are trustworthy. The WIND effect needs each park's
orientation (the compass bearing from home plate to center field). Those bearings below are
best-effort and should be verified against a real source before you lean on the wind number;
the coefficient is deliberately conservative and the whole factor is clamped so an error
can't blow up a projection. venue_ids should also be verified against a live slate.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Callable, Dict, Optional

import requests

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

# Tunable, documented coefficients (calibrate against your own results over time).
TEMP_BASELINE_F = 70.0
TEMP_COEF = 0.006      # +0.6% HR per °F above baseline (≈ +12% at 90°F)
WIND_COEF = 0.010      # +1.0% HR per mph of wind blowing OUT to center
HR_FACTOR_MIN, HR_FACTOR_MAX = 0.75, 1.30

# venue_id -> stadium. roof: "open" | "fixed" (always covered) | "retractable" (uncertain).
# cf_bearing = degrees (0=N, 90=E) from home plate toward center field. VERIFY these.
STADIUMS: Dict[int, Dict] = {
    1:    dict(name="Angel Stadium",        lat=33.800, lon=-117.883, roof="open",        cf_bearing=45),
    2:    dict(name="Chase Field",          lat=33.445, lon=-112.067, roof="retractable", cf_bearing=0),
    3:    dict(name="Oriole Park",          lat=39.284, lon=-76.622,  roof="open",        cf_bearing=30),
    4:    dict(name="Fenway Park",          lat=42.346, lon=-71.097,  roof="open",        cf_bearing=45),
    5:    dict(name="Wrigley Field",        lat=41.948, lon=-87.656,  roof="open",        cf_bearing=30),
    7:    dict(name="Rate Field",           lat=41.830, lon=-87.634,  roof="open",        cf_bearing=125),
    8:    dict(name="Great American Ball Park", lat=39.097, lon=-84.507, roof="open",     cf_bearing=100),
    10:   dict(name="Progressive Field",    lat=41.496, lon=-81.685,  roof="open",        cf_bearing=0),
    12:   dict(name="Coors Field",          lat=39.756, lon=-104.994, roof="open",        cf_bearing=0),
    14:   dict(name="Comerica Park",        lat=42.339, lon=-83.049,  roof="open",        cf_bearing=150),
    15:   dict(name="Daikin Park",          lat=29.757, lon=-95.355,  roof="retractable", cf_bearing=345),
    17:   dict(name="Kauffman Stadium",     lat=39.051, lon=-94.480,  roof="open",        cf_bearing=60),
    19:   dict(name="Dodger Stadium",       lat=34.074, lon=-118.240, roof="open",        cf_bearing=25),
    22:   dict(name="loanDepot park",       lat=25.778, lon=-80.220,  roof="retractable", cf_bearing=40),
    2392: dict(name="Target Field",         lat=44.982, lon=-93.278,  roof="open",        cf_bearing=90),
    2395: dict(name="Citizens Bank Park",   lat=39.906, lon=-75.166,  roof="open",        cf_bearing=0),
    2602: dict(name="Yankee Stadium",       lat=40.829, lon=-73.926,  roof="open",        cf_bearing=25),
    2680: dict(name="Petco Park",           lat=32.707, lon=-117.157, roof="open",        cf_bearing=0),
    2681: dict(name="Busch Stadium",        lat=38.622, lon=-90.193,  roof="open",        cf_bearing=70),
    2889: dict(name="American Family Field", lat=43.028, lon=-87.971, roof="retractable", cf_bearing=0),
    3289: dict(name="Truist Park",          lat=33.890, lon=-84.468,  roof="open",        cf_bearing=25),
    3309: dict(name="T-Mobile Park",        lat=47.591, lon=-122.332, roof="retractable", cf_bearing=0),
    3312: dict(name="Oracle Park",          lat=37.778, lon=-122.389, roof="open",        cf_bearing=90),
    4169: dict(name="Globe Life Field",     lat=32.747, lon=-97.084,  roof="retractable", cf_bearing=0),
    5325: dict(name="Tropicana Field",      lat=27.768, lon=-82.653,  roof="fixed",       cf_bearing=0),
    680:  dict(name="Rogers Centre",        lat=43.641, lon=-79.389,  roof="retractable", cf_bearing=0),
    32:   dict(name="PNC Park",             lat=40.447, lon=-80.006,  roof="open",        cf_bearing=120),
    31:   dict(name="Nationals Park",       lat=38.873, lon=-77.007,  roof="open",        cf_bearing=30),
}


def wind_out_component(wind_mph: float, wind_from_deg: float, cf_bearing: float) -> float:
    """Component of wind along the home-plate -> center-field axis (mph).

    wind_from_deg is meteorological 'direction the wind comes FROM'. Positive result = wind
    blowing OUT toward center (helps HR); negative = blowing IN (suppresses)."""
    blowing_to = (wind_from_deg + 180.0) % 360.0          # direction wind blows toward
    return wind_mph * math.cos(math.radians(blowing_to - cf_bearing))


def hr_factor(temp_f: Optional[float], out_wind_mph: float, roof: str) -> float:
    """Multiplier on a hitter's HR rate from weather. 1.0 = neutral / indoors."""
    if roof == "fixed":
        return 1.0
    temp_adj = 1.0 + TEMP_COEF * ((temp_f - TEMP_BASELINE_F) if temp_f is not None else 0.0)
    wind_adj = 1.0 + WIND_COEF * out_wind_mph
    return float(min(max(temp_adj * wind_adj, HR_FACTOR_MIN), HR_FACTOR_MAX))


def _fetch_open_meteo(lat: float, lon: float, date_str: str) -> Dict:
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "start_date": date_str, "end_date": date_str, "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def get_game_weather(venue_id: Optional[int], iso_utc: Optional[str],
                     fetcher: Optional[Callable] = None) -> Optional[Dict]:
    """Weather + HR factor for a game. None if the park is unknown (safe -> no adjustment).

    `fetcher(lat, lon, date_str) -> open-meteo-json` is injectable for testing.
    """
    park = STADIUMS.get(venue_id) if venue_id is not None else None
    if not park or "lat" not in park:
        return None

    # Fixed dome: skip the network call entirely, weather is irrelevant.
    if park["roof"] == "fixed":
        return {"park": park["name"], "roof": "fixed", "dome": True,
                "temp_f": None, "wind_mph": 0.0, "out_wind_mph": 0.0,
                "hr_factor": 1.0, "wind_desc": "indoors", "summary": "Indoors (fixed roof)"}

    if not iso_utc:
        return None
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        date_str = dt.strftime("%Y-%m-%d")
        fetch = fetcher or _fetch_open_meteo
        data = fetch(park["lat"], park["lon"], date_str)
        hourly = data["hourly"]
        times = hourly["time"]
        target = dt.strftime("%Y-%m-%dT%H:00")
        idx = times.index(target) if target in times else min(
            range(len(times)),
            key=lambda i: abs(datetime.fromisoformat(times[i]).replace(tzinfo=dt.tzinfo) - dt))
        temp = hourly["temperature_2m"][idx]
        wind = hourly["wind_speed_10m"][idx]
        wdir = hourly["wind_direction_10m"][idx]
    except Exception:
        return None

    out_wind = wind_out_component(wind, wdir, park["cf_bearing"])
    factor = hr_factor(temp, out_wind, park["roof"])

    if out_wind > 2:
        wind_desc = f"{wind:.0f} mph out to CF"
    elif out_wind < -2:
        wind_desc = f"{wind:.0f} mph in from CF"
    else:
        wind_desc = f"{wind:.0f} mph crosswind"

    return {
        "park": park["name"], "roof": park["roof"], "dome": False,
        "temp_f": round(temp), "wind_mph": round(wind), "wind_dir": round(wdir),
        "out_wind_mph": round(out_wind, 1), "hr_factor": round(factor, 3),
        "wind_desc": wind_desc,
        "summary": f"{round(temp)}°F · {wind_desc}",
        "approx_wind": park["roof"] == "retractable",
    }
