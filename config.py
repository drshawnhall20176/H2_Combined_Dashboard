"""
Static configuration: park factors, expected plate appearances by lineup spot,
default prop lines, and tuning constants.

These are sane defaults for an MVP. The two things most worth improving over time
are (1) PARK_FACTORS (regress them toward 1.0 and update yearly) and
(2) DEFAULT_LINES, which should ultimately be replaced by *real* sportsbook lines
pulled from an odds feed (see README -> "Adding real odds").
"""

from datetime import date

# Current MLB season. Auto-detected from today's date, override with --season.
def default_season() -> int:
    today = date.today()
    # Treat Jan/Feb as belonging to the prior season's offseason; otherwise use the year.
    return today.year if today.month >= 3 else today.year - 1


# Expected plate appearances by batting-order index (0 = leadoff ... 8 = nine-hole).
# Rough league averages for a 9-inning game. Leadoff hitters get the most PA.
LINEUP_SPOT_PA = [4.65, 4.55, 4.45, 4.35, 4.25, 4.10, 4.00, 3.90, 3.80]

# Used when a batter is projected to start but the order is not yet posted.
DEFAULT_UNKNOWN_PA = 4.25

# How many of a team's position players to treat as "projected starters" when the
# lineup is not posted. Ranked by season plate appearances (regulars float to the top).
PROJECTED_STARTERS_PER_TEAM = 9

# Park factors keyed by MLB venue id. >1.0 helps offense, <1.0 suppresses it.
# Separate HR and hits multipliers. Anything not listed defaults to 1.0 (neutral).
# These are approximate, multi-year-ish values — refine with current data when you can.
PARK_FACTORS = {
    # venue_id: {"hr": x, "hits": y}
    1:   {"hr": 1.18, "hits": 1.04},  # Chase Field (variable w/ roof, treat ~neutral-high)
    2:   {"hr": 0.95, "hits": 1.00},  # Oriole Park (post-2024 LF wall change ~neutral)
    3:   {"hr": 0.96, "hits": 1.08},  # Fenway Park (high hits, low HR)
    4:   {"hr": 1.10, "hits": 1.02},  # Truist Park
    5:   {"hr": 1.10, "hits": 1.03},  # Wrigley Field (wind-dependent)
    7:   {"hr": 1.30, "hits": 1.10},  # Coors Field
    9:   {"hr": 0.92, "hits": 0.96},  # Comerica Park
    12:  {"hr": 1.02, "hits": 1.00},  # loanDepot park (Miami)
    14:  {"hr": 1.05, "hits": 1.00},  # Rogers Centre
    15:  {"hr": 1.08, "hits": 1.02},  # Great American Ball Park (HR-friendly)
    17:  {"hr": 1.06, "hits": 1.02},  # Guaranteed Rate / Rate Field (Chicago AL)
    19:  {"hr": 0.98, "hits": 1.01},  # Kauffman Stadium (big OF, low HR, gap hits)
    22:  {"hr": 1.07, "hits": 1.01},  # Dodger Stadium
}

NEUTRAL_PARK = {"hr": 1.00, "hits": 1.00}

# Default prop lines used when no live sportsbook line is supplied.
# For batter HR the prop is "anytime HR" (a yes/no, no numeric line).
DEFAULT_LINES = {
    "batter_hr": None,           # anytime HR -> probability of >=1 HR
    "batter_total_bases": 1.5,   # over/under 1.5 total bases
    "batter_hits": 0.5,          # over/under 0.5 hits (i.e. "to record a hit")
    "batter_strikeouts": 0.5,    # over/under 0.5 strikeouts
    "pitcher_strikeouts": 5.5,
    "pitcher_outs": 17.5,        # ~5.2 innings
    "pitcher_walks": 1.5,
}

# Monte Carlo sample count. 20k is fast with numpy and plenty stable.
DEFAULT_SIMS = 20000

# Recent-form blending for batters (only used if --recent-form is on).
RECENT_GAMES_N = 15
RECENT_FORM_WEIGHT = 0.25  # 0 = ignore recent form, 1 = use only recent form

# Recent starts used to project a pitcher's innings/batters-faced.
RECENT_STARTS_N = 5

# HTTP
API_BASE = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 15
