"""
refresh_statcast.py — pull the season's Statcast batter data and cache it to disk.

Run this once a day (manually, or via Task Scheduler / cron). The dashboard reads the
cached file instantly and never blocks on Baseball Savant.

    python refresh_statcast.py            # current season
    python refresh_statcast.py 2026       # explicit season

Requires pybaseball:  pip install pybaseball
"""

import sys
from datetime import date

import statcast_data as SC


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else (date.today().year)
    print(f"Pulling Statcast batter data for {year} from Baseball Savant...")
    try:
        path = SC.refresh(year)
    except ImportError:
        print("pybaseball is not installed. Run:  pip install pybaseball")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"Refresh failed: {e}")
        print("If column names changed in your pybaseball version, tell Claude and we'll adjust.")
        return 1
    # quick sanity read-back
    lookup, k = SC.load(path)
    print(f"Cached {len(lookup)} batters. Calibration k = {round(k, 3) if k else 'n/a'}.")
    print("The Dinger Engine will use this automatically on its next refresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
