"""
smoke_test.py — exercises the LIVE data path (needs internet, no Streamlit).

Run it on your machine to confirm the engine is actually pulling real data before
you open the dashboard:

    python smoke_test.py                # today's date
    python smoke_test.py 2026-07-04     # a specific date

It checks the three layers the pages depend on: schedule, pitching slate, full slate.
If you see real names and non-zero stats (not all "TBD"/0), the data path works.
"""

import sys
import time
from datetime import datetime

import mlb_engine as E


def main(date_str: str) -> int:
    print(f"=== Live smoke test for {date_str} ===\n")

    # 1) Schedule
    games = E.get_schedule(date_str)
    print(f"[schedule]  {len(games)} game(s) found")
    if not games:
        print("  No games on this date. Try a date during the season with a full slate.")
        return 0
    g = games[0]
    print(f"  e.g. {g['away_name']} @ {g['home_name']} — {g['venue_name']} ({g['status']})")
    print(f"       probable SPs: home_id={g['home_pitcher_id']} away_id={g['away_pitcher_id']}\n")

    # 2) Pitching slate (lightweight)
    t0 = time.time()
    pitchers = E.build_pitching_slate(date_str)
    print(f"[pitching]  {len(pitchers)} probable starters in {time.time()-t0:.1f}s")
    for p in pitchers[:3]:
        print(f"  {p['Pitcher']:<22} ERA {p['ERA']:.2f}  FIP {p['FIP']:.2f}  "
              f"Δ {p['Delta']:+.2f}  K/9 {p['K/9']:.1f}")
    print()

    # 3) Full slate (hitters + concurrency)
    t0 = time.time()
    rows, meta = E.build_slate(date_str)
    elapsed = time.time() - t0
    confirmed = sum(1 for r in rows if r["Lineup"] == "Confirmed")
    print(f"[slate]     {len(meta)} games · {len(rows)} hitters in {elapsed:.1f}s "
          f"({confirmed} confirmed, {len(rows)-confirmed} projected)")
    for r in sorted(rows, key=lambda x: x["PowerIndex"], reverse=True)[:5]:
        print(f"  {r['Hitter']:<22} {r['Team']:<22} ISO {r['ISO']:.3f}  "
              f"OPS {r['OPS']:.3f}  vs {r['Opp Pitcher']} ({r['Advantage']})")

    # Basic sanity assertions
    print()
    problems = []
    if all(p["ERA"] == 0 for p in pitchers):
        problems.append("all pitcher ERAs are 0 — stats not parsing")
    if rows and all(r["OPS"] == 0 for r in rows):
        problems.append("all hitter OPS are 0 — stats not parsing")
    if not rows:
        problems.append("no hitters compiled — check boxscore/roster fetch")
    if problems:
        print("POSSIBLE ISSUES:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Looks healthy: real names and non-zero stats are flowing.")
    return 0


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    raise SystemExit(main(date_arg))
