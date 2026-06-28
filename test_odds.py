"""
test_odds.py — offline tests for odds math + edge join (no network).

    python test_odds.py     # or: pytest test_odds.py
"""

import projections as P
import odds_api as O


def test_implied_prob():
    assert round(O.implied_prob(-110), 4) == 0.5238
    assert round(O.implied_prob(100), 4) == 0.5
    assert round(O.implied_prob(120), 4) == 0.4545


def test_decimal():
    assert O.american_to_decimal(100) == 2.0
    assert O.american_to_decimal(-200) == 1.5
    assert O.american_to_decimal(150) == 2.5


def test_ev_percent():
    assert round(O.ev_percent(0.60, 120), 1) == 32.0
    assert round(O.ev_percent(0.50, -110), 2) == -4.55
    # break-even: fair price for 50% is +100 -> EV 0
    assert round(O.ev_percent(0.50, 100), 6) == 0.0


def test_devig():
    assert O.devig_two_way(-110, -110) == 0.5
    # favorite over: over -200 / under +160 -> fair over > 0.5
    fair = O.devig_two_way(-200, 160)
    assert 0.6 < fair < 0.72


def test_best_price_picks_highest_payout():
    # +120 pays more than -105; +120 should win
    book, price = O._best_price({"a": -105, "b": 120})
    assert price == 120


def test_parse_event_offers():
    event = {
        "bookmakers": [{
            "key": "fanduel",
            "markets": [{
                "key": "batter_hits",
                "outcomes": [
                    {"name": "Over", "description": "Aaron Judge", "point": 0.5, "price": -200},
                    {"name": "Under", "description": "Aaron Judge", "point": 0.5, "price": 160},
                ],
            }],
        }]
    }
    offers = O.parse_event_offers(event)
    assert len(offers) == 1
    o = offers[0]
    assert o["market"] == "batter_hits" and o["point"] == 0.5
    assert o["over"]["fanduel"] == -200 and o["under"]["fanduel"] == 160


def test_compute_edges_matches_and_ranks():
    slug = dict(plateAppearances=600, atBats=540, hits=165, doubles=34, triples=2,
                homeRuns=38, baseOnBalls=55, strikeOuts=140)
    row = {"Hitter": "José Ramírez", "Team": "CLE", "GameLabel": "CLE @ DET",
           "Opp Pitcher": "P", "Lineup": "Confirmed", "_stat": slug, "_exp_pa": 4.5, "_venue_id": None}
    meta = []
    index = P.build_projection_index([row], meta, sims=15000, seed=3)

    offers = [
        # book sends de-accented name; should still match
        {"market": "batter_hits", "player": "Jose Ramirez", "point": 0.5,
         "over": {"fd": -200}, "under": {"fd": 160}},
        # unmatched player
        {"market": "batter_hits", "player": "Nobody Here", "point": 0.5,
         "over": {"fd": -150}, "under": {"fd": 120}},
    ]
    edges, stats = O.compute_edges(index, offers)
    assert stats["matched"] == 1 and stats["unmatched"] == 1
    assert all("EV%" in e for e in edges)
    # sorted by EV% descending
    evs = [e["EV%"] for e in edges]
    assert evs == sorted(evs, reverse=True)
    # model name (with accent) is preserved in output
    assert edges[0]["Player"] == "José Ramírez"


def test_kelly_fraction():
    # p=0.60 at even money (+100): f* = (0.6*2 - 1)/(2-1) = 0.20
    assert abs(O.kelly_fraction(0.60, 100) - 0.20) < 1e-9
    assert O.kelly_fraction(0.50, 100) == 0.0      # fair odds -> no edge
    assert O.kelly_fraction(0.40, 100) == 0.0      # -EV -> clamped to 0


def test_kelly_stake_caps_and_fractions():
    # full f=0.20, quarter -> 0.05; 5% cap is not binding -> 0.05 * 1000 = 50
    assert O.kelly_stake(0.60, 100, 1000, fraction=0.25, cap_pct=0.05) == 50.0
    # tighter 2% cap binds -> 20
    assert O.kelly_stake(0.60, 100, 1000, fraction=0.25, cap_pct=0.02) == 20.0
    # -EV bet -> no stake
    assert O.kelly_stake(0.45, 100, 1000) == 0.0
    # small bankroll, half-Kelly -> small bet
    assert O.kelly_stake(0.58, 120, 50, fraction=0.5, cap_pct=0.05) > 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
