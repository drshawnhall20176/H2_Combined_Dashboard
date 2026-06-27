"""
test_engine.py — offline unit tests for mlb_engine's pure logic.

No network required. Run either way:
    python test_engine.py        # plain, no dependencies beyond the engine
    pytest test_engine.py         # if you have pytest installed
"""

import mlb_engine as E


# ----------------------------------------------------------------- FIP
def test_fip_known_value():
    # 13*8 + 3*(10+2) - 2*38 = 64; 64/42 = 1.5238; +3.17 = 4.69
    stat = dict(homeRuns=8, baseOnBalls=10, hitByPitch=2, strikeOuts=38, inningsPitched="42.0")
    assert E.calculate_fip(stat, constant=3.17) == 4.69


def test_fip_guards_zero_innings():
    assert E.calculate_fip(dict(inningsPitched=0)) == 0.0
    assert E.calculate_fip({}) == 0.0  # missing fields default to 0


def test_fip_handles_string_inputs():
    stat = dict(homeRuns="8", baseOnBalls="10", hitByPitch="2", strikeOuts="38", inningsPitched="42.0")
    assert E.calculate_fip(stat, constant=3.17) == 4.69


# ----------------------------------------------------------------- innings parsing
def test_parse_innings():
    assert round(E.parse_innings("6.2"), 3) == 6.667   # 6 and 2/3
    assert round(E.parse_innings("85.1"), 3) == 85.333
    assert E.parse_innings("7.0") == 7.0
    assert E.parse_innings(0) == 0.0
    assert E.parse_innings(None) == 0.0


# ----------------------------------------------------------------- platoon
def test_platoon_advantage():
    assert E.platoon_advantage("L", "R") == "Advantage"
    assert E.platoon_advantage("R", "L") == "Advantage"
    assert E.platoon_advantage("R", "R") == "Disadvantage"
    assert E.platoon_advantage("L", "L") == "Disadvantage"
    assert E.platoon_advantage("S", "R") == "Advantage"   # switch always
    assert E.platoon_advantage("S", "L") == "Advantage"
    assert E.platoon_advantage("R", "") == "Unknown"      # missing pitcher hand


# ----------------------------------------------------------------- power index
def test_power_index_ordering():
    slugger = E.power_index(iso=0.260, ops=0.920, advantage="Advantage")
    slap = E.power_index(iso=0.090, ops=0.680, advantage="Disadvantage")
    assert slugger == 42.0
    assert slap == 8.0
    assert slugger > slap


def test_power_index_advantage_bonus():
    with_adv = E.power_index(iso=0.150, ops=0.750, advantage="Advantage")
    without = E.power_index(iso=0.150, ops=0.750, advantage="Disadvantage")
    assert round(with_adv - without, 1) == 5.0


# ----------------------------------------------------------------- helpers
def test_safe_float():
    assert E.safe_float("3.14") == 3.14
    assert E.safe_float(None) == 0.0
    assert E.safe_float("not a number", default=-1) == -1


def test_strip_accents():
    assert E.strip_accents("José Ramírez") == "Jose Ramirez"


# ----------------------------------------------------------------- row assembly
def test_hitter_row_assembly():
    opp = E.PitcherMetrics(id=1, name="Ace", hand="R")
    raw = {
        "id": 99, "name": "José Ramírez", "bat_hand": "S",
        "stat": dict(homeRuns=30, hits=160, totalBases=300, avg="0.280", obp="0.360",
                     slg="0.520", ops="0.880", strikeOuts=90, plateAppearances=650),
    }
    row = E._hitter_row(raw, opp, "Guardians", "CLE @ DET (Game 1)", projected=False)
    assert row["ISO"] == 0.24                 # .520 - .280
    assert round(row["K%"], 3) == 0.138       # 90 / 650
    assert row["Advantage"] == "Advantage"    # switch vs RHP
    assert row["Lineup"] == "Confirmed"       # projected=False
    assert row["Opp Pitcher"] == "Ace"


def test_hitter_row_missing_pa_no_crash():
    # plateAppearances missing -> K% denominator guarded to 1, must not divide by zero
    opp = E.PitcherMetrics(id=1, name="Ace", hand="L")
    raw = {"id": 1, "name": "Rookie", "bat_hand": "R",
           "stat": dict(homeRuns=0, hits=0, totalBases=0, avg="0.000", slg="0.000",
                        ops="0.000", strikeOuts=0)}
    row = E._hitter_row(raw, opp, "Team", "label", projected=True)
    assert row["Lineup"] == "Projected"
    assert row["K%"] == 0.0


# ----------------------------------------------------------------- runner
def test_aggregate_pitching_splits():
    # Traded pitcher: two stints must sum, with innings rebuilt in MLB thirds format.
    splits = [
        {"stat": dict(strikeOuts=60, baseOnBalls=15, homeRuns=8, battersFaced=300,
                      gamesStarted=11, earnedRuns=30, hits=70, atBats=270,
                      inningsPitched="72.1", hitByPitch=3)},
        {"stat": dict(strikeOuts=40, baseOnBalls=10, homeRuns=5, battersFaced=200,
                      gamesStarted=7, earnedRuns=18, hits=45, atBats=180,
                      inningsPitched="48.2", hitByPitch=2)},
    ]
    agg = E._aggregate_pitching_splits(splits)
    assert agg["strikeOuts"] == 100 and agg["battersFaced"] == 500 and agg["gamesStarted"] == 18
    # 72.1 (72+1/3) + 48.2 (48+2/3) = 121.0 innings exactly
    assert agg["inningsPitched"] == "121.0"
    assert round(E.parse_innings(agg["inningsPitched"]), 2) == 121.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
