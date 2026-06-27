"""
test_statcast.py — offline tests for the Statcast layer (no Savant, no pybaseball).

    python test_statcast.py     # or: pytest test_statcast.py
"""

import os
import tempfile

import numpy as np
import pandas as pd

import projections as P
import statcast_data as SC


def _write_cache(tmp):
    df = pd.DataFrame([
        dict(player_id=1, name="Elite Barrel", pa=600, brl_pa=0.115, brl_pct=0.22,
             hardhit=0.60, avg_ev=95.8, slg=0.620, xslg=0.640, xiso=0.330),
        dict(player_id=2, name="League Avg", pa=550, brl_pa=0.055, brl_pct=0.08,
             hardhit=0.40, avg_ev=89.0, slg=0.420, xslg=0.415, xiso=0.155),
        dict(player_id=3, name="Slap", pa=500, brl_pa=0.015, brl_pct=0.03,
             hardhit=0.25, avg_ev=86.0, slg=0.350, xslg=0.345, xiso=0.075),
        dict(player_id=4, name="Low PA Noise", pa=40, brl_pa=0.250, brl_pct=0.45,
             hardhit=0.70, avg_ev=99.0, slg=0.800, xslg=0.520, xiso=0.260),
    ])
    path = os.path.join(tmp, "statcast_batters.csv")
    df.to_csv(path, index=False)
    return path


def test_load_and_calibration():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_cache(tmp)
        lookup, k = SC.load(path)
        assert len(lookup) == 4
        assert k is not None and 0.4 < k < 0.8           # plausible barrel->HR factor
        # elite barrels map to a much higher expected HR rate than a slap hitter
        assert SC.expected_hr_rate(lookup[1]["brl_pa"], k) > SC.expected_hr_rate(lookup[3]["brl_pa"], k)


def test_low_pa_excluded_from_calibration():
    # The 40-PA noise guy's 0.25 brl_pa must not drag the league mean (and thus k).
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_cache(tmp)
        _, k = SC.load(path)
        # mean of qualified brl_pa = (0.115+0.055+0.015)/3 = 0.0617 -> k = 0.033/0.0617 ~ 0.535
        assert abs(k - 0.535) < 0.05


def test_missing_file_is_graceful():
    lookup, k = SC.load("/no/such/file.csv")
    assert lookup == {} and k is None


def test_xhr_from_statcast_lookup():
    sc = {99: {"brl_pa": 0.10}}
    assert abs(P.xhr_from_statcast(99, sc, 0.6) - 0.06) < 1e-9
    assert P.xhr_from_statcast(99, sc, None) is None      # no calibration
    assert P.xhr_from_statcast(7, sc, 0.6) is None         # not in lookup
    assert P.xhr_from_statcast(None, sc, 0.6) is None


def test_statcast_regresses_cold_masher_up():
    # Elite contact, cold HR results -> Statcast prior should raise HR probability.
    cold = dict(plateAppearances=500, atBats=450, hits=130, doubles=30, triples=1,
                homeRuns=12, baseOnBalls=45, strikeOuts=110)
    rng = np.random.default_rng(1)
    base = P.batter_pa_probs(cold, P.NEUTRAL_PARK)              # league prior
    juiced = P.batter_pa_probs(cold, P.NEUTRAL_PARK, xhr_pa=0.059)  # barrel-implied prior
    assert juiced[P.HR] > base[P.HR]


def test_statcast_regresses_lucky_hitter_down():
    lucky = dict(plateAppearances=500, atBats=450, hits=120, doubles=20, triples=0,
                 homeRuns=28, baseOnBalls=45, strikeOuts=120)
    base = P.batter_pa_probs(lucky, P.NEUTRAL_PARK)
    pulled = P.batter_pa_probs(lucky, P.NEUTRAL_PARK, xhr_pa=0.024)
    assert pulled[P.HR] < base[P.HR]


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
