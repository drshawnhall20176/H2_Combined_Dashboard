"""
test_projections.py — offline tests for the projection engine (seeded, deterministic).

    python test_projections.py     # or: pytest test_projections.py
"""

import numpy as np
import projections as P


def _slugger():
    return dict(plateAppearances=600, atBats=540, hits=165, doubles=34, triples=2,
                homeRuns=38, baseOnBalls=55, strikeOuts=140)


def test_pa_probs_sum_to_one():
    probs = P.batter_pa_probs(_slugger(), P.NEUTRAL_PARK)
    assert probs is not None
    assert abs(probs.sum() - 1.0) < 1e-9
    assert (probs >= 0).all()


def test_low_sample_returns_none():
    assert P.batter_pa_probs(dict(plateAppearances=10), P.NEUTRAL_PARK) is None


def test_park_boosts_hr_rate():
    neutral = P.batter_pa_probs(_slugger(), P.NEUTRAL_PARK)[P.HR]
    coors = P.batter_pa_probs(_slugger(), P.PARK_FACTORS[7])[P.HR]
    assert coors > neutral  # Coors (venue 7) inflates HR


def test_probabilities_in_range():
    rng = np.random.default_rng(0)
    probs = P.batter_pa_probs(_slugger(), P.NEUTRAL_PARK)
    sim = P.simulate_batter(probs, 4.4, 20000, rng)
    for arr, line in ((sim["tb"], 1.5), (sim["hits"], 0.5), (sim["k"], 0.5)):
        p = float(np.mean(arr > line))
        assert 0.0 <= p <= 1.0
    hr_p = float(np.mean(sim["hr"] >= 1))
    assert 0.15 < hr_p < 0.40  # a 38-HR bat sits in a believable anytime-HR band


def test_pitcher_projection_sane():
    ace = dict(battersFaced=720, inningsPitched="180.0", gamesStarted=29,
               strikeOuts=235, baseOnBalls=42)
    proj = P.project_pitcher(ace)
    assert 5.0 < proj["exp_ip"] < 7.0
    assert proj["exp_k"] > proj["exp_bb"]


def test_fair_odds_roundtrip():
    assert P.prob_to_american(0.5) == -100
    assert P.prob_to_decimal(0.5) == 2.0
    # favorite gets negative american, dog positive
    assert P.prob_to_american(0.62) < 0
    assert P.prob_to_american(0.30) > 0


def test_build_signals_shape():
    row = {"Hitter": "X", "Team": "T", "GameLabel": "A @ B", "Opp Pitcher": "P",
           "Lineup": "Confirmed", "_stat": _slugger(), "_exp_pa": 4.4, "_venue_id": None}

    class PM:
        id = 1; name = "Ace"
        stat = dict(battersFaced=700, inningsPitched="170.0", gamesStarted=28,
                    strikeOuts=200, baseOnBalls=50)
    meta = [{"label": "A @ B", "home_name": "B", "away_name": "A", "home_pm": PM(), "away_pm": PM()}]
    sigs = P.build_signals([row], meta, sims=8000, seed=1)
    # 4 batter markets + 3 pitcher markets * 2 pitchers = 10
    assert len(sigs) == 10
    for s in sigs:
        assert 0.0 <= s["ModelProb"] <= 1.0
        assert s["FairDec"] is None or s["FairDec"] >= 1.0


def test_starter_gate_skips_relievers():
    # Pure reliever (no starts) listed as a probable -> must be skipped, not projected.
    assert P.project_pitcher(dict(battersFaced=45, inningsPitched="11.0",
                                  gamesStarted=0, strikeOuts=18, baseOnBalls=9)) is None
    # Too-thin starter sample -> skipped.
    assert P.project_pitcher(dict(battersFaced=50, inningsPitched="12.0",
                                  gamesStarted=2, strikeOuts=17, baseOnBalls=8)) is None


def test_shrinkage_tames_hot_small_sample():
    # 3 starts, 15 IP, 40% raw K rate. Pre-fix this exploded toward ~8.5+ K.
    proj = P.project_pitcher(dict(battersFaced=63, inningsPitched="15.0",
                                  gamesStarted=3, strikeOuts=25, baseOnBalls=4))
    assert proj is not None
    assert proj["exp_k"] < 7.0   # regressed, not inflated
    # And never above the clamp ceiling.
    assert proj["exp_k"] <= 0.45 * proj["exp_bf"] + 1e-9


def test_real_ace_stays_sane():
    proj = P.project_pitcher(dict(battersFaced=720, inningsPitched="180.0",
                                  gamesStarted=29, strikeOuts=235, baseOnBalls=42))
    assert 5.5 < proj["exp_ip"] < 7.0
    assert 6.0 < proj["exp_k"] < 9.5      # believable ace strikeout projection
    assert proj["exp_bb"] < 3.0


def test_batter_shrinkage_pulls_small_samples():
    # A 40-PA hot callup's HR rate must regress well below its raw 0.125.
    hot = P.batter_pa_probs(dict(plateAppearances=40, atBats=36, hits=16, doubles=4,
                                 triples=0, homeRuns=5, baseOnBalls=4, strikeOuts=6),
                            P.NEUTRAL_PARK)
    assert hot[P.HR] < 0.07          # far below raw 5/40 = 0.125
    # A full-season slugger barely moves.
    slug = P.batter_pa_probs(dict(plateAppearances=600, atBats=540, hits=165, doubles=34,
                                  triples=2, homeRuns=38, baseOnBalls=55, strikeOuts=140),
                             P.NEUTRAL_PARK)
    assert slug[P.HR] > 0.05         # still clearly an above-average power bat


def test_odds_ratio_identity():
    # League bat vs league pitcher at league rate must return league rate.
    assert abs(P.odds_ratio(0.033, 0.033, 0.033) - 0.033) < 1e-6
    # Better pitcher (lower allowed) pushes the matchup probability down.
    assert P.odds_ratio(0.065, 0.020, 0.033) < P.odds_ratio(0.065, 0.045, 0.033)


def test_matchup_moves_hr_with_pitcher_quality():
    slug = _slugger()
    hr_prone = dict(battersFaced=700, homeRuns=28, strikeOuts=120, baseOnBalls=70, hits=180)
    ace = dict(battersFaced=700, homeRuns=10, strikeOuts=210, baseOnBalls=40, hits=130)
    p_easy = P.batter_pa_probs(slug, P.NEUTRAL_PARK, opp_allowed=P.pitcher_allowed_rates(hr_prone))
    p_hard = P.batter_pa_probs(slug, P.NEUTRAL_PARK, opp_allowed=P.pitcher_allowed_rates(ace))
    # Same hitter projects for MORE HR vs the homer-prone arm, FEWER vs the ace.
    assert p_easy[P.HR] > p_hard[P.HR]
    # And more strikeouts vs the high-K ace.
    assert p_hard[P.K] > p_easy[P.K]


def test_pitcher_allowed_rates_guards():
    assert P.pitcher_allowed_rates(None) is None
    assert P.pitcher_allowed_rates(dict(battersFaced=10)) is None  # too thin


def test_handedness_split_applies():
    slug = _slugger()
    vs_r = dict(plateAppearances=420, atBats=380, hits=130, doubles=28, triples=1,
                homeRuns=32, baseOnBalls=35, strikeOuts=80)
    vs_l = dict(plateAppearances=180, atBats=165, hits=35, doubles=6, triples=0,
                homeRuns=6, baseOnBalls=12, strikeOuts=55)
    p_vsR = P.batter_pa_probs(slug, P.NEUTRAL_PARK, split_stat=vs_r)
    p_vsL = P.batter_pa_probs(slug, P.NEUTRAL_PARK, split_stat=vs_l)
    assert p_vsR[P.HR] > p_vsL[P.HR]   # mashes RHP
    assert p_vsL[P.K] > p_vsR[P.K]     # whiffs vs LHP
    # A tiny split sample barely moves the season rate (shrinkage).
    tiny = dict(plateAppearances=8, atBats=7, hits=5, doubles=2, triples=0,
                homeRuns=3, baseOnBalls=1, strikeOuts=1)
    base = P.batter_pa_probs(slug, P.NEUTRAL_PARK)
    p_tiny = P.batter_pa_probs(slug, P.NEUTRAL_PARK, split_stat=tiny)
    assert abs(p_tiny[P.HR] - base[P.HR]) < 0.01


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
