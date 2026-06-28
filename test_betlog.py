"""
test_betlog.py — offline tests for the bet log (temp SQLite, no network).

    python test_betlog.py     # or: pytest test_betlog.py
"""

import os
import tempfile

import betlog as B


def test_crud():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "bets.db")
        bid = B.add_bet(db, player="Altuve", game="HOU @ DET", market="Batter Total Bases",
                        side="Over", line=1.5, entry_odds=120, model_prob=0.58, stake=2.5)
        assert isinstance(bid, int)
        assert len(B.list_bets(db)) == 1
        B.update_bet(bid, db, result="win", close_odds=100)
        bet = B.list_bets(db)[0]
        assert bet["result"] == "win" and bet["close_odds"] == 100
        assert len(B.list_bets(db, settled=True)) == 1
        assert len(B.list_bets(db, settled=False)) == 0
        B.delete_bet(bid, db)
        assert len(B.list_bets(db)) == 0


def test_clv_pct():
    assert B.clv_pct(120, 100) == 10.0       # +120 vs +100 close -> beat by 10%
    assert B.clv_pct(-150, -150) == 0.0      # flat
    assert B.clv_pct(-110, -120) > 0         # took -110, closed -120 -> beat close
    assert B.clv_pct(100, None) is None      # no closing line


def test_bet_pnl():
    assert B.bet_pnl({"result": "win", "stake": 2.5, "entry_odds": 120}) == 3.0
    assert B.bet_pnl({"result": "loss", "stake": 2.0, "entry_odds": -110}) == -2.0
    assert B.bet_pnl({"result": "push", "stake": 2.0, "entry_odds": -110}) == 0.0
    assert B.bet_pnl({"result": None, "stake": 2.0, "entry_odds": -110}) is None  # unsettled


def test_summary():
    bets = [
        {"result": "win", "stake": 2.5, "entry_odds": 120, "close_odds": 100, "model_prob": 0.58},
        {"result": "loss", "stake": 2.0, "entry_odds": -110, "close_odds": -120, "model_prob": 0.64},
        {"result": None, "stake": 2.2, "entry_odds": -150, "close_odds": None, "model_prob": 0.72},
    ]
    s = B.summary(bets)
    assert s["wins"] == 1 and s["losses"] == 1 and s["open"] == 1
    assert s["profit"] == 1.0          # +3.00 win, -2.00 loss
    assert s["clv_n"] == 2             # two bets have closing lines
    assert s["beat_close_rate"] == 100.0  # both beat the close


def test_calibration():
    # 3 buckets, perfectly calibrated within each
    bets = []
    for _ in range(10):
        bets.append({"model_prob": 0.55, "result": "win", "stake": 1, "entry_odds": -110})
    for _ in range(10):
        bets.append({"model_prob": 0.55, "result": "loss", "stake": 1, "entry_odds": -110})
    cal = B.calibration(bets, n_bins=5)
    assert len(cal) == 1               # all in the 0.4-0.6 bucket
    assert cal[0]["n"] == 20
    assert cal[0]["actual"] == 0.5     # 10 wins of 20
    # unsettled bets are excluded
    assert B.calibration([{"model_prob": 0.6, "result": None}], n_bins=5) == []


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
