"""
test_weather.py — offline tests for weather math and parsing (no network).

    python test_weather.py    # or: pytest test_weather.py
"""

import weather as W


def test_wind_out_component():
    # CF due north (bearing 0). Wind FROM south blows straight out -> +full.
    assert round(W.wind_out_component(10, 180, 0), 1) == 10.0
    # Wind FROM north blows straight in -> -full.
    assert round(W.wind_out_component(10, 0, 0), 1) == -10.0
    # Wind FROM west is a crosswind -> ~0 out component.
    assert abs(W.wind_out_component(10, 270, 0)) < 0.01


def test_hr_factor():
    assert W.hr_factor(70, 0, "open") == 1.0                  # baseline neutral
    assert W.hr_factor(90, 0, "open") > 1.0                   # heat helps
    assert W.hr_factor(50, -10, "open") < 1.0                 # cold + wind in suppresses
    assert W.hr_factor(90, 10, "fixed") == 1.0               # dome ignores weather
    assert W.hr_factor(120, 40, "open") == W.HR_FACTOR_MAX    # clamp at the top
    assert W.hr_factor(20, -40, "open") == W.HR_FACTOR_MIN    # clamp at the bottom


def test_get_game_weather_parsing():
    def fake(lat, lon, date_str):
        return {"hourly": {
            "time": ["2026-06-28T21:00", "2026-06-28T22:00", "2026-06-28T23:00"],
            "temperature_2m": [78, 82, 85],
            "wind_speed_10m": [8, 12, 10],
            "wind_direction_10m": [180, 180, 200],
        }}
    wx = W.get_game_weather(12, "2026-06-28T22:00:00Z", fetcher=fake)  # Coors, cf_bearing 0
    assert wx["temp_f"] == 82 and wx["wind_mph"] == 12
    assert wx["out_wind_mph"] > 0          # wind from south = out to CF
    assert wx["hr_factor"] > 1.0


def test_graceful_degradation():
    assert W.get_game_weather(999999, "2026-06-28T22:00:00Z", fetcher=lambda *a: {}) is None
    assert W.get_game_weather(None, None) is None
    # fixed dome short-circuits without any fetch
    wx = W.get_game_weather(5325, "2026-06-28T22:00:00Z", fetcher=lambda *a: 1 / 0)
    assert wx["hr_factor"] == 1.0 and wx["dome"] is True


def test_no_duplicate_or_clobbered_parks():
    # the table should have distinct, populated parks (guards the placeholder-key bug)
    assert all("lat" in v for v in W.STADIUMS.values())
    assert W.STADIUMS[2]["name"] == "Chase Field"   # not clobbered by a placeholder


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
