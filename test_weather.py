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


def test_full_table_complete():
    # All 30 current parks present, each fully populated, names unique (guards typos/clobbers).
    assert len(W._STATIC_STADIUMS) == 30
    for v in W._STATIC_STADIUMS.values():
        assert {"name", "lat", "lon", "roof", "cf_bearing"} <= set(v)
        assert v["roof"] in ("open", "fixed", "retractable")
    names = [v["name"] for v in W._STATIC_STADIUMS.values()]
    assert len(names) == len(set(names))


def test_name_fallback_resolves():
    # venue_id missing from table, but the name matches -> still resolves.
    def fake(lat, lon, date_str):
        return {"hourly": {"time": ["2026-06-28T22:00"], "temperature_2m": [80],
                           "wind_speed_10m": [5], "wind_direction_10m": [180]}}
    wx = W.get_game_weather(999999, "2026-06-28T22:00:00Z", venue_name="Wrigley Field", fetcher=fake)
    assert wx is not None and wx["park"] == "Wrigley Field"
    # garbage name and id -> None
    assert W.get_game_weather(999999, "2026-06-28T22:00:00Z", venue_name="Nowhere Park", fetcher=fake) is None


def test_json_override_loading():
    import json
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "stadiums.json")
        with open(p, "w") as f:
            json.dump({"2": {"name": "Chase Field", "lat": 33.4, "lon": -112.0,
                             "roof": "retractable", "cf_bearing": 0}}, f)
        ov = W._load_overrides(p)
        assert 2 in ov and ov[2]["name"] == "Chase Field"   # JSON string key -> int
    assert W._load_overrides("/no/such/file.json") == {}     # absent -> empty


def test_sponsor_tolerant_matching():
    # A sponsor-prefixed current name still resolves to the right park...
    k = W._best_name_key(W._norm("UNIQLO Field at Dodger Stadium"), W._BY_NAME)
    assert k and W._BY_NAME[k]["name"] == "Dodger Stadium"
    k2 = W._best_name_key(W._norm("Whatever Co Field at Wrigley Field"), W._BY_NAME)
    assert k2 and W._BY_NAME[k2]["name"] == "Wrigley Field"
    # ...but short generic tokens must NOT match anything.
    assert W._best_name_key("park", W._BY_NAME) is None
    assert W._best_name_key("field", W._BY_NAME) is None


def test_override_merge_corrects_defaulted_bearing():
    # A refreshed entry under a sponsor name with a defaulted 0 bearing should be corrected
    # back to the curated bearing via name match, while keeping the API's id + coords.
    overrides = {111: {"name": "UNIQLO Field at Dodger Stadium", "lat": 34.07, "lon": -118.24,
                       "roof": "open", "cf_bearing": 0}}
    merged = W._merge_overrides(W._STATIC_STADIUMS, overrides)
    assert merged[111]["cf_bearing"] == 25      # corrected from the defaulted 0
    assert merged[111]["lat"] == 34.07          # API coords retained


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
