"""Read-only Home Assistant cache + value formatting for /frame/state."""

from __future__ import annotations

import http.client
import json
import logging
import urllib.error
from datetime import timedelta, timezone

import pytest

from wildlife.config import HAEntityConfig, HomeAssistantConfig
from wildlife.frame.ha import HACache, compass, display_value, flow_value, sun_text, volts_value


class FakeHA:
    """Stands in for ``urllib_fetch``: serves canned states and the forecast service."""

    def __init__(self, states: dict, forecast: list | None = None) -> None:
        self.states = states
        self.forecast = forecast
        self.down = False
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url))
        assert headers["Authorization"] == "Bearer tok"
        assert timeout == 3.0
        if self.down:
            raise urllib.error.URLError("connection refused")
        if "/api/services/weather/get_forecasts" in url:
            if self.forecast is None:
                raise urllib.error.HTTPError(url, 400, "Bad Request", None, None)
            entity_id = json.loads(body)["entity_id"]
            return {"changed_states": [], "service_response": {entity_id: {"forecast": self.forecast}}}
        entity_id = url.rsplit("/", 1)[1]
        if entity_id not in self.states:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return self.states[entity_id]


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


OUT = HAEntityConfig(id="sensor.outdoor_temp", slot="temps", label="Outside")
WX = HAEntityConfig(id="weather.home", slot="weather")
STATES = {
    "sensor.outdoor_temp": {
        "entity_id": "sensor.outdoor_temp",
        "state": "68.4",
        "attributes": {"unit_of_measurement": "°F", "friendly_name": "Outdoor", "token_ish": "x"},
    },
    "weather.home": {
        "entity_id": "weather.home",
        "state": "partlycloudy",
        "attributes": {
            "temperature": 72, "temperature_unit": "°F", "humidity": 30,
            "friendly_name": "Home", "entity_picture": "/api/x",
        },
    },
}


def _ha(*entities: HAEntityConfig) -> HomeAssistantConfig:
    return HomeAssistantConfig(url="http://ha:8123", token="tok", entities=list(entities))


def test_snapshot_returns_allow_listed_items_with_filtered_attrs() -> None:
    fake = FakeHA(STATES)
    snap = HACache(fetch=fake, clock=Clock()).snapshot(_ha(OUT, WX), refresh_seconds=15)
    assert (snap["ok"], snap["stale"], snap["age_seconds"]) == (True, False, 0)
    assert snap["items"][0] == {
        "slot": "temps", "label": "Outside", "state": "68.4", "unit": "°F",
        "attrs": {}, "display": "68.4°", "flow": None, "detail": "",
        "icon": None, "icon_tone": None, "level": None, "value_tone": None, "flow_tone": None,
        "volts": None,
    }
    weather = snap["items"][1]
    assert weather["attrs"] == {"temperature": 72, "temperature_unit": "°F", "humidity": 30}
    assert weather["label"] == "Home"  # falls back to friendly_name
    assert (weather["display"], weather["detail"]) == ("72°", "Partly cloudy")
    state_urls = {u for m, u in fake.calls if m == "GET"}
    assert state_urls == {
        "http://ha:8123/api/states/sensor.outdoor_temp",
        "http://ha:8123/api/states/weather.home",
    }


def test_results_are_cached_for_refresh_seconds() -> None:
    fake, clock = FakeHA(STATES), Clock()
    cache = HACache(fetch=fake, clock=clock)
    cache.snapshot(_ha(OUT), 15)
    n = len(fake.calls)
    clock.t += 14
    cache.snapshot(_ha(OUT), 15)
    assert len(fake.calls) == n
    clock.t += 1
    cache.snapshot(_ha(OUT), 15)
    assert len(fake.calls) == n + 1


def test_ha_down_serves_last_good_values_marked_stale() -> None:
    fake, clock = FakeHA(STATES), Clock()
    cache = HACache(fetch=fake, clock=clock)
    cache.snapshot(_ha(OUT), 15)
    fake.down = True
    clock.t += 20
    snap = cache.snapshot(_ha(OUT), 15)
    assert (snap["ok"], snap["stale"], snap["age_seconds"]) == (False, True, 20)
    assert snap["items"][0]["state"] == "68.4"


def test_ha_down_before_first_success_returns_empty() -> None:
    fake = FakeHA(STATES)
    fake.down = True
    snap = HACache(fetch=fake, clock=Clock()).snapshot(_ha(OUT), 15)
    assert snap == {"ok": False, "stale": True, "age_seconds": None, "items": [], "forecast": None}


def test_unknown_entity_does_not_break_the_strip() -> None:  # Review Focus 1
    bad = HAEntityConfig(id="sensor.typo_temp", slot="temps", label="Typo")
    snap = HACache(fetch=FakeHA(STATES), clock=Clock()).snapshot(_ha(OUT, bad), 15)
    assert snap["ok"] is True
    assert snap["items"][1] == {
        "slot": "temps", "label": "Typo", "state": None, "unit": None,
        "attrs": {}, "display": "—", "flow": None, "detail": "",
        "icon": None, "icon_tone": None, "level": None, "value_tone": None, "flow_tone": None,
        "volts": None,
    }


def test_config_change_invalidates_cache() -> None:  # Review Focus 5
    fake, clock = FakeHA(STATES), Clock()
    cache = HACache(fetch=fake, clock=clock)
    cache.snapshot(_ha(OUT), 15)
    snap = cache.snapshot(_ha(OUT, WX), 15)  # same instant, new allow-list
    assert [i["slot"] for i in snap["items"]] == ["temps", "weather"]


def test_unconfigured_ha_makes_no_requests() -> None:
    fake = FakeHA(STATES)
    snap = HACache(fetch=fake, clock=Clock()).snapshot(HomeAssistantConfig(), 15)
    assert fake.calls == []
    assert (snap["ok"], snap["items"], snap["forecast"]) == (False, [], None)


def test_forecast_high_low_from_first_daily_entry_cached_30_minutes() -> None:
    fake = FakeHA(STATES, forecast=[{"temperature": 78, "templow": 51}, {"temperature": 70}])
    clock = Clock()
    cache = HACache(fetch=fake, clock=clock)

    def posts() -> int:
        return sum(1 for m, _ in fake.calls if m == "POST")

    assert cache.snapshot(_ha(WX), 15)["forecast"] == {
        "high": 78, "low": 51, "display": "H 78°  L 51°"
    }
    assert posts() == 1
    clock.t += 1799
    cache.snapshot(_ha(WX), 15)
    assert posts() == 1
    clock.t += 1
    cache.snapshot(_ha(WX), 15)
    assert posts() == 2


def test_forecast_failure_does_not_affect_states() -> None:  # Review Focus 3
    snap = HACache(fetch=FakeHA(STATES, forecast=None), clock=Clock()).snapshot(_ha(WX), 15)
    assert snap["ok"] is True and snap["forecast"] is None
    assert snap["items"][0]["state"] == "partlycloudy"


def test_forecast_retried_after_a_minute_when_it_fails() -> None:
    fake, clock = FakeHA(STATES, forecast=None), Clock()
    cache = HACache(fetch=fake, clock=clock)
    cache.snapshot(_ha(WX), 15)
    fake.forecast = [{"temperature": 80, "templow": 50}]
    clock.t += 59
    assert cache.snapshot(_ha(WX), 15)["forecast"] is None
    clock.t += 1
    assert cache.snapshot(_ha(WX), 15)["forecast"] == {
        "high": 80, "low": 50, "display": "H 80°  L 50°"
    }


def test_no_forecast_request_without_weather_entity() -> None:
    fake = FakeHA(STATES)
    HACache(fetch=fake, clock=Clock()).snapshot(_ha(OUT), 15)
    assert all(m == "GET" for m, _ in fake.calls)


@pytest.mark.parametrize(  # Review Focus 2
    "slot,state,unit,attrs,expected",
    [
        ("temps", None, "°F", {}, "—"),
        ("temps", "unavailable", "°F", {}, "—"),
        ("temps", "unknown", None, {}, "—"),
        ("temps", "68.44", "°F", {}, "68.4°"),
        ("temps", "68.0", "°C", {}, "68°"),
        ("energy", "1234", "W", {}, "1.2 kW"),
        ("energy", "950", "W", {}, "950 W"),
        ("energy", "12.345", "kWh", {}, "12.3 kWh"),
        ("energy", "charging", None, {}, "charging"),
        ("energy", "98", "%", {}, "98%"),
        ("rain", "0.12", "in", {}, '0.12"'),
        ("rain", "0", "in", {}, '0.00"'),
        ("rain", "0.004", "in/h", {}, '0.00"/h'),
        ("rain", "1.255", "in", {}, '1.25"'),
        ("rain", "3.2", "mm", {}, "3.2 mm"),
        ("rain", "0.4", "mm/h", {}, "0.4 mm/h"),
        ("rain", "unavailable", "in", {}, "—"),
        ("energy", "98.44", "%", {}, "98.4%"),
        ("energy", "nan", "W", {}, "—"),
        ("temps", "heat", None, {"current_temperature": 70.5, "hvac_action": "heating"},
         "70.5° · heating"),
        ("temps", "off", None, {"current_temperature": 66}, "66°"),
        ("weather", "sunny", None, {"temperature": 72.2}, "72.2°"),
        ("weather", "sunny", None, {}, "—"),
    ],
)
def test_display_value(slot, state, unit, attrs, expected) -> None:
    assert display_value(slot, state, unit, attrs) == expected


def test_unexpected_http_errors_mark_stale_instead_of_raising() -> None:  # final review #1
    class Broken(FakeHA):
        def __call__(self, method, url, headers, body, timeout):
            raise http.client.IncompleteRead(b"partial")

    snap = HACache(fetch=Broken(STATES), clock=Clock()).snapshot(_ha(OUT, WX), 15)
    assert (snap["ok"], snap["stale"], snap["items"], snap["forecast"]) == (False, True, [], None)


@pytest.mark.parametrize(  # final review #2
    "attrs", [{"current_temperature": None, "hvac_action": "idle"}, {"current_temperature": "n/a"}]
)
def test_climate_without_a_reading_shows_a_dash(attrs: dict) -> None:
    assert display_value("temps", "heat", None, attrs) == "—"


def test_weather_with_non_numeric_temperature_shows_a_dash() -> None:
    assert display_value("weather", "sunny", None, {"temperature": "n/a"}) == "—"


@pytest.mark.parametrize(  # final review: a missing low rendered as "L 0°"
    "today,expected",
    [
        ({"temperature": 78, "templow": 51}, "H 78°  L 51°"),
        ({"temperature": 78.4, "templow": None}, "H 78°"),
        ({"temperature": None, "templow": 51}, "L 51°"),
        ({"temperature": "n/a"}, None),
    ],
)
def test_forecast_display_never_invents_values(today: dict, expected: str | None) -> None:
    snap = HACache(fetch=FakeHA(STATES, forecast=[today]), clock=Clock()).snapshot(_ha(WX), 15)
    assert (snap["forecast"] or {}).get("display") == expected


def test_old_forecast_is_dropped_after_six_hours_of_failures() -> None:  # minor #8
    fake, clock = FakeHA(STATES, forecast=[{"temperature": 78, "templow": 51}]), Clock()
    cache = HACache(fetch=fake, clock=clock)
    assert cache.snapshot(_ha(WX), 15)["forecast"]["display"] == "H 78°  L 51°"
    fake.forecast = None  # the forecast service starts failing
    clock.t += 6 * 3600 - 1
    assert cache.snapshot(_ha(WX), 15)["forecast"] is not None
    clock.t += 1
    assert cache.snapshot(_ha(WX), 15)["forecast"] is None


def test_ha_outage_logs_once_and_recovery_logs_once(caplog) -> None:  # minor #10
    fake, clock = FakeHA(STATES), Clock()
    cache = HACache(fetch=fake, clock=clock)
    cache.snapshot(_ha(OUT), 15)
    fake.down = True
    with caplog.at_level(logging.INFO, logger="wildlife.frame.ha"):
        for _ in range(5):
            clock.t += 15
            cache.snapshot(_ha(OUT), 15)
        fake.down = False
        clock.t += 15
        cache.snapshot(_ha(OUT), 15)
    messages = [r.getMessage() for r in caplog.records]
    assert sum("unavailable" in m for m in messages) == 1
    assert sum("reachable again" in m for m in messages) == 1


def test_unknown_entity_is_logged_once_per_config(caplog) -> None:  # minor #10
    bad = HAEntityConfig(id="sensor.typo_temp", slot="temps")
    fake, clock = FakeHA(STATES), Clock()
    cache = HACache(fetch=fake, clock=clock)
    with caplog.at_level(logging.WARNING, logger="wildlife.frame.ha"):
        for _ in range(3):
            cache.snapshot(_ha(OUT, bad), 15)
            clock.t += 15
    assert sum("sensor.typo_temp" in r.getMessage() for r in caplog.records) == 1


def test_repeated_forecast_failures_log_once(caplog) -> None:  # minor #10
    fake, clock = FakeHA(STATES, forecast=None), Clock()
    cache = HACache(fetch=fake, clock=clock)
    with caplog.at_level(logging.INFO, logger="wildlife.frame.ha"):
        for _ in range(4):
            cache.snapshot(_ha(WX), 15)
            clock.t += 61
    assert sum("forecast" in r.getMessage().lower() for r in caplog.records) == 1


# ---- battery flow (Victron shunt power: positive = charging) ------------------

SOC = HAEntityConfig(
    id="sensor.shop_battery_soc", slot="energy", label="Shop battery", flow="sensor.shop_shunt_power"
)


def _battery_states(power: str = "1312.4") -> dict:
    return {
        "sensor.shop_battery_soc": {
            "entity_id": "sensor.shop_battery_soc",
            "state": "75.3",
            "attributes": {"unit_of_measurement": "%"},
        },
        "sensor.shop_shunt_power": {
            "entity_id": "sensor.shop_shunt_power",
            "state": power,
            "attributes": {"unit_of_measurement": "W"},
        },
    }


@pytest.mark.parametrize(
    ("state", "unit", "expected"),
    [
        ("98.2", "W", "+98 W"),
        ("-420.4", "W", "−420 W"),
        ("1312.4", "W", "+1.3 kW"),
        ("-1000", "W", "−1 kW"),
        ("999.6", "W", "+1 kW"),  # rounds up into kW, never "+1000 W"
        ("-10512", "W", "−11 kW"),  # whole kW from 10 kW so "100%  −10.5 kW" can't wrap
        ("9960", "W", "+10 kW"),
        ("0.3", "W", "0 W"),
        ("-0.4", "W", "0 W"),  # never "−0 W"
        ("1.31", "kW", "+1.3 kW"),
        ("-0.42", "kW", "−420 W"),
        ("unavailable", "W", None),
        ("unknown", "W", None),
        (None, "W", None),
        ("nan", "W", None),
        ("inf", "W", None),
        ("charging", "W", None),
        ("-6.589", "A", "−6.6 A"),  # shunt current: same sign rule as power
        ("1.3", "A", "+1.3 A"),
        ("0.04", "A", "0 A"),
        ("-0.04", "A", "0 A"),  # never "−0 A"
        ("-226.4", "A", "−226 A"),  # whole amps from 100 A so the row can't wrap
        ("99.96", "A", "+100 A"),
        ("unavailable", "A", None),
        ("98", "V", None),  # neither power nor current
        ("98", None, None),
    ],
)
def test_flow_value(state, unit, expected) -> None:
    assert flow_value(state, unit) == expected


def test_battery_row_carries_signed_flow_and_whole_percent() -> None:
    fake = FakeHA(_battery_states())
    item = HACache(fetch=fake, clock=Clock()).snapshot(_ha(SOC), 15)["items"][0]
    assert (item["display"], item["flow"]) == ("75%", "+1.3 kW")
    assert {url.rsplit("/", 1)[1] for _, url in fake.calls} == {
        "sensor.shop_battery_soc",
        "sensor.shop_shunt_power",
    }


def test_rows_without_a_flow_sensor_are_unchanged() -> None:
    item = HACache(fetch=FakeHA(STATES), clock=Clock()).snapshot(_ha(OUT), 15)["items"][0]
    assert (item["display"], item["flow"]) == ("68.4°", None)


@pytest.mark.parametrize("broken", ["unavailable", "missing"])
def test_dead_shunt_leaves_just_the_percent(broken: str) -> None:
    states = _battery_states("unavailable")
    if broken == "missing":
        del states["sensor.shop_shunt_power"]  # HA answers 404
    snap = HACache(fetch=FakeHA(states), clock=Clock()).snapshot(_ha(SOC, OUT), 15)
    assert snap["ok"] is True
    assert [(i["display"], i["flow"]) for i in snap["items"]] == [("75%", None), ("—", None)]


def test_changing_the_flow_sensor_invalidates_cache() -> None:
    states = _battery_states()
    states["sensor.shed_shunt_power"] = {
        "entity_id": "sensor.shed_shunt_power",
        "state": "-420",
        "attributes": {"unit_of_measurement": "W"},
    }
    cache = HACache(fetch=FakeHA(states), clock=Clock())
    cache.snapshot(_ha(SOC), 15)
    moved = SOC.model_copy(update={"flow": "sensor.shed_shunt_power"})
    assert cache.snapshot(_ha(moved), 15)["items"][0]["flow"] == "−420 W"


# ---- weather conditions rows (wind, sun, humidity, pressure) ---------------

MDT = timezone(timedelta(hours=-6))


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [
        (0, "N"), (19, "NNE"), (11.2, "N"), (11.3, "NNE"), (90, "E"), (202.5, "SSW"),
        (348.75, "N"), (359, "N"), ("19", "NNE"), (None, None), ("unknown", None), ("nan", None),
    ],
)
def test_compass(degrees, expected) -> None:
    assert compass(degrees) == expected


@pytest.mark.parametrize(
    ("state", "unit", "expected"),
    [
        ("29.811", "inHg", "29.81 inHg"),  # pressure needs hundredths, like the rain gauge
        ("1.1", "mph", "1 mph"),  # wind in whole mph
        ("13.6", "mph", "14 mph"),
        ("96", "%", "96%"),
    ],
)
def test_conditions_values(state, unit, expected) -> None:
    assert display_value("conditions", state, unit, {}) == expected


def test_sun_text_shows_next_rise_and_set_in_local_time() -> None:
    attrs = {"next_rising": "2026-10-14T13:12:31.5+00:00", "next_setting": "2026-10-14T00:31:02+00:00"}
    assert sun_text(attrs, MDT) == "↑ 7:12 AM · ↓ 6:31 PM"


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"next_setting": "2026-10-14T00:31:02+00:00"}, "↓ 6:31 PM"),
        ({}, "—"),
        ({"next_rising": "garbage", "next_setting": None}, "—"),
    ],
)
def test_sun_text_leaves_out_missing_parts(attrs: dict, expected: str) -> None:
    assert sun_text(attrs, MDT) == expected


WIND = HAEntityConfig(
    id="sensor.wind_speed", slot="conditions", label="Wind",
    direction="sensor.wind_direction", gust="sensor.max_gust",
)


def _wind_states(speed: str = "1.1", direction: str = "19", gust: str = "13.6") -> dict:
    def row(entity_id: str, state: str, unit: str) -> dict:
        return {"entity_id": entity_id, "state": state, "attributes": {"unit_of_measurement": unit}}

    return {
        "sensor.wind_speed": row("sensor.wind_speed", speed, "mph"),
        "sensor.wind_direction": row("sensor.wind_direction", direction, "°"),
        "sensor.max_gust": row("sensor.max_gust", gust, "mph"),
    }


def test_wind_row_combines_speed_compass_and_gust() -> None:
    fake = FakeHA(_wind_states())
    item = HACache(fetch=fake, clock=Clock()).snapshot(_ha(WIND), 15)["items"][0]
    assert item["display"] == "1 mph NNE · gust 14"
    assert {url.rsplit("/", 1)[1] for _, url in fake.calls} == {
        "sensor.wind_speed", "sensor.wind_direction", "sensor.max_gust",
    }


def test_wind_row_drops_missing_companions() -> None:
    states = _wind_states(gust="unavailable")
    del states["sensor.wind_direction"]  # HA answers 404
    item = HACache(fetch=FakeHA(states), clock=Clock()).snapshot(_ha(WIND), 15)["items"][0]
    assert item["display"] == "1 mph"


def test_wind_row_without_a_speed_is_a_dash() -> None:
    item = HACache(fetch=FakeHA(_wind_states(speed="unknown")), clock=Clock()).snapshot(_ha(WIND), 15)
    assert item["items"][0]["display"] == "—"


def test_sun_row_through_the_cache_keeps_its_attributes_private() -> None:
    sun = HAEntityConfig(id="sun.sun", slot="conditions", label="Sun")
    states = {
        "sun.sun": {
            "entity_id": "sun.sun",
            "state": "above_horizon",
            "attributes": {
                "next_rising": "2026-10-14T13:12:31+00:00",
                "next_setting": "2026-10-14T00:31:02+00:00",
                "elevation": 31.2,
                "friendly_name": "Sun",
            },
        }
    }
    item = HACache(fetch=FakeHA(states), clock=Clock(), tz=MDT).snapshot(_ha(sun), 15)["items"][0]
    assert (item["label"], item["display"], item["attrs"]) == ("Sun", "↑ 7:12 AM · ↓ 6:31 PM", {})


def test_changing_wind_companions_invalidates_cache() -> None:
    states = _wind_states()
    states["sensor.other_gust"] = {"entity_id": "sensor.other_gust", "state": "30.2",
                                   "attributes": {"unit_of_measurement": "mph"}}
    cache = HACache(fetch=FakeHA(states), clock=Clock())
    cache.snapshot(_ha(WIND), 15)
    moved = WIND.model_copy(update={"gust": "sensor.other_gust"})
    assert cache.snapshot(_ha(moved), 15)["items"][0]["display"] == "1 mph NNE · gust 30"


# ---- design pass: icons, tones, battery level, whole watts ------------------

def _one(state: str, unit: str | None, **entity) -> dict:
    e = HAEntityConfig(id="sensor.thing", slot=entity.pop("slot", "energy"), label="Thing", **entity)
    states = {"sensor.thing": {"entity_id": "sensor.thing", "state": state,
                               "attributes": {"unit_of_measurement": unit} if unit else {}}}
    return HACache(fetch=FakeHA(states), clock=Clock()).snapshot(_ha(e), 15)["items"][0]


@pytest.mark.parametrize(
    ("state", "expected"),
    [("574.3", "574 W"), ("-0.3", "0 W"), ("999.4", "999 W"), ("999.6", "1 kW"), ("1312", "1.3 kW")],
)
def test_watts_are_whole_numbers(state: str, expected: str) -> None:
    assert display_value("energy", state, "W", {}) == expected


@pytest.mark.parametrize(
    ("soc", "level", "tone"),
    [("75.3", 75.3, "ok"), ("50", 50.0, "ok"), ("29", 29.0, "low"), ("25", 25.0, "low"),
     ("12.5", 12.5, "critical"), ("104", 100.0, "ok"), ("-3", 0.0, "critical"),
     ("unavailable", None, None)],
)
def test_battery_icon_fills_to_the_charge_in_its_level_colour(soc: str, level, tone) -> None:
    item = _one(soc, "%", icon="battery")
    assert (item["icon"], item["level"], item["icon_tone"]) == ("battery", level, tone)


@pytest.mark.parametrize(("watts", "tone"), [("672.4", "sun"), ("0", None), ("unknown", None)])
def test_solar_value_is_sun_coloured_only_while_producing(watts: str, tone) -> None:
    assert _one(watts, "W", icon="sun")["value_tone"] == tone


def test_flow_tone_marks_charging_only() -> None:
    states = _battery_states("12.0")
    states["sensor.shop_shunt_power"]["attributes"]["unit_of_measurement"] = "A"
    charging = HACache(fetch=FakeHA(states), clock=Clock()).snapshot(_ha(SOC), 15)["items"][0]
    assert (charging["flow"], charging["flow_tone"]) == ("+12 A", "in")
    states["sensor.shop_shunt_power"]["state"] = "-11.1"
    discharging = HACache(fetch=FakeHA(states), clock=Clock()).snapshot(_ha(SOC), 15)["items"][0]
    assert (discharging["flow"], discharging["flow_tone"]) == ("−11.1 A", None)  # normal at night


def test_plain_rows_have_no_icon_or_tones() -> None:
    item = _one("68.4", "°F", slot="temps")
    assert {k: item[k] for k in ("icon", "icon_tone", "level", "value_tone", "flow_tone")} == {
        "icon": None, "icon_tone": None, "level": None, "value_tone": None, "flow_tone": None,
    }


@pytest.mark.parametrize(
    ("condition", "icon", "tone"),
    [("rainy", "rain", "sky"), ("pouring", "rain", "sky"), ("sunny", "sun", "sun"),
     ("clear-night", "moon", None), ("partlycloudy", "partly", "sun"), ("cloudy", "cloud", None),
     ("lightning-rainy", "storm", "sky"), ("snowy", "snow", "sky"), ("fog", "fog", None),
     ("windy", "wind", None), ("exceptional", None, None)],
)
def test_weather_row_gets_a_condition_glyph(condition: str, icon, tone) -> None:
    states = {"weather.home": {**STATES["weather.home"], "state": condition}}
    item = HACache(fetch=FakeHA(states), clock=Clock()).snapshot(_ha(WX), 15)["items"][0]
    assert (item["icon"], item["icon_tone"]) == (icon, tone)


# ---- battery voltage from the shunt -----------------------------------------

@pytest.mark.parametrize(
    ("state", "unit", "expected"),
    [("53.21", "V", "53.2 V"), ("25.8", "V", "25.8 V"), ("25", "V", "25.0 V"),
     ("53210", "mV", "53.2 V"), ("unavailable", "V", None), ("nan", "V", None),
     ("12", "A", None), ("12", None, None)],
)
def test_volts_value(state, unit, expected) -> None:
    assert volts_value(state, unit) == expected


def test_battery_row_carries_the_shunt_voltage() -> None:
    entity = SOC.model_copy(update={"volts": "sensor.shop_shunt_voltage"})
    states = _battery_states()
    states["sensor.shop_shunt_voltage"] = {"entity_id": "sensor.shop_shunt_voltage", "state": "53.21",
                                           "attributes": {"unit_of_measurement": "V"}}
    fake = FakeHA(states)
    item = HACache(fetch=fake, clock=Clock()).snapshot(_ha(entity), 15)["items"][0]
    assert (item["display"], item["volts"], item["flow"]) == ("75%", "53.2 V", "+1.3 kW")
    assert "sensor.shop_shunt_voltage" in {url.rsplit("/", 1)[1] for _, url in fake.calls}


def test_missing_voltage_sensor_leaves_the_row_as_it_was() -> None:
    entity = SOC.model_copy(update={"volts": "sensor.gone_voltage"})  # HA answers 404
    item = HACache(fetch=FakeHA(_battery_states()), clock=Clock()).snapshot(_ha(entity), 15)["items"][0]
    assert (item["display"], item["volts"], item["flow"]) == ("75%", None, "+1.3 kW")
