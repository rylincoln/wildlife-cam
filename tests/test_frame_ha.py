"""Read-only Home Assistant cache + value formatting for /frame/state."""

from __future__ import annotations

import http.client
import json
import urllib.error

import pytest

from wildlife.config import HAEntityConfig, HomeAssistantConfig
from wildlife.frame.ha import HACache, display_value


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
        "attrs": {}, "display": "68.4°", "detail": "",
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
        "attrs": {}, "display": "—", "detail": "",
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
