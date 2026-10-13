"""Read-only Home Assistant access for the /frame dashboard.

The long-lived token never leaves this server: :class:`HACache` fetches only the
allow-listed entities from ``config.homeassistant`` and returns a filtered,
display-ready snapshot. One snapshot is shared by every viewer for
``refresh_seconds``; when HA is unreachable the last good values are served,
flagged ``stale``. A 404 for a single entity (typo, deleted entity) yields a "—"
row instead of failing the strip.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, tzinfo
from typing import Any

from wildlife.config import HAEntityConfig, HomeAssistantConfig

logger = logging.getLogger(__name__)

__all__ = ["KEPT_ATTRS", "HACache", "compass", "display_value", "flow_value", "sun_text", "urllib_fetch"]

KEPT_ATTRS = (
    "temperature",
    "temperature_unit",
    "current_temperature",
    "hvac_action",
    "condition",
    "humidity",
)
FORECAST_TTL_S = 1800.0
FORECAST_RETRY_S = 60.0
FORECAST_MAX_AGE_S = 6 * 3600.0  # never show a forecast older than this
_MISSING = ("", "unavailable", "unknown")
_DASH = "—"
CONDITIONS = {
    "clear-night": "Clear",
    "cloudy": "Cloudy",
    "exceptional": "Alert",
    "fog": "Fog",
    "hail": "Hail",
    "lightning": "Thunderstorm",
    "lightning-rainy": "Thunderstorm",
    "partlycloudy": "Partly cloudy",
    "pouring": "Heavy rain",
    "rainy": "Rain",
    "snowy": "Snow",
    "snowy-rainy": "Sleet",
    "sunny": "Sunny",
    "windy": "Windy",
    "windy-variant": "Windy",
}

_COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
COMPANIONS = ("flow", "direction", "gust")  # extra entities a row may name (fetched with it)
# Weather condition -> (glyph the page draws, colour tone). Unlisted conditions get no glyph.
WEATHER_GLYPHS = {
    "sunny": ("sun", "sun"),
    "clear-night": ("moon", None),
    "partlycloudy": ("partly", "sun"),
    "cloudy": ("cloud", None),
    "fog": ("fog", None),
    "rainy": ("rain", "sky"),
    "pouring": ("rain", "sky"),
    "lightning": ("storm", "sky"),
    "lightning-rainy": ("storm", "sky"),
    "snowy": ("snow", "sky"),
    "snowy-rainy": ("snow", "sky"),
    "hail": ("snow", "sky"),
    "windy": ("wind", None),
    "windy-variant": ("wind", None),
}
BATTERY_OK, BATTERY_LOW = 50.0, 25.0  # fill colour: ok at/above 50 %, low at/above 25 %, else critical

Fetch = Callable[[str, str, dict, "bytes | None", float], Any]


def urllib_fetch(method: str, url: str, headers: dict, body: bytes | None, timeout: float) -> Any:
    """Perform one HTTP request against Home Assistant and decode its JSON reply."""
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _num(value: Any) -> str:
    """Format a number with at most one decimal, dropping a trailing ``.0`` ("—" if not a number)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _DASH
    if math.isnan(number) or math.isinf(number):
        return _DASH
    number = round(number, 1)
    return str(int(number)) if number == int(number) else f"{number:.1f}"


def display_value(slot: str, state: str | None, unit: str | None, attrs: dict) -> str:
    """Human-readable value for one strip row (never raises, never 'nan')."""
    if state is None or state in _MISSING:
        return _DASH
    if "current_temperature" in attrs:  # climate entity: show the measured temperature
        reading = _num(attrs["current_temperature"])
        if reading == _DASH:  # HA sends null while the thermostat has no reading
            return _DASH
        action = attrs.get("hvac_action")
        return f"{reading}° · {action}" if action else f"{reading}°"
    if slot == "weather":
        reading = _num(attrs.get("temperature"))
        return _DASH if reading == _DASH else reading + "°"
    try:
        number = float(state)
    except (TypeError, ValueError):
        return str(state)
    if math.isnan(number) or math.isinf(number):
        return _DASH
    if unit in ("in", "in/h"):  # rain gauges: hundredths of an inch matter
        return f'{number:.2f}"' + ("/h" if unit == "in/h" else "")
    if unit == "inHg":  # barometric pressure moves in hundredths
        return f"{number:.2f} inHg"
    if unit in ("mph", "km/h"):  # wind: whole numbers read faster
        return f"{round(number)} {unit}"
    if unit == "W" and abs(round(number)) < 1000:  # whole watts; tenths are noise at a glance
        return f"{round(number) or 0} W"
    if unit == "W":
        number, unit = number / 1000, "kW"
    text = _num(number)
    if unit in ("°F", "°C"):
        return text + "°"
    if unit == "%":
        return text + "%"
    return f"{text} {unit}" if unit else text


def flow_value(state: str | None, unit: str | None) -> str | None:
    """Signed battery flow from a power (W/kW) or current (A) sensor.

    '+98 W', '−1.3 kW', '−6.6 A', '0 A'; None when there is no usable reading.
    """
    if unit not in ("W", "kW", "A"):
        return None
    try:
        value = float(state)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    if unit == "A":
        amps = round(value) if abs(value) >= 99.95 else round(value, 1)  # whole amps from 100 A
        sign = "+" if amps > 0 else "−" if amps < 0 else ""
        return f"{sign}{_num(abs(amps))} A"
    watts = value * (1000 if unit == "kW" else 1)
    whole = round(watts)
    sign = "+" if whole > 0 else "−" if whole < 0 else ""
    if abs(whole) >= 10000:  # whole kW keeps "100%  −11 kW" on one line
        return f"{sign}{round(abs(watts) / 1000)} kW"
    if abs(whole) >= 1000:
        return f"{sign}{_num(abs(watts) / 1000)} kW"
    return f"{sign}{abs(whole)} W"


def compass(degrees: Any) -> str | None:
    """16-point compass name for a bearing in degrees (19 -> 'NNE'); None if not a number."""
    try:
        value = float(degrees)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return _COMPASS[round(value / 22.5) % 16]


def _clock_at(value: Any, tz: tzinfo | None) -> str | None:
    """'7:12 AM' for an ISO timestamp in ``tz`` (None = this server's zone); None if unparseable."""
    try:
        moment = datetime.fromisoformat(str(value)).astimezone(tz)
    except ValueError:
        return None
    return f"{moment.hour % 12 or 12}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"


def sun_text(attrs: dict, tz: tzinfo | None = None) -> str:
    """'↑ 7:12 AM · ↓ 6:31 PM' from sun.sun's next rising/setting, leaving out missing parts."""
    rise = _clock_at(attrs.get("next_rising"), tz) if attrs.get("next_rising") else None
    sets = _clock_at(attrs.get("next_setting"), tz) if attrs.get("next_setting") else None
    parts = [f"↑ {rise}" if rise else "", f"↓ {sets}" if sets else ""]
    return " · ".join(p for p in parts if p) or _DASH


def _wind_text(display: str, direction: dict | None, gust: dict | None) -> str:
    """'1 mph NNE · gust 14': speed, then whichever companions have a reading."""
    if display == _DASH:
        return _DASH
    point = compass((direction or {}).get("state"))
    text = f"{display} {point}" if point else display
    try:
        gust_value = float((gust or {}).get("state"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return text
    if math.isnan(gust_value) or math.isinf(gust_value):
        return text
    return f"{text} · gust {round(gust_value)}"


def _number(state: Any) -> float | None:
    try:
        value = float(state)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def _icon_fields(entity: HAEntityConfig, state: Any, unit: str | None, flow: str | None) -> dict:
    """Glyph, its colour tone, battery fill level and value/flow tones for the page."""
    icon, icon_tone, level, value_tone = entity.icon, None, None, None
    number = _number(state)
    if entity.slot == "weather":
        icon, icon_tone = WEATHER_GLYPHS.get(str(state), (None, None))
    elif entity.icon == "battery" and unit == "%" and number is not None:
        level = min(100.0, max(0.0, number))
        icon_tone = "ok" if level >= BATTERY_OK else "low" if level >= BATTERY_LOW else "critical"
    elif entity.icon == "sun" and number is not None and number > 0:
        value_tone = "sun"  # producing
    return {
        "icon": icon,
        "icon_tone": icon_tone,
        "level": level,
        "value_tone": value_tone,
        "flow_tone": "in" if flow and flow.startswith("+") else None,  # discharging stays plain
    }


def _hilo(high: Any, low: Any) -> str | None:
    """'H 78°  L 51°' from today's forecast, leaving out missing parts (never 'L 0°')."""
    parts = []
    for prefix, value in (("H", high), ("L", low)):
        if _num(value) == _DASH:  # None, non-numeric, nan/inf
            continue
        parts.append(f"{prefix} {round(float(value))}°")
    return "  ".join(parts) or None


def _shape(
    entity: HAEntityConfig,
    payload: dict | None,
    extras: dict[str, dict | None] | None = None,
    tz: tzinfo | None = None,
) -> dict:
    payload = payload or {}
    extras = extras or {}
    attrs = payload.get("attributes") or {}
    state = payload.get("state")
    unit = attrs.get("unit_of_measurement")
    kept = {k: attrs[k] for k in KEPT_ATTRS if k in attrs}
    display = display_value(entity.slot, state, unit, kept)
    if entity.id.startswith("sun.") and state not in (None, *_MISSING):
        display = sun_text(attrs, tz)
    if entity.direction or entity.gust:
        display = _wind_text(display, extras.get("direction"), extras.get("gust"))
    flow = None
    if entity.flow:
        if unit == "%" and display != _DASH:
            display = f"{round(float(state))}%"  # whole percent leaves room for the flow
        flow_payload = extras.get("flow") or {}
        flow = flow_value(
            flow_payload.get("state"), (flow_payload.get("attributes") or {}).get("unit_of_measurement")
        )
    return {
        "slot": entity.slot,
        "label": entity.label or attrs.get("friendly_name") or entity.id,
        "state": state,
        "unit": unit,
        "attrs": kept,
        "display": display,
        "flow": flow,
        "detail": CONDITIONS.get(state, str(state or "")) if entity.slot == "weather" else "",
        **_icon_fields(entity, state, unit, flow),
    }


class HACache:
    """Shared, thread-safe, allow-listed snapshot of Home Assistant state."""

    def __init__(
        self,
        fetch: Fetch | None = None,
        clock: Callable[[], float] = time.monotonic,
        tz: tzinfo | None = None,  # sunrise/sunset zone; None = this server's
    ):
        self._fetch = fetch
        self._clock = clock
        self._tz = tz
        self._lock = threading.Lock()
        self._reset(None)

    def _reset(self, key: object) -> None:
        self._key = key
        self._items: list[dict] = []
        self._ok = False
        self._good_at: float | None = None
        self._attempt_at: float | None = None
        self._forecast: dict | None = None
        self._forecast_due = float("-inf")
        self._forecast_good_at: float | None = None
        # Log on transitions only, so an outage or a typo doesn't flood the gallery log.
        self._down = False
        self._forecast_failing = False
        self._missing_logged: set[str] = set()

    def snapshot(self, ha: HomeAssistantConfig, refresh_seconds: float) -> dict:
        """Return the current snapshot, refreshing from HA when it is older than allowed."""
        key = (
            ha.url,
            ha.token,
            ha.timeout_seconds,
            tuple((e.id, e.slot, e.label, e.flow, e.direction, e.gust, e.icon) for e in ha.entities),
        )
        with self._lock:
            if key != self._key:
                self._reset(key)  # allow-list/token/url changed: drop everything cached
            if not (ha.url and ha.token and ha.entities):
                return {"ok": False, "stale": True, "age_seconds": None, "items": [], "forecast": None}
            now = self._clock()
            if self._attempt_at is None or now - self._attempt_at >= refresh_seconds:
                self._attempt_at = now
                self._refresh_states(ha, now)
            if now >= self._forecast_due:
                self._refresh_forecast(ha, now)
            if self._forecast_good_at is not None and now - self._forecast_good_at >= FORECAST_MAX_AGE_S:
                self._forecast = None  # yesterday's H/L is worse than none
            age = None if self._good_at is None else round(now - self._good_at)
            return {
                "ok": self._ok,
                "stale": not self._ok,
                "age_seconds": age,
                "items": list(self._items),
                "forecast": self._forecast,
            }

    def _call(self, ha: HomeAssistantConfig, method: str, path: str, payload: dict | None = None):
        fetch = self._fetch or urllib_fetch  # module global resolved per call (monkeypatchable)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Authorization": f"Bearer {ha.token}", "Content-Type": "application/json"}
        return fetch(method, ha.url + path, headers, body, ha.timeout_seconds)

    def _state(self, ha: HomeAssistantConfig, entity_id: str) -> dict | None:
        """One entity's state, or None when HA doesn't know it (other HTTP errors propagate)."""
        try:
            payload = self._call(ha, "GET", f"/api/states/{entity_id}")
        except urllib.error.HTTPError as err:
            if err.code != 404:
                raise
            # Unknown entity: render a dash, keep the rest of the strip.
            if entity_id not in self._missing_logged:
                self._missing_logged.add(entity_id)
                logger.warning("Home Assistant has no entity %s (check homeassistant.entities)", entity_id)
            return None
        return payload if isinstance(payload, dict) else None

    def _refresh_states(self, ha: HomeAssistantConfig, now: float) -> None:
        items: list[dict] = []
        try:
            for entity in ha.entities:
                payload = self._state(ha, entity.id)
                extras = {
                    name: self._state(ha, getattr(entity, name))
                    for name in COMPANIONS
                    if getattr(entity, name)
                }
                items.append(_shape(entity, payload, extras, self._tz))
        except Exception as err:  # noqa: BLE001 - any HA fault (incl. http.client) -> stale
            if not self._down:
                logger.warning("Home Assistant unavailable (%s); serving last good values", err)
                self._down = True
            self._ok = False
            return
        if self._down:
            logger.info("Home Assistant reachable again")
            self._down = False
        self._items, self._ok, self._good_at = items, True, now

    def _refresh_forecast(self, ha: HomeAssistantConfig, now: float) -> None:
        weather = next((e for e in ha.entities if e.slot == "weather"), None)
        if weather is None:
            self._forecast, self._forecast_due = None, float("inf")
            return
        try:
            resp = self._call(
                ha,
                "POST",
                "/api/services/weather/get_forecasts?return_response",
                {"entity_id": weather.id, "type": "daily"},
            )
            service = resp.get("service_response") if isinstance(resp, dict) else None
            entry = service.get(weather.id) if isinstance(service, dict) else None
            days = entry.get("forecast") if isinstance(entry, dict) else None
        except Exception as err:  # noqa: BLE001 - forecast is optional; never fail the strip
            if not self._forecast_failing:
                logger.info("Weather forecast unavailable (%s); retrying every %ss", err, FORECAST_RETRY_S)
                self._forecast_failing = True
            self._forecast_due = now + FORECAST_RETRY_S
            return
        self._forecast_failing = False
        today = days[0] if isinstance(days, list) and days and isinstance(days[0], dict) else None
        if today is None:
            self._forecast = None
        else:
            high, low = today.get("temperature"), today.get("templow")
            self._forecast = {"high": high, "low": low, "display": _hilo(high, low)}
            self._forecast_good_at = now
        self._forecast_due = now + FORECAST_TTL_S
