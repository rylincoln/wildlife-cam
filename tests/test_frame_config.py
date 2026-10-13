"""Config models for the /frame kiosk dashboard and its Home Assistant source."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wildlife.config import (
    Config,
    DimConfig,
    FrameConfig,
    HAEntityConfig,
    HomeAssistantConfig,
)


def test_sections_exist_and_default_off() -> None:
    assert {"frame", "homeassistant"} <= set(Config.model_fields)
    frame = FrameConfig()
    assert frame.enabled is False and frame.cameras == [] and frame.refresh_seconds == 15
    assert frame.dim == DimConfig()
    ha = HomeAssistantConfig()
    assert (ha.url, ha.token, ha.timeout_seconds, ha.entities) == ("", "", 3.0, [])


def test_dim_defaults() -> None:
    dim = DimConfig()
    assert (dim.start, dim.end, dim.level, dim.wake_seconds) == ("22:00", "06:30", 0.12, 60)


@pytest.mark.parametrize("bad", ["7pm", "24:00", "6:30", "06:60", ""])
def test_dim_times_must_be_hhmm(bad: str) -> None:
    with pytest.raises(ValidationError):
        DimConfig(start=bad)


def test_dim_time_whitespace_is_trimmed() -> None:
    assert DimConfig(end=" 07:05 ").end == "07:05"


def test_dim_can_be_disabled_with_null() -> None:
    assert FrameConfig.model_validate({"dim": None}).dim is None


@pytest.mark.parametrize("level", [-0.1, 1.5])
def test_dim_level_bounds(level: float) -> None:
    with pytest.raises(ValidationError):
        DimConfig(level=level)


def test_refresh_seconds_bounds() -> None:
    with pytest.raises(ValidationError):
        FrameConfig(refresh_seconds=1)


def test_frame_parses_from_mapping() -> None:
    frame = FrameConfig.model_validate(
        {
            "enabled": True,
            "cameras": ["water_tower"],
            "refresh_seconds": 20,
            "dim": {"start": "21:00", "end": "07:00", "level": 0.1, "wake_seconds": 30},
        }
    )
    assert frame.enabled and frame.cameras == ["water_tower"] and frame.refresh_seconds == 20
    assert frame.dim is not None and frame.dim.start == "21:00" and frame.dim.wake_seconds == 30


def test_ha_url_trailing_slash_stripped() -> None:
    assert HomeAssistantConfig(url="http://192.168.1.33:8123/").url == "http://192.168.1.33:8123"


def test_entity_validation() -> None:
    assert HAEntityConfig(id="sensor.outdoor_temp", slot="temps").label == ""
    with pytest.raises(ValidationError):
        HAEntityConfig(id="sensor.outdoor_temp", slot="doors")
    with pytest.raises(ValidationError):
        HAEntityConfig(id="Sensor.Outdoor", slot="temps")
    with pytest.raises(ValidationError):
        HAEntityConfig(id="sensor.x/../../api", slot="temps")


def test_rain_slot_is_allowed() -> None:
    assert HAEntityConfig(id="sensor.station_daily_rain", slot="rain", label="Rain today").slot == "rain"


def test_energy_row_can_name_a_battery_flow_sensor() -> None:
    entity = HAEntityConfig(
        id="sensor.shop_battery_soc", slot="energy", label="Shop battery", flow="sensor.shop_shunt_power"
    )
    assert entity.flow == "sensor.shop_shunt_power"
    assert HAEntityConfig(id="sensor.shop_battery_soc", slot="energy").flow is None
    with pytest.raises(ValidationError):  # same allow-list rule as id: no path tricks into the HA API
        HAEntityConfig(id="sensor.shop_battery_soc", slot="energy", flow="sensor.x/../../api")


def test_conditions_slot_and_wind_companions() -> None:
    wind = HAEntityConfig(
        id="sensor.station_wind_speed", slot="conditions", label="Wind",
        direction="sensor.station_wind_direction", gust="sensor.station_max_gust",
    )
    assert (wind.slot, wind.direction, wind.gust) == (
        "conditions", "sensor.station_wind_direction", "sensor.station_max_gust"
    )
    assert HAEntityConfig(id="sun.sun", slot="conditions", label="Sun").direction is None
    for field in ("direction", "gust"):  # allow-listed like id
        with pytest.raises(ValidationError):
            HAEntityConfig(id="sensor.station_wind_speed", slot="conditions", **{field: "sensor.x/../api"})


def test_ha_timeout_is_capped_below_the_page_poll_timeout() -> None:  # minor #9
    with pytest.raises(ValidationError):
        HomeAssistantConfig(timeout_seconds=6)
    assert HomeAssistantConfig(timeout_seconds=5).timeout_seconds == 5
