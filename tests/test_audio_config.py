"""Validation tests for AudioConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wildlife.config import AudioConfig, Config


def test_audio_defaults_are_inert():
    a = AudioConfig()
    assert a.enabled is False
    assert a.stream == "sub"
    assert a.use_geo_filter is True
    assert a.confidence_threshold == pytest.approx(0.25)
    assert a.bandpass_fmin == 0
    assert a.min_confirmations == 2
    assert a.confirm_window_s == 15.0
    assert a.cooldown_s == 30.0
    assert a.active_hours == ""


def test_audio_rejects_bad_values():
    with pytest.raises(ValidationError):
        AudioConfig(stream="both")
    with pytest.raises(ValidationError):
        AudioConfig(confidence_threshold=1.5)
    with pytest.raises(ValidationError):
        AudioConfig(bandpass_fmin=-1)
    with pytest.raises(ValidationError):
        AudioConfig(min_confirmations=0)
    with pytest.raises(ValidationError):
        AudioConfig(active_hours="25:00-06:00")


def test_geo_filter_requires_lat_lon():
    # use_geo_filter true (default) but no coords -> error
    with pytest.raises(ValidationError):
        AudioConfig(enabled=True, use_geo_filter=True)
    # valid coords ok
    a = AudioConfig(use_geo_filter=True, latitude=37.2, longitude=-107.5)
    assert a.latitude == pytest.approx(37.2)
    # geo off -> coords optional
    assert AudioConfig(use_geo_filter=False).latitude is None


def test_lat_lon_ranges():
    with pytest.raises(ValidationError):
        AudioConfig(use_geo_filter=True, latitude=91.0, longitude=0.0)
    with pytest.raises(ValidationError):
        AudioConfig(use_geo_filter=True, latitude=0.0, longitude=181.0)


def test_config_without_audio_block_still_validates():
    # Mirror the ContinuousConfig inertness regression: a Config with no `audio`
    # key must validate with audio.enabled defaulting False.
    from tests.test_continuous_config import _minimal_config_dict  # reuse helper

    cfg = Config.model_validate(_minimal_config_dict())
    assert cfg.audio.enabled is False


def test_boundary_values_accepted():
    assert AudioConfig(confidence_threshold=0.0).confidence_threshold == 0.0
    assert AudioConfig(confidence_threshold=1.0).confidence_threshold == 1.0
    ok = AudioConfig(use_geo_filter=True, latitude=90.0, longitude=-180.0)
    assert ok.latitude == 90.0 and ok.longitude == -180.0
    assert AudioConfig(active_hours="22:00-06:00").active_hours == "22:00-06:00"


def test_disabled_config_may_omit_coords_even_with_geo_on():
    # coords are required only when audio is ENABLED with geo on; an inert config may omit them
    a = AudioConfig(enabled=False, use_geo_filter=True)
    assert a.enabled is False and a.latitude is None
