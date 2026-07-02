"""Validation tests for the RemoteConfig block and livestream.base_path."""

from __future__ import annotations

from wildlife.config import Config, LivestreamConfig, RemoteConfig


def _minimal_config_dict() -> dict:
    """A minimal mapping that Config.model_validate accepts (no cameras needed)."""
    return {
        "cameras": [],
        "event_source": "reolink_native",
        "capture": {
            "burst_frames": 5, "burst_interval_ms": 200, "stream": "main",
            "rtsp_timeout_s": 10, "max_concurrent": 1,
        },
        "detection": {
            "model_path": "models/yolov8s.pt", "device": "cpu",
            "animal_classes": ["bird"], "confidence_threshold": 0.5,
            "min_box_area_frac": 0.01, "save_best_only": True,
        },
        "dedupe": {"cooldown_s": 30},
        "storage": {"captures_dir": "/tmp/wl-caps", "db_path": "/tmp/wl.db"},
        "retention": {"max_age_days": 30},
        "gallery": {"host": "0.0.0.0", "port": 8080, "page_size": 60},
        "resource_guard": {},
    }


def test_remote_defaults_disabled() -> None:
    cfg = Config.model_validate(_minimal_config_dict())
    assert cfg.remote.enabled is False
    assert cfg.remote.share_secret_hash is None
    assert cfg.remote.block_admin is True
    assert cfg.remote.base_url == ""


def test_remote_block_parses() -> None:
    data = _minimal_config_dict()
    data["remote"] = {
        "enabled": True, "base_url": "https://cam.rlblais.org",
        "share_secret_hash": "pbkdf2:sha256:xxx", "block_admin": True,
    }
    cfg = Config.model_validate(data)
    assert cfg.remote.enabled is True
    assert cfg.remote.base_url == "https://cam.rlblais.org"


def test_base_path_defaults_empty() -> None:
    assert LivestreamConfig().base_path == ""


def test_base_path_normalized() -> None:
    assert LivestreamConfig(base_path="go2rtc").base_path == "/go2rtc"
    assert LivestreamConfig(base_path="/go2rtc/").base_path == "/go2rtc"
    assert LivestreamConfig(base_path="  ").base_path == ""


def test_remote_model_direct() -> None:
    r = RemoteConfig(enabled=True, base_url="https://x")
    assert r.enabled and r.base_url == "https://x" and r.block_admin is True
