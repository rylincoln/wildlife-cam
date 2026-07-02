"""Tests for the admin config writer -- validation, atomicity, and round-trip.

Hardware-free: exercises :mod:`wildlife.admin.config_io` against a throwaway
``config.yaml`` under ``tmp_path``. No cv2/torch/go2rtc/network involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wildlife.admin import config_io as cio
from wildlife.admin.config_io import ConfigError

# A minimal but complete, commented config the writer can round-trip.
_BASE = """\
cameras:
  - id: north_field
    host: 192.168.1.100
    username: admin
    password: "secret-pw"
    rtsp_main: "rtsp://{username}:{password}@{host}:554/Preview_01_main"  # keep me
    rtsp_sub: "rtsp://{username}:{password}@{host}:554/Preview_01_sub"
    onvif_port: 8000

event_source: reolink_native

capture:
  burst_frames: 5
  burst_interval_ms: 200
  stream: sub
  rtsp_timeout_s: 10
  max_concurrent: 1

detection:
  model_path: "models/yolov8s.pt"  # tune me
  device: cpu
  animal_classes:
    - bird
    - cat
  confidence_threshold: 0.55
  min_box_area_frac: 0.01
  save_best_only: true

dedupe:
  cooldown_s: 30

storage:
  captures_dir: "__CAPTURES__"
  db_path: "__DB__"

retention:
  max_age_days: 30

gallery:
  host: "0.0.0.0"
  port: 8080
  page_size: 60

resource_guard:
  detect_every_nth_event: 1
  max_burst_per_minute: 20
"""


def render_base(tmp_path: Path) -> str:
    """Fill the storage-path sentinels in ``_BASE`` (leaving RTSP ``{...}`` alone)."""
    return _BASE.replace("__CAPTURES__", str(tmp_path / "captures")).replace(
        "__DB__", str(tmp_path / "c.db")
    )


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(render_base(tmp_path), encoding="utf-8")
    return p


def test_update_preserves_comments_and_backs_up(config_path: Path) -> None:
    cfg = cio.update_section(config_path, "detection", {"confidence_threshold": 0.7})
    text = config_path.read_text()

    assert cfg.detection.confidence_threshold == 0.7
    # Comments and the un-interpolated RTSP template survive the write.
    assert "# tune me" in text
    assert "# keep me" in text
    assert "{username}" in text
    # A timestamped backup and a reload trigger were created.
    assert list(cio.backup_dir(config_path).glob("config.*.yaml"))
    assert cio.trigger_path(config_path).is_file()


def test_bad_range_and_bad_rtsp_rejected(config_path: Path) -> None:
    before = config_path.read_text()
    with pytest.raises(ConfigError):
        cio.update_section(config_path, "gallery", {"port": 99999})  # pydantic range
    with pytest.raises(ConfigError):
        cio.upsert_camera(
            config_path,
            {"id": "bad", "host": "h", "username": "u", "password": "p",
             "rtsp_main": "http://nope", "rtsp_sub": "rtsp://ok", "onvif_port": 8000},
        )  # semantic scheme check
    # The file on disk is unchanged after a rejected write.
    assert config_path.read_text() == before


def test_duplicate_camera_id_rejected(config_path: Path) -> None:
    def _add_dupe(doc) -> None:
        doc["cameras"].append({
            "id": "north_field", "host": "9.9.9.9", "username": "u", "password": "p",
            "rtsp_main": "rtsp://a", "rtsp_sub": "rtsp://b", "onvif_port": 8000,
        })

    with pytest.raises(ConfigError):
        cio.write(config_path, _add_dupe)


def test_upsert_new_then_edit_then_delete(config_path: Path) -> None:
    cfg = cio.upsert_camera(
        config_path,
        {"id": "south_trail", "host": "192.168.1.72", "username": "admin", "password": "pw",
         "rtsp_main": "rtsp://x/main", "rtsp_sub": "rtsp://x/sub", "onvif_port": 8000},
    )
    assert [c.id for c in cfg.cameras] == ["north_field", "south_trail"]

    # Editing keeps it a single row (upsert by id), and can rename.
    cfg = cio.upsert_camera(
        config_path,
        {"id": "south_gate", "host": "192.168.1.72", "username": "admin", "password": "pw",
         "rtsp_main": "rtsp://x/main", "rtsp_sub": "rtsp://x/sub", "onvif_port": 8000},
        original_id="south_trail",
    )
    assert [c.id for c in cfg.cameras] == ["north_field", "south_gate"]

    cfg = cio.delete_camera(config_path, "south_gate")
    assert [c.id for c in cfg.cameras] == ["north_field"]

    with pytest.raises(ConfigError):
        cio.delete_camera(config_path, "does_not_exist")


def test_update_sections_atomic(config_path: Path) -> None:
    cfg = cio.update_sections(
        config_path,
        {"detection": {"device": "mps"}, "capture": {"stream": "main", "burst_frames": 3}},
    )
    assert cfg.detection.device == "mps"
    assert cfg.capture.stream == "main"
    assert cfg.capture.burst_frames == 3
    # One combined write => exactly one backup for the two-section change.
    assert len(list(cio.backup_dir(config_path).glob("config.*.yaml"))) == 1


def test_set_admin_password_no_reload(config_path: Path) -> None:
    cfg = cio.set_admin_password(config_path, "pbkdf2:sha256:fake$hash")
    assert cfg.admin.password_hash == "pbkdf2:sha256:fake$hash"
    assert cfg.admin.enabled is True
    # Password changes don't affect the detector/go2rtc, so no reload is queued.
    assert not cio.trigger_path(config_path).is_file()


def test_read_raw_keeps_templates(config_path: Path) -> None:
    raw = cio.read_raw(config_path)
    assert raw["cameras"][0]["rtsp_main"].startswith("rtsp://{username}")


def test_add_duplicate_id_rejected(config_path: Path) -> None:
    # Adding (original_id=None) a camera whose id already exists must error
    # rather than silently overwrite the existing one.
    with pytest.raises(ConfigError):
        cio.upsert_camera(
            config_path,
            {"id": "north_field", "host": "10.0.0.9", "username": "u", "password": "attacker",
             "rtsp_main": "rtsp://a/main", "rtsp_sub": "rtsp://a/sub", "onvif_port": 8000},
        )
    # The original camera is untouched.
    assert cio.read_raw(config_path)["cameras"][0]["host"] == "192.168.1.100"


def test_unknown_rtsp_placeholder_rejected(config_path: Path) -> None:
    with pytest.raises(ConfigError):
        cio.upsert_camera(
            config_path,
            {"id": "cam2", "host": "h", "username": "u", "password": "p",
             "rtsp_main": "rtsp://{host}:554/Preview_{channel}_main",
             "rtsp_sub": "rtsp://{host}:554/sub", "onvif_port": 8000},
        )


def test_backups_are_owner_only(config_path: Path) -> None:
    cio.update_section(config_path, "detection", {"confidence_threshold": 0.7})
    backups = list(cio.backup_dir(config_path).glob("config.*.yaml"))
    assert backups
    for b in backups:
        assert (b.stat().st_mode & 0o777) == 0o600
