"""Configuration loading and validation for the wildlife detection system.

Parses ``config.yaml`` (mirrors ``config.example.yaml`` field-for-field) into a
validated :class:`Config` tree using pydantic v2. Responsibilities beyond plain
type checking:

* Interpolate ``{username}``/``{password}``/``{host}`` templates into each
  camera's RTSP URLs.
* Expand ``~`` and ``$ENV`` references in storage paths to absolute
  :class:`pathlib.Path` values.
* Enforce sensible numeric ranges (confidence in ``0..1``, ports in
  ``1..65535``, ``burst_frames >= 1``, box-area fraction in ``0..1`` ...).

This module depends only on pydantic v2 and PyYAML. It MUST NOT import
``torch``/``cv2``/``numpy`` so it stays importable in hardware-free tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "CameraConfig",
    "CaptureConfig",
    "DetectionConfig",
    "DedupeConfig",
    "StorageConfig",
    "RetentionConfig",
    "GalleryConfig",
    "ResourceGuardConfig",
    "LivestreamConfig",
    "AdminConfig",
    "Config",
    "load_config",
]


class CameraConfig(BaseModel):
    """A single camera's connection details and RTSP stream templates."""

    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1)
    host: str
    username: str
    password: str
    rtsp_main: str
    rtsp_sub: str
    onvif_port: int = Field(default=8000, ge=1, le=65535)

    @model_validator(mode="after")
    def _interpolate_rtsp(self) -> "CameraConfig":
        """Fill ``{username}``/``{password}``/``{host}`` into the RTSP URLs.

        Templates are only formatted when they actually contain a ``{`` so that
        fully-resolved URLs pass through untouched (and are idempotent if the
        model is validated twice). An unknown/invalid placeholder (e.g.
        ``{channel}`` or ``{0}``) is re-raised as a ``ValueError`` so pydantic
        turns it into a readable field error instead of leaking a raw
        ``KeyError``/``IndexError`` to callers.
        """
        for attr in ("rtsp_main", "rtsp_sub"):
            value = getattr(self, attr)
            if "{" not in value:
                continue
            try:
                setattr(
                    self,
                    attr,
                    value.format(
                        username=self.username, password=self.password, host=self.host
                    ),
                )
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    f"{attr} has an unknown placeholder {exc}; only "
                    "{username}, {password} and {host} are supported"
                ) from exc
        return self


class CaptureConfig(BaseModel):
    """RTSP burst-grab behaviour for a single event."""

    burst_frames: int = Field(ge=1)
    burst_interval_ms: int = Field(ge=0)
    stream: Literal["main", "sub"]
    rtsp_timeout_s: int = Field(ge=1)
    max_concurrent: int = Field(ge=1)


class DetectionConfig(BaseModel):
    """YOLO model selection, device, and the animal-class / confidence gate."""

    model_path: str
    device: Literal["mps", "cpu"]
    animal_classes: list[str]
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    min_box_area_frac: float = Field(ge=0.0, le=1.0)
    save_best_only: bool


class DedupeConfig(BaseModel):
    """Per-camera cooldown to suppress re-triggers."""

    cooldown_s: int = Field(ge=0)


class StorageConfig(BaseModel):
    """On-disk capture directory, SQLite path, and image encoding settings."""

    captures_dir: Path
    db_path: Path
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    thumbnail_px: int = Field(default=320, ge=1)

    @field_validator("captures_dir", "db_path", mode="before")
    @classmethod
    def _expand_path(cls, value: str | Path) -> Path:
        """Expand ``~`` and environment variables, returning an absolute-ish Path.

        Runs in ``before`` mode so the raw string (which may contain ``~`` or
        ``$VARS``) is expanded prior to pydantic coercing it to :class:`Path`.
        """
        return Path(os.path.expandvars(os.path.expanduser(str(value))))


class RetentionConfig(BaseModel):
    """Pruning policy applied by ``scripts/prune.py``."""

    max_age_days: int = Field(ge=0)
    min_confidence_keep: float = Field(default=0.0, ge=0.0, le=1.0)


class GalleryConfig(BaseModel):
    """Flask gallery bind address and pagination."""

    host: str
    port: int = Field(ge=1, le=65535)
    page_size: int = Field(ge=1)


class ResourceGuardConfig(BaseModel):
    """Co-tenancy throttles to stay polite alongside the media server."""

    detect_every_nth_event: int = Field(default=1, ge=1)
    max_burst_per_minute: int = Field(default=20, ge=1)


class LivestreamConfig(BaseModel):
    """Optional on-demand live view served by a companion go2rtc binary.

    Additive feature: the detector pipeline is untouched. When ``enabled``, the
    gallery embeds go2rtc's player (``stream.html``) so viewers can watch a
    camera live, picking the lighter sub stream or the 4K main stream. The
    ``*_listen`` fields configure the generated ``go2rtc.yaml`` bind addresses.
    """

    enabled: bool = False
    go2rtc_port: int = Field(default=1984, ge=1, le=65535)  # browser-reachable go2rtc api port
    go2rtc_url: str | None = None  # full base override e.g. "http://192.168.1.50:1984"; if None, gallery derives from request host
    api_listen: str = ":1984"  # go2rtc api bind (for generated go2rtc.yaml)
    webrtc_listen: str = ":8555"
    rtsp_listen: str = ":8554"
    default_stream: Literal["sub", "main"] = "sub"
    allow_main: bool = True  # expose the Main toggle in the UI
    # go2rtc player transport per tile. NOTE: the player always prefers MSE over
    # WebRTC whenever "mse" is present in the list (it ignores the string order),
    # so to *force* WebRTC you must omit "mse". Reolink main is HEVC/4K (MSE plays
    # it in Safari; WebRTC mostly can't), while the sub stream is H264 High profile
    # which Safari's MSE rejects (bad codec string) but WebRTC decodes fine -- hence
    # different defaults per tile.
    main_mode: str = "webrtc,mse"  # HEVC 4K -> MSE
    sub_mode: str = "webrtc"  # H264 High -> WebRTC (omit "mse" so it isn't chosen)


class AdminConfig(BaseModel):
    """Optional password-gated admin UI served by the gallery.

    When ``password_hash`` is set (via the ``wildlife-admin-password`` CLI), the
    gallery exposes ``/admin`` routes for editing detection/camera config behind
    HTTP Basic Auth. Only a Werkzeug password *hash* is stored -- the plaintext
    is never persisted. With ``enabled`` true but no hash set, the admin routes
    fail **closed** (403 with instructions) so a fresh deployment can't be
    reconfigured by anyone on the LAN before a password exists.
    """

    enabled: bool = True
    password_hash: str | None = None


class Config(BaseModel):
    """Top-level validated configuration tree for the whole system."""

    cameras: list[CameraConfig]
    event_source: Literal["reolink_native", "onvif_bridge"]
    capture: CaptureConfig
    detection: DetectionConfig
    dedupe: DedupeConfig
    storage: StorageConfig
    retention: RetentionConfig
    gallery: GalleryConfig
    resource_guard: ResourceGuardConfig
    livestream: LivestreamConfig = Field(default_factory=LivestreamConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)

    @model_validator(mode="after")
    def _unique_camera_ids(self) -> "Config":
        """Reject duplicate camera ids -- they'd collide as dict keys downstream.

        The worker keys its camera registry and go2rtc keys its streams by
        ``camera.id``; two cameras sharing an id would silently shadow one
        another, so fail validation loudly instead.
        """
        seen: set[str] = set()
        dupes: set[str] = set()
        for cam in self.cameras:
            if cam.id in seen:
                dupes.add(cam.id)
            seen.add(cam.id)
        if dupes:
            raise ValueError(
                "Duplicate camera id(s): " + ", ".join(sorted(dupes))
            )
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a :class:`Config`.

    Reads the file at ``path`` with :func:`yaml.safe_load` and validates the
    resulting mapping via :meth:`Config.model_validate`. All interpolation,
    path expansion, and range checks happen during validation.

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        A fully validated :class:`Config` instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the data fails schema/range validation.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config.model_validate(data)
