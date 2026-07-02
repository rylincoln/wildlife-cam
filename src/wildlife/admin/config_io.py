"""Safe, validated, comment-preserving read/modify/write of ``config.yaml``.

This is the anti-footgun core of the admin UI. Every mutation goes through
:func:`write`, which:

1. Loads the file with ``ruamel.yaml`` in **round-trip** mode so the operator's
   comments and formatting survive edits.
2. Applies a caller-supplied mutation to the in-memory document.
3. Serialises the mutated document to text **once**, then validates *that exact
   text* through the same ``yaml.safe_load`` -> :meth:`Config.model_validate`
   path the worker/gallery/go2rtc daemons use at load time. Validating the bytes
   we are about to persist guarantees the daemons will accept the file -- there
   is no drift between "what we checked" and "what we wrote".
4. Runs extra :func:`semantic_issues` checks (RTSP scheme, model-file presence).
   Any ``"error"``-level issue aborts the write.
5. Backs up the previous file (timestamped, pruned) and writes the new one
   **atomically** (temp file + ``os.replace``) so a crash mid-write can never
   leave a truncated ``config.yaml``.
6. Touches the reload trigger so the root reloader daemon re-validates,
   regenerates ``go2rtc.yaml``, and restarts the affected services.

``ruamel.yaml`` is an optional dependency (the ``admin`` extra); it is imported
lazily with a clear install hint so the rest of the package never requires it.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Callable

import yaml as _pyyaml
from pydantic import ValidationError

from wildlife.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigError",
    "BACKUP_DIRNAME",
    "TRIGGER_NAME",
    "STATUS_NAME",
    "trigger_path",
    "backup_dir",
    "read_raw",
    "read_apply_status",
    "semantic_issues",
    "write",
    "upsert_camera",
    "delete_camera",
    "update_section",
    "update_sections",
    "set_admin_password",
]

#: Directory (relative to the config file) that holds timestamped backups.
BACKUP_DIRNAME = ".config-backups"
#: File (relative to the config file) whose modification wakes the reloader.
TRIGGER_NAME = ".reload.trigger"
#: File (relative to the config file) the reloader writes its last-apply outcome to.
STATUS_NAME = ".reload.status.json"
#: How many timestamped backups to retain.
MAX_BACKUPS = 20
#: Canonical camera field order for newly-added cameras (readability only).
_CAMERA_FIELD_ORDER = (
    "id",
    "host",
    "username",
    "password",
    "rtsp_main",
    "rtsp_sub",
    "onvif_port",
)


class ConfigError(ValueError):
    """A proposed config edit failed schema or semantic validation.

    The message is human-readable and safe to show in the admin UI (it never
    contains secrets -- only field paths and reasons).
    """


# --------------------------------------------------------------------------- #
# ruamel / path helpers
# --------------------------------------------------------------------------- #
def _yaml():
    """Return a configured round-trip ``ruamel.yaml.YAML`` (lazy import)."""
    try:
        from ruamel.yaml import YAML
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "The admin feature needs ruamel.yaml. Install it with: "
            "pip install 'wildlife-detect[admin]'"
        ) from exc
    y = YAML()  # round-trip mode preserves comments + formatting
    y.preserve_quotes = True
    y.width = 4096  # don't wrap long RTSP URLs
    y.indent(mapping=2, sequence=4, offset=2)  # matches config.example.yaml
    return y


def trigger_path(config_path: str | Path) -> Path:
    """Path of the reload-trigger file for a given config file (same directory)."""
    return Path(config_path).resolve().parent / TRIGGER_NAME


def backup_dir(config_path: str | Path) -> Path:
    """Path of the backup directory for a given config file (same directory)."""
    return Path(config_path).resolve().parent / BACKUP_DIRNAME


def read_raw(config_path: str | Path) -> dict:
    """Return the on-disk config as a plain dict (templates un-interpolated).

    Unlike a loaded :class:`Config` (whose RTSP URLs have credentials baked in),
    this preserves the raw ``rtsp://{username}...`` templates -- what the edit
    forms should show. Returns ``{}`` if the file is missing.
    """
    path = Path(config_path)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return _pyyaml.safe_load(fh) or {}


def read_apply_status(config_path: str | Path) -> dict | None:
    """Return the reloader's last-apply status (``.reload.status.json``) or None."""
    import json

    spath = Path(config_path).resolve().parent / STATUS_NAME
    if not spath.is_file():
        return None
    try:
        return json.loads(spath.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _format_validation_error(exc: ValidationError) -> str:
    """Turn a pydantic :class:`ValidationError` into a compact, safe message."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def semantic_issues(config: Config) -> list[tuple[str, str]]:
    """Return ``(level, message)`` checks beyond what pydantic enforces.

    ``level`` is ``"error"`` (blocks the write) or ``"warning"`` (surfaced in the
    UI but allowed). Schema-level rules (types, ranges, port bounds, unique
    camera ids) already live in :mod:`wildlife.config` and are not repeated here.
    """
    issues: list[tuple[str, str]] = []

    for cam in config.cameras:
        for field in ("rtsp_main", "rtsp_sub"):
            url = getattr(cam, field)
            if not str(url).lower().startswith("rtsp://"):
                issues.append(
                    ("error", f"camera {cam.id}: {field} must be an rtsp:// URL")
                )

    model_path = Path(os.path.expanduser(config.detection.model_path))
    if not model_path.is_file():
        issues.append(
            (
                "warning",
                f"detection.model_path {config.detection.model_path!r} does not "
                "exist yet on disk -- detection will fail until the model is present",
            )
        )

    if not config.cameras:
        issues.append(("warning", "no cameras configured -- the detector has nothing to do"))

    return issues


def validate_text(text: str) -> Config:
    """Validate YAML *text* exactly as the daemons will load it.

    Raises :class:`ConfigError` on any schema or blocking semantic problem.
    """
    try:
        data = _pyyaml.safe_load(text)
    except _pyyaml.YAMLError as exc:  # pragma: no cover - ruamel usually emits valid YAML
        raise ConfigError(f"produced invalid YAML: {exc}") from exc
    try:
        config = Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc

    blocking = [msg for level, msg in semantic_issues(config) if level == "error"]
    if blocking:
        raise ConfigError("; ".join(blocking))
    return config


# --------------------------------------------------------------------------- #
# Atomic write + backup + trigger
# --------------------------------------------------------------------------- #
def _backup(config_path: Path) -> Path | None:
    """Copy the current config to a timestamped backup; prune to ``MAX_BACKUPS``."""
    if not config_path.is_file():
        return None
    bdir = backup_dir(config_path)
    bdir.mkdir(parents=True, exist_ok=True)
    # Backups are byte-for-byte copies of config.yaml (plaintext camera creds +
    # the admin hash), so keep them owner-only like the live file.
    try:
        bdir.chmod(0o700)
    except OSError:  # pragma: no cover - best effort
        pass
    stamp = time.strftime("%Y%m%dT%H%M%S")
    dest = bdir / f"config.{stamp}.yaml"
    # Include a short counter suffix if two writes land in the same second.
    n = 1
    while dest.exists():
        dest = bdir / f"config.{stamp}.{n}.yaml"
        n += 1
    dest.write_bytes(config_path.read_bytes())
    try:
        dest.chmod(0o600)
    except OSError:  # pragma: no cover - best effort
        pass
    backups = sorted(bdir.glob("config.*.yaml"))
    for stale in backups[:-MAX_BACKUPS]:
        try:
            stale.unlink()
        except OSError:  # pragma: no cover - best-effort prune
            logger.debug("Could not prune backup %s", stale)
    return dest


def _atomic_write(config_path: Path, text: str) -> None:
    """Write ``text`` to ``config_path`` atomically (temp file + ``os.replace``)."""
    directory = config_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = None, None
    try:
        import tempfile

        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".config.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = None  # now owned by fh
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, config_path)
        tmp = None
    finally:
        if fd is not None:
            os.close(fd)
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)


def touch_trigger(config_path: str | Path) -> Path:
    """Write a timestamp to the reload trigger so launchd WatchPaths fires.

    Modifying (not replacing) the file is what reliably trips ``WatchPaths``; the
    content is informational only.
    """
    tpath = trigger_path(config_path)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    with open(tpath, "w", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} reload requested\n")
        fh.flush()
        os.fsync(fh.fileno())
    return tpath


# --------------------------------------------------------------------------- #
# Core read/modify/write
# --------------------------------------------------------------------------- #
def write(
    config_path: str | Path,
    mutate: Callable[[object], None],
    *,
    trigger_reload: bool = True,
) -> Config:
    """Round-trip load, apply ``mutate``, validate, back up, and atomically write.

    Args:
        config_path: Path to ``config.yaml``.
        mutate: Callback given the round-trip document (a ruamel ``CommentedMap``);
            it should modify it in place. It must not need a return value.
        trigger_reload: When true (default), touch the reload trigger after a
            successful write so the reloader daemon applies the change.

    Returns:
        The validated :class:`Config` that was persisted.

    Raises:
        ConfigError: If the mutated document fails schema or semantic validation.
            The file on disk is left untouched in this case.
    """
    config_path = Path(config_path)
    yaml = _yaml()

    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as fh:
            doc = yaml.load(fh)
    else:  # pragma: no cover - config.yaml always exists in practice
        from ruamel.yaml.comments import CommentedMap

        doc = CommentedMap()

    mutate(doc)

    # Serialise once; validate and persist the *same* bytes.
    buf = io.StringIO()
    yaml.dump(doc, buf)
    text = buf.getvalue()

    config = validate_text(text)  # raises ConfigError on any problem

    _backup(config_path)
    _atomic_write(config_path, text)
    logger.info("Wrote validated config to %s", config_path)
    if trigger_reload:
        # A trigger failure (e.g. a permissions edge case) must not undo or error
        # an already-persisted, valid config write -- the change is saved; only
        # auto-apply is missed, which the dashboard surfaces separately.
        try:
            touch_trigger(config_path)
            logger.info("Reload trigger touched (%s)", trigger_path(config_path))
        except OSError:
            logger.warning(
                "Config saved but could not touch the reload trigger; a manual "
                "restart of the worker/go2rtc may be needed.",
                exc_info=True,
            )
    return config


# --------------------------------------------------------------------------- #
# High-level mutations
# --------------------------------------------------------------------------- #
def _ensure_map(doc, key):
    """Return ``doc[key]`` as a mapping, creating an empty one if absent."""
    from ruamel.yaml.comments import CommentedMap

    if key not in doc or doc[key] is None:
        doc[key] = CommentedMap()
    return doc[key]


def upsert_camera(config_path: str | Path, camera: dict, *, original_id: str | None = None) -> Config:
    """Add a new camera or update an existing one, keyed by id.

    ``camera`` is a plain dict of the camera fields (already-typed values;
    ``onvif_port`` an int). ``original_id`` identifies the row being edited when
    the id itself is changing; defaults to ``camera["id"]``.
    """
    from ruamel.yaml.comments import CommentedMap

    target_id = original_id or camera["id"]

    def _mutate(doc) -> None:
        cams = doc.get("cameras")
        if cams is None:
            from ruamel.yaml.comments import CommentedSeq

            cams = CommentedSeq()
            doc["cameras"] = cams
        # Find existing row by id.
        idx = next((i for i, c in enumerate(cams) if c.get("id") == target_id), None)
        # An "add" (no original_id) that collides with an existing id would
        # silently overwrite that camera -- reject it instead of losing data.
        if idx is not None and original_id is None:
            raise ConfigError(f"a camera with id {target_id!r} already exists")
        if idx is None:
            row = CommentedMap()
            for field in _CAMERA_FIELD_ORDER:
                if field in camera:
                    row[field] = camera[field]
            cams.append(row)
        else:
            row = cams[idx]
            for field in _CAMERA_FIELD_ORDER:
                if field in camera:
                    row[field] = camera[field]

    return write(config_path, _mutate)


def delete_camera(config_path: str | Path, camera_id: str) -> Config:
    """Remove the camera with ``camera_id`` from the config."""

    def _mutate(doc) -> None:
        cams = doc.get("cameras") or []
        idx = next((i for i, c in enumerate(cams) if c.get("id") == camera_id), None)
        if idx is None:
            raise ConfigError(f"no camera with id {camera_id!r} to delete")
        del cams[idx]

    return write(config_path, _mutate)


def update_section(config_path: str | Path, section: str, values: dict) -> Config:
    """Merge ``values`` into a top-level mapping section (e.g. ``detection``)."""
    return update_sections(config_path, {section: values})


def update_sections(config_path: str | Path, sections: dict[str, dict]) -> Config:
    """Merge several top-level sections in one atomic, validated write.

    ``sections`` maps a top-level key (``"detection"``, ``"capture"``, ...) to the
    field/value pairs to set. Doing them together means one backup, one
    validation, and one reload for a multi-section form save.
    """

    def _mutate(doc) -> None:
        for section, values in sections.items():
            target = _ensure_map(doc, section)
            for key, val in values.items():
                target[key] = val

    return write(config_path, _mutate)


def set_admin_password(config_path: str | Path, password_hash: str) -> Config:
    """Persist a Werkzeug password *hash* under ``admin.password_hash``.

    Does not trigger a reload -- changing the admin password doesn't affect the
    worker/go2rtc, and the gallery reads it live per request.
    """

    def _mutate(doc) -> None:
        admin = _ensure_map(doc, "admin")
        admin["enabled"] = True
        admin["password_hash"] = password_hash

    return write(config_path, _mutate, trigger_reload=False)
