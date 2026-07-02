"""Flask blueprint for the password-gated admin UI (``/admin``).

Mounted by the gallery only when it knows the config file path (so it can write
edits). Every route is guarded by :func:`wildlife.admin.auth.check_admin_auth`
via a ``before_request`` hook.

The blueprint holds no state of its own: it reads the current config through the
gallery's live-reloading ``get_config`` callable and writes changes through
:mod:`wildlife.admin.config_io`, which validates + backs up + triggers the
reloader. Camera "Test" runs the isolated :mod:`wildlife.admin.probe` subprocess.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from flask import (
    Blueprint,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.security import generate_password_hash

from wildlife.admin import config_io
from wildlife.admin.auth import check_admin_auth
from wildlife.admin.config_io import ConfigError

logger = logging.getLogger(__name__)

__all__ = ["create_admin_blueprint"]

# Default RTSP templates offered when adding a new (Reolink) camera.
_RTSP_MAIN_DEFAULT = "rtsp://{username}:{password}@{host}:554/Preview_01_main"
_RTSP_SUB_DEFAULT = "rtsp://{username}:{password}@{host}:554/Preview_01_sub"
# Wall-clock budgets (seconds) for the probe subprocess.
_TEST_TIMEOUT_S = 8  # per-operation, handed to the probe
_TEST_SUBPROC_TIMEOUT = 60  # hard cap on the whole "Test" (both streams + reolink)
_SAVE_CHECK_SUBPROC_TIMEOUT = 25  # quick single-stream check on save
_MIN_PASSWORD_LEN = 8


def _run_probe(params: dict, timeout: int) -> dict:
    """Invoke the probe subprocess with ``params`` on stdin; return its JSON."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "wildlife.admin.probe"],
            input=json.dumps(params),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"camera test timed out after {timeout}s"}
    except OSError as exc:  # pragma: no cover - launch failure
        return {"error": f"could not launch probe: {exc}"}
    if proc.stdout:
        try:
            return json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError):
            pass
    return {"error": (proc.stderr or "camera test produced no result").strip()[:400]}


def create_admin_blueprint(config_path: str | Path, get_config: Callable[[], object]) -> Blueprint:
    """Build the ``/admin`` blueprint bound to ``config_path`` and ``get_config``."""
    config_path = str(config_path)
    bp = Blueprint("admin", __name__, url_prefix="/admin")

    # ------------------------------------------------------------------ #
    # Auth guard for every admin route.
    # ------------------------------------------------------------------ #
    @bp.before_request
    def _guard():
        # CSRF defence: browsers auto-attach cached Basic-Auth credentials to
        # cross-site form POSTs, so a same-origin check is what actually stops a
        # malicious page from driving these state-changing endpoints. Requests
        # with no Origin/Referer (e.g. curl) are allowed -- auth still applies.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("Origin") or request.headers.get("Referer") or ""
            netloc = urlsplit(origin).netloc
            if netloc and netloc != request.host:
                return Response("Cross-origin admin request blocked.", 403)
        return check_admin_auth(get_config())

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _stored_password(camera_id: str | None) -> str:
        """Return the saved password for an existing camera (for blank-on-edit)."""
        if not camera_id:
            return ""
        cam = next((c for c in get_config().cameras if c.id == camera_id), None)
        return cam.password if cam else ""

    def _camera_params(form, *, original_id: str | None) -> dict:
        """Assemble camera fields from a form, filling a blank password from disk."""
        password = form.get("password", "")
        if not password:
            password = _stored_password(original_id)
        return {
            "id": (form.get("id") or "").strip(),
            "host": (form.get("host") or "").strip(),
            "username": (form.get("username") or "").strip(),
            "password": password,
            "rtsp_main": (form.get("rtsp_main") or "").strip(),
            "rtsp_sub": (form.get("rtsp_sub") or "").strip(),
            "onvif_port": (form.get("onvif_port") or "8000").strip(),
        }

    # ------------------------------------------------------------------ #
    # Dashboard
    # ------------------------------------------------------------------ #
    @bp.route("/", strict_slashes=False)
    def dashboard():
        cfg = get_config()
        warnings = [m for level, m in config_io.semantic_issues(cfg) if level == "warning"]
        status = config_io.read_apply_status(config_path)
        reloader_installed = Path("/Library/LaunchDaemons/com.wildlife.reload.plist").exists()
        # "Pending" only if the trigger was touched *after* the last apply ran
        # (the trigger file persists, so its mere existence isn't enough).
        tpath = config_io.trigger_path(config_path)
        spath = tpath.with_name(config_io.STATUS_NAME)
        pending = tpath.exists() and (
            not spath.is_file() or tpath.stat().st_mtime > spath.stat().st_mtime
        )
        return render_template(
            "admin/dashboard.html",
            cfg=cfg,
            warnings=warnings,
            status=status,
            reloader_installed=reloader_installed,
            pending=pending,
        )

    # ------------------------------------------------------------------ #
    # Detection / capture tuning
    # ------------------------------------------------------------------ #
    @bp.route("/detection", methods=["GET", "POST"])
    def detection():
        cfg = get_config()
        if request.method == "GET":
            return render_template("admin/detection.html", v=_detection_view(cfg), errors=[])

        errors: list[str] = []

        def fnum(name: str, cast):
            raw = (request.form.get(name) or "").strip()
            try:
                return cast(raw)
            except (TypeError, ValueError):
                errors.append(f"{name}: {raw!r} is not a valid number")
                return None

        sections = {
            "detection": {
                "model_path": (request.form.get("model_path") or "").strip(),
                "device": request.form.get("device") or "mps",
                "animal_classes": _parse_classes(request.form.get("animal_classes") or ""),
                "confidence_threshold": fnum("confidence_threshold", float),
                "min_box_area_frac": fnum("min_box_area_frac", float),
                "save_best_only": "save_best_only" in request.form,
            },
            "capture": {
                "stream": request.form.get("stream") or "sub",
                "burst_frames": fnum("burst_frames", int),
                "burst_interval_ms": fnum("burst_interval_ms", int),
                "rtsp_timeout_s": fnum("rtsp_timeout_s", int),
                "max_concurrent": fnum("max_concurrent", int),
            },
            "dedupe": {"cooldown_s": fnum("cooldown_s", int)},
            "resource_guard": {
                "detect_every_nth_event": fnum("detect_every_nth_event", int),
                "max_burst_per_minute": fnum("max_burst_per_minute", int),
            },
        }

        if not errors:
            try:
                config_io.update_sections(config_path, sections)
                flash("Detection settings saved. Applying to the detector…", "ok")
                return redirect(url_for("admin.detection"))
            except ConfigError as exc:
                errors.append(str(exc))

        # Re-render with the submitted (raw) values on any error.
        return render_template(
            "admin/detection.html", v=_detection_view_from_form(request.form), errors=errors
        )

    # ------------------------------------------------------------------ #
    # Cameras
    # ------------------------------------------------------------------ #
    @bp.route("/cameras")
    def cameras():
        raw = config_io.read_raw(config_path)
        rows = raw.get("cameras") or []
        return render_template("admin/cameras.html", cameras=rows)

    @bp.route("/cameras/new", methods=["GET", "POST"])
    def camera_new():
        if request.method == "GET":
            return render_template(
                "admin/camera_form.html",
                v={
                    "id": "",
                    "host": "",
                    "username": "admin",
                    "rtsp_main": _RTSP_MAIN_DEFAULT,
                    "rtsp_sub": _RTSP_SUB_DEFAULT,
                    "onvif_port": "8000",
                },
                original_id="",
                is_new=True,
                errors=[],
            )
        return _save_camera(original_id=None, is_new=True)

    @bp.route("/cameras/<cam_id>/edit", methods=["GET", "POST"])
    def camera_edit(cam_id: str):
        if request.method == "GET":
            raw = config_io.read_raw(config_path)
            row = next((c for c in (raw.get("cameras") or []) if c.get("id") == cam_id), None)
            if row is None:
                flash(f"No camera with id {cam_id!r}.", "err")
                return redirect(url_for("admin.cameras"))
            return render_template(
                "admin/camera_form.html",
                v={
                    "id": row.get("id", ""),
                    "host": row.get("host", ""),
                    "username": row.get("username", ""),
                    "rtsp_main": row.get("rtsp_main", _RTSP_MAIN_DEFAULT),
                    "rtsp_sub": row.get("rtsp_sub", _RTSP_SUB_DEFAULT),
                    "onvif_port": str(row.get("onvif_port", 8000)),
                },
                original_id=cam_id,
                is_new=False,
                errors=[],
            )
        return _save_camera(original_id=cam_id, is_new=False)

    def _save_camera(*, original_id: str | None, is_new: bool):
        """Shared POST handler for add + edit: validate, test, write."""
        params = _camera_params(request.form, original_id=original_id)
        errors: list[str] = []
        for field in ("id", "host", "username", "rtsp_main", "rtsp_sub"):
            if not params[field]:
                errors.append(f"{field} is required")
        try:
            onvif_port = int(params["onvif_port"])
        except (TypeError, ValueError):
            onvif_port = None
            errors.append("onvif_port must be a number")

        skip_test = "skip_test" in request.form
        if not errors and not skip_test:
            stream = getattr(get_config().capture, "stream", "sub")
            result = _run_probe(
                {**params, "onvif_port": onvif_port, "streams": [stream],
                 "reolink": False, "timeout_s": _TEST_TIMEOUT_S},
                timeout=_SAVE_CHECK_SUBPROC_TIMEOUT,
            )
            sres = (result.get("streams") or {}).get(stream) or {}
            if not sres.get("ok"):
                detail = sres.get("error") or result.get("error") or "no frame decoded"
                errors.append(
                    f"Camera test failed on the {stream!r} stream: {detail}. "
                    "Fix it, or tick “Save without testing”."
                )

        if not errors:
            try:
                params["onvif_port"] = onvif_port
                config_io.upsert_camera(config_path, params, original_id=original_id)
                flash(f"Camera {params['id']!r} saved. Restarting detector + go2rtc…", "ok")
                return redirect(url_for("admin.cameras"))
            except ConfigError as exc:
                errors.append(str(exc))

        view = {k: (request.form.get(k) or "") for k in
                ("id", "host", "username", "rtsp_main", "rtsp_sub", "onvif_port")}
        return render_template(
            "admin/camera_form.html",
            v=view, original_id=original_id or "", is_new=is_new, errors=errors,
        )

    @bp.route("/cameras/<cam_id>/delete", methods=["POST"])
    def camera_delete(cam_id: str):
        try:
            config_io.delete_camera(config_path, cam_id)
            flash(f"Camera {cam_id!r} removed. Restarting detector + go2rtc…", "ok")
        except ConfigError as exc:
            flash(str(exc), "err")
        return redirect(url_for("admin.cameras"))

    @bp.route("/test-camera", methods=["POST"])
    def test_camera():
        """AJAX: probe a (possibly unsaved) camera; return the JSON report."""
        data = request.get_json(silent=True) or request.form
        original_id = (data.get("original_id") or "").strip() or None
        params = _camera_params(data, original_id=original_id)
        try:
            params["onvif_port"] = int(params["onvif_port"])
        except (TypeError, ValueError):
            return jsonify({"error": "onvif_port must be a number"})
        for field in ("host", "username", "rtsp_main", "rtsp_sub"):
            if not params[field]:
                return jsonify({"error": f"{field} is required to run a test"})
        params.update({"streams": ["sub", "main"], "reolink": True, "timeout_s": _TEST_TIMEOUT_S})
        return jsonify(_run_probe(params, timeout=_TEST_SUBPROC_TIMEOUT))

    # ------------------------------------------------------------------ #
    # Admin password
    # ------------------------------------------------------------------ #
    @bp.route("/password", methods=["GET", "POST"])
    def password():
        if request.method == "GET":
            return render_template("admin/password.html", errors=[])
        errors: list[str] = []
        pw1 = request.form.get("password") or ""
        pw2 = request.form.get("confirm") or ""
        if len(pw1) < _MIN_PASSWORD_LEN:
            errors.append(f"password must be at least {_MIN_PASSWORD_LEN} characters")
        if pw1 != pw2:
            errors.append("passwords do not match")
        if errors:
            return render_template("admin/password.html", errors=errors)
        config_io.set_admin_password(config_path, generate_password_hash(pw1))
        flash("Admin password updated. You may be prompted to re-authenticate.", "ok")
        return redirect(url_for("admin.dashboard"))

    return bp


# --------------------------------------------------------------------------- #
# View-model helpers (module-level so they're easy to unit test)
# --------------------------------------------------------------------------- #
def _parse_classes(text: str) -> list[str]:
    """Split a textarea of animal classes on newlines/commas; strip + dedupe."""
    seen: set[str] = set()
    out: list[str] = []
    for token in text.replace(",", "\n").split("\n"):
        name = token.strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _detection_view(cfg) -> dict:
    """Build the detection form's string values from the live config."""
    d, c, ded, rg = cfg.detection, cfg.capture, cfg.dedupe, cfg.resource_guard
    return {
        "model_path": d.model_path,
        "device": d.device,
        "animal_classes": "\n".join(d.animal_classes),
        "confidence_threshold": d.confidence_threshold,
        "min_box_area_frac": d.min_box_area_frac,
        "save_best_only": d.save_best_only,
        "stream": c.stream,
        "burst_frames": c.burst_frames,
        "burst_interval_ms": c.burst_interval_ms,
        "rtsp_timeout_s": c.rtsp_timeout_s,
        "max_concurrent": c.max_concurrent,
        "cooldown_s": ded.cooldown_s,
        "detect_every_nth_event": rg.detect_every_nth_event,
        "max_burst_per_minute": rg.max_burst_per_minute,
    }


def _detection_view_from_form(form) -> dict:
    """Re-populate the detection form from a rejected submission's raw values."""
    return {
        "model_path": form.get("model_path", ""),
        "device": form.get("device", "mps"),
        "animal_classes": form.get("animal_classes", ""),
        "confidence_threshold": form.get("confidence_threshold", ""),
        "min_box_area_frac": form.get("min_box_area_frac", ""),
        "save_best_only": "save_best_only" in form,
        "stream": form.get("stream", "sub"),
        "burst_frames": form.get("burst_frames", ""),
        "burst_interval_ms": form.get("burst_interval_ms", ""),
        "rtsp_timeout_s": form.get("rtsp_timeout_s", ""),
        "max_concurrent": form.get("max_concurrent", ""),
        "cooldown_s": form.get("cooldown_s", ""),
        "detect_every_nth_event": form.get("detect_every_nth_event", ""),
        "max_burst_per_minute": form.get("max_burst_per_minute", ""),
    }
