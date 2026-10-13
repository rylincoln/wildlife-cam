"""LAN-only /frame kiosk routes: the page, its state JSON, and a last-seen probe.

* ``GET /frame``       -- the 1280x800 dashboard (two MJPEG tiles + HA strip).
* ``GET /frame/state`` -- server time, night-dim state and the allow-listed HA
  snapshot; also records the polling client so ``/frame/info`` can report it.
* ``GET /frame/wildlife`` -- recent camera sightings + recent distinct bird IDs.
* ``GET /frame/sightings?before=<ts>`` -- the next page of older sightings for the
  scrolling Recent sightings panel (``next`` is the cursor, null at the end).
* ``GET /frame/info``  -- plain-text last poller (IP, time, user agent) to find the
  frame on the LAN (its Wi-Fi MAC/IP change on every boot).

The tunnel gate in :mod:`wildlife.gallery.app` 404s all of these remotely.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime

from flask import Blueprint, Response, abort, jsonify, render_template, request

from wildlife.config import Config
from wildlife.frame.ha import HACache
from wildlife.frame.schedule import in_window
from wildlife.frame.wildlife import CAMERA_KINDS, build_wildlife, sightings_page

__all__ = ["create_frame_blueprint"]


logger = logging.getLogger(__name__)

BIRD_SCAN = 60  # recent audio rows scanned for up to 12 distinct species
# Recent camera rows scanned per provenance for up to 12 events: a burst event can
# save up to burst_frames rows (more with several objects).
SIGHTING_SCAN = 150
SIGHTINGS_PAGE = 12  # one more is fetched to know whether older history exists
_NO_HA = {"ok": False, "stale": True, "age_seconds": None, "items": [], "forecast": None}


def _dim_state(cfg: Config, now: datetime) -> dict:
    """Night-dim state for ``now``; full brightness when ``frame.dim`` is null."""
    dim = cfg.frame.dim
    return {
        "active": dim is not None and in_window(now.time(), dim.start, dim.end),
        "level": dim.level if dim is not None else 1.0,
        "wake_seconds": dim.wake_seconds if dim is not None else 0,
        "start": dim.start if dim is not None else None,  # lets the page keep the schedule
        "end": dim.end if dim is not None else None,  # while the server is unreachable
    }


def _now() -> datetime:
    """Local wall-clock time on the gallery host (module-level so tests can pin it)."""
    return datetime.now()


def create_frame_blueprint(
    get_config: Callable[[], Config],
    go2rtc_base: Callable[[], str],
    recent_captures: Callable[[str, int], list[dict]] | None = None,
    sighting_events: Callable[[str | None, int], list[dict]] | None = None,
) -> Blueprint:
    """Build the frame blueprint.

    Args:
        get_config: Returns the current (live-reloaded) config.
        go2rtc_base: Returns the browser-reachable go2rtc base URL for LAN clients,
            e.g. ``http://192.168.1.193:1984/go2rtc``.
        recent_captures: ``(source_kind, limit) -> serialized captures`` (newest
            first) for ``/frame/wildlife``; without it the panel stays empty.
        sighting_events: ``(before, limit) -> serialized event captures`` (newest
            first, each with ``event_latest``) for ``/frame/sightings`` paging.
    """
    bp = Blueprint("frame", __name__)
    cache = HACache()
    seen: dict = {"ip": None, "ua": "", "at": None}
    seen_lock = threading.Lock()

    @bp.before_request
    def _refuse_cloudflare():
        # Tunnel traffic always carries Cloudflare headers. Refuse it even when
        # remote.enabled is off (e.g. cloudflared left running): HA data stays on the LAN.
        if request.headers.get("Cf-Connecting-IP") or request.headers.get("Cf-Ray"):
            abort(404)

    def _enabled_config() -> Config:
        cfg = get_config()
        if not cfg.frame.enabled:
            abort(404)
        return cfg

    def _camera_ids(cfg: Config) -> list[str]:
        known = [c.id for c in cfg.cameras]
        chosen = [cid for cid in cfg.frame.cameras if cid in known]
        return chosen or known

    @bp.route("/frame")
    def frame_page():
        cfg = _enabled_config()
        base = go2rtc_base()
        cameras = [
            {
                "id": cid,
                "label": cid.replace("_", " "),
                "mjpeg": f"{base}/api/stream.mjpeg?src={cid}_mjpeg",
                "mjpeg_hd": f"{base}/api/stream.mjpeg?src={cid}_mjpeg_hd",
            }
            for cid in _camera_ids(cfg)
        ]
        return render_template(
            "frame.html",
            cameras=cameras,
            mjpeg_enabled=cfg.livestream.enabled and cfg.livestream.mjpeg is not None,
            refresh_seconds=cfg.frame.refresh_seconds,
            dim=_dim_state(cfg, _now()),  # initial state: a reload at night must not flash bright
        )

    @bp.route("/frame/state")
    def frame_state():
        cfg = _enabled_config()
        now = _now()
        with seen_lock:
            seen.update(ip=request.remote_addr, ua=(request.user_agent.string or "")[:200], at=now)
        try:
            snapshot = cache.snapshot(cfg.homeassistant, cfg.frame.refresh_seconds)
        except Exception:  # HA must never take the clock and dimming down
            logger.exception("Home Assistant snapshot failed")
            snapshot = dict(_NO_HA)
        resp = jsonify({"now": now.timestamp(), "dim": _dim_state(cfg, now), "ha": snapshot})
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @bp.route("/frame/wildlife")
    def frame_wildlife():
        _enabled_config()
        data: dict = {"sighting": None, "sightings": [], "birds": []}
        if recent_captures is not None:
            try:
                cameras = [c for kind in CAMERA_KINDS for c in recent_captures(kind, SIGHTING_SCAN)]
                data = build_wildlife(cameras, recent_captures("audio", BIRD_SCAN), _now())
            except Exception:  # the panel is a nicety; never break the page over it
                logger.exception("Wildlife lookup failed")
        resp = jsonify(data)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @bp.route("/frame/sightings")
    def frame_sightings():
        _enabled_config()
        before = request.args.get("before") or None
        if before is not None:
            try:
                datetime.fromisoformat(before)
            except ValueError:
                abort(400)
        data: dict = {"sightings": [], "next": None}
        if sighting_events is not None:
            try:
                rows = sighting_events(before, SIGHTINGS_PAGE + 1)
                items = sightings_page(rows[:SIGHTINGS_PAGE], _now())
                more = len(rows) > SIGHTINGS_PAGE and bool(items)
                data = {"sightings": items, "next": items[-1]["ts"] if more else None}
            except Exception:  # the panel is a nicety; never break the page over it
                logger.exception("Sightings page lookup failed")
        resp = jsonify(data)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @bp.route("/frame/info")
    def frame_info():
        _enabled_config()
        with seen_lock:
            ip, ua, at = seen["ip"], seen["ua"], seen["at"]
        if ip is None:
            return Response("no polls yet\n", mimetype="text/plain")
        text = f"last poll: {ip} at {at.isoformat(timespec='seconds')}\nuser agent: {ua}\n"
        return Response(text, mimetype="text/plain")

    return bp
