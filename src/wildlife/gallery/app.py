"""Flask gallery application for browsing saved wildlife captures.

A small, read-only web UI over the SQLite capture database written by the
worker. It renders a paginated, newest-first thumbnail grid with filters
(camera, class, date range, minimum confidence), serves full images and
thumbnails from the on-disk captures tree, and exposes a JSON endpoint
(``/api/captures``) the front-end uses for incremental pagination.

The app is intentionally light: server-rendered HTML plus a little vanilla
JavaScript (no frontend framework). It only ever reads the database, so it can
run concurrently with the worker thanks to SQLite WAL mode.

Design notes
------------
* A fresh :class:`wildlife.store.Store` is opened per request and stashed on
  Flask's :data:`~flask.g`, then closed in ``teardown``. This gives every
  request thread its own SQLite connection (sidestepping ``check_same_thread``
  concerns) at negligible cost for a low-traffic LAN gallery.
* Image/thumbnail routes resolve the stored *relative* path against
  ``config.storage.captures_dir`` and guard against path traversal before
  serving the file.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, time
from pathlib import Path

from flask import (
    Flask,
    abort,
    g,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)

from wildlife.config import Config, load_config
from wildlife.store import Store
from wildlife.remote import capability as _cap

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y-%m-%d"


# --------------------------------------------------------------------------- #
# Request-parameter parsing helpers
# --------------------------------------------------------------------------- #
def _parse_date(value: str | None, *, end_of_day: bool = False) -> datetime | None:
    """Parse a ``YYYY-MM-DD`` string into a day-bounded :class:`datetime`.

    Returns ``None`` for blank/invalid input so a bad filter simply widens the
    result set instead of erroring. ``end_of_day`` snaps to the last instant of
    the day so date-range filters are inclusive of the end date.
    """
    if not value:
        return None
    try:
        day = datetime.strptime(value, _DATE_FMT).date()
    except ValueError:
        return None
    return datetime.combine(day, time.max if end_of_day else time.min)


def _parse_float(value: str | None) -> float | None:
    """Parse an optional float; ``None``/blank/invalid all become ``None``."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_page(value: str | None) -> int:
    """Parse a 1-based page number, clamping junk/low values to ``1``."""
    try:
        page = int(value) if value is not None else 1
    except (TypeError, ValueError):
        page = 1
    return max(page, 1)


def _parse_filters(args) -> dict:
    """Extract the gallery filter set from a request's query args.

    Keeps both the raw strings (for re-populating the form) and the coerced
    values (``datetime``/``float``/``None``) passed to :meth:`Store.query`.
    """
    camera = (args.get("camera") or "").strip() or None
    label = (args.get("label") or "").strip() or None
    start_raw = (args.get("start") or "").strip()
    end_raw = (args.get("end") or "").strip()
    min_conf_raw = (args.get("min_confidence") or "").strip()
    source_kind = (args.get("source_kind") or "").strip() or None
    return {
        "camera": camera,
        "label": label,
        "start_raw": start_raw,
        "end_raw": end_raw,
        "min_confidence_raw": min_conf_raw,
        "start": _parse_date(start_raw),
        "end": _parse_date(end_raw, end_of_day=True),
        "min_confidence": _parse_float(min_conf_raw),
        "source_kind": source_kind,
    }


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app(config: Config, config_path: str | Path | None = None) -> Flask:
    """Build the Flask gallery app for a validated :class:`Config`.

    Args:
        config: Fully validated configuration tree. ``config.storage`` locates
            the SQLite DB and captures directory; ``config.gallery.page_size``
            controls pagination.
        config_path: Path to ``config.yaml``. When given, the gallery (a) reloads
            request-time settings (cameras, livestream) live when the file changes
            on disk, so admin edits show up without a restart, and (b) mounts the
            password-gated ``/admin`` blueprint that can edit that file. Storage
            paths and page size are still read once at boot.

    Returns:
        A configured :class:`flask.Flask` instance with the gallery routes
        registered. Bind/serve it via :func:`main` (or your own runner).
    """
    app = Flask(__name__)
    # Random per-process key: only used to sign flash-message cookies in /admin.
    # Auth itself is stateless HTTP Basic, so a fresh key each restart is fine.
    app.secret_key = os.urandom(32)

    storage = config.storage
    captures_dir = Path(storage.captures_dir).resolve()
    page_size = config.gallery.page_size

    app.config.update(
        WILDLIFE_CONFIG=config,
        CAPTURES_DIR=captures_dir,
        PAGE_SIZE=page_size,
    )

    # Live config reload: request-time reads go through get_config(), which
    # reloads config.yaml when its mtime changes so admin edits (and the
    # reloader) are reflected in the gallery without restarting it. Storage/DB
    # paths captured above are intentionally *not* hot-swapped.
    _cfg_cache: dict = {"mtime": None, "config": config}

    def get_config() -> Config:
        if config_path is None:
            return _cfg_cache["config"]
        try:
            mtime = os.path.getmtime(config_path)
        except OSError:
            return _cfg_cache["config"]
        if mtime != _cfg_cache["mtime"]:
            # Advance the remembered mtime first so an invalid file on disk is
            # parsed (and logged) once per change, not on every request.
            _cfg_cache["mtime"] = mtime
            try:
                _cfg_cache["config"] = load_config(config_path)
                logger.info("Reloaded config from %s", config_path)
            except Exception:  # noqa: BLE001 - keep serving the last good config
                logger.exception("Failed to reload %s; keeping previous config", config_path)
        return _cfg_cache["config"]

    _rate_limiter = _cap.RateLimiter()

    def _via_tunnel() -> bool:
        """True when the request arrived via the local cloudflared connector AND
        remote access is enabled -- i.e. the shared-secret gate applies. LAN
        clients (non-loopback remote_addr) and remote-disabled configs are False."""
        return get_config().remote.enabled and _cap.is_loopback(request.remote_addr)

    @app.before_request
    def _remote_gate():
        if not _via_tunnel():
            return None  # LAN or remote-disabled -> unchanged behavior
        remote = get_config().remote
        path = request.path
        if remote.block_admin and (path == "/admin" or path.startswith("/admin/")):
            abort(404)  # /admin is never reachable over the tunnel
        if path.startswith("/static/"):
            return None  # styling assets carry no data; keep the shared page rendered
        if not remote.share_secret_hash:
            logger.warning("remote.enabled but no share_secret_hash set; run wildlife-share-secret")
            abort(404)  # fail closed
        ip = request.headers.get("Cf-Connecting-IP") or request.remote_addr
        if _rate_limiter.blocked(ip):
            abort(404)  # treat a rate-limited IP exactly like a bad key (no oracle)
        provided = request.args.get("key")
        if _cap.secret_ok(remote.share_secret_hash, provided):
            _rate_limiter.reset(ip)
            g._set_share_cookie = provided  # emitted in after_request
            return None
        if _cap.secret_ok(remote.share_secret_hash, request.cookies.get(_cap.COOKIE_NAME)):
            return None
        if provided is not None:
            _rate_limiter.record_fail(ip)
        abort(404)

    @app.after_request
    def _remote_headers(resp):
        if getattr(g, "_set_share_cookie", None):
            resp.set_cookie(
                _cap.COOKIE_NAME, g._set_share_cookie,
                max_age=60 * 60 * 24 * 90, secure=True, httponly=True, samesite="Lax",
            )
        if get_config().remote.enabled:
            resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    def _new_store() -> Store:
        return Store(
            db_path=storage.db_path,
            captures_dir=storage.captures_dir,
            jpeg_quality=storage.jpeg_quality,
            thumbnail_px=storage.thumbnail_px,
        )

    # Ensure the schema exists (idempotent, sets WAL) so a brand-new deployment
    # serves an empty gallery rather than erroring on a missing table.
    boot = _new_store()
    try:
        boot.init_schema()
    finally:
        boot.close()

    def get_store() -> Store:
        """Return a per-request :class:`Store`, opening one lazily on first use."""
        store = getattr(g, "_store", None)
        if store is None:
            store = _new_store()
            g._store = store
        return store

    @app.teardown_appcontext
    def _close_store(_exc: BaseException | None = None) -> None:
        store = getattr(g, "_store", None)
        if store is not None:
            store.close()
            g._store = None

    # ----------------------------------------------------------------------- #
    # Livestream (go2rtc) helpers + routes
    # ----------------------------------------------------------------------- #
    def _live_base(remote: bool) -> str:
        """Resolve the browser-reachable go2rtc base for iframe embeds.

        Over the tunnel (``remote``) go2rtc is reverse-routed at a same-origin
        sub-path (``livestream.base_path``, e.g. ``/go2rtc``) and served under
        ``api.base_path``, so we embed a relative URL -- no host, no port, https by
        inheritance. On the LAN we hit go2rtc directly on its api port (honoring an
        explicit ``go2rtc_url`` override), with the same sub-path appended.
        """
        ls = get_config().livestream
        if remote:
            return ls.base_path
        if ls.go2rtc_url:
            return ls.go2rtc_url.rstrip("/") + ls.base_path
        host = request.host.split(":")[0]
        return f"http://{host}:{ls.go2rtc_port}{ls.base_path}"

    def _stream_iframe_src(base: str, stream_name: str, mode: str) -> str:
        """Build the go2rtc ``stream.html`` embed URL for a single stream name."""
        return f"{base}/stream.html?src={stream_name}&mode={mode}"

    def _camera_live(base: str, camera, *, remote: bool) -> dict:
        """Shape a camera into its ``sub``/``main`` iframe URLs.

        Over the tunnel both tiles are forced to ``mode=mse`` because WebRTC cannot
        traverse a Cloudflare Tunnel; on the LAN the configured per-tile transports
        (``sub_mode``/``main_mode``) are kept for best latency.
        """
        ls = get_config().livestream
        sub_mode = "mse" if remote else ls.sub_mode
        main_mode = "mse" if remote else ls.main_mode
        return {
            "id": camera.id,
            "sub_src": _stream_iframe_src(base, f"{camera.id}_sub", sub_mode),
            "main_src": _stream_iframe_src(base, f"{camera.id}_main", main_mode),
        }

    @app.route("/live")
    def live():
        """Render the live grid of every camera's go2rtc player embed."""
        cfg = get_config()
        ls = cfg.livestream
        if not ls.enabled:
            abort(404)
        remote = _via_tunnel()
        base = _live_base(remote)
        cameras = [_camera_live(base, cam, remote=remote) for cam in cfg.cameras]
        return render_template(
            "live.html", cameras=cameras, default_stream=ls.default_stream,
            allow_main=ls.allow_main, single=False, remote=remote,
        )

    @app.route("/live/<camera_id>")
    def live_camera(camera_id: str):
        """Render a single enlarged live player for one camera id."""
        cfg = get_config()
        ls = cfg.livestream
        if not ls.enabled:
            abort(404)
        camera = next((c for c in cfg.cameras if c.id == camera_id), None)
        if camera is None:
            abort(404)
        remote = _via_tunnel()
        base = _live_base(remote)
        cameras = [_camera_live(base, camera, remote=remote)]
        return render_template(
            "live.html", cameras=cameras, default_stream=ls.default_stream,
            allow_main=ls.allow_main, single=True, remote=remote,
        )

    # ----------------------------------------------------------------------- #
    # Serialization + query helpers
    # ----------------------------------------------------------------------- #
    def _serialize(row: dict) -> dict:
        """Shape a DB row dict into the JSON/template payload for one capture."""
        cid = row["id"]
        return {
            "id": cid,
            "camera_id": row.get("camera_id"),
            "label": row.get("label"),
            "confidence": row.get("confidence"),
            "event_ts": row.get("event_ts"),
            "capture_ts": row.get("capture_ts"),
            "width": row.get("width"),
            "height": row.get("height"),
            "box": [
                row.get("box_x1"),
                row.get("box_y1"),
                row.get("box_x2"),
                row.get("box_y2"),
            ],
            "thumb_url": url_for("thumb", capture_id=cid),
            "image_url": url_for("image", capture_id=cid),
            "source_kind": row.get("source_kind"),
            "audio_url": url_for("audio", capture_id=cid) if row.get("audio_path") else None,
        }

    def _query_page(filters: dict, page: int) -> tuple[list[dict], bool]:
        """Run a filtered, paged query; return ``(serialized_rows, has_more)``.

        Fetches ``page_size + 1`` rows to cheaply detect whether a next page
        exists, then trims to ``page_size``.
        """
        offset = (page - 1) * page_size
        rows = get_store().query(
            camera_id=filters["camera"],
            label=filters["label"],
            start=filters["start"],
            end=filters["end"],
            min_confidence=filters["min_confidence"],
            source_kind=filters["source_kind"],
            limit=page_size + 1,
            offset=offset,
        )
        has_more = len(rows) > page_size
        return [_serialize(r) for r in rows[:page_size]], has_more

    # ----------------------------------------------------------------------- #
    # Routes
    # ----------------------------------------------------------------------- #
    @app.route("/")
    def index():
        """Render the server-side thumbnail grid plus filter controls."""
        filters = _parse_filters(request.args)
        page = _parse_page(request.args.get("page"))
        captures, has_more = _query_page(filters, page)
        store = get_store()
        return render_template(
            "index.html",
            captures=captures,
            cameras=store.distinct_cameras(),
            labels=store.distinct_labels(),
            filters=filters,
            page=page,
            page_size=page_size,
            has_more=has_more,
            livestream_enabled=get_config().livestream.enabled,
            admin_enabled=bool(config_path) and get_config().admin.enabled,
        )

    @app.route("/api/captures")
    def api_captures():
        """Return a JSON page of captures for the requested filters/page."""
        filters = _parse_filters(request.args)
        page = _parse_page(request.args.get("page"))
        captures, has_more = _query_page(filters, page)
        return jsonify(
            {
                "page": page,
                "page_size": page_size,
                "count": len(captures),
                "has_more": has_more,
                "captures": captures,
            }
        )

    @app.route("/image/<int:capture_id>")
    def image(capture_id: int):
        """Serve the full-resolution JPEG for a capture id."""
        return _serve_file(capture_id, "image_path")

    @app.route("/thumb/<int:capture_id>")
    def thumb(capture_id: int):
        """Serve the thumbnail JPEG for a capture id."""
        return _serve_file(capture_id, "thumb_path")

    @app.route("/audio/<int:capture_id>")
    def audio(capture_id: int):
        """Serve the AAC/.m4a clip for an audio capture (range-enabled)."""
        row = get_store().get(capture_id)
        if not row or not row.get("audio_path"):
            abort(404)
        full = (captures_dir / row["audio_path"]).resolve()
        try:
            full.relative_to(captures_dir)
        except ValueError:
            abort(403)
        if not full.is_file():
            abort(404)
        return send_file(full, mimetype="audio/mp4", conditional=True)

    def _serve_file(capture_id: int, key: str):
        """Resolve a stored relative path safely and stream the JPEG."""
        row = get_store().get(capture_id)
        if not row:
            abort(404)
        rel = row.get(key)
        if not rel:
            abort(404)
        full = (captures_dir / rel).resolve()
        # Guard against path traversal (e.g. a poisoned relative path).
        try:
            full.relative_to(captures_dir)
        except ValueError:
            logger.warning("Refusing to serve out-of-tree path: %s", full)
            abort(403)
        if not full.is_file():
            abort(404)
        return send_file(full, mimetype="image/jpeg", conditional=True)

    # ----------------------------------------------------------------------- #
    # Admin (config editing) -- only when we know the config path to write to.
    # ----------------------------------------------------------------------- #
    if config_path is not None:
        from wildlife.admin.routes import create_admin_blueprint

        app.register_blueprint(create_admin_blueprint(config_path, get_config, get_store))

    return app


def main() -> None:
    """Console entry point: load config and run the gallery server.

    Usage::

        python -m wildlife.gallery.app [config.yaml]

    Binds to ``config.gallery.host``/``config.gallery.port``. The browsing routes
    apply no auth (LAN-only by design); the ``/admin`` config editor is separately
    password-gated. Exposing beyond the LAN requires a reverse proxy with TLS.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    app = create_app(config, config_path=config_path)
    host, port = config.gallery.host, config.gallery.port
    logger.info("Starting wildlife gallery on http://%s:%s", host, port)
    # threaded=True: each request thread gets its own per-request Store/connection.
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
