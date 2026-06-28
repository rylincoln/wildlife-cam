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
    return {
        "camera": camera,
        "label": label,
        "start_raw": start_raw,
        "end_raw": end_raw,
        "min_confidence_raw": min_conf_raw,
        "start": _parse_date(start_raw),
        "end": _parse_date(end_raw, end_of_day=True),
        "min_confidence": _parse_float(min_conf_raw),
    }


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app(config: Config) -> Flask:
    """Build the Flask gallery app for a validated :class:`Config`.

    Args:
        config: Fully validated configuration tree. ``config.storage`` locates
            the SQLite DB and captures directory; ``config.gallery.page_size``
            controls pagination.

    Returns:
        A configured :class:`flask.Flask` instance with the gallery routes
        registered. Bind/serve it via :func:`main` (or your own runner).
    """
    app = Flask(__name__)

    storage = config.storage
    captures_dir = Path(storage.captures_dir).resolve()
    page_size = config.gallery.page_size

    app.config.update(
        WILDLIFE_CONFIG=config,
        CAPTURES_DIR=captures_dir,
        PAGE_SIZE=page_size,
    )

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

    return app


def main() -> None:
    """Console entry point: load config and run the gallery server.

    Usage::

        python -m wildlife.gallery.app [config.yaml]

    Binds to ``config.gallery.host``/``config.gallery.port``. No auth is applied
    (LAN-only by design); exposing beyond the LAN requires a reverse proxy with
    auth + TLS.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    app = create_app(config)
    host, port = config.gallery.host, config.gallery.port
    logger.info("Starting wildlife gallery on http://%s:%s", host, port)
    # threaded=True: each request thread gets its own per-request Store/connection.
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
