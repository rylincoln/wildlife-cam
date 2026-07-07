"""Persistence layer for wildlife captures.

This module owns the on-disk capture artifacts (full JPEG + thumbnail) and the
SQLite metadata index described in spec section 6.5. It deliberately depends on
*only* numpy, Pillow and the stdlib ``sqlite3`` module so that it stays free of
heavy/hardware deps (no torch / cv2 / ultralytics) and can be unit-tested on any
machine.

Frames handed to :meth:`Store.save_capture` are expected to be OpenCV-style BGR
``np.ndarray`` images; they are converted to RGB before being written with
Pillow. Image paths persisted in the database are stored *relative* to
``captures_dir`` so the capture tree can be relocated without rewriting rows.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from wildlife.models import Detection

logger = logging.getLogger(__name__)

# Schema is verbatim from spec section 6.5.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS captures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT    NOT NULL,
    event_ts    TEXT    NOT NULL,   -- ISO8601, when the camera event fired
    capture_ts  TEXT    NOT NULL,   -- ISO8601, when the frame was saved
    label       TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    box_x1 REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL,
    image_path  TEXT    NOT NULL,   -- relative to captures_dir
    thumb_path  TEXT    NOT NULL,
    width INTEGER, height INTEGER,
    original_label TEXT,               -- model's label before a human reclassify
    reviewed       INTEGER NOT NULL DEFAULT 0,
    reviewed_at    TEXT,               -- ISO8601 of the last human action
    source_kind    TEXT    NOT NULL DEFAULT 'reolink',  -- provenance: reolink | continuous | audio
    audio_path     TEXT                                 -- relative clip path for source_kind='audio'
);
CREATE INDEX IF NOT EXISTS idx_capture_ts ON captures(capture_ts);
CREATE INDEX IF NOT EXISTS idx_camera     ON captures(camera_id);
CREATE INDEX IF NOT EXISTS idx_label      ON captures(label);
"""

# Columns returned by SELECT * and surfaced in row dicts, in declaration order.
_COLUMNS: tuple[str, ...] = (
    "id",
    "camera_id",
    "event_ts",
    "capture_ts",
    "label",
    "confidence",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "image_path",
    "thumb_path",
    "width",
    "height",
    "original_label",
    "reviewed",
    "reviewed_at",
    "source_kind",
    "audio_path",
)

# Columns added after the original schema; applied idempotently by _migrate().
_COLUMN_ADDITIONS: tuple[tuple[str, str], ...] = (
    ("original_label", "TEXT"),
    ("reviewed", "INTEGER NOT NULL DEFAULT 0"),
    ("reviewed_at", "TEXT"),
    ("source_kind", "TEXT NOT NULL DEFAULT 'reolink'"),
    ("audio_path", "TEXT"),
)


def _iso(ts: datetime | str) -> str:
    """Return an ISO8601 string for a datetime (pass-through for strings)."""
    if isinstance(ts, datetime):
        return ts.isoformat()
    return str(ts)


def _now_iso() -> str:
    """Timestamp for human edits, matching the ISO8601 TEXT convention."""
    return datetime.now().isoformat(timespec="seconds")


def _sanitize(part: str) -> str:
    """Make a string safe to embed in a filename path segment."""
    return "".join(c if c.isalnum() or c in ("-", ".") else "_" for c in part)


def _safe_unlink(captures_dir: Path, rel: str | None) -> tuple[bool, int]:
    """Delete ``captures_dir/rel`` if present and safely inside ``captures_dir``.

    Returns ``(deleted, bytes_freed)``. Missing files count as not-deleted with
    zero bytes (keeps the operation idempotent). Paths that resolve outside the
    captures tree are refused as a safety check.
    """
    if not rel:
        return (False, 0)

    target = captures_dir / rel
    root = captures_dir.resolve()
    try:
        resolved = target.resolve()
    except OSError:
        return (False, 0)

    if root != resolved and root not in resolved.parents:
        logger.warning("Refusing to delete path outside captures_dir: %s", target)
        return (False, 0)

    if not resolved.exists():
        return (False, 0)

    try:
        size = resolved.stat().st_size
        resolved.unlink()
        return (True, size)
    except OSError as exc:
        logger.warning("Could not delete %s: %s", resolved, exc)
        return (False, 0)


def _prune_empty_dirs(root: Path) -> int:
    """Remove now-empty subdirectories under ``root`` (bottom-up). Returns count."""
    if not root.is_dir():
        return 0
    removed = 0
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        directory = Path(dirpath)
        if directory == root:
            continue
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
                removed += 1
        except OSError:
            # Non-empty or vanished concurrently; leave it be.
            pass
    return removed


def _chunked(seq: list, size: int) -> Iterable[list]:
    """Yield ``seq`` in lists of at most ``size`` (for SQLite IN-clause limits)."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _sweep_empty_capture_dirs(captures_dir: Path, rels: Iterable[str]) -> int:
    """Remove now-empty ``YYYY/MM/DD`` dirs for the given relative file paths.

    Scoped to the deleted captures' own dated directories (not the whole tree),
    and skips the current day's directory because the worker may be writing
    there. Walks up MM/YYYY so an emptied day frees its parents too. Returns the
    number of directories removed. Never raises.
    """
    now = datetime.now()
    today = (captures_dir / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}").resolve()
    root = captures_dir.resolve()
    day_dirs = {(captures_dir / rel).parent for rel in rels if rel}
    removed = 0
    for day in day_dirs:
        for directory in (day, day.parent, day.parent.parent):
            try:
                resolved = directory.resolve()
            except OSError:
                continue
            if resolved == today or resolved == root or root not in resolved.parents:
                continue
            try:
                if not any(directory.iterdir()):
                    directory.rmdir()
                    removed += 1
            except OSError:
                pass
    return removed


def _build_filters(
    *,
    camera_id: str | None,
    label: str | None,
    start: datetime | str | None,
    end: datetime | str | None,
    min_confidence: float | None,
    reviewed: bool | None,
    source_kind: str | None = None,
) -> tuple[list[str], list[Any]]:
    """Build the shared AND-combined WHERE clauses + params for query/count."""
    clauses: list[str] = []
    params: list[Any] = []
    if camera_id is not None:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if label is not None:
        clauses.append("label = ?")
        params.append(label)
    if start is not None:
        clauses.append("capture_ts >= ?")
        params.append(_iso(start))
    if end is not None:
        clauses.append("capture_ts <= ?")
        params.append(_iso(end))
    if min_confidence is not None:
        clauses.append("confidence >= ?")
        params.append(float(min_confidence))
    if reviewed is not None:
        clauses.append("reviewed = ?")
        params.append(1 if reviewed else 0)
    if source_kind is not None:
        clauses.append("source_kind = ?")
        params.append(source_kind)
    return clauses, params


class Store:
    """SQLite-backed metadata index plus JPEG/thumbnail writer for captures.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file (parent dirs are created).
    captures_dir:
        Root directory under which the ``YYYY/MM/DD`` capture tree is written.
    jpeg_quality:
        Pillow JPEG quality for the full-size image (1-95 sensible range).
    thumbnail_px:
        Target *width* in pixels of the generated thumbnail. Thumbnails are only
        ever downscaled, never enlarged.
    """

    def __init__(
        self,
        db_path: str | Path,
        captures_dir: str | Path,
        jpeg_quality: int = 85,
        thumbnail_px: int = 320,
    ) -> None:
        self.db_path = Path(db_path)
        self.captures_dir = Path(captures_dir)
        self.jpeg_quality = int(jpeg_quality)
        self.thumbnail_px = int(thumbnail_px)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.captures_dir.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False: the worker writes from one thread while the
        # (separate, read-only) gallery Store reads from Flask request threads.
        # WAL mode (set in init_schema) keeps those readers from blocking.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Per-connection PRAGMAs. busy_timeout MUST be set on every connection
        # (not just the boot/worker one) so a gallery/admin write waits for the
        # worker's WAL writer instead of failing immediately with "database is
        # locked". journal_mode=WAL is persistent (file header) so it stays in
        # init_schema; busy_timeout/synchronous are per-connection and do not.
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        # Serialize writes from THIS instance; cross-connection writes are
        # arbitrated by SQLite WAL + busy_timeout, not by this lock.
        self._write_lock = threading.Lock()

    # ----------------------------------------------------------------- schema
    def init_schema(self) -> None:
        """Create the captures table + indexes, enable WAL, and migrate (idempotent)."""
        with self._write_lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.executescript(_SCHEMA_SQL)
            self._migrate()
            self._conn.commit()
        logger.debug("Initialised capture schema at %s", self.db_path)

    def _migrate(self) -> None:
        """Add any missing post-original columns. Safe across re-runs and races."""
        existing = {r["name"] for r in self._conn.execute("PRAGMA table_info(captures)")}
        for name, decl in _COLUMN_ADDITIONS:
            if name in existing:
                continue
            try:
                self._conn.execute(f"ALTER TABLE captures ADD COLUMN {name} {decl}")
            except sqlite3.OperationalError as exc:
                # Another process (worker vs gallery boot) added it between the
                # check and the ALTER — a schema error busy_timeout can't absorb.
                if "duplicate column name" not in str(exc).lower():
                    raise

    # ------------------------------------------------------------- write path
    def _write_image_and_thumb(
        self, image: Image.Image, abs_dir: Path, stem: str, width: int
    ) -> tuple[str, str]:
        """Write ``image`` as a full JPEG + a downscaled thumbnail; return (rel, thumb_rel).

        Shared by :meth:`save_capture` and :meth:`save_audio_capture` (both pass
        an already-RGB PIL image). Guards against filename collisions within one
        event.
        """
        image_abs = abs_dir / f"{stem}.jpg"
        if image_abs.exists():
            n = 1
            while (abs_dir / f"{stem}-{n}.jpg").exists():
                n += 1
            stem = f"{stem}-{n}"
            image_abs = abs_dir / f"{stem}.jpg"
        thumb_abs = abs_dir / f"{stem}_thumb.jpg"

        image.save(image_abs, format="JPEG", quality=self.jpeg_quality)
        thumb = image
        if width > self.thumbnail_px:
            new_h = max(1, round(image.height * self.thumbnail_px / width))
            thumb = image.resize((self.thumbnail_px, new_h), Image.LANCZOS)
        thumb.save(thumb_abs, format="JPEG", quality=self.jpeg_quality)

        rel_dir = abs_dir.relative_to(self.captures_dir)
        return (
            (rel_dir / f"{stem}.jpg").as_posix(),
            (rel_dir / f"{stem}_thumb.jpg").as_posix(),
        )

    def save_capture(
        self,
        *,
        camera_id: str,
        event_ts: datetime,
        capture_ts: datetime,
        frame: np.ndarray,
        det: Detection,
        source_kind: str = "reolink",
    ) -> int:
        """Persist one capture: write JPEG + thumbnail and insert a metadata row.

        ``frame`` is a BGR (OpenCV) ``np.ndarray``; it is converted to RGB before
        being written. Files land at
        ``<captures_dir>/YYYY/MM/DD/<camera_id>_<capture_ts>_<label>_<conf>.jpg``
        with a sibling ``*_thumb.jpg``. The image/thumb paths recorded in the row
        are relative to ``captures_dir``.

        Returns the new row's integer primary key.
        """
        height, width = int(frame.shape[0]), int(frame.shape[1])

        # BGR -> RGB. Copy to a contiguous array so Pillow is happy with the view.
        if frame.ndim == 3 and frame.shape[2] == 3:
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            image = Image.fromarray(rgb, mode="RGB")
        elif frame.ndim == 2:
            image = Image.fromarray(np.ascontiguousarray(frame), mode="L")
        else:  # pragma: no cover - defensive; unexpected channel count
            raise ValueError(f"Unsupported frame shape: {frame.shape!r}")

        # Build the dated directory + filenames.
        rel_dir = Path(
            f"{capture_ts.year:04d}",
            f"{capture_ts.month:02d}",
            f"{capture_ts.day:02d}",
        )
        abs_dir = self.captures_dir / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)

        # Box coordinates double as a uniqueness token in the filename. In
        # all-positives mode (save_best_only=false) two detections from a single
        # event can share camera/capture_ts/label/2dp-confidence, so a box-derived
        # token keeps their files (and the DB rows pointing at them) distinct
        # instead of silently overwriting.
        x1, y1, x2, y2 = det.box_xyxy
        box_token = f"{int(x1)}-{int(y1)}-{int(x2)}-{int(y2)}"

        ts_compact = capture_ts.strftime("%Y%m%dT%H%M%S_%f")
        stem = "_".join(
            (
                _sanitize(camera_id),
                ts_compact,
                _sanitize(det.label),
                f"{det.confidence:.2f}",
                box_token,
            )
        )

        image_rel, thumb_rel = self._write_image_and_thumb(image, abs_dir, stem, width)

        with self._write_lock:
            cur = self._conn.execute(
                """
                INSERT INTO captures (
                    camera_id, event_ts, capture_ts, label, confidence,
                    box_x1, box_y1, box_x2, box_y2,
                    image_path, thumb_path, width, height, source_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    _iso(event_ts),
                    _iso(capture_ts),
                    det.label,
                    float(det.confidence),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    image_rel,
                    thumb_rel,
                    width,
                    height,
                    source_kind,
                ),
            )
            self._conn.commit()
            capture_id = int(cur.lastrowid)

        logger.info(
            "Saved capture id=%d camera=%s label=%s conf=%.3f -> %s",
            capture_id,
            camera_id,
            det.label,
            det.confidence,
            image_rel,
        )
        return capture_id

    def save_audio_capture(
        self,
        *,
        camera_id: str,
        event_ts: datetime,
        capture_ts: datetime,
        species: str,
        confidence: float,
        spectrogram_rgb: np.ndarray,
        clip_bytes: bytes | None,
        source_kind: str = "audio",
    ) -> int:
        """Persist one audio detection: spectrogram JPEG + thumbnail + optional clip.

        ``spectrogram_rgb`` is an RGB ``uint8`` image (H, W, 3). The clip (AAC/.m4a
        bytes) is written to ``audio_path`` when given, else that column is NULL.
        Box columns are NULL (audio has no bounding box). Returns the new row id.
        """
        image = Image.fromarray(np.ascontiguousarray(spectrogram_rgb), mode="RGB")
        height, width = int(spectrogram_rgb.shape[0]), int(spectrogram_rgb.shape[1])

        rel_dir = Path(
            f"{capture_ts.year:04d}", f"{capture_ts.month:02d}", f"{capture_ts.day:02d}"
        )
        abs_dir = self.captures_dir / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)

        ts_compact = capture_ts.strftime("%Y%m%dT%H%M%S_%f")
        stem = "_".join(
            (_sanitize(camera_id), ts_compact, _sanitize(species), f"{confidence:.2f}", "audio")
        )
        image_rel, thumb_rel = self._write_image_and_thumb(image, abs_dir, stem, width)

        audio_rel: str | None = None
        if clip_bytes is not None:
            # Reuse the (collision-resolved) image stem so the clip sits beside it.
            clip_stem = Path(image_rel).stem
            clip_abs = abs_dir / f"{clip_stem}.m4a"
            clip_abs.write_bytes(clip_bytes)
            audio_rel = (rel_dir / f"{clip_stem}.m4a").as_posix()

        with self._write_lock:
            cur = self._conn.execute(
                """
                INSERT INTO captures (
                    camera_id, event_ts, capture_ts, label, confidence,
                    image_path, thumb_path, width, height, source_kind, audio_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id, _iso(event_ts), _iso(capture_ts), species, float(confidence),
                    image_rel, thumb_rel, width, height, source_kind, audio_rel,
                ),
            )
            self._conn.commit()
            capture_id = int(cur.lastrowid)

        logger.info(
            "Saved audio capture id=%d camera=%s species=%s conf=%.3f clip=%s",
            capture_id, camera_id, species, confidence, audio_rel or "-",
        )
        return capture_id

    # -------------------------------------------------------------- read path
    def get(self, capture_id: int) -> dict[str, Any] | None:
        """Return the row for ``capture_id`` as a dict, or None if absent."""
        row = self._conn.execute(
            "SELECT * FROM captures WHERE id = ?", (capture_id,)
        ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def query(
        self,
        *,
        camera_id: str | None = None,
        label: str | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        min_confidence: float | None = None,
        reviewed: bool | None = None,
        source_kind: str | None = None,
        limit: int = 60,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return matching rows as dicts, newest first.

        All filters are optional and AND-combined. ``start``/``end`` bound
        ``capture_ts`` inclusively; ``reviewed`` filters on the human-review flag;
        ``source_kind`` matches the provenance column exactly (e.g. ``"audio"``).
        Results are ordered by ``capture_ts`` desc (ties by ``id`` desc) and
        paginated via ``limit``/``offset``.
        """
        clauses, params = _build_filters(
            camera_id=camera_id, label=label, start=start, end=end,
            min_confidence=min_confidence, reviewed=reviewed, source_kind=source_kind,
        )
        sql = "SELECT * FROM captures"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY capture_ts DESC, id DESC LIMIT ? OFFSET ?"
        params.extend((int(limit), int(offset)))
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(
        self,
        *,
        camera_id: str | None = None,
        label: str | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        min_confidence: float | None = None,
        reviewed: bool | None = None,
        source_kind: str | None = None,
    ) -> int:
        """Return the number of rows matching the same filters as :meth:`query`."""
        clauses, params = _build_filters(
            camera_id=camera_id, label=label, start=start, end=end,
            min_confidence=min_confidence, reviewed=reviewed, source_kind=source_kind,
        )
        sql = "SELECT COUNT(*) FROM captures"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return int(self._conn.execute(sql, params).fetchone()[0])

    def distinct_cameras(self) -> list[str]:
        """Return the sorted set of camera ids present in the index."""
        rows = self._conn.execute(
            "SELECT DISTINCT camera_id FROM captures ORDER BY camera_id"
        ).fetchall()
        return [r["camera_id"] for r in rows]

    def distinct_labels(self) -> list[str]:
        """Return the sorted set of labels present in the index."""
        rows = self._conn.execute(
            "SELECT DISTINCT label FROM captures ORDER BY label"
        ).fetchall()
        return [r["label"] for r in rows]

    # ------------------------------------------------------------- delete path
    def delete(self, capture_id: int) -> bool:
        """Delete one capture: unlink both files, then remove the row.

        Files are removed first (idempotently, path-guarded); the DB delete then
        proceeds regardless of unlink outcome. Returns True if a row was removed.
        """
        row = self.get(capture_id)
        if row is None:
            return False
        rels = [row.get("image_path"), row.get("thumb_path"), row.get("audio_path")]
        for rel in rels:
            _safe_unlink(self.captures_dir, rel)
        with self._write_lock:
            cur = self._conn.execute("DELETE FROM captures WHERE id = ?", (capture_id,))
            self._conn.commit()
        _sweep_empty_capture_dirs(self.captures_dir, [r for r in rels if r])
        return cur.rowcount == 1

    def delete_many(self, ids: Iterable[int]) -> int:
        """Delete many captures by id. Returns the number of rows actually removed."""
        id_list = [int(i) for i in ids]
        if not id_list:
            return 0
        total = 0
        swept: list[str] = []
        for chunk in _chunked(id_list, 900):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT image_path, thumb_path, audio_path FROM captures "
                f"WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            for r in rows:
                for key in ("image_path", "thumb_path", "audio_path"):
                    _safe_unlink(self.captures_dir, r[key])
                    if r[key]:
                        swept.append(r[key])
            with self._write_lock:
                cur = self._conn.execute(
                    f"DELETE FROM captures WHERE id IN ({placeholders})", chunk
                )
                self._conn.commit()
                total += cur.rowcount
        _sweep_empty_capture_dirs(self.captures_dir, swept)
        return total

    # ------------------------------------------------------------ update path
    def update_label(self, capture_id: int, new_label: str) -> dict[str, Any] | None:
        """Reclassify one capture. Records ``original_label`` on first edit.

        Returns the updated row dict, or None if ``capture_id`` does not exist.
        Raises ValueError on a blank label (the DB column is NOT NULL and must
        never receive an empty string).
        """
        label = (new_label or "").strip()
        if not label:
            raise ValueError("new_label must be a non-empty string")
        with self._write_lock:
            cur = self._conn.execute(
                """
                UPDATE captures
                   SET original_label = COALESCE(original_label, label),
                       label = ?, reviewed = 1, reviewed_at = ?
                 WHERE id = ?
                """,
                (label, _now_iso(), capture_id),
            )
            self._conn.commit()
        if cur.rowcount == 0:
            return None
        return self.get(capture_id)

    def update_label_many(self, ids: Iterable[int], new_label: str) -> int:
        """Reclassify many captures. Returns the number of rows touched."""
        label = (new_label or "").strip()
        if not label:
            raise ValueError("new_label must be a non-empty string")
        id_list = [int(i) for i in ids]
        if not id_list:
            return 0
        now = _now_iso()
        total = 0
        with self._write_lock:
            for chunk in _chunked(id_list, 900):
                placeholders = ",".join("?" for _ in chunk)
                cur = self._conn.execute(
                    f"""
                    UPDATE captures
                       SET original_label = COALESCE(original_label, label),
                           label = ?, reviewed = 1, reviewed_at = ?
                     WHERE id IN ({placeholders})
                    """,
                    [label, now, *chunk],
                )
                total += cur.rowcount
            self._conn.commit()
        return total

    def mark_reviewed_many(self, ids: Iterable[int]) -> int:
        """Mark many captures reviewed without changing their label.

        Returns the number newly marked (rows already reviewed are not counted).
        """
        id_list = [int(i) for i in ids]
        if not id_list:
            return 0
        now = _now_iso()
        total = 0
        with self._write_lock:
            for chunk in _chunked(id_list, 900):
                placeholders = ",".join("?" for _ in chunk)
                cur = self._conn.execute(
                    f"UPDATE captures SET reviewed = 1, reviewed_at = ? "
                    f"WHERE id IN ({placeholders}) AND reviewed = 0",
                    [now, *chunk],
                )
                total += cur.rowcount
            self._conn.commit()
        return total

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a plain dict over the declared columns."""
        return {key: row[key] for key in _COLUMNS}
