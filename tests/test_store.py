"""Unit tests for :mod:`wildlife.store`.

These tests are hardware-free: they exercise the SQLite schema, the JPEG +
thumbnail writer, the dated file layout, relative-path bookkeeping and the query
helpers using a throwaway ``tmp_path`` database and a synthetic numpy BGR frame.
No torch / cv2 imports.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wildlife.models import Detection
from wildlife.store import Store


def _make_frame(width: int = 160, height: int = 120) -> np.ndarray:
    """A synthetic BGR frame (black) with a solid colored box drawn in it."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # BGR red box (blue=0, green=0, red=255) so the JPEG is non-degenerate.
    frame[30:90, 40:120] = (0, 0, 255)
    return frame


def _det(label: str = "bird", confidence: float = 0.87) -> Detection:
    return Detection(
        label=label,
        confidence=confidence,
        box_xyxy=(40.0, 30.0, 120.0, 90.0),
        box_area_frac=0.25,
    )


def _new_store(tmp_path: Path, **kwargs) -> Store:
    store = Store(
        db_path=tmp_path / "captures.db",
        captures_dir=tmp_path / "captures",
        **kwargs,
    )
    store.init_schema()
    return store


def test_init_schema_creates_table_and_indexes(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        # Table exists.
        tables = {
            r[0]
            for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "captures" in tables

        # Indexes from spec 6.5 exist.
        indexes = {
            r[0]
            for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert {"idx_capture_ts", "idx_camera", "idx_label"} <= indexes

        # WAL mode is enabled.
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        store.close()


def test_save_capture_writes_files_and_returns_id(tmp_path: Path) -> None:
    store = _new_store(tmp_path, jpeg_quality=85, thumbnail_px=64)
    try:
        event_ts = datetime(2026, 6, 28, 12, 0, 0)
        capture_ts = datetime(2026, 6, 28, 12, 0, 1, 123456)
        frame = _make_frame()

        capture_id = store.save_capture(
            camera_id="north_field",
            event_ts=event_ts,
            capture_ts=capture_ts,
            frame=frame,
            det=_det("bird", 0.87),
        )
        assert isinstance(capture_id, int)
        assert capture_id > 0

        row = store.get(capture_id)
        assert row is not None

        # Files live under YYYY/MM/DD relative to captures_dir.
        rel_dir = Path("2026", "06", "28")
        assert Path(row["image_path"]).parent == rel_dir
        assert Path(row["thumb_path"]).parent == rel_dir
        assert row["image_path"].endswith(".jpg")
        assert row["thumb_path"].endswith("_thumb.jpg")

        image_abs = store.captures_dir / row["image_path"]
        thumb_abs = store.captures_dir / row["thumb_path"]
        assert image_abs.is_file()
        assert thumb_abs.is_file()

        # Paths are relative, not absolute.
        assert not Path(row["image_path"]).is_absolute()
        assert not Path(row["thumb_path"]).is_absolute()
    finally:
        store.close()


def test_row_roundtrips_fields(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        event_ts = datetime(2026, 6, 28, 9, 30, 0)
        capture_ts = datetime(2026, 6, 28, 9, 30, 2)
        det = _det("dog", 0.73)
        capture_id = store.save_capture(
            camera_id="south_trail",
            event_ts=event_ts,
            capture_ts=capture_ts,
            frame=_make_frame(160, 120),
            det=det,
        )

        row = store.get(capture_id)
        assert row is not None
        assert row["id"] == capture_id
        assert row["camera_id"] == "south_trail"
        assert row["event_ts"] == event_ts.isoformat()
        assert row["capture_ts"] == capture_ts.isoformat()
        assert row["label"] == "dog"
        assert row["confidence"] == 0.73
        assert (row["box_x1"], row["box_y1"], row["box_x2"], row["box_y2"]) == (
            40.0,
            30.0,
            120.0,
            90.0,
        )
        assert row["width"] == 160
        assert row["height"] == 120

        # query() returns the same record.
        results = store.query()
        assert len(results) == 1
        assert results[0]["id"] == capture_id
        # All declared columns present in the dict.
        for key in (
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
        ):
            assert key in results[0]
    finally:
        store.close()


def test_get_missing_returns_none(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        assert store.get(999) is None
    finally:
        store.close()


def test_query_filters(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        base = datetime(2026, 6, 28, 8, 0, 0)
        # Three distinct captures across cameras/labels/confidences/times.
        ids = []
        ids.append(
            store.save_capture(
                camera_id="north_field",
                event_ts=base,
                capture_ts=datetime(2026, 6, 28, 8, 0, 0),
                frame=_make_frame(),
                det=_det("bird", 0.90),
            )
        )
        ids.append(
            store.save_capture(
                camera_id="south_trail",
                event_ts=base,
                capture_ts=datetime(2026, 6, 28, 8, 5, 0),
                frame=_make_frame(),
                det=_det("dog", 0.60),
            )
        )
        ids.append(
            store.save_capture(
                camera_id="north_field",
                event_ts=base,
                capture_ts=datetime(2026, 6, 28, 8, 10, 0),
                frame=_make_frame(),
                det=_det("bird", 0.50),
            )
        )

        # camera_id filter.
        north = store.query(camera_id="north_field")
        assert {r["camera_id"] for r in north} == {"north_field"}
        assert len(north) == 2

        # label filter.
        dogs = store.query(label="dog")
        assert len(dogs) == 1
        assert dogs[0]["label"] == "dog"

        # min_confidence filter.
        high = store.query(min_confidence=0.7)
        assert len(high) == 1
        assert high[0]["confidence"] == 0.90

        # Newest first ordering (08:10 > 08:05 > 08:00).
        ordered = store.query()
        ts_order = [r["capture_ts"] for r in ordered]
        assert ts_order == sorted(ts_order, reverse=True)

        # start/end bound capture_ts inclusively.
        windowed = store.query(
            start=datetime(2026, 6, 28, 8, 5, 0),
            end=datetime(2026, 6, 28, 8, 5, 0),
        )
        assert len(windowed) == 1
        assert windowed[0]["camera_id"] == "south_trail"

        # limit / offset pagination.
        page1 = store.query(limit=2, offset=0)
        page2 = store.query(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 1
        assert {r["id"] for r in page1}.isdisjoint({r["id"] for r in page2})
    finally:
        store.close()


def test_distinct_cameras_and_labels(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        ts = datetime(2026, 6, 28, 8, 0, 0)
        store.save_capture(
            camera_id="north_field",
            event_ts=ts,
            capture_ts=ts,
            frame=_make_frame(),
            det=_det("bird", 0.9),
        )
        store.save_capture(
            camera_id="south_trail",
            event_ts=ts,
            capture_ts=datetime(2026, 6, 28, 8, 1, 0),
            frame=_make_frame(),
            det=_det("dog", 0.8),
        )
        store.save_capture(
            camera_id="north_field",
            event_ts=ts,
            capture_ts=datetime(2026, 6, 28, 8, 2, 0),
            frame=_make_frame(),
            det=_det("bird", 0.7),
        )

        assert store.distinct_cameras() == ["north_field", "south_trail"]
        assert store.distinct_labels() == ["bird", "dog"]
    finally:
        store.close()


def test_thumbnail_is_smaller(tmp_path: Path) -> None:
    store = _new_store(tmp_path, thumbnail_px=64)
    try:
        ts = datetime(2026, 6, 28, 8, 0, 0)
        capture_id = store.save_capture(
            camera_id="north_field",
            event_ts=ts,
            capture_ts=ts,
            frame=_make_frame(160, 120),
            det=_det("bird", 0.9),
        )
        row = store.get(capture_id)
        assert row is not None

        image_abs = store.captures_dir / row["image_path"]
        thumb_abs = store.captures_dir / row["thumb_path"]

        with Image.open(image_abs) as full, Image.open(thumb_abs) as thumb:
            # Full image keeps source dimensions.
            assert full.size == (160, 120)
            # Thumbnail downscaled to thumbnail_px wide, aspect preserved.
            assert thumb.width == 64
            assert thumb.height == 48
            assert thumb.width < full.width
            assert thumb.height < full.height

        # Thumbnail byte size is smaller than the full image.
        assert thumb_abs.stat().st_size < image_abs.stat().st_size
    finally:
        store.close()


def test_new_columns_present_and_defaulted(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        cid = store.save_capture(
            camera_id="cam", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
        )
        row = store.get(cid)
        assert row["reviewed"] == 0
        assert row["original_label"] is None
        assert "reviewed_at" in row
    finally:
        store.close()


def test_busy_timeout_set_per_connection(tmp_path: Path) -> None:
    # A Store that never called init_schema must still have busy_timeout.
    store = Store(db_path=tmp_path / "captures.db", captures_dir=tmp_path / "captures")
    try:
        assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        # busy_timeout alone is a weak assertion (sqlite3.connect's default
        # timeout=5.0 already yields 5000). synchronous defaults to FULL=2, and
        # our __init__ sets it to NORMAL=1, so this genuinely pins the
        # per-connection PRAGMA block: it fails if that line is removed.
        assert store._conn.execute("PRAGMA synchronous").fetchone()[0] == 1
    finally:
        store.close()


def test_migration_adds_columns_to_old_db(tmp_path: Path) -> None:
    import sqlite3
    db = tmp_path / "captures.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL, event_ts TEXT NOT NULL, capture_ts TEXT NOT NULL,
            label TEXT NOT NULL, confidence REAL NOT NULL,
            box_x1 REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL,
            image_path TEXT NOT NULL, thumb_path TEXT NOT NULL,
            width INTEGER, height INTEGER
        );
        INSERT INTO captures (camera_id,event_ts,capture_ts,label,confidence,image_path,thumb_path)
        VALUES ('cam','2020-01-01T00:00:00','2020-01-01T00:00:00','deer',0.9,'a.jpg','a_thumb.jpg');
        """
    )
    conn.commit()
    conn.close()

    store = Store(db_path=db, captures_dir=tmp_path / "captures")
    try:
        store.init_schema()  # ALTER-adds the 3 columns without error
        row = store.get(1)
        assert row["reviewed"] == 0
        assert row["original_label"] is None
        assert "reviewed_at" in row
        store.init_schema()  # idempotent second run — no error
    finally:
        store.close()


class _StaleSchemaConn:
    """Delegates to a real connection but reports an empty table_info.

    Simulates a stale schema snapshot (as a concurrent process would hold
    after another connection added the columns), forcing _migrate to attempt
    the ALTERs against columns that already exist.
    """

    def __init__(self, real):
        self._real = real

    def execute(self, sql, *args, **kwargs):
        if "table_info" in sql:
            return []
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_migrate_tolerates_concurrent_duplicate_column(tmp_path: Path) -> None:
    store = _new_store(tmp_path)  # fully migrated: the 3 columns already exist
    real = store._conn
    try:
        store._conn = _StaleSchemaConn(real)  # pre-check now sees no columns
        store._migrate()  # attempts ALTERs on existing columns -> must not raise
    finally:
        store._conn = real
        store.close()


def test_safe_unlink_removes_and_guards(tmp_path: Path) -> None:
    from wildlife.store import _safe_unlink

    captures = tmp_path / "captures"
    (captures / "2020" / "01" / "01").mkdir(parents=True)
    f = captures / "2020" / "01" / "01" / "x.jpg"
    f.write_bytes(b"hello")

    # Deletes a real file, reports bytes freed.
    deleted, size = _safe_unlink(captures, "2020/01/01/x.jpg")
    assert deleted is True and size == 5 and not f.exists()

    # Missing file is tolerated (idempotent), not an error.
    assert _safe_unlink(captures, "2020/01/01/x.jpg") == (False, 0)

    # Path escaping captures_dir is refused.
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    assert _safe_unlink(captures, "../secret.txt") == (False, 0)
    assert outside.exists()


def test_delete_removes_row_and_files(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        cid = store.save_capture(
            camera_id="cam", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
        )
        row = store.get(cid)
        img = store.captures_dir / row["image_path"]
        thumb = store.captures_dir / row["thumb_path"]
        assert img.exists() and thumb.exists()

        assert store.delete(cid) is True
        assert store.get(cid) is None
        assert not img.exists() and not thumb.exists()
        # The now-empty dated dir is swept.
        assert not (store.captures_dir / "2020" / "01" / "01").exists()
    finally:
        store.close()


def test_delete_unknown_id_and_missing_files(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        assert store.delete(999) is False  # unknown id, no exception
        cid = store.save_capture(
            camera_id="cam", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
        )
        row = store.get(cid)
        (store.captures_dir / row["image_path"]).unlink()  # file already gone
        assert store.delete(cid) is True  # tolerates the missing file
        assert store.get(cid) is None
    finally:
        store.close()


def test_delete_many_counts_only_removed(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        ids = [
            store.save_capture(
                camera_id="cam", event_ts=datetime(2020, 1, 1),
                capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
            )
            for _ in range(3)
        ]
        # Two real ids + one bogus + one duplicate -> 3 rows actually removed.
        removed = store.delete_many([ids[0], ids[1], 999, ids[0], ids[2]])
        assert removed == 3
        assert all(store.get(i) is None for i in ids)
    finally:
        store.close()


def test_delete_many_spans_chunk_boundary(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        rows = [
            ("cam", "2020-01-01T00:00:00", "2020-01-01T00:00:00", "deer", 0.9,
             f"2020/01/01/f{i}.jpg", f"2020/01/01/f{i}_thumb.jpg")
            for i in range(901)
        ]
        store._conn.executemany(
            "INSERT INTO captures "
            "(camera_id,event_ts,capture_ts,label,confidence,image_path,thumb_path) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        store._conn.commit()
        ids = [r["id"] for r in store.query(limit=2000)]
        assert len(ids) == 901
        assert store.delete_many(ids) == 901   # sums rowcount across two chunks
        assert store.count() == 0
    finally:
        store.close()


def test_delete_many_removes_files(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        cid = store.save_capture(
            camera_id="cam", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
        )
        row = store.get(cid)
        img = store.captures_dir / row["image_path"]
        thumb = store.captures_dir / row["thumb_path"]
        assert img.exists() and thumb.exists()
        assert store.delete_many([cid]) == 1
        assert not img.exists() and not thumb.exists()
        assert store.get(cid) is None
    finally:
        store.close()


def test_update_label_records_provenance_once(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        cid = store.save_capture(
            camera_id="cam", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(label="bird"),
        )
        r1 = store.update_label(cid, "deer")
        assert r1["label"] == "deer"
        assert r1["original_label"] == "bird"   # captured on first reclassify
        assert r1["reviewed"] == 1
        assert r1["reviewed_at"]                 # timestamp set

        r2 = store.update_label(cid, "elk")      # a -> b -> c
        assert r2["label"] == "elk"
        assert r2["original_label"] == "bird"    # NOT overwritten
    finally:
        store.close()


def test_update_label_validation_and_missing(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        cid = store.save_capture(
            camera_id="cam", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
        )
        assert store.update_label(999, "deer") is None   # unknown id
        with pytest.raises(ValueError):
            store.update_label(cid, "   ")                 # blank label
    finally:
        store.close()


def test_bulk_update_and_mark_reviewed(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        ids = [
            store.save_capture(
                camera_id="cam", event_ts=datetime(2020, 1, 1),
                capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
            )
            for _ in range(3)
        ]
        assert store.update_label_many(ids, "deer") == 3
        assert all(store.get(i)["label"] == "deer" for i in ids)
        assert all(store.get(i)["reviewed"] == 1 for i in ids)

        # All already reviewed -> mark_reviewed_many counts only 0->1 transitions.
        assert store.mark_reviewed_many(ids) == 0
    finally:
        store.close()


def test_count_and_reviewed_filter(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        ids = [
            store.save_capture(
                camera_id="cam", event_ts=datetime(2020, 1, 1),
                capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
            )
            for _ in range(3)
        ]
        assert store.count() == 3
        store.update_label(ids[0], "deer")  # marks one reviewed

        assert store.count(reviewed=True) == 1
        assert store.count(reviewed=False) == 2
        assert len(store.query(reviewed=False)) == 2
        assert len(store.query(reviewed=True)) == 1
        # count mirrors query length under the same filters.
        assert store.count(camera_id="cam") == len(store.query(camera_id="cam", limit=1000))
    finally:
        store.close()
