"""Tests for the retention prune predicate, incl. the reviewed exemption."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from wildlife.models import Detection
from wildlife.store import Store

# scripts/ is not an importable package; load prune.py by path.
_PRUNE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "prune.py"
_spec = importlib.util.spec_from_file_location("prune", _PRUNE_PATH)
prune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune)


def test_build_where_variants() -> None:
    assert prune._build_where("2020-01-01", 0.0, True) == ("capture_ts < ?", ["2020-01-01"])
    assert prune._build_where("2020-01-01", 0.5, False) == (
        "(capture_ts < ? OR confidence < ?)", ["2020-01-01", 0.5],
    )
    assert prune._build_where("2020-01-01", 0.5, True) == (
        "(capture_ts < ? OR (confidence < ? AND reviewed = 0))", ["2020-01-01", 0.5],
    )


def _frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_reviewed_low_conf_capture_is_spared(tmp_path: Path) -> None:
    store = Store(db_path=tmp_path / "captures.db", captures_dir=tmp_path / "captures")
    store.init_schema()
    try:
        recent = datetime.now()  # not old enough for the age rule
        det = Detection(label="deer", confidence=0.2, box_xyxy=(1, 1, 5, 5), box_area_frac=0.1)
        kept = store.save_capture(camera_id="c", event_ts=recent, capture_ts=recent, frame=_frame(), det=det)
        pruned = store.save_capture(camera_id="c", event_ts=recent, capture_ts=recent, frame=_frame(), det=det)
        store.update_label(kept, "deer")  # marks reviewed (confidence stays 0.2)

        where, params = prune._build_where("1970-01-01T00:00:00", 0.5, True)
        matched = {
            r["id"]
            for r in store._conn.execute(
                f"SELECT id FROM captures WHERE {where}", params
            ).fetchall()
        }
        assert pruned in matched      # unreviewed low-conf is pruned
        assert kept not in matched    # reviewed low-conf is spared
    finally:
        store.close()


def _write_prune_config(config_path: Path, *, captures_dir: Path, db_path: Path) -> None:
    """Write a minimal-but-valid config.yaml for driving ``prune.main()`` in tests."""
    config_path.write_text(
        "cameras:\n"
        "  - id: cam1\n"
        "    host: 192.168.1.50\n"
        "    username: admin\n"
        "    password: pw\n"
        '    rtsp_main: "rtsp://x/main"\n'
        '    rtsp_sub: "rtsp://x/sub"\n'
        "event_source: reolink_native\n"
        "capture:\n"
        "  burst_frames: 1\n"
        "  burst_interval_ms: 100\n"
        "  stream: sub\n"
        "  rtsp_timeout_s: 5\n"
        "  max_concurrent: 1\n"
        "detection:\n"
        '  model_path: "models/yolov8s.pt"\n'
        "  device: cpu\n"
        "  animal_classes: [bird]\n"
        "  confidence_threshold: 0.5\n"
        "  min_box_area_frac: 0.01\n"
        "  save_best_only: true\n"
        "dedupe:\n"
        "  cooldown_s: 10\n"
        "storage:\n"
        f'  captures_dir: "{captures_dir}"\n'
        f'  db_path: "{db_path}"\n'
        "retention:\n"
        "  max_age_days: 30\n"
        "  min_confidence_keep: 0.0\n"
        "gallery:\n"
        '  host: "0.0.0.0"\n'
        "  port: 8080\n"
        "  page_size: 60\n"
        "resource_guard: {}\n"
    )


def test_prune_runs_against_legacy_db_missing_audio_path_column(
    tmp_path: Path, monkeypatch
) -> None:
    """``prune.main()`` must not crash selecting ``audio_path`` on a pre-migration DB.

    A previous fix added ``audio_path`` to prune's SELECT and unlink loop
    without the same has-column guard already used for ``reviewed``. Against a
    ``captures`` table that predates the audio-clip migration, that raised
    ``sqlite3.OperationalError: no such column: audio_path`` and the whole
    prune run failed. Build a legacy (pre-audio_path) schema by hand -- same
    approach as ``test_store.test_source_kind_migrates_onto_a_legacy_db`` --
    and drive prune through its real CLI entry point (``prune.main()``).
    """
    captures_dir = tmp_path / "captures"
    captures_dir.mkdir()
    db_path = tmp_path / "legacy.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE captures ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, "
        "event_ts TEXT NOT NULL, capture_ts TEXT NOT NULL, label TEXT NOT NULL, "
        "confidence REAL NOT NULL, box_x1 REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL, "
        "image_path TEXT NOT NULL, thumb_path TEXT NOT NULL, width INTEGER, height INTEGER)"
    )
    image_rel, thumb_rel = "old.jpg", "old_thumb.jpg"
    (captures_dir / image_rel).write_bytes(b"jpg-bytes")
    (captures_dir / thumb_rel).write_bytes(b"thumb-bytes")
    conn.execute(
        "INSERT INTO captures (camera_id, event_ts, capture_ts, label, confidence, "
        "image_path, thumb_path) VALUES ('cam1', ?, ?, 'bird', 0.9, ?, ?)",
        ("2000-01-01T00:00:00", "2000-01-01T00:00:00", image_rel, thumb_rel),
    )
    conn.commit()
    conn.close()

    config_path = tmp_path / "config.yaml"
    _write_prune_config(config_path, captures_dir=captures_dir, db_path=db_path)

    monkeypatch.setattr(sys, "argv", ["prune.py", "--config", str(config_path)])
    exit_code = prune.main()  # must not raise sqlite3.OperationalError
    assert exit_code == 0

    verify = sqlite3.connect(str(db_path))
    try:
        remaining = verify.execute("SELECT id FROM captures").fetchall()
    finally:
        verify.close()
    assert remaining == []  # old row was actually pruned
    assert not (captures_dir / image_rel).exists()
    assert not (captures_dir / thumb_rel).exists()
