"""Tests for the retention prune predicate, incl. the reviewed exemption."""

from __future__ import annotations

import importlib.util
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
