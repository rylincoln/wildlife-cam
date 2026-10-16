"""The gallery's Review filter and the /frame confidence floor, over review_log."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from tests.test_frame_routes import _client, _config, _get
from wildlife.models import Detection
from wildlife.review import ReviewLog
from wildlife.store import Store

# name -> (latest verdict or None, reviewer confidence)
ROWS = {
    "fox": ("animal", 0.92),
    "maybe": ("animal", 0.55),
    "moth": ("not_animal", 0.7),
    "blob": ("unsure", 0.4),
    "fresh": (None, None),
}


def _seed(cfg) -> dict[str, int]:
    store = Store(cfg.storage.db_path, cfg.storage.captures_dir)
    store.init_schema()
    ids = {}
    for minute, name in enumerate(ROWS):
        ts = datetime(2026, 10, 15, 3, minute)
        ids[name] = store.save_capture(
            camera_id="south_trail", event_ts=ts, capture_ts=ts, frame=np.zeros((32, 64, 3), np.uint8),
            det=Detection(label="red_fox", confidence=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0), box_area_frac=0.1),
            source_kind="continuous",
        )
    store.close()
    ReviewLog(cfg.storage.db_path).close()
    conn = sqlite3.connect(cfg.storage.db_path)
    for name, (verdict, rc) in ROWS.items():
        if verdict is None:
            continue
        if name == "blob":  # an earlier verdict the latest one supersedes
            conn.execute(
                "INSERT INTO review_log (capture_id, representative_id, camera_id, capture_ts, label, "
                "verdict, reviewer_confidence, action, reviewed_at) "
                "VALUES (?, ?, 'south_trail', 't', 'red_fox', 'animal', 0.9, 'kept', 'now')",
                (ids[name], ids[name]),
            )
        conn.execute(
            "INSERT INTO review_log (capture_id, representative_id, camera_id, capture_ts, label, "
            "verdict, reviewer_confidence, action, reviewed_at) "
            "VALUES (?, ?, 'south_trail', 't', 'red_fox', ?, ?, 'kept', 'now')",
            (ids[name], ids[name], verdict, rc),
        )
    conn.commit()
    conn.close()
    return ids


def _ids(client, query: str = "") -> set[int]:
    data = _get(client, "/api/captures" + query).get_json()
    return {c["id"] for c in data["captures"]}


def test_gallery_hides_rejected_and_unsure_by_default_while_review_runs(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.review.enabled = True
    ids = _seed(cfg)
    client = _client(cfg)
    by = {v: k for k, v in ids.items()}
    assert {by[i] for i in _ids(client)} == {"fox", "maybe", "fresh"}
    assert {by[i] for i in _ids(client, "?review=unsure")} == {"blob"}
    assert {by[i] for i in _ids(client, "?review=confirmed")} == {"fox", "maybe"}
    assert len(_ids(client, "?review=all")) == 5
    assert len(_ids(client, "?review=bogus")) == 3  # unknown -> the default
    assert 'name="review"' in _get(client, "/").get_data(as_text=True)


def test_gallery_shows_everything_while_review_is_off(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed(cfg)
    client = _client(cfg)
    assert len(_ids(client)) == 5
    assert 'name="review"' not in _get(client, "/").get_data(as_text=True)


def test_review_filters_before_any_review_ran(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    store = Store(cfg.storage.db_path, cfg.storage.captures_dir)
    store.init_schema()
    store.save_capture(
        camera_id="south_trail", event_ts=datetime(2026, 10, 15), capture_ts=datetime(2026, 10, 15),
        frame=np.zeros((32, 64, 3), np.uint8),
        det=Detection(label="red_fox", confidence=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0), box_area_frac=0.1),
    )
    assert len(store.query(review="hide_rejected")) == 1
    assert store.query(review="unsure") == [] and store.count(review="confirmed") == 0
    store.close()


def test_frame_needs_a_sure_reviewer(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.review.enabled = True
    ids = _seed(cfg)
    sightings = _get(_client(cfg), "/frame/wildlife").get_json()["sightings"]
    assert [s["thumb_url"] for s in sightings] == [f"/thumb/{ids['fox']}"]  # not the 0.55 "maybe"
