"""wildlife-replay: ground-truth loading and variant scoring."""

from __future__ import annotations

import json
import sqlite3

import numpy as np

from wildlife.frame_features import box_brightness_ratio, ir_glow_reject, is_ir_frame
from wildlife.replay import Row, _glow_keep, load_truth, score
from wildlife.review import ReviewLog


def _row(truth, ir=True, glow=1.0, conf=0.6):
    return Row(1, "cam1", "2026-10-01T03:00:00", "red_fox", conf, truth, ir, glow)


def test_score_counts_kept_reals_and_removed_fps():
    rows = [_row("real", glow=0.5), _row("real", ir=False, glow=3.0),
            _row("fp", glow=2.0), _row("fp", glow=1.1)]
    by = {r["variant"]: r for r in score(rows)}
    base = by["current pipeline (everything saved)"]
    assert (base["real_kept"], base["fp_removed"], base["precision"]) == (2, 0, 0.5)
    gate = by["IR glow gate 1.2x"]
    # The colour frame's bright box is never judged; only the IR fp at 2.0x goes.
    assert (gate["real_kept"], gate["fp_removed"]) == (2, 1)


def test_glow_variant_matches_worker_gate():
    rng = np.random.default_rng(0)
    for _ in range(50):
        level = int(rng.integers(5, 120))
        frame = np.full((120, 160, 3), level, np.uint8)
        if rng.random() < 0.3:
            frame[..., 1] = min(255, level + 40)  # colour frame
        frame[30:70, 40:90] = int(rng.integers(0, 255))
        box = (40, 30, 90, 70)
        row = Row(1, "c", "t", "x", 0.5, "fp", is_ir_frame(frame), box_brightness_ratio(frame, box))
        for t in (1.0, 1.2, 1.5):
            assert _glow_keep(t)(row) == (not ir_glow_reject(frame, box, t))


def test_load_truth_from_review_log_overridden_by_label_file(tmp_path):
    db = tmp_path / "c.db"
    ReviewLog(db).close()  # creates review_log
    conn = sqlite3.connect(db)
    rows = [(1, "animal"), (2, "not_animal"), (3, "unsure"), (4, "not_animal"), (4, "person")]
    for cid, verdict in rows:
        conn.execute(
            "INSERT INTO review_log (capture_id, representative_id, camera_id, capture_ts, label, "
            "verdict, action, reviewed_at) VALUES (?, ?, 'c', 't', 'x', ?, 'kept', 'now')",
            (cid, cid, verdict),
        )
    conn.commit()
    conn.close()
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"2": "real", "9": "fp"}))

    truth = load_truth(db, str(labels))
    # 3 (unsure) is excluded; 4 takes its latest verdict; the file overrides 2.
    assert truth == {1: "real", 2: "real", 4: "real", 9: "fp"}
