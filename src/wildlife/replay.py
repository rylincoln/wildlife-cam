"""Replay saved captures through candidate pipeline gates and score them.

Ground truth comes from ``review_log`` (the ``wildlife-review`` verdicts:
animal/person = real, not_animal = false positive, unsure = left out) and/or a
hand-label JSON file (``{"<capture id>": "real" | "fp"}``), which wins on conflict.

For each labelled capture still on disk it computes the per-box features the
worker's gates use -- with the very same functions from
:mod:`wildlife.frame_features` -- then reports, per variant, how many real
detections survive and how many false positives it removes.

Limits worth remembering when reading the numbers: only captures the live
pipeline *saved* are here, so a variant can only lose real animals, never find
new ones -- "real kept" is recall relative to today's pipeline, not absolute.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from wildlife.config import load_config
from wildlife.frame_features import box_brightness_ratio, is_ir_frame

# Decode JPEGs at half resolution (PIL draft mode): ~4x faster on the 4K frames,
# and a mean-brightness ratio is unaffected by the downscale.
_DRAFT_SCALE = 2


@dataclass
class Row:
    id: int
    camera_id: str
    capture_ts: str
    label: str
    confidence: float
    truth: str  # "real" | "fp"
    ir: bool
    glow: float  # box_brightness_ratio


Variant = tuple[str, Callable[[Row], bool]]  # (name, keep?)


def _glow_keep(max_ratio: float) -> Callable[[Row], bool]:
    # Mirrors frame_features.ir_glow_reject: reject only IR frames above max_ratio.
    return lambda r: not (r.ir and r.glow > max_ratio)


VARIANTS: list[Variant] = [
    ("current pipeline (everything saved)", lambda r: True),
    ("confidence >= 0.50", lambda r: r.confidence >= 0.50),
    ("confidence >= 0.60", lambda r: r.confidence >= 0.60),
    ("confidence >= 0.70", lambda r: r.confidence >= 0.70),
    ("IR glow gate 1.5x", _glow_keep(1.5)),
    ("IR glow gate 1.2x", _glow_keep(1.2)),
    ("IR glow gate 1.0x", _glow_keep(1.0)),
    ("IR glow gate 1.2x + conf >= 0.50", lambda r: _glow_keep(1.2)(r) and r.confidence >= 0.50),
    ("drop every IR detection", lambda r: not r.ir),
]


def load_truth(db_path: str | Path, labels_file: str | None) -> dict[int, str]:
    """capture id -> "real" | "fp" from review_log (latest verdict) overlaid by a label file."""
    truth: dict[int, str] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        has_log = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_log'"
        ).fetchone()
        if has_log:
            for cid, verdict in conn.execute(
                "SELECT capture_id, verdict FROM review_log r WHERE id = "
                "(SELECT MAX(id) FROM review_log WHERE capture_id = r.capture_id)"
            ):
                if verdict in ("animal", "person"):
                    truth[int(cid)] = "real"
                elif verdict == "not_animal":
                    truth[int(cid)] = "fp"
    finally:
        conn.close()
    if labels_file:
        for cid, lab in json.loads(Path(labels_file).read_text()).items():
            if lab not in ("real", "fp"):
                raise SystemExit(f"{labels_file}: label for {cid} must be 'real' or 'fp', got {lab!r}")
            truth[int(cid)] = lab
    return truth


def load_rows(db_path, captures_dir, truth: dict[int, str], since: str | None) -> list[Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    sql = ("SELECT id, camera_id, capture_ts, label, confidence, box_x1, box_y1, box_x2, box_y2, "
           "image_path FROM captures WHERE source_kind != 'audio'")
    args: list = []
    if since:
        sql += " AND capture_ts >= ?"
        args.append(since)
    rows: list[Row] = []
    try:
        for r in conn.execute(sql, args):
            cid = int(r["id"])
            if cid not in truth:
                continue
            path = Path(captures_dir) / r["image_path"]
            if not path.is_file():
                continue
            with Image.open(path) as im:
                full_w = im.width
                im.draft("RGB", (im.width // _DRAFT_SCALE, im.height // _DRAFT_SCALE))
                rgb = np.asarray(im.convert("RGB"))
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            s = bgr.shape[1] / full_w
            box = (r["box_x1"] * s, r["box_y1"] * s, r["box_x2"] * s, r["box_y2"] * s)
            rows.append(Row(cid, r["camera_id"], r["capture_ts"], r["label"],
                            float(r["confidence"]), truth[cid], is_ir_frame(bgr),
                            box_brightness_ratio(bgr, box)))
    finally:
        conn.close()
    return rows


def score(rows: list[Row], variants: list[Variant] = VARIANTS) -> list[dict]:
    real = [r for r in rows if r.truth == "real"]
    fps = [r for r in rows if r.truth == "fp"]
    out = []
    for name, keep in variants:
        real_kept = sum(1 for r in real if keep(r))
        fp_kept = sum(1 for r in fps if keep(r))
        kept = real_kept + fp_kept
        out.append({
            "variant": name,
            "real_kept": real_kept, "real_total": len(real),
            "fp_removed": len(fps) - fp_kept, "fp_total": len(fps),
            "precision": real_kept / kept if kept else float("nan"),
        })
    return out


def print_table(results: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'variant':36s} {'real kept':>11s} {'FPs removed':>13s} {'precision':>10s}")
    for r in results:
        print(f"  {r['variant']:36s} {r['real_kept']:5d}/{r['real_total']:<5d} "
              f"{r['fp_removed']:6d}/{r['fp_total']:<6d} {r['precision']:9.0%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wildlife-replay", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=os.environ.get("WILDLIFE_CONFIG", "config.yaml"))
    parser.add_argument("--labels", help='JSON {"<capture id>": "real"|"fp"}; overrides review_log')
    parser.add_argument("--since", help="only captures at/after this ISO date")
    parser.add_argument("--by-camera", action="store_true", help="also score each camera separately")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    truth = load_truth(config.storage.db_path, args.labels)
    if not truth:
        print("No labels yet: run `wildlife-review run --dry-run` or pass --labels.", file=sys.stderr)
        return 1
    rows = load_rows(config.storage.db_path, config.storage.captures_dir, truth, args.since)
    n_real = sum(r.truth == "real" for r in rows)
    print(f"{len(rows)} labelled captures on disk: {n_real} real, {len(rows) - n_real} false positives "
          f"({sum(r.ir for r in rows)} in IR night mode).")
    print_table(score(rows), "All cameras")
    if args.by_camera:
        for cam in sorted({r.camera_id for r in rows}):
            print_table(score([r for r in rows if r.camera_id == cam]), cam)
    return 0


if __name__ == "__main__":
    sys.exit(main())
