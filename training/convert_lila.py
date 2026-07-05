#!/usr/bin/env python3
"""Convert a LILA / COCO-Camera-Traps dataset into a YOLO label pool.

The bootstrap for *before you have your own captures*: turn Caltech Camera Traps,
ENA24, etc. into YOLO training data. Handles both shapes the research pinned:

* **Native species boxes** (e.g. ENA24): each annotation has a ``bbox`` (absolute
  pixels) with a species ``category_id`` -> converted directly.
* **Image-level species + separate animal boxes** (e.g. Caltech): species is an
  image-level category; geometry comes from a LILA **MegaDetector-results** file
  (``--md-results``) whose ``bbox_relative`` boxes are class-agnostic "animal" and
  get the image's species.

Output pool: ``<out>/<group>/<name>.jpg`` (symlink) + sibling ``.txt``, grouped by
``seq_id`` (then ``location``, then a per-image key for sequence-less datasets)
so prepare_dataset.py's default parent-dir grouping keeps burst frames together
while independent images (e.g. ENA24) still split per-image. Categories outside
the active class set are **not**
silently turned into background — those images are skipped, so the model never
learns a real animal as "nothing".

Example::

    python training/convert_lila.py --images ~/data/caltech/cct_images \
        --cct ~/data/caltech/caltech_bboxes_20200316.json \
        --md-results ~/data/caltech/caltech_camera_traps_mdv5a.0.0_results.json \
        --out ~/wildlife/labeled/caltech --tiers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import species  # noqa: E402

# Source category name (normalized) -> our class. None/absent = unmapped (skip image).
_ALIASES: dict[str, str] = {
    "deer": "mule_deer", "mule_deer": "mule_deer",
    "white_tailed_deer": "mule_deer", "whitetailed_deer": "mule_deer",
    "elk": "elk", "moose": "moose", "pronghorn": "pronghorn",
    "bighorn": "bighorn_sheep", "bighorn_sheep": "bighorn_sheep",
    "coyote": "coyote", "fox": "red_fox", "red_fox": "red_fox",
    "gray_fox": "gray_fox", "grey_fox": "gray_fox",
    "cougar": "mountain_lion", "mountain_lion": "mountain_lion", "puma": "mountain_lion",
    "bobcat": "bobcat", "lynx": "canada_lynx", "canada_lynx": "canada_lynx",
    "raccoon": "raccoon", "northern_raccoon": "raccoon",
    "skunk": "striped_skunk", "striped_skunk": "striped_skunk",
    "squirrel": "tree_squirrel", "tree_squirrel": "tree_squirrel",
    "eastern_gray_squirrel": "tree_squirrel", "eastern_fox_squirrel": "tree_squirrel",
    "gray_squirrel": "tree_squirrel", "fox_squirrel": "tree_squirrel",
    "rabbit": "cottontail", "cottontail": "cottontail", "eastern_cottontail": "cottontail",
    "snowshoe_hare": "snowshoe_hare",
    "american_crow": "bird", "crow": "bird", "raven": "bird", "magpie": "bird",
    "porcupine": "porcupine", "marmot": "yellow_bellied_marmot",
    "yellow_bellied_marmot": "yellow_bellied_marmot",
    "badger": "badger", "american_badger": "badger",
    "turkey": "wild_turkey", "wild_turkey": "wild_turkey", "bird": "bird",
    "bear": "black_bear", "black_bear": "black_bear", "american_black_bear": "black_bear",
    "human": "person", "person": "person",
}
_EMPTY = "__empty__"  # sentinel for the CCT 'empty' category


def _norm_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def abs_bbox_to_yolo(bbox, w, h):
    """CCT absolute [xmin,ymin,bw,bh] pixels -> YOLO (xc,yc,w,h) normalized."""
    x, y, bw, bh = bbox
    return (x + bw / 2.0) / w, (y + bh / 2.0) / h, bw / w, bh / h


def norm_bbox_to_yolo(bbox):
    """Normalized [xmin,ymin,bw,bh] (MD bbox_relative) -> YOLO (xc,yc,w,h) normalized."""
    x, y, bw, bh = bbox
    return x + bw / 2.0, y + bh / 2.0, bw, bh


def _map_category(name: str, active: set[str]) -> str | None:
    """Return our active class for a source category name, ``_EMPTY``, or None (unmapped)."""
    n = _norm_name(name)
    if n in ("empty", "blank", "none"):
        return _EMPTY
    cls = _ALIASES.get(n)
    return cls if (cls and cls in active) else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="Dataset images root (file_name is relative to this).")
    p.add_argument("--cct", required=True, help="COCO-Camera-Traps JSON (species categories + annotations).")
    p.add_argument("--md-results", default=None, help="LILA MegaDetector-results JSON (bbox_relative boxes).")
    p.add_argument("--out", required=True, help="Output label pool.")
    p.add_argument("--tiers", default="12")
    p.add_argument("--no-support", action="store_true")
    p.add_argument("--conf-thresh", type=float, default=0.2, help="Min MD detection confidence.")
    p.add_argument("--min-box-rel-area", type=float, default=0.0,
                   help="Drop boxes smaller than this fraction of the frame. The Idaho set has no "
                        "native boxes, so its MegaDetector fired on distant specks across whole "
                        "sequences, all inheriting the sequence-level label (89%% of pronghorn boxes "
                        "were <1%% of frame -> 'pronghorn = speck'). Re-convert idaho with 0.01 to prune.")
    p.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    args = p.parse_args()

    images_root = Path(args.images).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    classes = species.training_classes(tuple(args.tiers), include_support=not args.no_support)
    active = set(classes)
    cls_index = {n: i for i, n in enumerate(classes)}
    species.write_manifest(out, classes)  # so prepare_dataset uses these exact ids

    cct = json.loads(Path(args.cct).expanduser().read_text())
    cat_name = {c["id"]: c["name"] for c in cct.get("categories", [])}
    images = {im["id"]: im for im in cct.get("images", [])}
    anns_by_img: dict[str, list[dict]] = defaultdict(list)
    for a in cct.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    md_by_file: dict[str, list[dict]] = {}
    if args.md_results:
        md = json.loads(Path(args.md_results).expanduser().read_text())
        for im in md.get("images", []):
            md_by_file[im["file"]] = im.get("detections", [])

    stats = Counter()
    unmapped: Counter = Counter()

    def _emit(image: dict, lines: list[str]) -> bool:
        """Materialize one image + label; return False if the source is missing."""
        file_name = image["file_name"]
        src_img = images_root / file_name
        if not src_img.is_file():
            return False  # e.g. privacy-removed images absent from the public zip
        # Group by sequence/event so prepare_dataset's default split keeps bursts
        # together. Datasets with no seq/location (e.g. ENA24) are independent
        # images -> give each its own group so they split per-image instead of all
        # landing in one split.
        group = str(image.get("seq_id") or image.get("location") or f"img_{image['id']}")
        safe = file_name.replace("/", "__").replace("\\", "__")
        dst_img = out / group / safe
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if args.copy:
            import shutil

            shutil.copy2(src_img, dst_img)
        else:
            os.symlink(src_img, dst_img)
        dst_img.with_suffix(".txt").write_text(("\n".join(lines) + "\n") if lines else "")
        return True

    for img_id, image in images.items():
        anns = anns_by_img.get(img_id, [])
        mapped_cats = [_map_category(cat_name.get(a["category_id"], ""), active) for a in anns]
        # If any non-empty category is unmapped/out-of-tier, skip (avoid false background).
        if any(m is None for m in mapped_cats):
            stats["skip_unmapped"] += 1
            for a, m in zip(anns, mapped_cats):
                if m is None:
                    unmapped[cat_name.get(a["category_id"], "?")] += 1
            continue

        # Pure-empty image -> background negative. Require a non-empty list so a
        # *zero-annotation* image (unknown content) isn't a vacuous-True background.
        if mapped_cats and all(m == _EMPTY for m in mapped_cats):
            stats["background" if _emit(image, []) else "skip_missing"] += 1
            continue

        lines: list[str] = []
        # (a) Native per-annotation boxes (species-labeled). Only take this path for
        # boxes we can actually convert -- an absolute bbox with no image dims must
        # fall through to (b)'s MegaDetector boxes instead of being dropped.
        boxed = [
            (a, m) for a, m in zip(anns, mapped_cats)
            if m not in (None, _EMPTY)
            and ("bbox_relative" in a or ("bbox" in a and image.get("width") and image.get("height")))
        ]
        if boxed:
            w = image.get("width")
            h = image.get("height")
            for a, m in boxed:
                if "bbox_relative" in a:
                    xc, yc, bw, bh = norm_bbox_to_yolo(a["bbox_relative"])
                elif w and h:
                    xc, yc, bw, bh = abs_bbox_to_yolo(a["bbox"], w, h)
                else:
                    stats["skip_no_dims"] += 1
                    continue
                if bw * bh < args.min_box_rel_area:
                    stats["skip_tiny_box"] += 1
                    continue
                lines.append(f"{cls_index[m]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        else:
            # (b) Image-level species + MegaDetector animal boxes.
            species_present = {m for m in mapped_cats if m not in (None, _EMPTY)}
            dets = md_by_file.get(image["file_name"], [])
            if len(species_present) != 1 or not dets:
                stats["skip_no_boxes"] += 1  # ambiguous species, or no boxes available
                continue
            (only_cls,) = tuple(species_present)
            # A 'person' image's individuals are MegaDetector category 2 (human),
            # not 1 (animal) -- pick the matching category or the box is dropped.
            want = "2" if only_cls == "person" else "1"
            for d in dets:
                if str(d.get("category")) != want or float(d.get("conf", 0)) < args.conf_thresh:
                    continue
                box = d.get("bbox_relative") or d.get("bbox")  # both normalized in LILA/MD
                if not box:
                    continue
                xc, yc, bw, bh = norm_bbox_to_yolo(box)
                if bw * bh < args.min_box_rel_area:
                    stats["skip_tiny_box"] += 1
                    continue
                lines.append(f"{cls_index[only_cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            stats["labeled" if _emit(image, lines) else "skip_missing"] += 1
        else:
            stats["skip_no_boxes"] += 1

    print(f"Converted into {out}")
    for k in ("labeled", "background", "skip_unmapped", "skip_no_boxes", "skip_no_dims",
              "skip_tiny_box", "skip_missing"):
        if stats[k]:
            print(f"  {k}: {stats[k]}")
    if unmapped:
        print("  top unmapped source categories (add to _ALIASES or ignore):")
        for name, n in unmapped.most_common(12):
            print(f"    {name}: {n}")
    print(f"\nNext: python training/prepare_dataset.py --pool {out} --out <dataset> --tiers {args.tiers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
