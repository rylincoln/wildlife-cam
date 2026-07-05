#!/usr/bin/env python3
"""Box iconic photos with MegaDetector v6 and force a KNOWN species label.

The iNat/GBIF companion to ``convert_lila.py``. Downloaded iNaturalist photos
(``download_inat.py``) have no bounding boxes, but we already KNOW each folder's
species from the taxon query -- so MegaDetector supplies only the *geometry* and
we assign the label ourselves. This sidesteps the failure that made pronghorn
weak (the Idaho set's noisy sequence-level MegaDetector boxes): here we favour
precision hard -- one confident box per iconic photo, tight area/aspect gates,
ambiguous frames dropped to ``review.csv`` rather than mislabeled.

MegaDetector v6 (``weights/MDV6-yolov10e.pt``) runs on the ALREADY-INSTALLED
ultralytics -- no SpeciesNet / PytorchWildlife in the main env (those re-pin
torch and would break the training toolchain). MD is class-agnostic
(animal/person/vehicle), so it boxes species it was never named on -- exactly
what the 5 empty classes need.

CRITICAL: the raw MDv6 ``.pt`` loaded via ultralytics is 0-indexed
{0:animal,1:person,2:vehicle}, NOT the MegaDetector *JSON* convention
(1/2/3). We resolve categories strictly by ``md.names`` (asserted at load), never
by a hard-coded integer -- a single wrong index would silently label PERSON boxes
as the animal species.

Output pool mirrors convert_lila: ``<out>/<observation_id>/<photo_id>.jpg``
(symlink) + sibling ``.txt`` + one ``classes.txt`` manifest, so
``prepare_dataset.py`` merges it and keeps each observation on one split side.

Example::

    python training/label_boxes.py --images ~/wildlife/data/inat/pronghorn \
        --assume-species pronghorn --out ~/wildlife/labeled/inat_pronghorn --tiers 12
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import species  # noqa: E402

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install the training extra:\n"
              "    pip install -e '.[train]'", file=sys.stderr)
        raise SystemExit(1)
    return YOLO


def _dhash(path: Path, size: int = 8) -> int | None:
    """64-bit difference hash for near-duplicate detection (iNat reuses photos)."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    except Exception:  # noqa: BLE001 - a corrupt/unreadable image just isn't deduped
        return None
    px = list(img.tobytes())  # 'L' mode: one byte per pixel, row-major (getdata() is deprecated)
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _resolve_categories(names: dict) -> tuple[int, int, set[int]]:
    """Map animal/person/vehicle to this checkpoint's class indices, BY NAME.

    Guards the 0/1/2-vs-1/2/3 inversion: we abort unless the model really is a
    MegaDetector-style animal/person/vehicle detector.
    """
    by_name = {str(v).strip().lower(): int(k) for k, v in names.items()}
    if "animal" not in by_name or "person" not in by_name:
        raise SystemExit(
            f"Weights are not a MegaDetector animal/person/vehicle detector (names={names}). "
            "Point --weights at MDV6-yolov10e.pt.")
    vehicles = {by_name[n] for n in ("vehicle", "vehicles") if n in by_name}
    return by_name["animal"], by_name["person"], vehicles


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="Input dir of <observation_id>/<photo>.jpg (download_inat.py).")
    p.add_argument("--assume-species", required=True, help="The KNOWN class for every photo here (forced label).")
    p.add_argument("--out", required=True, help="Output label pool.")
    p.add_argument("--weights", default="weights/MDV6-yolov10e.pt", help="MegaDetector v6 ultralytics .pt.")
    p.add_argument("--tiers", default="12")
    p.add_argument("--no-support", action="store_true")
    p.add_argument("--device", default="mps", help="mps | cpu | 0")
    p.add_argument("--imgsz", type=int, default=1280, help="Detector inference size (higher finds smaller subjects).")
    p.add_argument("--conf", type=float, default=0.5, help="Min detection confidence (precision over recall).")
    p.add_argument("--min-area", type=float, default=0.015, help="Drop boxes < this fraction of the frame (specks).")
    p.add_argument("--max-area", type=float, default=0.92, help="Drop boxes > this fraction (MD failed to localize).")
    p.add_argument("--max-aspect", type=float, default=6.0, help="Drop absurdly thin/wide boxes.")
    p.add_argument("--keep-multi", action="store_true",
                   help="Keep ALL qualifying boxes on a frame (each gets the forced label) instead of "
                        "dropping multi-animal frames. Use for HERD species (bighorn_sheep, pronghorn) "
                        "where iconic photos are legitimately multi-animal; the single-species folder "
                        "means every box is that species (conf/area/aspect gates screen MD false boxes).")
    p.add_argument("--dhash-thresh", type=int, default=4, help="Hamming <= this = near-duplicate (drop). -1 disables.")
    p.add_argument("--review-csv", default=None, help="Where to write flagged frames (default: <out>/review.csv).")
    p.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N images (smoke test).")
    args = p.parse_args()

    images_root = Path(args.images).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not images_root.is_dir():
        print(f"Images dir not found: {images_root}", file=sys.stderr)
        return 1

    classes = species.training_classes(tuple(args.tiers), include_support=not args.no_support)
    active = set(classes)
    if args.assume_species not in active:
        print(f"--assume-species {args.assume_species!r} not in the active class set "
              f"({', '.join(classes)}). Check --tiers/--no-support.", file=sys.stderr)
        return 1
    cls_index = {n: i for i, n in enumerate(classes)}
    forced_id = cls_index[args.assume_species]
    want_person = args.assume_species == "person"
    out.mkdir(parents=True, exist_ok=True)
    species.write_manifest(out, classes)

    files = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No images under {images_root}.", file=sys.stderr)
        return 1

    YOLO = _load_yolo()
    md = YOLO(str(Path(args.weights).expanduser()))
    animal_id, person_id, vehicle_ids = _resolve_categories(md.names)
    target_id = person_id if want_person else animal_id
    print(f"MegaDetector names={md.names} -> animal={animal_id} person={person_id} "
          f"vehicle={sorted(vehicle_ids) or '-'}")
    print(f"Boxing {len(files)} photo(s) as '{args.assume_species}' (id {forced_id}), "
          f"conf>={args.conf} imgsz={args.imgsz} device={args.device}")

    stats = Counter()
    seen_hashes: list[int] = []
    review: list[dict] = []

    def _flag(path: Path, reason: str, n: int, best: float) -> None:
        review.append({"filepath": str(path), "assume_species": args.assume_species,
                       "reason": reason, "n_target_boxes": n, "best_conf": f"{best:.3f}"})

    # Predict ONE image at a time (not a batched/streamed list). ultralytics silently
    # skips any image it can't decode, so a positional zip(files, results) would pair
    # every later file with the WRONG photo's boxes -- the exact geometric poisoning
    # this pipeline exists to prevent (and res.path is synthetic 'imageN.jpg' for list
    # sources here, so it can't be mapped back either). Per-image keeps path<->result
    # locked; an undecodable photo yields an empty result we tally as read_error.
    # FP32 on MPS (ultralytics default; half-precision on Apple GPU is discouraged/buggy).
    for path in files:
        res_list = md.predict(source=str(path), conf=args.conf, imgsz=args.imgsz,
                              device=args.device, verbose=False)
        if not res_list:
            stats["review_read_error"] += 1
            _flag(path, "read_error", 0, 0.0)
            continue
        b = res_list[0].boxes
        cls_ids = [int(c) for c in b.cls.tolist()] if b is not None and len(b) else []
        confs = [float(c) for c in b.conf.tolist()] if cls_ids else []
        xywhn = b.xywhn.tolist() if cls_ids else []          # normalized center xywh == YOLO format
        xywh = b.xywh.tolist() if cls_ids else []            # pixels, for aspect

        person_present = any(cid == person_id and cf >= args.conf for cid, cf in zip(cls_ids, confs))

        # Candidate boxes of the target category passing the geometric gates.
        cands: list[tuple[float, tuple[float, float, float, float]]] = []
        for cid, cf, (xc, yc, wn, hn), (_px, _py, pw, ph) in zip(cls_ids, confs, xywhn, xywh):
            if cid != target_id or cf < args.conf:
                continue
            area = wn * hn
            if area < args.min_area or area > args.max_area:
                continue
            asp = max(pw / max(ph, 1e-6), ph / max(pw, 1e-6))
            if asp > args.max_aspect:
                continue
            cands.append((cf, (xc, yc, wn, hn)))

        best_conf = max((c for c, _ in cands), default=0.0)

        if not cands:
            stats["review_no_box"] += 1
            _flag(path, "no_box", 0, best_conf)
            continue
        # An animal frame that also contains a person is likely staged/handled
        # (captive bighorn, held porcupine) -- drop rather than risk a bad exemplar.
        if not want_person and person_present:
            stats["review_person_in_frame"] += 1
            _flag(path, "person_in_frame", len(cands), best_conf)
            continue
        # >=2 confident animal boxes on an iconic photo is ambiguous (herd or
        # mixed-species or a mis-fire) -- one image-level species can't be trusted
        # for every box, so drop by default. --keep-multi overrides for herd species
        # (single-species folder -> every box is that species). People frames may
        # legitimately have several, so want_person never drops on count.
        if not want_person and not args.keep_multi and len(cands) >= 2:
            stats["review_multi"] += 1
            _flag(path, "multi_box", len(cands), best_conf)
            continue

        # Near-duplicate guard: iNat reuses photos across observations; the same
        # image in train AND val silently inflates val mAP.
        if args.dhash_thresh >= 0:
            h = _dhash(path)
            if h is not None:
                if any(_hamming(h, s) <= args.dhash_thresh for s in seen_hashes):
                    stats["review_dup"] += 1
                    _flag(path, "near_duplicate", len(cands), best_conf)
                    continue
                seen_hashes.append(h)

        # person or --keep-multi: keep all qualifying boxes; else the single best animal box.
        boxes = cands if (want_person or args.keep_multi) else cands[:1]
        lines = [f"{forced_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}" for _cf, (xc, yc, wn, hn) in boxes]

        rel = path.relative_to(images_root)          # <obs_id>/<photo>.jpg -> preserve grouping
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            import shutil
            shutil.copy2(path, dst)
        else:
            os.symlink(path, dst)
        dst.with_suffix(".txt").write_text("\n".join(lines) + "\n")
        stats["labeled"] += 1
        stats["boxes"] += len(lines)

    review_csv = Path(args.review_csv).expanduser() if args.review_csv else out / "review.csv"
    with open(review_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["filepath", "assume_species", "reason", "n_target_boxes", "best_conf"])
        w.writeheader()
        w.writerows(review)

    print(f"\nLabeled {stats['labeled']}/{len(files)} photo(s) -> {out}  ({stats['boxes']} boxes)")
    for k in ("review_no_box", "review_person_in_frame", "review_multi", "review_dup", "review_read_error"):
        if stats[k]:
            print(f"  dropped {k[7:]}: {stats[k]}")
    print(f"  flagged {len(review)} frame(s) in {review_csv}")
    if not stats["labeled"]:
        print("  WARNING: zero labels written -- check --weights, --device, and that photos contain the species.",
              file=sys.stderr)
        return 1
    print(f"\nNext: python training/prepare_dataset.py --pool ~/wildlife/labeled --out ~/wildlife/dataset_v2 "
          f"--tiers {args.tiers} --max-per-class 1200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
