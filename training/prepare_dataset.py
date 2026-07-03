#!/usr/bin/env python3
"""Assemble a YOLO detection dataset from a pool of labeled images.

Takes a *pool* of images (each with a sibling ``<stem>.txt`` YOLO label, or none =
background negative) and produces an Ultralytics-ready dataset::

    <out>/images/{train,val}/...   (symlinks by default, --copy to copy)
    <out>/labels/{train,val}/...
    <out>/data.yaml               (names from training/species.py)

The one non-obvious thing it gets right is the **burst-safe split**: motion
cameras fire bursts of near-identical frames, so putting some frames of an event
in ``train`` and others in ``val`` leaks information and inflates val mAP. Images
are grouped (default: by their immediate parent directory; override with
``--group-regex``) and every image in a group lands on the *same* side of the
split. Keep one event's/sequence's frames in one subfolder (autolabel.py and
convert_lila.py already do this) and the default just works.

Examples::

    # split a pool of labeled captures, 15% to val, Tier-1 class names
    python training/prepare_dataset.py --pool ~/wildlife/labeled --out ~/wildlife/dataset

    # group your own captures by "<camera>_<event-timestamp>"
    python training/prepare_dataset.py --pool ~/wildlife/labeled --out ~/wildlife/dataset \
        --group-regex '([A-Za-z0-9]+_\\d{8}T\\d{6})'
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import species  # noqa: E402  (local module, after path bootstrap)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_label(image: Path, pool: Path, labels_root: Path | None) -> Path | None:
    """Locate an image's YOLO label: sibling ``<stem>.txt``, else under ``labels_root``.

    Under ``labels_root`` the source subtree is mirrored (``labels_root/<rel>.txt``)
    to match how autolabel writes them; a flat ``<stem>.txt`` is a legacy fallback.
    """
    sibling = image.with_suffix(".txt")
    if sibling.is_file():
        return sibling
    if labels_root is not None:
        mirrored = labels_root / image.relative_to(pool).with_suffix(".txt")
        if mirrored.is_file():
            return mirrored
        flat = labels_root / f"{image.stem}.txt"
        if flat.is_file():
            return flat
    return None


def _group_key(image: Path, pool: Path, group_regex: re.Pattern | None) -> str:
    """Return the split-grouping key for an image (keeps bursts together)."""
    rel = image.relative_to(pool)
    if group_regex is not None:
        m = group_regex.search(str(rel))
        if not m:
            # Falling back to parent-dir here would silently split a burst across
            # train/val -- fail loudly so naming/regex gets fixed instead.
            raise ValueError(
                f"--group-regex did not match {rel}; fix the regex or filenames "
                "so every image yields a group key (else burst frames leak across splits)."
            )
        return m.group(1) if m.groups() else m.group(0)
    # Default: the image's immediate parent directory (relative to the pool).
    rel_parent = image.parent.relative_to(pool) if image.parent != pool else Path(".")
    return str(rel_parent)


def _assign_split(group: str, val_frac: float, seed: int) -> str:
    """Deterministically map a group key to ``"train"``/``"val"`` (stable across runs)."""
    digest = hashlib.sha1(f"{seed}:{group}".encode()).hexdigest()
    # Top 8 hex digits -> [0,1). Groups with fraction < val_frac go to val.
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if frac < val_frac else "train"


def _materialize(src: Path, dst: Path, copy: bool) -> None:
    """Place ``src`` at ``dst`` via symlink (default) or copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True, help="Directory of labeled images (recursive).")
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument("--labels", default=None, help="Optional separate labels dir (else sibling .txt).")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of GROUPS held out for val.")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap labeled instances per class per split (stratified balancing across "
                             "sources). Prefer this over Ultralytics --fraction, which takes the first-N "
                             "sorted files and would train on just one source.")
    parser.add_argument("--seed", type=int, default=1, help="Split seed (deterministic).")
    parser.add_argument("--tiers", default="1", help="Species tiers for names, e.g. '1' or '12'.")
    parser.add_argument("--no-support", action="store_true", help="Exclude person/black_bear/bird classes.")
    parser.add_argument("--group-regex", default=None, help="Regex; group 1 (or whole match) is the split key.")
    parser.add_argument("--group-by", choices=["parent", "image"], default="parent",
                        help="'parent' (default, burst-safe) or 'image' (per-image split, for "
                             "sequence-less datasets like ENA24).")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    args = parser.parse_args()

    if yaml is None:
        print("PyYAML is required (pip install pyyaml).", file=sys.stderr)
        return 1

    pool = Path(args.pool).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    labels_root = Path(args.labels).expanduser().resolve() if args.labels else None
    if not pool.is_dir():
        print(f"Pool not found: {pool}", file=sys.stderr)
        return 1

    # Prefer class manifests written by autolabel/convert_lila so label ids line up
    # with names. Search recursively (convert writes one per --out subdir) and in
    # the labels dir, and require agreement -- merged sub-pools labeled with
    # different tiers/flags would otherwise carry conflicting ids under one data.yaml.
    manifest_files = list(pool.rglob(species.MANIFEST))
    if labels_root:
        manifest_files += list(labels_root.rglob(species.MANIFEST))
    manifests = {tuple(m) for m in (species.read_manifest(mf.parent) for mf in manifest_files) if m}
    if len(manifests) > 1:
        print(
            "ERROR: conflicting class manifests under the pool -- sub-pools were "
            "labeled with different tiers/flags. Re-run those with matching --tiers.",
            file=sys.stderr,
        )
        return 1
    if manifests:
        classes = list(next(iter(manifests)))
        print(f"Using class manifest ({len(classes)} classes) from {len(manifest_files)} pool(s)")
    else:
        classes = species.training_classes(tuple(args.tiers), include_support=not args.no_support)
        print(f"No class manifest found; using --tiers {args.tiers} ({len(classes)} classes)")
    n_classes = len(classes)
    group_regex = re.compile(args.group_regex) if args.group_regex else None

    images = sorted(p for p in pool.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    if not images:
        print(
            f"No images found under {pool}.\n"
            "Nothing to do yet — point --pool at labeled captures (see training/README.md).",
            file=sys.stderr,
        )
        return 1

    # Group images so burst/sequence frames never split across train/val.
    # --group-by image makes each image its own group (per-image split) for
    # datasets with no sequence/event structure.
    groups: dict[str, list[Path]] = defaultdict(list)
    for img in images:
        key = str(img.relative_to(pool)) if args.group_by == "image" else _group_key(img, pool, group_regex)
        groups[key].append(img)

    split_counts = Counter()
    class_counts: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    backgrounds = Counter()
    bad_class_ids = 0

    # Shuffle group order (seeded) so a --max-per-class cap fills each class from a
    # MIX of sources, not just whichever pool sorts first (e.g. caltech). Split
    # assignment is by group hash, so this doesn't change which split a group lands in.
    group_items = list(groups.items())
    random.Random(args.seed).shuffle(group_items)

    skipped_capped = 0
    for group, imgs in group_items:
        split = _assign_split(group, args.val_frac, args.seed)
        for img in imgs:
            rel = img.relative_to(pool)
            label = _find_label(img, pool, labels_root)

            # Parse+validate the label FIRST (needed for the per-class cap): drop
            # malformed/out-of-range lines rather than crashing or writing garbage.
            valid: list[str] = []
            img_classes: list[str] = []
            if label is not None:
                for line in label.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except ValueError:
                        bad_class_ids += 1
                        continue
                    if 0 <= cid < n_classes:
                        valid.append(line)
                        img_classes.append(classes[cid])
                    else:
                        bad_class_ids += 1

            # Stratified cap: skip a labeled image only when EVERY class it contains
            # is already at the cap for this split (so rare classes are never starved).
            if args.max_per_class and img_classes and all(
                class_counts[split][c] >= args.max_per_class for c in img_classes
            ):
                skipped_capped += 1
                continue

            img_dst = out / "images" / split / rel
            _materialize(img, img_dst, args.copy)
            split_counts[split] += 1

            if not valid:
                backgrounds[split] += 1
                continue
            for c in img_classes:
                class_counts[split][c] += 1
            lbl_dst = out / "labels" / split / rel.with_suffix(".txt")
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.write_text("\n".join(valid) + "\n")

    # A training set needs both splits populated. A flat pool collapses to one
    # group (all images to one side), so fail loudly rather than emit a broken set.
    if not split_counts["train"] or not split_counts["val"]:
        print(
            f"ERROR: degenerate split (train={split_counts['train']} val={split_counts['val']}). "
            "A flat pool groups all images together; use per-event subfolders or "
            "--group-regex, or adjust --val-frac.",
            file=sys.stderr,
        )
        return 1

    # Write data.yaml.
    data = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "names": species.data_yaml_names(classes),
    }
    with open(out / "data.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)

    # Report.
    print(f"Dataset written to {out}")
    print(f"  groups: {len(groups)}  |  images: train={split_counts['train']} val={split_counts['val']}"
          f"  |  background(no-label): train={backgrounds['train']} val={backgrounds['val']}")
    print(f"  classes ({n_classes}): {', '.join(classes)}")
    print("  per-class instances (train / val):")
    for name in classes:
        t, v = class_counts["train"][name], class_counts["val"][name]
        flag = "  <-- LOW, needs more data" if (t + v) < 50 else ""
        print(f"    {name:22} {t:6d} / {v:<6d}{flag}")
    print("\nNext: python training/train.py --data", out / "data.yaml")
    if bad_class_ids:
        print(f"  ERROR: {bad_class_ids} label line(s) were malformed or had a class id outside "
              f"0..{n_classes - 1} (names/tiers/manifest mismatch); those lines were dropped.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
