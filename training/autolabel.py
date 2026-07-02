#!/usr/bin/env python3
"""Turn a folder of images into YOLO labels using Google SpeciesNet.

This is the labeling loop from the plan: point it at camera captures (or
downloaded per-species photos), and it runs the SpeciesNet ensemble
(MegaDetector boxes + species classifier + USA/CO geofencing) and writes a YOLO
``.txt`` next to each image, plus a ``review.csv`` flagging anything it wasn't
confident about for a human to fix (the gallery is a natural review UI later).

Two ways to run:

* ``--run --images DIR`` — invoke SpeciesNet for you (needs the ``autolabel``
  extra: ``pip install -e '.[autolabel]'``), writing/reading ``predictions.json``.
* ``--from-json predictions.json --images DIR`` — parse a ``predictions.json`` you
  already produced (works with no extra installed).

Mapping is keyed on the **scientific binomial** (genus, species) parsed from
SpeciesNet's 7-field label, which avoids its common-name quirks (it emits
``puma`` not "mountain lion", ``grey fox``, ``northern raccoon``). Only classes in
the active tier set (see training/species.py) are written; anything else goes to
``review.csv`` so you never silently mislabel.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import species  # noqa: E402

# SpeciesNet label -> our class, keyed by (genus, species) lowercased. Binomials
# are stable across SpeciesNet's common-name spellings.
SCIENTIFIC_TO_CLASS: dict[tuple[str, str], str] = {
    ("homo", "sapiens"): "person",
    ("ursus", "americanus"): "black_bear",
    ("odocoileus", "hemionus"): "mule_deer",
    ("odocoileus", "virginianus"): "mule_deer",  # regional model lumps deer
    ("cervus", "canadensis"): "elk",
    ("canis", "latrans"): "coyote",
    ("vulpes", "vulpes"): "red_fox",
    ("urocyon", "cinereoargenteus"): "gray_fox",
    ("puma", "concolor"): "mountain_lion",
    ("lynx", "rufus"): "bobcat",
    ("lynx", "canadensis"): "canada_lynx",
    ("meleagris", "gallopavo"): "wild_turkey",
    ("procyon", "lotor"): "raccoon",
    ("mephitis", "mephitis"): "striped_skunk",
    ("alces", "alces"): "moose",
    ("antilocapra", "americana"): "pronghorn",
    ("ovis", "canadensis"): "bighorn_sheep",
    ("oreamnos", "americanus"): "mountain_goat",
    ("erethizon", "dorsatum"): "porcupine",
    ("marmota", "flaviventris"): "yellow_bellied_marmot",
    ("taxidea", "taxus"): "badger",
    ("martes", "americana"): "pine_marten",
    ("dendragapus", "obscurus"): "dusky_grouse",
}
# Genus-only rollups we'll accept but flag for review (regionally dominant).
GENUS_FALLBACK: dict[str, str] = {"odocoileus": "mule_deer"}


def parse_label(label: str) -> tuple[str, str, str, str]:
    """Return (klass, genus, species, common_name) from a 7-field SpeciesNet label."""
    parts = label.split(";")
    parts += [""] * (7 - len(parts))
    _uuid, klass, _order, _family, genus, sp, common = parts[:7]
    return klass.lower(), genus.lower(), sp.lower(), common


def map_species(label: str, active: set[str]) -> tuple[str | None, str]:
    """Map a SpeciesNet label to an active class. Returns (class_or_None, status)."""
    klass, genus, sp, _common = parse_label(label)
    cls = SCIENTIFIC_TO_CLASS.get((genus, sp))
    if cls and cls in active:
        return cls, "labeled"
    if not sp and genus in GENUS_FALLBACK and GENUS_FALLBACK[genus] in active:
        return GENUS_FALLBACK[genus], "review:genus"
    # Generic bird fallback: an unmatched bird becomes 'bird' if that class exists.
    if klass == "aves" and "bird" in active:
        return "bird", "review:genus"
    return None, "review:unmapped"


def _norm_box_to_yolo(bbox: list[float]) -> tuple[float, float, float, float]:
    """SpeciesNet bbox [xmin,ymin,w,h] (normalized) -> YOLO (xc,yc,w,h) normalized."""
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0, w, h


def process(predictions: list[dict], active_classes: list[str], det_thresh: float,
            class_thresh: float, labels_out: Path | None, images_root: Path | None = None) -> list[dict]:
    """Write YOLO labels for each prediction; return review rows.

    A row is added to review whenever a human should look: an animal was detected
    but not confidently labeled; a confident species has no detection box (so the
    frame would otherwise be a silent background negative); a genus-level guess was
    used; or the frame has multiple animals (one image-level species can't be
    trusted for every box).
    """
    active = set(active_classes)
    cls_index = {name: i for i, name in enumerate(active_classes)}
    person_ok = "person" in active
    review: list[dict] = []

    def _flag(filepath, top_label, species_cls, top_score, n_animal, reason):
        review.append({"filepath": str(filepath), "top_common_name": parse_label(top_label)[3],
                       "mapped_class": species_cls or "", "score": f"{top_score:.3f}",
                       "n_animal_boxes": n_animal, "status": reason})

    for pred in predictions:
        filepath = Path(pred["filepath"])
        dets = pred.get("detections") or []
        top_label = pred.get("prediction", "")
        top_score = float(pred.get("prediction_score", 0.0))

        species_cls, status = (None, "blank")
        if top_label:
            species_cls, status = map_species(top_label, active)
            if species_cls and top_score < class_thresh:
                status = "review:low_conf"
        elif dets:
            status = "review:no_species"  # detector ran but the classifier didn't

        lines: list[str] = []
        n_animal = 0
        n_labeled = 0  # animal boxes we actually labeled (person boxes don't count)
        for det in dets:
            if float(det.get("conf", 0.0)) < det_thresh:
                continue
            category = str(det.get("category", ""))
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            xc, yc, w, h = _norm_box_to_yolo(bbox)
            if category == "2" and person_ok:  # human
                lines.append(f"{cls_index['person']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            elif category == "1":  # animal
                n_animal += 1
                if species_cls and status in ("labeled", "review:genus") and top_score >= class_thresh:
                    lines.append(f"{cls_index[species_cls]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    n_labeled += 1
            # category "3" (vehicle) -> skipped

        # Write the label (empty = background negative), mirroring the source subtree
        # under --labels-out so identical stems in different folders don't collide.
        if labels_out is not None:
            try:
                rel = filepath.resolve().relative_to(images_root.resolve()) if images_root else Path(filepath.name)
            except ValueError:
                rel = Path(filepath.name)
            dst = (labels_out / rel).with_suffix(".txt")
        else:
            dst = filepath.with_suffix(".txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(("\n".join(lines) + "\n") if lines else "")

        # Flag for human review so nothing is silently mis/under-labeled.
        if n_animal and not n_labeled:
            _flag(filepath, top_label, species_cls, top_score, n_animal,
                  status if status.startswith("review") else "review:unmapped")
        elif n_labeled and n_animal > 1:
            _flag(filepath, top_label, species_cls, top_score, n_animal, "review:multi")
        elif n_labeled and status == "review:genus":
            _flag(filepath, top_label, species_cls, top_score, n_animal, "review:genus")
        elif not lines and species_cls and top_score >= class_thresh:
            # Confident species but nothing was written -> would be a silent
            # background negative; flag it instead.
            _flag(filepath, top_label, species_cls, top_score, n_animal, "review:no_box")
    return review


def _run_speciesnet(images: Path, out_json: Path, country: str, admin1: str | None) -> None:
    cmd = [sys.executable, "-m", "speciesnet.scripts.run_model",
           "--folders", str(images), "--predictions_json", str(out_json), "--country", country]
    if admin1:
        cmd += ["--admin1_region", admin1]
    print("Running SpeciesNet:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", help="Folder of images (required with --run).")
    p.add_argument("--from-json", help="Existing SpeciesNet predictions.json to parse.")
    p.add_argument("--run", action="store_true", help="Invoke SpeciesNet to produce predictions.")
    p.add_argument("--predictions-json", default=None, help="Where --run writes predictions (default: <images>/predictions.json).")
    p.add_argument("--country", default="USA")
    p.add_argument("--admin1", default="CO", help="ISO 3166-2 subdivision, e.g. CO.")
    p.add_argument("--tiers", default="12", help="Active species tiers, e.g. '1' or '12'.")
    p.add_argument("--no-support", action="store_true")
    p.add_argument("--det-thresh", type=float, default=0.2, help="Min detection confidence.")
    p.add_argument("--class-thresh", type=float, default=0.5, help="Min species confidence to auto-label.")
    p.add_argument("--labels-out", default=None, help="Write labels here instead of next to images.")
    p.add_argument("--review-csv", default="review.csv")
    args = p.parse_args()

    classes = species.training_classes(tuple(args.tiers), include_support=not args.no_support)
    # Record the class list next to the labels so prepare_dataset uses the same ids.
    manifest_dir = args.labels_out or args.images
    if manifest_dir:
        species.write_manifest(manifest_dir, classes)

    if args.run:
        if not args.images:
            print("--run requires --images.", file=sys.stderr)
            return 1
        images = Path(args.images).expanduser().resolve()
        out_json = Path(args.predictions_json).expanduser() if args.predictions_json else images / "predictions.json"
        try:
            _run_speciesnet(images, out_json, args.country, args.admin1)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"SpeciesNet run failed ({exc}). Install it with: pip install -e '.[autolabel]'", file=sys.stderr)
            return 1
        pred_path = out_json
    elif args.from_json:
        pred_path = Path(args.from_json).expanduser()
    else:
        print("Provide --run (with --images) or --from-json.", file=sys.stderr)
        return 1

    data = json.loads(Path(pred_path).read_text())
    predictions = data.get("predictions", [])
    labels_out = Path(args.labels_out).expanduser() if args.labels_out else None
    images_root = Path(args.images).expanduser().resolve() if args.images else None
    review = process(predictions, classes, args.det_thresh, args.class_thresh, labels_out, images_root)

    with open(args.review_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["filepath", "top_common_name", "mapped_class",
                                           "score", "n_animal_boxes", "status"])
        w.writeheader()
        w.writerows(review)

    labeled = len(predictions) - len(review)
    print(f"Processed {len(predictions)} image(s): ~{labeled} auto-labeled, {len(review)} flagged in {args.review_csv}")
    print(f"Active classes ({len(classes)}): {', '.join(classes)}")
    print("Next: correct review.csv rows, then  python training/prepare_dataset.py --pool", args.images or "<images>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
