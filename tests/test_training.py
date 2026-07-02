"""Hardware-free tests for the offline training scaffold (training/).

Covers the pure-Python logic that must be correct before any real training run:
the class taxonomy (single source of truth) and the burst-safe train/val split.
No torch/ultralytics/speciesnet involved.
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(_TRAINING))

import autolabel  # noqa: E402
import convert_lila  # noqa: E402
import prepare_dataset  # noqa: E402
import species  # noqa: E402

# SpeciesNet 7-field labels (verbatim binomials from taxonomy_release.txt).
_MULE_DEER = "uuid;mammalia;artiodactyla;cervidae;odocoileus;hemionus;mule deer"
_PUMA = "uuid;mammalia;carnivora;felidae;puma;concolor;puma"
_HUMAN = "uuid;mammalia;primates;hominidae;homo;sapiens;human"
_FOX_SQUIRREL = "uuid;mammalia;rodentia;sciuridae;sciurus;niger;fox squirrel"


def test_training_classes_support_first_and_deduped() -> None:
    t1 = species.training_classes(("1",))
    assert t1[:3] == ["person", "black_bear", "bird"]  # support classes lead
    assert t1.count("mule_deer") == 1
    # Tiers compose without duplicating support.
    t12 = species.training_classes(("1", "2"))
    assert t12[:3] == ["person", "black_bear", "bird"]
    assert "bighorn_sheep" in t12 and len(t12) == len(set(t12))
    # Opting out of support drops person/bear/bird.
    wild = species.training_classes(("1",), include_support=False)
    assert "person" not in wild and wild[0] == "mule_deer"


def test_data_yaml_and_config_snippet() -> None:
    classes = species.training_classes(("1",))
    names = species.data_yaml_names(classes)
    assert names[0] == "person" and names[3] == "mule_deer"
    snippet = species.config_animal_classes(classes)
    assert snippet.startswith("animal_classes:")
    assert "  - mule_deer" in snippet


def test_assign_split_is_deterministic_and_balanced() -> None:
    # Same group+seed is stable; distribution roughly honors val_frac.
    a = prepare_dataset._assign_split("evt_42", 0.2, seed=1)
    b = prepare_dataset._assign_split("evt_42", 0.2, seed=1)
    assert a == b
    val = sum(prepare_dataset._assign_split(f"g{i}", 0.2, 1) == "val" for i in range(2000))
    assert 0.15 < val / 2000 < 0.25


def test_group_key_default_parent_and_regex() -> None:
    pool = Path("/pool")
    img = pool / "evt_b" / "f3.jpg"
    assert prepare_dataset._group_key(img, pool, None) == "evt_b"
    import re

    rx = re.compile(r"(cam\d+_\d{8})")
    img2 = pool / "sub" / "cam1_20260115_f0.jpg"
    assert prepare_dataset._group_key(img2, pool, rx) == "cam1_20260115"


def test_prepare_dataset_no_burst_leak(tmp_path: Path) -> None:
    classes = species.training_classes(("1",))
    deer = classes.index("mule_deer")
    pool = tmp_path / "pool"
    for g in [f"evt_{c}" for c in "abcdefgh"]:
        gd = pool / g
        gd.mkdir(parents=True)
        for i in range(4):  # a burst of near-identical frames
            (gd / f"f{i}.jpg").write_bytes(b"x")
            (gd / f"f{i}.txt").write_text(f"{deer} 0.5 0.5 0.2 0.3\n")
        (gd / "empty.jpg").write_bytes(b"x")  # background negative (no label)

    out = tmp_path / "dataset"
    r = subprocess.run(
        [sys.executable, str(_TRAINING / "prepare_dataset.py"),
         "--pool", str(pool), "--out", str(out), "--val-frac", "0.3", "--seed", "1"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr

    # No event's frames may appear in both splits.
    where = defaultdict(set)
    for split in ("train", "val"):
        for img in (out / "images" / split).rglob("*.jpg"):
            where[img.parent.name].add(split)
    assert all(len(s) == 1 for s in where.values()), where

    import yaml

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"][deer] == "mule_deer"
    assert data["train"] == "images/train" and data["val"] == "images/val"


# --------------------------------------------------------------------------- #
# autolabel (SpeciesNet -> YOLO)
# --------------------------------------------------------------------------- #
def test_map_species_uses_binomial_not_common_name() -> None:
    active = set(species.training_classes(("1", "2")))
    # 'puma' common name would miss 'mountain lion'; binomial (puma, concolor) hits.
    assert autolabel.map_species(_PUMA, active) == ("mountain_lion", "labeled")
    assert autolabel.map_species(_MULE_DEER, active) == ("mule_deer", "labeled")
    assert autolabel.map_species(_HUMAN, active) == ("person", "labeled")
    # fox squirrel isn't in tier1/2 -> unmapped (goes to review, never mislabeled).
    assert autolabel.map_species(_FOX_SQUIRREL, active)[0] is None


def test_autolabel_process_writes_and_reviews(tmp_path: Path) -> None:
    classes = species.training_classes(("1",))
    deer, person = classes.index("mule_deer"), classes.index("person")
    preds = [
        {"filepath": "deer.jpg", "prediction": _MULE_DEER, "prediction_score": 0.9,
         "detections": [{"category": "1", "conf": 0.9, "bbox": [0.5, 0.3, 0.2, 0.4]}]},
        {"filepath": "person.jpg", "prediction": _HUMAN, "prediction_score": 0.95,
         "detections": [{"category": "2", "conf": 0.8, "bbox": [0.1, 0.1, 0.2, 0.2]}]},
        {"filepath": "squirrel.jpg", "prediction": _FOX_SQUIRREL, "prediction_score": 0.9,
         "detections": [{"category": "1", "conf": 0.9, "bbox": [0.4, 0.4, 0.1, 0.1]}]},
        {"filepath": "lowconf.jpg", "prediction": _MULE_DEER, "prediction_score": 0.3,
         "detections": [{"category": "1", "conf": 0.9, "bbox": [0.5, 0.5, 0.1, 0.1]}]},
        {"filepath": "vehicle.jpg", "prediction": "", "prediction_score": 0.0,
         "detections": [{"category": "3", "conf": 0.9, "bbox": [0.2, 0.2, 0.3, 0.3]}]},
    ]
    review = autolabel.process(preds, classes, det_thresh=0.2, class_thresh=0.5, labels_out=tmp_path)

    # deer box: corner->center xc=0.6 yc=0.5 w=0.2 h=0.4
    assert (tmp_path / "deer.txt").read_text().strip() == f"{deer} 0.600000 0.500000 0.200000 0.400000"
    assert (tmp_path / "person.txt").read_text().strip().startswith(f"{person} ")
    # unmapped species + low-conf species -> empty label, flagged for review
    assert (tmp_path / "squirrel.txt").read_text().strip() == ""
    assert (tmp_path / "lowconf.txt").read_text().strip() == ""
    statuses = {r["status"] for r in review}
    assert "review:unmapped" in statuses and "review:low_conf" in statuses


# --------------------------------------------------------------------------- #
# convert_lila (COCO-Camera-Traps -> YOLO)
# --------------------------------------------------------------------------- #
def test_bbox_conversions() -> None:
    # absolute pixels -> normalized center
    xc, yc, w, h = convert_lila.abs_bbox_to_yolo([100, 50, 200, 150], 1024, 768)
    assert round(xc, 5) == round(200 / 1024, 5) and round(w, 5) == round(200 / 1024, 5)
    # already-normalized corner -> center (width/height pass through)
    xc, yc, w, h = convert_lila.norm_bbox_to_yolo([0.0, 0.2762, 0.1539, 0.2825])
    assert round(xc, 5) == 0.07695 and w == 0.1539


def test_category_mapping_and_empty() -> None:
    active = set(species.training_classes(("1", "2")))
    assert convert_lila._map_category("Cougar", active) == "mountain_lion"
    assert convert_lila._map_category("American Black Bear", active) == "black_bear"
    assert convert_lila._map_category("empty", active) == convert_lila._EMPTY
    assert convert_lila._map_category("Domestic Dog", active) is None  # unmapped -> skip


def test_convert_lila_native_boxes_and_skips(tmp_path: Path) -> None:
    imgs = tmp_path / "imgs"
    (imgs / "S1").mkdir(parents=True)
    for f in ("f1.jpg", "f2.jpg", "f3.jpg"):
        (imgs / "S1" / f).write_bytes(b"x")
    cct = {
        "categories": [{"id": 0, "name": "empty"}, {"id": 1, "name": "American Black Bear"},
                       {"id": 2, "name": "Domestic Dog"}],
        "images": [
            {"id": "a", "file_name": "S1/f1.jpg", "width": 100, "height": 100, "seq_id": "S1"},
            {"id": "b", "file_name": "S1/f2.jpg", "width": 100, "height": 100, "seq_id": "S1"},
            {"id": "c", "file_name": "S1/f3.jpg", "width": 100, "height": 100, "seq_id": "S1"},
        ],
        "annotations": [
            {"id": "1", "image_id": "a", "category_id": 1, "bbox": [10, 20, 30, 40]},  # bear box
            {"id": "2", "image_id": "b", "category_id": 0},                            # empty
            {"id": "3", "image_id": "c", "category_id": 2, "bbox": [0, 0, 10, 10]},    # dog -> skip
        ],
    }
    cct_path = tmp_path / "cct.json"
    cct_path.write_text(__import__("json").dumps(cct))
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_TRAINING / "convert_lila.py"), "--images", str(imgs),
         "--cct", str(cct_path), "--out", str(out), "--tiers", "1"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    bear_cls = species.training_classes(("1",)).index("black_bear")
    # bear image labeled: xc=(10+15)/100=.25 yc=(20+20)/100=.40 w=.30 h=.40
    assert (out / "S1" / "S1__f1.txt").read_text().strip() == f"{bear_cls} 0.250000 0.400000 0.300000 0.400000"
    # empty image -> background (empty label)
    assert (out / "S1" / "S1__f2.txt").read_text().strip() == ""
    # dog image -> skipped entirely (no false background)
    assert not (out / "S1" / "S1__f3.txt").exists()
    # a class manifest was written so downstream tools reuse these ids
    assert species.read_manifest(out) == species.training_classes(("1",))


def _prep(pool: Path, out: Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(_TRAINING / "prepare_dataset.py"),
         "--pool", str(pool), "--out", str(out), *extra],
        capture_output=True, text=True,
    )


def test_manifest_keeps_class_ids_consistent(tmp_path: Path) -> None:
    # A pool labeled against tiers "12" must keep those ids even if prepare_dataset
    # is invoked with a different --tiers (the manifest wins, preventing mislabels).
    pool = tmp_path / "pool"
    pool.mkdir()
    twelve = species.training_classes(("1", "2"))
    species.write_manifest(pool, twelve)
    bighorn = twelve.index("bighorn_sheep")  # only exists in tier 2
    for i in range(8):  # multiple events so the split is non-degenerate
        ev = pool / f"evt_{i}"
        ev.mkdir()
        (ev / "a.jpg").write_bytes(b"x")
        (ev / "a.txt").write_text(f"{bighorn} 0.5 0.5 0.2 0.2\n")

    out = tmp_path / "ds"
    r = _prep(pool, out, "--tiers", "1", "--val-frac", "0.34")  # deliberately wrong tier
    assert r.returncode == 0, r.stderr
    import yaml

    names = yaml.safe_load((out / "data.yaml").read_text())["names"]
    assert names[bighorn] == "bighorn_sheep"  # manifest honored, not tiers-1


def test_prepare_flat_pool_degenerate_split_errors(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    for i in range(4):  # flat pool -> one group -> empty split
        (pool / f"f{i}.jpg").write_bytes(b"x")
    r = _prep(pool, tmp_path / "ds")
    assert r.returncode == 1 and "degenerate split" in r.stderr


def test_prepare_conflicting_manifests_error(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    (pool / "ds1").mkdir(parents=True)
    (pool / "ds2").mkdir(parents=True)
    species.write_manifest(pool / "ds1", species.training_classes(("1",)))
    species.write_manifest(pool / "ds2", species.training_classes(("1", "2")))
    (pool / "ds1" / "x.jpg").write_bytes(b"x")
    r = _prep(pool, tmp_path / "ds")
    assert r.returncode == 1 and "conflicting class manifests" in r.stderr


def test_prepare_malformed_label_dropped_and_flagged(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    for i in range(8):  # enough events for a non-degenerate split
        ev = pool / f"e{i}"
        ev.mkdir(parents=True)
        (ev / "a.jpg").write_bytes(b"x")
        (ev / "a.txt").write_text("# a stray comment line\n0 0.5 0.5 0.2 0.2\n")
    r = _prep(pool, tmp_path / "ds", "--val-frac", "0.34")
    assert r.returncode == 1 and "malformed" in r.stderr  # surfaced, not crashed


def test_prepare_group_regex_nonmatch_fails(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "nomatch.jpg").write_bytes(b"x")
    r = _prep(pool, tmp_path / "ds", "--group-regex", r"(cam\d+_\d{8})")
    # The ValueError surfaces on stderr (non-zero exit) rather than leaking a burst.
    assert r.returncode != 0 and "did not match" in r.stderr


# --------------------------------------------------------------------------- #
# autolabel review-flag fixes
# --------------------------------------------------------------------------- #
def test_person_box_does_not_suppress_animal_review(tmp_path: Path) -> None:
    classes = species.training_classes(("1",))
    # Frame with a labeled person AND an unmapped animal: the animal must still be flagged.
    preds = [{"filepath": "mix.jpg", "prediction": _FOX_SQUIRREL, "prediction_score": 0.9,
              "detections": [{"category": "2", "conf": 0.9, "bbox": [0.1, 0.1, 0.1, 0.1]},
                             {"category": "1", "conf": 0.9, "bbox": [0.6, 0.6, 0.2, 0.2]}]}]
    review = autolabel.process(preds, classes, 0.2, 0.5, tmp_path)
    assert any(r["status"] == "review:unmapped" for r in review)


def test_confident_species_no_box_flagged(tmp_path: Path) -> None:
    classes = species.training_classes(("1",))
    # Classifier confident (mule deer) but the detector produced no box -> must be
    # flagged, not written as a silent background negative.
    preds = [{"filepath": "nobox.jpg", "prediction": _MULE_DEER, "prediction_score": 0.95,
              "detections": []}]
    review = autolabel.process(preds, classes, 0.2, 0.5, tmp_path)
    assert (tmp_path / "nobox.txt").read_text().strip() == ""
    assert any(r["status"] == "review:no_box" for r in review)
