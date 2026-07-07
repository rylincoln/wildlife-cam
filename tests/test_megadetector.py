"""Unit tests for the MegaDetector second-pass decision logic (hardware-free).

These exercise wildlife.megadetector.second_pass_decision / _iou with synthetic
Detections and a real MegadetectorConfig — no torch/ultralytics, no model load.
"""

from __future__ import annotations

from wildlife.config import MegadetectorConfig
from wildlife.megadetector import _iou, second_pass_decision
from wildlife.models import Detection


def _det(label, conf, box=(100.0, 100.0, 200.0, 500.0), area=0.05) -> Detection:
    return Detection(label=label, confidence=conf, box_xyxy=box, box_area_frac=area)


def _cfg(**over) -> MegadetectorConfig:
    base = dict(person_override=True, rescue_misses=True, suppress_false_positives=False)
    base.update(over)
    return MegadetectorConfig(**base)


# --------------------------------------------------------------------------- #
# _iou
# --------------------------------------------------------------------------- #
def test_iou_identical_and_disjoint():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.0 < _iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1.0


# --------------------------------------------------------------------------- #
# person override
# --------------------------------------------------------------------------- #
def test_person_override_relabels_all_positives():
    bird = _det("bird", 0.7)
    md = [_det("person", 0.92)]  # same box -> IoU 1.0
    _bd, _bf, positives, reason = second_pass_decision(
        md_dets=md, best_det=None, best_frame=None,
        positives=[("f0", bird), ("f1", bird)], representative_frame="rep",
        save_best_only=False, cfg=_cfg(),
    )
    assert [d.label for _f, d in positives] == ["person", "person"]
    assert positives[0][1].confidence == 0.92  # takes the MD person confidence
    assert positives[0][1].box_xyxy == bird.box_xyxy  # keeps the species box (frame-aligned)
    assert "relabeled 2 -> person" in reason


def test_person_override_save_best_only():
    bird = _det("bird", 0.6)
    md = [_det("person", 0.88)]
    best_det, best_frame, _pos, reason = second_pass_decision(
        md_dets=md, best_det=bird, best_frame="bf", positives=[],
        representative_frame="rep", save_best_only=True, cfg=_cfg(),
    )
    assert best_det.label == "person" and best_frame == "bf"
    assert "-> person" in reason


def test_no_override_when_iou_below_match():
    bird = _det("bird", 0.7, box=(100, 100, 200, 500))
    md = [_det("person", 0.95, box=(1000, 1000, 1100, 1400))]  # far away -> IoU 0
    _bd, _bf, positives, reason = second_pass_decision(
        md_dets=md, best_det=None, best_frame=None,
        positives=[("f", bird)], representative_frame="rep",
        save_best_only=False, cfg=_cfg(),
    )
    assert positives[0][1].label == "bird" and reason == ""


def test_override_ignores_vehicle_even_when_person_override_on():
    bird = _det("bird", 0.7)
    md = [_det("vehicle", 0.99)]  # vehicle not in default classes AND never a person
    _bd, _bf, positives, reason = second_pass_decision(
        md_dets=md, best_det=None, best_frame=None,
        positives=[("f", bird)], representative_frame="rep",
        save_best_only=False, cfg=_cfg(classes=["person", "animal", "vehicle"]),
    )
    assert positives[0][1].label == "bird" and reason == ""


# --------------------------------------------------------------------------- #
# recall net (rescue)
# --------------------------------------------------------------------------- #
def test_rescue_when_species_kept_nothing_all_positives():
    md = [_det("animal", 0.55, area=0.08)]
    _bd, _bf, positives, reason = second_pass_decision(
        md_dets=md, best_det=None, best_frame=None, positives=[],
        representative_frame="rep", save_best_only=False, cfg=_cfg(),
    )
    assert len(positives) == 1
    frame, det = positives[0]
    assert frame == "rep" and det.label == "animal" and det.confidence == 0.55
    assert "rescued a missed animal" in reason


def test_rescue_save_best_only_sets_best():
    md = [_det("animal", 0.6, area=0.08)]
    best_det, best_frame, _pos, reason = second_pass_decision(
        md_dets=md, best_det=None, best_frame=None, positives=[],
        representative_frame="rep", save_best_only=True, cfg=_cfg(),
    )
    assert best_det is not None and best_det.label == "animal" and best_frame == "rep"


def test_rescue_respects_min_area_frac():
    md = [_det("animal", 0.9, area=0.001)]  # a speck
    _bd, _bf, positives, reason = second_pass_decision(
        md_dets=md, best_det=None, best_frame=None, positives=[],
        representative_frame="rep", save_best_only=False, cfg=_cfg(),
        min_area_frac=0.01,
    )
    assert positives == [] and reason == ""


def test_rescue_excludes_classes_not_in_list():
    md = [_det("vehicle", 0.9, area=0.2)]
    _bd, _bf, positives, _reason = second_pass_decision(
        md_dets=md, best_det=None, best_frame=None, positives=[],
        representative_frame="rep", save_best_only=False,
        cfg=_cfg(classes=["person", "animal"]),  # vehicle excluded
    )
    assert positives == []


def test_rescue_disabled_leaves_empty():
    md = [_det("animal", 0.9, area=0.2)]
    _bd, _bf, positives, reason = second_pass_decision(
        md_dets=md, best_det=None, best_frame=None, positives=[],
        representative_frame="rep", save_best_only=False, cfg=_cfg(rescue_misses=False),
    )
    assert positives == [] and reason == ""


# --------------------------------------------------------------------------- #
# false-positive suppression
# --------------------------------------------------------------------------- #
def test_suppress_drops_unsupported_detection_when_enabled():
    bird = _det("bird", 0.7)
    _bd, _bf, positives, reason = second_pass_decision(
        md_dets=[], best_det=None, best_frame=None,  # MD sees nothing
        positives=[("f", bird)], representative_frame="rep",
        save_best_only=False, cfg=_cfg(suppress_false_positives=True),
    )
    assert positives == [] and "suppressed 1 false positive" in reason


def test_suppress_off_by_default_keeps_detection():
    bird = _det("bird", 0.7)
    _bd, _bf, positives, reason = second_pass_decision(
        md_dets=[], best_det=None, best_frame=None,
        positives=[("f", bird)], representative_frame="rep",
        save_best_only=False, cfg=_cfg(),  # suppress False
    )
    assert positives[0][1].label == "bird" and reason == ""


def test_suppress_save_best_only_nulls_best():
    bird = _det("bird", 0.7)
    best_det, best_frame, _pos, reason = second_pass_decision(
        md_dets=[], best_det=bird, best_frame="bf", positives=[],
        representative_frame="rep", save_best_only=True,
        cfg=_cfg(suppress_false_positives=True),
    )
    assert best_det is None and best_frame is None and "suppressed" in reason


def test_kept_detection_confirmed_by_md_survives_suppression():
    bird = _det("bird", 0.7, box=(100, 100, 200, 500))
    md = [_det("animal", 0.6, box=(105, 105, 205, 505))]  # overlaps -> confirmed real
    _bd, _bf, positives, _reason = second_pass_decision(
        md_dets=md, best_det=None, best_frame=None,
        positives=[("f", bird)], representative_frame="rep",
        save_best_only=False, cfg=_cfg(suppress_false_positives=True),
    )
    assert positives[0][1].label == "bird"  # kept: MD confirms something is there
