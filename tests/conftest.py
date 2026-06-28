"""Shared pytest fixtures and helpers for the wildlife test suite.

Everything here is hardware-free: tests build :class:`Detection` instances and a
sample :class:`DetectionConfig` directly so the gate logic can be exercised with
no torch/cv2/ultralytics imports.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from wildlife.config import DetectionConfig
from wildlife.models import Detection

# Labels treated as "wildlife" in the sample config below.
SAMPLE_ANIMAL_CLASSES = ["bird", "cat", "dog", "horse", "bear"]


def _make_detection(label: str, conf: float, area_frac: float = 0.2) -> Detection:
    """Build a :class:`Detection` with a dummy box for gate-logic tests.

    The actual box geometry is irrelevant to the gate, which only inspects
    ``label``, ``confidence`` and ``box_area_frac`` -- so we supply a fixed,
    plausible ``box_xyxy`` and let ``area_frac`` be controlled independently.
    """
    return Detection(
        label=label,
        confidence=conf,
        box_xyxy=(0.0, 0.0, 100.0, 100.0),
        box_area_frac=area_frac,
    )


@pytest.fixture
def make_detection() -> Callable[..., Detection]:
    """Return the ``make_detection(label, conf, area_frac=0.2)`` factory."""
    return _make_detection


@pytest.fixture
def det_config() -> DetectionConfig:
    """A sample :class:`DetectionConfig` built directly via the pydantic model."""
    return DetectionConfig(
        model_path="models/yolov8s.pt",
        device="cpu",
        animal_classes=list(SAMPLE_ANIMAL_CLASSES),
        confidence_threshold=0.55,
        min_box_area_frac=0.01,
        save_best_only=True,
    )
