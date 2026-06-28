#!/usr/bin/env python3
"""Run YOLO on one image and confirm inference is really on MPS (the M1 GPU).

Build-order step 4: validates the Ultralytics + Metal path and the COCO label
mapping. It builds a :class:`~wildlife.detect.Detector`, runs ``infer`` on a
still (an ``--image`` you provide, or a generated synthetic frame), prints every
detection, and -- crucially -- asserts the *actual* device in use is ``mps`` when
MPS was requested, so a silent CPU fallback fails the test loudly.

Run it directly from a checkout::

    python scripts/test_detect.py                         # synthetic frame, mps
    python scripts/test_detect.py --image some_animal.jpg
    python scripts/test_detect.py --device cpu            # don't require MPS

Exit status:
* ``0`` -- inference ran on the requested device.
* ``1`` -- ``--device mps`` was requested but MPS is unavailable (CPU fallback),
  or the image/model could not be loaded.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


def _bootstrap_src_path() -> None:
    """Allow running straight from a checkout without an editable install."""
    src = Path(__file__).resolve().parent.parent / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_src_path()


def _setup_logging(verbose: bool) -> None:
    """Surface detector logs (model load, device resolution, fallbacks)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def _resolve_model(model_arg: str) -> str:
    """Return a usable model path.

    Prefer the path as given; if it doesn't exist, try the same path under the
    repo root (so ``models/yolov8s.pt`` resolves when run from any cwd). If still
    not found, return the original string so Ultralytics can auto-download a
    known weight name (e.g. ``yolov8s.pt``).
    """
    candidate = Path(model_arg)
    if candidate.is_file():
        return str(candidate)
    repo_root = Path(__file__).resolve().parent.parent
    alt = repo_root / model_arg
    if alt.is_file():
        return str(alt)
    print(
        f"NOTE: model {model_arg!r} not found locally; "
        "Ultralytics will try to download it by name."
    )
    return model_arg


def _load_frame(image_path: str) -> np.ndarray:
    """Load a still image as a BGR numpy frame (OpenCV channel order)."""
    import cv2  # lazy: heavy import only needed when an image is supplied

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path!r}")
    return frame


def _synthetic_frame() -> np.ndarray:
    """Generate a deterministic 640x640 BGR noise frame as a fallback input.

    A synthetic frame exercises the full load->infer->label path without needing
    a sample on disk. It won't contain real animals, so zero detections is the
    expected and acceptable outcome here -- the point is to confirm the device.
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)


def main() -> int:
    """Build the detector, run one inference, print results, confirm the device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", default=None, help="Path to a still image (BGR). Omit for a synthetic frame."
    )
    parser.add_argument(
        "--model",
        default="models/yolov8s.pt",
        help="Path to Ultralytics weights or a .mlpackage (default: models/yolov8s.pt).",
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default="mps",
        help="Requested inference device (default: mps).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level module logging."
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    # Report what torch thinks about MPS up front for a clear diagnostic.
    try:
        import torch

        mps_available = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001 - torch import shouldn't hard-crash the test
        mps_available = False
        print(f"WARNING: could not query torch MPS availability: {exc!r}")
    print(f"torch.backends.mps.is_available() = {mps_available}")

    # Defer the heavy import until after the lightweight argparse/logging setup.
    from wildlife.detect import Detector

    if args.image:
        frame = _load_frame(args.image)
        print(f"Loaded image {args.image!r} -> shape={frame.shape} dtype={frame.dtype}")
    else:
        frame = _synthetic_frame()
        print(
            f"No --image given; using synthetic frame shape={frame.shape} "
            "(expect zero detections)."
        )

    model_path = _resolve_model(args.model)
    detector = Detector(model_path=model_path, device=args.device)
    device_in_use = detector.device_in_use()
    print(f"Requested device={args.device!r}  ->  device_in_use={device_in_use!r}")

    detections = detector.infer(frame)
    print(f"\nDetections ({len(detections)}):")
    if not detections:
        print("  (none)")
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        print(
            f"  {det.label:<12} conf={det.confidence:.3f} "
            f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
            f"area_frac={det.box_area_frac:.4f}"
        )

    # Hard gate: a request for MPS must actually run on MPS.
    print()
    if args.device == "mps" and device_in_use != "mps":
        print(
            "RESULT: FAIL -- 'mps' was requested but inference fell back to "
            f"{device_in_use!r}. Check your Apple Silicon / PyTorch MPS build."
        )
        return 1

    assert device_in_use == args.device, (
        f"device mismatch: requested {args.device!r}, using {device_in_use!r}"
    )
    print(f"RESULT: PASS -- inference confirmed on {device_in_use!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
