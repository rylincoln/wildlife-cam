#!/usr/bin/env python3
"""Fine-tune a YOLO detector on SW-Colorado wildlife and export to Core ML.

Two-stage transfer learning from COCO-pretrained weights (the Ultralytics
recommended recipe for a modest, off-domain dataset):

* **Stage 1** — ``freeze=10``: freeze the backbone, adapt only neck + head so the
  pretrained features aren't washed out by a small dataset.
* **Stage 2** — unfreeze everything at a low LR. Because ``optimizer=auto``
  *ignores* ``lr0``, stage 2 sets ``optimizer=AdamW`` explicitly so the low LR
  actually takes effect.

Then exports a Core ML ``.mlpackage`` that runs on the Apple Neural Engine — 3×
faster than CPU and off the GPU shaders the media server wants. Point
``detection.model_path`` at it and it drops into ``detect.py`` with no code change
(``model.names`` become your species; make ``config.yaml`` ``animal_classes``
match).

Example::

    python training/train.py --data ~/wildlife/dataset/data.yaml --model yolo11s.pt

Requires the training extra:  pip install -e '.[train]'   (ultralytics + coremltools)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import species  # noqa: E402  (local module)


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is not installed. Install the training extra:\n"
            "    pip install -e '.[train]'",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return YOLO


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Path to data.yaml (from prepare_dataset.py).")
    p.add_argument("--model", default="yolo11s.pt",
                   help="COCO-pretrained base .pt (yolo11s.pt, yolov8s.pt, yolo26s.pt). Auto-downloads.")
    p.add_argument("--imgsz", type=int, default=640, help="Train/infer size; use 1280 for small/distant animals.")
    p.add_argument("--batch", type=int, default=16, help="Batch size (-1 = auto, CUDA only).")
    p.add_argument("--device", default="mps", help="mps (Apple GPU) | cpu | 0 (CUDA).")
    p.add_argument("--freeze", type=int, default=10, help="Backbone layers to freeze in stage 1.")
    p.add_argument("--epochs-stage1", type=int, default=20)
    p.add_argument("--epochs-stage2", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.001, help="Stage-2 lr0 (with optimizer=AdamW).")
    p.add_argument("--patience", type=int, default=15, help="Early-stopping patience.")
    p.add_argument("--mosaic", type=float, default=0.5, help="Mosaic aug prob (lower for small datasets).")
    p.add_argument("--fraction", type=float, default=1.0, help="Fraction of the train set to use (quick runs).")
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers (MPS may force 0).")
    p.add_argument("--no-val", action="store_true",
                   help="Skip per-epoch validation (big MPS speedup); one final val still runs.")
    p.add_argument("--cache", choices=["none", "ram", "disk"], default="none",
                   help="Cache decoded images (disk/ram) so epochs after the first skip JPEG "
                        "decode -- the big win when MPS training is data-loading bound.")
    p.add_argument("--hours", type=float, default=None,
                   help="Max training time (hours); overrides epochs. Ultralytics trains as many "
                        "epochs as fit and picks the val-best checkpoint. Best for a bounded overnight run.")
    p.add_argument("--single-stage", action="store_true", help="Skip the frozen stage; one unfrozen run.")
    p.add_argument("--project", default="runs/wildlife", help="Output root for runs.")
    p.add_argument("--name", default="sw_co", help="Run name.")
    p.add_argument("--no-export", action="store_true", help="Skip the Core ML export step.")
    p.add_argument("--quantize", choices=["none", "8", "16"], default="8",
                   help="Core ML quantization: 8=INT8 (smallest/fastest), 16=FP16, none=FP32.")
    args = p.parse_args()

    data = Path(args.data).expanduser()
    if not data.is_file():
        print(f"data.yaml not found: {data}\nRun training/prepare_dataset.py first.", file=sys.stderr)
        return 1

    YOLO = _load_yolo()
    model = YOLO(args.model)

    common = dict(
        data=str(data), imgsz=args.imgsz, batch=args.batch, device=args.device,
        patience=args.patience, mosaic=args.mosaic, project=args.project, fraction=args.fraction,
        workers=args.workers, val=not args.no_val,
        cache={"none": False, "ram": True, "disk": "disk"}[args.cache],
    )
    if args.hours:
        common["time"] = args.hours  # Ultralytics: time-bound overrides epochs

    if not args.single_stage and args.epochs_stage1 > 0:
        print(f"[stage 1] freeze={args.freeze}, {args.epochs_stage1} epochs")
        r1 = model.train(**common, epochs=args.epochs_stage1, freeze=args.freeze, name=f"{args.name}-s1")
        # Stage 2 is a *fresh* run started from stage-1's best weights (the
        # documented two-stage recipe -- not resume=, which can't change freeze/lr).
        s1_best = Path(r1.save_dir) / "weights" / "best.pt"
        if s1_best.is_file():
            model = YOLO(str(s1_best))

    stage2_epochs = args.epochs_stage2 if not args.single_stage else (args.epochs_stage1 + args.epochs_stage2)
    print(f"[stage 2] unfrozen, optimizer=AdamW lr0={args.lr}, {stage2_epochs} epochs")
    results = model.train(
        **common, epochs=stage2_epochs, optimizer="AdamW", lr0=args.lr, name=f"{args.name}-s2"
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    print("Validation summary:")
    metrics = model.val(data=str(data), imgsz=args.imgsz, device=args.device)
    try:
        print(f"  mAP50-95={metrics.box.map:.3f}  mAP50={metrics.box.map50:.3f}")
    except Exception:  # noqa: BLE001 - metrics surface varies by version
        pass

    if args.no_export:
        print("Skipping Core ML export (--no-export). Point detection.model_path at best.pt for a quick test.")
        return 0

    # End-to-end (NMS-free) models need no embedded NMS. Detect this from the
    # loaded model (a renamed YOLO26 checkpoint won't have 'yolo26' in its path).
    is_end2end = bool(getattr(getattr(model, "model", None), "end2end", False)) or (
        "yolo26" in Path(args.model).stem.lower()
    )
    # Deploy hint from the TRAINED model's own class names (not hardcoded tiers),
    # so detection.animal_classes always matches what the model actually predicts.
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    animal_classes = species.config_animal_classes(names)

    export_kwargs: dict = dict(format="coreml", imgsz=args.imgsz)
    if not is_end2end:
        export_kwargs["nms"] = True
    if args.quantize != "none":
        export_kwargs["quantize"] = int(args.quantize)
    print(f"\n[export] Core ML: {export_kwargs}")
    try:
        deploy_path = model.export(**export_kwargs)
        print(f"\nExported Core ML model: {deploy_path}")
    except Exception as exc:  # noqa: BLE001 - export is best-effort; the .pt still deploys
        deploy_path = best
        print(f"\nCore ML export failed ({type(exc).__name__}: {exc}).")
        print("This env's torch is likely too new for coremltools; deploy the .pt on MPS "
              "instead (detect.py loads it directly via device=mps -- no code change).")

    print(f"\nDeploy: set detection.model_path to:\n  {deploy_path}")
    print("and detection.animal_classes to the trained model's classes:\n")
    print(animal_classes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
