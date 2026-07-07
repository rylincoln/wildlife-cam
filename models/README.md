# Models

YOLO weights for the detection pass live here. Weight files are **gitignored**
(`*.pt`, `*.mlpackage`, `*.onnx`) — they are downloaded or exported on the host,
never committed.

## Deployed model

The deployed detector is **`models/wildlife_sw_co.pt`** — a fine-tuned **21-class
southwest-Colorado species** model (`yolo11s`, val mAP50 ≈ 0.81), referenced by
`detection.model_path` in `config.yaml`. It is **produced by the training toolchain**
(see [`training/README.md`](../training/README.md)) and copied here by hand — it is
*not* downloadable from Ultralytics. `detect.py` reads the class names from the model
itself; `config.yaml` `detection.animal_classes` is the 20-species allowlist that
survives the gate (`person` is trained but kept off it). It runs on the Mac GPU via **MPS**.

### Base checkpoints (for training only)

`yolov8s.pt` / `yolo11s.pt` are the stock COCO backbones the fine-tune *starts from*.
Ultralytics **auto-downloads** these on first use (`YOLO("yolo11s.pt")`), caching them
locally. On their own they only know the 80 COCO labels — of local wildlife, just
**bear** and generic **bird** (deer, elk, fox, mountain lion, etc. aren't COCO
classes) — which is exactly why the fine-tuned model above exists.

## Newer / alternative models

You can point `detection.model_path` at any Ultralytics-compatible checkpoint.
Newer options worth trying:

- **`yolo11s`** — the YOLO11 small model; improved accuracy/efficiency over v8.
- **`yolo26n`** — a YOLO26 nano model; lighter still, useful if you want to
  minimize GPU contention with the co-tenant media server.

Larger variants (`...m`, `...l`, `...x`) trade speed for accuracy if your host
has the headroom.

> Swapping models needs no code change — `detect.py` reads `model.names` and
> `gate.py` filters by `detection.animal_classes`. Retrain/extend the deployed
> model with the [`training/`](../training/README.md) toolchain.

## Core ML export (currently non-functional)

The intended fast path is exporting to Core ML so inference runs on the **Apple
Neural Engine** (lower power, off the GPU shaders the media server may want):

```python
from ultralytics import YOLO
YOLO("models/wildlife_sw_co.pt").export(format="coreml")   # -> a .mlpackage
```

**This currently fails on this stack** — the installed torch (2.12) is too new for
`coremltools`, so `.export(format="coreml")` raises. `train.py` catches the failure
and falls back to deploying the `.pt`. Until the torch/`coremltools` versions
realign, the model runs as a **`.pt` on MPS** (which is what ships). The `[coreml]`
extra (`uv pip install -e ".[coreml]"`) is kept for when export works again.
