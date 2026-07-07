# Models

YOLO weights for the detection pass live here. Weight files are **gitignored**
(`*.pt`, `*.mlpackage`, `*.onnx`) — they are downloaded or exported on the host,
never committed.

## Default (stock) model

Out of the box, `config.example.yaml` points `detection.model_path` at a stock
**`models/yolov8s.pt`** — no manual download needed: Ultralytics **auto-downloads**
the weights on first use (`YOLO("yolov8s.pt")`), caching them locally. `yolov8s`
("small") is a good accuracy/speed balance on an M1 GPU, but it only knows the 80
COCO labels — of local wildlife, just **bear** and generic **bird** (deer, elk, fox,
mountain lion, etc. aren't COCO classes). Runs on the Mac GPU via **MPS**.

## Your fine-tuned model

Once you fine-tune a detector on your local species with the
[`training/`](../training/README.md) toolchain, copy the resulting `best.pt` here
(e.g. `models/wildlife_<region>.pt`) and point `detection.model_path` at it, with
`detection.animal_classes` set to the trained class names (`train.py` prints the
exact block). Swapping models needs **no code change** — `detect.py` reads the class
names from the model itself and `gate.py` filters by `animal_classes`. Fine-tuned
weights are produced locally and copied in by hand; they aren't downloadable from
Ultralytics. These also run on **MPS**.

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
YOLO("models/your_model.pt").export(format="coreml")   # -> a .mlpackage
```

**This currently fails on this stack** — the installed torch (2.12) is too new for
`coremltools`, so `.export(format="coreml")` raises. `train.py` catches the failure
and falls back to deploying the `.pt`. Until the torch/`coremltools` versions
realign, the model runs as a **`.pt` on MPS** (which is what ships). The `[coreml]`
extra (`uv pip install -e ".[coreml]"`) is kept for when export works again.
