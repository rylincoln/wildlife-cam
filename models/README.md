# Models

YOLO weights for the detection pass live here. Weight files are **gitignored**
(`*.pt`, `*.mlpackage`, `*.onnx`) — they are downloaded or exported on the host,
never committed.

## Default model

The default is **`models/yolov8s.pt`**, referenced by `detection.model_path` in
`config.yaml`. You do not need to download it manually: Ultralytics
**auto-downloads** the weights on first use when you load
`YOLO("models/yolov8s.pt")` (or `YOLO("yolov8s.pt")`), caching them locally.

`yolov8s` ("small") is a good balance of accuracy and speed on an M1 GPU. It uses
the stock 80-class COCO labels; the relevant animal classes (bird, cat, dog,
horse, sheep, cow, bear, elephant, zebra, giraffe) are configured under
`detection.animal_classes`.

## Newer / alternative models

You can point `detection.model_path` at any Ultralytics-compatible checkpoint.
Newer options worth trying:

- **`yolo11s`** — the YOLO11 small model; improved accuracy/efficiency over v8.
- **`yolo26n`** — a YOLO26 nano model; lighter still, useful if you want to
  minimize GPU contention with the co-tenant media server.

Larger variants (`...m`, `...l`, `...x`) trade speed for accuracy if your host
has the headroom.

> Note: deer, elk, and fox are **not** COCO classes. For local species, plan to
> fine-tune on your own gallery captures later (see spec section 10).

## Core ML export (optional, for the Apple Neural Engine)

For lower power and better co-tenancy, you can export the model to Core ML so
inference can run on the **ANE** instead of the GPU shaders the media server may
want:

```python
from ultralytics import YOLO

YOLO("models/yolov8s.pt").export(format="coreml")
# -> produces models/yolov8s.mlpackage
```

Install the export toolchain with the `coreml` extra:

```bash
uv pip install -e ".[coreml]"   # coremltools
```

Then set `detection.model_path` to the resulting `.mlpackage`. The default path
stays on **MPS** for simplicity; reach for Core ML only if you observe GPU
contention with the media server.
