# Training a SW-Colorado wildlife detector

Stock `yolov8s.pt` only knows 80 COCO classes — of local wildlife, just **bear**
and generic **bird**. To detect mule deer, elk, cougar, coyote, fox, turkey, etc.
you need a model *trained* on them. This directory is the offline toolchain for
that. Nothing here runs in the live app; you produce a Core ML model and point
`detection.model_path` at it (no code change — `detect.py` reads `model.names`,
and `gate.py` filters by `detection.animal_classes`, which must match).

## The pipeline

```
   public datasets ─┐
                     ├─▶ convert_lila.py ─┐
 your captures ─▶ autolabel.py (SpeciesNet) ─┴─▶ pool of images + YOLO .txt
                     │
                     ▼
            prepare_dataset.py  (burst-safe train/val split + data.yaml)
                     │
                     ▼
                 train.py   (2-stage fine-tune → best.pt → Core ML .mlpackage)
                     │
                     ▼
        copy .mlpackage → models/ ; set model_path + animal_classes ; restart worker
```

`species.py` is the **single source of truth** for class names, so `data.yaml`
and `config.yaml` can't drift. Class ids are the list index — keep the order
stable once you start labeling.

## Install

```bash
pip install -e '.[train]'      # ultralytics + coremltools (fine-tune + export)
pip install -e '.[autolabel]'  # speciesnet (auto-labeling; bundles MegaDetector)
```

## Phase 0 — before you have your own captures (bootstrap on public data)

The best directly-boxed, commercially-usable sets for our carnivores/ungulates:

| Dataset | Species | Get it |
|---|---|---|
| **Caltech Camera Traps** | cougar, bobcat, coyote, fox, deer, raccoon, skunk | [lila.science/datasets/caltech-camera-traps](https://lila.science/datasets/caltech-camera-traps) (`cct_images.tar.gz`, `caltech_bboxes_20200316.json`) |
| **ENA24** | black bear, deer, turkey, coyote, fox, skunk | [lila.science/datasets/ena24detection](https://lila.science/datasets/ena24detection) (100% boxed, ~3.6 GB) |

For elk/moose/pronghorn/bighorn, add **Idaho Camera Traps** (image-level → use its
[LILA MegaDetector-results JSON](https://lila.science/megadetector-results-for-camera-trap-datasets/)
for boxes; note its **non-commercial** license) and fill rare species from
**iNaturalist via GBIF** (filter to CC0/CC-BY + region).

```bash
# ENA24 (native species boxes):
python training/convert_lila.py --images ~/data/ena24/images \
    --cct ~/data/ena24/ena24.json --out ~/wildlife/labeled/ena24 --tiers 12

# Caltech (image-level species + MegaDetector boxes):
python training/convert_lila.py --images ~/data/caltech/cct_images \
    --cct ~/data/caltech/caltech_bboxes_20200316.json \
    --md-results ~/data/caltech/caltech_camera_traps_mdv5a.0.0_results.json \
    --out ~/wildlife/labeled/caltech --tiers 12

# Split (burst-safe) into a trainable dataset:
python training/prepare_dataset.py --pool ~/wildlife/labeled --out ~/wildlife/dataset --tiers 12
python training/train.py --data ~/wildlife/dataset/data.yaml --model yolo11s.pt
```

Categories outside your active tiers are **skipped**, never turned into
background — so the model won't learn a real animal as "nothing".

## Phase 1 — once captures are flowing (the real loop)

Your gallery captures are the highest-value data, especially **night/IR** frames
(where public data is weak). Auto-label them, correct the flagged ones, retrain:

```bash
# Run SpeciesNet over captures (geofenced to Colorado) and write YOLO labels:
python training/autolabel.py --run --images ~/wildlife/captures --country USA --admin1 CO --tiers 12

# -> writes a .txt next to each image + review.csv of low-confidence / unmapped ones.
# Fix review.csv rows (SpeciesNet maps by scientific name; puma=mountain_lion,
# grey fox=gray_fox, northern raccoon=raccoon are handled for you).

python training/prepare_dataset.py --pool ~/wildlife/captures --out ~/wildlife/dataset --tiers 12 \
    --group-regex '([A-Za-z0-9]+_\d{8}T\d{6})'      # group by <camera>_<event-timestamp>
python training/train.py --data ~/wildlife/dataset/data.yaml --model yolo11s.pt
```

## Deploy

```bash
cp -r runs/wildlife/sw_co-s2/weights/best.mlpackage models/wildlife_sw_co.mlpackage
```
`train.py` prints the exact `animal_classes` block for the model it just trained
(derived from the model's own `model.names`, so it always matches) — paste that
into `config.yaml`, set `detection.model_path: "models/wildlife_sw_co.mlpackage"`,
and restart the worker (or use the admin UI → it validates + restarts). Any
trained class you omit from `animal_classes` is silently dropped by the gate, so
keep them in sync. The `.mlpackage` runs on the **Apple Neural Engine** — ~3× faster than
CPU and off the GPU shaders go2rtc wants.

## Tips (from the research)

- **Two-stage fine-tune** (default): stage 1 `freeze=10` adapts head/neck without
  wrecking pretrained features; stage 2 unfreezes at `lr0=0.001` (with
  `optimizer=AdamW`, because `optimizer=auto` silently ignores `lr0`).
- **~150–300 labeled images/species** gets a usable model; variety (day/night-IR,
  blur, partial animals, multiple cameras) matters more than raw count. Keep some
  empty frames as background negatives.
- **`--imgsz 1280`** if animals are small/distant in your framing.
- **A custom model replaces COCO** — that's why `person`/`black_bear`/`bird` are in
  the training set (see `SUPPORT` in `species.py`); they won't fall back to COCO.
- **YOLO26** (`--model yolo26s.pt`) is NMS-free; `train.py` drops the `nms` export
  flag automatically. **YOLO11** (`yolo11s.pt`) is the well-worn default here.
- Watch **per-class mAP50-95** and the confusion matrix — similar mammals get
  confused, and rare classes need oversampling.

## Scripts

| Script | Does |
|---|---|
| `species.py` | canonical class taxonomy (tiers + support); emits `data.yaml` names & `animal_classes` |
| `convert_lila.py` | COCO-Camera-Traps + LILA MD-results JSON → YOLO label pool |
| `autolabel.py` | SpeciesNet `predictions.json` → YOLO labels + `review.csv` |
| `prepare_dataset.py` | burst-safe train/val split + `data.yaml` |
| `train.py` | 2-stage fine-tune + Core ML export |
