# Training a local wildlife detector

Stock `yolov8s.pt` only knows 80 COCO classes — of local wildlife, just **bear**
and generic **bird**. To detect the deer, elk, foxes, big cats, turkeys, etc. of
**your** region you need a model *trained* on them. This directory is the offline
toolchain for that. Define your taxonomy in `species.py` — it ships a
**southwest-Colorado set as a worked example**, and the commands below use those
species and datasets; edit it and swap in datasets/queries for your own area.

Nothing here runs in the live app; you produce a `best.pt`, copy it into `models/`,
and point `detection.model_path` at it — no code change, because `detect.py` reads
`model.names` and `gate.py` filters by `detection.animal_classes` (which must
match). It runs on the Mac GPU via **MPS**. (Core ML export to a `.mlpackage` for
the Neural Engine is the intended fast path but currently fails — the installed
torch is too new for `coremltools` — so `train.py` falls back to the `.pt`.)

## The pipeline

```
 LILA datasets ──▶ download_lila.py (target subset) ─▶ convert_lila.py ──┐
                                                                          │
 iNaturalist ────▶ download_inat.py (CC photos) ─▶ label_boxes.py ───────┤─▶ pool of
                                        (MegaDetector v6 boxes)           │   images + .txt
 your captures ──────────▶ autolabel.py (SpeciesNet) ────────────────────┘
                     │
                     ▼
            prepare_dataset.py  (burst-safe train/val split + data.yaml)
                     │
                     ▼
                 train.py   (2-stage fine-tune → best.pt)
                     │
                     ▼
      copy best.pt → models/ ; set model_path + animal_classes ; restart worker  (runs on MPS)
```

`species.py` is the **single source of truth** for class names, so `data.yaml`
and `config.yaml` can't drift. Class ids are the list index — keep the order
stable once you start labeling.

## Install

```bash
uv pip install -e '.[train]'      # ultralytics (+ coremltools; the venv is uv-managed, no pip)
uv pip install -e '.[autolabel]'  # speciesnet (auto-labeling; bundles MegaDetector)
```

`label_boxes.py` (the iNaturalist path) needs **no extra** — it runs MegaDetector v6
on the already-installed `ultralytics`; the weights download to `weights/` on first
use. `download_inat.py` uses only `requests`/`urllib`.

## Phase 0 — before you have your own captures (bootstrap on public data)

The public archives are huge (Caltech 105 GB, Idaho ~1.45 TB) and mostly empty
frames, so `download_lila.py` fetches only a **target-species subset** from each
dataset's per-image URL. Good starting sets:

| Dataset | Adds | Size (full → subset) |
|---|---|---|
| **ENA24** | black bear, deer, turkey, coyote, fox, skunk, raccoon, bobcat, squirrel | 3.6 GB (small — grab the whole zip) |
| **Caltech Camera Traps** | **mountain_lion**, bobcat, coyote, deer, raccoon, skunk, rabbit, bird | 105 GB → ~14 GB (38k boxed target imgs) |
| **Idaho Camera Traps** | **elk, moose, pronghorn, bighorn**, more cougar/bear/coyote | 1.45 TB → ~6 GB (cap 4k/class, MD-boxed) |

Idaho/Caltech-image-level boxes come from the
[LILA MegaDetector-results JSON](https://lila.science/megadetector-results-for-camera-trap-datasets/);
Idaho is **non-commercial**. For species the camera-trap sets lack (marmot,
snowshoe hare, porcupine, bighorn, …), fill from **iNaturalist** with
`download_inat.py` → `label_boxes.py` (see below). It queries the iNat API
directly for research-grade, CC-licensed, region-filtered photos — **not** GBIF
(for these species ~95–99% of Colorado GBIF media is just re-ingested iNaturalist,
and the rest is museum-specimen photos that hurt a trail-cam detector). Since iNat
photos aren't boxed, `label_boxes.py` runs MegaDetector v6 to box them and forces
the known species label.

```bash
# Fill an empty class from iNaturalist (Colorado; drop --place-id for full range):
python training/download_inat.py --class yellow_bellied_marmot --place-id 34 \
    --per-species 1200 --out ~/wildlife/data/inat/yellow_bellied_marmot
python training/label_boxes.py --images ~/wildlife/data/inat/yellow_bellied_marmot \
    --assume-species yellow_bellied_marmot --out ~/wildlife/labeled/inat_yellow_bellied_marmot
```

```bash
# ENA24 — small; just download the zip + json, then convert (native boxes):
python training/convert_lila.py --images ~/wildlife/data/ena24/images \
    --cct ~/wildlife/data/ena24/ena24.json --out ~/wildlife/labeled/ena24 --tiers 12

# Caltech — download the boxed target subset, then convert:
python training/download_lila.py --metadata ~/wildlife/data/caltech/caltech_bboxes.json \
    --base-url https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images \
    --out ~/wildlife/data/caltech/images --tiers 12 --boxed-only
python training/convert_lila.py --images ~/wildlife/data/caltech/images \
    --cct ~/wildlife/data/caltech/caltech_bboxes.json --out ~/wildlife/labeled/caltech --tiers 12

# Idaho — no native boxes: filter by MegaDetector-results, cap per class:
python training/download_lila.py --metadata ~/wildlife/data/idaho/idaho-camera-traps.json \
    --base-url https://storage.googleapis.com/public-datasets-lila/idaho-camera-traps/public \
    --md-results ~/wildlife/data/idaho/idaho-camera-traps_mdv5a.0.0_results.json \
    --out ~/wildlife/data/idaho/images --tiers 12 --max-per-class 4000
python training/convert_lila.py --images ~/wildlife/data/idaho/images \
    --cct ~/wildlife/data/idaho/idaho-camera-traps.json \
    --md-results ~/wildlife/data/idaho/idaho-camera-traps_mdv5a.0.0_results.json \
    --out ~/wildlife/labeled/idaho --tiers 12

# Merge all pools into one burst-safe split (the class manifest keeps ids consistent):
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
cp runs/wildlife/sw_co-s2/weights/best.pt models/wildlife_sw_co.pt
```
`train.py` prints the exact `animal_classes` block for the model it just trained
(derived from the model's own `model.names`, so it always matches) — paste that
into `config.yaml`, set `detection.model_path: "models/wildlife_sw_co.pt"`,
and restart the worker (or use the admin UI → it validates + restarts). Any
trained class you omit from `animal_classes` is silently dropped by the gate, so
keep them in sync. The `.pt` runs on the Mac GPU via **MPS**. (Core ML export to a
`.mlpackage` on the Neural Engine — lower power, off the GPU shaders go2rtc wants —
is the intended target but currently fails on this torch/`coremltools`, so `.pt`
on MPS is the shipped deploy path; `train.py` prints this same hint when export fails.)

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
| `download_lila.py` | fetch a target-species subset of a LILA dataset (skips the huge full archive) |
| `convert_lila.py` | COCO-Camera-Traps + LILA MD-results JSON → YOLO label pool (`--min-box-rel-area` prunes specks) |
| `download_inat.py` | fetch CC-licensed, region-filtered research-grade photos from the iNaturalist API |
| `label_boxes.py` | box iNat photos with MegaDetector v6 and force the known species label → YOLO pool |
| `autolabel.py` | SpeciesNet `predictions.json` → YOLO labels + `review.csv` |
| `prepare_dataset.py` | burst-safe train/val split + `data.yaml` (`--max-per-class` for stratified balance) |
| `train.py` | 2-stage fine-tune → `best.pt` (deploy on MPS; Core ML export attempted, falls back to `.pt`) |
