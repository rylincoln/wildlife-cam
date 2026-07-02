# Wildlife Detection System — Build Specification (macOS / Apple Silicon)

A fully-local, motion-triggered wildlife detection system that pulls frames from PoE
cameras, runs object detection on an Apple Silicon Mac (GPU/MPS), and saves only frames
containing a detected animal above a confidence threshold. Includes a small local web
gallery for browsing captures. No cloud, no subscriptions.

> **Audience:** This document is an implementation brief for Claude CLI. Build it as a
> GitHub repo in a local VS Code environment on the Mac mini itself (no cross-device
> deployment — you develop and run on the same machine). Treat each numbered component
> as an incremental, testable unit.

---

## 1. Goals & Non-Goals

### Goals

- Subscribe to camera-side detection events; keep the host near-idle until an event fires.
- On event: open the camera's RTSP stream, grab a short burst of frames, run them
  through a YOLO model on the Mac's GPU (MPS), and save **only** frames where an _animal_
  class is detected at or above a configurable confidence threshold.
- Persist each capture to local disk with a row in SQLite.
- Serve a small local web gallery (thumbnails, filter by camera/date/class/confidence).
- Run reliably as a background `launchd` service, survive reboots, coexist politely with
  the media server already running on this machine.

### Non-Goals (v1)

- No cloud upload, no push notifications (gallery only — per design decision).
- No species-level fine-tuning yet (stock COCO model first; fine-tune later).
- No live video recording — we save still frames only.

---

## 2. Host & Hardware (as deployed)

| Component | Detail                                                                                                |
| --------- | ----------------------------------------------------------------------------------------------------- |
| Compute   | **Apple M1 Mac mini, 8GB** — already running 24/7 as a media-serving appliance                        |
| Inference | M1 GPU via **MPS** (Metal Performance Shaders) / optionally Core ML. No external NPU.                 |
| Storage   | Mac mini internal SSD (or an attached drive if you want more capture headroom)                        |
| Cameras   | 2x **Reolink E1 Outdoor SE PoE** (4K, PTZ, person/vehicle/animal AI, RTSP+ONVIF)                      |
| Network   | **TP-Link TL-SG1005P** 802.3at PoE switch powering both cameras; mini on same LAN                     |
| Cable     | **GEARit Cat6 direct-burial, solid copper, UV-resistant** (bulk, terminate to length), <=100m per run |

### Critical host notes

- **Co-tenancy with the media server is the main consideration, not RAM or PCIe.** Both
  the media server and this detector are bursty/idle workloads (media spikes on a stream
  request + transcode; detector spikes on a camera motion event). Collisions are rare and
  the M1 absorbs them, but design defensively:
  - **8GB is shared.** Keep the detector lean: no long in-memory frame history, write
    captures to disk promptly, modest model size. Budget ~1-2GB resident for the detector.
  - **GPU/Media Engine is shared** between media transcoding and MPS inference. Throttle
    detection to every Nth frame and one burst per event so you're not fighting a
    transcode for the GPU. In practice this never bites; the knob exists if it does.
- **macOS always-on hygiene:** disable system sleep (`pmset`), ensure the service runs at
  boot without an interactive login (a LaunchDaemon, not a LaunchAgent — see section 6.6),
  and set the detector to relaunch on crash. The media-server appliance setup likely
  already has sleep/boot handled; this slots into that pattern.
- **No Pi, no Hailo, no PCIe, no NPU keying.** All of that is gone. Inference is native
  Apple Silicon GPU. This removes the most error-prone parts of the original plan.

---

## 3. Architecture

```
Camera A (PoE) -- ONVIF/TCP "animal" event --+
                                             |
Camera B (PoE) -- ONVIF/TCP "animal" event --+
                                             v
                                    +------------------+
                                    |  event_listener  |  subscribes to camera events
                                    +--------+---------+  (one task per camera)
                                             | enqueue (camera_id, ts)
                                             v
                                    +------------------+
                                    |   capture_queue  |  serializes capture+infer work
                                    +--------+---------+
                                             v
                                    +------------------+
                                    | detection_worker |  1. open RTSP (main), grab burst
                                    |  (YOLO on MPS)   |  2. infer on M1 GPU
                                    |                  |  3. gate: animal class + conf
                                    |                  |  4. dedupe/cooldown
                                    +--------+---------+
                                             | on positive
                                             v
                          +------------------+------------------+
                          v                                     v
                  write JPEG -> disk                  insert row -> SQLite
                  <captures_dir>/                     captures.db
                  YYYY/MM/DD/<cam>_<ts>_<class>_<conf>.jpg
                                             |
                                             v
                                    +------------------+
                                    |  gallery (Flask) |  reads SQLite, serves thumbnails
                                    +------------------+  bound to LAN
```

**Trigger strategy (chosen):** camera-side motion/animal detection. The host stays
near-idle until a camera pushes an event, then does a short burst grab + inference. On the
M1 this is comfortable even alongside media serving.

**Two-stage filter:** the camera's on-board AI animal class is a _loose_ pre-filter
(tuned for pets/common animals). The YOLO pass on the Mac is the _precise_ gate. Configure
the camera detection sensitivity loosely so it doesn't miss deer/elk/fox its model wasn't
trained on; let the Mac-side YOLO + confidence threshold do the real work.

---

## 4. Repo Layout

```
wildlife-detect/
|- README.md                      # quickstart + summary
|- SPEC.md                        # this document
|- pyproject.toml                 # deps, pinned
|- config.example.yaml            # copy -> config.yaml, gitignored
|- .gitignore                     # config.yaml, *.db, captures/, .venv, models/*.pt, *.mlpackage
|- src/
|  +- wildlife/
|     |- __init__.py
|     |- config.py                # load + validate config.yaml (pydantic)
|     |- events/
|     |  |- __init__.py
|     |  |- base.py               # EventSource ABC: yields CameraEvent
|     |  |- reolink_native.py     # Reolink TCP/ONVIF push listener (reolink-aio)
|     |  +- onvif_bridge.py       # ONVIF pull-point subscriber (fallback)
|     |- capture.py               # RTSP burst grab via OpenCV/FFmpeg, stream lifecycle
|     |- detect.py                # YOLO inference on MPS (Ultralytics), returns Detections
|     |- gate.py                  # animal-class + confidence + box-area + dedupe logic
|     |- store.py                 # SQLite schema, insert, query; file + thumbnail writing
|     |- worker.py                # single-consumer loop tying it together
|     +- gallery/
|        |- __init__.py
|        |- app.py                # Flask app
|        |- templates/index.html
|        +- static/style.css
|- models/
|  +- README.md                   # which YOLO weights; optional Core ML conversion notes
|- scripts/
|  |- test_rtsp.py                # verify RTSP URL + grab one frame per camera
|  |- test_events.py              # print events as they arrive (no detection)
|  |- test_detect.py              # run YOLO on a still image, confirm MPS is used
|  +- prune.py                    # retention: delete captures older than N days
|- launchd/
|  |- com.wildlife.detect.plist   # the worker + listeners (LaunchDaemon)
|  +- com.wildlife.gallery.plist  # the Flask gallery (LaunchDaemon)
+- tests/
   |- test_gate.py                # unit tests for the gate logic (no hardware)
   +- test_store.py               # unit tests for SQLite layer (tmp db)
```

---

## 5. Configuration

`config.example.yaml`:

```yaml
cameras:
  - id: north_field
    host: 192.168.1.101
    username: admin
    password: "CHANGE_ME"
    rtsp_main: "rtsp://{username}:{password}@{host}:554/Preview_01_main"
    rtsp_sub: "rtsp://{username}:{password}@{host}:554/Preview_01_sub"
    onvif_port: 8000
  - id: south_trail
    host: 192.168.1.102
    username: admin
    password: "CHANGE_ME"
    rtsp_main: "rtsp://{username}:{password}@{host}:554/Preview_01_main"
    rtsp_sub: "rtsp://{username}:{password}@{host}:554/Preview_01_sub"
    onvif_port: 8000

event_source: reolink_native # reolink_native | onvif_bridge

capture:
  burst_frames: 5 # grab this many frames per event
  burst_interval_ms: 200 # spacing between grabs
  stream: main # main | sub  (8GB is fine with main; sub is lighter)
  rtsp_timeout_s: 10
  max_concurrent: 1 # serialize capture+infer; one camera at a time

detection:
  model_path: "models/yolov8s.pt" # Ultralytics weights; or a .mlpackage for Core ML
  device: "mps" # mps | cpu  (mps = M1 GPU)
  animal_classes: # COCO subset treated as "wildlife"
    - bird
    - cat
    - dog
    - horse
    - sheep
    - cow
    - bear
    - elephant
    - zebra
    - giraffe
  confidence_threshold: 0.55 # tune: raise for fewer false positives
  min_box_area_frac: 0.01 # ignore specks < 1% of frame area
  save_best_only: true # one best frame per event (vs all positives)

dedupe:
  cooldown_s: 30 # suppress re-triggers from same camera within window

storage:
  captures_dir: "~/wildlife/captures"
  db_path: "~/wildlife/captures.db"
  jpeg_quality: 85
  thumbnail_px: 320

retention:
  max_age_days: 30 # prune.py deletes captures older than this
  min_confidence_keep: 0.0 # optionally also prune low-confidence captures

gallery:
  host: "0.0.0.0" # bind to LAN
  port: 8080
  page_size: 60

resource_guard: # be a good co-tenant with the media server
  detect_every_nth_event: 1 # raise to skip events if GPU contention ever appears
  max_burst_per_minute: 20 # hard cap on how often we'll fire inference
```

`config.py` loads this with **pydantic**, validates types/ranges, expands `~` in paths,
and interpolates the `{username}/{password}/{host}` templates into the RTSP URLs.

---

## 6. Component Specs

### 6.1 Event sources (`events/`)

`base.EventSource` is an abstract base class:

```python
def stream(self) -> Iterator[CameraEvent]: ...
# CameraEvent = dataclass(camera_id: str, event_ts: datetime, kind: str)
```

**`reolink_native.py` (preferred).** Use the `reolink-aio` library (the same library
Home Assistant's Reolink integration uses). It supports the camera's push hierarchy: TCP
push -> ONVIF push -> ONVIF long-poll -> fast-poll, picking the fastest that works.
Subscribe per camera, yield a `CameraEvent` when an _animal_ (or motion, if animal events
aren't separately exposed) event fires. Each camera runs its listener as its own asyncio
task or thread; both feed the shared queue. Auto-reconnect with backoff on drop; log clearly.

**`onvif_bridge.py` (fallback).** If native push is flaky, fall back to ONVIF PullPoint
subscription (`onvif-zeep`). We only care about the _on_ edge (trigger), so the known
Reolink "no motion-off event" quirk is harmless here.

### 6.2 Capture (`capture.py`)

```python
def grab_burst(camera: CameraConfig, n: int, interval_ms: int,
               stream: str, timeout_s: int) -> list[np.ndarray]:
    """Open RTSP, grab n frames spaced interval_ms apart, close stream. Returns frames."""
```

- Use OpenCV's FFmpeg backend (`cv2.VideoCapture(url, cv2.CAP_FFMPEG)`). macOS + Homebrew
  FFmpeg decodes Reolink H.265 fine; the M1 has hardware HEVC decode.
- **Always close the stream** in a `finally` block — Reolink cameras drop connections when
  hit by multiple concurrent streams from the same host IP. Open -> grab -> close.
- Set a small buffer (`cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)`) so stale frames don't queue.
- `stream: sub` is available as a lighter option, but on 8GB with main-stream 4K you're
  fine. Use main for capture quality; switch to sub only if you want to reduce load while
  the media server is busy.
- Respect `max_concurrent: 1` — only the single worker calls this.

### 6.3 Detection (`detect.py`)

```python
class Detector:
    def __init__(self, model_path: str, device: str = "mps"): ...
    def infer(self, frame: np.ndarray) -> list[Detection]: ...
# Detection = dataclass(label: str, confidence: float, box_xyxy: tuple, box_area_frac: float)
```

- Use **Ultralytics YOLO** with the MPS device:
  ```python
  from ultralytics import YOLO
  model = YOLO(model_path)            # e.g. "yolov8s.pt"
  results = model(frame, device="mps", verbose=False)
  ```
- Map results to `Detection` objects with COCO labels, confidence, box geometry; compute
  `box_area_frac` = box area / frame area.
- Load the model **once** at worker startup and reuse across all events (model load is the
  expensive step; inference per frame is cheap on the M1 GPU).
- **Optional Core ML path:** for lower power / better co-tenancy, export the model to Core
  ML (`model.export(format="coreml")`) and run via `coremltools` so inference can use the
  Apple Neural Engine instead of the GPU shaders the media server may want. Document both;
  default to MPS for simplicity, note Core ML as the optimization if GPU contention shows.
- Confirm MPS is actually used (not silently falling back to CPU) in `test_detect.py`.

### 6.4 Gate (`gate.py`) — core logic, fully unit-testable, no hardware

```python
def select_keepers(dets: list[Detection], cfg: DetectionConfig) -> list[Detection]:
    return [d for d in dets
            if d.label in cfg.animal_classes
            and d.confidence >= cfg.confidence_threshold
            and d.box_area_frac >= cfg.min_box_area_frac]

def pick_best(keepers: list[Detection]) -> Detection | None:
    return max(keepers, key=lambda d: d.confidence) if keepers else None
```

Plus a `Deduper` tracking last-save time per camera, suppressing saves within `cooldown_s`,
and enforcing `max_burst_per_minute`. No I/O, no hardware deps — this is the part with real
unit tests (`tests/test_gate.py`).

### 6.5 Store (`store.py`)

SQLite schema:

```sql
CREATE TABLE IF NOT EXISTS captures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT    NOT NULL,
    event_ts    TEXT    NOT NULL,   -- ISO8601, when the camera event fired
    capture_ts  TEXT    NOT NULL,   -- ISO8601, when the frame was saved
    label       TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    box_x1 REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL,
    image_path  TEXT    NOT NULL,   -- relative to captures_dir
    thumb_path  TEXT    NOT NULL,
    width INTEGER, height INTEGER,
    original_label TEXT,               -- model's label before a human reclassify
    reviewed       INTEGER NOT NULL DEFAULT 0,
    reviewed_at    TEXT                -- ISO8601 of the last human action
);
CREATE INDEX IF NOT EXISTS idx_capture_ts ON captures(capture_ts);
CREATE INDEX IF NOT EXISTS idx_camera     ON captures(camera_id);
CREATE INDEX IF NOT EXISTS idx_label      ON captures(label);
```

- File layout: `captures_dir/YYYY/MM/DD/<camera_id>_<capture_ts>_<label>_<conf>.jpg`
  plus a `_thumb.jpg` sibling.
- Write JPEG at `jpeg_quality`; generate a `thumbnail_px`-wide thumbnail at save time
  (Pillow). Thumbnails keep the gallery fast.
- Enable SQLite **WAL mode** so the gallery can read while the worker writes.
- `original_label`/`reviewed`/`reviewed_at` were added after v1, for the admin
  capture-management UI (see 6.7 and the "Managing captures" section of
  `README.md`). `init_schema()` applies them via an idempotent migration
  (`ALTER TABLE ... ADD COLUMN`, guarded so it's a no-op on an already-migrated
  or brand-new DB), so upgrading in place never loses existing rows.

### 6.6 Worker (`worker.py`)

The single consumer:

1. Load config, init Detector (model onto MPS), init Store, init Deduper.
2. Start one EventSource listener per camera, all feeding a `queue.Queue` (or asyncio queue).
3. Loop: pop `(camera_id, event_ts)` -> deduper/cooldown + rate-cap check -> `grab_burst` ->
   for each frame `infer` -> `select_keepers` -> `pick_best` -> if best: write file + thumb +
   SQLite row, update deduper -> log outcome (kept / nothing-above-threshold + reason).
4. Graceful shutdown on SIGTERM (launchd stop): drain queue, close streams, close DB.

Run it as a **LaunchDaemon** (`launchd/com.wildlife.detect.plist`) so it starts at boot
without a login session, with `KeepAlive=true` to relaunch on crash and `RunAtLoad=true`.
Logs go to a file under `~/wildlife/logs/` (or `/usr/local/var/log/`). The gallery runs as
a second LaunchDaemon (`com.wildlife.gallery.plist`).

> Note: LaunchDaemons run as root by default — set `UserName` in the plist to your user so
> paths under `~` and the captures dir resolve to your account, or use absolute paths.

### 6.7 Gallery (`gallery/app.py`)

Minimal Flask app, read-only against SQLite:

- `/` — paginated thumbnail grid, newest first; filters: camera, date range, class,
  min-confidence. Lazy-load full images on click (modal/lightbox).
- `/image/<id>` and `/thumb/<id>` — serve full JPEG / thumbnail.
- `/api/captures` — JSON for the grid (filters + `page`).
- Bind to `0.0.0.0:8080`; reach it from any device on your LAN (like the media server UI).
  No auth in v1 (LAN-only); README notes that exposing beyond LAN needs auth + TLS.
- Keep it light: server-rendered grid + a little vanilla JS. No heavy frontend framework.

**Admin capture management** (post-v1; password-gated, part of the optional
`/admin` blueprint — see `README.md`'s Admin section):

- `GET /admin/captures` — the same filters as `/` plus a `reviewed` state, over a
  selectable thumbnail grid.
- `POST /admin/captures/<id>/delete` — permanently removes the row and both
  JPEGs (full + thumbnail), reusing the same file-deletion helpers as
  `scripts/prune.py`.
- `POST /admin/captures/<id>/reclassify` — relabels a capture; the first edit
  records the model's original label in `original_label` and sets `reviewed`.
- `POST /admin/captures/bulk` — delete, reclassify, or mark-reviewed over a set
  of selected ids in one request.

---

## 7. Build Order (incremental, each step testable)

1. **Repo skeleton** — pyproject, config loading + validation, .gitignore. Commit.
2. **`scripts/test_rtsp.py`** — pull one frame from each camera's RTSP URL. Validates
   credentials, URLs, and the FFmpeg/OpenCV HEVC decode path. **Hard gate.**
3. **`scripts/test_events.py`** — prove camera events reach the Mac. Print each event.
   Validates the `reolink_native` path. **Hard gate.**
4. **`detect.py` + `scripts/test_detect.py`** — run YOLO on a saved still, print
   detections, and **confirm it's running on MPS** (not CPU fallback). Validates the
   Ultralytics + Metal path and label mapping.
5. **`gate.py` + `tests/test_gate.py`** — pure logic, full unit tests. No hardware.
6. **`store.py` + `tests/test_store.py`** — schema, insert, query, file + thumb writing.
7. **`capture.py`** — burst grab with strict stream lifecycle.
8. **`worker.py`** — wire it together; run interactively, watch logs, trigger by walking
   in front of a camera.
9. **`gallery/`** — Flask app over the populated DB.
10. **launchd plists** — install both LaunchDaemons, load, reboot, confirm auto-start.
11. **`scripts/prune.py` + a launchd-scheduled (StartCalendarInterval) run** — retention.
12. **Admin capture management** (post-v1, builds on the optional admin editor) —
    `store.py` gains `delete`/`delete_many`/`update_label`/`update_label_many`/
    `mark_reviewed_many`; the `/admin/captures` routes (6.7) wrap them; retention
    (step 11) is updated to spare `reviewed` rows from `min_confidence_keep`.

---

## 8. Dependencies (pin in pyproject.toml)

- `ultralytics` — YOLO + MPS inference (pulls in `torch` with Metal support)
- `torch` / `torchvision` — Apple Silicon builds include MPS automatically
- `reolink-aio` — Reolink native event push + camera control
- `onvif-zeep` — ONVIF fallback
- `opencv-python` — frame handling (full build is fine on macOS; FFmpeg backend for RTSP)
- `pillow` — thumbnails
- `pydantic` — config validation
- `flask` — gallery
- `pyyaml` — config
- `pytest` — tests
- (optional) `coremltools` — Core ML export path for ANE inference

System: install **FFmpeg via Homebrew** (`brew install ffmpeg`) for robust RTSP/HEVC
decode. Use a Python virtualenv (`python3 -m venv .venv`). Develop in VS Code on the mini.

> No Hailo SDK, no `.hef` compilation, no Pi OS setup. The model is a stock Ultralytics
> `.pt` file downloaded on first run, optionally exported to Core ML.

---

## 9. Operational Concerns

- **Co-tenancy:** the detector shares the M1 with the media server. Both bursty; collisions
  rare. Use `resource_guard` (rate cap, optional event skipping) and consider the Core ML
  path if you ever observe transcode/inference GPU contention. Run the detector at a
  slightly nice'd priority if desired.
- **Always-on hygiene:** `sudo pmset -a sleep 0` (or appropriate display/disk settings),
  LaunchDaemons with `RunAtLoad` + `KeepAlive`. Confirm the system already won't sleep
  (it's a media appliance, so likely set) — the detector must keep running headless.
- **Storage math:** gated JPEGs at ~300KB-1MB each. Even hundreds/day is trivial for the
  internal SSD over a `max_age_days` window. `prune.py` (scheduled daily) enforces
  retention. Point `captures_dir` at a roomy volume if you want long history.
- **Logging:** structured logs to a file (launchd captures stdout/stderr too). Log every
  event received and every capture decision (kept / rejected + reason) for tuning.
- **Tuning loop:** start `confidence_threshold` ~0.5-0.6. Review the gallery; raise if
  junk, lower if missing animals. Use the cameras' motion zones to mask roads/trees/feeders.
- **Night:** disable the camera spotlight if it spooks wildlife; rely on IR/color-night.
  YOLO runs on grayscale IR frames a lot — plan to fine-tune on your own night captures
  later for local species (deer/elk/fox aren't COCO classes).
- **Reolink connection discipline:** open -> grab -> close. Don't run other clients (the
  Reolink app live view, another NVR) hammering the same camera while the worker runs.

---

## 10. Future (post-v1, noted not built)

- Fine-tune a detector on local species from your own captures (the gallery becomes your
  labeling source). Export the fine-tuned model to Core ML for efficient ANE inference.
- Optional push notifications (ntfy/Telegram) on high-confidence captures.
- Optional off-host backup (NAS/external) of the captures tree.
- Per-camera PTZ presets so auto-tracking framing is consistent for time-comparison.
- If the media server and detector ever genuinely contend, split the detector onto a
  dedicated low-power box (the original Pi/Hailo plan remains a valid fallback).
