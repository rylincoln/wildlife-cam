# Wildlife Detection System

Fully-local, motion-triggered wildlife detection for PoE cameras, running on an
Apple Silicon Mac. It subscribes to camera-side detection events, grabs a short
RTSP frame burst per event, runs YOLO object detection on the Mac GPU (MPS), and
saves **only** frames containing an animal at or above a confidence threshold.
Captures land on local disk with a row in SQLite, and a small read-only Flask
gallery lets you browse them on your LAN. No cloud, no subscriptions.

See [`spec.md`](spec.md) for the full build specification (architecture, build
order, operational notes).

---

## Requirements

- macOS on Apple Silicon (M1 or newer) — inference runs on the GPU via MPS.
- Python 3.12 (3.11–3.13 supported).
- Homebrew FFmpeg for robust RTSP / H.265 (HEVC) decode.
- Reolink (or ONVIF-capable) PoE cameras reachable on the LAN.

---

## Quickstart

### 1. System dependencies

```bash
brew install ffmpeg
```

### 2. Create the virtualenv and install

[`uv`](https://github.com/astral-sh/uv) is recommended, but plain `python -m venv`
+ `pip` works too.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[detect,cameras,dev]"
```

Optional extras:

- `coreml` — export the model to Core ML for Apple Neural Engine (ANE) inference
  (`uv pip install -e ".[coreml]"`). See [`models/README.md`](models/README.md).

### 3. Configure

```bash
cp config.example.yaml config.yaml
# edit config.yaml: set each camera host, username, password
```

`config.yaml` is gitignored — your credentials never get committed. The RTSP URLs
use `{username}/{password}/{host}` templates that `wildlife.config` interpolates
at load time, and `~` / `$ENV` in the storage paths are expanded automatically.

---

## On-device build & test order (from spec section 7)

Run these in order. The first two are **HARD GATES** — do not proceed until they
pass against your real cameras.

1. **`scripts/test_rtsp.py`** — pull one frame from each camera's RTSP URL.
   Validates credentials, URLs, and the FFmpeg/OpenCV HEVC decode path.
   **Hard gate.**
2. **`scripts/test_events.py`** — prove camera detection events reach the Mac;
   prints each event as it arrives (no detection). Validates the
   `reolink_native` event path. **Hard gate.**
3. **`scripts/test_detect.py`** — run YOLO on a saved still image, print
   detections, and confirm inference is actually running on **MPS** (not a
   silent CPU fallback).
4. Unit tests (no hardware): `pytest` covers the gate logic and the SQLite store.

```bash
python scripts/test_rtsp.py        # hard gate 1
python scripts/test_events.py      # hard gate 2
python scripts/test_detect.py path/to/still.jpg
pytest
```

---

## Running

### Worker (event listener + capture + detection)

```bash
wildlife-worker            # uses ./config.yaml
# or: wildlife-worker /path/to/config.yaml
```

The worker loads the YOLO model once at startup, runs one event listener per
camera into a shared queue, and for each event does: cooldown/rate-cap check →
RTSP burst grab → infer each frame → gate on animal class + confidence + box
area → save the best frame (JPEG + thumbnail + SQLite row). It logs every
capture decision (kept / rejected + reason) for tuning. It shuts down gracefully
on `SIGTERM`/`SIGINT` (drains the queue, closes streams and the DB).

### Gallery (read-only web UI)

```bash
wildlife-gallery           # binds config.gallery host/port (default 0.0.0.0:8080)
```

Open `http://<mac-LAN-ip>:8080` from any device on your LAN. Paginated thumbnail
grid, newest first, with filters for camera, date range, class, and minimum
confidence; click a thumbnail to load the full image.

> **Security:** the gallery has **no authentication** and is intended for
> LAN-only use (like the media-server UI on the same machine). Do **not** expose
> it to the internet without adding auth + TLS in front of it.

---

## Running as a service (launchd)

For 24/7 operation, install the worker and gallery as **LaunchDaemons** (not
LaunchAgents) so they start at boot without an interactive login and relaunch on
crash. Example plists live under `launchd/`.

- Use a **LaunchDaemon** (`/Library/LaunchDaemons/com.wildlife.detect.plist` and
  `com.wildlife.gallery.plist`).
- Set `<key>UserName</key>` to your account so paths under `~` and the captures
  directory resolve to your user (LaunchDaemons run as root by default), or use
  absolute paths in `config.yaml`.
- Set `RunAtLoad` **and** `KeepAlive` to `true` so they start at boot and
  relaunch on crash.
- Point stdout/stderr to a log file under `~/wildlife/logs/`.

Install and load:

```bash
sudo cp launchd/com.wildlife.detect.plist   /Library/LaunchDaemons/
sudo cp launchd/com.wildlife.gallery.plist  /Library/LaunchDaemons/
sudo launchctl load /Library/LaunchDaemons/com.wildlife.detect.plist
sudo launchctl load /Library/LaunchDaemons/com.wildlife.gallery.plist
```

### Keep the Mac awake

The detector must keep running headless, so disable sleep:

```bash
sudo pmset -a sleep 0
```

(A media-server appliance likely already has this set.)

---

## Retention

`scripts/prune.py` deletes captures older than `retention.max_age_days` (and,
optionally, below `retention.min_confidence_keep`). Schedule it daily via a
`StartCalendarInterval` LaunchDaemon.

---

## Layout

- `src/wildlife/` — the package (config, events, capture, detect, gate, store,
  worker, gallery).
- `scripts/` — on-device test + maintenance scripts.
- `models/` — YOLO weights (auto-downloaded on first run; gitignored). See
  [`models/README.md`](models/README.md).
- `launchd/` — example LaunchDaemon plists.
- `tests/` — hardware-free unit tests.

For full details, read [`spec.md`](spec.md).
