# Wildlife Detection System

Fully-local wildlife detection for PoE cameras, running on an Apple Silicon Mac.
It grabs a short RTSP frame burst per detection event, runs a **YOLO detector** on
the Mac GPU (MPS), and saves **only** frames containing a target animal at or above
a confidence threshold. It works out of the box with a stock model, and the
included [`training/`](training/README.md) toolchain lets you fine-tune a detector
on **your own local wildlife** — the species stock models don't know — which drops
in with no code change.

Three detection paths feed the same pipeline, all optional and independent:

- **Camera AI events** — subscribe to the camera's onboard detection and burst-grab on each.
- **Continuous motion gate** — an always-on MOG2 motion detector fires *your* model, catching wildlife the camera's person/car AI ignores.
- **BirdNET audio bird-ID** — identify birds by song from the camera's audio (CPU-side), saved as playable spectrograms.

Captures land on local disk with a row in SQLite, and a Flask gallery lets you
browse them on your LAN (with an optional password-gated admin editor and
Cloudflare-Tunnel remote access). No cloud, no subscriptions.

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

### Automated setup (recommended)

One script installs everything — Homebrew ffmpeg, the Python 3.12 venv + project,
and the optional [go2rtc](#live-view-optional) binary — then creates
`~/wildlife/logs`, copies `config.yaml` from the example, and runs the unit tests:

```bash
./scripts/setup_macos.sh
```

Afterward, edit `config.yaml` with your camera IPs/credentials. To do it by hand
instead — or to understand each step — follow the manual steps below.

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

Optional extras (installed as needed; see their sections): `admin` (config
editor), `audio` (BirdNET), `train` / `autolabel` (the training toolchain).

- `coreml` — **currently non-functional on this stack.** Core ML export was meant
  to target the Apple Neural Engine, but the installed torch (2.12) is too new for
  `coremltools`, so `.export(format="coreml")` fails. Detection runs as a `.pt` on
  **MPS** instead (no Core ML needed). The `[coreml]` extra still
  exists for when the toolchain versions realign. See [`models/README.md`](models/README.md).

> **Note:** the shipped `.venv` is **uv-managed** and has no `pip` binary — use
> `uv pip install …` (as above), not `pip`. `pip` only works in a hand-made
> `python -m venv`.

### 3. Configure

```bash
cp config.example.yaml config.yaml
# edit config.yaml: set each camera host, username, password
```

`config.yaml` is gitignored — your credentials never get committed. The RTSP URLs
use `{username}/{password}/{host}` templates that `wildlife.config` interpolates
at load time, and `~` / `$ENV` in the storage paths are expanded automatically.

### Test mode (no cameras)

Run the whole app — gallery + Live tab — against a built-in test video, with no
real cameras and without touching your `config.yaml`:

```bash
./scripts/run_demo.sh                 # virtual test pattern, seeded gallery, http://localhost:8080
./scripts/run_demo.sh --source mux    # a real public video feed in the Live tab
./scripts/run_demo.sh --detect        # also run a one-shot capture -> YOLO on the feed
```

It starts go2rtc against the test source, launches the gallery with the Live tab
on, seeds some synthetic captures so the grid has content, prints the URLs, and
cleans everything up on `Ctrl-C`. All demo state lives in `./.demo` (gitignored).
Flags: `--source virtual|mux|<go2rtc-source>`, `--port N`, `--seed N`, `--detect`.

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
python scripts/test_detect.py --image path/to/still.jpg
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
grid, newest first, with filters for camera, date range, class, minimum
confidence, and **source** — *Camera AI* (`reolink`), *Motion* (`continuous`), or
*Audio (birds)* (`audio`) — so you can isolate each detection path. Click a
thumbnail to load the full image; audio detections open a playable spectrogram.

> **Security:** the gallery has **no authentication** and is intended for
> LAN-only use (like the media-server UI on the same machine). Do **not** expose
> it to the internet without adding auth + TLS in front of it.

---

## Live view (optional)

The detector saves still frames only — it never records video. If you also want
a **live look** at a camera, the gallery can embed an on-demand low-latency
player backed by [**go2rtc**](https://github.com/AlexxIT/go2rtc), an external
binary that runs as a third service. This is fully **additive**: the worker
(`worker.py` / `capture.py` / `detect.py`) is untouched and keeps doing its own
short RTSP bursts. go2rtc pulls a camera's RTSP feed **only while someone is
watching** and re-publishes it as WebRTC/MSE.

For each camera, two streams are exposed: `<camera_id>_sub` (lighter, the
default) and `<camera_id>_main` (full 4K). The `/live` pages let a viewer toggle
between **Sub** and **Main** per camera.

### 1. Install go2rtc

go2rtc is a single static binary — no dependencies. Download the **arm64**
(Apple Silicon) build; on macOS the release ships as a **`.zip`** (unzip it to
get the `go2rtc` binary):

- from [`AlexxIT/go2rtc` releases](https://github.com/AlexxIT/go2rtc/releases)
  (`go2rtc_mac_arm64.zip`), or
- from [`bropat/go2rtc-static`](https://github.com/bropat/go2rtc-static).

```bash
# example: download + unzip, then install to a Homebrew bin dir already on PATH
cd /tmp
curl -L -o go2rtc_mac_arm64.zip \
  https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_mac_arm64.zip
unzip -o go2rtc_mac_arm64.zip                  # -> ./go2rtc
mv go2rtc /opt/homebrew/bin/go2rtc
chmod +x /opt/homebrew/bin/go2rtc
# if macOS Gatekeeper blocks it: xattr -d com.apple.quarantine /opt/homebrew/bin/go2rtc
```

(Install it wherever you like — just note the path; the LaunchDaemon below
points at `/opt/homebrew/bin/go2rtc` and has a comment showing where to edit it.)

### 2. Generate `go2rtc.yaml` from your config

```bash
wildlife-stream-config                 # reads ./config.yaml -> writes ./go2rtc.yaml
# or: wildlife-stream-config config.yaml go2rtc.yaml
```

This reads your cameras (reusing the same credential-interpolated RTSP URLs the
detector uses) and the `livestream:` block in `config.yaml`, then writes
`go2rtc.yaml` with one `<id>_main` and one `<id>_sub` stream per camera plus the
api/webrtc/rtsp listen ports. `go2rtc.yaml` is **gitignored** because it embeds
camera RTSP credentials (just like `config.yaml`). Re-run it whenever you change
cameras or ports.

### 3. Test-run go2rtc in the foreground

```bash
go2rtc -config go2rtc.yaml
```

Open `http://<mac-LAN-ip>:1984/` (go2rtc's own dashboard) and confirm each
camera's `_sub` / `_main` stream plays. Stop it with `Ctrl-C` once it works.

### 4. Enable it in the gallery

Add a `livestream:` block to `config.yaml` and set `enabled: true`, then restart
the gallery:

```yaml
livestream:
  enabled: true            # off by default; turning this on reveals /live
  go2rtc_port: 1984        # browser-reachable go2rtc api port
  # go2rtc_url:            # optional full override, e.g. "http://192.168.1.50:1984";
                           # if unset, the gallery derives it from the request host
  default_stream: sub      # sub | main  (sub is lighter; default)
  allow_main: true         # show the Main (4K) toggle in the UI
  main_mode: "webrtc,mse"  # player transport for the 4K main tile (HEVC → MSE in Safari)
  sub_mode: "webrtc"       # player transport for the sub tile
```

> **Reolink codec note:** the player prefers MSE whenever `mse` is listed (the
> string order is ignored), so omit `mse` to force WebRTC. Reolink's 4K **main**
> is HEVC — Safari plays it via **MSE** but not WebRTC — while the **sub** stream
> is H.264 *High profile*, which Safari's **MSE** refuses (unsupported codec
> string) but WebRTC decodes. That's why the two tiles default to different
> transports; adjust per your cameras/browser if needed.

Now browse to `http://<mac-LAN-ip>:8080/live` for a grid of all cameras, or
`http://<mac-LAN-ip>:8080/live/<camera_id>` for a single camera. (When
`livestream.enabled` is `false`, the `/live` routes return 404 and the gallery
shows no live link.)

### Networking & security

- **LAN ports to allow** on the Mac mini: **1984** (go2rtc api + player),
  **8555** (WebRTC), **8554** (RTSP). These are the defaults from the
  `livestream` block (`go2rtc_port` / `webrtc_listen` / `rtsp_listen`).
- **On-demand:** go2rtc connects to a camera only while that stream is being
  watched, so an idle `/live` page costs nothing and respects Reolink's
  open→watch→close connection discipline.
- **LAN-only / no auth:** like the gallery, go2rtc has **no authentication** by
  default and is intended for LAN-only use. Do **not** expose ports 1984/8555/
  8554 to the internet. go2rtc can add HTTP basic auth via `api.username` /
  `api.password` in `go2rtc.yaml` if you need a minimal gate.

---

## Admin (config editor, optional)

A password-gated **`/admin`** section in the gallery lets you edit detection
tuning and add/configure cameras from the browser — without hand-editing
`config.yaml`. It is built to be hard to footgun:

- **Validated writes.** Every save is checked through the *same*
  pyyaml → pydantic path the daemons load with (types, ranges, unique camera
  ids, RTSP scheme). A bad edit is rejected with a field-level message and the
  file on disk is left untouched.
- **Test-before-save.** Adding/editing a camera runs a real probe — it grabs a
  live RTSP frame and connects via `reolink-aio` (verifying credentials,
  reachability, channels, and AI object types). On save, the detector's capture
  stream is grabbed once to confirm it works before the config is written
  (tick *Save without testing* to override).
- **Safe on disk.** Writes are atomic (temp file + rename), preserve your YAML
  comments, and keep timestamped backups in `.config-backups/`.

**Enable it:**

```bash
# 1. Ensure the admin extra is installed (setup_macos.sh already includes it):
uv pip install -p .venv -e '.[admin]'          # adds ruamel.yaml

# 2. Set an admin password (stored only as a hash in config.yaml):
.venv/bin/wildlife-admin-password

# 3. Open http://<mac-LAN-ip>:8080/admin  (log in with any username + that password)
```

**How saved changes take effect.** Detection/camera edits are consumed by the
*worker* and *go2rtc*, which must restart to pick them up — the gallery can't do
that itself. The optional **reloader daemon** (`com.wildlife.reload`, installed
by `scripts/install_launchd.sh`) watches for edits and, as root, re-validates the
config, regenerates `go2rtc.yaml`, and restarts the worker + go2rtc within a few
seconds. The dashboard shows the last-apply status. If the reloader isn't
installed, saves still persist — just restart manually:

```bash
sudo launchctl kickstart -k system/com.wildlife.detect
sudo launchctl kickstart -k system/com.wildlife.stream
```

**Security.** The admin section is separately password-protected (unlike the
open, read-only gallery), but the password crosses the LAN via HTTP Basic Auth —
keep it LAN-only, or front it with a reverse proxy providing TLS. With no
password set, `/admin` fails closed (403).

### Managing captures

Browse to **`/admin/captures`** (the "Captures" tab) to review and clean up
detections. It reuses the gallery filters (camera, class, date, plus a
**Review** state) over a selectable thumbnail grid:

- **Reclassify** a capture to another species from the dropdown. The first edit
  records the model's original prediction (`original_label`) and marks the row
  **reviewed**, so human-corrected captures are usable later as clean training
  data. (Reclassifying is DB-only — the on-disk filename keeps its original
  label, which is harmless because files are resolved by their stored path.)
- **Delete** a capture. This is **permanent**: it removes the SQLite row *and*
  the JPEGs (full + thumbnail) — and, for audio detections, the `.m4a` clip —
  mirroring `scripts/prune.py`. There is no undo. Deleting is the disposal path
  for false positives.
- **Bulk**: tick the checkboxes to delete, reclassify, or mark-reviewed many at
  once. Selection applies to the current page.
- A **"Mark reviewed"** action and the **Unreviewed** filter let you work
  through a backlog without re-seeing handled captures.

Reviewed captures are exempt from retention's `min_confidence_keep` rule (a
human-confirmed low-confidence capture is not auto-pruned); they are still
subject to `max_age_days`, so export anything you want to keep permanently.

---

## Remote access (Cloudflare Tunnel)

Reach the gallery + live view from anywhere at one URL (e.g. `https://cam.example.com`),
protected by a **shared secret in the link** — no ports forwarded, home IP hidden,
`/admin` unreachable remotely, and your LAN unchanged. Free Cloudflare plan.

**How the gate works.** Requests arriving through the tunnel (from the local
`cloudflared`, i.e. a loopback address) must carry the secret: as `?key=…` on the first
visit, then a `wl_key` cookie. LAN requests (real private IPs) are never gated. `/admin`
returns 404 over the tunnel. Live video is go2rtc's MSE player under `/go2rtc` (WebRTC
can't cross a tunnel), gated at Cloudflare's edge by one WAF rule checking the same cookie.

### One-time setup

> On the production Mac mini you can run `./scripts/setup_remote.sh` (after creating a
> tunnel in the dashboard and exporting `CF_TUNNEL_TOKEN`) to do the host-side setup —
> it configures `config.yaml`, mints the share secret, regenerates `go2rtc.yaml`,
> restarts go2rtc, and prints the remaining dashboard/camera steps with values filled in.

1. `brew install cloudflared` (keep current: `brew upgrade cloudflared`).
2. Cloudflare dashboard → Zero Trust → Networks → Tunnels → Create → Cloudflared → name
   it; copy the token; `sudo cloudflared service install <TOKEN>` (boot daemon).
3. Add two **public hostnames** on that tunnel (order matters — the `/go2rtc` one first):
   - `cam` . `example.com`, **Path** `go2rtc` → HTTP `localhost:1984`
   - `cam` . `example.com`, (no path) → HTTP `localhost:8080`
4. In `config.yaml` set `remote.base_url: "https://cam.example.com"` and
   `livestream.base_path: "/go2rtc"`; regenerate go2rtc config
   (`wildlife-stream-config`) and restart go2rtc; the gallery picks up `config.yaml` changes automatically.
5. Run `wildlife-share-secret` → note the printed **share link** and **raw secret**.
6. Add a Cloudflare **WAF custom rule**: *Block* when the path starts with `/go2rtc` and
   the `wl_key` cookie ≠ the raw secret. (Free-plan fallback: a small Cloudflare Worker on
   `cam.example.com/go2rtc/*` that checks the `wl_key` cookie.)
7. Set each camera's **sub-stream to H.264 Main/Baseline** so MSE live plays in every
   browser incl. Safari/iOS.
8. Keep **Total TLS OFF** for the zone so `cam.` stays out of Certificate Transparency logs.

### Using / rotating

- Share `https://cam.example.com/?key=<secret>`; the recipient's browser stores the
  cookie so the key only appears once.
- **Rotate / revoke everyone:** re-run `wildlife-share-secret` and update the WAF rule
  with the new secret. Old links stop working.

> **Security note.** This is view-only "anyone with the link" access: a forwarded link
> grants access until you rotate. `/admin` is never exposed remotely. See
> `docs/superpowers/specs/2026-07-02-remote-access-cloudflare-tunnel-design.md` for the
> full threat model and the go2rtc-exposure trade-off.

---

## Continuous (motion-gated) detection

By default the pipeline only looks when Reolink's onboard AI fires. Continuous
detection adds a second, always-on producer per camera that runs a cheap MOG2
motion gate on the go2rtc **sub** restream and fires your own YOLO on motion — so
*your* fine-tuned model, not Reolink's person/car AI, decides what counts. This
catches small, distant, and nocturnal wildlife the camera silently ignores.

**Requires the go2rtc daemon running** (installed and started per
[Live view (optional)](#live-view-optional)): the motion reader consumes go2rtc's
`_sub` restream directly at `rtsp://127.0.0.1:<rtsp_listen port>/<camera id>_sub`,
and bursts are grabbed back through go2rtc's configured burst stream
(`capture.stream`, `main` by default) — no second direct camera session. This is
independent of `livestream.enabled`, which only controls the gallery's in-browser
Live tab; go2rtc generates and serves the per-camera `_main`/`_sub` streams
regardless of that setting, so the daemon just needs to be running. Because the
motion reader keeps a client connected, go2rtc holds that `_sub` upstream open
continuously (rather than only while someone is watching the live page). No
launchd start-order changes are needed: the
shipped daemons start the worker (`com.wildlife.detect`) before go2rtc
(`com.wildlife.stream`), but the motion reader retries with backoff until go2rtc is
reachable, so at boot you may briefly see it reconnecting in the logs until go2rtc
comes up.

Enable it in `config.yaml`:

    continuous:
      enabled: true

**Motion masks (effectively required in busy scenes).** A yard with a road,
swaying canopy, a flag, or moving water will trip MOG2 constantly. Add per-camera
`motion_mask` ignore polygons (normalized 0..1 coordinates) to blank those regions:

    cameras:
      - id: north_field
        # ...
        motion_mask:
          - [[0.0, 0.7], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]]  # ignore the bottom 30%

**Tuning knobs** (start with the defaults, then adjust on real footage):
`min_area_frac` and `refractory_s` are the dominant levers for cutting trigger
volume; `active_hours` duty-cycles by time of day; `algorithm: frame_diff` is a
lighter fallback for a weaker host. Continuous captures are tagged
`source_kind = "continuous"` in the database (vs `"reolink"`) so you can tell the
two paths apart when tuning.

Because a second producer now feeds the same queue, the global
`resource_guard.max_burst_per_minute` cap (which limits how often inference
fires) may need raising so continuous events aren't starved by Reolink ones; the
per-camera `dedupe.cooldown_s` and `resource_guard.detect_every_nth_event`
throttles apply to continuous events for free.

---

## Audio bird-ID (optional)

A second detection modality: a per-camera analyzer identifies birds by song using
[BirdNET](https://github.com/birdnet-team/birdnet), reading the camera's audio off the
same go2rtc restream. It runs **CPU-side** (no contention with YOLO on the GPU) and saves
each confirmed detection as a **playable spectrogram** — a spectrogram thumbnail in the
gallery grid; click it for the spectrogram plus an audio player with a synced playhead.

**Install the extra** (heavy — pulls TensorFlow):

    uv pip install -e ".[audio]"

> `birdnet` brings **full TensorFlow (~1 GB)** plus scipy/pandas/pyarrow/soundfile/kagglehub.
> It is CPU-only. On first run it downloads model weights from Kaggle Hub (needs network +
> a writable cache once). On the 8 GB mini, watch memory alongside torch/YOLO.

**Requires the go2rtc daemon** (like continuous detection) and a camera **audio track**.
Some Reolink models only carry audio on the *main* stream — confirm with
`ffprobe rtsp://127.0.0.1:8554/<id>_sub` and set `audio.stream: main` if `_sub` has no audio.
Prefer selecting **AAC/16000** on the camera over G.711 8 kHz (which clips high-frequency
calls).

**Enable it** in `config.yaml` with your real coordinates (they stay in the gitignored
`config.yaml`):

    audio:
      enabled: true
      latitude: 37.228274
      longitude: -107.519089

**Tuning (cut wind/false positives):** `min_confirmations` + `confirm_window_s` are the
dominant lever (wind won't reproduce the *same* species repeatedly); raise
`confidence_threshold`; set `bandpass_fmin` (e.g. 300 Hz) to band-limit low-frequency wind;
`use_geo_filter` trims implausible species by location/season; `active_hours` duty-cycles.
Audio detections are tagged `source_kind = "audio"` in the DB and filterable in the gallery.

---

## Running as a service (launchd)

For 24/7 operation, install the worker and gallery as **LaunchDaemons** (not
LaunchAgents) so they start at boot without an interactive login and relaunch on
crash. Example plists live under `launchd/`. The easiest path is
`sudo bash scripts/install_launchd.sh`, which installs all daemons
(worker, gallery, go2rtc, and the admin reloader), creates the log dir, and
arms the reload trigger.

- Use a **LaunchDaemon** (`/Library/LaunchDaemons/com.wildlife.detect.plist` and
  `com.wildlife.gallery.plist`). If you enabled live view, there is an optional
  **third** LaunchDaemon, `com.wildlife.stream.plist`, that runs go2rtc (see
  [Live view (optional)](#live-view-optional)). If you use the admin editor,
  there is an optional **fourth**, `com.wildlife.reload.plist`, that applies
  saved config changes (see [Admin](#admin-config-editor-optional)); it runs as
  **root** (no `UserName`) and is `WatchPaths`-triggered, so it stays idle until
  an edit occurs.
- The plists ship with a `__USER__` / `/Users/__USER__` placeholder (so the repo
  carries no personal username). `scripts/install_launchd.sh` substitutes your
  deploy user + home automatically; if you install a plist by hand, replace
  `__USER__` first, or use absolute paths in `config.yaml`.
- Set `RunAtLoad` **and** `KeepAlive` to `true` so they start at boot and
  relaunch on crash.
- Point stdout/stderr to a log file under `~/wildlife/logs/`.

Install and load — easiest is the script (fills in `__USER__` for you):

```bash
sudo bash scripts/install_launchd.sh          # installs all daemons, substituting your user

# ...or by hand (replace __USER__ with your username first):
sed "s/__USER__/$USER/g; s#/Users/__USER__#$HOME#g" launchd/com.wildlife.detect.plist \
  | sudo tee /Library/LaunchDaemons/com.wildlife.detect.plist >/dev/null
sudo launchctl load /Library/LaunchDaemons/com.wildlife.detect.plist
```

### Keep the Mac awake

The detector must keep running headless, so disable sleep:

```bash
sudo pmset -a sleep 0
```

(A media-server appliance likely already has this set.)

---

## Detecting local wildlife (training)

The stock COCO model only names a handful of animals (bear, generic bird) and none
of the deer, elk, foxes, big cats, and so on that fill most regions — so
`detection.animal_classes` can't filter for what a COCO model never predicts. The
[`training/`](training/README.md) toolchain fine-tunes a detector on **your own
local species** and drops in with **no code change**: `detect.py` reads class names
from the model itself and `gate.py` keeps only detections whose label is in
`animal_classes`, so you just point `detection.model_path` at your `.pt` and match
`animal_classes`. Define your region's taxonomy in `training/species.py` (it ships
a southwest-Colorado set as a worked example — edit it for your area).

The workflow:

- **Bootstrap from public data** before you have your own captures — LILA
  camera-trap datasets (`download_lila.py` → `convert_lila.py`) plus an
  **iNaturalist gap-fill** for species the camera-trap sets lack
  (`download_inat.py` → `label_boxes.py`, which boxes CC-licensed photos with
  **MegaDetector v6** and forces the known species label).
- **Auto-label your own captures** with **SpeciesNet** (`autolabel.py`).
- **Two-stage YOLO fine-tune** (`train.py`), then **deploy the `best.pt` on MPS**:
  copy it into `models/`, set `model_path` + `animal_classes`, restart the worker.

> Core ML export (a `.mlpackage` for the Apple Neural Engine) is the intended fast
> path but currently **fails** — the installed torch is too new for `coremltools` —
> so the model runs as a `.pt` on the GPU (MPS) instead. `train.py` already falls
> back to the `.pt` and prints the deploy hint. Dataset choices and exact commands
> are in [`training/README.md`](training/README.md).

---

## Retention

`scripts/prune.py` deletes captures older than `retention.max_age_days` (and,
optionally, below `retention.min_confidence_keep`) — removing the SQLite row, the
JPEGs, and any audio clip. Human-reviewed captures are exempt from the
confidence rule. Schedule it daily via a `StartCalendarInterval` LaunchDaemon —
no prune plist ships in `launchd/`, so author your own from one of the examples there.

---

## Layout

- `src/wildlife/` — the package: `config`, `capture`, `detect`, `gate`, `store`,
  `worker`; `motion` + `audio` + `audio_gate` + `_colormap` (continuous & BirdNET
  detection); `events/` (`reolink_native`, `continuous_motion`, `audio_detection`,
  `onvif_bridge`); and the `gallery/`, `admin/`, `remote/`, `stream/` subpackages.
- `scripts/` — on-device test + maintenance scripts (`setup_macos.sh`,
  `run_demo.sh`, `install_launchd.sh`, `setup_remote.sh`, `prune.py`, `test_*.py`).
- `training/` — offline toolchain to fine-tune a local-species detector: public-data
  bootstrap (LILA `download_lila`/`convert_lila` + iNaturalist `download_inat`/
  `label_boxes`) → autolabel → split → two-stage fine-tune → deploy `best.pt` on
  MPS. See [`training/README.md`](training/README.md).
- `models/` — model weights (gitignored). Stock base checkpoints (e.g. `yolov8s.pt`)
  auto-download; your fine-tuned `.pt` from the training toolchain is copied in by
  hand. See [`models/README.md`](models/README.md).
- `launchd/` — example LaunchDaemon plists (worker, gallery, go2rtc stream, reloader).
- `docs/` — design specs and implementation plans for the larger features.
- `tests/` — hardware-free unit tests.

For full details, read [`spec.md`](spec.md).

---

## Acknowledgements

This project stands on some excellent open-source work:

- [**go2rtc**](https://github.com/AlexxIT/go2rtc) — the on-demand WebRTC/MSE
  restreamer behind the Live view, continuous motion, and audio paths.
- [**Ultralytics YOLO**](https://github.com/ultralytics/ultralytics) — the object
  detector framework, used stock and as the base for fine-tuning a local model.
- [**BirdNET**](https://github.com/birdnet-team/birdnet) — the acoustic model
  behind the audio bird-ID feature (pulls TensorFlow via the `[audio]` extra).
- [**MegaDetector**](https://github.com/microsoft/CameraTraps) / Pytorch-Wildlife —
  the camera-trap animal detector used (v6, via Ultralytics) to box the iNaturalist
  gap-fill images in `training/label_boxes.py`.
- [**SpeciesNet**](https://github.com/google/cameratrapai) — used to auto-label
  training images with species predictions (bundles MegaDetector).
- [**LILA BC**](https://lila.science/) camera-trap datasets (ENA24, Caltech
  Camera Traps, Idaho, etc.) and [**iNaturalist**](https://www.inaturalist.org/) —
  the public data used to bootstrap a local-species detector before you have your
  own captures. Please honor each dataset's citation terms and each iNaturalist
  observation's CC license / attribution.
- [**Reolink**](https://reolink.com/) cameras and the
  [`reolink-aio`](https://github.com/starkillerOG/reolink_aio) library for the
  native event path.
- [**Cloudflare Tunnel**](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
  (`cloudflared`) — the zero-port-forward remote-access path.

---

## License

This project's own source is **MIT** — see [`LICENSE`](LICENSE).

> **Dependency licenses differ.** The MIT license covers only the code in this
> repository. Notably, the `ultralytics` package is **AGPL-3.0** (not MIT), and a
> model **fine-tuned from Ultralytics YOLO** carries the same AGPL considerations —
> running it as a network-accessible service is the AGPL trigger. The training
> toolchain also uses SpeciesNet and MegaDetector, and the `[audio]` extra pulls
> TensorFlow. You are responsible for complying with the licenses of the model
> weights, datasets, and dependencies you install and distribute.

---

## Status

This is a personal project, shared in the hope it's useful. It's provided
**as-is**, with no warranty and no commitment to support, maintenance, or
reviewing issues/PRs (see the warranty disclaimer in [`LICENSE`](LICENSE)).
Feel free to fork it and adapt it to your own cameras and local species.
