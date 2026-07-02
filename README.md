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
grid, newest first, with filters for camera, date range, class, and minimum
confidence; click a thumbnail to load the full image.

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
  both JPEG files (full + thumbnail), mirroring `scripts/prune.py`. There is no
  undo. Deleting is the disposal path for false positives.
- **Bulk**: tick the checkboxes to delete, reclassify, or mark-reviewed many at
  once. Selection applies to the current page.
- A **"Mark reviewed"** action and the **Unreviewed** filter let you work
  through a backlog without re-seeing handled captures.

Reviewed captures are exempt from retention's `min_confidence_keep` rule (a
human-confirmed low-confidence capture is not auto-pruned); they are still
subject to `max_age_days`, so export anything you want to keep permanently.

---

## Remote access (Cloudflare Tunnel)

Reach the gallery + live view from anywhere at one URL (e.g. `https://cam.rlblais.org`),
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
> restarts services, and prints the remaining dashboard/camera steps with values filled in.

1. `brew install cloudflared` (keep current: `brew upgrade cloudflared`).
2. Cloudflare dashboard → Zero Trust → Networks → Tunnels → Create → Cloudflared → name
   it; copy the token; `sudo cloudflared service install <TOKEN>` (boot daemon).
3. Add two **public hostnames** on that tunnel (order matters — the `/go2rtc` one first):
   - `cam` . `rlblais.org`, **Path** `go2rtc` → HTTP `localhost:1984`
   - `cam` . `rlblais.org`, (no path) → HTTP `localhost:8080`
4. In `config.yaml` set `remote.base_url: "https://cam.rlblais.org"` and
   `livestream.base_path: "/go2rtc"`; regenerate go2rtc config
   (`wildlife-stream-config`) and restart go2rtc; the gallery picks up `config.yaml` changes automatically.
5. Run `wildlife-share-secret` → note the printed **share link** and **raw secret**.
6. Add a Cloudflare **WAF custom rule**: *Block* when the path starts with `/go2rtc` and
   the `wl_key` cookie ≠ the raw secret. (Free-plan fallback: a small Cloudflare Worker on
   `cam.rlblais.org/go2rtc/*` that checks the `wl_key` cookie.)
7. Set each camera's **sub-stream to H.264 Main/Baseline** so MSE live plays in every
   browser incl. Safari/iOS.
8. Keep **Total TLS OFF** for the zone so `cam.` stays out of Certificate Transparency logs.

### Using / rotating

- Share `https://cam.rlblais.org/?key=<secret>`; the recipient's browser stores the
  cookie so the key only appears once.
- **Rotate / revoke everyone:** re-run `wildlife-share-secret` and update the WAF rule
  with the new secret. Old links stop working.

> **Security note.** This is view-only "anyone with the link" access: a forwarded link
> grants access until you rotate. `/admin` is never exposed remotely. See
> `docs/superpowers/specs/2026-07-02-remote-access-cloudflare-tunnel-design.md` for the
> full threat model and the go2rtc-exposure trade-off.

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

The stock COCO model only names **bear** and generic **bird** among local
species — mule deer, elk, cougar, coyote, fox, turkey, etc. are not COCO classes,
so `detection.animal_classes` can't filter for what the model never predicts. The
[`training/`](training/README.md) toolchain fine-tunes a detector on
southwest-Colorado species and exports it to Core ML; you then just point
`detection.model_path` at the `.mlpackage` and match `animal_classes` (no code
change). The recommended loop — auto-label with **SpeciesNet**, two-stage YOLO
fine-tune, Core ML export — plus which public datasets to bootstrap from before
you have your own captures, is in [`training/README.md`](training/README.md).

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
- `training/` — offline toolchain to fine-tune a local-species detector (auto-label
  → split → train → Core ML export). See [`training/README.md`](training/README.md).
- `models/` — YOLO weights (auto-downloaded on first run; gitignored). See
  [`models/README.md`](models/README.md).
- `launchd/` — example LaunchDaemon plists.
- `tests/` — hardware-free unit tests.

For full details, read [`spec.md`](spec.md).

---

## Acknowledgements

This project stands on some excellent open-source work:

- [**go2rtc**](https://github.com/AlexxIT/go2rtc) — the on-demand WebRTC/MSE
  restreamer behind the optional Live view.
- [**Ultralytics YOLO**](https://github.com/ultralytics/ultralytics) — the object
  detector (stock COCO model and the fine-tuning toolchain).
- [**SpeciesNet**](https://github.com/google/cameratrapai) — used to auto-label
  training images with species predictions.
- [**LILA BC**](https://lila.science/) camera-trap datasets (ENA24, Caltech
  Camera Traps, Idaho, etc.) — the public data used to bootstrap a local-species
  detector before you have your own captures. Please honor each dataset's
  citation and attribution terms.
- [**Reolink**](https://reolink.com/) cameras and the
  [`reolink-aio`](https://github.com/starkillerOG/reolink_aio) library for the
  native event path.

---

## License

This project's own source is **MIT** — see [`LICENSE`](LICENSE).

> **Dependency licenses differ.** The MIT license covers only the code in this
> repository. Notably, the stock YOLO model and the `ultralytics` package are
> **AGPL-3.0** (not MIT), and the training toolchain also uses SpeciesNet. You
> are responsible for complying with the licenses of the model weights and
> dependencies you install and distribute — especially if you run this as a
> network-accessible service or redistribute it.

---

## Status

This is a personal project, shared in the hope it's useful. It's provided
**as-is**, with no warranty and no commitment to support, maintenance, or
reviewing issues/PRs (see the warranty disclaimer in [`LICENSE`](LICENSE)).
Feel free to fork it and adapt it to your own cameras and local species.
