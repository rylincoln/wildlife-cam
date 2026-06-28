#!/usr/bin/env bash
#
# run_demo.sh — run the whole app in TEST MODE with no real cameras.
#
# Brings up go2rtc against a test video source (re-published as local RTSP),
# starts the gallery with the Live tab enabled, and seeds some synthetic
# captures so the grid has content. Everything lives in ./.demo (gitignored)
# and your real config.yaml is never touched. Ctrl-C cleans everything up.
#
# Usage:
#   ./scripts/run_demo.sh                      # virtual test pattern, port 8080, 12 captures
#   ./scripts/run_demo.sh --source mux         # real public video feed in the Live tab
#   ./scripts/run_demo.sh --port 8090 --seed 0 # custom port, no seeded captures
#   ./scripts/run_demo.sh --detect             # also run a one-shot capture->YOLO on the feed
#   ./scripts/run_demo.sh --source 'ffmpeg:https://host/stream.m3u8#video=h264'  # any go2rtc source
#
# Flags:
#   --source NAME|URL  virtual (default) | mux | any go2rtc source string
#   --port N           gallery port (default 8080)
#   --seed N           synthetic gallery captures to seed (default 12)
#   --detect           after startup, grab a frame off the feed and run YOLO once
#   -h, --help         show this help
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PATH="/opt/homebrew/bin:$PATH"

SOURCE="virtual"; PORT="8080"; SEED="12"; DETECT="0"
while [ $# -gt 0 ]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --port)   PORT="$2";   shift 2 ;;
    --seed)   SEED="$2";   shift 2 ;;
    --detect) DETECT="1";  shift ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown flag: $1 (try --help)"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }

VENV_PY="$REPO_DIR/.venv/bin/python"
[ -x "$VENV_PY" ] || { warn "No .venv found — run ./scripts/setup_macos.sh first."; exit 1; }

# Resolve the requested source into a go2rtc source string.
case "$SOURCE" in
  virtual) SRC='ffmpeg:virtual?video&size=1280x720#video=h264' ;;   # built-in moving test pattern (no network)
  mux)     SRC='ffmpeg:https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8#video=h264' ;;  # public test video
  *)       SRC="$SOURCE" ;;                                          # pass any go2rtc source through
esac

# Find go2rtc; degrade to gallery-only if it isn't installed.
GO2RTC_BIN="$(command -v go2rtc 2>/dev/null || true)"
[ -z "$GO2RTC_BIN" ] && [ -x "$REPO_DIR/bin/go2rtc" ] && GO2RTC_BIN="$REPO_DIR/bin/go2rtc"
if [ -n "$GO2RTC_BIN" ]; then LIVE="1"; else
  LIVE="0"; warn "go2rtc not found — starting gallery only (no Live tab). Install it with scripts/setup_macos.sh."
fi

DEMO="$REPO_DIR/.demo"
rm -rf "$DEMO"; mkdir -p "$DEMO"

GO2RTC_PID=""; GALLERY_PID=""; CLEANED="0"
cleanup() {
  [ "$CLEANED" = "1" ] && return; CLEANED="1"
  printf '\n'; say "Shutting down test mode"
  [ -n "$GALLERY_PID" ] && kill "$GALLERY_PID" 2>/dev/null || true
  [ -n "$GO2RTC_PID" ]  && kill "$GO2RTC_PID"  2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Generate .demo/config.yaml + .demo/go2rtc.yaml and seed synthetic captures.
say "Building demo config (source=$SOURCE, port=$PORT, seed=$SEED, live=$LIVE)"
"$VENV_PY" - "$DEMO" "$SRC" "$PORT" "$SEED" "$LIVE" <<'PY'
import sys, os, yaml, datetime as dt
import numpy as np
demo, src, port, seed, live = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
repo = os.path.dirname(demo)
raw = yaml.safe_load(open(os.path.join(repo, "config.example.yaml")))

cams = []
for cid in ("north_field", "south_trail"):
    cams.append({
        "id": cid, "host": "127.0.0.1", "username": "x", "password": "x",
        "rtsp_main": f"rtsp://127.0.0.1:8554/{cid}_main",
        "rtsp_sub":  f"rtsp://127.0.0.1:8554/{cid}_sub",
        "onvif_port": 8000,
    })
raw["cameras"] = cams
raw["gallery"]["host"] = "0.0.0.0"
raw["gallery"]["port"] = port
raw["storage"]["captures_dir"] = os.path.join(demo, "captures")
raw["storage"]["db_path"] = os.path.join(demo, "captures.db")
raw["capture"]["rtsp_timeout_s"] = 20
raw["livestream"]["enabled"] = live
raw["livestream"]["go2rtc_url"] = None  # derive from the browser's request host
yaml.safe_dump(raw, open(os.path.join(demo, "config.yaml"), "w"))

streams = {}
for cid in ("north_field", "south_trail"):
    streams[f"{cid}_main"] = src
    streams[f"{cid}_sub"] = src
g2 = {"streams": streams, "api": {"listen": ":1984"},
      "rtsp": {"listen": ":8554"}, "webrtc": {"listen": ":8555"},
      "log": {"level": "info"}}
yaml.safe_dump(g2, open(os.path.join(demo, "go2rtc.yaml"), "w"), sort_keys=False)

if seed > 0:
    sys.path.insert(0, os.path.join(repo, "src"))
    from wildlife.config import load_config
    from wildlife.store import Store
    from wildlife.models import Detection
    cfg = load_config(os.path.join(demo, "config.yaml"))
    st = Store(cfg.storage.db_path, cfg.storage.captures_dir,
               cfg.storage.jpeg_quality, cfg.storage.thumbnail_px)
    st.init_schema()
    labels = ["deer", "bird", "fox", "raccoon", "cat", "dog"]
    colors = [(40,160,255),(80,220,120),(200,120,255),(255,180,60),(120,200,255),(255,120,120)]
    now = dt.datetime.now()
    for i in range(seed):
        f = np.zeros((180,240,3), np.uint8); f[:] = (24,28,34); f[40:140,50:190] = colors[i % len(colors)]
        ts = now - dt.timedelta(minutes=7*i)
        st.save_capture(camera_id=cams[i % 2]["id"], event_ts=ts, capture_ts=ts, frame=f,
                        det=Detection(labels[i % len(labels)], round(0.55+0.05*(i % 5), 2), (50,40,190,140), 0.27))
    st.close()
print(f"wrote {demo}/config.yaml + go2rtc.yaml; seeded {seed} capture(s)")
PY

# Start go2rtc (if available).
if [ "$LIVE" = "1" ]; then
  say "Starting go2rtc ($SOURCE)"
  "$GO2RTC_BIN" -config "$DEMO/go2rtc.yaml" > "$DEMO/go2rtc.log" 2>&1 &
  GO2RTC_PID=$!
  for _ in $(seq 1 20); do curl -fsS -o /dev/null http://127.0.0.1:1984/api 2>/dev/null && break; sleep 0.3; done
  echo "go2rtc up (pid $GO2RTC_PID) — dashboard http://localhost:1984"
fi

# Optional one-shot capture -> YOLO on the feed.
if [ "$DETECT" = "1" ] && [ "$LIVE" = "1" ]; then
  say "One-shot capture -> YOLO on the feed (needs content with COCO objects to show hits)"
  "$VENV_PY" - "$DEMO" <<'PY'
import sys
from wildlife.config import load_config
from wildlife.capture import grab_burst
from wildlife.detect import Detector
cfg = load_config(f"{sys.argv[1]}/config.yaml")
frames = grab_burst(cfg.cameras[0], 1, 0, "main", cfg.capture.rtsp_timeout_s)
print(f"grabbed {len(frames)} frame(s); shape={frames[0].shape if frames else '-'}")
if frames:
    det = Detector(cfg.detection.model_path, cfg.detection.device)
    print("device_in_use:", det.device_in_use())
    ds = det.infer(frames[0])
    print(f"detections ({len(ds)}):", ", ".join(f"{d.label}({d.confidence:.2f})" for d in ds) or "(none in this frame)")
PY
fi

say "Test mode ready"
HOSTHINT="localhost"
echo "  Gallery:        http://$HOSTHINT:$PORT/"
[ "$LIVE" = "1" ] && echo "  Live view:      http://$HOSTHINT:$PORT/live"
[ "$LIVE" = "1" ] && echo "  go2rtc console: http://$HOSTHINT:1984/"
echo "  (LAN devices: swap 'localhost' for this Mac's IP)"
echo "  Press Ctrl-C to stop everything."
echo

# Gallery in the background + wait: a signal interrupts the wait so cleanup()
# runs and reaps go2rtc too (a foreground child would block the trap).
"$REPO_DIR/.venv/bin/wildlife-gallery" "$DEMO/config.yaml" &
GALLERY_PID=$!
wait "$GALLERY_PID" || true
