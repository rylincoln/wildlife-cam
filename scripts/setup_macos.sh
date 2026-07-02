#!/usr/bin/env bash
#
# setup_macos.sh — one-shot setup for the Wildlife Detection stack on
# macOS / Apple Silicon. Idempotent: safe to re-run.
#
# Installs / sets up:
#   1. Homebrew ffmpeg          (RTSP / H.265 decode + go2rtc virtual source)
#   2. Python 3.12 venv + the project with the detect/cameras/dev/admin extras
#   3. go2rtc (arm64)           (optional Live view companion)
#   4. ~/wildlife/logs and a config.yaml copied from the example
#   5. Hardware-free unit tests as a smoke check
#
# Env overrides:
#   GO2RTC_VERSION=latest          # or a tag like v1.9.14
#   GO2RTC_BIN_DIR=/opt/homebrew/bin   # where to install the go2rtc binary
#   SKIP_GO2RTC=1                  # skip the go2rtc step entirely
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON_VERSION="3.12"
GO2RTC_VERSION="${GO2RTC_VERSION:-latest}"
GO2RTC_BIN_DIR="${GO2RTC_BIN_DIR:-/opt/homebrew/bin}"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- 0. platform
say "Checking platform"
[ "$(uname -s)" = "Darwin" ] || warn "Not macOS — this script targets macOS/Apple Silicon."
ARCH="$(uname -m)"
[ "$ARCH" = "arm64" ] || warn "Arch is '$ARCH', not arm64 — torch/go2rtc assume Apple Silicon."

# ---------------------------------------------------------------- 1. homebrew
say "Checking Homebrew"
if ! have brew; then
  warn "Homebrew not found. Install it from https://brew.sh, then re-run this script."
  exit 1
fi
echo "brew: $(brew --version | head -1)"

# ---------------------------------------------------------------- 2. ffmpeg
say "Installing ffmpeg (RTSP/HEVC decode + go2rtc virtual test source)"
if have ffmpeg; then
  echo "ffmpeg already present: $(ffmpeg -version | head -1)"
else
  brew install ffmpeg
fi

# ------------------------------------------------------- 3. python venv + pkg
say "Creating Python $PYTHON_VERSION venv and installing the project (.[detect,cameras,dev,admin])"
if have uv; then
  [ -d .venv ] || uv venv --python "$PYTHON_VERSION" .venv
  uv pip install -p .venv -e '.[detect,cameras,dev,admin]'
else
  warn "uv not found (recommended: 'brew install uv'); falling back to python$PYTHON_VERSION + pip."
  if ! have "python$PYTHON_VERSION"; then
    warn "python$PYTHON_VERSION not found — install with 'brew install python@$PYTHON_VERSION' and re-run."
    exit 1
  fi
  [ -d .venv ] || "python$PYTHON_VERSION" -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/pip install -e '.[detect,cameras,dev,admin]'
fi
echo "venv ready: $(./.venv/bin/python --version)"
./.venv/bin/python - <<'PY'
import torch
print(f"torch {torch.__version__}  mps_available={torch.backends.mps.is_available()}")
PY

# ---------------------------------------------------------------- 4. go2rtc
if [ "${SKIP_GO2RTC:-0}" = "1" ]; then
  say "Skipping go2rtc (SKIP_GO2RTC=1)"
else
  say "Installing go2rtc ($GO2RTC_VERSION, arm64) for the optional Live view"
  if [ "$GO2RTC_VERSION" = "latest" ]; then
    GO2RTC_URL="https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_mac_arm64.zip"
  else
    GO2RTC_URL="https://github.com/AlexxIT/go2rtc/releases/download/${GO2RTC_VERSION}/go2rtc_mac_arm64.zip"
  fi
  # Fall back to a repo-local ./bin if the requested dir isn't writable.
  if ! { [ -d "$GO2RTC_BIN_DIR" ] && [ -w "$GO2RTC_BIN_DIR" ]; }; then
    warn "$GO2RTC_BIN_DIR not writable; installing go2rtc to ./bin instead."
    GO2RTC_BIN_DIR="$REPO_DIR/bin"
  fi
  mkdir -p "$GO2RTC_BIN_DIR"
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  curl -fL -o "$TMP/go2rtc.zip" "$GO2RTC_URL"
  unzip -o "$TMP/go2rtc.zip" -d "$TMP" >/dev/null
  mv "$TMP/go2rtc" "$GO2RTC_BIN_DIR/go2rtc"
  chmod +x "$GO2RTC_BIN_DIR/go2rtc"
  # curl-downloaded binaries usually aren't quarantined, but clear it just in case.
  xattr -d com.apple.quarantine "$GO2RTC_BIN_DIR/go2rtc" 2>/dev/null || true
  echo "go2rtc: $("$GO2RTC_BIN_DIR/go2rtc" --version 2>&1 | head -1)  ->  $GO2RTC_BIN_DIR/go2rtc"
  if [ "$GO2RTC_BIN_DIR" != "/opt/homebrew/bin" ]; then
    warn "go2rtc is at $GO2RTC_BIN_DIR/go2rtc — update ProgramArguments in"
    warn "launchd/com.wildlife.stream.plist to point there (it defaults to /opt/homebrew/bin/go2rtc)."
  fi
fi

# ---------------------------------------------------------------- 5. runtime dirs + config
say "Creating ~/wildlife/logs"
mkdir -p "$HOME/wildlife/logs"

say "Config"
if [ -f config.yaml ]; then
  echo "config.yaml already exists — leaving it untouched."
else
  cp config.example.yaml config.yaml
  echo "Created config.yaml from config.example.yaml — EDIT camera host/username/password."
fi

# ---------------------------------------------------------------- 6. smoke check
say "Running hardware-free unit tests"
if ./.venv/bin/python -m pytest -q; then
  TEST_NOTE="unit tests passed"
else
  TEST_NOTE="unit tests FAILED — see output above"
  warn "$TEST_NOTE"
fi

# ---------------------------------------------------------------- done
say "Setup complete ($TEST_NOTE). Next steps:"
cat <<EOF
  1) Edit config.yaml — camera IPs / credentials (and 'livestream.enabled: true' for Live view).
  2) Validate cameras (needs them on your LAN — these are the spec's hard gates):
       .venv/bin/python scripts/test_rtsp.py
       .venv/bin/python scripts/test_events.py
  3) Run the detector + gallery:
       .venv/bin/wildlife-worker
       .venv/bin/wildlife-gallery            # http://<mac-ip>:8080/
  4) Optional Live view:
       .venv/bin/wildlife-stream-config      # writes ./go2rtc.yaml from config.yaml
       go2rtc -config go2rtc.yaml            # or install the com.wildlife.stream LaunchDaemon
EOF
