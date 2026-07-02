#!/usr/bin/env bash
#
# setup_remote.sh -- host-side setup for Cloudflare Tunnel remote access, run on
# the PRODUCTION Mac mini from the repo root (NOT the dev machine). Idempotent
# where practical. Pairs with a few Cloudflare-dashboard + camera steps it prints
# at the end with values filled in.
#
# Automates: install/upgrade cloudflared; connect the tunnel as a boot daemon
# from your dashboard token; set config.yaml (remote.base_url, livestream.base_path);
# mint the share secret; regenerate go2rtc.yaml; restart go2rtc (the stream service).
#
# Note: re-running rotates the share secret and invalidates previously-shared links.
#
# Usage:
#   HOST=cam.rlblais.org CF_TUNNEL_TOKEN=eyJ... ./scripts/setup_remote.sh
#   HOST=cam.rlblais.org ./scripts/setup_remote.sh        # host config only; prints tunnel steps
#   ./scripts/setup_remote.sh --dry-run                   # echo actions, change nothing
#
set -euo pipefail

HOST="${HOST:-cam.rlblais.org}"
CONFIG="${CONFIG:-config.yaml}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

run() { echo "+ $*"; [[ "$DRY_RUN" == "1" ]] || "$@"; }
note() { printf '\n\033[1m%s\033[0m\n' "$*"; }

[[ "$(uname)" == "Darwin" ]] || { echo "This script is for macOS (the prod Mac mini)." >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "No $CONFIG here -- run from the repo root on prod." >&2; exit 1; }
command -v brew >/dev/null || { echo "Homebrew required (https://brew.sh)." >&2; exit 1; }
VENV="./.venv/bin"
[[ -x "$VENV/python" ]] || { echo "No ./.venv -- install the project first (see README)." >&2; exit 1; }

# 1. cloudflared
if command -v cloudflared >/dev/null; then
  run brew upgrade cloudflared || true
else
  run brew install cloudflared
fi

# 2. Connect the tunnel as a boot daemon (token from the dashboard tunnel).
if [[ -z "${CF_TUNNEL_TOKEN:-}" ]]; then
  note "No CF_TUNNEL_TOKEN set -- skipping the tunnel service install."
  echo "Create a tunnel in the Cloudflare dashboard (Zero Trust -> Networks -> Tunnels),"
  echo "copy its connector token, then re-run: CF_TUNNEL_TOKEN=... ./scripts/setup_remote.sh"
else
  echo "+ sudo cloudflared service install <token redacted>"
  [[ "$DRY_RUN" == "1" ]] || sudo cloudflared service install "$CF_TUNNEL_TOKEN"
fi

# 3. Wildlife config: public URL + go2rtc sub-path (comment-preserving write).
note "Updating $CONFIG (remote.base_url, livestream.base_path)"
run "$VENV/python" - "$CONFIG" "$HOST" <<'PY'
import sys
from wildlife.admin.config_io import update_sections
update_sections(sys.argv[1], {
    "remote": {"base_url": f"https://{sys.argv[2]}"},
    "livestream": {"base_path": "/go2rtc"},
})
print("config updated")
PY

# 4. Mint the share secret (prints the share link + the raw secret).
note "Minting the share secret"
run "$VENV/wildlife-share-secret" "$CONFIG"

# 5. Regenerate go2rtc.yaml (adds api.base_path) + restart services.
run "$VENV/wildlife-stream-config" "$CONFIG" go2rtc.yaml
run sudo launchctl kickstart -k system/com.wildlife.stream || true

# 6. Finish in the Cloudflare dashboard / on the cameras.
note "ALMOST DONE -- finish these (values filled in for you):"
cat <<EOF

A) Tunnel public hostnames (your tunnel -> Published routes) -- add TWO, /go2rtc FIRST:
     1. Subdomain "${HOST%%.*}", Domain "${HOST#*.}", Path "go2rtc" -> HTTP  localhost:1984
     2. Subdomain "${HOST%%.*}", Domain "${HOST#*.}", (no Path)     -> HTTP  localhost:8080

B) WAF custom rule (Security -> WAF -> Custom rules -> Create):
     Expression:  (starts_with(http.request.uri.path, "/go2rtc") and not http.request.cookies["wl_key"][0] eq "<PASTE THE RAW SECRET PRINTED ABOVE>")
     Action:      Block
   (Free-plan fallback: a Worker on ${HOST}/go2rtc/* that checks the wl_key cookie.)

C) Cameras: set each sub-stream to H.264 Main or Baseline profile.

D) SSL/TLS -> Edge Certificates: confirm Total TLS is OFF (keeps ${HOST} out of CT logs).

Then, from OFF your LAN, open the printed share link and verify the gallery + live view.
EOF
