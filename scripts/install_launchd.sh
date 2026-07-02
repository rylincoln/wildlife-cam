#!/usr/bin/env bash
#
# install_launchd.sh — install the wildlife-cam LaunchDaemons for 24/7 operation.
#
# Copies launchd/com.wildlife.{detect,gallery,stream}.plist into
# /Library/LaunchDaemons, sets root:wheel 0644, and (re)loads them via launchctl
# so they start now (RunAtLoad) and at every boot, relaunching on crash
# (KeepAlive). Idempotent: safe to re-run after editing a plist or config.
#
# The plists contain absolute paths and a UserName — edit them for your machine
# first (see the header comment in each). This script only copies + loads them;
# it derives the repo root from its own location and the deploy user from the
# sudo invoker (for the log directory), but does NOT rewrite the plist contents.
#
# Usage:
#   sudo bash scripts/install_launchd.sh
#
# Manage afterwards (all need sudo):
#   sudo launchctl bootout system /Library/LaunchDaemons/com.wildlife.detect.plist   # stop+unload one
#   sudo launchctl kickstart -k system/com.wildlife.detect                           # restart one
#   sudo launchctl list | grep com.wildlife                                          # status
#
set -uo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "This must run as root. Re-run:  sudo bash $0"
  exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST=/Library/LaunchDaemons

# Deploy user = whoever invoked sudo (falls back to the repo owner), used only to
# create the log directory with the right ownership. The daemons themselves run
# as the UserName set inside each plist.
DEPLOY_USER="${SUDO_USER:-$(stat -f%Su "$REPO")}"
DEPLOY_HOME="$(eval echo "~${DEPLOY_USER}")"
install -d -o "$DEPLOY_USER" -g staff "$DEPLOY_HOME/wildlife/logs"

rc=0
for L in detect gallery stream; do
  SRC="$REPO/launchd/com.wildlife.$L.plist"
  DST="$DEST/com.wildlife.$L.plist"
  printf '\n==> com.wildlife.%s\n' "$L"
  if [ ! -f "$SRC" ]; then echo "    MISSING source: $SRC"; rc=1; continue; fi
  cp "$SRC" "$DST" && chown root:wheel "$DST" && chmod 644 "$DST" \
    || { echo "    copy failed"; rc=1; continue; }
  # Unload any prior instance so bootstrap doesn't error on "already loaded".
  launchctl bootout system "$DST" 2>/dev/null
  if launchctl bootstrap system "$DST" 2>/tmp/wl_boot.err; then
    launchctl enable "system/com.wildlife.$L" 2>/dev/null
    echo "    bootstrapped + enabled (RunAtLoad -> starting now)"
  elif launchctl load -w "$DST" 2>/dev/null; then
    echo "    loaded (legacy launchctl load)"
  else
    echo "    FAILED: $(cat /tmp/wl_boot.err 2>/dev/null)"; rc=1
  fi
done

printf '\n==> Loaded wildlife daemons (PID / laststatus / label):\n'
launchctl list | grep com.wildlife || echo "   (none listed — check errors above)"
printf '\nLogs: %s/wildlife/logs/{detect,gallery,stream}.{out,err}.log\n' "$DEPLOY_HOME"
exit $rc
