#!/usr/bin/env bash
#
# install_review_agent.sh — install the daily Claude-vision review job as a
# LaunchAgent for the current user (no sudo).
#
# Unlike the detector/gallery/stream daemons (install_launchd.sh, root), the
# review runs in the user's login session so the `claude` CLI can read its
# login from the user's keychain. Copies launchd/com.wildlife.review.plist into
# ~/Library/LaunchAgents with the __USER__ placeholders filled in, and (re)loads
# it. Idempotent.
#
# Usage:
#   bash scripts/install_review_agent.sh            # install / update
#   bash scripts/install_review_agent.sh --remove   # unload and delete
#
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this as the user whose claude login the review should use, not root."
  exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL=com.wildlife.review
DST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

launchctl bootout "$DOMAIN" "$DST" 2>/dev/null || true
if [ "${1:-}" = "--remove" ]; then
  rm -f "$DST"
  echo "Removed $LABEL."
  exit 0
fi

install -d "$HOME/Library/LaunchAgents" "$HOME/wildlife/logs"
sed -e "s#/Users/__USER__#${HOME}#g" -e "s#__USER__#$(id -un)#g" \
  "$REPO/launchd/$LABEL.plist" > "$DST"
chmod 644 "$DST"
plutil -lint "$DST" >/dev/null
launchctl bootstrap "$DOMAIN" "$DST"
launchctl enable "$DOMAIN/$LABEL"

echo "Installed $LABEL (daily 09:00; a no-op until review.enabled is true)."
echo "Run now:  launchctl kickstart $DOMAIN/$LABEL"
echo "Logs:     $HOME/wildlife/logs/review.{out,err}.log"
