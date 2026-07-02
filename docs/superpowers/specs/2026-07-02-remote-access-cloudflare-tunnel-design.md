# Remote access via Cloudflare Tunnel — design

**Date:** 2026-07-02
**Status:** Draft for review
**Domain:** `rlblais.org` (managed in Cloudflare; user's account)

## 1. Goal

Reach the wildlife-cam system from the public internet without port-forwarding or
exposing the Mac mini's IP, using a **Cloudflare Tunnel** on the **free plan**. Three
surfaces, three different trust levels ("layered"):

| Surface | Who can reach it remotely | Mechanism |
|---|---|---|
| Gallery + `/live` page (photos, filters, the live *page*) | **Anyone with the secret link** | Capability URL (token) enforced by the Flask app, tunnel traffic only |
| Live **video** tiles (go2rtc streams) | **You + anyone you invite by email** | Cloudflare Access (email one-time-PIN) on a second hostname |
| `/admin` (config editor, daemon restart) | **You only** | Cloudflare Access at the edge + existing Basic Auth on the LAN |

This layering was chosen because the three surfaces carry very different risk: the
admin can rewrite config and restart services; the live video is the only piece that
brushes against Cloudflare's video-serving ToS; the JPEG gallery is low-risk and the
thing you actually want to share.

### Non-goals

- **Shareable-by-link live video.** Deliberately out of scope for v1 — see §11. Making
  live "anyone with the link" requires a reverse proxy (Caddy) in front of go2rtc,
  because go2rtc has no cookie auth of its own, and it is the higher-risk config under
  Cloudflare's video ToS. v1 gates live behind Access instead. This decision is
  **reversible**; the Flask/tunnel work here doesn't block it.
- **Changing LAN behavior.** On the LAN the gallery stays open and `/admin` stays
  Basic-Auth-gated exactly as today. All new gating applies *only* to tunnel traffic.
- **Replacing go2rtc / recording video.** Untouched.
- **Multi-user capability tokens, single-use tokens, self-serve token UI.** One
  rotatable token for v1 (§6).

## 2. Decisions locked with the user

1. **Access model:** Layered/Hybrid (recommended option), confirmed.
2. **Live view:** In scope for v1.
3. **Tunnel style:** Remote-managed (dashboard token) named tunnel, installed as a
   boot **launch daemon**.
4. **Live-view auth (Option A):** Gated by Cloudflare Access (you + invited), *not*
   shareable-by-link. Chosen as the default per "go with recommendation"; flagged
   reversible. **← confirm on review.**

## 3. Verified constraints that shape the design

From research (high confidence; sources in the research transcript):

1. **WebRTC cannot cross a Cloudflare Tunnel.** The tunnel carries only HTTP/HTTPS +
   WebSocket; it never forwards go2rtc's DTLS/SRTP media (UDP/TCP 8555). Remote live
   view **must** use go2rtc's **MSE** transport (fMP4 over a WebSocket at
   `/api/ws`), which Cloudflare proxies natively on all plans. We must **force
   `mode=mse`** for remote — leaving `webrtc,mse` makes the player try WebRTC first
   and hang for several seconds on ICE timeout before falling back.
2. **go2rtc's API + MSE WebSocket share one origin/port** (default `:1984`). The whole
   `:1984` origin must be routed through one tunnel hostname so the page and `/api/ws`
   are same-origin.
3. **Codec/browser matrix for remote MSE** (Reolink-specific, already noted in README):
   the 4K **main** is HEVC → MSE plays it in **Safari** and HEVC-capable Chrome only;
   the **sub** is H.264 *High profile* → Safari's MSE rejects it (bad codec string)
   but Chrome/Firefox MSE decode it. Since WebRTC (the sub's LAN transport) is
   unavailable remotely, **no single stream plays MSE in every browser** without a
   camera-side change. See §7.3 for mitigations.
4. **Video ToS still exists** (old §2.8 moved into the CDN terms). A tunnel hostname is
   always proxied — you cannot grey-cloud it. For a personal cam the JPEG gallery is
   low risk and *occasional, authenticated* live viewing is low practical risk (worst
   realistic case is a per-stream "video restricted" interstitial, not a ban).
   **Gating live behind Access keeps it a private authenticated service** — the safest
   posture — which is a second reason for Option A.
5. **Certificate Transparency:** issuing a TLS cert normally publishes the hostname to
   public CT logs. **But** Cloudflare Universal SSL is a `*.rlblais.org` **wildcard**,
   so first-level names like `cam.rlblais.org` / `stream.rlblais.org` are **not**
   individually written to CT (only `rlblais.org` and `*.rlblais.org` appear) — as long
   as **Total TLS / per-hostname certs are NOT enabled**. Obscurity is a bonus, never
   the control.
6. **Cloudflare Access on the free plan** covers up to 50 users, supports email
   One-Time-PIN with no external IdP, and can protect **only `/admin`** by creating a
   self-hosted app scoped to that path (more-specific path wins; no app on the bare
   hostname ⇒ the rest stays open). Access injects a signed `Cf-Access-Jwt-Assertion`
   header the origin can verify (JWKS by `kid`, check `aud`/`iss`/`exp`).

## 4. Architecture

```
                         Internet
                            │  (HTTPS only; home IP never exposed)
                   ┌────────┴─────────┐
                   │  Cloudflare edge  │
                   │  + Access (OTP)   │
                   └────────┬─────────┘
             cam.rlblais.org│ stream.rlblais.org
                   ┌────────┴─────────┐
                   │   cloudflared     │  (launchd daemon on the Mac mini)
                   │  named tunnel     │
                   └───┬──────────┬────┘
        localhost:8080 │          │ localhost:1984
            ┌──────────┴───┐  ┌───┴──────────┐
            │ Flask gallery │  │   go2rtc     │
            │  / , /live    │  │  MSE streams │
            │  /admin       │  └──────────────┘
            └──────────────┘
```

- **One named tunnel** with **two public hostnames**:
  - `cam.rlblais.org` → `http://localhost:8080` (Flask gallery)
  - `stream.rlblais.org` → `http://localhost:1984` (go2rtc)
- **Two Cloudflare Access apps** (both allow only your email(s) via OTP):
  - App 1: `cam.rlblais.org/admin` (path-scoped; leaves the rest of `cam.` open)
  - App 2: `stream.rlblais.org` (whole hostname; gates all live video)
- **Flask app** enforces the **capability link** on tunnel traffic to `cam.rlblais.org`
  (gallery + `/live` page), and accepts a verified Access JWT as an alternative to
  Basic Auth on `/admin`.

### 4.1 How the three gates compose (LAN vs remote)

The same Flask process serves both LAN clients (real private IPs) and cloudflared
(which connects from `127.0.0.1`). We distinguish them by **`request.remote_addr`
being loopback** — robust because the only things that reach `:8080` are LAN clients
and cloudflared, and a remote attacker cannot make `remote_addr` a LAN IP (they can't
reach `:8080` directly at all; only cloudflared is exposed).

| Route | LAN (`remote_addr` = private IP) | Remote (`remote_addr` = 127.0.0.1, via tunnel) |
|---|---|---|
| `/`, `/live`, images, `/api/captures` | Open (unchanged) | **Capability token required** |
| `/admin/*` | Basic Auth (unchanged) | **Cloudflare Access** at edge; app accepts verified Access JWT (no Basic prompt) |
| live video (`stream.` host) | go2rtc open on LAN (unchanged) | **Cloudflare Access** at edge |

## 5. Cloudflare-side setup (runbook, no repo code)

Done once in the Cloudflare dashboard + on the Mac. Captured here so implementation
can script/document it; values (tunnel token, Access AUD tags, team domain) feed the
`config.yaml` additions in §6.

1. **Install:** `brew install cloudflared` (arm64 auto-selected).
2. **Create tunnel (remote-managed):** Zero Trust → Networks → Tunnels → Create →
   Cloudflared → name `wildlife`. Copy the connector **token**.
3. **Install as boot daemon:** `sudo cloudflared service install <TOKEN>` → launchd
   `com.cloudflare.cloudflared`, starts at boot. (Prefer `--token-file` / `TUNNEL_TOKEN`
   over pasting the token into shell history.)
4. **Add public hostnames** (Routes → Add public hostname), which auto-creates proxied
   CNAMEs:
   - `cam` . `rlblais.org` → HTTP `localhost:8080`
   - `stream` . `rlblais.org` → HTTP `localhost:1984`
5. **Zero Trust → Settings → Authentication:** add **One-time PIN** login method.
6. **Access app 1 (admin):** Access → Applications → Add → Self-hosted. Domain
   `cam.rlblais.org`, **Path `admin`**. Session e.g. 24h. Login method: OTP only.
   Policy: Allow, Include → Emails → `rylincoln@gmail.com` (+ any invitees). **Record
   the Application AUD tag.**
7. **Access app 2 (stream):** same, Domain `stream.rlblais.org`, **no path** (whole
   host). Same allow-policy. **Record its AUD tag** (only needed if we later verify it
   at origin; go2rtc itself won't).
8. **Confirm CT posture:** ensure **Total TLS is OFF** for the zone so first-level
   labels stay out of CT (§3.5). Optionally enable CT **monitoring** alerts.
9. **Keep current:** `brew upgrade cloudflared` periodically (Cloudflare supports
   ~1 year of releases).

## 6. Flask / repo changes

### 6.1 New `remote` config block

```yaml
remote:
  enabled: false                 # off by default; turning on activates the capability gate
  base_url: "https://cam.rlblais.org"     # canonical public URL (for building share links)
  share_token_hash: null         # Werkzeug hash of the capability token; set via wildlife-share-token
  # Live view over the tunnel:
  stream_base_url: "https://stream.rlblais.org"  # public go2rtc origin for remote /live embeds
  # Admin over the tunnel (Cloudflare Access JWT verification):
  access_team_domain: null       # e.g. "https://<team>.cloudflareaccess.com"
  access_admin_aud: null         # AUD tag of the /admin Access app
```

New pydantic `RemoteConfig` model (mirrors `AdminConfig`/`LivestreamConfig` style),
added to `Config` with a `default_factory` so existing configs keep working untouched.
When `remote.enabled` is false, **nothing changes** — the app behaves exactly as today.

### 6.2 Capability-link gate (gallery)

A `before_request` hook installed in `create_app` (`gallery/app.py`), applied to
gallery routes (not `/admin`, not `/static`, not the token-entry route):

- **Scope:** only when `remote.enabled` **and** `request.remote_addr` is loopback
  (tunnel traffic). LAN requests skip the gate entirely.
- **Entry:** `GET /s/<token>` — path, not query (lower log/Referer leakage). Validate
  with **constant-time** compare against `remote.share_token_hash`. On success set an
  `HttpOnly; Secure; SameSite=Lax` cookie carrying the token, then **302 to `/`** so
  the token leaves the address bar/history/Referer.
- **Per-request check:** valid cookie ⇒ allow; else `404` (not `401` — don't advertise
  that a gate exists).
- **Hardening:** `Referrer-Policy: no-referrer` on all responses (via `after_request`);
  simple in-memory **rate-limit** on `/s/<token>` keyed by `Cf-Connecting-IP`
  (256-bit token makes brute force infeasible, but cheap defense). Token is **reusable
  + rotatable** (not single-use), so chat link-preview unfurlers can't burn it.
- **Revocation/rotation:** re-run `wildlife-share-token` → new hash → old links dead.

New CLI **`wildlife-share-token`** (`wildlife.remote.share_token:main`, entry in
`[project.scripts]`): generate `secrets.token_urlsafe(32)` (256-bit), write its hash to
`config.yaml` (reusing the admin config-writing path / atomic write), and print the
full share URL `${base_url}/s/<token>` **once**.

### 6.3 Request-aware live embeds (`/live`)

`_live_base()` and the per-tile `mode` are made **request-aware** so LAN keeps its
low-latency WebRTC while remote uses MSE:

- **Remote (loopback `remote_addr`, or host == the public cam host):**
  base = `remote.stream_base_url` (`https://stream.rlblais.org`), **`mode=mse`** forced
  for both tiles. This fixes the current hardcoded `http://<host>:1984` (mixed-content)
  and the WebRTC-won't-traverse problem in one place.
- **LAN:** unchanged — derives `http://<host>:1984`, keeps configured `sub_mode`/
  `main_mode`.

No change to go2rtc or the generated `go2rtc.yaml` is required for this (the existing
`:1984` api is what cloudflared points at). The `livestream.*_mode` config defaults
stay as-is for LAN.

### 6.4 Admin: accept a verified Access JWT

`check_admin_auth` (`admin/auth.py`) gains a first check: if a
**`Cf-Access-Jwt-Assertion`** header is present and **verifies** (fetch JWKS from
`${access_team_domain}/cdn-cgi/access/certs`, select key by `kid`, RS256, check
`aud == access_admin_aud`, `iss == access_team_domain`, `exp`/`iat`) ⇒ authorized (skip
Basic Auth). Else fall back to the existing Basic Auth. Result:

- **Remote `/admin`:** edge Access blocks unauthenticated requests before they reach
  the app; authenticated requests arrive with a valid JWT ⇒ no second Basic-Auth
  prompt. A forged header fails signature verification (safe against a LAN attacker
  hitting `:8080` directly).
- **LAN `/admin`:** no JWT ⇒ Basic Auth as today. Fully backward compatible when
  `remote.access_*` is unset.

New deps in the **`admin`** extra: `pyjwt>=2.8`, `cryptography`. JWKS cached with a
short TTL; keys matched by `kid` (they rotate ~6-weekly).

### 6.5 Files touched (estimate)

- `src/wildlife/config.py` — add `RemoteConfig`, wire into `Config`.
- `src/wildlife/gallery/app.py` — capability `before_request`/`after_request`,
  `/s/<token>` route, request-aware `_live_base()` + mode.
- `src/wildlife/admin/auth.py` — Access-JWT path.
- `src/wildlife/remote/` (new) — `share_token.py` (CLI), `access_jwt.py` (verify),
  `capability.py` (gate helpers).
- `pyproject.toml` — `wildlife-share-token` script; `pyjwt`/`cryptography` in `admin`.
- `config.example.yaml` — commented `remote:` block.
- `README.md` — "Remote access (Cloudflare Tunnel)" section + the §5 runbook.

## 7. Failure modes & error handling

1. **Tunnel down / cloudflared not running:** Cloudflare serves its own error page;
   nothing to handle in-app. Runbook covers `launchctl` / `brew services` + log paths.
2. **`remote.enabled` but no `share_token_hash`:** fail **closed** for tunnel traffic
   (all gallery routes `404` remotely) with a log line telling the operator to run
   `wildlife-share-token` — mirrors the admin "fail closed" posture. LAN unaffected.
3. **Bad/expired token:** `404` (no oracle). Rate-limited.
4. **Access JWKS unreachable / JWT invalid:** admin JWT path fails → falls back to
   Basic Auth (so remote admin still works if you know the password); log the JWKS
   error. Never fail *open*.
5. **Live tile won't play remotely (codec):** see §7.3 — surface a short "your browser
   may not support this stream remotely; try Safari, or the other tile" note on `/live`
   when remote.
6. **Unauthenticated link-holder opens `/live` remotely:** the go2rtc iframe targets the
   Access-gated `stream.rlblais.org`, so for a viewer who is *not* logged into Access the
   iframe renders the Cloudflare login (which may frame-bust to blank). This is expected
   under Option A (live = you-only). The `/live` template should show a "live view
   requires login" hint when the viewer lacks an Access session, so blank tiles aren't a
   mystery. For **you**, one OTP login covers both Access apps via SSO (the
   `CF_Authorization` cookie on `stream.` is `SameSite=None`, so it's sent in the iframe).

### 7.3 Codec caveat (call it out, don't silently ship)

Because WebRTC is unavailable remotely and MSE has the Reolink codec matrix (§3.3),
**cross-browser remote live view isn't guaranteed out of the box.** Options, cheapest
first, documented for the user to choose at implementation:

- **Set the camera sub-stream to H.264 Main/Baseline** (Reolink web UI) so `sub` +
  MSE plays in all browsers including Safari. *Recommended* — no CPU cost, best compat.
- **Accept Safari-for-main:** HEVC main tile plays in Safari / HEVC-capable Chrome;
  fine if that's your remote browser.
- **Transcode via go2rtc/ffmpeg** to H.264 for a remote-friendly stream — CPU cost on
  the Mac; last resort.

The spec's code changes (force MSE remotely) are correct regardless; the codec choice
is a camera/config decision layered on top.

## 8. Security posture (summary)

- **No inbound ports / no exposed home IP** — only outbound cloudflared.
- **Admin:** identity (Access OTP) at the edge + app-side Basic Auth on LAN; JWT
  verified at origin (defense in depth) so `/admin` can't be reached by spoofing.
- **Live:** identity (Access OTP) — private authenticated service (also the safest ToS
  posture).
- **Gallery:** 256-bit capability token, path→cookie→redirect, `no-referrer`, revocable,
  rate-limited, HTTPS-only. Weaknesses acknowledged: a shared/forwarded link grants
  access until rotated; that's the accepted trade for "anyone with the link."
- **Obscurity is a bonus, not a control:** first-level hostnames stay out of CT while
  Total TLS is off, but security rests on the token/Access, never the name.
- **Fail closed** everywhere the gate is enabled but unconfigured.

## 9. Testing

- **Unit (no hardware/network):**
  - Capability gate: loopback vs LAN `remote_addr` branching; `/s/<token>` sets cookie
    + 302; bad token → 404; missing hash → 404 remote / open LAN; `Referrer-Policy`
    header present; constant-time compare used.
  - `wildlife-share-token`: generates ≥256-bit token, writes a verifying hash, prints
    URL; rotation invalidates the old hash.
  - Access-JWT: valid signed token (test keypair) → authorized; wrong `aud`/`iss`/`exp`
    / bad signature → falls back to Basic Auth; JWKS `kid` selection.
  - Request-aware `_live_base()`/mode: remote → `stream_base_url` + `mse`; LAN →
    local + configured modes.
  - `RemoteConfig` validation + `remote.enabled=false` is a no-op (regression guard).
- **Manual (against real Cloudflare, in the runbook):** tunnel reachable; `/s/<token>`
  flow from a phone off-LAN; `/admin` OTP; live tile plays remotely in the target
  browser; LAN behavior unchanged.

## 10. Build order (phases)

1. **Config + no-op guard:** `RemoteConfig`, wired in, `enabled=false` changes nothing.
   Tests green. (Safe to merge alone.)
2. **Capability gate + `wildlife-share-token`:** the gallery-sharing win. Tunnel +
   `cam.rlblais.org` runbook. Ships independently of live/admin.
3. **Admin Access JWT:** Access app 1 + origin verification.
4. **Remote live view:** `stream.rlblais.org` hostname + Access app 2 + request-aware
   embeds/MSE. Codec decision (§7.3).
5. **Docs:** README remote-access section + runbook.

Each phase is independently testable and mergeable.

## 11. Future / explicitly deferred

- **Shareable-by-link live (Option B):** add **Caddy** on the Mac as a front proxy for
  one origin, using `forward_auth` to a tiny Flask endpoint to check the capability
  cookie, routing `/` → gallery and `/stream/*` → go2rtc (Caddy proxies the MSE
  WebSocket natively). Or a lower-latency-sacrificing Flask proxy of go2rtc's HTTP
  MP4/HLS to avoid WebSocket entirely. Higher ToS exposure.
- **Multiple / single-use / expiring capability tokens** and a self-serve admin UI to
  mint/revoke them.
- **go2rtc LAN auth** (`api.username/password`) if the LAN should also be gated.

## 12. Open items to confirm on review

1. **Live-view auth = Option A (Access-gated, not shareable)** — chosen as the default
   from "go with recommendation"; confirm, or switch to Option B (adds Caddy).
2. **Hostnames** `cam.rlblais.org` / `stream.rlblais.org` — OK, or different labels?
3. **Whom to allow** in Access (just `rylincoln@gmail.com`, or invitees/`@rlblais.org`)?
4. **Sub-stream codec** for reliable cross-browser remote live (§7.3) — willing to set
   the camera sub-stream to H.264 Main/Baseline?
5. Values needed at implementation time (from your CF dashboard): tunnel token, the two
   Access **AUD** tags, your **team domain**.
