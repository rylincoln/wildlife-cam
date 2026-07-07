# Remote access via Cloudflare Tunnel — design

**Date:** 2026-07-02
**Status:** Draft for review (v2 — reflects the "one hostname, shared-secret link, shareable live, no admin" direction)
**Domain:** `example.com` (managed in Cloudflare; user's account)

## 1. Goal

Reach the wildlife-cam **gallery and live view** from the public internet over one
tidy URL, protected by a single **shared secret in the link** ("anyone with the link"),
with **no home IP or inbound ports exposed**. Uses a **named Cloudflare Tunnel** on the
**free plan**.

Confirmed with the user:

1. **Shareable live too** — live video rides the same secret link (not a separate login).
2. **One hostname → the existing app**, exposing **gallery + `/live`**, **no `/admin`**.
3. **Access gated by a shared secret in a URL param — keep it simple.**
4. **Yes** — willing to set the camera **sub-stream to H.264 Main/Baseline** for
   cross-browser remote live.
5. **Live playback = Option A (robust / Safari-friendly)** — go2rtc's own player under a
   path, MSE transport, gated at the edge. Chosen because "shareable" implies friends on
   iPhones (Safari). See §5 for the alternative. **← confirm on review.**

### Non-goals

- **Remote `/admin`.** `/admin` is blocked for tunnel traffic (404). Config editing
  stays LAN-only, exactly as today. This removes Cloudflare Access, JWT verification, a
  second hostname, and any new Python auth deps from scope.
- **Identity/login (Cloudflare Access).** Not used at all. The gate is a shared secret.
- **Changing LAN behavior.** On the LAN the gallery stays open, `/admin` stays
  Basic-Auth-gated, and live stays WebRTC. New *gating* applies **only** to tunnel
  traffic. The one LAN-visible side effect of enabling remote live is that go2rtc moves
  under the `base_path` prefix (`:1984/go2rtc/`); with remote off (`base_path: ""`) LAN is
  byte-for-byte as today.
- **Recording / changing the detector or go2rtc's streams.** Untouched.
- **Per-user / single-use / expiring links.** One rotatable shared secret (§6.2).

## 2. Verified constraints (high-confidence research; see transcript)

1. **WebRTC cannot cross a Cloudflare Tunnel** (HTTP/WebSocket only). Remote live must use
   go2rtc's **MSE** transport (fMP4 over a WebSocket at `/api/ws`), which Cloudflare
   proxies natively on all plans. Force **`mode=mse`** for remote (leaving `webrtc,mse`
   makes the player hang on an ICE timeout first).
2. **go2rtc supports a sub-path** via `api.base_path` (e.g. `/go2rtc`), so its player +
   API can live at `cam.example.com/go2rtc/…` behind a reverse route without breaking its
   internal `/api/...` calls.
3. **cloudflared supports path-based routing under one hostname** — multiple ingress
   rules with the same hostname and different `path`, most-specific first, catch-all last
   (dashboard exposes a "Path" field that writes the same rules).
4. **Codec/browser reality:** remote MSE needs **H.264 Main/Baseline** to play in every
   browser incl. Safari (Reolink's default sub is H.264 *High*, which Safari's MSE
   rejects; the 4K main is HEVC = Safari-only). Hence the camera sub-stream change (§7.3).
5. **MSE-over-WebSocket avoids the cloudflared response-buffering gotcha** that would
   affect a long-lived MP4 HTTP stream — another reason to prefer the MSE path.
6. **Video ToS** still exists (moved into CDN terms); a tunnel hostname is always proxied.
   For a personal cam behind a secret, with occasional low-volume viewing, practical risk
   is low (worst realistic case: a per-stream interstitial, not a ban). Keep it modest.
7. **Certificate Transparency:** first-level names like `cam.example.com` stay out of CT
   under Cloudflare's `*.example.com` **wildcard** Universal SSL — **as long as Total TLS
   is OFF**. Obscurity is a bonus, never the control.

## 3. Architecture

```
                         Internet
                            │  HTTPS only; home IP never exposed
                   ┌────────┴─────────────────────┐
                   │  Cloudflare edge              │
                   │  + 1 WAF rule on /go2rtc/*    │  (blocks if secret cookie absent)
                   └────────┬─────────────────────┘
                 cam.example.com  (one hostname, path-routed)
                   ┌────────┴─────────┐
                   │   cloudflared     │  (launchd daemon; named tunnel)
                   │  /go2rtc/* ─┐     │
                   │  /*  ───────┼──┐  │
                   └─────────────┼──┼──┘
             localhost:1984 ◄────┘  └────► localhost:8080
             ┌──────────────┐         ┌──────────────┐
             │   go2rtc      │         │ Flask gallery │
             │ base_path=    │         │  / , /live    │
             │  /go2rtc      │         │  (/admin 404  │
             │  MSE player   │         │   over tunnel)│
             └──────────────┘         └──────────────┘
```

- **One named tunnel**, **one hostname** `cam.example.com`, path-routed by cloudflared:
  - `cam.example.com/go2rtc/*` → `http://localhost:1984` (go2rtc; `base_path=/go2rtc`)
  - `cam.example.com/*` → `http://localhost:8080` (Flask gallery)
- **Gate = one shared secret**, enforced in two complementary places:
  - **Flask** gates everything it serves (`/`, `/live`, images, `/api/captures`) on
    tunnel traffic, and **sets the secret cookie** when it sees a valid `?key=`.
  - **One Cloudflare WAF custom rule** gates `/go2rtc/*` (which bypasses Flask) by
    requiring the same secret cookie. (Free-plan fallback: a tiny Cloudflare Worker on
    that path — see §6.4.)
- The secret cookie is host-scoped, so it's automatically sent to both `/` and `/go2rtc/*`.

### 3.1 LAN vs remote (how we tell them apart)

Same Flask process serves LAN clients (real private IPs) and cloudflared (connects from
`127.0.0.1`). We branch on **`request.remote_addr` being loopback** — robust because the
only things reaching `:8080` are LAN clients and cloudflared, and a remote attacker can't
make `remote_addr` a LAN IP (they can't reach `:8080` directly; only cloudflared is
exposed). *Assumption: cloudflared runs on the same Mac as the gallery (it does).*

| Route | LAN (`remote_addr` = private IP) | Remote (`remote_addr` = 127.0.0.1) |
|---|---|---|
| `/`, `/live`, images, `/api/captures` | open (unchanged) | **secret required** (Flask) |
| `/admin/*` | Basic Auth (unchanged) | **404** (blocked) |
| live video | go2rtc direct on `:1984`, WebRTC (unchanged) | `/go2rtc/*`, MSE, **secret required** (WAF) |

## 4. The share flow (end to end)

1. You send a friend `https://cam.example.com/?key=<SECRET>`.
2. Their browser hits Flask (catch-all route). Flask validates `?key`, sets an
   `HttpOnly; Secure; SameSite=Lax` cookie `wl_key=<SECRET>` for `cam.example.com`, and
   serves the gallery. (`Referrer-Policy: no-referrer` on all responses.)
3. They browse photos and open `/live`. The page embeds go2rtc's player from
   `/go2rtc/stream.html?src=<cam>_sub&mode=mse` (same origin). Those requests carry the
   `wl_key` cookie, so the **WAF rule** lets them through to go2rtc, which serves MSE.
4. A stranger hitting `cam.example.com` **without** the key (and without the cookie) gets
   **404** from Flask; a stranger hitting `/go2rtc/*` without the cookie is **blocked** by
   the WAF rule. Rotating the secret (regenerate + update config + WAF rule) invalidates
   every old link.

## 5. Live-playback decision (Option A chosen)

| | **Option A — go2rtc player under `/go2rtc` (chosen)** | Option B — Flask proxies `stream.mp4` |
|---|---|---|
| Works on Safari/iOS | **Yes** (MSE, H.264 Main) | No (bare `stream.mp4` often blank on Safari) |
| Latency | ~1 s (MSE) | ~1–3 s + higher start delay |
| App code | ~none (embed only) | streaming reverse-proxy w/ careful teardown |
| cloudflared buffering risk | none (WebSocket) | real (needs current cloudflared, named tunnel) |
| Gate enforcement | Flask (app) + 1 WAF rule (go2rtc path) | Flask only (one place) |
| go2rtc exposure | its UI/API reachable under `/go2rtc` (secret-gated) | go2rtc never exposed |

**Chosen: Option A**, because the whole point is *sharing* live with friends (iPhones →
Safari), and A is the only one that plays there reliably; its extra cost is one
declarative Cloudflare rule, not fragile code. **Trade-off accepted:** go2rtc's own
UI/API is reachable under the secret-gated `/go2rtc` path (mitigation in §8).

## 6. Repo / config changes

### 6.1 New `remote` config block + `livestream.base_path`

```yaml
remote:
  enabled: false                     # off by default; on activates the gate for tunnel traffic
  base_url: "https://cam.example.com"
  share_secret_hash: null            # Werkzeug hash of the shared secret; set via wildlife-share-secret
  block_admin: true                  # 404 /admin for tunnel (loopback) traffic

livestream:
  # ...existing fields...
  base_path: ""                      # set "/go2rtc" when exposing go2rtc under the tunnel subpath
```

New `RemoteConfig` pydantic model (mirrors `AdminConfig`), added to `Config` with a
`default_factory`. **When `remote.enabled` is false, nothing changes** — the app behaves
exactly as today. `livestream.base_path` defaults to `""` (LAN-direct, unchanged); it is
applied to both the generated `go2rtc.yaml` **and** the gallery's embed URLs so LAN and
remote stay consistent.

### 6.2 Shared-secret gate (Flask)

`before_request` + `after_request` in `create_app` (`gallery/app.py`), active only when
`remote.enabled` **and** `request.remote_addr` is loopback:

- **Admin block:** path starts with `/admin` → `404` (always, regardless of secret).
- **Secret check:** if `remote.share_secret_hash` is unset → **fail closed** (`404` all
  tunnel traffic; log "run wildlife-share-secret"). Else:
  - valid `?key=` (constant-time compare vs hash) → set cookie `wl_key` (`HttpOnly;
    Secure; SameSite=Lax`, long expiry) = the provided secret, continue;
  - else valid `wl_key` cookie → continue;
  - else → `404` (no oracle that a gate exists).
- **Headers:** `Referrer-Policy: no-referrer` on every response.
- Static assets (`/static/*`) and the health of internal links all ride the cookie, so the
  `?key=` only needs to appear once (on the shared link).

New CLI **`wildlife-share-secret`** (`wildlife.remote.share_secret:main`, in
`[project.scripts]`): generate `secrets.token_urlsafe(32)` (256-bit), store its **hash**
in `config.yaml` (atomic write, reusing the admin config-writing path), and print, once:
the plaintext secret (for the WAF rule), the share URL `${base_url}/?key=<secret>`, and a
reminder to paste the secret into the Cloudflare WAF rule.

### 6.3 Request-aware live embeds (`/live`)

`_live_base()` + per-tile `mode` + `base_path` become request-aware:

- **Remote (loopback, or host == public host):** base = same-origin `${base_path}` (e.g.
  `/go2rtc`), **`mode=mse`** forced. Fixes today's hardcoded `http://<host>:1984`
  (mixed-content) and the WebRTC-can't-traverse problem.
- **LAN:** `http://<host>:1984${base_path}` + configured `sub_mode`/`main_mode` (WebRTC).
  Note: because `api.base_path` moves go2rtc's root for LAN too, LAN embeds include the
  same prefix — the only LAN-visible change (the go2rtc dashboard is now at
  `:1984/go2rtc/`). If `base_path` is left `""` (remote off), LAN is byte-for-byte as today.

### 6.4 go2rtc + Cloudflare edge (config, not app code)

- **`stream/config_gen.py`:** emit `api.base_path: <livestream.base_path>` into
  `go2rtc.yaml` when set. (No other go2rtc change; streams unchanged.)
- **Cloudflare WAF custom rule** (dashboard, free plan, ≤5 rules): 
  *Block* when `starts_with(http.request.uri.path, "/go2rtc")` **and**
  `not http.request.cookies["wl_key"][0] == "<SECRET>"`.
  **Verify at implementation that free-plan WAF custom rules support cookie matching;**
  **fallback:** a tiny Cloudflare **Worker** bound to `cam.example.com/go2rtc/*` that
  checks the `wl_key` cookie and either `fetch()`es through or returns `403` (also free).

### 6.5 Files touched (estimate)

- `src/wildlife/config.py` — `RemoteConfig`; `LivestreamConfig.base_path`.
- `src/wildlife/gallery/app.py` — secret gate (`before_request`/`after_request`),
  request-aware `_live_base()`/mode/base_path.
- `src/wildlife/remote/share_secret.py` (new) — the CLI + hash/verify helpers.
- `src/wildlife/stream/config_gen.py` — emit `api.base_path`.
- `pyproject.toml` — `wildlife-share-secret` script. **No new runtime deps.**
- `config.example.yaml` — commented `remote:` block + `livestream.base_path`.
- `README.md` — "Remote access (Cloudflare Tunnel)" section + the §9 runbook.

## 7. Failure modes & edge cases

1. **`remote.enabled` but no secret hash:** fail **closed** (404 all tunnel traffic); LAN
   unaffected. Logged with the fix command.
2. **Wrong/missing key:** `404` (no oracle). 256-bit secret ⇒ brute force infeasible; add
   a light in-memory rate-limit on failed `?key` keyed by `Cf-Connecting-IP`.
3. **WAF rule missing/misconfigured:** `/go2rtc/*` could be left open (obscurity only) or
   fully blocked. Runbook includes a verification step (hit `/go2rtc/` with and without
   the cookie). Worker fallback if WAF free can't match cookies.
4. **`base_path` mismatch** (go2rtc vs cloudflared vs embed): live page 404s under
   `/go2rtc`. All three read from `livestream.base_path`; runbook verifies once.
5. **Safari with H.264 High (camera not reconfigured):** sub tile won't play in Safari.
   Requires the §7.3 camera change; `/live` shows a brief "try the other tile / a
   Chromium browser" hint when remote.
6. **Tunnel down:** Cloudflare serves its own error page; runbook covers `launchctl`/
   `brew services` + log paths.
7. **Secret leaks (forwarded link, history, logs):** accepted trade for "anyone with the
   link." Mitigated by cookie (key appears once), `no-referrer`, and easy rotation.

### 7.3 Camera codec (one-time setup)

Set each Reolink **sub-stream to H.264, Main or Baseline profile** (camera web UI /
`/admin` camera settings) so MSE plays it in **all** browsers including Safari. The 4K
main stays HEVC (Safari-only over MSE) — fine as the "HD in Safari" tile. Documented in
the runbook; no code depends on it beyond forcing `mode=mse`.

## 8. Security posture

- **No inbound ports, home IP hidden** — only outbound cloudflared.
- **`/admin` unreachable remotely** (404 on tunnel traffic) — config editing stays LAN +
  Basic Auth.
- **Shared secret:** 256-bit, hashed in `config.yaml`, cookie is `HttpOnly/Secure/Lax`,
  `no-referrer`, rate-limited, HTTPS-only, **rotatable** (regenerate → update config +
  WAF rule). It is **view-only** access.
- **go2rtc exposure (accepted trade of Option A):** its UI/API is reachable under the
  secret-gated `/go2rtc`. Mitigations: (a) the secret gate covers it; (b) *optional*
  hardening — restrict the WAF/Worker to only the paths the player needs
  (`/go2rtc/stream.html`, `/go2rtc/api/ws`, `/go2rtc/api/frame*`, static assets) and block
  go2rtc's config/edit endpoints; (c) note LAN go2rtc is already open today. Recommend
  shipping with the gate and adding path-restriction if you ever widen who gets the link.
- **Obscurity is a bonus, not a control:** keep **Total TLS OFF** so `cam.` stays out of
  CT; security rests on the secret.
- **Fail closed** when enabled-but-unconfigured.

## 9. Cloudflare + host setup (runbook, done once)

1. `brew install cloudflared` (keep it current: `brew upgrade cloudflared`).
2. Zero Trust → Networks → Tunnels → Create → Cloudflared → name `wildlife`; copy the
   **token**. `sudo cloudflared service install <TOKEN>` (boot launch daemon; prefer
   `--token-file`/`TUNNEL_TOKEN` over shell history).
3. Public hostnames (order matters — specific path first):
   - `cam` . `example.com`, **Path `go2rtc`** → HTTP `localhost:1984`
   - `cam` . `example.com`, (no path) → HTTP `localhost:8080`
4. Run `wildlife-share-secret`; note the printed secret + share URL.
5. Set `remote.enabled: true`, `remote.base_url`, `livestream.base_path: "/go2rtc"` in
   `config.yaml`; regenerate `go2rtc.yaml` (`wildlife-stream-config`); restart gallery +
   go2rtc.
6. Add the **WAF custom rule** on `/go2rtc/*` requiring `wl_key == <secret>` (or deploy
   the Worker fallback).
7. Set each camera **sub-stream to H.264 Main/Baseline**.
8. Confirm **Total TLS OFF** for the zone.
9. Verify: from off-LAN, `/?key=…` loads the gallery; `/live` plays on an iPhone (Safari)
   and a Chromium browser; `/admin` returns 404; `/go2rtc/` without the cookie is blocked;
   LAN behavior unchanged.

## 10. Build order (phases)

1. **Config + no-op guard:** `RemoteConfig`, `livestream.base_path`, wired in;
   `enabled=false` changes nothing. Tests green. (Mergeable alone.)
2. **Secret gate + `wildlife-share-secret`:** Flask gate, admin-block, cookie, headers,
   rate-limit; CLI. Tunnel + `cam.example.com` catch-all runbook → gallery shareable.
3. **Remote live:** `livestream.base_path` → `go2rtc.yaml`; request-aware embeds/MSE; the
   `/go2rtc` cloudflared route + WAF rule; camera codec. → live shareable.
4. **Docs:** README remote-access section + runbook.

Each phase is independently testable and mergeable.

## 11. Testing

- **Unit (no hardware/network):**
  - Gate: loopback vs LAN branching; `?key` valid → cookie set + served; valid cookie →
    served; bad/missing → 404; missing hash → 404 remote / open LAN; `/admin` → 404 remote,
    Basic Auth on LAN; `Referrer-Policy` header; constant-time compare; rate-limit.
  - `wildlife-share-secret`: ≥256-bit secret; writes a verifying hash; prints secret + URL;
    rotation invalidates the old hash.
  - Request-aware `_live_base()`/mode/base_path: remote → `${base_path}` + `mse`; LAN →
    `http://host:1984` + configured modes.
  - `config_gen`: emits `api.base_path` iff `livestream.base_path` set.
  - `RemoteConfig` validation + `remote.enabled=false` no-op regression.
- **Manual (runbook §9):** the off-LAN verification checklist, incl. iPhone/Safari live.

## 12. Considered alternatives

- **Option B (Flask proxies `stream.mp4`):** simplest code + no edge rule + go2rtc never
  exposed, but **no Safari/iOS**, higher latency, and cloudflared-buffering-sensitive.
  Rejected because sharing implies iPhone viewers. (Kept here in case Safari support turns
  out not to matter — it's a smaller change.)
- **Cloudflare Access (identity login):** dropped entirely per "shared secret, keep it
  simple, no admin."
- **Second hostname for go2rtc:** dropped per "one hostname."
- **WebSocket-proxy go2rtc through Flask:** would unify the gate in one place with MSE, but
  needs WS-bridge code in a WSGI app — not "simple."

## 13. Open items to confirm on review

1. **Live playback = Option A** (go2rtc under `/go2rtc`, edge WAF rule) — confirm, or take
   Option B (simpler, no Safari).
2. **Hostname `cam.example.com`** and **subpath `/go2rtc`** — OK, or different labels?
3. **go2rtc-exposure trade** (§8) — ship with the plain gate, or include the optional
   path-restriction hardening from day one?
4. **Secret in `?key=` query param** — OK as specified (cookie after first hit +
   `no-referrer`), or do you want the cleaner `/s/<token>`-path-then-redirect form?
5. Values needed at **implementation** time (from your CF dashboard): tunnel token; and
   you'll paste the generated secret into the WAF rule.
