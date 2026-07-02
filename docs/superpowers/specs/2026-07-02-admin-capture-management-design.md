# Admin capture management — design

**Date:** 2026-07-02
**Status:** Draft for review
**Feature:** Manage detected captures (delete, reclassify, review) from the password-gated `/admin` section.

---

## 1. Overview & goals

The gallery is read-only. Once the detector saves a capture (a SQLite row + a full JPEG +
a thumbnail), there is no in-app way to correct a mislabel or remove a bad frame — the only
deletion path is the `scripts/prune.py` retention CLI. This feature adds a **`/admin/captures`**
management view that lets an authenticated LAN admin:

- **Delete** captures (single, or many via multi-select) — removing the DB row *and* both
  on-disk files, mirroring the exact contract `prune.py` already uses.
- **Reclassify** a capture's label (single or bulk) from a constrained dropdown, recording
  provenance (what the model originally predicted) so corrected captures are usable later as
  clean training data.
- **Review-triage** — mark captures reviewed and filter to the unreviewed backlog, so a human
  can work through detections without re-seeing what they've already handled.

It lives entirely behind the existing admin auth + CSRF guard. The public gallery stays
read-only and unauthenticated.

### Design principles

- **Follow existing patterns.** Admin blueprint routes, `before_request` auth + same-origin
  CSRF, `flash('ok'|'err')` + PRG for form posts, the `test-camera` `fetch()`+JSON precedent for
  the bulk endpoint, one stylesheet with existing tokens, vanilla-JS IIFE per page.
- **The Store is the integrity boundary.** All DB/file mutation logic lives in `store.py` with
  validation, so a route (or a future caller) can't corrupt data. Routes are thin.
- **Reuse the reference deletion contract.** `prune.py`'s `_safe_unlink` / `_prune_empty_dirs`
  are lifted into the package and shared, so there is one implementation of "safely remove a
  capture's files."
- **Convention over config.** No new `config.yaml` surface in v1; a couple of module constants.

---

## 2. Scope

### In scope (v1)

- New `Store` methods: `delete`, `delete_many`, `update_label`, `update_label_many`,
  `mark_reviewed_many`, `count`, plus a shared file-deletion helper.
- Schema evolution: three nullable/defaulted columns (`original_label`, `reviewed`,
  `reviewed_at`) added via an **idempotent migration**, plus `_COLUMNS` update and a
  per-connection `busy_timeout`.
- Admin blueprint: extended factory signature + new routes (`GET /admin/captures`,
  `POST …/<id>/delete`, `POST …/<id>/reclassify`, `POST /admin/captures/bulk`).
- New template `templates/admin/captures.html` (grid + selection + bulk toolbar + reclassify +
  reused lightbox), CSS additions, a "Captures" nav link.
- `prune.py` refactor to import the shared helpers, plus a retention rule that spares
  human-reviewed low-confidence captures.
- Unit + Flask route tests; docs updates (README, spec.md).

### Out of scope (v1) — deliberate

- **A `false_positive` label sentinel.** Deleting is the disposal path for false positives.
  (Rationale in §11. A dedicated `disposition` column for a "keep-as-hard-negative" bucket is a
  clean future addition.)
- **Free-text relabeling.** The reclassify target set is constrained to keep the label space —
  and the public gallery's label filter — clean.
- **Cross-page selection.** Selection is per-page (§8.4).
- **Undo / soft-delete / trash.** Delete is permanent (as `prune.py` already is).
- **File renaming on reclassify.** The on-disk filename bakes in the old label, but nothing ever
  parses it back — resolution is by the stored `image_path`/`thumb_path`. Reclassify is DB-only.

### Decisions flagged for your confirmation

These are my best-judgment defaults (you were away). Each is easy to change; call out any:

1. **Drop the `false_positive` sentinel** — delete is the disposal path. *(§11)*
2. **Include the review-triage surface** (`reviewed` badge + "unreviewed only" filter +
   "Mark reviewed" bulk action). Without it, the provenance columns would be write-only. If you
   want the barest tool, we cut this and keep only delete + reclassify (provenance still recorded
   on reclassify). *(§8.6)*
3. **Provenance columns + migration** (`original_label`, `reviewed`, `reviewed_at`). *(§4)*
4. **Constrained reclassify dropdown**, no free-text. *(§8.5)*
5. **Prune spares reviewed rows from the `min_confidence_keep` rule** (not from age-based
   retention). *(§11)*
6. **Offset pagination + in-place DOM updates + an "unreviewed" triage workflow** rather than
   keyset pagination. *(§8.4)*

---

## 3. Architecture at a glance

```
                 ┌─────────────────────────── gallery/app.py (Flask app factory) ────────────┐
                 │  create_app(config, config_path)                                            │
                 │    • per-request get_store() on flask.g   • app.config[CAPTURES_DIR]         │
                 │    • public read-only routes: / /api/captures /image/<id> /thumb/<id>        │
                 │    • register_blueprint(create_admin_blueprint(                              │
                 │          config_path, get_config, get_store, captures_dir))  ◄── EXTENDED    │
                 └───────────────────────────────────────┬──────────────────────────────────────┘
                                                          │
                 ┌────────────────────── admin/routes.py (blueprint, url_prefix=/admin) ───────┐
                 │  @before_request _guard: same-origin CSRF (non-GET) + check_admin_auth       │
                 │  NEW: GET /captures  POST /captures/<id>/delete  /captures/<id>/reclassify    │
                 │       POST /captures/bulk (JSON)                                             │
                 └───────────────────────────────────────┬──────────────────────────────────────┘
                                                          │  (thin routes → Store)
                 ┌──────────────────────────────── store.py (integrity boundary) ─────────────┐
                 │  __init__: + PRAGMA busy_timeout, synchronous=NORMAL (per connection)        │
                 │  init_schema: WAL (persistent) + idempotent ALTER TABLE migration            │
                 │  NEW: delete / delete_many / update_label / update_label_many /              │
                 │       mark_reviewed_many / count   (+ shared _safe_unlink/_prune_empty_dirs)  │
                 └─────────────────────────────────────────────────────────────────────────────┘
                                                          ▲
                                       scripts/prune.py ──┘  imports the shared helpers; spares reviewed rows
```

---

## 4. Data model changes (`store.py`)

### 4.1 New columns

Add three columns to the `captures` table:

| Column | Type | Nullability | Meaning |
|---|---|---|---|
| `original_label` | `TEXT` | nullable (NULL until first reclassify) | The label the model produced, captured the first time a human reclassifies. Never overwritten thereafter. |
| `reviewed` | `INTEGER NOT NULL DEFAULT 0` | defaulted | `1` once a human has reclassified or explicitly marked the capture reviewed. |
| `reviewed_at` | `TEXT` | nullable | ISO8601 timestamp of the last human action, consistent with the `capture_ts` convention. |

Both the `CREATE TABLE` in `_SCHEMA_SQL` (for fresh DBs) **and** the migration (for deployed DBs)
must define them.

### 4.2 `_COLUMNS` must be extended  *(hardening finding — blocker)*

`_row_to_dict` builds row dicts with `{key: row[key] for key in _COLUMNS}` over `SELECT *`.
If the new columns are **not** appended to the module-level `_COLUMNS` tuple, they are silently
dropped from every `get()`/`query()`/`update_label()` return value and the provenance/review
UI renders nothing. **Append `original_label`, `reviewed`, `reviewed_at` to `_COLUMNS`.**
A test asserts the keys are present in `get()`/`query()` output.

### 4.3 Idempotent migration in `init_schema()`  *(hardening finding — blocker + major)*

`init_schema()` today runs `executescript(_SCHEMA_SQL)` (which is `CREATE TABLE IF NOT EXISTS` +
indexes). Add, after that, a small migration that adds any missing column:

```python
def _migrate(self) -> None:
    existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(captures)")}
    additions = [
        ("original_label", "TEXT"),
        ("reviewed", "INTEGER NOT NULL DEFAULT 0"),
        ("reviewed_at", "TEXT"),
    ]
    for name, decl in additions:
        if name in existing:
            continue
        try:
            self._conn.execute(f"ALTER TABLE captures ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError as exc:
            # Cross-process race: another process (worker vs gallery boot) added it first.
            if "duplicate column name" not in str(exc).lower():
                raise
    self._conn.commit()
```

Called under `self._write_lock`, right after `executescript`, in `init_schema()`.

- **Why the `try/except`, not just the `PRAGMA` pre-check:** the worker and the gallery-boot
  Store can call `init_schema()` concurrently at launch (launchd starts both). A pure
  check-then-`ALTER` is a cross-process TOCTOU — both read "missing", both `ALTER`, the second
  raises `duplicate column name` (a schema error, **not** `SQLITE_BUSY`, so `busy_timeout` won't
  absorb it) and crashes startup. Treating `duplicate column name` as success makes it safe.
- `reviewed INTEGER NOT NULL DEFAULT 0` backfills existing rows correctly (SQLite returns the
  constant default for pre-existing rows).
- Test: run `init_schema()` twice on a pre-populated DB, and run it on two separate connections
  against a pre-populated DB — no error, columns present, existing rows have `reviewed = 0`.

### 4.4 Per-connection `PRAGMA busy_timeout`  *(hardening finding — blocker)*

Set `PRAGMA busy_timeout=5000` **and** `PRAGMA synchronous=NORMAL` in `Store.__init__`
(immediately after `sqlite3.connect`), **not** in `init_schema()`.

- **Why not `init_schema`:** `busy_timeout` and `synchronous` are *per-connection* and
  non-persistent (unlike `journal_mode=WAL`, which lives in the file header). Per-request
  gallery/admin Stores never call `init_schema()` — only the boot Store and the worker do. If
  `busy_timeout` were set only in `init_schema`, the exact connections doing the new admin
  **writes** would default to `busy_timeout=0` and hit an immediate "database is locked" the
  instant the worker is mid-INSERT — turning normal concurrency into bogus `err` flashes.
- `journal_mode=WAL` stays in `init_schema()` (persistent; one-time is fine).
- **`_write_lock` does not coordinate across connections/processes.** Each `Store` instance has
  its own `threading.Lock`; the worker and each per-request admin Store are different objects and
  different processes. The real cross-writer arbiter is **WAL single-writer + `busy_timeout`**.
  `_write_lock` only orders writes *within one instance*; keep using it for that, and document
  this in the method docstrings. Keep DB write transactions short and do all file I/O **outside**
  the transaction so a bulk delete never starves the worker past `busy_timeout`.

---

## 5. Shared file-deletion helpers (lift out of `prune.py`)

`scripts/prune.py` is not an importable package module, so its `_safe_unlink(captures_dir, rel)`
and `_prune_empty_dirs(root)` can't be reused as-is. Lift both into `src/wildlife/store.py`
(module-level functions) verbatim; `scripts/prune.py` imports them back so there is one
implementation.

- `_safe_unlink(captures_dir, rel) -> tuple[bool, int]`: resolves the target, **refuses paths
  that escape `captures_dir`** (`root == resolved or root in resolved.parents`), tolerates
  missing files (returns `(False, 0)`), never raises. This is the path-traversal guard and the
  idempotency guarantee.
- `_prune_empty_dirs(root) -> int`: bottom-up `os.walk` removing now-empty dirs.
- Only cosmetic change: the logger name moves from `logging.getLogger("prune")` to
  `wildlife.store`. Behavior identical.

---

## 6. Store API (new methods)

All mutations acquire `self._write_lock`, run a **short** transaction, `commit()`, and derive
their return count from `cursor.rowcount` (not from a prior `get()` — see §6.6). File I/O
happens outside the lock/transaction. `image_path`/`thumb_path` are relative to `captures_dir`;
every unlink goes through `_safe_unlink`.

### 6.1 `delete(capture_id: int) -> bool`

1. `row = self.get(capture_id)`; if `None`, return `False` (no exception).
2. Outside the write lock: `_safe_unlink` for `row["image_path"]` **and** `row["thumb_path"]`
   (files first, tolerant of missing/failed unlink — do **not** gate the DB delete on unlink).
3. Under `_write_lock`: `cur = execute("DELETE FROM captures WHERE id=?", (capture_id,))`,
   `commit()`. Return `cur.rowcount == 1`.
4. Best-effort: sweep **only the deleted capture's dated directory** (`YYYY/MM/DD` derived from
   `image_path`), **skipping the current day** (the worker may be mid-write there). Do **not**
   run the full-tree `_prune_empty_dirs(captures_dir)` per delete — see §6.7.

### 6.2 `delete_many(ids: Iterable[int]) -> int`

1. Coerce every id to `int` (the route also validates — defense in depth).
2. Chunk ids to ≤ 900 per statement (SQLite variable limit). For each chunk: `SELECT id,
   image_path, thumb_path WHERE id IN (?, ?, …)` to gather paths.
3. Outside the transaction: `_safe_unlink` both files for every gathered row.
4. Under `_write_lock`: for each chunk `DELETE FROM captures WHERE id IN (…)`, accumulate
   `cur.rowcount`; single `commit()`. Return total rowcount (**rows actually removed**).
5. Sweep the affected dated dirs once (scoped, skip current day).

### 6.3 `update_label(capture_id: int, new_label: str) -> dict | None`

1. Validate `new_label` is non-blank (`.strip()`); else raise `ValueError` (Store is the
   integrity boundary — `''` passes `NOT NULL` but must not reach the DB).
2. Under `_write_lock`, a **single** statement (so provenance is captured correctly):
   ```sql
   UPDATE captures
      SET original_label = COALESCE(original_label, label),
          label = ?, reviewed = 1, reviewed_at = ?
    WHERE id = ?
   ```
   `commit()`. Using `COALESCE(original_label, label)` in the same UPDATE evaluates against the
   **old** row, so `original_label` records the true first label. A two-statement version
   (set `label` first, then backfill `original_label` from `label`) would corrupt provenance
   (`bird→deer` would record `original_label='deer'`).
3. If `cur.rowcount == 0`, return `None`; else return `self.get(capture_id)` (now includes the
   new provenance columns via §4.2).

### 6.4 `update_label_many(ids, new_label) -> int`

Same validation; single `UPDATE … WHERE id IN (…)` with the same `COALESCE` expression, chunked.
Return total `rowcount` (rows touched). "Touched" is the defined semantics — reclassifying to a
label some rows already have still marks them reviewed and counts them (documented; see §6.6).

### 6.5 `mark_reviewed_many(ids) -> int`  *(review-triage feature)*

`UPDATE captures SET reviewed = 1, reviewed_at = ? WHERE id IN (…) AND reviewed = 0` — the
`reviewed = 0` guard means `rowcount` counts only genuinely newly-reviewed rows. Does not touch
`label`/`original_label`.

### 6.6 `count(**filters) -> int`

`SELECT COUNT(*)` reusing the exact same WHERE-clause builder as `query`, extended with an
optional `reviewed` filter (`None` = all, `True`/`False` = filter). Also extend `query()` with
the same `reviewed` param. Drives the results header ("N captures", "M unreviewed") and
pagination bounds.

### 6.7 Semantics notes (from adversarial review)

- **`affected` = rows actually removed / touched**, taken from `cursor.rowcount`, never from
  `get()`-existence (two admins deleting overlapping id sets, or racing `prune`, would each
  count a row that only one DELETE removed, and the flash would lie).
- **Files-first, then row** matches `prune.py`. A crash between leaves a "ghost row" pointing at
  deleted files; that's tolerable because `_serve_file` already 404s on missing files and
  `AUTOINCREMENT` never reuses ids, so a stale link can't resolve to a different capture. Delete
  is idempotent, so a retry cleans up the ghost row.
- **Scoped empty-dir sweep, skipping the current day.** The worker does
  `mkdir(parents=True, exist_ok=True)` then `image.save(...)` with a small window between. A
  full-tree bottom-up sweep on every interactive delete could `rmdir` today's freshly-created
  directory before the worker's save, making the worker drop that capture. Scope the sweep to the
  deleted rows' dated dirs and skip the current day; leave the whole-tree sweep to `prune`.

---

## 7. Admin blueprint wiring

### 7.1 Factory signature change (`admin/routes.py` + `gallery/app.py`)

`create_admin_blueprint` currently takes `(config_path, get_config)`. Extend it to
`(config_path, get_config, get_store, captures_dir)`. At the single registration call site in
`create_app` (only reached when `config_path is not None`), both `get_store` (the per-request
closure) and `captures_dir` (the already-`.resolve()`d local, == `app.config["CAPTURES_DIR"]`)
are in scope — pass them directly:

```python
app.register_blueprint(
    create_admin_blueprint(config_path, get_config, get_store, captures_dir)
)
```

The admin routes call `get_store()` to obtain the same per-request `Store` (on `flask.g`) the
gallery uses, so connections are still per-request and torn down by the existing
`teardown_appcontext`.

### 7.2 Serialize helper

The gallery's `_serialize` / `_query_page` are **closures inside `create_app`** and not
importable. The admin blueprint defines its own small `_serialize_capture(row)` →
`{id, camera_id, label, confidence, capture_ts, reviewed, original_label,
thumb_url, image_url}` where `thumb_url = url_for("thumb", capture_id=id)` and
`image_url = url_for("image", capture_id=id)`. These point at the **existing** gallery
`/thumb/<id>` and `/image/<id>` routes — consistent with the current posture (images are already
served unauthenticated on the LAN; the admin grid just references them). The module-level filter
parsers (`_parse_filters`, `_parse_date`, `_parse_float`, `_parse_page`) in `app.py` **are**
importable and are reused.

### 7.3 Routes (all auto-guarded by `@before_request _guard` = same-origin CSRF + Basic Auth)

| Route | Method | Purpose | Response |
|---|---|---|---|
| `/admin/captures` | GET | Filtered, paginated grid | Renders `admin/captures.html` |
| `/admin/captures/<int:id>/delete` | POST | Single delete | PRG: `flash('ok'/'err')` → redirect back to filtered list |
| `/admin/captures/<int:id>/reclassify` | POST | Single reclassify (form `new_label`) | PRG: validate → `flash` → redirect |
| `/admin/captures/bulk` | POST (JSON) | Bulk delete/reclassify/mark-reviewed | JSON `{ok, action, affected}` / `{ok:false, error}` |

**GET `/admin/captures`:** parse filters (`camera`, `label`, `start`, `end`, `min_confidence`,
plus new `reviewed` ∈ {all, unreviewed, reviewed}) and `page`. Call `store.query(...)` and
`store.count(...)`. Render with serialized rows, the filter option lists
(`distinct_cameras()`, and reclassify labels = `animal_classes ∪ distinct_labels()` sorted),
`page`, `has_more`, `total`, `reviewed` state.

**Single POST routes** are `<int:id>` (the converter rejects non-int), POST-only (a mutating GET
would bypass the Origin check — GETs are exempt — and be CSRF-able via `<img>`). Reclassify
validates `new_label` against the server-side allow-list (`animal_classes ∪ distinct_labels ∪
current label`); on failure re-flash `err` and redirect. Redirect preserves the current filter
query string (carried as hidden form fields).

**Bulk POST `/admin/captures/bulk`** (mirrors the `test-camera` fetch+JSON precedent):
- **Require JSON.** Read `request.get_json(silent=True)`; if `None`, return JSON 400/415. Do
  **not** use the `get_json() or request.form` fallback — accepting form-encoded bodies would let
  a cross-origin HTML form skip the CORS preflight and rely on the Origin heuristic alone.
- Body: `{action: "delete"|"reclassify"|"mark_reviewed", ids: [int], label?: str}`.
- **Strictly coerce every id with `int(x)`**; reject the whole request (JSON 400) on any non-int.
  This is the SQL-injection guard for the `IN (…)` clause — combined with placeholder binding in
  the Store, never string-formatting.
- **Cap `len(ids)` ≤ `BULK_MAX_IDS` (1000)**; reject larger batches (self-inflicted-DoS guard).
- For `reclassify`, validate `label` against the allow-list.
- Dispatch to `store.delete_many` / `update_label_many` / `mark_reviewed_many`.
- Return `{ok: true, action, affected}`.

---

## 8. Frontend (`templates/admin/captures.html`, `style.css`, `admin/base.html`)

### 8.1 Template shell

`captures.html` `{% extends "admin/base.html" %}` with `{% block title %}` / `{% block content %}`
/ `{% block scripts %}`. `base.html` already renders `flash` banners and the `{% if errors %}`
block — the single-item PRG routes reuse those for free.

**Wider container:** `admin/base.html`'s `<main class="admin-main">` is `max-width: 900px`, which
yields ~3 grid columns — cramped for triage. Add a `{% block main_class %}admin-main{% endblock %}`
to `base.html` so `captures.html` can set `admin-main admin-main-wide`, and add
`.admin-main-wide { max-width: 1400px; }` to `style.css`.

### 8.2 Two feedback paths (documented, deliberate)

- **Single-item** delete/reclassify → **PRG** (form POST, `flash`, full-page redirect). Works
  **without JavaScript** — graceful degradation, matching the camera-CRUD pattern.
- **Bulk** → **JSON fetch + inline live region** (`#bulk-status`), updating the grid **in place**.
  Requires JS (a progressive enhancement, like `test-camera`).

An `aria-live="polite"` `#bulk-status` banner (reusing `.banner`/`.banner-ok`/`.banner-err`)
reports e.g. "Deleted 12 · 2 skipped".

### 8.3 In-place grid update after bulk actions

On a successful bulk response, the JS reconciles the DOM rather than reloading:
- **delete:** remove the matching `.card[data-id]` nodes; clear selection; update the results
  header + toolbar count; if the page is now empty and `page > 1`, navigate to the previous page;
  else show the empty state (§8.7).
- **reclassify:** update each card's `.meta .label` (and reviewed badge); clear selection.
- **mark_reviewed:** add the reviewed badge to each card; clear selection.

A manual **Refresh** link is provided; residual offset drift (§8.4) is documented in helper text.

### 8.4 Pagination + selection

- **Offset pagination** (`ORDER BY capture_ts DESC, id DESC LIMIT ? OFFSET ?`, reusing `query`).
  Prev/next with **disabled states** at the ends and a "Page N" indicator (`count()` gives the
  total so we can show "Page N of M").
- **Known limitation (documented):** offset paging can skip/repeat rows when the row set shifts
  (a large delete, or the worker inserting newest-first while you page). We accept this for a
  low-traffic LAN tool instead of adding keyset pagination, **because in-place mutation avoids
  reloading the page under you, and the recommended triage workflow sidesteps drift entirely:**
  filter to **"unreviewed", stay on page 1**, and mark/relabel — reviewed rows leave the filter,
  page 1 refills, and you never rely on a stale offset. (Keyset pagination is a clean future
  upgrade if exhaustive ordered triage is ever needed.)
- **Selection is per-page.** A JS `Set` of `data-id` + a `.card.selected` class, seeded empty on
  each page load, cleared on navigate/filter. A sticky bulk toolbar shows a live "N selected"
  count and a "Select all on page" checkbox; helper text states "Selection applies to the current
  page." No cross-page selection in v1.

### 8.5 Reclassify UI

- The target set is `animal_classes ∪ distinct_labels() ∪ current label`, de-duplicated and
  sorted deterministically. **Always includes the capture's current label** even if it's no
  longer in `animal_classes` (a legacy label stays selectable and preselected).
- **Bulk reclassify:** a `<select>` in the toolbar (populated server-side) + an "Apply" button
  that POSTs the selection to `/admin/captures/bulk`.
- **Single reclassify:** a small `<select>`+form in the **lightbox** (see §8.8) for careful
  one-at-a-time review, backed by the PRG route.
- No free-text (keeps the label space + the public gallery filter clean).

### 8.6 Card markup & the click conflict  *(hardening finding — major)*

Reuse `.grid`/`.card`/`.meta`. Per card:
- A **selection checkbox** placed as a **sibling of `img.thumb`** (top-left), with
  `pointer-events: auto`. It must **not** live inside `.meta` (which is `pointer-events: none`).
  A visible check badge gives a **non-color** selected indicator (not just a border tint) — plus
  the native checkbox is keyboard-focusable and Space-toggleable (accessibility).
- The `.meta` overlay (`label`/`conf`/`cam`/`ts`) as in the gallery, plus a `.pill.pill-ok`
  **"reviewed"** badge when `reviewed`.
- A per-card **× delete** control (top-right, on hover), backed by the PRG single-delete form with
  an `onsubmit="return confirm(...)"` guard.
- **Guard the lightbox click:** the reused delegated handler is
  `grid.addEventListener("click", e => { const card = e.target.closest(".card"); if (card) openLightbox(card); })`.
  It must be guarded so clicks on the checkbox / delete button / any `input,select,button` do
  **not** open the lightbox (`if (e.target.closest("input,select,button")) return;`). Shift-click
  range-select is supported (expected in grids).

### 8.7 Empty states

Mirror `index.html`'s `#empty` block (the `.empty` style exists): distinguish **"No captures
yet"** (total zero) from **"No captures match these filters"** (filtered zero, with a Reset link).

### 8.8 Lightbox reuse

The lightbox is an inline IIFE bound to element IDs in `index.html` (not a shared partial), so its
markup + JS are **duplicated** into `captures.html` (its CSS is already shared). Enhancements for
the admin context:
- Add **focus management**: move focus to `.lb-close` on open, restore on close.
- **Namespace keyboard handling** so that while the lightbox is open it swallows arrows/Escape and
  selection keys are ignored.
- Add a **reclassify `<select>` + delete button** in the lightbox caption for the focused capture
  (PRG routes).
- (Optional future cleanup: factor the lightbox into a shared `_lightbox.html` + `lightbox.js` to
  avoid drift. Out of scope for v1.)

### 8.9 CSS additions (to `style.css`, using existing tokens)

`.admin-main-wide`; card selection checkbox positioning + check badge; `.card.selected` outline
(with the badge as the non-color cue); sticky `.bulk-toolbar`; `#bulk-status`; reviewed `.pill`;
per-card `×` button; disabled pagination buttons. Reuse `--accent` for selection; **danger red is
hardcoded** (`#f85149` border / `#ff8781` text / `rgba(248,81,73,…)` fill) — reuse those exact
values or the `.btn.danger` class, not a new token.

### 8.10 Navigation & discoverability

Add a **"Captures"** link to the admin nav in `admin/base.html` (alongside Dashboard / Detection /
Cameras / Password). **No** per-card "manage" link from the public gallery — the gallery is
unauthenticated, so such a link would either 401-prompt every visitor or advertise the admin
surface. The existing `admin_enabled`-gated Admin nav link is the entry point.

---

## 9. Security (consolidated)

- **All new routes sit on the admin blueprint**, so `@before_request _guard` (same-origin CSRF for
  non-GET + `check_admin_auth`) covers every one — including the JSON bulk route (the guard runs
  for every dispatched request regardless of method/content-type). **Never** attach a
  capture-mutation or capture-listing route to the gallery `app` object (unauthenticated). Note the
  per-request `Store` is a read-write class; it stays read-only in the gallery only because the
  public routes call read methods — do not add a mutation convenience route to `app`.
- **Mutations are POST-only.** A mutating GET would bypass the Origin check and be exploitable via
  `<img>`/`<link>`. Route tests assert 405 on GET.
- **CSRF:** same-origin `Origin`/`Referer` check. A same-origin `fetch()` carries the browser's
  cached Basic-Auth creds and sends `Origin=host`, so it both authenticates and passes. The
  no-`Origin` (curl) allowance is fine — browsers always attach `Origin` on cross-origin
  state-changing requests, so a no-Origin request is not a browser-CSRF vector (and still needs
  Basic Auth). The bulk route additionally **requires JSON** (a second, independent CSRF barrier —
  form-encoded cross-origin posts skip preflight; JSON doesn't).
- **SQL injection:** every statement uses `?` placeholders, including the dynamic `IN (…)` built as
  `",".join("?" * len(chunk))` with ints bound as params — never string-formatted. Bulk ids are
  `int()`-coerced at the route boundary; labels are additionally allow-listed.
- **Path traversal:** file deletion goes through `_safe_unlink` (resolve-then-contain), blocking
  `../`, absolute-path, and symlink escapes. A refused unlink does **not** abort the row delete.
- **XSS:** Jinja autoescape stays on (no `| safe`); the `#bulk-status` element and any JS echoing a
  label/count use `element.textContent` (the codebase convention), never `innerHTML`
  concatenation. The server-side label allow-list means `distinct_labels()` can never be seeded
  with markup.
- **Abuse:** `BULK_MAX_IDS` cap + a `confirm()` before bulk delete. Rate-limiting adds little on a
  single-shared-password LAN box.

---

## 10. Concurrency & correctness (consolidated)

Covered inline above; the load-bearing points:
- `busy_timeout` + `synchronous=NORMAL` in `__init__` (per connection); WAL in `init_schema`
  (persistent). `_write_lock` orders writes within one instance only. *(§4.4)*
- Migration idempotent across re-runs **and** concurrent connections (`duplicate column name`
  tolerated). *(§4.3)*
- `_COLUMNS` extended so provenance columns surface. *(§4.2)*
- `affected` from `rowcount`; single-`UPDATE` `COALESCE` provenance; blank-label rejected in Store;
  files-first idempotent delete; scoped empty-dir sweep skipping the current day; short DB txns
  with file I/O outside them. *(§6)*
- Pagination drift acknowledged + mitigated by in-place updates and the unreviewed-page-1
  workflow. *(§8.4)*

---

## 11. Retention interaction (`scripts/prune.py`)

- After lifting the helpers, `prune.py` imports `_safe_unlink` / `_prune_empty_dirs` from
  `wildlife.store` (one implementation).
- **Spare reviewed captures from the confidence rule.** `prune`'s deletion predicate is
  `capture_ts < cutoff OR (min_confidence_keep > 0 AND confidence < min_confidence_keep)` using the
  *original model* confidence. A human who confirms a rare-species capture that happened to have
  low model confidence would otherwise get it silently auto-pruned. Change the confidence branch to
  exclude `reviewed = 1` rows (`… AND confidence < ? AND reviewed = 0`). **Age-based retention still
  applies** to reviewed rows (disk is finite; document that you should export curated captures if
  you want them permanently). Add a prune test: a reviewed low-confidence row survives the
  confidence rule; an unreviewed one doesn't.
- **Why no `false_positive` sentinel:** overloading the `NOT NULL label` column with a disposition
  value leaks it into `distinct_labels()`, which powers the **public** gallery's Class filter and
  every card's visible label — surfacing an internal review verdict into the open browse UI and
  conflating "species" with "disposition." It also has undefined retention (a `false_positive` with
  0.95 confidence would be retained exactly like a real animal). Delete is the disposal path. If a
  "keep as hard-negative for training" bucket is wanted later, add a dedicated `disposition` column
  (kept out of `distinct_labels()`), not a magic label.

---

## 12. Testing plan

**Store unit tests (`tests/test_store.py`):**
- `delete`: removes row + both files; returns `True`; missing files tolerated; unknown id →
  `False` (no exception); traversal-guarded path refused but row still deleted.
- `delete_many`: `affected == rows actually removed` via rowcount, incl. some-missing / duplicate
  ids; files unlinked via `_safe_unlink`.
- `update_label`: sets label; `original_label` = COALESCE (a→b→c keeps `'a'`); `reviewed=1`,
  `reviewed_at` set; returns updated dict **including** new columns; unknown id → `None`; blank
  label → `ValueError`; same-label is defined (touched + reviewed).
- `update_label_many` / `mark_reviewed_many` rowcount semantics (`mark_reviewed_many` counts only
  `reviewed=0→1`).
- **New columns present** in `get()`/`query()` output (guards the `_COLUMNS` omission).
- **Migration idempotent**: `init_schema()` twice on a pre-populated DB; and on two connections
  against a pre-populated DB (duplicate-column tolerated); existing rows get `reviewed=0`.
- **`busy_timeout` is set on a fresh (non-boot) Store** (`PRAGMA busy_timeout` > 0).
- `count(...)` matches `len(query(...))` across filters incl. `reviewed`.

**Prune tests (`tests/`):** reviewed low-confidence row survives `min_confidence_keep`; unreviewed
one is pruned; age rule still deletes reviewed rows.

**Flask route tests (`tests/test_admin_routes.py` style — `_auth()` Basic header, `ctx` fixture):**
- **401** without Basic Auth on `GET /admin/captures`, `POST …/delete`, `…/reclassify`,
  `…/bulk`.
- **403** cross-origin (`Origin` mismatch) POST on `…/delete`, `…/reclassify`, `…/bulk`.
- **405** GET on the POST-only mutation routes.
- delete route removes row + files, flashes `ok`, 302.
- reclassify updates label + provenance, 302; invalid label → err re-render.
- bulk: happy-path delete / reclassify / mark_reviewed → JSON `affected`; bad `action` → error;
  **non-JSON body → 400/415**; **non-int id → 400**; **> cap ids → 400**; label not in
  allow-list → error.
- grid renders; reviewed badge shows; empty state renders.

---

## 13. Docs to update

- **`README.md`** — add a "Managing captures" subsection under Admin: how to reach
  `/admin/captures`, that delete is **permanent** and removes both files (mirrors `prune`),
  reclassify records provenance, the reviewed filter / bulk actions, no undo.
- **`spec.md`** — §6.5 schema (three new columns), the gallery/admin section (new routes), and the
  build order.
- **`config.example.yaml`** — no new config; a one-line comment near `retention:` noting reviewed
  rows are spared the `min_confidence_keep` rule.

---

## 14. Build order

1. **Store foundation** — lift helpers into `store.py`; `busy_timeout`/`synchronous` in
   `__init__`; migration + `_SCHEMA_SQL` + `_COLUMNS`; `delete`/`delete_many`/`update_label`/
   `update_label_many`/`mark_reviewed_many`/`count`. Unit tests. *(No UI yet — the hard part,
   fully testable in isolation.)*
2. **`prune.py` refactor** — import shared helpers; reviewed-exemption; prune tests.
3. **Admin wiring** — extend `create_admin_blueprint` signature + registration; serialize helper;
   the four routes. Route tests (auth/CSRF/405/validation) before the template.
4. **Template + CSS + nav** — `captures.html` (grid, selection, bulk toolbar, reclassify, lightbox,
   feedback, empty states, pagination); `style.css` additions; `admin/base.html` nav link +
   `main_class` block.
5. **Docs** — README, spec.md, config comment.

Each step is independently verifiable; step 1 carries the correctness risk and is pure Store/DB.

---

## 15. Open questions for the reviewer

See the six flagged decisions in §2. In particular: confirm (a) dropping `false_positive`,
(b) including the review-triage surface, and (c) the prune reviewed-exemption — those three shape
the most code.
