# Admin Capture Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a password-gated `/admin/captures` view to delete, reclassify, and review-triage detected captures, backed by new `Store` methods and admin routes.

**Architecture:** All DB/file mutation logic lives in `wildlife.store.Store` (the integrity boundary); thin admin-blueprint routes call it. Deletion reuses the file-safety helpers lifted out of `scripts/prune.py`. The UI is a server-rendered grid (works without JS) enhanced with a vanilla-JS selection + bulk-action layer, following the existing gallery/admin conventions.

**Tech Stack:** Python 3.11–3.13, Flask, SQLite (stdlib `sqlite3`), Pillow + numpy (in `store.py` only), Jinja templates, one hand-written stylesheet, vanilla JS. `pytest` for tests.

## Global Constraints

- **`store.py` dependency floor:** stdlib + `numpy` + `Pillow` only — never import torch/cv2/ultralytics/Flask into it.
- **All SQL uses `?` placeholders** with a params list; dynamic `IN (…)` clauses are built as `",".join("?" for _ in chunk)` with values bound — never string-formatted.
- **All Store writes** acquire `self._write_lock`, run a short transaction, and `commit()`; file I/O happens outside the lock/transaction.
- **File deletion** always goes through `_safe_unlink` (path-traversal-guarded, missing-file-tolerant) and never aborts the DB delete on unlink failure.
- **New admin routes live on the `admin` blueprint** (never on the gallery `app`), are POST-only for mutations, and inherit the existing `before_request` auth + same-origin CSRF guard.
- **Tests are hardware-free** (no network/GPU); storage points under `tmp_path`.
- **Follow existing patterns:** row-as-dict (`_COLUMNS`), `flash('ok'|'err')` + PRG for form posts, `jsonify` for the AJAX bulk route, `_serialize`-style payloads, existing CSS tokens/classes.
- **Reference spec:** `docs/superpowers/specs/2026-07-02-admin-capture-management-design.md`.

---

## File Structure

**Modified:**
- `src/wildlife/store.py` — schema columns + migration + per-connection PRAGMAs; lifted `_safe_unlink`/`_prune_empty_dirs`; new `_chunked`/`_sweep_empty_capture_dirs`/`_now_iso`; `delete`/`delete_many`/`update_label`/`update_label_many`/`mark_reviewed_many`/`count`; `query` gains a `reviewed` filter.
- `scripts/prune.py` — import the lifted helpers; reviewed-aware retention predicate.
- `src/wildlife/admin/routes.py` — extended factory signature; capture routes; module-level helpers.
- `src/wildlife/gallery/app.py` — pass `get_store` into `create_admin_blueprint`.
- `src/wildlife/gallery/templates/admin/base.html` — "Captures" nav link + a `main_class` block.
- `src/wildlife/gallery/static/style.css` — capture-management styles.
- `README.md`, `spec.md`, `config.example.yaml` — docs.

**Created:**
- `src/wildlife/gallery/templates/admin/captures.html` — the management page.
- Tests slot into existing `tests/test_store.py`, `tests/test_admin_routes.py`, and a new `tests/test_prune.py`.

---

## Task 1: Schema evolution — columns, migration, per-connection PRAGMAs

**Files:**
- Modify: `src/wildlife/store.py` (`_SCHEMA_SQL` 32-48, `_COLUMNS` 51-66, `__init__` 97-118, `init_schema` 121-128)
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: three new columns (`original_label TEXT`, `reviewed INTEGER NOT NULL DEFAULT 0`, `reviewed_at TEXT`) surfaced in every row dict; `Store._migrate()`; per-connection `PRAGMA busy_timeout=5000`. Consumed by all later Store methods and the admin routes.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_store.py`:

```python
def test_new_columns_present_and_defaulted(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        cid = store.save_capture(
            camera_id="cam", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
        )
        row = store.get(cid)
        assert row["reviewed"] == 0
        assert row["original_label"] is None
        assert "reviewed_at" in row
    finally:
        store.close()


def test_busy_timeout_set_per_connection(tmp_path: Path) -> None:
    # A Store that never called init_schema must still have busy_timeout.
    store = Store(db_path=tmp_path / "captures.db", captures_dir=tmp_path / "captures")
    try:
        assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        store.close()


def test_migration_adds_columns_to_old_db(tmp_path: Path) -> None:
    import sqlite3
    db = tmp_path / "captures.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL, event_ts TEXT NOT NULL, capture_ts TEXT NOT NULL,
            label TEXT NOT NULL, confidence REAL NOT NULL,
            box_x1 REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL,
            image_path TEXT NOT NULL, thumb_path TEXT NOT NULL,
            width INTEGER, height INTEGER
        );
        INSERT INTO captures (camera_id,event_ts,capture_ts,label,confidence,image_path,thumb_path)
        VALUES ('cam','2020-01-01T00:00:00','2020-01-01T00:00:00','deer',0.9,'a.jpg','a_thumb.jpg');
        """
    )
    conn.commit()
    conn.close()

    store = Store(db_path=db, captures_dir=tmp_path / "captures")
    try:
        store.init_schema()  # ALTER-adds the 3 columns without error
        row = store.get(1)
        assert row["reviewed"] == 0
        assert row["original_label"] is None
        assert "reviewed_at" in row
        store.init_schema()  # idempotent second run — no error
    finally:
        store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_store.py::test_new_columns_present_and_defaulted tests/test_store.py::test_busy_timeout_set_per_connection tests/test_store.py::test_migration_adds_columns_to_old_db -v`
Expected: FAIL (`KeyError: 'reviewed'` / `busy_timeout` is `0`).

- [ ] **Step 3: Add the columns to `_SCHEMA_SQL`**

In `src/wildlife/store.py`, change the `CREATE TABLE` tail (line 43) from:

```python
    width INTEGER, height INTEGER
);
```

to:

```python
    width INTEGER, height INTEGER,
    original_label TEXT,               -- model's label before a human reclassify
    reviewed       INTEGER NOT NULL DEFAULT 0,
    reviewed_at    TEXT                -- ISO8601 of the last human action
);
```

- [ ] **Step 4: Extend `_COLUMNS` and add the migration list**

Append to the `_COLUMNS` tuple (after `"height",` at line 65):

```python
    "original_label",
    "reviewed",
    "reviewed_at",
)
```

Immediately below the `_COLUMNS` definition, add:

```python
# Columns added after the original schema; applied idempotently by _migrate().
_COLUMN_ADDITIONS: tuple[tuple[str, str], ...] = (
    ("original_label", "TEXT"),
    ("reviewed", "INTEGER NOT NULL DEFAULT 0"),
    ("reviewed_at", "TEXT"),
)
```

- [ ] **Step 5: Set per-connection PRAGMAs in `__init__`**

In `__init__`, replace the connection block (lines 115-118):

```python
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Serialize writes from this instance; reads remain concurrent under WAL.
        self._write_lock = threading.Lock()
```

with:

```python
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Per-connection PRAGMAs. busy_timeout MUST be set on every connection
        # (not just the boot/worker one) so a gallery/admin write waits for the
        # worker's WAL writer instead of failing immediately with "database is
        # locked". journal_mode=WAL is persistent (file header) so it stays in
        # init_schema; busy_timeout/synchronous are per-connection and do not.
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        # Serialize writes from THIS instance; cross-connection writes are
        # arbitrated by SQLite WAL + busy_timeout, not by this lock.
        self._write_lock = threading.Lock()
```

- [ ] **Step 6: Add the migration to `init_schema` and define `_migrate`**

Replace `init_schema` (lines 121-128) with:

```python
    def init_schema(self) -> None:
        """Create the captures table + indexes, enable WAL, and migrate (idempotent)."""
        with self._write_lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.executescript(_SCHEMA_SQL)
            self._migrate()
            self._conn.commit()
        logger.debug("Initialised capture schema at %s", self.db_path)

    def _migrate(self) -> None:
        """Add any missing post-original columns. Safe across re-runs and races."""
        existing = {r["name"] for r in self._conn.execute("PRAGMA table_info(captures)")}
        for name, decl in _COLUMN_ADDITIONS:
            if name in existing:
                continue
            try:
                self._conn.execute(f"ALTER TABLE captures ADD COLUMN {name} {decl}")
            except sqlite3.OperationalError as exc:
                # Another process (worker vs gallery boot) added it between the
                # check and the ALTER — a schema error busy_timeout can't absorb.
                if "duplicate column name" not in str(exc).lower():
                    raise
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_store.py -v`
Expected: PASS (all, including the existing suite).

- [ ] **Step 8: Commit**

```bash
git add src/wildlife/store.py tests/test_store.py
git commit -m "store: add provenance/review columns + idempotent migration + per-conn busy_timeout"
```

---

## Task 2: Lift the file-deletion helpers into the package

**Files:**
- Modify: `src/wildlife/store.py` (add imports + module-level helpers)
- Modify: `scripts/prune.py` (delete local copies 83-132, import from `wildlife.store`)
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `wildlife.store._safe_unlink(captures_dir: Path, rel: str | None) -> tuple[bool, int]` and `wildlife.store._prune_empty_dirs(root: Path) -> int`. Consumed by Task 3 and `scripts/prune.py`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_store.py`:

```python
def test_safe_unlink_removes_and_guards(tmp_path: Path) -> None:
    from wildlife.store import _safe_unlink

    captures = tmp_path / "captures"
    (captures / "2020" / "01" / "01").mkdir(parents=True)
    f = captures / "2020" / "01" / "01" / "x.jpg"
    f.write_bytes(b"hello")

    # Deletes a real file, reports bytes freed.
    deleted, size = _safe_unlink(captures, "2020/01/01/x.jpg")
    assert deleted is True and size == 5 and not f.exists()

    # Missing file is tolerated (idempotent), not an error.
    assert _safe_unlink(captures, "2020/01/01/x.jpg") == (False, 0)

    # Path escaping captures_dir is refused.
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    assert _safe_unlink(captures, "../secret.txt") == (False, 0)
    assert outside.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store.py::test_safe_unlink_removes_and_guards -v`
Expected: FAIL (`ImportError: cannot import name '_safe_unlink'`).

- [ ] **Step 3: Add `import os` and the lifted helpers to `store.py`**

At the top of `src/wildlife/store.py`, add `import os` to the stdlib imports (after `import logging`). Then add these module-level functions just below `_sanitize` (after line 78):

```python
def _safe_unlink(captures_dir: Path, rel: str | None) -> tuple[bool, int]:
    """Delete ``captures_dir/rel`` if present and safely inside ``captures_dir``.

    Returns ``(deleted, bytes_freed)``. Missing files count as not-deleted with
    zero bytes (keeps the operation idempotent). Paths that resolve outside the
    captures tree are refused as a safety check.
    """
    if not rel:
        return (False, 0)

    target = captures_dir / rel
    root = captures_dir.resolve()
    try:
        resolved = target.resolve()
    except OSError:
        return (False, 0)

    if root != resolved and root not in resolved.parents:
        logger.warning("Refusing to delete path outside captures_dir: %s", target)
        return (False, 0)

    if not resolved.exists():
        return (False, 0)

    try:
        size = resolved.stat().st_size
        resolved.unlink()
        return (True, size)
    except OSError as exc:
        logger.warning("Could not delete %s: %s", resolved, exc)
        return (False, 0)


def _prune_empty_dirs(root: Path) -> int:
    """Remove now-empty subdirectories under ``root`` (bottom-up). Returns count."""
    if not root.is_dir():
        return 0
    removed = 0
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        directory = Path(dirpath)
        if directory == root:
            continue
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
                removed += 1
        except OSError:
            # Non-empty or vanished concurrently; leave it be.
            pass
    return removed
```

- [ ] **Step 4: Replace the copies in `scripts/prune.py` with an import**

In `scripts/prune.py`, delete the two function definitions `_safe_unlink` (lines 83-113) and `_prune_empty_dirs` (lines 116-132). Then change the import line 45 from:

```python
from wildlife.config import load_config  # noqa: E402
```

to:

```python
from wildlife.config import load_config  # noqa: E402
from wildlife.store import _prune_empty_dirs, _safe_unlink  # noqa: E402
```

(`import os` at line 29 stays; it is still used elsewhere. `_table_exists` and the rest are untouched.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_store.py::test_safe_unlink_removes_and_guards tests/ -k "prune or store" -v`
Expected: PASS (the new test plus any existing prune/store tests).

- [ ] **Step 6: Commit**

```bash
git add src/wildlife/store.py scripts/prune.py tests/test_store.py
git commit -m "store: lift _safe_unlink/_prune_empty_dirs into the package; prune imports them"
```

---

## Task 3: `Store.delete` and `delete_many`

**Files:**
- Modify: `src/wildlife/store.py` (add helpers + methods after `distinct_labels`, before `close`)
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `_safe_unlink` (Task 2), the new columns (Task 1).
- Produces: `Store.delete(capture_id: int) -> bool`, `Store.delete_many(ids: Iterable[int]) -> int`, and module-level `_chunked` / `_sweep_empty_capture_dirs`. Consumed by the admin delete routes (Task 7).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_store.py`:

```python
def test_delete_removes_row_and_files(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        cid = store.save_capture(
            camera_id="cam", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
        )
        row = store.get(cid)
        img = store.captures_dir / row["image_path"]
        thumb = store.captures_dir / row["thumb_path"]
        assert img.exists() and thumb.exists()

        assert store.delete(cid) is True
        assert store.get(cid) is None
        assert not img.exists() and not thumb.exists()
        # The now-empty dated dir is swept.
        assert not (store.captures_dir / "2020" / "01" / "01").exists()
    finally:
        store.close()


def test_delete_unknown_id_and_missing_files(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        assert store.delete(999) is False  # unknown id, no exception
        cid = store.save_capture(
            camera_id="cam", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
        )
        row = store.get(cid)
        (store.captures_dir / row["image_path"]).unlink()  # file already gone
        assert store.delete(cid) is True  # tolerates the missing file
        assert store.get(cid) is None
    finally:
        store.close()


def test_delete_many_counts_only_removed(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        ids = [
            store.save_capture(
                camera_id="cam", event_ts=datetime(2020, 1, 1),
                capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
            )
            for _ in range(3)
        ]
        # Two real ids + one bogus + one duplicate -> 3 rows actually removed.
        removed = store.delete_many([ids[0], ids[1], 999, ids[0], ids[2]])
        assert removed == 3
        assert all(store.get(i) is None for i in ids)
    finally:
        store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_store.py::test_delete_removes_row_and_files tests/test_store.py::test_delete_unknown_id_and_missing_files tests/test_store.py::test_delete_many_counts_only_removed -v`
Expected: FAIL (`AttributeError: 'Store' object has no attribute 'delete'`).

- [ ] **Step 3: Add `Iterable` import and the module-level helpers**

In `src/wildlife/store.py`, change `from typing import Any` to:

```python
from typing import Any, Iterable
```

Add these module-level helpers just below `_prune_empty_dirs`:

```python
def _chunked(seq: list, size: int) -> Iterable[list]:
    """Yield ``seq`` in lists of at most ``size`` (for SQLite IN-clause limits)."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _sweep_empty_capture_dirs(captures_dir: Path, rels: Iterable[str]) -> int:
    """Remove now-empty ``YYYY/MM/DD`` dirs for the given relative file paths.

    Scoped to the deleted captures' own dated directories (not the whole tree),
    and skips the current day's directory because the worker may be writing
    there. Walks up MM/YYYY so an emptied day frees its parents too. Returns the
    number of directories removed. Never raises.
    """
    now = datetime.now()
    today = (captures_dir / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}").resolve()
    root = captures_dir.resolve()
    day_dirs = {(captures_dir / rel).parent for rel in rels if rel}
    removed = 0
    for day in day_dirs:
        for directory in (day, day.parent, day.parent.parent):
            try:
                resolved = directory.resolve()
            except OSError:
                continue
            if resolved == today or resolved == root or root not in resolved.parents:
                continue
            try:
                if not any(directory.iterdir()):
                    directory.rmdir()
                    removed += 1
            except OSError:
                pass
    return removed
```

- [ ] **Step 4: Add `delete` and `delete_many` methods**

In `src/wildlife/store.py`, add these methods to the `Store` class just after `distinct_labels` (after line 321), before the `close` method:

```python
    # ------------------------------------------------------------- delete path
    def delete(self, capture_id: int) -> bool:
        """Delete one capture: unlink both files, then remove the row.

        Files are removed first (idempotently, path-guarded); the DB delete then
        proceeds regardless of unlink outcome. Returns True if a row was removed.
        """
        row = self.get(capture_id)
        if row is None:
            return False
        rels = [row.get("image_path"), row.get("thumb_path")]
        for rel in rels:
            _safe_unlink(self.captures_dir, rel)
        with self._write_lock:
            cur = self._conn.execute("DELETE FROM captures WHERE id = ?", (capture_id,))
            self._conn.commit()
        _sweep_empty_capture_dirs(self.captures_dir, [r for r in rels if r])
        return cur.rowcount == 1

    def delete_many(self, ids: Iterable[int]) -> int:
        """Delete many captures by id. Returns the number of rows actually removed."""
        id_list = [int(i) for i in ids]
        if not id_list:
            return 0
        total = 0
        swept: list[str] = []
        for chunk in _chunked(id_list, 900):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT image_path, thumb_path FROM captures WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            for r in rows:
                for key in ("image_path", "thumb_path"):
                    _safe_unlink(self.captures_dir, r[key])
                    if r[key]:
                        swept.append(r[key])
            with self._write_lock:
                cur = self._conn.execute(
                    f"DELETE FROM captures WHERE id IN ({placeholders})", chunk
                )
                self._conn.commit()
                total += cur.rowcount
        _sweep_empty_capture_dirs(self.captures_dir, swept)
        return total
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_store.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/wildlife/store.py tests/test_store.py
git commit -m "store: add delete/delete_many with scoped empty-dir sweep"
```

---

## Task 4: `Store.update_label`, `update_label_many`, `mark_reviewed_many`

**Files:**
- Modify: `src/wildlife/store.py` (add `_now_iso` helper + three methods)
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: the new columns (Task 1), `_chunked` (Task 3).
- Produces: `update_label(capture_id: int, new_label: str) -> dict | None`, `update_label_many(ids, new_label: str) -> int`, `mark_reviewed_many(ids) -> int`, `_now_iso() -> str`. Consumed by the admin reclassify/bulk routes (Task 7).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_store.py`:

```python
import pytest  # add at top of file if not already imported


def test_update_label_records_provenance_once(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        cid = store.save_capture(
            camera_id="cam", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(label="bird"),
        )
        r1 = store.update_label(cid, "deer")
        assert r1["label"] == "deer"
        assert r1["original_label"] == "bird"   # captured on first reclassify
        assert r1["reviewed"] == 1
        assert r1["reviewed_at"]                 # timestamp set

        r2 = store.update_label(cid, "elk")      # a -> b -> c
        assert r2["label"] == "elk"
        assert r2["original_label"] == "bird"    # NOT overwritten
    finally:
        store.close()


def test_update_label_validation_and_missing(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        cid = store.save_capture(
            camera_id="cam", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
        )
        assert store.update_label(999, "deer") is None   # unknown id
        with pytest.raises(ValueError):
            store.update_label(cid, "   ")                 # blank label
    finally:
        store.close()


def test_bulk_update_and_mark_reviewed(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        ids = [
            store.save_capture(
                camera_id="cam", event_ts=datetime(2020, 1, 1),
                capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
            )
            for _ in range(3)
        ]
        assert store.update_label_many(ids, "deer") == 3
        assert all(store.get(i)["label"] == "deer" for i in ids)
        assert all(store.get(i)["reviewed"] == 1 for i in ids)

        # All already reviewed -> mark_reviewed_many counts only 0->1 transitions.
        assert store.mark_reviewed_many(ids) == 0
    finally:
        store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_store.py::test_update_label_records_provenance_once tests/test_store.py::test_update_label_validation_and_missing tests/test_store.py::test_bulk_update_and_mark_reviewed -v`
Expected: FAIL (`AttributeError: ... 'update_label'`).

- [ ] **Step 3: Add `_now_iso` and the three methods**

In `src/wildlife/store.py`, add a module-level helper next to `_iso` (after line 73):

```python
def _now_iso() -> str:
    """Timestamp for human edits, matching the ISO8601 TEXT convention."""
    return datetime.now().isoformat(timespec="seconds")
```

Add these methods to the `Store` class immediately after `delete_many` (from Task 3):

```python
    # ------------------------------------------------------------ update path
    def update_label(self, capture_id: int, new_label: str) -> dict[str, Any] | None:
        """Reclassify one capture. Records ``original_label`` on first edit.

        Returns the updated row dict, or None if ``capture_id`` does not exist.
        Raises ValueError on a blank label (the DB column is NOT NULL and must
        never receive an empty string).
        """
        label = (new_label or "").strip()
        if not label:
            raise ValueError("new_label must be a non-empty string")
        with self._write_lock:
            cur = self._conn.execute(
                """
                UPDATE captures
                   SET original_label = COALESCE(original_label, label),
                       label = ?, reviewed = 1, reviewed_at = ?
                 WHERE id = ?
                """,
                (label, _now_iso(), capture_id),
            )
            self._conn.commit()
        if cur.rowcount == 0:
            return None
        return self.get(capture_id)

    def update_label_many(self, ids: Iterable[int], new_label: str) -> int:
        """Reclassify many captures. Returns the number of rows touched."""
        label = (new_label or "").strip()
        if not label:
            raise ValueError("new_label must be a non-empty string")
        id_list = [int(i) for i in ids]
        if not id_list:
            return 0
        now = _now_iso()
        total = 0
        with self._write_lock:
            for chunk in _chunked(id_list, 900):
                placeholders = ",".join("?" for _ in chunk)
                cur = self._conn.execute(
                    f"""
                    UPDATE captures
                       SET original_label = COALESCE(original_label, label),
                           label = ?, reviewed = 1, reviewed_at = ?
                     WHERE id IN ({placeholders})
                    """,
                    [label, now, *chunk],
                )
                total += cur.rowcount
            self._conn.commit()
        return total

    def mark_reviewed_many(self, ids: Iterable[int]) -> int:
        """Mark many captures reviewed without changing their label.

        Returns the number newly marked (rows already reviewed are not counted).
        """
        id_list = [int(i) for i in ids]
        if not id_list:
            return 0
        now = _now_iso()
        total = 0
        with self._write_lock:
            for chunk in _chunked(id_list, 900):
                placeholders = ",".join("?" for _ in chunk)
                cur = self._conn.execute(
                    f"UPDATE captures SET reviewed = 1, reviewed_at = ? "
                    f"WHERE id IN ({placeholders}) AND reviewed = 0",
                    [now, *chunk],
                )
                total += cur.rowcount
            self._conn.commit()
        return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wildlife/store.py tests/test_store.py
git commit -m "store: add update_label/update_label_many/mark_reviewed_many with COALESCE provenance"
```

---

## Task 5: `Store.count` + `reviewed` filter on `query`

**Files:**
- Modify: `src/wildlife/store.py` (extract `_build_filters`, extend `query`, add `count`)
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `query(..., reviewed: bool | None = None)` and `count(*, camera_id=None, label=None, start=None, end=None, min_confidence=None, reviewed=None) -> int`. Consumed by the admin captures list route (Task 7).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_store.py`:

```python
def test_count_and_reviewed_filter(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        ids = [
            store.save_capture(
                camera_id="cam", event_ts=datetime(2020, 1, 1),
                capture_ts=datetime(2020, 1, 1), frame=_make_frame(), det=_det(),
            )
            for _ in range(3)
        ]
        assert store.count() == 3
        store.update_label(ids[0], "deer")  # marks one reviewed

        assert store.count(reviewed=True) == 1
        assert store.count(reviewed=False) == 2
        assert len(store.query(reviewed=False)) == 2
        assert len(store.query(reviewed=True)) == 1
        # count mirrors query length under the same filters.
        assert store.count(camera_id="cam") == len(store.query(camera_id="cam", limit=1000))
    finally:
        store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store.py::test_count_and_reviewed_filter -v`
Expected: FAIL (`TypeError: query() got an unexpected keyword argument 'reviewed'`).

- [ ] **Step 3: Extract the filter builder and extend `query`**

In `src/wildlife/store.py`, add a module-level helper just above the `Store` class (after `_sanitize`/helpers, before `class Store`):

```python
def _build_filters(
    *,
    camera_id: str | None,
    label: str | None,
    start: datetime | str | None,
    end: datetime | str | None,
    min_confidence: float | None,
    reviewed: bool | None,
) -> tuple[list[str], list[Any]]:
    """Build the shared AND-combined WHERE clauses + params for query/count."""
    clauses: list[str] = []
    params: list[Any] = []
    if camera_id is not None:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if label is not None:
        clauses.append("label = ?")
        params.append(label)
    if start is not None:
        clauses.append("capture_ts >= ?")
        params.append(_iso(start))
    if end is not None:
        clauses.append("capture_ts <= ?")
        params.append(_iso(end))
    if min_confidence is not None:
        clauses.append("confidence >= ?")
        params.append(float(min_confidence))
    if reviewed is not None:
        clauses.append("reviewed = ?")
        params.append(1 if reviewed else 0)
    return clauses, params
```

Replace the body of `query` (lines 281-307, the `clauses`/`params` construction through the return) so its signature gains `reviewed` and it delegates to `_build_filters`. The new `query` reads:

```python
    def query(
        self,
        *,
        camera_id: str | None = None,
        label: str | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        min_confidence: float | None = None,
        reviewed: bool | None = None,
        limit: int = 60,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return matching rows as dicts, newest first.

        All filters are optional and AND-combined. ``start``/``end`` bound
        ``capture_ts`` inclusively; ``reviewed`` filters on the human-review flag.
        Results are ordered by ``capture_ts`` desc (ties by ``id`` desc) and
        paginated via ``limit``/``offset``.
        """
        clauses, params = _build_filters(
            camera_id=camera_id, label=label, start=start, end=end,
            min_confidence=min_confidence, reviewed=reviewed,
        )
        sql = "SELECT * FROM captures"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY capture_ts DESC, id DESC LIMIT ? OFFSET ?"
        params.extend((int(limit), int(offset)))
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]
```

- [ ] **Step 4: Add `count`**

Add immediately after `query`:

```python
    def count(
        self,
        *,
        camera_id: str | None = None,
        label: str | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        min_confidence: float | None = None,
        reviewed: bool | None = None,
    ) -> int:
        """Return the number of rows matching the same filters as :meth:`query`."""
        clauses, params = _build_filters(
            camera_id=camera_id, label=label, start=start, end=end,
            min_confidence=min_confidence, reviewed=reviewed,
        )
        sql = "SELECT COUNT(*) FROM captures"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return int(self._conn.execute(sql, params).fetchone()[0])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_store.py -v`
Expected: PASS (including existing `query` tests — the refactor is behavior-preserving).

- [ ] **Step 6: Commit**

```bash
git add src/wildlife/store.py tests/test_store.py
git commit -m "store: add count() and a reviewed filter (shared _build_filters)"
```

---

## Task 6: Prune spares reviewed low-confidence captures

**Files:**
- Modify: `scripts/prune.py` (add `_has_reviewed_column` + `_build_where`; use them in `main`)
- Test: `tests/test_prune.py` (new)

**Interfaces:**
- Consumes: the `reviewed` column (Task 1), a Store-created DB.
- Produces: `scripts/prune.py::_build_where(cutoff_iso: str, min_conf: float, has_reviewed: bool) -> tuple[str, list]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prune.py`:

```python
"""Tests for the retention prune predicate, incl. the reviewed exemption."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np

from wildlife.models import Detection
from wildlife.store import Store

# scripts/ is not an importable package; load prune.py by path.
_PRUNE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "prune.py"
_spec = importlib.util.spec_from_file_location("prune", _PRUNE_PATH)
prune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune)


def test_build_where_variants() -> None:
    assert prune._build_where("2020-01-01", 0.0, True) == ("capture_ts < ?", ["2020-01-01"])
    assert prune._build_where("2020-01-01", 0.5, False) == (
        "(capture_ts < ? OR confidence < ?)", ["2020-01-01", 0.5],
    )
    assert prune._build_where("2020-01-01", 0.5, True) == (
        "(capture_ts < ? OR (confidence < ? AND reviewed = 0))", ["2020-01-01", 0.5],
    )


def _frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_reviewed_low_conf_capture_is_spared(tmp_path: Path) -> None:
    store = Store(db_path=tmp_path / "captures.db", captures_dir=tmp_path / "captures")
    store.init_schema()
    try:
        recent = datetime.now()  # not old enough for the age rule
        det = Detection(label="deer", confidence=0.2, box_xyxy=(1, 1, 5, 5), box_area_frac=0.1)
        kept = store.save_capture(camera_id="c", event_ts=recent, capture_ts=recent, frame=_frame(), det=det)
        pruned = store.save_capture(camera_id="c", event_ts=recent, capture_ts=recent, frame=_frame(), det=det)
        store.update_label(kept, "deer")  # marks reviewed (confidence stays 0.2)

        where, params = prune._build_where("1970-01-01T00:00:00", 0.5, True)
        matched = {
            r["id"]
            for r in store._conn.execute(
                f"SELECT id FROM captures WHERE {where}", params
            ).fetchall()
        }
        assert pruned in matched      # unreviewed low-conf is pruned
        assert kept not in matched    # reviewed low-conf is spared
    finally:
        store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prune.py -v`
Expected: FAIL (`AttributeError: module 'prune' has no attribute '_build_where'`).

- [ ] **Step 3: Add the helpers to `scripts/prune.py`**

Add these two functions to `scripts/prune.py` just after `_table_exists` (after line 80):

```python
def _has_reviewed_column(conn: sqlite3.Connection) -> bool:
    """True if the captures table has the ``reviewed`` column (post-migration)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(captures)")}
    return "reviewed" in cols


def _build_where(cutoff_iso: str, min_conf: float, has_reviewed: bool) -> tuple[str, list]:
    """Build the prune match predicate.

    Age always applies. When ``min_conf > 0`` the confidence rule applies too,
    but human-reviewed rows are exempt from it (a curator confirmed the capture,
    so a low *model* confidence must not auto-delete it). Age retention still
    applies to reviewed rows.
    """
    if min_conf <= 0.0:
        return ("capture_ts < ?", [cutoff_iso])
    if has_reviewed:
        return ("(capture_ts < ? OR (confidence < ? AND reviewed = 0))", [cutoff_iso, float(min_conf)])
    return ("(capture_ts < ? OR confidence < ?)", [cutoff_iso, float(min_conf)])
```

- [ ] **Step 4: Use them in `main`**

In `scripts/prune.py::main`, replace the inline predicate block (lines 177-182):

```python
    # Build the match predicate (shared by SELECT and DELETE).
    where = "capture_ts < ?"
    params: list[object] = [cutoff_iso]
    if min_conf > 0.0:
        where = "(capture_ts < ? OR confidence < ?)"
        params = [cutoff_iso, float(min_conf)]

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn):
            print("Database has no 'captures' table; nothing to prune.")
            return 0
```

with:

```python
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn):
            print("Database has no 'captures' table; nothing to prune.")
            return 0

        # Build the match predicate (shared by SELECT and DELETE); reviewed rows
        # are exempt from the confidence rule when the column is present.
        where, params = _build_where(cutoff_iso, min_conf, _has_reviewed_column(conn))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_prune.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/prune.py tests/test_prune.py
git commit -m "prune: spare human-reviewed captures from the min_confidence rule"
```

---

## Task 7: Admin capture routes + wiring + server-rendered page

**Files:**
- Modify: `src/wildlife/admin/routes.py` (factory signature, capture routes, module-level helpers, `current_app` import, gallery-parser imports)
- Modify: `src/wildlife/gallery/app.py:392` (pass `get_store`)
- Modify: `src/wildlife/gallery/templates/admin/base.html` (nav link + `main_class` block)
- Create: `src/wildlife/gallery/templates/admin/captures.html` (server-rendered grid + per-card forms + filters + pagination + empty state)
- Modify: `src/wildlife/gallery/static/style.css` (`.admin-main-wide`, reviewed pill)
- Test: `tests/test_admin_routes.py`

**Interfaces:**
- Consumes: all Store methods (Tasks 1-5).
- Produces: endpoints `admin.captures_index` (GET `/admin/captures`), `admin.capture_delete` (POST `/admin/captures/<int:capture_id>/delete`), `admin.capture_reclassify` (POST `/admin/captures/<int:capture_id>/reclassify`), `admin.captures_bulk` (POST `/admin/captures/bulk`). The template + JS layer (Task 8) builds on the `admin.captures_index` page.

- [ ] **Step 1: Write the failing route tests**

Add to `tests/test_admin_routes.py`. First, add a small helper that seeds a capture (put it near the top, after `_auth`):

```python
def _seed_capture(cfgp: Path) -> int:
    """Insert one capture into the configured DB; return its id."""
    from datetime import datetime
    import numpy as np
    from wildlife.config import load_config
    from wildlife.models import Detection
    from wildlife.store import Store

    cfg = load_config(cfgp)
    store = Store(db_path=cfg.storage.db_path, captures_dir=cfg.storage.captures_dir)
    store.init_schema()
    try:
        return store.save_capture(
            camera_id="north_field", event_ts=datetime(2020, 1, 1),
            capture_ts=datetime(2020, 1, 1),
            frame=np.zeros((120, 160, 3), dtype=np.uint8),
            det=Detection(label="deer", confidence=0.8, box_xyxy=(1, 1, 5, 5), box_area_frac=0.1),
        )
    finally:
        store.close()
```

Then the tests:

```python
def _origin(host: str = "localhost") -> dict:
    return {"Origin": f"http://{host}"}


def test_captures_auth_and_render(ctx) -> None:
    client, cfgp = ctx
    _seed_capture(cfgp)
    # Auth required.
    assert client.get("/admin/captures").status_code == 401
    assert client.post("/admin/captures/bulk", json={}).status_code == 401
    # Renders when authed.
    r = client.get("/admin/captures", headers=_auth())
    assert r.status_code == 200
    assert b"deer" in r.data


def test_capture_delete_route(ctx) -> None:
    client, cfgp = ctx
    cid = _seed_capture(cfgp)
    r = client.post(f"/admin/captures/{cid}/delete", headers=_auth())
    assert r.status_code == 302
    # GET on a POST-only route is rejected.
    assert client.get(f"/admin/captures/{cid}/delete", headers=_auth()).status_code == 405


def test_capture_reclassify_route(ctx) -> None:
    client, cfgp = ctx
    cid = _seed_capture(cfgp)
    # Invalid label -> flash err, still 302 (PRG).
    r = client.post(f"/admin/captures/{cid}/reclassify", headers=_auth(),
                    data={"new_label": "not_a_class"})
    assert r.status_code == 302
    # Valid label ('deer' is the capture's current label, always allowed).
    r = client.post(f"/admin/captures/{cid}/reclassify", headers=_auth(),
                    data={"new_label": "deer"})
    assert r.status_code == 302


def test_captures_bulk_route(ctx) -> None:
    client, cfgp = ctx
    cid = _seed_capture(cfgp)
    hdr = {**_auth(), **_origin()}
    # Happy path.
    r = client.post("/admin/captures/bulk", headers=hdr,
                    json={"action": "delete", "ids": [cid]})
    assert r.status_code == 200 and r.get_json() == {"ok": True, "action": "delete", "affected": 1}
    # Bad action.
    assert client.post("/admin/captures/bulk", headers=hdr,
                       json={"action": "nope", "ids": [1]}).status_code == 400
    # Non-JSON body.
    assert client.post("/admin/captures/bulk", headers=hdr, data="x").status_code == 400
    # Non-int id.
    assert client.post("/admin/captures/bulk", headers=hdr,
                       json={"action": "delete", "ids": ["1; DROP"]}).status_code == 400
    # Cross-origin POST is blocked by the guard.
    assert client.post("/admin/captures/bulk", headers={**_auth(), "Origin": "http://evil.example"},
                       json={"action": "delete", "ids": [1]}).status_code == 403
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_admin_routes.py -k captures -v`
Expected: FAIL (routes 404 / factory `TypeError` once wiring changes land).

- [ ] **Step 3: Extend the factory signature and pass `get_store`**

In `src/wildlife/admin/routes.py`, add `current_app` to the flask import block (line 23-32) and import the gallery parsers at the top (after line 37):

```python
from wildlife.gallery.app import _parse_filters, _parse_page
```

Change the factory signature (line 75) from:

```python
def create_admin_blueprint(config_path: str | Path, get_config: Callable[[], object]) -> Blueprint:
    """Build the ``/admin`` blueprint bound to ``config_path`` and ``get_config``."""
```

to:

```python
def create_admin_blueprint(
    config_path: str | Path,
    get_config: Callable[[], object],
    get_store: Callable[[], object],
) -> Blueprint:
    """Build the ``/admin`` blueprint.

    ``get_store`` returns the gallery's per-request :class:`~wildlife.store.Store`
    so capture-management routes read/write the same DB. File deletion is owned by
    the Store (it holds ``captures_dir``), so the blueprint needs no path itself.
    """
```

In `src/wildlife/gallery/app.py`, change the registration (line 392) from:

```python
        app.register_blueprint(create_admin_blueprint(config_path, get_config))
```

to:

```python
        app.register_blueprint(create_admin_blueprint(config_path, get_config, get_store))
```

- [ ] **Step 4: Add the capture routes inside the blueprint**

In `src/wildlife/admin/routes.py`, add these routes just before `return bp` (line 343). They close over `get_store` / `get_config`:

```python
    # ------------------------------------------------------------------ #
    # Captures (browse + manage)
    # ------------------------------------------------------------------ #
    def _captures_redirect() -> str:
        """Back to the captures list, preserving filters via the same-origin referrer."""
        ref = request.referrer
        if ref and urlsplit(ref).netloc == request.host:
            return ref
        return url_for("admin.captures_index")

    @bp.route("/captures")
    def captures_index():
        store = get_store()
        filters = _parse_filters(request.args)
        reviewed = _parse_reviewed(request.args.get("reviewed"))
        page = _parse_page(request.args.get("page"))
        page_size = current_app.config["PAGE_SIZE"]
        offset = (page - 1) * page_size
        rows = store.query(
            camera_id=filters["camera"], label=filters["label"],
            start=filters["start"], end=filters["end"],
            min_confidence=filters["min_confidence"], reviewed=reviewed,
            limit=page_size + 1, offset=offset,
        )
        has_more = len(rows) > page_size
        rows = rows[:page_size]
        total = store.count(
            camera_id=filters["camera"], label=filters["label"],
            start=filters["start"], end=filters["end"],
            min_confidence=filters["min_confidence"], reviewed=reviewed,
        )
        # Args to carry across Prev/Next links (everything except the page number).
        query_args = {k: v for k, v in request.args.items() if k != "page"}
        return render_template(
            "admin/captures.html",
            captures=[_serialize_capture(r) for r in rows],
            cameras=store.distinct_cameras(),
            labels=store.distinct_labels(),
            reclassify_labels=_reclassify_labels(get_config(), store),
            filters=filters,
            reviewed=request.args.get("reviewed") or "",
            query_args=query_args,
            page=page, page_size=page_size, has_more=has_more, total=total,
        )

    @bp.route("/captures/<int:capture_id>/delete", methods=["POST"])
    def capture_delete(capture_id: int):
        if get_store().delete(capture_id):
            flash(f"Capture #{capture_id} deleted.", "ok")
        else:
            flash(f"Capture #{capture_id} not found.", "err")
        return redirect(_captures_redirect())

    @bp.route("/captures/<int:capture_id>/reclassify", methods=["POST"])
    def capture_reclassify(capture_id: int):
        store = get_store()
        new_label = (request.form.get("new_label") or "").strip()
        row = store.get(capture_id)
        if row is None:
            flash(f"Capture #{capture_id} not found.", "err")
            return redirect(_captures_redirect())
        allowed = _reclassify_labels(get_config(), store, current=row.get("label"))
        if new_label not in allowed:
            flash(f"{new_label!r} is not a valid label.", "err")
            return redirect(_captures_redirect())
        store.update_label(capture_id, new_label)
        flash(f"Capture #{capture_id} reclassified to {new_label!r}.", "ok")
        return redirect(_captures_redirect())

    @bp.route("/captures/bulk", methods=["POST"])
    def captures_bulk():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "JSON body required"}), 400
        action = data.get("action")
        if action not in ("delete", "reclassify", "mark_reviewed"):
            return jsonify({"ok": False, "error": "unknown action"}), 400
        raw_ids = data.get("ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            return jsonify({"ok": False, "error": "ids must be a non-empty list"}), 400
        try:
            ids = [int(x) for x in raw_ids]
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "ids must be integers"}), 400
        if len(ids) > _BULK_MAX_IDS:
            return jsonify({"ok": False, "error": f"too many ids (max {_BULK_MAX_IDS})"}), 400
        store = get_store()
        if action == "delete":
            affected = store.delete_many(ids)
        elif action == "mark_reviewed":
            affected = store.mark_reviewed_many(ids)
        else:  # reclassify
            label = (data.get("label") or "").strip()
            if label not in _reclassify_labels(get_config(), store):
                return jsonify({"ok": False, "error": "invalid label"}), 400
            affected = store.update_label_many(ids, label)
        return jsonify({"ok": True, "action": action, "affected": affected})
```

- [ ] **Step 5: Add the module-level helpers**

In `src/wildlife/admin/routes.py`, add near the other module constants (after line 50) and view helpers (after `_detection_view_from_form`, line 399):

```python
_BULK_MAX_IDS = 1000
```

```python
def _parse_reviewed(value: str | None) -> bool | None:
    """Map the ``reviewed`` filter param to a tri-state (all / reviewed / unreviewed)."""
    if value == "reviewed":
        return True
    if value == "unreviewed":
        return False
    return None


def _reclassify_labels(cfg, store, *, current: str | None = None) -> list[str]:
    """The allowed reclassify targets: configured classes ∪ existing labels ∪ current."""
    labels = set(getattr(cfg.detection, "animal_classes", []) or [])
    labels.update(store.distinct_labels())
    if current:
        labels.add(current)
    return sorted(labels)


def _serialize_capture(row: dict) -> dict:
    """Shape a capture row for the admin grid (reuses the gallery image routes)."""
    cid = row["id"]
    return {
        "id": cid,
        "camera_id": row.get("camera_id"),
        "label": row.get("label"),
        "confidence": row.get("confidence"),
        "capture_ts": row.get("capture_ts"),
        "reviewed": bool(row.get("reviewed")),
        "original_label": row.get("original_label"),
        "thumb_url": url_for("thumb", capture_id=cid),
        "image_url": url_for("image", capture_id=cid),
    }
```

- [ ] **Step 6: Add the "Captures" nav link + `main_class` block to `base.html`**

In `src/wildlife/gallery/templates/admin/base.html`, add a nav link after the Cameras link (line 18):

```html
        <a class="nav-link{% if ep.startswith('admin.capture') %} active{% endif %}" href="{{ url_for('admin.captures_index') }}">Captures</a>
```

Change the `<main>` tag (line 25) from `<main class="admin-main">` to:

```html
  <main class="{% block main_class %}admin-main{% endblock %}">
```

- [ ] **Step 7: Create the server-rendered `captures.html`**

Create `src/wildlife/gallery/templates/admin/captures.html`:

```html
{% extends "admin/base.html" %}
{% block title %}Captures{% endblock %}
{% block main_class %}admin-main admin-main-wide{% endblock %}

{% block content %}
<form class="filters" method="get" action="{{ url_for('admin.captures_index') }}">
  <label class="field"><span>Camera</span>
    <select name="camera">
      <option value="">All cameras</option>
      {% for cam in cameras %}
      <option value="{{ cam }}" {% if filters.camera == cam %}selected{% endif %}>{{ cam }}</option>
      {% endfor %}
    </select>
  </label>
  <label class="field"><span>Class</span>
    <select name="label">
      <option value="">All classes</option>
      {% for lbl in labels %}
      <option value="{{ lbl }}" {% if filters.label == lbl %}selected{% endif %}>{{ lbl }}</option>
      {% endfor %}
    </select>
  </label>
  <label class="field"><span>Review</span>
    <select name="reviewed">
      <option value="" {% if not reviewed %}selected{% endif %}>All</option>
      <option value="unreviewed" {% if reviewed == 'unreviewed' %}selected{% endif %}>Unreviewed</option>
      <option value="reviewed" {% if reviewed == 'reviewed' %}selected{% endif %}>Reviewed</option>
    </select>
  </label>
  <label class="field"><span>From</span><input type="date" name="start" value="{{ filters.start_raw }}"></label>
  <label class="field"><span>To</span><input type="date" name="end" value="{{ filters.end_raw }}"></label>
  <div class="field actions">
    <button type="submit" class="btn primary">Apply</button>
    <a class="btn ghost" href="{{ url_for('admin.captures_index') }}">Reset</a>
  </div>
</form>

<p class="muted">{{ total }} capture{{ '' if total == 1 else 's' }} match.</p>

{% if not captures %}
<div class="empty"><p>No captures match these filters.</p></div>
{% endif %}

<section id="grid" class="grid" aria-live="polite">
  {% for c in captures %}
  <article class="card" data-id="{{ c.id }}" data-full="{{ c.image_url }}"
           data-label="{{ c.label }}" data-confidence="{{ c.confidence }}"
           data-camera="{{ c.camera_id }}" data-ts="{{ c.capture_ts }}">
    <img class="thumb" loading="lazy" src="{{ c.thumb_url }}" alt="{{ c.label }} on {{ c.camera_id }}">
    {% if c.reviewed %}<span class="pill pill-ok card-reviewed">reviewed</span>{% endif %}
    <div class="meta">
      <span class="label">{{ c.label }}</span>
      <span class="conf">{{ (c.confidence * 100) | round(0) | int }}%</span>
      <span class="cam">{{ c.camera_id }}</span>
      <span class="ts">{{ (c.capture_ts or '')[:19] | replace('T', ' ') }}</span>
    </div>
    <div class="card-actions">
      <form method="post" action="{{ url_for('admin.capture_reclassify', capture_id=c.id) }}">
        <select name="new_label" aria-label="Reclassify">
          {% for lbl in reclassify_labels %}
          <option value="{{ lbl }}" {% if lbl == c.label %}selected{% endif %}>{{ lbl }}</option>
          {% endfor %}
        </select>
        <button type="submit" class="btn ghost">Relabel</button>
      </form>
      <form method="post" action="{{ url_for('admin.capture_delete', capture_id=c.id) }}"
            onsubmit="return confirm('Delete capture #{{ c.id }}? This removes the image files and cannot be undone.');">
        <button type="submit" class="btn danger">Delete</button>
      </form>
    </div>
  </article>
  {% endfor %}
</section>

<div class="pager">
  {% if page > 1 %}<a class="btn ghost" href="{{ url_for('admin.captures_index', page=page-1, **query_args) }}">&#8249; Prev</a>{% else %}<span class="btn ghost" aria-disabled="true">&#8249; Prev</span>{% endif %}
  <span class="muted">Page {{ page }}</span>
  {% if has_more %}<a class="btn ghost" href="{{ url_for('admin.captures_index', page=page+1, **query_args) }}">Next &#8250;</a>{% else %}<span class="btn ghost" aria-disabled="true">Next &#8250;</span>{% endif %}
</div>
{% endblock %}
```

- [ ] **Step 8: Add the wide-container + reviewed-pill CSS**

Append to `src/wildlife/gallery/static/style.css`:

```css
/* --- Admin captures management --- */
.admin-main-wide { max-width: 1400px; }
.card .card-reviewed { position: absolute; top: 8px; right: 8px; z-index: 2; }
.card-actions { position: absolute; inset: auto 0 0 0; display: flex; gap: 6px;
  padding: 6px; background: rgba(0,0,0,0.35); pointer-events: auto; }
.card-actions form { display: flex; gap: 4px; margin: 0; }
.card-actions select { max-width: 130px; }
.pager { display: flex; align-items: center; gap: 12px; margin: 18px 0; }
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/test_admin_routes.py -v`
Expected: PASS (captures tests + the existing admin suite).

- [ ] **Step 10: Commit**

```bash
git add src/wildlife/admin/routes.py src/wildlife/gallery/app.py \
        src/wildlife/gallery/templates/admin/base.html \
        src/wildlife/gallery/templates/admin/captures.html \
        src/wildlife/gallery/static/style.css tests/test_admin_routes.py
git commit -m "admin: capture management routes + server-rendered /admin/captures page"
```

---

## Task 8: Selection + bulk-action + lightbox JS layer

**Files:**
- Modify: `src/wildlife/gallery/templates/admin/captures.html` (bulk toolbar, `#bulk-status`, per-card checkbox, lightbox markup, `{% block scripts %}`)
- Modify: `src/wildlife/gallery/static/style.css` (toolbar, selection, checkbox, `#bulk-status`)
- Test: `tests/test_admin_routes.py` (render assertions)

**Interfaces:**
- Consumes: `admin.captures_bulk` (Task 7). No new backend.

- [ ] **Step 1: Write the failing render test**

Add to `tests/test_admin_routes.py`:

```python
def test_captures_page_has_bulk_ui(ctx) -> None:
    client, cfgp = ctx
    _seed_capture(cfgp)
    r = client.get("/admin/captures", headers=_auth())
    html = r.data
    assert b'id="bulk-status"' in html
    assert b'class="bulk-toolbar"' in html
    assert b'class="select-box"' in html   # per-card selection checkbox
    assert b'id="lightbox"' in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admin_routes.py::test_captures_page_has_bulk_ui -v`
Expected: FAIL (elements absent).

- [ ] **Step 3: Add the toolbar, checkbox, status region, and lightbox markup**

In `captures.html`, insert the bulk toolbar + status region immediately **before** the `<section id="grid" ...>` line:

```html
<div class="bulk-toolbar" hidden>
  <label class="select-all"><input type="checkbox" id="selectAll"> Select all on page</label>
  <span id="selCount" class="muted" role="status" aria-live="polite">0 selected</span>
  <select id="bulkLabel" aria-label="Bulk reclassify to">
    {% for lbl in reclassify_labels %}<option value="{{ lbl }}">{{ lbl }}</option>{% endfor %}
  </select>
  <button type="button" class="btn ghost" id="bulkRelabel">Relabel selected</button>
  <button type="button" class="btn ghost" id="bulkReview">Mark reviewed</button>
  <button type="button" class="btn danger" id="bulkDelete">Delete selected</button>
  <span class="muted select-note">Selection applies to this page only.</span>
</div>
<div id="bulk-status" class="banner" role="status" aria-live="polite" hidden></div>
```

Add a selection checkbox as the **first child of each `.card`** (immediately after the `<article ...>` opening tag, before the `<img>`):

```html
    <input type="checkbox" class="select-box" aria-label="Select capture #{{ c.id }}">
```

Add the lightbox markup at the end of the `{% block content %}` (after the `.pager` div):

```html
<div id="lightbox" class="lightbox" hidden role="dialog" aria-modal="true" aria-label="Capture viewer">
  <button class="lb-btn lb-close" aria-label="Close">&times;</button>
  <button class="lb-btn lb-prev" aria-label="Previous">&#8249;</button>
  <button class="lb-btn lb-next" aria-label="Next">&#8250;</button>
  <figure class="lb-figure">
    <div class="lb-spinner" id="lbSpinner"></div>
    <img id="lbImg" alt="">
    <figcaption id="lbCap" class="lb-cap"></figcaption>
  </figure>
</div>
```

- [ ] **Step 4: Add the `{% block scripts %}` IIFE**

Append to `captures.html`:

```html
{% block scripts %}
<script>
(function () {
  "use strict";
  const grid = document.getElementById("grid");
  const toolbar = document.querySelector(".bulk-toolbar");
  const selCount = document.getElementById("selCount");
  const selectAll = document.getElementById("selectAll");
  const status = document.getElementById("bulk-status");
  const selected = new Set();

  function refresh() {
    selCount.textContent = selected.size + " selected";
    toolbar.hidden = selected.size === 0 && !anyBox();
  }
  function anyBox() { return grid.querySelector(".select-box") !== null; }
  if (anyBox()) toolbar.hidden = false;

  function setSelected(card, on) {
    const id = card.dataset.id;
    card.classList.toggle("selected", on);
    card.querySelector(".select-box").checked = on;
    if (on) selected.add(id); else selected.delete(id);
  }

  // Selection via checkbox; everything else on the card opens the lightbox.
  grid.addEventListener("change", (e) => {
    const box = e.target.closest(".select-box");
    if (!box) return;
    setSelected(box.closest(".card"), box.checked);
    refresh();
  });
  selectAll.addEventListener("change", () => {
    grid.querySelectorAll(".card").forEach((c) => setSelected(c, selectAll.checked));
    refresh();
  });

  function showStatus(msg, ok) {
    status.hidden = false;
    status.textContent = msg;
    status.className = "banner " + (ok ? "banner-ok" : "banner-err");
  }

  async function bulk(action, extra) {
    const ids = Array.from(selected).map(Number);
    if (!ids.length) return;
    if (action === "delete" &&
        !confirm("Delete " + ids.length + " capture(s)? This removes image files and cannot be undone.")) return;
    try {
      const res = await fetch("{{ url_for('admin.captures_bulk') }}", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(Object.assign({ action: action, ids: ids }, extra || {})),
      });
      const data = await res.json();
      if (!data.ok) { showStatus(data.error || "Action failed", false); return; }
      applyResult(action, ids, extra);
      showStatus(labelFor(action) + " " + data.affected + " capture(s).", true);
    } catch (err) {
      showStatus("Request failed: " + err, false);
    }
  }
  function labelFor(a) { return a === "delete" ? "Deleted" : a === "reclassify" ? "Reclassified" : "Marked reviewed"; }

  function applyResult(action, ids, extra) {
    ids.forEach((id) => {
      const card = grid.querySelector('.card[data-id="' + id + '"]');
      if (!card) return;
      if (action === "delete") { card.remove(); return; }
      if (action === "reclassify" && extra && extra.label) {
        card.querySelector(".meta .label").textContent = extra.label;
        card.dataset.label = extra.label;
      }
      if (!card.querySelector(".card-reviewed")) {
        const pill = document.createElement("span");
        pill.className = "pill pill-ok card-reviewed";
        pill.textContent = "reviewed";
        card.appendChild(pill);
      }
    });
    selected.clear();
    grid.querySelectorAll(".select-box").forEach((b) => (b.checked = false));
    grid.querySelectorAll(".card.selected").forEach((c) => c.classList.remove("selected"));
    selectAll.checked = false;
    refresh();
    if (!grid.querySelector(".card")) location.reload();
  }

  document.getElementById("bulkDelete").addEventListener("click", () => bulk("delete"));
  document.getElementById("bulkReview").addEventListener("click", () => bulk("mark_reviewed"));
  document.getElementById("bulkRelabel").addEventListener("click", () =>
    bulk("reclassify", { label: document.getElementById("bulkLabel").value }));

  // ----- Lightbox (image click only; ignore controls) ----- //
  const lb = document.getElementById("lightbox");
  const lbImg = document.getElementById("lbImg");
  const lbCap = document.getElementById("lbCap");
  const lbSpinner = document.getElementById("lbSpinner");
  let current = null, lastFocus = null;
  const cards = () => Array.from(grid.querySelectorAll(".card"));

  function open(card) {
    current = card;
    lastFocus = document.activeElement;
    lbSpinner.hidden = false;
    lbImg.classList.remove("ready");
    lbImg.onload = () => { lbSpinner.hidden = true; lbImg.classList.add("ready"); };
    lbImg.onerror = () => { lbSpinner.hidden = true; };
    lbImg.src = card.dataset.full;
    lbCap.innerHTML = '<strong></strong><span class="lb-cam"></span><span class="lb-ts"></span>';
    lbCap.querySelector("strong").textContent = card.dataset.label || "";
    lbCap.querySelector(".lb-cam").textContent = card.dataset.camera || "";
    lbCap.querySelector(".lb-ts").textContent = (card.dataset.ts || "").slice(0, 19).replace("T", " ");
    lb.hidden = false;
    document.body.classList.add("noscroll");
    lb.querySelector(".lb-close").focus();
  }
  function close() {
    lb.hidden = true; lbImg.removeAttribute("src"); current = null;
    document.body.classList.remove("noscroll");
    if (lastFocus) lastFocus.focus();
  }
  function step(d) {
    if (!current) return;
    const all = cards(); const i = all.indexOf(current);
    let n = i + d; if (n < 0) n = 0; if (n >= all.length) n = all.length - 1;
    open(all[n]);
  }
  grid.addEventListener("click", (e) => {
    if (e.target.closest("input,select,button,form,a")) return;  // controls don't open the lightbox
    const card = e.target.closest(".card");
    if (card) open(card);
  });
  lb.addEventListener("click", (e) => {
    if (e.target === lb || e.target.closest(".lb-close")) close();
    else if (e.target.closest(".lb-prev")) step(-1);
    else if (e.target.closest(".lb-next")) step(1);
  });
  document.addEventListener("keydown", (e) => {
    if (lb.hidden) return;                 // selection keys ignored only while open
    if (e.key === "Escape") close();
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "ArrowRight") step(1);
  });
})();
</script>
{% endblock %}
```

- [ ] **Step 5: Add the toolbar/selection CSS**

Append to `src/wildlife/gallery/static/style.css`:

```css
.bulk-toolbar { position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap;
  align-items: center; gap: 10px; padding: 10px 12px; margin-bottom: 12px;
  background: var(--bg-elev-2); border: 1px solid var(--border); border-radius: var(--radius-sm); }
.bulk-toolbar .select-note { margin-left: auto; }
.card .select-box { position: absolute; top: 8px; left: 8px; z-index: 3; width: 20px; height: 20px;
  pointer-events: auto; cursor: pointer; }
.card.selected { outline: 3px solid var(--accent); outline-offset: -3px; }
#bulk-status { margin-bottom: 12px; }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_admin_routes.py -v`
Expected: PASS.

- [ ] **Step 7: Manually verify the interactive layer**

Run the demo and click through selection + bulk delete/relabel + lightbox (there is no JS unit-test harness in this repo, so this is a manual gate):

Run: `./scripts/run_demo.sh --seed 12` then open `http://localhost:8080/admin/captures` (log in), confirm: checkboxes select cards, the sticky toolbar appears, bulk delete removes cards + shows the status banner, relabel updates the label + adds the reviewed pill, clicking a thumbnail opens the lightbox, arrows/Escape work.
Expected: all behaviors work; no console errors.

- [ ] **Step 8: Commit**

```bash
git add src/wildlife/gallery/templates/admin/captures.html \
        src/wildlife/gallery/static/style.css tests/test_admin_routes.py
git commit -m "admin: selection + bulk actions + lightbox for /admin/captures"
```

---

## Task 9: Documentation

**Files:**
- Modify: `README.md` (Admin section, ~252-298)
- Modify: `spec.md` (schema §6.5, gallery/admin section, build order)
- Modify: `config.example.yaml` (retention comment)

**Interfaces:** none (docs only).

- [ ] **Step 1: Add a "Managing captures" subsection to `README.md`**

Under the Admin section, add:

```markdown
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
```

- [ ] **Step 2: Update `spec.md`**

In `spec.md` §6.5, add the three columns (`original_label TEXT`, `reviewed INTEGER NOT NULL DEFAULT 0`, `reviewed_at TEXT`) to the `captures` schema and note the idempotent migration. In the gallery/admin section, list the new routes (`GET /admin/captures`, `POST /admin/captures/<id>/delete`, `.../reclassify`, `POST /admin/captures/bulk`). Add "capture management" to the admin build-order step.

- [ ] **Step 3: Add a retention comment to `config.example.yaml`**

Next to the `retention:` block, add:

```yaml
  # Human-reviewed captures (reclassified or marked reviewed in /admin/captures)
  # are exempt from min_confidence_keep, but still subject to max_age_days.
```

- [ ] **Step 4: Commit**

```bash
git add README.md spec.md config.example.yaml
git commit -m "docs: document /admin/captures management + reviewed retention exemption"
```

---

## Final verification

- [ ] Run the full suite: `pytest -q` — expect all green.
- [ ] Manual smoke test via `./scripts/run_demo.sh --seed 12` on `/admin/captures` (single delete, single relabel, bulk delete, bulk relabel, mark reviewed, filters, lightbox).
- [ ] Confirm the public gallery (`/`) still has no capture-mutation routes and no auth prompt.

---

## Self-review notes (author)

- **Spec coverage:** §4 columns/migration/busy_timeout → Task 1; §5 helper lift → Task 2; §6.1-6.2 delete → Task 3; §6.3-6.5 update/mark → Task 4; §6.6 count + reviewed filter → Task 5; §11 prune exemption → Task 6; §7 routes + wiring → Task 7; §8 template/selection/bulk/lightbox/CSS/nav → Tasks 7-8; §12 tests distributed across tasks; §13 docs → Task 9. `false_positive` sentinel intentionally absent (out of scope, §2).
- **Deviation from spec §7.1:** the factory takes `get_store` but **not** `captures_dir` — the Store owns file deletion via its own `captures_dir`, so the blueprint never needs the path. Noted in Task 7 Step 3.
- **Type consistency:** `delete_many`/`update_label_many`/`mark_reviewed_many` all take `Iterable[int]` and return `int`; `update_label` returns `dict | None`; `count`/`query` share `_build_filters`; route endpoints (`admin.captures_index`, `admin.capture_delete`, `admin.capture_reclassify`, `admin.captures_bulk`) are referenced consistently in the template `url_for`s and tests.
- **No JS unit harness** exists in the repo; Task 8's automated test asserts the markup is present and the interactive behavior is a documented manual gate (Task 8 Step 7).
