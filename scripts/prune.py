#!/usr/bin/env python3
"""Retention pruning: delete old (and optionally low-confidence) captures.

Build-order step 11. Enforces ``retention.max_age_days`` (and, when configured,
``retention.min_confidence_keep``) from the config by removing the JPEG, its
thumbnail, and the SQLite row for every matching capture, then sweeping away any
now-empty ``YYYY/MM/DD`` directories.

A capture is pruned when **either**:

* its ``capture_ts`` is older than ``max_age_days``, **or**
* ``min_confidence_keep > 0`` and its ``confidence`` is below that value.

The script is safe and idempotent: it never touches files outside
``captures_dir``, tolerates already-missing files, deletes rows by primary key,
and a second run finds nothing left to do. Use ``--dry-run`` to preview.

Run it directly from a checkout::

    python scripts/prune.py
    python scripts/prune.py --dry-run
    python scripts/prune.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path


def _bootstrap_src_path() -> None:
    """Allow running straight from a checkout without an editable install."""
    src = Path(__file__).resolve().parent.parent / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_src_path()

from wildlife.config import load_config  # noqa: E402

logger = logging.getLogger("prune")


def _resolve_config(path_arg: str) -> Path:
    """Find the config file, falling back to the repo root next to ``scripts/``."""
    candidate = Path(path_arg)
    if candidate.is_file():
        return candidate
    repo_root = Path(__file__).resolve().parent.parent
    alt = repo_root / path_arg
    if alt.is_file():
        return alt
    raise SystemExit(
        f"Config file not found: {path_arg!r}. "
        "Copy config.example.yaml to config.yaml and edit it."
    )


def _human_bytes(num: int) -> str:
    """Format a byte count compactly (e.g. ``12.3 MB``)."""
    size = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _table_exists(conn: sqlite3.Connection) -> bool:
    """Return True if the ``captures`` table is present."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='captures'"
    ).fetchone()
    return row is not None


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


def main() -> int:
    """Prune captures per the retention policy. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without touching files or the DB.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    config = load_config(_resolve_config(args.config))
    captures_dir = Path(config.storage.captures_dir)
    db_path = Path(config.storage.db_path)
    max_age_days = config.retention.max_age_days
    min_conf = config.retention.min_confidence_keep

    cutoff = datetime.now() - timedelta(days=max_age_days)
    cutoff_iso = cutoff.isoformat()

    print("Retention prune" + (" (DRY RUN)" if args.dry_run else ""))
    print(f"  captures_dir : {captures_dir}")
    print(f"  db_path      : {db_path}")
    print(f"  max_age_days : {max_age_days}  -> delete capture_ts < {cutoff_iso}")
    if min_conf > 0.0:
        print(f"  min_conf_keep: {min_conf}  -> also delete confidence < {min_conf}")
    print()

    if not db_path.exists():
        print(f"No database at {db_path}; nothing to prune.")
        return 0

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

        rows = conn.execute(
            f"SELECT id, image_path, thumb_path, capture_ts, confidence "
            f"FROM captures WHERE {where} ORDER BY capture_ts",
            params,
        ).fetchall()

        if not rows:
            print("Nothing matches the retention policy. Up to date.")
            return 0

        print(f"{len(rows)} capture(s) match the policy.\n")

        files_deleted = 0
        files_missing = 0
        bytes_freed = 0
        ids_to_delete: list[int] = []

        for row in rows:
            ids_to_delete.append(int(row["id"]))
            verb = "would delete" if args.dry_run else "deleting"
            logger.debug(
                "%s id=%d capture_ts=%s conf=%.3f",
                verb,
                row["id"],
                row["capture_ts"],
                row["confidence"],
            )
            for rel in (row["image_path"], row["thumb_path"]):
                if args.dry_run:
                    # Still account for what we'd remove, without unlinking.
                    target = (captures_dir / rel) if rel else None
                    if target is not None and target.exists():
                        files_deleted += 1
                        bytes_freed += target.stat().st_size
                    else:
                        files_missing += 1
                    continue
                deleted, size = _safe_unlink(captures_dir, rel)
                if deleted:
                    files_deleted += 1
                    bytes_freed += size
                else:
                    files_missing += 1

        dirs_removed = 0
        if not args.dry_run:
            with conn:  # transactional batch delete
                conn.executemany(
                    "DELETE FROM captures WHERE id = ?",
                    [(rid,) for rid in ids_to_delete],
                )
            dirs_removed = _prune_empty_dirs(captures_dir)

        print()
        print("Summary:")
        print(f"  rows {'matched' if args.dry_run else 'deleted'} : {len(ids_to_delete)}")
        print(f"  files removed      : {files_deleted}  ({_human_bytes(bytes_freed)})")
        print(f"  files already gone : {files_missing}")
        if not args.dry_run:
            print(f"  empty dirs removed : {dirs_removed}")
        if args.dry_run:
            print("\n(DRY RUN -- no files or rows were actually deleted.)")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
