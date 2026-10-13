"""Claude-vision review of saved camera captures: find and delete the false positives.

The on-device pipeline (species YOLO + MegaDetector + heuristic guards) is fooled
at night by insects at the IR illuminator -- both models call a close moth wing
an animal at >0.9 confidence. This pass shows each saved detection to Claude (the
full frame with the box drawn, plus a close-up of the box) and asks whether a
real animal or person is there.

``wildlife-review run``:

1. Select saved image captures with no ``review_log`` row yet (audio excluded,
   and nothing younger than a couple of minutes so a burst still being saved
   isn't split).
2. Group them per burst event (camera + ``capture_ts``) and cluster overlapping
   boxes within an event: a burst's frames show the same object, so one request
   on the cluster's highest-confidence row covers all of them.
3. Ask Claude for a structured verdict.
4. Delete a cluster when the verdict is ``not_animal`` at
   ``review.delete_min_confidence`` or above -- unless a human reviewed or
   reclassified any capture in it (a human call always wins).
5. Write one ``review_log`` row per capture, deleted ones included, so the
   verdicts outlive the images and serve as ground truth for ``wildlife-replay``.

``run --dry-run`` records ``would_delete`` instead of deleting, and
``wildlife-review apply`` deletes those later -- so the backlog can be labelled,
scored against pipeline variants, and only then cleaned.

Two backends (``review.backend``): ``claude_cli`` runs the Claude Code CLI
(``claude -p``) on its own login, so a Claude subscription pays and no API key
is needed; ``api`` calls the Anthropic API with a key.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ValidationError, field_validator

from wildlife.config import Config, ReviewConfig, load_config
from wildlife.store import Store

logger = logging.getLogger("wildlife.review")

# Server-side refusal fallback: if the reviewer model declines a request, the API
# re-runs it on Anthropic's recommended fallback model within the same call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Boxes in one burst event this overlapping are the same object in successive frames.
_CLUSTER_IOU = 0.5
# Leave very recent captures for the next run: the worker may still be saving
# the rest of that burst.
_MIN_AGE = timedelta(minutes=2)
# Image sizing for the request. The full frame is downscaled to Claude's
# recommended long edge; the close-up is upscaled when small so a distant animal
# isn't a dozen pixels wide.
_FULL_MAX_PX = 1568
_CROP_MAX_PX = 768
_CROP_MIN_PX = 512
_CROP_MIN_SIDE_SRC = 256  # source pixels: never crop tighter than this
_JPEG_QUALITY = 88
_MAX_TOKENS = 4096
# claude_cli: one review normally takes a few seconds.
_CLI_TIMEOUT_S = 180
# Where a bare review.claude_bin is looked for when it isn't on PATH (launchd's
# PATH is minimal).
_CLI_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "~/.local/bin", "~/.claude/local")

VERDICTS = ("animal", "person", "not_animal", "unsure")
ARTIFACTS = (
    "none",
    "insect",
    "rain_snow_or_dust",
    "flare_glare_or_web",
    "vegetation_rock_or_shadow",
    "static_object_or_text",
    "corrupted_frame",
    "other",
)

_REVIEW_LOG_SQL = """
CREATE TABLE IF NOT EXISTS review_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    capture_id          INTEGER NOT NULL,   -- may no longer exist once deleted
    representative_id   INTEGER NOT NULL,   -- the capture whose image was judged
    camera_id           TEXT    NOT NULL,
    capture_ts          TEXT    NOT NULL,
    image_path          TEXT,
    label               TEXT    NOT NULL,   -- detector label at review time
    confidence          REAL,               -- detector confidence
    verdict             TEXT    NOT NULL,   -- animal | person | not_animal | unsure
    species             TEXT,
    artifact            TEXT,
    reviewer_confidence REAL,
    reason              TEXT,
    action              TEXT    NOT NULL,   -- kept | kept_human | would_delete | deleted
    model               TEXT,
    reviewed_at         TEXT    NOT NULL,
    applied_at          TEXT                -- when a would_delete was carried out
);
CREATE INDEX IF NOT EXISTS idx_review_capture ON review_log(capture_id);
CREATE INDEX IF NOT EXISTS idx_review_action  ON review_log(action);
"""


# --------------------------------------------------------------------- models
class Verdict(BaseModel):
    """The reviewer's structured answer for one flagged region."""

    reason: str
    verdict: Literal["animal", "person", "not_animal", "unsure"]
    species: str
    artifact: str
    confidence: float

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))


@dataclass
class Candidate:
    """One saved capture row awaiting review."""

    id: int
    camera_id: str
    capture_ts: str
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    image_path: str
    human_reviewed: bool


@dataclass
class Cluster:
    """Captures in one burst event that show the same object."""

    members: list[Candidate] = field(default_factory=list)

    @property
    def rep(self) -> Candidate:
        return self.members[0]  # highest confidence (members are added in that order)


@dataclass
class RunSummary:
    clusters: int = 0
    captures: int = 0
    failed: int = 0
    actions: Counter = field(default_factory=Counter)
    verdicts: Counter = field(default_factory=Counter)
    input_tokens: int = 0
    output_tokens: int = 0
    stopped: str | None = None  # why the run ended early, if it did


@dataclass
class Usage:
    input_tokens: int = 0  # cache reads and writes included
    output_tokens: int = 0


# ----------------------------------------------------------------- review log
class ReviewLog:
    """The ``review_log`` table in the captures DB (own connection; WAL-safe)."""

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.executescript(_REVIEW_LOG_SQL)
        self._conn.commit()

    def pending(self, since: str | None = None, now: datetime | None = None) -> list[Candidate]:
        """Image captures with no review yet, newest first."""
        cutoff = ((now or datetime.now()) - _MIN_AGE).isoformat()
        sql = """
            SELECT id, camera_id, capture_ts, label, confidence,
                   box_x1, box_y1, box_x2, box_y2, image_path, reviewed, original_label
              FROM captures c
             WHERE source_kind != 'audio'
               AND capture_ts < ?
               AND NOT EXISTS (SELECT 1 FROM review_log r WHERE r.capture_id = c.id)
        """
        args: list[Any] = [cutoff]
        if since:
            sql += " AND capture_ts >= ?"
            args.append(since)
        sql += " ORDER BY capture_ts DESC, confidence DESC"
        return [_candidate(r) for r in self._conn.execute(sql, args)]

    def by_ids(self, ids: list[int]) -> list[Candidate]:
        q = ",".join("?" * len(ids))
        sql = f"""
            SELECT id, camera_id, capture_ts, label, confidence,
                   box_x1, box_y1, box_x2, box_y2, image_path, reviewed, original_label
              FROM captures WHERE id IN ({q}) AND source_kind != 'audio'
        """
        return [_candidate(r) for r in self._conn.execute(sql, ids)]

    def record(self, cluster: Cluster, v: Verdict, action: str, model: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        rows = [
            (
                m.id, cluster.rep.id, m.camera_id, m.capture_ts, m.image_path, m.label,
                m.confidence, v.verdict, v.species, v.artifact, v.confidence, v.reason,
                action, model, now,
            )
            for m in cluster.members
        ]
        self._conn.executemany(
            """INSERT INTO review_log (capture_id, representative_id, camera_id, capture_ts,
                   image_path, label, confidence, verdict, species, artifact,
                   reviewer_confidence, reason, action, model, reviewed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self._conn.commit()

    def would_delete_ids(self) -> list[int]:
        """Captures a dry run flagged that still exist and are still not human-reviewed."""
        sql = """
            SELECT DISTINCT r.capture_id FROM review_log r
              JOIN captures c ON c.id = r.capture_id
             WHERE r.action = 'would_delete'
               AND c.reviewed = 0 AND c.original_label IS NULL
        """
        return [int(r[0]) for r in self._conn.execute(sql)]

    def mark_applied(self, ids: list[int]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        for i in range(0, len(ids), 900):
            chunk = ids[i : i + 900]
            q = ",".join("?" * len(chunk))
            self._conn.execute(
                f"UPDATE review_log SET action = 'deleted', applied_at = ? "
                f"WHERE action = 'would_delete' AND capture_id IN ({q})",
                [now, *chunk],
            )
        self._conn.commit()

    def summary(self, since: str | None = None) -> list[sqlite3.Row]:
        sql = """SELECT camera_id, verdict, action, artifact, COUNT(*) AS n
                   FROM review_log {where} GROUP BY 1,2,3,4 ORDER BY n DESC"""
        if since:
            return list(self._conn.execute(sql.format(where="WHERE capture_ts >= ?"), [since]))
        return list(self._conn.execute(sql.format(where="")))

    def close(self) -> None:
        self._conn.close()


def _candidate(r: sqlite3.Row) -> Candidate:
    return Candidate(
        id=int(r["id"]),
        camera_id=r["camera_id"],
        capture_ts=r["capture_ts"],
        label=r["label"],
        confidence=float(r["confidence"]),
        box=(float(r["box_x1"]), float(r["box_y1"]), float(r["box_x2"]), float(r["box_y2"])),
        image_path=r["image_path"],
        human_reviewed=bool(r["reviewed"]) or r["original_label"] is not None,
    )


# ----------------------------------------------------------------- clustering
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cluster_candidates(cands: list[Candidate]) -> list[Cluster]:
    """Group per burst event, then merge overlapping boxes within each event."""
    events: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in cands:
        events[(c.camera_id, c.capture_ts)].append(c)
    clusters: list[Cluster] = []
    for key in sorted(events, key=lambda k: k[1], reverse=True):
        local: list[Cluster] = []
        for c in sorted(events[key], key=lambda c: c.confidence, reverse=True):
            home = next((cl for cl in local if iou(cl.rep.box, c.box) >= _CLUSTER_IOU), None)
            if home is None:
                local.append(Cluster([c]))
            else:
                home.members.append(c)
        clusters.extend(local)
    return clusters


def decide_action(cluster: Cluster, v: Verdict, cfg: ReviewConfig, dry_run: bool) -> str:
    """What to do with a judged cluster. Only a confident not_animal ever deletes."""
    if v.verdict != "not_animal" or v.confidence < cfg.delete_min_confidence:
        return "kept"
    if any(m.human_reviewed for m in cluster.members):
        return "kept_human"
    return "would_delete" if dry_run else "deleted"


# ------------------------------------------------------------------- request
def build_images(path: Path, box) -> tuple[bytes, bytes]:
    """JPEG bytes for (full frame with the box drawn, close-up of the box)."""
    with Image.open(path) as src:
        im = src.convert("RGB")
    w, h = im.size
    x1, y1 = max(0.0, box[0]), max(0.0, box[1])
    x2, y2 = min(float(w), box[2]), min(float(h), box[3])

    # Close-up: the box plus half its longer side of context all round, never
    # narrower than _CROP_MIN_SIDE_SRC source pixels.
    pad = 0.5 * max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max((x2 - x1) / 2 + pad, _CROP_MIN_SIDE_SRC / 2)
    half_h = max((y2 - y1) / 2 + pad, _CROP_MIN_SIDE_SRC / 2)
    crop = im.crop((
        int(max(0, cx - half_w)), int(max(0, cy - half_h)),
        int(min(w, cx + half_w)), int(min(h, cy + half_h)),
    ))
    long_edge = max(crop.size)
    target = min(_CROP_MAX_PX, max(_CROP_MIN_PX, long_edge))
    if target != long_edge:
        s = target / long_edge
        crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)

    full = im.copy()
    full.thumbnail((_FULL_MAX_PX, _FULL_MAX_PX), Image.LANCZOS)
    s = full.width / w
    ImageDraw.Draw(full).rectangle(
        [x1 * s, y1 * s, x2 * s, y2 * s], outline=(255, 0, 0), width=max(3, round(full.width / 300))
    )
    return _jpeg(full), _jpeg(crop)


def _jpeg(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=_JPEG_QUALITY)
    return buf.getvalue()


def system_prompt(cfg: ReviewConfig, species: list[str]) -> str:
    return f"""You review detections from an automatic wildlife camera pipeline. {cfg.site_notes}

An on-device detector flagged a region in a saved frame. You get two images: the full frame with the flagged region outlined in red, and a close-up of that region. Decide whether a real animal or person is present.

This pipeline produces many false positives. The common ones here:
- Insects, mostly moths, flying close to the infrared illuminator at night. They appear as bright streaks or arcs (motion blur), or as large glowing out-of-focus ovals that can carry a striped, vein-like texture (a wing a few centimetres from the lens). They are often cut off by the frame edge and can look large.
- Rain, snow or dust lit by the illuminator: thin bright streaks.
- Lens flare, glare, IR bloom, spider-web strands.
- Vegetation, rocks, shadows, a bucket or post, or burned-in on-screen text such as the camera name in a corner.
- Corrupted video frames: flat grey, green or magenta areas, blocky smears.

Real animals at night look like solid bodies on or moving across the ground, at a size consistent with their distance, usually darker than or similar to the lit ground around them. Eyeshine (small bright paired dots) indicates a real animal. A real animal can be small, distant, partly hidden by brush, or motion-blurred.

If a real animal or person is clearly visible anywhere in the full frame, even outside the red box, the photo is worth keeping: answer for that animal and say where it is.

Deleting a photo of a real animal costs far more than keeping a false positive. Answer "not_animal" only when you can tell what the flagged region actually is (an insect, a streak, a plant...) and no animal is visible. If you cannot tell, answer "unsure".

Fields:
- reason: one short sentence on what you see in the flagged region.
- verdict: "animal", "person", "not_animal" or "unsure".
- species: for an animal or person, the closest match from the allowed list ("other_animal" for a clear animal that fits none of them, such as a dog, cat or livestock); "none" otherwise. Allowed: {", ".join(species)}.
- artifact: for not_animal, what the region is; "none" otherwise.
- confidence: 0.0 to 1.0, how sure you are of the verdict."""


def output_schema(species: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "species": {"type": "string", "enum": [*species, "other_animal", "none"]},
            "artifact": {"type": "string", "enum": list(ARTIFACTS)},
            "confidence": {"type": "number"},
        },
        "required": ["reason", "verdict", "species", "artifact", "confidence"],
        "additionalProperties": False,
    }


class AbortRun(RuntimeError):
    """No point making more requests this run; verdicts already in are kept."""


class AuthError(AbortRun):
    """The reviewer can't be reached: credentials missing or rejected, or no CLI."""


class PlanLimitError(AbortRun):
    """The subscription's usage window is (nearly) spent; the rest waits for the next run."""


def _declined() -> Verdict:
    # The reviewer refused the request. Keep the capture; record it as unsure.
    return Verdict(reason="reviewer declined", verdict="unsure", species="none",
                   artifact="none", confidence=0.0)


class Reviewer:
    """Builds the request for one representative capture and hands it to a backend."""

    def __init__(self, backend, cfg: ReviewConfig, species: list[str], captures_dir: Path) -> None:
        self.backend = backend
        self.cfg = cfg
        self.captures_dir = captures_dir
        self.system = system_prompt(cfg, species)
        self.schema = output_schema(species)

    def review(self, cand: Candidate) -> tuple[Verdict | None, Usage | None]:
        """Return (verdict, usage). verdict is None when the call failed (retry next run)."""
        full, crop = build_images(self.captures_dir / cand.image_path, cand.box)
        content = [
            {"type": "text", "text": "Image 1: full frame, flagged region outlined in red."},
            _image_block(full),
            {"type": "text", "text": "Image 2: close-up of the flagged region."},
            _image_block(crop),
            {"type": "text", "text": f"Camera: {cand.camera_id}. Local time: {cand.capture_ts[:16].replace('T', ' ')}."},
        ]
        return self.backend(cand.id, self.system, self.schema, content)


def _image_block(data: bytes) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg",
                   "data": base64.standard_b64encode(data).decode("ascii")},
    }


class ApiBackend:
    """The Anthropic Messages API (an API key), with server-side refusal fallback."""

    def __init__(self, client, cfg: ReviewConfig) -> None:
        self.client = client
        self.cfg = cfg

    def __call__(self, cid: int, system: str, schema: dict, content: list) -> tuple[Verdict | None, Usage | None]:
        import anthropic

        try:
            resp = self.client.beta.messages.create(
                model=self.cfg.model,
                max_tokens=_MAX_TOKENS,
                betas=[_FALLBACK_BETA],
                fallbacks="default",
                system=system,
                output_config={
                    "effort": self.cfg.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                messages=[{"role": "user", "content": content}],
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise AuthError(str(exc)) from exc
        except anthropic.BadRequestError as exc:
            logger.error("Capture %d: request rejected: %s", cid, exc.message)
            return None, None
        except anthropic.RateLimitError:
            logger.warning("Capture %d: rate limited after retries; will retry next run.", cid)
            return None, None
        except anthropic.APIStatusError as exc:
            logger.warning("Capture %d: API error %s; will retry next run.", cid, exc.status_code)
            return None, None
        except anthropic.APIConnectionError:
            logger.warning("Capture %d: network error; will retry next run.", cid)
            return None, None

        u = resp.usage
        usage = Usage(
            (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
            + (getattr(u, "cache_read_input_tokens", 0) or 0),
            u.output_tokens or 0,
        )
        if resp.stop_reason == "refusal":  # even the fallback declined
            return _declined(), usage
        if resp.stop_reason == "max_tokens":
            logger.warning("Capture %d: response truncated; will retry next run.", cid)
            return None, usage
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            return Verdict.model_validate_json(text), usage
        except ValidationError as exc:
            logger.warning("Capture %d: unparseable verdict (%s); will retry next run.", cid, exc)
            return None, usage


def resolve_claude_bin(name: str) -> str:
    if os.sep in name or name.startswith("~"):
        return str(Path(name).expanduser())
    found = shutil.which(name)
    if found:
        return found
    for d in _CLI_DIRS:
        cand = Path(d).expanduser() / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    raise AuthError(f"`{name}` is not on PATH or in {', '.join(_CLI_DIRS)}: install Claude Code "
                    "or set review.claude_bin")


class CliBackend:
    """The Claude Code CLI (``claude -p``) on its own login: a Claude subscription, no API key.

    Each review is one headless CLI run with no tools, no session saved and every
    customisation off (``--safe-mode``: no CLAUDE.md, hooks, plugins or MCP), the
    two images sent as a stream-json user message, and the verdict returned as
    ``--json-schema`` structured output.
    """

    def __init__(self, cfg: ReviewConfig, runner=subprocess.run) -> None:
        self.cfg = cfg
        self.bin = resolve_claude_bin(cfg.claude_bin)
        self.runner = runner
        self.env = dict(os.environ)
        token_file = Path(cfg.oauth_token_file).expanduser()
        if not self.env.get("CLAUDE_CODE_OAUTH_TOKEN") and token_file.is_file():
            token = token_file.read_text(encoding="utf-8").strip()
            if token:
                self.env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        # Set once a response reports the plan's usage past max_plan_utilization:
        # requests already running finish, later ones stop before starting.
        self.exhausted: str | None = None

    def __call__(self, cid: int, system: str, schema: dict, content: list) -> tuple[Verdict | None, Usage | None]:
        if self.exhausted:
            raise PlanLimitError(self.exhausted)
        cmd = [
            self.bin, "-p", "--safe-mode",
            "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--model", self.cfg.model, "--effort", self.cfg.effort,
            "--tools", "", "--no-session-persistence",
            "--system-prompt", system, "--json-schema", json.dumps(schema),
        ]
        message = json.dumps({"type": "user", "message": {"role": "user", "content": content}})
        try:
            proc = self.runner(cmd, input=message + "\n", capture_output=True, text=True,
                               timeout=_CLI_TIMEOUT_S, env=self.env, cwd=tempfile.gettempdir())
        except subprocess.TimeoutExpired:
            logger.warning("Capture %d: claude CLI timed out; will retry next run.", cid)
            return None, None

        result, limits = None, None
        for line in proc.stdout.splitlines():
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "result":
                result = msg
            elif msg.get("type") == "rate_limit_event":
                limits = msg.get("rate_limit_info") or {}
        if result is None:
            logger.warning("Capture %d: claude CLI exited %s with no result (%s); will retry next run.",
                           cid, proc.returncode, (proc.stderr or "").strip()[-300:])
            return None, None

        u = result.get("usage") or {}
        usage = Usage(
            (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
            + (u.get("cache_read_input_tokens") or 0),
            u.get("output_tokens") or 0,
        )
        self._note_limits(limits)
        if result.get("is_error"):
            status, text = result.get("api_error_status"), str(result.get("result") or "")
            if status in (401, 403) or "authenticat" in text.lower() or "/login" in text:
                raise AuthError(text)
            if status == 429 or (limits or {}).get("status") == "rejected":
                raise PlanLimitError(text or "plan usage limit reached")
            logger.warning("Capture %d: claude CLI error (%s): %s; will retry next run.", cid, status, text[:300])
            return None, usage
        if result.get("stop_reason") == "refusal":
            return _declined(), usage
        structured = result.get("structured_output")
        if structured is None:
            logger.warning("Capture %d: no structured verdict (%s); will retry next run.",
                           cid, result.get("subtype"))
            return None, usage
        try:
            return Verdict.model_validate(structured), usage
        except ValidationError as exc:
            logger.warning("Capture %d: unparseable verdict (%s); will retry next run.", cid, exc)
            return None, usage

    def _note_limits(self, limits: dict | None) -> None:
        windows = (limits or {}).get("unifiedWindows") or {}
        for name, w in windows.items():
            used = (w or {}).get("utilization")
            if used is not None and used >= self.cfg.max_plan_utilization:
                self.exhausted = (f"plan usage at {used:.0%} of the {name.replace('_', '-')} window "
                                  f"(review.max_plan_utilization {self.cfg.max_plan_utilization:.0%})")


def make_client(cfg: ReviewConfig):
    """Anthropic client: env/profile credentials first, then ``cfg.api_key_file``."""
    import anthropic

    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return anthropic.Anthropic()
    key_file = Path(cfg.api_key_file).expanduser()
    if key_file.is_file():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return anthropic.Anthropic(api_key=key)
    return anthropic.Anthropic()  # an `ant auth login` profile, if any


def make_backend(cfg: ReviewConfig, client=None):
    """An injected API client wins (tests); otherwise ``cfg.backend`` decides."""
    if client is not None or cfg.backend == "api":
        return ApiBackend(client or make_client(cfg), cfg)
    return CliBackend(cfg)


# ------------------------------------------------------------------- commands
def _judge(reviewer: Reviewer, clusters: list[Cluster], concurrency: int):
    """Yield (cluster, verdict | None, usage) as requests complete.

    On an AbortRun (auth failure, plan limit) queued requests are skipped, ones
    already running are still yielded, and the AbortRun is raised at the end.
    """
    stop = threading.Event()

    def task(cl: Cluster):
        return None if stop.is_set() else reviewer.review(cl.rep)

    abort: AbortRun | None = None
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(task, cl): cl for cl in clusters}
        for fut in as_completed(futures):
            try:
                out = fut.result()
            except AbortRun as exc:
                abort = abort or exc
                stop.set()
                continue
            if out is not None:
                yield futures[fut], *out
    if abort is not None:
        raise abort


def run_review(config: Config, *, dry_run: bool, limit: int | None, since: str | None,
               client=None, backend=None) -> RunSummary:
    cfg = config.review
    log = ReviewLog(config.storage.db_path)
    store = Store(config.storage.db_path, config.storage.captures_dir)
    summary = RunSummary()
    try:
        clusters = cluster_candidates(log.pending(since=since))[: limit or cfg.max_per_run]
        summary.clusters = len(clusters)
        if not clusters:
            logger.info("Nothing to review.")
            return summary
        backend = backend or make_backend(cfg, client)
        logger.info("Reviewing %d cluster(s) / %d capture(s)%s with %s via %s.",
                    len(clusters), sum(len(c.members) for c in clusters),
                    " (dry run)" if dry_run else "", cfg.model,
                    "the Anthropic API" if isinstance(backend, ApiBackend) else "the claude CLI")
        reviewer = Reviewer(backend, cfg, config.detection.animal_classes, Path(config.storage.captures_dir))
        try:
            for cluster, verdict, usage in _judge(reviewer, clusters, cfg.concurrency):
                if usage is not None:
                    summary.input_tokens += usage.input_tokens
                    summary.output_tokens += usage.output_tokens
                if verdict is None:
                    summary.failed += 1
                    continue
                action = decide_action(cluster, verdict, cfg, dry_run)
                if action == "deleted":
                    store.delete_many([m.id for m in cluster.members])
                log.record(cluster, verdict, action, cfg.model)
                summary.captures += len(cluster.members)
                summary.actions[action] += len(cluster.members)
                summary.verdicts[verdict.verdict] += 1
                logger.info("#%d %s %s@%.2f -> %s %.2f (%s) [%s]: %s",
                            cluster.rep.id, cluster.rep.camera_id, cluster.rep.label,
                            cluster.rep.confidence, verdict.verdict, verdict.confidence,
                            verdict.artifact if verdict.verdict == "not_animal" else verdict.species,
                            action, verdict.reason)
        except PlanLimitError as exc:
            summary.stopped = str(exc)
            logger.warning("Stopping early: %s. Unreviewed captures wait for the next run.", exc)
    finally:
        log.close()
        store.close()
    logger.info("Done: %d cluster(s), %d failed; captures by action %s; tokens in=%d out=%d.",
                summary.clusters, summary.failed, dict(summary.actions),
                summary.input_tokens, summary.output_tokens)
    return summary


def apply_pending(config: Config) -> int:
    log = ReviewLog(config.storage.db_path)
    store = Store(config.storage.db_path, config.storage.captures_dir)
    try:
        ids = log.would_delete_ids()
        n = store.delete_many(ids)
        log.mark_applied(ids)
        logger.info("Deleted %d capture(s) flagged by earlier dry runs.", n)
        return n
    finally:
        log.close()
        store.close()


def check(config: Config, ids: list[int], client=None, backend=None) -> None:
    """Judge specific captures and print JSON lines. Writes nothing."""
    cfg = config.review
    log = ReviewLog(config.storage.db_path)
    try:
        clusters = [Cluster([c]) for c in log.by_ids(ids)]
    finally:
        log.close()
    reviewer = Reviewer(backend or make_backend(cfg, client), cfg,
                        config.detection.animal_classes, Path(config.storage.captures_dir))
    for cluster, verdict, usage in _judge(reviewer, clusters, cfg.concurrency):
        c = cluster.rep
        print(json.dumps({"id": c.id, "label": c.label, "detector_confidence": round(c.confidence, 3),
                          **(verdict.model_dump() if verdict else {"verdict": None}),
                          "tokens_in": usage.input_tokens if usage else None,
                          "tokens_out": usage.output_tokens if usage else None}), flush=True)


def report(config: Config, since: str | None) -> None:
    log = ReviewLog(config.storage.db_path)
    try:
        rows = log.summary(since)
    finally:
        log.close()
    print(f"{'camera':24s} {'verdict':11s} {'action':13s} {'artifact':26s} {'n':>5s}")
    for r in rows:
        print(f"{r['camera_id']:24s} {r['verdict']:11s} {r['action']:13s} "
              f"{(r['artifact'] or ''):26s} {r['n']:5d}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wildlife-review", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=os.environ.get("WILDLIFE_CONFIG", "config.yaml"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="review unreviewed captures and delete false positives")
    p_run.add_argument("--dry-run", action="store_true", help="record would_delete instead of deleting")
    p_run.add_argument("--limit", type=int, help="max requests this run (default review.max_per_run)")
    p_run.add_argument("--since", help="only captures at/after this ISO date")
    p_run.add_argument("--scheduled", action="store_true", help="no-op unless review.enabled")
    sub.add_parser("apply", help="delete captures earlier dry runs flagged")
    p_check = sub.add_parser("check", help="judge specific capture ids; prints JSON, writes nothing")
    p_check.add_argument("ids", nargs="+", type=int)
    p_rep = sub.add_parser("report", help="summarise review_log")
    p_rep.add_argument("--since")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    config = load_config(args.config)
    try:
        if args.cmd == "run":
            if args.scheduled and not config.review.enabled:
                logger.info("review.enabled is false; nothing to do.")
                return 0
            run_review(config, dry_run=args.dry_run, limit=args.limit, since=args.since)
        elif args.cmd == "apply":
            apply_pending(config)
        elif args.cmd == "check":
            check(config, args.ids)
        elif args.cmd == "report":
            report(config, args.since)
    except AuthError as exc:
        if config.review.backend == "api":
            logger.error("Anthropic credentials missing or rejected (%s). Set ANTHROPIC_API_KEY, "
                         "run `ant auth login`, or put a key in %s.", exc, config.review.api_key_file)
        else:
            logger.error("The claude CLI can't run the review (%s). Run `claude auth login` as this "
                         "user, or put a `claude setup-token` token in %s.", exc,
                         config.review.oauth_token_file)
        return 2
    except PlanLimitError as exc:  # `check` has no partial-run handling of its own
        logger.warning("Stopped: %s.", exc)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
