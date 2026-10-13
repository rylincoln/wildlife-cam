"""wildlife-review: clustering, delete policy, dry-run/apply, and the API request shape.

The Anthropic client is faked, so nothing here touches the network.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from wildlife.config import Config, ReviewConfig
from wildlife.models import Detection
from wildlife.review import (
    AuthError,
    Candidate,
    Cluster,
    apply_pending,
    build_images,
    cluster_candidates,
    decide_action,
    output_schema,
    run_review,
)
from wildlife.store import Store

SPECIES = ["person", "mule_deer", "red_fox"]


def _config(tmp_path: Path, **review) -> Config:
    return Config.model_validate({
        "cameras": [{
            "id": "cam1", "host": "192.168.1.101", "username": "admin", "password": "x",
            "rtsp_main": "rtsp://{username}:{password}@{host}:554/main",
            "rtsp_sub": "rtsp://{username}:{password}@{host}:554/sub",
        }],
        "event_source": "reolink_native",
        "capture": {"burst_frames": 3, "burst_interval_ms": 100, "stream": "main",
                    "rtsp_timeout_s": 5, "max_concurrent": 1},
        "detection": {"model_path": "m.pt", "device": "cpu", "animal_classes": SPECIES,
                      "confidence_threshold": 0.4, "min_box_area_frac": 0.01,
                      "save_best_only": False},
        "dedupe": {"cooldown_s": 0},
        "storage": {"captures_dir": str(tmp_path / "caps"), "db_path": str(tmp_path / "c.db")},
        "retention": {"max_age_days": 30},
        "gallery": {"host": "0.0.0.0", "port": 8080, "page_size": 60},
        "resource_guard": {"detect_every_nth_event": 1, "max_burst_per_minute": 20},
        "review": {"concurrency": 1, **review},
    })


def _save(store: Store, ts: datetime, box, conf=0.8, label="red_fox") -> int:
    frame = np.full((240, 320, 3), 30, np.uint8)
    det = Detection(label=label, confidence=conf, box_xyxy=box, box_area_frac=0.1)
    return store.save_capture(camera_id="cam1", event_ts=ts, capture_ts=ts, frame=frame, det=det)


def _response(verdict="not_animal", confidence=0.95, stop="end_turn"):
    body = {"reason": "test", "verdict": verdict, "confidence": confidence,
            "species": "red_fox" if verdict == "animal" else "none",
            "artifact": "insect" if verdict == "not_animal" else "none"}
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=json.dumps(body))],
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20))


class _FakeClient:
    """Answers by the local time in the request's last text block."""

    def __init__(self, by_time: dict[str, object]):
        self.by_time = by_time
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        text = kw["messages"][0]["content"][-1]["text"]
        answer = next(v for t, v in self.by_time.items() if t in text)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def world(tmp_path):
    """Four events: an insect burst (2 overlapping frames), a real fox, a human-
    reviewed capture the reviewer calls an insect, and a low-confidence call."""
    cfg = _config(tmp_path)
    store = Store(cfg.storage.db_path, cfg.storage.captures_dir)
    store.init_schema()
    base = datetime.now() - timedelta(hours=3)
    t = [base + timedelta(minutes=i) for i in range(4)]
    ids = {
        "insect": [_save(store, t[0], (10, 10, 100, 100), 0.9), _save(store, t[0], (12, 12, 102, 102), 0.7)],
        "fox": [_save(store, t[1], (50, 50, 150, 150))],
        "human": [_save(store, t[2], (50, 50, 150, 150))],
        "unsure": [_save(store, t[3], (50, 50, 150, 150))],
    }
    store.update_label(ids["human"][0], "red_fox")  # a human reclassified it
    store.close()
    times = {k: t[i].strftime("%Y-%m-%d %H:%M") for i, k in enumerate(["insect", "fox", "human", "unsure"])}
    client = _FakeClient({
        times["insect"]: _response("not_animal", 0.95),
        times["fox"]: _response("animal", 0.9),
        times["human"]: _response("not_animal", 0.97),
        times["unsure"]: _response("not_animal", 0.6),  # below delete_min_confidence
    })
    return cfg, ids, client


def _log(cfg) -> dict[int, str]:
    conn = sqlite3.connect(cfg.storage.db_path)
    try:
        return {cid: action for cid, action in conn.execute("SELECT capture_id, action FROM review_log")}
    finally:
        conn.close()


def _exists(cfg, cid) -> bool:
    conn = sqlite3.connect(cfg.storage.db_path)
    try:
        return conn.execute("SELECT 1 FROM captures WHERE id=?", (cid,)).fetchone() is not None
    finally:
        conn.close()


def test_run_deletes_only_confident_non_animals_and_respects_humans(world):
    cfg, ids, client = world
    summary = run_review(cfg, dry_run=False, limit=None, since=None, client=client)

    assert len(client.calls) == 4  # the two insect frames shared one request
    assert summary.actions == {"deleted": 2, "kept": 2, "kept_human": 1}
    for cid in ids["insect"]:
        assert not _exists(cfg, cid)
    for key in ("fox", "human", "unsure"):
        assert _exists(cfg, ids[key][0])
    log = _log(cfg)
    assert {log[c] for c in ids["insect"]} == {"deleted"}  # verdicts outlive the images
    assert log[ids["human"][0]] == "kept_human"
    assert log[ids["unsure"][0]] == "kept"
    # Everything is now reviewed: a second run makes no requests.
    run_review(cfg, dry_run=False, limit=None, since=None, client=client)
    assert len(client.calls) == 4


def test_dry_run_then_apply(world):
    cfg, ids, client = world
    run_review(cfg, dry_run=True, limit=None, since=None, client=client)
    assert all(_exists(cfg, c) for c in ids["insect"])
    assert {_log(cfg)[c] for c in ids["insect"]} == {"would_delete"}

    assert apply_pending(cfg) == 2
    assert not any(_exists(cfg, c) for c in ids["insect"])
    assert {_log(cfg)[c] for c in ids["insect"]} == {"deleted"}
    assert _exists(cfg, ids["human"][0])


def test_request_shape(world):
    cfg, _, client = world
    run_review(cfg, dry_run=True, limit=1, since=None, client=client)
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["schema"] == output_schema(SPECIES)
    images = [b for b in kw["messages"][0]["content"] if b["type"] == "image"]
    assert len(images) == 2 and all(b["source"]["media_type"] == "image/jpeg" for b in images)
    assert "red_fox" in kw["system"]


def test_refusal_is_kept_as_unsure(world):
    cfg, ids, client = world
    for k in client.by_time:
        client.by_time[k] = _response(stop="refusal")
    summary = run_review(cfg, dry_run=False, limit=None, since=None, client=client)
    assert summary.verdicts == {"unsure": 4}
    assert all(_exists(cfg, c) for group in ids.values() for c in group)


def test_auth_failure_aborts_without_touching_anything(world):
    import anthropic
    import httpx2

    cfg, ids, client = world
    err = anthropic.AuthenticationError(
        "invalid x-api-key",
        response=httpx2.Response(401, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")),
        body=None,
    )
    for k in client.by_time:
        client.by_time[k] = err
    with pytest.raises(AuthError):
        run_review(cfg, dry_run=False, limit=None, since=None, client=client)
    assert _log(cfg) == {}
    assert all(_exists(cfg, c) for group in ids.values() for c in group)


def _cand(cid, ts, box, conf, human=False):
    return Candidate(cid, "cam1", ts, "red_fox", conf, box, "x.jpg", human)


def test_cluster_candidates_merges_overlap_within_an_event_only():
    cands = [
        _cand(1, "T1", (0, 0, 10, 10), 0.5),
        _cand(2, "T1", (1, 1, 11, 11), 0.9),    # same object as 1, higher confidence
        _cand(3, "T1", (50, 50, 60, 60), 0.7),  # a second object in the same burst
        _cand(4, "T2", (0, 0, 10, 10), 0.6),    # same box, different event
    ]
    clusters = cluster_candidates(cands)
    assert sorted(len(c.members) for c in clusters) == [1, 1, 2]
    merged = next(c for c in clusters if len(c.members) == 2)
    assert merged.rep.id == 2


def test_decide_action():
    cfg = ReviewConfig(delete_min_confidence=0.85)
    from wildlife.review import Verdict

    def v(verdict, conf):
        return Verdict(reason="r", verdict=verdict, species="none", artifact="none", confidence=conf)

    plain = Cluster([_cand(1, "T", (0, 0, 1, 1), 0.5)])
    human = Cluster([_cand(1, "T", (0, 0, 1, 1), 0.5), _cand(2, "T", (0, 0, 1, 1), 0.4, human=True)])
    assert decide_action(plain, v("not_animal", 0.9), cfg, dry_run=False) == "deleted"
    assert decide_action(plain, v("not_animal", 0.9), cfg, dry_run=True) == "would_delete"
    assert decide_action(plain, v("not_animal", 0.8), cfg, dry_run=False) == "kept"
    assert decide_action(plain, v("unsure", 1.0), cfg, dry_run=False) == "kept"
    assert decide_action(plain, v("animal", 0.99), cfg, dry_run=False) == "kept"
    assert decide_action(human, v("not_animal", 0.99), cfg, dry_run=False) == "kept_human"


def test_build_images_sizes(tmp_path):
    path = tmp_path / "f.jpg"
    Image.new("RGB", (3840, 2160), (40, 40, 40)).save(path)
    full, crop = build_images(path, (3700, 2000, 3760, 2050))  # tiny box in a corner
    with Image.open(__import__("io").BytesIO(full)) as f:
        assert max(f.size) == 1568
    with Image.open(__import__("io").BytesIO(crop)) as c:
        assert 512 <= max(c.size) <= 768


# ------------------------------------------------------------- claude CLI backend
class _FakeCli:
    """Stands in for subprocess.run on `claude -p`: answers by the local time in the
    stream-json user message, like _FakeClient, with a canned CLI event stream."""

    def __init__(self, by_time: dict[str, dict], utilization: float = 0.1):
        self.by_time = by_time
        self.utilization = utilization
        self.calls: list[dict] = []

    def __call__(self, cmd, *, input, **kw):
        self.calls.append({"cmd": cmd, "input": input, **kw})
        content = json.loads(input)["message"]["content"]
        result = next(v for t, v in self.by_time.items() if t in content[-1]["text"])
        limits = {"status": "allowed", "unifiedWindows": {
            "five_hour": {"utilization": self.utilization}, "seven_day": {"utilization": 0.01}}}
        lines = [{"type": "system", "subtype": "init"},
                 {"type": "rate_limit_event", "rate_limit_info": limits},
                 {"type": "result", "subtype": "success", "is_error": False, "stop_reason": "end_turn",
                  "usage": {"input_tokens": 2, "cache_creation_input_tokens": 4000,
                            "cache_read_input_tokens": 700, "output_tokens": 50}, **result}]
        return SimpleNamespace(returncode=0, stdout="\n".join(json.dumps(x) for x in lines), stderr="")


def _cli_verdict(verdict, confidence=0.95):
    return {"structured_output": json.loads(_response(verdict, confidence).content[0].text)}


@pytest.fixture
def cli_world(world, monkeypatch):
    cfg, ids, client = world
    times = {}
    for t, resp in client.by_time.items():
        body = json.loads(resp.content[0].text)
        times[t] = _cli_verdict(body["verdict"], body["confidence"])
    monkeypatch.setattr("wildlife.review.resolve_claude_bin", lambda name: "/usr/bin/claude")
    return cfg, ids, _FakeCli(times)


def _cli_backend(cfg, runner):
    from wildlife.review import CliBackend

    return CliBackend(cfg.review, runner=runner)


def test_cli_backend_runs_review_on_the_cli_login(cli_world, monkeypatch, tmp_path):
    cfg, ids, runner = cli_world
    token = tmp_path / "token"
    token.write_text("sk-ant-oat-test\n")
    cfg.review.oauth_token_file = str(token)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    summary = run_review(cfg, dry_run=False, limit=None, since=None, backend=_cli_backend(cfg, runner))

    assert len(runner.calls) == 4
    assert summary.actions == {"deleted": 2, "kept": 2, "kept_human": 1}
    assert not any(_exists(cfg, c) for c in ids["insect"])
    assert (summary.input_tokens, summary.output_tokens) == (4 * 4702, 4 * 50)
    call = runner.calls[0]
    cmd = call["cmd"]
    assert cmd[:2] == ["/usr/bin/claude", "-p"] and "--safe-mode" in cmd
    opt = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
    assert opt["--model"] == "claude-opus-5-5" and opt["--effort"] == "low"
    assert opt["--tools"] == "" and opt["--input-format"] == "stream-json"
    assert json.loads(opt["--json-schema"]) == output_schema(SPECIES)
    assert "red_fox" in opt["--system-prompt"]
    images = [b for b in json.loads(call["input"])["message"]["content"] if b["type"] == "image"]
    assert len(images) == 2
    assert call["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-test"


def test_cli_auth_failure_aborts(cli_world):
    cfg, ids, runner = cli_world
    for t in runner.by_time:
        runner.by_time[t] = {"is_error": True, "api_error_status": 401,
                             "result": "Failed to authenticate. API Error: 401 Invalid bearer token"}
    with pytest.raises(AuthError):
        run_review(cfg, dry_run=False, limit=None, since=None, backend=_cli_backend(cfg, runner))
    assert _log(cfg) == {}
    assert all(_exists(cfg, c) for group in ids.values() for c in group)


def test_cli_stops_when_plan_usage_passes_the_limit(cli_world):
    cfg, ids, runner = cli_world
    runner.utilization = 0.85  # over the default max_plan_utilization of 0.8
    summary = run_review(cfg, dry_run=True, limit=None, since=None, backend=_cli_backend(cfg, runner))

    assert len(runner.calls) == 1  # concurrency 1: the first answer is kept, then it stops
    assert summary.clusters == 4 and summary.captures >= 1
    assert "five-hour" in summary.stopped
    # The rest are still pending for the next run.
    runner.utilization = 0.1
    run_review(cfg, dry_run=True, limit=None, since=None, backend=_cli_backend(cfg, runner))
    assert len(runner.calls) == 4


def test_cli_refusal_and_missing_output_keep_the_capture(cli_world):
    cfg, ids, runner = cli_world
    times = list(runner.by_time)
    runner.by_time[times[0]] = {"stop_reason": "refusal"}
    runner.by_time[times[1]] = {"subtype": "error_max_structured_output_retries"}
    summary = run_review(cfg, dry_run=False, limit=None, since=None, backend=_cli_backend(cfg, runner))
    assert summary.verdicts["unsure"] == 1 and summary.failed == 1
    assert all(_exists(cfg, c) for c in ids["insect"])  # the refused burst
    assert ids["fox"][0] not in _log(cfg)  # the failed one is retried next run


def test_resolve_claude_bin(tmp_path, monkeypatch):
    from wildlife.review import resolve_claude_bin

    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_claude_bin("/opt/x/claude") == "/opt/x/claude"
    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert resolve_claude_bin("claude") == str(exe)
    with pytest.raises(AuthError):
        resolve_claude_bin("no-such-claude-binary")
