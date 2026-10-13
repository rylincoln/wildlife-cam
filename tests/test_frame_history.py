"""Scrollable Recent sightings: Store.query_events paging + GET /frame/sightings."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from werkzeug.security import generate_password_hash

import wildlife.frame.routes as frame_routes
from tests.test_frame_routes import SECRET, _client, _config, _get
from wildlife.config import FrameConfig, RemoteConfig
from wildlife.frame.wildlife import build_wildlife
from wildlife.review import ReviewLog
from wildlife.store import Store

NOW = datetime(2026, 10, 12, 17, 0)
KINDS = ("reolink", "continuous")


def _store(db: Path) -> Store:
    store = Store(db, db.parent / "captures")
    store.init_schema()
    return store


def _cap(db: Path, camera: str, event_ts: str, capture_ts: str, conf: float = 0.9,
         kind: str = "reolink", label: str = "coyote") -> int:
    conn = sqlite3.connect(db)
    cur = conn.execute(
        "INSERT INTO captures (camera_id, event_ts, capture_ts, label, confidence, image_path, "
        "thumb_path, source_kind) VALUES (?, ?, ?, ?, ?, 'i.jpg', 't.jpg', ?)",
        (camera, event_ts, capture_ts, label, conf, kind),
    )
    conn.commit()
    conn.close()
    return cur.lastrowid


def _verdict(db: Path, capture_id: int, verdict: str) -> None:
    ReviewLog(db).close()
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO review_log (capture_id, representative_id, camera_id, capture_ts, label, "
        "verdict, action, reviewed_at) VALUES (?, ?, 'cam', 't', 'x', ?, 'kept', 'now')",
        (capture_id, capture_id, verdict),
    )
    conn.commit()
    conn.close()


def _events(store: Store, **kwargs) -> list[tuple[int, str]]:
    return [(r["id"], r["event_latest"]) for r in store.query_events(source_kinds=KINDS, **kwargs)]


# ---- Store.query_events ------------------------------------------------------

def test_events_are_newest_first_one_best_frame_each(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    store = _store(db)
    _cap(db, "water_tower", "2026-10-12T06:00:00", "2026-10-12T06:00:01", 0.55)
    best = _cap(db, "water_tower", "2026-10-12T06:00:00", "2026-10-12T06:00:02", 0.91)
    _cap(db, "water_tower", "2026-10-12T06:00:00", "2026-10-12T06:00:03", 0.70)
    other = _cap(db, "fenceline", "2026-10-12T06:00:00", "2026-10-12T06:00:00.5")  # same time, other camera
    old = _cap(db, "water_tower", "2026-10-11T22:00:00", "2026-10-11T22:00:01", kind="continuous")
    _cap(db, "water_tower", "2026-10-12T07:00:00", "2026-10-12T07:00:00", kind="audio")  # birds don't count
    assert _events(store, limit=10) == [
        (best, "2026-10-12T06:00:03"),
        (other, "2026-10-12T06:00:00.5"),
        (old, "2026-10-11T22:00:01"),
    ]


def test_paging_back_covers_every_event_once(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    store = _store(db)
    ids = [_cap(db, "water_tower", f"2026-10-0{d}T12:00:00", f"2026-10-0{d}T12:00:01") for d in range(1, 6)]
    page1 = _events(store, limit=2)
    page2 = _events(store, limit=2, before=page1[-1][1])
    page3 = _events(store, limit=2, before=page2[-1][1])
    assert _events(store, limit=2, before=page3[-1][1]) == []
    seen = [i for page in (page1, page2, page3) for i, _ in page]
    assert seen == ids[::-1]


def test_a_burst_is_never_split_across_pages(tmp_path: Path) -> None:
    # Camera A's burst straddles camera B's: grouping happens before the page cut, so A's
    # early frame never reappears as an event of its own on the next page.
    db = tmp_path / "c.db"
    store = _store(db)
    a_early = _cap(db, "a", "2026-10-12T10:00:00", "2026-10-12T10:00:00", 0.95)
    _cap(db, "a", "2026-10-12T10:00:00", "2026-10-12T10:00:05", 0.40)
    b = _cap(db, "b", "2026-10-12T10:00:02", "2026-10-12T10:00:03")
    page1 = _events(store, limit=1)
    assert page1 == [(a_early, "2026-10-12T10:00:05")]  # best frame of A, keyed by A's newest
    assert _events(store, limit=5, before=page1[-1][1]) == [(b, "2026-10-12T10:00:03")]


def test_confirmed_only_keeps_reviewer_confirmed_frames(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    store = _store(db)
    assert store.query_events(source_kinds=KINDS, confirmed_only=True) == []  # no review yet
    moth = _cap(db, "a", "2026-10-12T01:00:00", "2026-10-12T01:00:00", 0.99)
    fox = _cap(db, "a", "2026-10-12T02:00:00", "2026-10-12T02:00:00")
    _cap(db, "a", "2026-10-12T03:00:00", "2026-10-12T03:00:00")  # not reviewed yet
    _verdict(db, moth, "not_animal")
    _verdict(db, fox, "animal")
    assert [i for i, _ in _events(store, confirmed_only=True)] == [fox]
    assert len(_events(store)) == 3


# ---- GET /frame/sightings ----------------------------------------------------

def _rows(n: int, start_day: int = 28) -> list[dict]:
    return [
        {"id": i, "camera_id": "water_tower", "label": "coyote", "confidence": 0.9,
         "capture_ts": f"2026-09-{start_day - i:02d}T12:00:00", "event_latest": f"2026-09-{start_day - i:02d}T12:00:00",
         "thumb_url": f"/thumb/{i}", "image_url": f"/image/{i}"}
        for i in range(n)
    ]


def test_route_returns_a_page_and_the_cursor_for_the_next(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(frame_routes, "_now", lambda: NOW)
    calls: list = []

    def events(before, limit):
        calls.append((before, limit))
        return _rows(limit)  # one more than a page: there is more history

    monkeypatch.setattr(Store, "query_events", lambda self, **kw: events(kw.get("before"), kw["limit"]))
    resp = _get(_client(_config(tmp_path)), "/frame/sightings?before=2026-10-01T00:00:00")
    assert resp.status_code == 200 and resp.headers["Cache-Control"] == "no-store"
    data = resp.get_json()
    assert calls == [("2026-10-01T00:00:00", 13)]
    assert len(data["sightings"]) == 12
    first = data["sightings"][0]
    assert (first["label"], first["camera"], first["time"], first["ts"]) == (
        "Coyote", "water tower", "Sep 28, 12:00 PM", "2026-09-28T12:00:00"
    )
    assert data["next"] == data["sightings"][-1]["ts"] == "2026-09-17T12:00:00"


def test_route_last_page_has_no_cursor(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Store, "query_events", lambda self, **kw: _rows(3))
    data = _get(_client(_config(tmp_path)), "/frame/sightings").get_json()
    assert (len(data["sightings"]), data["next"]) == (3, None)


@pytest.mark.parametrize("bad", ["yesterday", "2026-13-40", "1;DROP TABLE captures"])
def test_route_rejects_a_malformed_cursor(tmp_path: Path, bad: str) -> None:
    assert _get(_client(_config(tmp_path)), f"/frame/sightings?before={bad}").status_code == 400


def test_route_survives_a_store_failure(tmp_path: Path, monkeypatch) -> None:
    def broken(self, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(Store, "query_events", broken)
    assert _get(_client(_config(tmp_path)), "/frame/sightings").get_json() == {"sightings": [], "next": None}


def test_route_is_lan_only_and_off_with_the_frame(tmp_path: Path) -> None:
    remote = RemoteConfig(enabled=True, share_secret_hash=generate_password_hash(SECRET))
    assert _client(_config(tmp_path, remote=remote)).get(f"/frame/sightings?key={SECRET}").status_code == 404
    client = _client(_config(tmp_path))
    assert _get(client, "/frame/sightings", headers={"Cf-Ray": "8a1b2c3d4e5f-SJC"}).status_code == 404
    off = _client(_config(tmp_path, frame=FrameConfig(enabled=False)))
    assert _get(off, "/frame/sightings").status_code == 404


def test_route_reads_the_real_store_confirmed_only_while_review_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(frame_routes, "_now", lambda: NOW)
    cfg = _config(tmp_path)
    db = cfg.storage.db_path
    _store(db).close()
    fox = _cap(db, "south_trail", "2026-10-12T05:20:00", "2026-10-12T05:20:01", kind="continuous")
    moth = _cap(db, "south_trail", "2026-10-12T05:10:00", "2026-10-12T05:10:01", kind="continuous")
    _verdict(db, fox, "animal")
    _verdict(db, moth, "not_animal")
    data = _get(_client(cfg), "/frame/sightings").get_json()
    assert [s["thumb_url"] for s in data["sightings"]] == [f"/thumb/{fox}", f"/thumb/{moth}"]
    cfg.review.enabled = True
    data = _get(_client(cfg), "/frame/sightings").get_json()
    assert [s["thumb_url"] for s in data["sightings"]] == [f"/thumb/{fox}"]
    assert data["sightings"][0]["ts"] == "2026-10-12T05:20:01"


# ---- first page (from /frame/wildlife) carries the cursor too ----------------

def test_wildlife_sightings_carry_their_event_time() -> None:
    caps = [
        {"id": 2, "camera_id": "a", "event_ts": "2026-10-12T06:00:00", "capture_ts": "2026-10-12T06:00:01",
         "label": "elk", "confidence": 0.9, "thumb_url": "/thumb/2", "image_url": "/image/2"},
        {"id": 3, "camera_id": "a", "event_ts": "2026-10-12T06:00:00", "capture_ts": "2026-10-12T06:00:04",
         "label": "elk", "confidence": 0.5, "thumb_url": "/thumb/3", "image_url": "/image/3"},
    ]
    sighting = build_wildlife(caps, [], NOW)["sightings"][0]
    assert (sighting["thumb_url"], sighting["ts"]) == ("/thumb/2", "2026-10-12T06:00:04")


# ---- page --------------------------------------------------------------------

def test_page_scrolls_the_sightings_grid_and_loads_older_pages(tmp_path: Path) -> None:
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    assert '"/frame/sightings?before=" + encodeURIComponent(' in body
    assert 'sightingList.addEventListener("scroll"' in body
    assert "No older sightings" in body
    css = (Path(frame_routes.__file__).resolve().parents[1] / "gallery" / "static" / "frame.css").read_text()
    assert "#sighting-list" in css and "overflow-y: auto" in css
