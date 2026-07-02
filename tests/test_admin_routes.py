"""Flask-level tests for the password-gated ``/admin`` blueprint.

Hardware-free: the camera *probe* (which shells out to cv2/reolink) is
monkeypatched, so no network or GPU is touched. Storage points under ``tmp_path``.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from werkzeug.security import generate_password_hash

from wildlife.admin import config_io as cio
from wildlife.config import load_config
from wildlife.gallery.app import create_app
from tests.test_admin_config_io import render_base  # reuse the minimal config

_PASSWORD = "supersecret"


def _auth(pw: str = _PASSWORD) -> dict:
    token = base64.b64encode(f"admin:{pw}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _seed_capture(cfgp: Path, label: str = "deer") -> int:
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
            det=Detection(label=label, confidence=0.8, box_xyxy=(1, 1, 5, 5), box_area_frac=0.1),
        )
    finally:
        store.close()


def _open_store(cfgp):
    from wildlife.config import load_config
    from wildlife.store import Store
    cfg = load_config(cfgp)
    s = Store(db_path=cfg.storage.db_path, captures_dir=cfg.storage.captures_dir)
    return s


@pytest.fixture()
def ctx(tmp_path: Path):
    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(render_base(tmp_path), "utf-8")
    cio.set_admin_password(cfgp, generate_password_hash(_PASSWORD))
    app = create_app(load_config(cfgp), config_path=str(cfgp))
    app.config.update(TESTING=True)
    return app.test_client(), cfgp


def test_auth_required(ctx) -> None:
    client, _ = ctx
    assert client.get("/admin/").status_code == 401
    assert client.get("/admin/", headers=_auth("wrong")).status_code == 401
    assert client.get("/admin/detection").status_code == 401
    assert client.post("/admin/test-camera", json={}).status_code == 401


def test_pages_render_when_authed(ctx) -> None:
    client, _ = ctx
    for path in ("/admin/", "/admin/detection", "/admin/cameras",
                 "/admin/cameras/new", "/admin/cameras/north_field/edit", "/admin/password"):
        assert client.get(path, headers=_auth()).status_code == 200


def test_detection_save_valid_and_invalid(ctx) -> None:
    client, cfgp = ctx
    common = {
        "model_path": "models/yolov8s.pt", "device": "mps", "animal_classes": "bird\ncat, dog",
        "min_box_area_frac": "0.01", "stream": "sub", "burst_frames": "5",
        "burst_interval_ms": "200", "rtsp_timeout_s": "10", "max_concurrent": "1",
        "cooldown_s": "30", "detect_every_nth_event": "1", "max_burst_per_minute": "20",
    }
    r = client.post("/admin/detection", headers=_auth(),
                    data={**common, "confidence_threshold": "0.6", "save_best_only": "on"})
    assert r.status_code == 302
    cfg = load_config(cfgp)
    assert cfg.detection.confidence_threshold == 0.6
    assert cfg.detection.animal_classes == ["bird", "cat", "dog"]

    # Out-of-range confidence is rejected and re-rendered with an error.
    r = client.post("/admin/detection", headers=_auth(),
                    data={**common, "confidence_threshold": "2.0"})
    assert r.status_code == 200
    assert b"Could not save" in r.data
    assert load_config(cfgp).detection.confidence_threshold == 0.6  # unchanged


def test_camera_save_runs_probe_and_gates(ctx, monkeypatch) -> None:
    client, cfgp = ctx
    calls = {"n": 0}

    def fake_probe(params, timeout):
        calls["n"] += 1
        return {"streams": {"sub": {"ok": True, "width": 640, "height": 360}}}

    monkeypatch.setattr("wildlife.admin.routes._run_probe", fake_probe)
    r = client.post("/admin/cameras/new", headers=_auth(), data={
        "id": "south_trail", "host": "192.168.1.72", "username": "admin", "password": "pw",
        "rtsp_main": "rtsp://{username}:{password}@{host}:554/Preview_01_main",
        "rtsp_sub": "rtsp://{username}:{password}@{host}:554/Preview_01_sub", "onvif_port": "8000",
    })
    assert r.status_code == 302
    assert calls["n"] == 1  # the pre-save frame check ran
    assert "south_trail" in [c.id for c in load_config(cfgp).cameras]


def test_camera_save_rejected_when_probe_fails(ctx, monkeypatch) -> None:
    client, cfgp = ctx
    monkeypatch.setattr(
        "wildlife.admin.routes._run_probe",
        lambda params, timeout: {"streams": {"sub": {"ok": False, "error": "no frame decoded"}}},
    )
    r = client.post("/admin/cameras/new", headers=_auth(), data={
        "id": "broken", "host": "10.0.0.9", "username": "admin", "password": "pw",
        "rtsp_main": "rtsp://a/main", "rtsp_sub": "rtsp://a/sub", "onvif_port": "8000",
    })
    assert r.status_code == 200
    assert b"Camera test failed" in r.data
    assert "broken" not in [c.id for c in load_config(cfgp).cameras]  # not written


def test_camera_save_skip_test_bypasses_probe(ctx, monkeypatch) -> None:
    client, cfgp = ctx

    def boom(params, timeout):  # must not be called
        raise AssertionError("probe should be skipped")

    monkeypatch.setattr("wildlife.admin.routes._run_probe", boom)
    r = client.post("/admin/cameras/new", headers=_auth(), data={
        "id": "manual", "host": "10.0.0.5", "username": "admin", "password": "pw",
        "rtsp_main": "rtsp://a/main", "rtsp_sub": "rtsp://a/sub", "onvif_port": "8000",
        "skip_test": "on",
    })
    assert r.status_code == 302
    assert "manual" in [c.id for c in load_config(cfgp).cameras]


def test_camera_edit_blank_password_keeps_stored(ctx, monkeypatch) -> None:
    client, cfgp = ctx
    monkeypatch.setattr(
        "wildlife.admin.routes._run_probe",
        lambda params, timeout: {"streams": {"sub": {"ok": True, "width": 640, "height": 360}}},
    )
    # Edit north_field, leaving password blank -> stored password must persist.
    r = client.post("/admin/cameras/north_field/edit", headers=_auth(), data={
        "id": "north_field", "host": "192.168.1.99", "username": "admin", "password": "",
        "rtsp_main": "rtsp://{username}:{password}@{host}:554/Preview_01_main",
        "rtsp_sub": "rtsp://{username}:{password}@{host}:554/Preview_01_sub",
        "onvif_port": "8000", "original_id": "north_field",
    })
    assert r.status_code == 302
    cam = load_config(cfgp).cameras[0]
    assert cam.host == "192.168.1.99"
    assert cam.password == "secret-pw"  # unchanged
    assert cio.read_raw(cfgp)["cameras"][0]["password"] == "secret-pw"


def test_camera_delete(ctx) -> None:
    client, cfgp = ctx
    r = client.post("/admin/cameras/north_field/delete", headers=_auth())
    assert r.status_code == 302
    assert load_config(cfgp).cameras == []


def test_test_camera_endpoint_returns_probe_json(ctx, monkeypatch) -> None:
    client, _ = ctx
    monkeypatch.setattr(
        "wildlife.admin.routes._run_probe",
        lambda params, timeout: {"streams": {"sub": {"ok": True, "width": 640, "height": 360}},
                                 "reolink": {"attempted": True, "ok": True, "channels": [0]}},
    )
    r = client.post("/admin/test-camera", headers=_auth(), json={
        "host": "1.2.3.4", "username": "admin", "password": "pw",
        "rtsp_main": "rtsp://a/main", "rtsp_sub": "rtsp://a/sub", "onvif_port": "8000",
    })
    assert r.status_code == 200
    assert r.get_json()["streams"]["sub"]["ok"] is True


def test_test_camera_validates_required_fields(ctx) -> None:
    client, _ = ctx
    r = client.post("/admin/test-camera", headers=_auth(),
                    json={"username": "admin", "onvif_port": "8000"})
    assert r.status_code == 200
    assert "error" in r.get_json()


def test_password_change(ctx) -> None:
    client, cfgp = ctx
    r = client.post("/admin/password", headers=_auth(),
                    data={"password": "brand-new-pw", "confirm": "brand-new-pw"})
    assert r.status_code == 302
    # New password authenticates; old one no longer does.
    assert client.get("/admin/", headers=_auth("brand-new-pw")).status_code == 200
    assert client.get("/admin/", headers=_auth(_PASSWORD)).status_code == 401


def test_password_change_mismatch_rejected(ctx) -> None:
    client, _ = ctx
    r = client.post("/admin/password", headers=_auth(),
                    data={"password": "brand-new-pw", "confirm": "different-pw"})
    assert r.status_code == 200
    assert b"do not match" in r.data


def test_admin_disabled_returns_404(tmp_path: Path) -> None:
    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(render_base(tmp_path), "utf-8")
    cio.write(cfgp, lambda doc: doc.__setitem__("admin", {"enabled": False}))
    app = create_app(load_config(cfgp), config_path=str(cfgp))
    app.config.update(TESTING=True)
    assert app.test_client().get("/admin/", headers=_auth()).status_code == 404


def test_no_password_fails_closed(tmp_path: Path) -> None:
    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(render_base(tmp_path), "utf-8")
    app = create_app(load_config(cfgp), config_path=str(cfgp))
    app.config.update(TESTING=True)
    assert app.test_client().get("/admin/").status_code == 403


def test_cross_origin_post_blocked_but_same_origin_ok(ctx) -> None:
    client, _ = ctx
    body = {"password": "brand-new-pw", "confirm": "brand-new-pw"}
    # A cross-origin POST (attacker page) is rejected before auth even runs.
    r = client.post("/admin/password", headers={**_auth(), "Origin": "http://evil.example"},
                    data=body)
    assert r.status_code == 403
    # A same-origin POST (Origin matches host) is allowed through.
    r = client.post("/admin/password",
                    headers={**_auth(), "Origin": "http://localhost"}, data=body)
    assert r.status_code == 302


def test_gallery_live_reloads_config_without_restart(ctx) -> None:
    """An external edit to config.yaml is reflected on the next request."""
    client, cfgp = ctx
    # Detection page reflects the seed value...
    assert b"0.55" in client.get("/admin/detection", headers=_auth()).data
    # ...then an out-of-band write (as the reloader/CLI would do) is picked up
    # by the same running app via its mtime-cached get_config().
    cio.update_section(cfgp, "detection", {"confidence_threshold": 0.9})
    body = client.get("/admin/detection", headers=_auth()).data
    assert b"0.9" in body


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
    # The row is actually gone (302 alone would pass even on a no-op).
    s = _open_store(cfgp)
    assert s.get(cid) is None
    s.close()
    # GET on a POST-only route is rejected.
    assert client.get(f"/admin/captures/{cid}/delete", headers=_auth()).status_code == 405


def test_capture_reclassify_route(ctx) -> None:
    client, cfgp = ctx
    cid = _seed_capture(cfgp)  # label "deer"
    _seed_capture(cfgp, label="elk")  # so "elk" is a real target in distinct_labels
    # Invalid label -> flash err, still 302 (PRG) -- and nothing changes.
    r = client.post(f"/admin/captures/{cid}/reclassify", headers=_auth(),
                    data={"new_label": "not_a_class"})
    assert r.status_code == 302
    s = _open_store(cfgp)
    assert s.get(cid)["label"] == "deer"  # rejected -> unchanged
    s.close()
    # Valid relabel to an existing label -> row updated + marked reviewed.
    r = client.post(f"/admin/captures/{cid}/reclassify", headers=_auth(),
                    data={"new_label": "elk"})
    assert r.status_code == 302
    s = _open_store(cfgp)
    row = s.get(cid)
    assert row["label"] == "elk"
    assert row["reviewed"] == 1
    s.close()
    # GET on a POST-only route is rejected.
    assert client.get(f"/admin/captures/{cid}/reclassify", headers=_auth()).status_code == 405


def test_captures_page_has_bulk_ui(ctx) -> None:
    client, cfgp = ctx
    _seed_capture(cfgp)
    r = client.get("/admin/captures", headers=_auth())
    html = r.data
    assert b'id="bulk-status"' in html
    assert b'class="bulk-toolbar"' in html
    assert b'class="select-box"' in html   # per-card selection checkbox
    assert b'id="lightbox"' in html


def test_captures_bulk_route(ctx) -> None:
    client, cfgp = ctx
    cid = _seed_capture(cfgp)  # reclassified below, then deleted last
    hdr = {**_auth(), **_origin()}
    # Invalid reclassify label is rejected (allow-list enforced).
    assert client.post("/admin/captures/bulk", headers=hdr,
                       json={"action": "reclassify", "ids": [cid],
                             "label": "not_a_class"}).status_code == 400
    # Valid bulk reclassify updates the row ('deer' is in distinct_labels).
    r = client.post("/admin/captures/bulk", headers=hdr,
                    json={"action": "reclassify", "ids": [cid], "label": "deer"})
    assert r.status_code == 200 and r.get_json() == {"ok": True, "action": "reclassify", "affected": 1}
    s = _open_store(cfgp)
    assert s.get(cid)["label"] == "deer"
    s.close()
    # Bad action.
    assert client.post("/admin/captures/bulk", headers=hdr,
                       json={"action": "nope", "ids": [1]}).status_code == 400
    # Non-JSON body.
    assert client.post("/admin/captures/bulk", headers=hdr, data="x").status_code == 400
    # Non-int id.
    assert client.post("/admin/captures/bulk", headers=hdr,
                       json={"action": "delete", "ids": ["1; DROP"]}).status_code == 400
    # GET on a POST-only route is rejected.
    assert client.get("/admin/captures/bulk", headers=_auth()).status_code == 405
    # Cross-origin POST is blocked by the guard.
    assert client.post("/admin/captures/bulk", headers={**_auth(), "Origin": "http://evil.example"},
                       json={"action": "delete", "ids": [1]}).status_code == 403
    # Happy path (delete last so we don't remove a row still under test).
    r = client.post("/admin/captures/bulk", headers=hdr,
                    json={"action": "delete", "ids": [cid]})
    assert r.status_code == 200 and r.get_json() == {"ok": True, "action": "delete", "affected": 1}
    s = _open_store(cfgp)
    assert s.get(cid) is None
    s.close()
