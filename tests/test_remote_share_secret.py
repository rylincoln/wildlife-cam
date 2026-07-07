"""Tests for set_remote_secret and the wildlife-share-secret CLI."""

from __future__ import annotations

from pathlib import Path

from werkzeug.security import check_password_hash

from wildlife.admin import config_io as cio
from wildlife.config import load_config
from wildlife.remote import share_secret
from tests.test_admin_config_io import render_base


def test_set_remote_secret_persists_and_enables(tmp_path: Path) -> None:
    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(render_base(tmp_path), "utf-8")

    cio.set_remote_secret(str(cfgp), "pbkdf2:sha256:deadbeef")

    cfg = load_config(cfgp)
    assert cfg.remote.enabled is True
    assert cfg.remote.share_secret_hash == "pbkdf2:sha256:deadbeef"


def test_cli_mints_verifiable_secret_and_prints_link(tmp_path, monkeypatch, capsys) -> None:
    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(render_base(tmp_path), "utf-8")
    # Give it a base_url so the printed link is concrete.
    cio.update_section(str(cfgp), "remote", {"base_url": "https://cam.example.com"})

    monkeypatch.setattr("sys.argv", ["wildlife-share-secret", str(cfgp)])
    assert share_secret.main() == 0

    out = capsys.readouterr().out
    # Extract the printed raw secret and confirm it verifies against the stored hash.
    line = next(ln for ln in out.splitlines() if "Raw secret:" in ln)
    secret = line.split("Raw secret:")[1].strip()
    assert len(secret) >= 40  # token_urlsafe(32) is ~43 chars
    cfg = load_config(cfgp)
    assert check_password_hash(cfg.remote.share_secret_hash, secret)
    assert f"https://cam.example.com/?key={secret}" in out


def test_cli_rotation_invalidates_old(tmp_path, monkeypatch) -> None:
    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(render_base(tmp_path), "utf-8")
    monkeypatch.setattr("sys.argv", ["wildlife-share-secret", str(cfgp)])

    share_secret.main()
    first = load_config(cfgp).remote.share_secret_hash
    share_secret.main()
    second = load_config(cfgp).remote.share_secret_hash
    assert first != second  # a fresh secret each run
