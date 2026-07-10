"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_xdg_data_home(tmp_path, monkeypatch):
    """Points paths.py's XDG_DATA_HOME fallback at an isolated tmp directory
    for every test. Most tests pass `runs_root` explicitly to AutumnApp and
    never touch this, but `queue_sessions_root` defaults to
    `paths.sessions_root()` when omitted -- without this, any test that
    constructs `AutumnApp(...)` without that kwarg would read/write the real
    user's `~/.local/share/autumn/sessions`."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
