"""Shared test fixtures."""

import pytest

from autumn import local_models, paths


@pytest.fixture(autouse=True)
def _isolated_xdg_data_home(tmp_path, monkeypatch):
    """Points paths.py's XDG_DATA_HOME fallback at an isolated tmp directory
    for every test. Most tests pass `runs_root` explicitly to AutumnApp and
    never touch this, but `queue_sessions_root` defaults to
    `paths.sessions_root()` when omitted -- without this, any test that
    constructs `AutumnApp(...)` without that kwarg would read/write the real
    user's `~/.local/share/autumn/sessions`."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))


@pytest.fixture(autouse=True)
def _seed_local_model_catalog(_isolated_xdg_data_home, tmp_path):
    """Pre-installs one fake local model into the isolated XDG models root
    (see `_isolated_xdg_data_home` above). ModelPickerScreen only appears
    when the local model catalog is empty, so without this every test that
    constructs `AutumnApp(...)` without its own `model_catalog_root` would
    land on ModelPickerScreen instead of InputScreen the moment that screen
    shipped. Tests that specifically exercise the empty-catalog picker pass
    their own empty `model_catalog_root`, which bypasses this seed."""
    source = tmp_path / "conftest-seed-model.gguf"
    source.write_bytes(b"fake gguf")
    local_models.install_model(paths.models_root(), name="conftest-seed-model", source_path=source)
