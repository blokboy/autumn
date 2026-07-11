"""Behavioral tests for the first-run model picker/download flow."""

import dataclasses
import hashlib
from pathlib import Path

import pytest
from textual.widgets import SelectionList

import local_models
import screens.model_picker_screen as model_picker_screen
from app import AutumnApp
from curated_models import CURATED_MODELS
from screens.dashboard_screen import DashboardScreen
from screens.input_screen import InputScreen
from screens.model_picker_screen import ModelPickerScreen

_FAKE_CONTENT = b"fake gguf bytes"
_FAKE_CONTENT_SHA256 = hashlib.sha256(_FAKE_CONTENT).hexdigest()


def _fake_download_file(url: str, destination: Path, on_progress) -> None:
    destination.write_bytes(_FAKE_CONTENT)
    if on_progress is not None:
        on_progress(len(_FAKE_CONTENT), len(_FAKE_CONTENT))


@pytest.fixture(autouse=True)
def _patch_curated_models_checksum(monkeypatch):
    """`_fake_download_file` above always writes the same fixed payload
    regardless of which curated model was picked -- these tests exercise
    the picker/download-screen flow, not real HF downloads or checksum
    verification itself (see test_model_downloader.py for that). Point
    each curated model's pinned checksum at that fixed payload's checksum
    so `download_and_install`'s real checksum verification doesn't reject
    it here."""
    patched = [dataclasses.replace(entry, sha256=_FAKE_CONTENT_SHA256) for entry in CURATED_MODELS]
    monkeypatch.setattr(model_picker_screen, "CURATED_MODELS", patched)


async def test_empty_catalog_shows_model_picker_first(tmp_path):
    app = AutumnApp(runs_root=tmp_path, model_catalog_root=tmp_path / "models")

    async with app.run_test() as pilot:
        await pilot.pause()

        assert isinstance(app.screen, ModelPickerScreen)
        selection_list = app.screen.query_one(SelectionList)
        assert selection_list.option_count == len(CURATED_MODELS)

        first = CURATED_MODELS[0]
        row_text = str(selection_list.get_option_at_index(0).prompt)
        assert first.name in row_text
        assert first.vendor in row_text
        assert "GB" in row_text


async def test_nonempty_catalog_skips_picker_and_shows_input_screen(tmp_path):
    catalog_root = tmp_path / "models"
    source = tmp_path / "existing.gguf"
    source.write_bytes(b"existing")
    local_models.install_model(catalog_root, name="existing", source_path=source)

    app = AutumnApp(runs_root=tmp_path, model_catalog_root=catalog_root)

    async with app.run_test() as pilot:
        await pilot.pause()

        assert isinstance(app.screen, InputScreen)


async def test_skipping_picker_lands_on_input_screen_without_installing(tmp_path):
    catalog_root = tmp_path / "models"
    app = AutumnApp(runs_root=tmp_path, model_catalog_root=catalog_root)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        assert isinstance(app.screen, InputScreen)
        assert local_models.list_models(catalog_root) == []


async def test_skipped_picker_reappears_on_next_launch(tmp_path):
    catalog_root = tmp_path / "models"

    app = AutumnApp(runs_root=tmp_path, model_catalog_root=catalog_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    # A fresh AutumnApp instance against the same (still-empty) catalog root
    # simulates the next `autumn` launch.
    relaunched = AutumnApp(runs_root=tmp_path, model_catalog_root=catalog_root)
    async with relaunched.run_test() as pilot:
        await pilot.pause()

        assert isinstance(relaunched.screen, ModelPickerScreen)


async def test_enter_with_nothing_selected_skips_like_escape(tmp_path):
    catalog_root = tmp_path / "models"
    app = AutumnApp(runs_root=tmp_path, model_catalog_root=catalog_root)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ModelPickerScreen)

        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, InputScreen)
        assert local_models.list_models(catalog_root) == []


async def test_picking_a_model_downloads_installs_and_enters_dashboard(tmp_path):
    catalog_root = tmp_path / "models"
    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=catalog_root,
        model_download_fn=_fake_download_file,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ModelPickerScreen)

        await pilot.press("space")
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert isinstance(app.screen, DashboardScreen)
        installed = local_models.get_default(catalog_root)
        assert installed is not None
        assert installed.name == CURATED_MODELS[0].name
        assert installed.path.read_bytes() == b"fake gguf bytes"


async def test_installed_picker_model_never_reappears(tmp_path):
    catalog_root = tmp_path / "models"
    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=catalog_root,
        model_download_fn=_fake_download_file,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")
        await pilot.press("enter")
        await pilot.pause(0.2)

    relaunched = AutumnApp(runs_root=tmp_path, model_catalog_root=catalog_root)
    async with relaunched.run_test() as pilot:
        await pilot.pause()

        assert isinstance(relaunched.screen, InputScreen)


async def test_picking_multiple_models_downloads_installs_all_and_focuses_models_tab(tmp_path):
    catalog_root = tmp_path / "models"
    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=catalog_root,
        model_download_fn=_fake_download_file,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ModelPickerScreen)

        # Check the first two rows (space toggles, down moves the highlight).
        await pilot.press("space")
        await pilot.press("down")
        await pilot.press("space")
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert isinstance(app.screen, DashboardScreen)
        installed_names = {model.name for model in local_models.list_models(catalog_root)}
        assert installed_names == {CURATED_MODELS[0].name, CURATED_MODELS[1].name}

        # First selected becomes the default (install_model's empty-catalog rule).
        default = local_models.get_default(catalog_root)
        assert default is not None
        assert default.name == CURATED_MODELS[0].name

        # Ambiguous default (two freshly-installed models) -> land on Models tab.
        from textual.widgets import TabbedContent

        assert app.screen.query_one(TabbedContent).active == "models-tab"
