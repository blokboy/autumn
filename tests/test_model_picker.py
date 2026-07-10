"""Behavioral tests for the first-run model picker/download flow."""

from pathlib import Path

from textual.widgets import Label, ListView

from autumn import local_models
from autumn.app import AutumnApp
from autumn.curated_models import CURATED_MODELS
from autumn.screens.dashboard_screen import DashboardScreen
from autumn.screens.input_screen import InputScreen
from autumn.screens.model_picker_screen import ModelPickerScreen


def _fake_download_file(url: str, destination: Path, on_progress) -> None:
    destination.write_bytes(b"fake gguf bytes")
    if on_progress is not None:
        on_progress(len(b"fake gguf bytes"), len(b"fake gguf bytes"))


async def test_empty_catalog_shows_model_picker_first(tmp_path):
    app = AutumnApp(runs_root=tmp_path, model_catalog_root=tmp_path / "models")

    async with app.run_test() as pilot:
        await pilot.pause()

        assert isinstance(app.screen, ModelPickerScreen)
        list_view = app.screen.query_one(ListView)
        assert len(list_view) == len(CURATED_MODELS)

        first = CURATED_MODELS[0]
        row_text = str(app.screen.query(Label)[1].content)
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
        await pilot.press("enter")
        await pilot.pause(0.2)

    relaunched = AutumnApp(runs_root=tmp_path, model_catalog_root=catalog_root)
    async with relaunched.run_test() as pilot:
        await pilot.pause()

        assert isinstance(relaunched.screen, InputScreen)
