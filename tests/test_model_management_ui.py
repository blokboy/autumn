"""Behavioral tests for dashboard local-model management."""

from textual.widgets import ListView, Static, TabbedContent

from autumn import local_models
from autumn.app import AutumnApp
from autumn.widgets.model_catalog_view import ModelCatalogView


async def test_models_tab_lists_installed_models_and_sets_default(tmp_path):
    first = tmp_path / "first.gguf"
    first.write_bytes(b"first")
    second = tmp_path / "second.gguf"
    second.write_bytes(b"second")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="first", source_path=first)
    local_models.install_model(catalog_root, name="second", source_path=second)
    app = AutumnApp(runs_root=tmp_path, model_catalog_root=catalog_root)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one(TabbedContent).active = "models-tab"
        await pilot.pause()

        view = app.screen.query_one("#models", ModelCatalogView)
        summary = view.query_one("#model-catalog-summary", Static).content
        assert "2 installed models" in str(summary)
        assert "Default: first" in str(summary)

        model_list = view.query_one("#model-list", ListView)
        model_list.index = 1
        await pilot.press("d")
        await pilot.pause()

        assert local_models.get_default(catalog_root).name == "second"
        summary = view.query_one("#model-catalog-summary", Static).content
        assert "Default: second" in str(summary)
