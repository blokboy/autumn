"""First-run landing screen: shown instead of InputScreen when Autumn's
local model catalog is empty, offering a curated list of small GGUF models
to download and install as the default local model.

Skipping (Escape) lands on InputScreen exactly as bare `autumn` always has,
just without a local model configured yet -- since the trigger is "the
catalog is empty," this screen naturally reappears next launch until either
this picker or `autumn models install` puts something in the catalog."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Label, ListItem, ListView

from autumn.curated_models import CURATED_MODELS, CuratedModel
from autumn.screens.input_screen import InputScreen
from autumn.screens.model_download_screen import ModelDownloadScreen


def _row_text(entry: CuratedModel) -> str:
    return f"{entry.name} ({entry.vendor}) -- ~{entry.approx_size_gb:.1f}GB"


class ModelPickerScreen(Screen):
    """Pick one of `CURATED_MODELS` to download, or skip to InputScreen."""

    BINDINGS = [("escape", "skip", "Skip")]

    DEFAULT_CSS = """
    ModelPickerScreen {
        align: center middle;
    }
    ModelPickerScreen #model-picker-frame {
        width: auto;
        height: auto;
        align: center middle;
    }
    ModelPickerScreen .picker-title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        margin-bottom: 1;
    }
    ModelPickerScreen .picker-hint {
        width: 100%;
        content-align: center middle;
        text-opacity: 70%;
        margin-top: 1;
    }
    ModelPickerScreen ListView {
        width: 64;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker-frame"):
            yield Label("Pick a model to get started", classes="picker-title")
            yield ListView(
                *[ListItem(Label(_row_text(entry))) for entry in CURATED_MODELS],
                id="model-picker-list",
            )
            yield Label("enter to download   •   escape to skip", classes="picker-hint")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        list_view = self.query_one("#model-picker-list", ListView)
        if list_view.index is None:
            return
        entry = CURATED_MODELS[list_view.index]
        self.app.switch_screen(ModelDownloadScreen(entry))

    def action_skip(self) -> None:
        self.app.switch_screen(InputScreen())
