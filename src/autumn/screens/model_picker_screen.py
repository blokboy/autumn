"""First-run landing screen: shown instead of InputScreen when Autumn's
local model catalog is empty, offering a curated list of small GGUF models
to download and install.

Skipping (Escape) lands on InputScreen exactly as bare `autumn` always has,
just without a local model configured yet -- since the trigger is "the
catalog is empty," this screen naturally reappears next launch until either
this picker or `autumn models install` puts something in the catalog."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Label, SelectionList

from autumn.curated_models import CURATED_MODELS, CuratedModel
from autumn.screens.input_screen import InputScreen
from autumn.screens.model_download_screen import ModelDownloadScreen


def _row_text(entry: CuratedModel) -> str:
    return f"{entry.name} ({entry.vendor}) -- ~{entry.approx_size_gb:.1f}GB"


class ModelPickerScreen(Screen):
    """Pick one or more of `CURATED_MODELS` to download, or skip to
    InputScreen. Space toggles a row; Enter downloads whatever's checked, or
    skips (same as Escape) if nothing is checked."""

    BINDINGS = [
        Binding("escape", "skip", "Skip"),
        Binding("enter", "confirm", "Download selected", priority=True),
    ]

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
    ModelPickerScreen SelectionList {
        width: 64;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker-frame"):
            yield Label("Pick one or more models to get started", classes="picker-title")
            yield SelectionList(
                *[(_row_text(entry), index) for index, entry in enumerate(CURATED_MODELS)],
                id="model-picker-list",
            )
            yield Label(
                "space to toggle   •   enter to download   •   escape to skip",
                classes="picker-hint",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(SelectionList).focus()

    def action_confirm(self) -> None:
        selection_list = self.query_one("#model-picker-list", SelectionList)
        selected = [CURATED_MODELS[index] for index in selection_list.selected]
        if not selected:
            self.action_skip()
            return
        self.app.switch_screen(ModelDownloadScreen(selected))

    def action_skip(self) -> None:
        self.app.switch_screen(InputScreen())
