"""Dashboard view over Autumn's installed local model catalog."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, ListItem, ListView, Static

from autumn.models import LocalModel


def _summary(models: list[LocalModel]) -> str:
    count = len(models)
    noun = "model" if count == 1 else "models"
    default = next((model.name for model in models if model.is_default), "--")
    return f"{count} installed {noun} | Default: {default} | Press d to set highlighted default"


def _row(model: LocalModel) -> str:
    marker = "*" if model.is_default else " "
    context = f" | ctx {model.context_window}" if model.context_window is not None else ""
    return f"{marker} {model.name} | {model.backend}{context}"


class ModelCatalogView(Vertical):
    """Lists installed local models and exposes the highlighted model name."""

    DEFAULT_CSS = """
    ModelCatalogView {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
    }
    ModelCatalogView #model-catalog-summary {
        margin-bottom: 1;
    }
    """

    def __init__(self, models: list[LocalModel], *args, **kwargs) -> None:
        self._models = models
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Static(_summary(self._models), id="model-catalog-summary")
        yield ListView(*[ListItem(Label(_row(model))) for model in self._models], id="model-list")

    def on_mount(self) -> None:
        self.border_title = "Models"

    @property
    def selected_model_name(self) -> str | None:
        model_list = self.query_one("#model-list", ListView)
        if model_list.index is None or model_list.index >= len(self._models):
            return None
        return self._models[model_list.index].name

    def refresh_from_models(self, models: list[LocalModel]) -> None:
        self._models = models
        self.query_one("#model-catalog-summary", Static).update(_summary(models))
        model_list = self.query_one("#model-list", ListView)
        index = model_list.index or 0
        model_list.clear()
        for model in models:
            model_list.append(ListItem(Label(_row(model))))
        model_list.index = min(index, len(models) - 1) if models else None
