"""Dashboard view over Autumn's unified model catalog (local models plus any
eligible provider-backed candidates), grouped by source."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, Tree

from autumn.models import CatalogEntry


def _summary(entries: list[CatalogEntry]) -> str:
    if not entries:
        return "No local models installed. Use `autumn models install` to add one."
    count = len(entries)
    noun = "model" if count == 1 else "models"
    default = next((entry.name for entry in entries if entry.is_default), "--")
    return f"{count} installed {noun} | Default: {default} | Press d to set highlighted default"


def _row(entry: CatalogEntry) -> str:
    marker = "*" if entry.is_default else " "
    context = f" | ctx {entry.context_window}" if entry.context_window is not None else ""
    return f"{marker} {entry.name} | {entry.backend}{context}"


def _grouped(entries: list[CatalogEntry]) -> list[tuple[str, list[CatalogEntry]]]:
    """Groups `entries` by `.group`, preserving first-seen group order (so
    "Local" -- always built first by `catalog.build_entries` -- renders
    above any provider groups)."""
    groups: dict[str, list[CatalogEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.group, []).append(entry)
    return list(groups.items())


class ModelCatalogView(Vertical):
    """Lists the unified model catalog as a tree grouped by source, and
    exposes the highlighted entry."""

    DEFAULT_CSS = """
    ModelCatalogView {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
    }
    ModelCatalogView #model-catalog-summary {
        margin-bottom: 1;
    }
    ModelCatalogView #model-tree {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self, entries: list[CatalogEntry], *args, **kwargs) -> None:
        self._entries = entries
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Static(_summary(self._entries), id="model-catalog-summary")
        tree: Tree[CatalogEntry] = Tree("models", id="model-tree")
        tree.show_root = False
        yield tree

    def on_mount(self) -> None:
        self.border_title = "Models"
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        tree = self.query_one("#model-tree", Tree)
        tree.clear()
        for group_name, group_entries in _grouped(self._entries):
            group_node = tree.root.add(group_name, expand=True)
            for entry in group_entries:
                group_node.add_leaf(_row(entry), data=entry)

    @property
    def selected_entry(self) -> CatalogEntry | None:
        tree = self.query_one("#model-tree", Tree)
        node = tree.cursor_node
        if node is None:
            return None
        return node.data

    def refresh_from_entries(self, entries: list[CatalogEntry]) -> None:
        self._entries = entries
        self.query_one("#model-catalog-summary", Static).update(_summary(entries))
        self._rebuild_tree()
