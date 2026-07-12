"""Sidebar listing every known run (live + historical), status-styled via the
`status-{value}` CSS classes."""

from pathlib import Path

from textual.binding import Binding
from textual.widgets import Label, ListItem, ListView

from models import DashboardState, RunStatus, RunSummary

_STATUS_CLASSES = tuple(f"status-{status.value}" for status in RunStatus)


def _format_label(name: str, status: RunStatus, is_live: bool) -> str:
    marker = "● " if is_live else "  "
    return f"{marker}{name}  [{status.value}]"


class RunListItem(ListItem):
    """A single run entry in the sidebar, styled by the run's status."""

    def __init__(self, summary: RunSummary) -> None:
        self.run_dir = summary.run_dir
        self._label = Label(_format_label(summary.name, summary.status, summary.is_live))
        super().__init__(self._label)
        self._apply_status_class(summary.status)

    def _apply_status_class(self, status: RunStatus) -> None:
        for css_class in _STATUS_CLASSES:
            self.remove_class(css_class)
        self.add_class(f"status-{status.value}")

    def refresh_from_summary(self, summary: RunSummary) -> None:
        self._label.update(_format_label(summary.name, summary.status, summary.is_live))
        self._apply_status_class(summary.status)

    def refresh_from_live_state(self, state: DashboardState) -> None:
        self._label.update(_format_label(state.run_name, state.status, True))
        self._apply_status_class(state.status)


class RunSidebar(ListView):
    """Lists every known run. Selection is surfaced to `DashboardScreen` via
    ListView's own `Highlighted` message rather than a bespoke message class --
    arrow/j/k already move the cursor per the existing bindings, so treating
    every cursor move as "select this run to preview" (rather than requiring a
    separate Enter-to-select step) is the natural fit here.
    """

    DEFAULT_CSS = """
    RunSidebar {
        width: 32;
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("j", "cursor_down", "Cursor down", show=False),
        Binding("k", "cursor_up", "Cursor up", show=False),
    ]

    def __init__(
        self,
        summaries: list[RunSummary],
        live_state: DashboardState | None = None,
        *args,
        **kwargs,
    ) -> None:
        self._items_by_run_dir: dict[Path, RunListItem] = {}
        items = [self._make_item(summary) for summary in summaries]
        super().__init__(*items, *args, **kwargs)
        if live_state is not None:
            self.refresh_from_state(live_state)

    def _make_item(self, summary: RunSummary) -> RunListItem:
        item = RunListItem(summary)
        self._items_by_run_dir[summary.run_dir] = item
        return item

    def on_mount(self) -> None:
        self.border_title = "Jobs"

    @property
    def selected_run_dir(self) -> Path | None:
        item = self.highlighted_child
        return item.run_dir if isinstance(item, RunListItem) else None

    async def update_runs(self, summaries: list[RunSummary]) -> None:
        """Reconciles the sidebar with a fresh `registry.scan()` + `merge_live()`
        result. The common case -- the same run_dirs in the same order, nothing
        appeared or finished since the last scan -- updates labels in place so
        cursor position is never disturbed. Anything else (a new run appeared,
        one dropped off, or the live-first/newest-first order shifted) rebuilds
        the list and restores the previous selection by run_dir if it still
        exists. `clear()`/`extend()` are both async (their widget removal/mount
        is message-queued, not immediate), so this must await the removal
        before mounting replacements -- otherwise the stale and fresh ListItems
        briefly coexist and `self.index` ends up pointing at the wrong row.
        """
        new_order = [summary.run_dir for summary in summaries]
        if new_order == list(self._items_by_run_dir.keys()):
            for summary in summaries:
                self._items_by_run_dir[summary.run_dir].refresh_from_summary(summary)
            return

        previously_selected = self.selected_run_dir
        await self.clear()
        self._items_by_run_dir = {}
        items = [self._make_item(summary) for summary in summaries]
        await self.extend(items)
        if previously_selected is not None and previously_selected in self._items_by_run_dir:
            self.index = new_order.index(previously_selected)

    def refresh_from_state(self, state: DashboardState) -> None:
        """Updates the live run's row in place as its `DashboardState.version`
        bumps, without rebuilding the list or disturbing cursor position."""
        item = self._items_by_run_dir.get(state.run_dir)
        if item is not None:
            item.refresh_from_live_state(state)
