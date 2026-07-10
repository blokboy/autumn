"""Main dashboard screen: sidebar of runs + tabbed detail view for the selected run.

Owns the merged run registry and decides, per sidebar selection, which
`DashboardState` the three tab widgets should render. The Overview/Candidates/Log
widgets never know whether that state is live or historical -- they only ever see
`refresh_from_state(state)` calls; live-vs-historical is entirely a concern of
this screen.
"""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import ListView, TabbedContent, TabPane

from autumn import registry
from autumn.models import DashboardState, RunStatus
from autumn.widgets.candidates_table import CandidatesTable
from autumn.widgets.log_view import LogView
from autumn.widgets.overview_pane import OverviewPane
from autumn.widgets.run_sidebar import RunListItem, RunSidebar

# How often to poll the live DashboardState.version for changes made off-screen
# (e.g. by DashboardCallback via app.call_from_thread). DashboardState is a plain
# dataclass, not a Textual reactive/message-emitting object, so a lightweight
# timer poll is the simplest, most idiomatic way to notice mutations pushed in
# from a background thread without threading a reactive/message all the way
# through DashboardCallback.
_POLL_INTERVAL_SECONDS = 0.1

# How often to re-scan the runs root from disk. Much coarser than the live poll
# above: this is a filesystem walk (registry.scan), not an in-memory version
# check, and only needs to be fresh enough to notice runs appearing/finishing --
# not fast enough to feel "live".
_REGISTRY_POLL_INTERVAL_SECONDS = 2.0


def _empty_state(runs_root: Path) -> DashboardState:
    """Placeholder shown when the runs root has nothing in it yet (fresh browse
    mode)."""
    return DashboardState(run_name="(no runs found)", run_dir=runs_root, status=RunStatus.UNKNOWN)


class DashboardScreen(Screen):
    """Hosts the run sidebar and the Overview/Candidates/Log tabs for whichever
    run is currently selected."""

    DEFAULT_CSS = """
    DashboardScreen > Horizontal {
        height: 1fr;
    }
    DashboardScreen TabbedContent {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(
        self,
        runs_root: Path,
        live_state: DashboardState | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._runs_root = Path(runs_root)
        self._live_state = live_state
        self._live_run_dir = live_state.run_dir if live_state is not None else None
        self._summaries = registry.merge_live(registry.scan(self._runs_root), live_state)
        self._last_seen_live_version = live_state.version if live_state is not None else -1

        initial = self._summaries[0] if self._summaries else None
        self._selected_run_dir = initial.run_dir if initial is not None else None
        self._displayed_state = (
            self._state_for(initial.run_dir) if initial is not None else _empty_state(self._runs_root)
        )

    def _state_for(self, run_dir: Path) -> DashboardState:
        if run_dir == self._live_run_dir:
            return self._live_state
        return registry.load_dashboard_state(run_dir)

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield RunSidebar(self._summaries, live_state=self._live_state, id="sidebar")
            with TabbedContent():
                yield TabPane("Overview", OverviewPane(self._displayed_state, id="overview"))
                yield TabPane("Candidates", CandidatesTable(self._displayed_state, id="candidates"))
                yield TabPane("Log", LogView(self._displayed_state, id="log"))

    def on_mount(self) -> None:
        self.set_interval(_POLL_INTERVAL_SECONDS, self._poll_live_state)
        self.set_interval(_REGISTRY_POLL_INTERVAL_SECONDS, self._rescan_registry)

    def on_list_view_highlighted(self, message: ListView.Highlighted) -> None:
        item = message.item
        if not isinstance(item, RunListItem):
            return
        self._select_run(item.run_dir)

    def _select_run(self, run_dir: Path) -> None:
        if run_dir == self._selected_run_dir:
            return
        self._selected_run_dir = run_dir
        self._displayed_state = self._state_for(run_dir)
        self._refresh_tabs(self._displayed_state)

    @property
    def selected_run_dir(self) -> Path | None:
        """The run_dir currently highlighted in the sidebar (live or historical),
        or None if the sidebar is empty. Used by AutumnApp's `r` (resume) action
        to know which run to act on."""
        return self._selected_run_dir

    def selected_run_status(self) -> RunStatus | None:
        """Status of the currently-selected run: the live in-memory status if
        it's this process's own live run (always fresher than a disk scan),
        otherwise inferred from disk. None if nothing is selected."""
        if self._selected_run_dir is None:
            return None
        if self._selected_run_dir == self._live_run_dir and self._live_state is not None:
            return self._live_state.status
        return registry.infer_status(self._selected_run_dir)

    def promote_to_live(self, state: DashboardState) -> None:
        """Adopts `state` as this screen's live run, called by AutumnApp right
        after it resumes a historical run via `r`. Pins the run first in the
        sidebar (via the next registry rescan, reusing the same merge_live path
        as any other live run) and immediately switches the displayed tabs to
        follow it, since resuming is presumed to mean "and now watch it"."""
        self._live_state = state
        self._live_run_dir = state.run_dir
        self._last_seen_live_version = -1
        self._selected_run_dir = state.run_dir
        self._displayed_state = state
        self._refresh_tabs(state)
        self.run_worker(self._rescan_registry())

    def _refresh_tabs(self, state: DashboardState) -> None:
        self.query_one("#overview", OverviewPane).refresh_from_state(state)
        self.query_one("#candidates", CandidatesTable).refresh_from_state(state)
        self.query_one("#log", LogView).refresh_from_state(state)

    def _poll_live_state(self) -> None:
        state = self._live_state
        if state is None or state.version == self._last_seen_live_version:
            return
        self._last_seen_live_version = state.version
        self.query_one(RunSidebar).refresh_from_state(state)
        if self._selected_run_dir == self._live_run_dir:
            self._refresh_tabs(state)

    async def _rescan_registry(self) -> None:
        self._summaries = registry.merge_live(registry.scan(self._runs_root), self._live_state)
        await self.query_one(RunSidebar).update_runs(self._summaries)
