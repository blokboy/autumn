"""The `autumn` Textual application."""

import threading
from pathlib import Path

from textual.app import App

from autumn import runner
from autumn.cli import LaunchSpec
from autumn.dashboard_callback import DashboardCallback
from autumn.fixtures import dry_run_events
from autumn.models import DashboardState, LiveRunSpec, RunStatus
from autumn.screens.confirm_screen import ConfirmScreen
from autumn.screens.dashboard_screen import DashboardScreen
from autumn.screens.help_screen import HelpScreen
from autumn.screens.input_screen import InputScreen

_QUIT_WHILE_RUNNING_MESSAGE = (
    "Quitting will terminate the in-progress run immediately. "
    "Press Q instead to stop cleanly."
)


class AutumnApp(App):
    """Torlink-styled dashboard: either launches (and live-tracks) one GEPA
    optimization run, or -- with no run_name/run_dir given -- just browses the
    run registry under runs_root with nothing pinned live."""

    CSS_PATH = "styles/autumn.tcss"
    TITLE = "autumn"

    BINDINGS = [
        ("q", "request_quit", "Quit"),
        ("Q", "stop_run", "Stop"),
        ("r", "resume", "Resume"),
        ("question_mark", "toggle_help", "Help"),
    ]

    def __init__(
        self,
        runs_root: Path,
        run_name: str | None = None,
        run_dir: Path | None = None,
        script_path: Path | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.runs_root = runs_root
        self.run_name = run_name
        self.run_dir = run_dir
        self.script_path = script_path
        self.dry_run = dry_run

        # Launch mode iff both run_name and run_dir are given (cli.py's `run`
        # subcommand always supplies both together); otherwise this is
        # browse-only, with no live state and no background thread.
        if run_name is not None and run_dir is not None:
            self.state: DashboardState | None = DashboardState(run_name=run_name, run_dir=run_dir)
            self._dashboard_callback: DashboardCallback | None = DashboardCallback(self, self.state)
        else:
            self.state = None
            self._dashboard_callback = None

    def on_mount(self) -> None:
        # Launch mode (both run_name/run_dir given, cli.py's `run` subcommand)
        # goes straight to the dashboard, unaffected by InputScreen below.
        # Browse-only construction (cli.py's `_browse`, i.e. bare `autumn`)
        # lands on InputScreen first instead of jumping straight into browse
        # mode -- InputScreen itself decides whether to fall through to browse
        # (empty Enter) or promote into a live run (`gepa ...`, see
        # `launch_gepa_run` below).
        if self.run_name is not None and self.run_dir is not None:
            # Kept as a direct reference (rather than looked up later via
            # query_one) because DashboardScreen sits at the *bottom* of the
            # screen stack -- query_one/query only search the currently active
            # screen, which by the time `r` is pressed could be a ConfirmScreen
            # modal pushed on top of it.
            self._dashboard_screen = DashboardScreen(self.runs_root, live_state=self.state)
            self.push_screen(self._dashboard_screen)
            self._start_live_run()
        else:
            self.push_screen(InputScreen())

    def _start_live_run(self) -> None:
        """Kicks off this process's live run on a background thread (dry-run
        replay or a real `runner.launch`) against whatever `_dashboard_callback`
        currently is. Shared by `on_mount`'s launch-mode construction and
        `launch_gepa_run` (InputScreen's `gepa ...` path), so the two ways of
        starting a live run can't drift apart."""
        if self._dashboard_callback is None:
            return
        if self.dry_run:
            thread = threading.Thread(
                target=dry_run_events.replay,
                args=(self._dashboard_callback,),
                daemon=True,
            )
            thread.start()
        elif self.script_path is not None:
            spec = LiveRunSpec(
                script_path=self.script_path, run_dir=self.run_dir, run_name=self.run_name
            )
            runner.launch(self._dashboard_callback, spec)

    def enter_browse_mode(self) -> None:
        """InputScreen's empty-Enter path: the same bare-browse DashboardScreen
        `_browse()`'s launch-mode-free `AutumnApp` construction produces (no
        live_state), swapped in for InputScreen with `switch_screen` rather
        than pushed on top of it -- there's nothing to go "back" to."""
        self._dashboard_screen = DashboardScreen(self.runs_root)
        self.switch_screen(self._dashboard_screen)

    def launch_gepa_run(self, spec: LaunchSpec) -> None:
        """InputScreen's `gepa <script> ...` path: promotes this already-mounted,
        browse-only AutumnApp into a live run, identically to what launch-mode
        construction + `on_mount` do together for `autumn run <script>`."""
        self.run_name = spec.run_name
        self.run_dir = spec.run_dir
        self.script_path = spec.script_path
        self.dry_run = spec.dry_run

        state = DashboardState(run_name=spec.run_name, run_dir=spec.run_dir)
        self.state = state
        self._dashboard_callback = DashboardCallback(self, state)

        self._dashboard_screen = DashboardScreen(self.runs_root, live_state=state)
        self.switch_screen(self._dashboard_screen)
        self._start_live_run()

    def action_toggle_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_request_quit(self) -> None:
        """`q`: quits immediately unless this process's own live run is still
        RUNNING, in which case quitting would kill it mid-optimization (GEPA
        runs on a background thread of this same process -- there is no way to
        detach), so confirm first. `Q` is the clean alternative, mentioned in
        the confirm message itself."""
        if self.state is not None and self.state.status is RunStatus.RUNNING:
            def on_result(confirmed: bool) -> None:
                if confirmed:
                    self.exit()

            self.push_screen(
                ConfirmScreen(_QUIT_WHILE_RUNNING_MESSAGE, title="Quit?", confirm_label="Quit anyway"),
                on_result,
            )
        else:
            self.exit()

    def action_stop_run(self) -> None:
        """`Q`: confirm, then touch `<run_dir>/gepa.stop` -- GEPA's own
        built-in FileStopper watches this file and lets the current iteration
        finish before the engine (and this run's background thread) exits on
        its own. No-op if this process has no RUNNING live run."""
        if self.state is None or self.state.status is not RunStatus.RUNNING:
            return
        state = self.state

        def on_result(confirmed: bool) -> None:
            if confirmed:
                # In --dry-run mode nothing has created run_dir on disk (the
                # replay never touches the filesystem) -- a real `autumn run`
                # always has it already (runner.launch's mkdir happens before
                # the thread starts), but this guards the dry-run path too.
                state.run_dir.mkdir(parents=True, exist_ok=True)
                (state.run_dir / "gepa.stop").touch()
                state.append_log("warn", "graceful stop requested")

        self.push_screen(
            ConfirmScreen("Send graceful stop signal to this run?", confirm_label="Stop"),
            on_result,
        )

    def action_resume(self) -> None:
        """`r`: on a STOPPED/FAILED historical run selected in the sidebar,
        re-launches it against the same run_dir, recovering the original
        script path from autumn_meta.json. GEPA's own GEPAState.load(run_dir)
        resumption then kicks in automatically inside the user's script --
        autumn only has to get the script running again, not replay state
        itself."""
        screen = self._dashboard_screen
        run_dir = screen.selected_run_dir
        if run_dir is None:
            return
        if self.state is not None and self.state.status is RunStatus.RUNNING:
            self.notify("A run is already active in this session.", severity="warning")
            return
        if screen.selected_run_status() not in (RunStatus.STOPPED, RunStatus.FAILED):
            return

        spec = runner.recover_launch_spec(run_dir)
        if spec is None:
            self.notify(f"Can't resume {run_dir.name}: no autumn_meta.json to recover it from.", severity="error")
            return

        state = DashboardState(run_name=spec.run_name, run_dir=spec.run_dir)
        callback = DashboardCallback(self, state)
        self.state = state
        self._dashboard_callback = callback
        runner.launch(callback, spec)
        screen.promote_to_live(state)
