"""The `autumn` Textual application."""

import os
import threading
from pathlib import Path

from textual.app import App
from textual.theme import Theme

from autumn import palette, paths, queue_store, runner
from autumn.cli import LaunchSpec, LaunchSpecError, parse_command_line
from autumn.dashboard_callback import DashboardCallback
from autumn.fixtures import dry_run_events
from autumn.models import DashboardState, LiveRunSpec, RunStatus
from autumn.screens.confirm_screen import ConfirmScreen
from autumn.screens.dashboard_screen import DashboardScreen
from autumn.screens.help_screen import HelpScreen
from autumn.screens.input_screen import InputScreen

# Retints Textual's own built-in widget chrome (Input focus border, Button,
# scrollbars, DataTable cursor, etc.) to match styles/autumn.tcss's fall
# palette -- without this, anything not explicitly styled in autumn.tcss
# falls back to Textual's default blue accent, which clashes.
_AUTUMN_THEME = Theme(
    name="autumn",
    primary=palette.ACCENT_PRIMARY,
    secondary=palette.ACCENT_SECONDARY,
    warning=palette.WARNING,
    error=palette.ERROR,
    success=palette.SUCCESS,
    accent=palette.ACCENT_PRIMARY,
    foreground=palette.CHROME,
    background=palette.BACKGROUND,
    surface=palette.BACKGROUND,
    panel=palette.BACKGROUND,
    dark=True,
)

_QUIT_WHILE_RUNNING_MESSAGE = (
    "Quitting will terminate the in-progress run immediately. "
    "Press Q instead to stop cleanly."
)


def _resume_prompt_message(item_count: int, session_count: int) -> str:
    items = "item" if item_count == 1 else "items"
    sessions = "session" if session_count == 1 else "sessions"
    return f"Found {item_count} pending {items} from {session_count} previous {sessions}."


# How often to check whether this process's own live run has left RUNNING, so
# the next queued command bar item (if any) can auto-start. Coarser than a
# rendering concern -- reuses the same lightweight-timer-poll idiom
# DashboardScreen already uses for its own live-state/registry polling, since
# DashboardState is a plain dataclass mutated from a background thread, not a
# Textual reactive/message emitter.
_QUEUE_POLL_INTERVAL_SECONDS = 0.2


class AutumnApp(App):
    """Torlink-styled (fall-toned) dashboard: either launches (and live-tracks) one GEPA
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
        queue_sessions_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.register_theme(_AUTUMN_THEME)
        self.theme = "autumn"
        self.runs_root = runs_root
        self.run_name = run_name
        self.run_dir = run_dir
        self.script_path = script_path
        self.dry_run = dry_run

        # This process's own on-disk pending-queue session file (see
        # queue_store.py) -- one file per AutumnApp instance, so concurrent
        # `autumn` processes never write the same file. Persisted on every
        # queue mutation (see submit_command/_advance_queue), not just at
        # quit, so a crash or kill -9 never loses queued commands.
        self._queue_sessions_root = queue_sessions_root or paths.sessions_root()
        self._queue_session_path = queue_store.session_path(
            self._queue_sessions_root, queue_store.new_session_id()
        )

        # Launch mode iff both run_name and run_dir are given (cli.py's `run`
        # subcommand always supplies both together); otherwise this is
        # browse-only, with no live state and no background thread.
        if run_name is not None and run_dir is not None:
            self.state: DashboardState | None = DashboardState(run_name=run_name, run_dir=run_dir)
            self._dashboard_callback: DashboardCallback | None = DashboardCallback(self, self.state)
        else:
            self.state = None
            self._dashboard_callback = None

        # In-memory-only queue of commands submitted via CommandBar while a
        # run was already live (see submit_command below); each entry is
        # either a LaunchSpec (a `gepa ...` command) or a plain str (a stub
        # prompt, queued only so it auto-advances in the same order it was
        # submitted). `_queue_watch_state` is whichever DashboardState
        # `_poll_queue_advance` is currently watching for a RUNNING -> terminal
        # transition -- always `self.state` as of the last time a live run was
        # (re)started, so the poll never fires twice for the same run.
        self.pending_queue: list[LaunchSpec | str] = []
        self._queue_watch_state: DashboardState | None = self.state

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
            self._offer_resume_or_input_screen()

        self.set_interval(_QUEUE_POLL_INTERVAL_SECONDS, self._poll_queue_advance)

    def _offer_resume_or_input_screen(self) -> None:
        """Bare `autumn`'s landing decision (never reached by `autumn run
        <script.py>`, which always takes the launch-mode branch above): if
        other sessions left a non-empty queue behind (and are confirmed dead
        via `queue_store.discover_resumable`'s pid check -- a still-live
        session's file is never surfaced here), ask before InputScreen whether
        to resume it. No leftover sessions -> InputScreen exactly as before."""
        leftover = queue_store.discover_resumable(self._queue_sessions_root, self._queue_session_path)
        if not leftover:
            self.push_screen(InputScreen())
            return

        merged = queue_store.load_and_merge(leftover)
        message = _resume_prompt_message(len(merged), len(leftover))

        def on_result(confirmed: bool) -> None:
            queue_store.delete_files(leftover)
            if confirmed:
                self.pending_queue = merged
                self._persist_queue()
                # Not enter_browse_mode()'s switch_screen: by the time this
                # callback runs, ConfirmScreen has already popped itself back
                # off, leaving the app's implicit base screen on top -- which
                # (unlike InputScreen) was never itself pushed via
                # push_screen, so it has no result-callback slot for
                # switch_screen to pop. push_screen (as the on_mount
                # launch-mode branch above also does from this same base
                # screen) is the correct call here.
                self._dashboard_screen = DashboardScreen(self.runs_root)
                self.push_screen(self._dashboard_screen)
                # DashboardScreen mounts asynchronously -- _advance_queue's
                # _refresh_queue_panel needs its CommandBar already mounted
                # (via promote_to_live/query_one), so this must wait for that
                # mount to actually land rather than running inline.
                self.call_after_refresh(self._advance_queue)
            else:
                self.push_screen(InputScreen())

        self.push_screen(
            ConfirmScreen(message, title="Resume?", confirm_label="Resume", cancel_label="Start fresh"),
            on_result,
        )

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

    def _adopt_live_spec(self, spec: LaunchSpec) -> DashboardState:
        """Updates run_name/run_dir/script_path/dry_run and constructs a fresh
        DashboardState + DashboardCallback for `spec`, without deciding how the
        dashboard screen should reflect it -- callers differ: `launch_gepa_run`
        needs a DashboardScreen that doesn't exist yet, while CommandBar's
        immediate-launch and queue-auto-advance paths promote an already-mounted
        one. Also (re)points `_queue_watch_state` at the new state, so
        `_poll_queue_advance` watches whichever run is live now."""
        self.run_name = spec.run_name
        self.run_dir = spec.run_dir
        self.script_path = spec.script_path
        self.dry_run = spec.dry_run

        state = DashboardState(run_name=spec.run_name, run_dir=spec.run_dir)
        self.state = state
        self._dashboard_callback = DashboardCallback(self, state)
        self._queue_watch_state = state
        return state

    def launch_gepa_run(self, spec: LaunchSpec) -> None:
        """InputScreen's `gepa <script> ...` path: promotes this already-mounted,
        browse-only AutumnApp into a live run, identically to what launch-mode
        construction + `on_mount` do together for `autumn run <script>`."""
        state = self._adopt_live_spec(spec)
        self._dashboard_screen = DashboardScreen(self.runs_root, live_state=state)
        self.switch_screen(self._dashboard_screen)
        self._start_live_run()

    def _launch_spec_now(self, spec: LaunchSpec) -> None:
        """Launches `spec` against the already-mounted DashboardScreen (via
        `promote_to_live`, the same mechanism `action_resume` uses) rather than
        replacing it -- used by `submit_command`'s immediate-launch path and by
        `_advance_queue`, both of which always run with a DashboardScreen
        already on screen (CommandBar only exists there)."""
        state = self._adopt_live_spec(spec)
        self._dashboard_screen.promote_to_live(state)
        self._start_live_run()

    def _run_is_live(self) -> bool:
        return self.state is not None and self.state.status is RunStatus.RUNNING

    def _refresh_queue_panel(self) -> None:
        self._dashboard_screen.refresh_queue(self.pending_queue)

    def _persist_queue(self) -> None:
        """Rewrites this process's own queue session file to match
        `pending_queue` exactly (deleting it once the queue is empty) --
        called after every append and every pop so an on-disk copy is always
        current, not just at quit (see queue_store.py)."""
        queue_store.persist_queue(self._queue_session_path, self.pending_queue, pid=os.getpid())

    def submit_command(self, text: str) -> None:
        """Handles one line submitted via CommandBar (`:` on DashboardScreen).

        A `gepa <script> ...` command launches immediately if no run is live,
        or is appended to `pending_queue` if one is. A non-`gepa` prompt shows
        the same stub notice InputScreen shows when nothing is live, but is
        also queued (rather than shown immediately) when a run is live, so it
        auto-advances in the same submitted order once its turn comes up
        instead of jumping the queue. Malformed `gepa ...` syntax is reported
        immediately either way -- parsing happens at submit time, not launch
        time, so a bad command never even makes it into the queue.
        """
        text = text.strip()
        if not text:
            return

        try:
            spec = parse_command_line(text)
        except LaunchSpecError as exc:
            self.notify(str(exc), severity="error")
            return

        item: LaunchSpec | str = spec if spec is not None else text

        if self._run_is_live():
            self.pending_queue.append(item)
            self._persist_queue()
            self._refresh_queue_panel()
            return

        if spec is not None:
            self._launch_spec_now(spec)
        else:
            self.notify("Prompt execution not implemented yet", severity="warning")

    def _poll_queue_advance(self) -> None:
        state = self.state
        if state is None or state is not self._queue_watch_state:
            return
        if state.status is RunStatus.RUNNING:
            return
        # Stop watching this now-finished run so this fires exactly once per
        # run, then work through the queue -- stub entries just show their
        # notice and fall through to the next item, a gepa entry launches and
        # returns (it re-points _queue_watch_state at the new run itself).
        self._queue_watch_state = None
        self._advance_queue()

    def _advance_queue(self) -> None:
        while self.pending_queue:
            item = self.pending_queue.pop(0)
            self._persist_queue()
            self._refresh_queue_panel()
            if isinstance(item, str):
                self.notify("Prompt execution not implemented yet", severity="warning")
                continue
            self._launch_spec_now(item)
            return
        self._refresh_queue_panel()

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
        self._queue_watch_state = state
        runner.launch(callback, spec)
        screen.promote_to_live(state)
