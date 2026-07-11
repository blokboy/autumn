"""The `autumn` Textual application."""

import os
import threading
import time
from pathlib import Path
from typing import Callable

from textual.app import App
from textual.theme import Theme

import catalog
import chat_store
import local_llm
import local_models
import model_downloader
import model_router
import palette
import paths
import queue_store
import runner
from cli import LaunchSpec, LaunchSpecError, parse_command_line
from dashboard_callback import DashboardCallback
from fixtures import dry_run_events
from groq_runner import GroqRunner, GroqRuntimeError
from local_model_runner import LocalModelRunner, LocalModelRuntimeError
from models import (
    ChatMessage,
    DashboardState,
    LiveRunSpec,
    LocalModel,
    ModelChoice,
    PromptRoutingPolicy,
    RunStatus,
    ToolCitation,
)
from screens.confirm_screen import ConfirmScreen
from screens.dashboard_screen import DashboardScreen
from screens.help_screen import HelpScreen
from screens.input_screen import InputScreen
from screens.model_picker_screen import ModelPickerScreen

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


def _resume_prompt_message(
    item_count: int,
    session_count: int,
    chat_message_count: int = 0,
    chat_session_count: int = 0,
) -> str:
    items = "item" if item_count == 1 else "items"
    sessions = "session" if session_count == 1 else "sessions"
    parts = []
    if item_count:
        parts.append(f"{item_count} pending {items} from {session_count} previous {sessions}")
    if chat_message_count:
        chat_messages = "chat message" if chat_message_count == 1 else "chat messages"
        chat_sessions = "session" if chat_session_count == 1 else "sessions"
        parts.append(
            f"{chat_message_count} pending {chat_messages} from {chat_session_count} previous {chat_sessions}"
        )
    return "Found " + " and ".join(parts) + "."


def _model_status_from_choice(choice: ModelChoice) -> str:
    if choice.backend == "builtin":
        return f"Fallback: {choice.name} ({choice.reason})"
    return f"Model: {choice.name} ({choice.reason})"


def _message_looks_like_error_response(message: ChatMessage) -> bool:
    text = message.text.strip().lower()
    return text.startswith(("error:", "error ", "failed:", "failure:", "runtime error:"))


def _model_status_for_reply(message: ChatMessage, choice: ModelChoice) -> str:
    if choice.backend != "builtin" and _message_looks_like_error_response(message):
        model = message.model or choice.name
        return f"Model error: {model} ({choice.reason}): {message.text}"
    return _model_status_from_choice(choice)


def _append_stream_marker(text: str, marker: str) -> str:
    """Appends a stream-interruption marker (e.g. `"[stopped]"` or
    `"[interrupted: ...]"`) to whatever partial text streamed in so far --
    with a leading space when there's prior text, or standing alone (no
    leading space) when nothing streamed in before the interruption."""
    return f"{text} {marker}" if text else marker


# How often (in seconds, wall-clock) a streaming Groq reply is allowed to
# trigger a `call_from_thread` chat refresh. Streaming re-renders the whole
# transcript per refresh_chat call (see ChatView.refresh_from_messages), so
# firing one on every token would flood the Textual event loop -- this
# throttles that down to a UI-perceptible cadence. The in-progress message's
# `.text` itself is still updated on every chunk regardless of this throttle;
# only the on-screen refresh is rate-limited, and a final refresh always
# happens when the stream ends (success, cancel, or error) regardless of how
# recently the last one fired.
_STREAM_REFRESH_INTERVAL_SECONDS = 0.1


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
        chat_sessions_root: Path | None = None,
        model_catalog_root: Path | None = None,
        local_model_runner: LocalModelRunner | None = None,
        groq_runner: GroqRunner | None = None,
        is_model_runtime_available: model_router.RuntimeAvailability | None = None,
        prompt_routing_policy: PromptRoutingPolicy | None = None,
        model_download_fn: model_downloader.DownloadFile | None = None,
        initial_queue: list[LaunchSpec | str] | None = None,
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
        self._chat_sessions_root = chat_sessions_root or paths.chat_sessions_root()
        self._chat_session_path = chat_store.session_path(
            self._chat_sessions_root, chat_store.new_session_id()
        )
        self.chat_messages: list[ChatMessage] = []
        self._chat_model_status: str | None = None
        # Transient "Searching docs for '...'"-style status shown while a
        # Groq tool-call round is in flight (see
        # docs/prd/chat-search-tools.md, "Tool-call round"). Deliberately
        # NOT part of ChatMessage/chat_store persistence -- it's not part of
        # the conversation, it's a fleeting UI signal that a background
        # tool call is running, and it's meaningless once that call has
        # already resolved (e.g. across a resumed session). Set from
        # GroqRunner.generate_stream's `on_status` callback in
        # `_run_stream_reply` below, cleared once the final answer's first
        # chunk arrives (or, defensively, at the end of any turn).
        self._chat_tool_status: str | None = None
        self._model_catalog_root = model_catalog_root or paths.models_root()
        self._local_model_runner = local_model_runner or LocalModelRunner()
        # Unlike `_local_model_runner`, not eagerly defaulted here:
        # `GroqRunner()`'s real client construction raises if `GROQ_API_KEY`
        # is unset (the common case for most installs/tests), so the real
        # default is only constructed lazily in `_answer_prompt_async`, at
        # the point a Groq choice is actually reached -- which, per
        # `groq_policy.build_policy`'s catalog gating, only happens when the
        # key is already set.
        self._groq_runner = groq_runner
        # The `threading.Event` for whichever Groq stream is currently
        # in-flight, if any -- created fresh per generation in
        # `_run_groq_stream` and cleared back to None once that generation
        # finishes (success, cancel, or error). `cancel_active_generation`
        # (wired to DashboardScreen's escape binding) sets it if present;
        # guarded clearing (see `_run_groq_stream`) means a late callback from
        # an already-finished generation never clobbers a newer one's event.
        self._active_cancel_event: threading.Event | None = None
        self._is_model_runtime_available = is_model_runtime_available
        self._prompt_routing_policy = prompt_routing_policy
        # None means "use model_downloader's real Hugging Face download";
        # tests inject a fake here to avoid real network calls from
        # ModelDownloadScreen.
        self._model_download_fn = model_download_fn

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
        # either a LaunchSpec (a `gepa ...` command) or a plain str (a chat
        # prompt, queued so it is answered in the same order it was submitted).
        # `_queue_watch_state` is whichever DashboardState
        # `_poll_queue_advance` is currently watching for a RUNNING -> terminal
        # transition -- always `self.state` as of the last time a live run was
        # (re)started, so the poll never fires twice for the same run.
        self.pending_queue: list[LaunchSpec | str] = list(initial_queue or [])
        self._queued_prompt_refs: list[ChatMessage | None] = []
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
            self._dashboard_screen = DashboardScreen(
                self.runs_root,
                live_state=self.state,
                chat_messages=self.chat_messages,
                chat_model_status=self._chat_model_status,
                model_catalog_root=self._model_catalog_root,
                prompt_routing_policy=self._prompt_routing_policy,
            )
            self.push_screen(self._dashboard_screen)
            if self.pending_queue:
                self._persist_queue()
                self.call_after_refresh(self._refresh_queue_panel)
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
        leftover_chats = chat_store.discover_resumable(
            self._chat_sessions_root, self._chat_session_path
        )
        if not leftover and not leftover_chats:
            if not local_models.list_models(self._model_catalog_root):
                self.push_screen(ModelPickerScreen())
            else:
                self.push_screen(InputScreen())
            return

        merged = queue_store.load_and_merge(leftover)
        merged_chat = chat_store.load_and_merge(leftover_chats)
        message = _resume_prompt_message(
            len(merged),
            len(leftover),
            len(merged_chat),
            len(leftover_chats),
        )

        def on_result(confirmed: bool) -> None:
            queue_store.delete_files(leftover)
            chat_store.delete_files(leftover_chats)
            if confirmed:
                self.pending_queue = merged
                self.chat_messages = merged_chat
                self._rebuild_queued_prompt_refs()
                self._persist_queue()
                self._persist_chat()
                # Not enter_browse_mode()'s switch_screen: by the time this
                # callback runs, ConfirmScreen has already popped itself back
                # off, leaving the app's implicit base screen on top -- which
                # (unlike InputScreen) was never itself pushed via
                # push_screen, so it has no result-callback slot for
                # switch_screen to pop. push_screen (as the on_mount
                # launch-mode branch above also does from this same base
                # screen) is the correct call here.
                self._dashboard_screen = DashboardScreen(
                    self.runs_root,
                    chat_messages=self.chat_messages,
                    chat_model_status=self._chat_model_status,
                    model_catalog_root=self._model_catalog_root,
                    prompt_routing_policy=self._prompt_routing_policy,
                )
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

    def enter_browse_mode(self, *, initial_tab: str | None = None) -> None:
        """InputScreen's empty-Enter path: the same bare-browse DashboardScreen
        `_browse()`'s launch-mode-free `AutumnApp` construction produces (no
        live_state), swapped in for InputScreen with `switch_screen` rather
        than pushed on top of it -- there's nothing to go "back" to.

        `initial_tab` lets a caller land on a specific tab instead of the
        default first one -- used by `finish_model_download` to open on the
        Models tab when multiple models were just installed and there's no
        unambiguous default to pick for the user."""
        self._dashboard_screen = DashboardScreen(
            self.runs_root,
            chat_messages=self.chat_messages,
            chat_model_status=self._chat_model_status,
            model_catalog_root=self._model_catalog_root,
            prompt_routing_policy=self._prompt_routing_policy,
            initial_tab=initial_tab,
        )
        self.switch_screen(self._dashboard_screen)

    def _persist_chat(self) -> None:
        chat_store.persist_chat(self._chat_session_path, self.chat_messages, pid=os.getpid())

    @property
    def model_catalog_root(self) -> Path:
        return self._model_catalog_root

    @property
    def model_download_fn(self) -> model_downloader.DownloadFile | None:
        return self._model_download_fn

    def set_default_model(self, group: str, name: str) -> None:
        try:
            catalog.set_default(
                self._model_catalog_root, group, name, policy=self._prompt_routing_policy
            )
        except ValueError as exc:
            self.notify(str(exc), severity="warning")
            return
        self.notify(f"Default model set to {name}", severity="information")

    def finish_model_download(self, *, focus_models_tab: bool = False) -> None:
        """ModelDownloadScreen's success path: continues into the dashboard
        exactly like InputScreen's empty-Enter path (`enter_browse_mode`).

        `focus_models_tab` is set when more than one model was just
        downloaded -- the first one installed becomes the default
        automatically (`local_models.install_model`'s empty-catalog rule),
        but with several newly-installed models to choose from, landing on
        the Models tab lets the user confirm/change that pick instead of it
        being silently implicit."""
        self.enter_browse_mode(initial_tab="models-tab" if focus_models_tab else None)

    def _append_user_prompt(self, text: str) -> ChatMessage:
        message = ChatMessage(role="user", text=text)
        self.chat_messages.append(message)
        self._persist_chat()
        if hasattr(self, "_dashboard_screen"):
            self._dashboard_screen.refresh_chat(self.chat_messages, self._chat_model_status)
        return message

    def _insert_assistant_message(
        self,
        message: ChatMessage,
        *,
        insert_after: ChatMessage | None = None,
    ) -> None:
        """Places `message` into `self.chat_messages` at the position a
        finished reply to the prompt it's answering would occupy: appended at
        the end normally, or immediately after `insert_after` when a prompt
        was answered out of submission order (queued replies -- see
        `test_live_run_queues_prompt_replies_in_order_without_duplicate_users`).
        Split out of `_append_assistant_reply` so a streaming reply's
        still-empty placeholder message can be inserted at the right spot
        immediately, before any text has streamed in, then have that same
        object's `.text` mutated in place as chunks arrive."""
        if insert_after is None:
            self.chat_messages.append(message)
        else:
            insert_at = next(
                (
                    index + 1
                    for index, existing in enumerate(self.chat_messages)
                    if existing is insert_after
                ),
                len(self.chat_messages),
            )
            self.chat_messages.insert(insert_at, message)

    def _refresh_chat_display(self) -> None:
        if hasattr(self, "_dashboard_screen"):
            self._dashboard_screen.refresh_chat(
                self.chat_messages, self._chat_model_status, self._chat_tool_status
            )

    def _finalize_assistant_reply(
        self,
        message: ChatMessage,
        choice: ModelChoice,
        *,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Whatever a *finished* assistant reply needs beyond already being
        present in `self.chat_messages`: status line, persistence, a final
        chat refresh, and the queue's on_complete hook. Does not insert --
        callers that haven't already inserted `message` should use
        `_append_assistant_reply` instead."""
        self._chat_model_status = _model_status_for_reply(message, choice)
        # Belt-and-suspenders clear: `_run_stream_reply`'s `on_chunk` already
        # clears this the moment the first chunk of a real answer arrives,
        # but any turn that finishes without ever reaching `on_chunk` (e.g.
        # a decide-call error surfacing before the tool round completes)
        # should still not leave a stale "Searching..." status showing once
        # the turn is over.
        self._chat_tool_status = None
        self._persist_chat()
        self._refresh_chat_display()
        if on_complete is not None:
            on_complete()

    def _append_assistant_reply(
        self,
        message: ChatMessage,
        choice: ModelChoice,
        *,
        insert_after: ChatMessage | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self._insert_assistant_message(message, insert_after=insert_after)
        self._finalize_assistant_reply(message, choice, on_complete=on_complete)

    def cancel_active_generation(self) -> None:
        """Wired to DashboardScreen's escape binding: stops whatever stream
        (Groq or local llama.cpp) is currently in-flight, if any, keeping
        the partial text that's already streamed in (see
        `_run_stream_reply`). No-op if nothing is currently streaming."""
        if self._active_cancel_event is not None:
            self._active_cancel_event.set()

    def _run_stream_reply(
        self,
        *,
        choice: ModelChoice,
        insert_after: ChatMessage | None,
        on_complete: Callable[[], None] | None,
        stream: Callable[..., None],
        error_type: type[Exception],
        finalize_success: Callable[[ChatMessage], None] | None = None,
        supports_tool_events: bool = False,
    ) -> None:
        """Shared skeleton for a streaming reply, used by both
        `_run_groq_stream` and `_run_local_stream`: inserts an empty
        placeholder `ChatMessage` into `self.chat_messages`, streams chunks
        into it via `stream(on_chunk=..., cancel_event=...)` (already bound
        to whichever runner/model/messages produced it) with throttled UI
        refreshes, and on completion appends a `[stopped]`/`[interrupted:
        ...]` marker for cancellation/errors -- keeping whatever text
        already streamed in either way, never substituting a different reply
        (see module docstring / PRD "Error handling mid-stream").

        `finalize_success`, if given, post-processes the message only when
        the stream completed cleanly (no error, not cancelled) -- e.g.
        local's trailing-whitespace strip, which only makes sense for output
        that actually finished rather than partial text.

        `supports_tool_events`, if `True`, additionally passes `on_status`/
        `on_citation` callbacks to `stream(...)` -- only `_run_groq_stream`
        sets this, since only `GroqRunner.generate_stream` has anything to
        report through them (see its docstring). `_run_local_stream` leaves
        this `False` and its `stream` lambda keeps the exact
        `on_chunk`/`cancel_event`-only signature it always had, so this
        addition never touches the local `llama.cpp` streaming path."""
        message = ChatMessage(role="assistant", text="", model=choice.name)
        cancel_event = threading.Event()
        self._active_cancel_event = cancel_event
        self._chat_tool_status = None

        self.call_from_thread(self._insert_assistant_message, message, insert_after=insert_after)
        self.call_from_thread(self._refresh_chat_display)

        last_refresh = time.monotonic()

        def on_chunk(text: str) -> None:
            nonlocal last_refresh
            # The final answer's first chunk is exactly the moment a
            # "Searching docs for '...'" status (if any) should disappear --
            # see acceptance criteria in the #26 issue. Direct attribute
            # write, no call_from_thread needed (same as `message.text`
            # below): it's a plain Python object mutation, not a Textual
            # widget update, and the throttled refresh just below picks it
            # up on whichever call actually fires the refresh.
            if self._chat_tool_status is not None:
                self._chat_tool_status = None
            message.text += text
            now = time.monotonic()
            if now - last_refresh >= _STREAM_REFRESH_INTERVAL_SECONDS:
                last_refresh = now
                self.call_from_thread(self._refresh_chat_display)

        def on_status(text: str) -> None:
            self._chat_tool_status = text
            self.call_from_thread(self._refresh_chat_display)

        def on_citation(citation: ToolCitation) -> None:
            message.citation = citation

        error: Exception | None = None
        try:
            if supports_tool_events:
                stream(on_chunk=on_chunk, cancel_event=cancel_event, on_status=on_status, on_citation=on_citation)
            else:
                stream(on_chunk=on_chunk, cancel_event=cancel_event)
        except error_type as exc:
            error = exc

        # Guard clearing: only clear if this is still the event this
        # generation set -- a later generation could already have replaced it
        # by the time this callback runs.
        if self._active_cancel_event is cancel_event:
            self._active_cancel_event = None

        if error is not None:
            message.text = _append_stream_marker(message.text, f"[interrupted: {error}]")
        elif cancel_event.is_set():
            message.text = _append_stream_marker(message.text, "[stopped]")
        elif finalize_success is not None:
            finalize_success(message)

        # `choice` is untouched here even on cancel/error -- deliberately
        # different from the offline-tiny fallback path in
        # `_answer_prompt_async` (the non-executable provider stub), which
        # discards the failed choice and regenerates a whole new
        # offline-tiny reply. Throwing away real partial output the user
        # already saw would be worse than a status line that still says
        # "Model: <model>".
        self.call_from_thread(
            self._finalize_assistant_reply,
            message,
            choice,
            on_complete=on_complete,
        )

    def _run_groq_stream(
        self,
        *,
        choice: ModelChoice,
        snapshot: list[ChatMessage],
        insert_after: ChatMessage | None,
        on_complete: Callable[[], None] | None,
    ) -> None:
        """Runs on `_answer_prompt_async`'s background thread for a Groq
        choice: streams the reply token-by-token into a placeholder
        `ChatMessage` already sitting in `self.chat_messages` via
        `_run_stream_reply`."""
        groq_runner = self._groq_runner or GroqRunner()
        self._run_stream_reply(
            choice=choice,
            insert_after=insert_after,
            on_complete=on_complete,
            stream=lambda *, on_chunk, cancel_event, on_status, on_citation: groq_runner.generate_stream(
                snapshot,
                choice.name,
                on_chunk=on_chunk,
                cancel_event=cancel_event,
                on_status=on_status,
                on_citation=on_citation,
            ),
            error_type=GroqRuntimeError,
            supports_tool_events=True,
        )

    def _run_local_stream(
        self,
        *,
        choice: ModelChoice,
        snapshot: list[ChatMessage],
        insert_after: ChatMessage | None,
        on_complete: Callable[[], None] | None,
    ) -> None:
        """Runs on `_answer_prompt_async`'s background thread for a local
        llama.cpp choice: streams the reply incrementally (via `Popen`, see
        `LocalModelRunner.generate_stream`) into a placeholder `ChatMessage`
        already sitting in `self.chat_messages` via `_run_stream_reply`. A
        `LocalModelRuntimeError` (missing `llama-cli`, non-zero exit, crash)
        is handled the same as a mid-stream Groq error: the partial text
        already streamed in is kept with an `[interrupted: ...]` marker
        appended, not discarded in favor of a fresh offline-tiny reply."""
        model = LocalModel(
            name=choice.name,
            backend=choice.backend,
            path=choice.path,
            context_window=choice.context_window,
            is_default=True,
        )
        self._run_stream_reply(
            choice=choice,
            insert_after=insert_after,
            on_complete=on_complete,
            stream=lambda *, on_chunk, cancel_event: self._local_model_runner.generate_stream(
                snapshot, model, on_chunk=on_chunk, cancel_event=cancel_event
            ),
            error_type=LocalModelRuntimeError,
            finalize_success=lambda message: setattr(message, "text", message.text.strip()),
        )

    def _answer_prompt_async(
        self,
        *,
        prompt: str,
        snapshot: list[ChatMessage],
        insert_after: ChatMessage | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        def run() -> None:
            choice = model_router.choose_model(
                prompt=prompt,
                catalog_root=self._model_catalog_root,
                is_runtime_available=self._is_model_runtime_available,
                policy=self._prompt_routing_policy,
            )
            if choice.backend == "llama.cpp" and choice.path is not None:
                self._run_local_stream(
                    choice=choice,
                    snapshot=snapshot,
                    insert_after=insert_after,
                    on_complete=on_complete,
                )
                return
            elif choice.backend == "provider" and choice.provider == "groq":
                self._run_groq_stream(
                    choice=choice,
                    snapshot=snapshot,
                    insert_after=insert_after,
                    on_complete=on_complete,
                )
                return
            elif choice.backend == "provider":
                choice = ModelChoice(
                    name=local_llm.OFFLINE_TINY_MODEL,
                    backend="builtin",
                    path=None,
                    reason=f"provider {choice.name} not executable yet",
                )
                message = local_llm.generate_response(snapshot, choice)
            else:
                message = local_llm.generate_response(snapshot, choice)
            self.call_from_thread(
                self._append_assistant_reply,
                message,
                choice,
                insert_after=insert_after,
                on_complete=on_complete,
            )

        threading.Thread(target=run, daemon=True).start()

    def open_chat_prompt(self, text: str) -> None:
        user_message = self._append_user_prompt(text)
        self._dashboard_screen = DashboardScreen(
            self.runs_root,
            chat_messages=self.chat_messages,
            chat_model_status=self._chat_model_status,
            model_catalog_root=self._model_catalog_root,
            prompt_routing_policy=self._prompt_routing_policy,
            initial_tab="chat-tab",
        )
        self.switch_screen(self._dashboard_screen)
        self.call_after_refresh(
            self._answer_prompt_async,
            prompt=text,
            snapshot=[user_message],
        )

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
        self._dashboard_screen = DashboardScreen(
            self.runs_root,
            live_state=state,
            chat_messages=self.chat_messages,
            chat_model_status=self._chat_model_status,
            model_catalog_root=self._model_catalog_root,
            prompt_routing_policy=self._prompt_routing_policy,
        )
        self.switch_screen(self._dashboard_screen)
        self._start_live_run()

    def launch_gepa_runs(self, specs: list[LaunchSpec]) -> None:
        """Launches the first parsed GEPA run and queues the rest in order."""
        if not specs:
            return
        self.pending_queue.extend(specs[1:])
        if len(specs) > 1:
            self._persist_queue()
        self.launch_gepa_run(specs[0])
        if len(specs) > 1:
            self.call_after_refresh(self._refresh_queue_panel)

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

    def _rebuild_queued_prompt_refs(self) -> None:
        """Best-effort link from restored prompt queue entries to persisted
        user messages, favoring the newest matching chat messages for duplicate
        prompt text.
        """
        prompts = [item for item in self.pending_queue if isinstance(item, str)]
        refs: list[ChatMessage | None] = []
        cursor = len(self.chat_messages) - 1
        for prompt in reversed(prompts):
            found: ChatMessage | None = None
            for index in range(cursor, -1, -1):
                message = self.chat_messages[index]
                if message.role == "user" and message.text == prompt:
                    found = message
                    cursor = index - 1
                    break
            refs.append(found)
        self._queued_prompt_refs = list(reversed(refs))

    def _queued_prompt_context(self, prompt: str) -> tuple[list[ChatMessage], ChatMessage]:
        ref = self._queued_prompt_refs.pop(0) if self._queued_prompt_refs else None
        if ref is None or not any(message is ref for message in self.chat_messages):
            ref = self._append_user_prompt(prompt)
        index = self._chat_message_index(ref)
        return list(self.chat_messages[: index + 1]), ref

    def _chat_message_index(self, ref: ChatMessage) -> int:
        return next(index for index, message in enumerate(self.chat_messages) if message is ref)

    def submit_command(self, text: str) -> None:
        """Handles one line submitted via CommandBar (`:` on DashboardScreen).

        A `gepa <script> ...` command launches immediately if no run is live,
        or is appended to `pending_queue` if one is. A non-`gepa` prompt is
        answered immediately when possible; while a run is live, its user
        message is displayed/persisted immediately and the answer is queued in
        submission order. Malformed `gepa ...` syntax is reported immediately
        either way -- parsing happens at submit time, not launch time, so a bad
        command never even makes it into the queue.
        """
        text = text.strip()
        if not text:
            return

        try:
            specs = parse_command_line(text)
        except LaunchSpecError as exc:
            self.notify(str(exc), severity="error")
            return

        if self._run_is_live():
            if specs is None:
                self._queued_prompt_refs.append(self._append_user_prompt(text))
                self.pending_queue.append(text)
            else:
                self.pending_queue.extend(specs)
            self._persist_queue()
            self._refresh_queue_panel()
            return

        if specs is not None:
            self.pending_queue.extend(specs[1:])
            if len(specs) > 1:
                self._persist_queue()
            self._launch_spec_now(specs[0])
            if len(specs) > 1:
                self._refresh_queue_panel()
        else:
            user_message = self._append_user_prompt(text)
            self._answer_prompt_async(
                prompt=text,
                snapshot=list(self.chat_messages[: self._chat_message_index(user_message) + 1]),
            )

    def _poll_queue_advance(self) -> None:
        state = self.state
        if state is None or state is not self._queue_watch_state:
            return
        if state.status is RunStatus.RUNNING:
            return
        # Stop watching this now-finished run so this fires exactly once per
        # run, then work through the queue -- prompt entries answer and resume
        # queue advancement on completion; a gepa entry launches and returns
        # (it re-points _queue_watch_state at the new run itself).
        self._queue_watch_state = None
        self._advance_queue()

    def _advance_queue(self) -> None:
        while self.pending_queue:
            item = self.pending_queue.pop(0)
            self._persist_queue()
            self._refresh_queue_panel()
            if isinstance(item, str):
                snapshot, user_message = self._queued_prompt_context(item)
                self._answer_prompt_async(
                    prompt=item,
                    snapshot=snapshot,
                    insert_after=user_message,
                    on_complete=self._advance_queue,
                )
                return
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
