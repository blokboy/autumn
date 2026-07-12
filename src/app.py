"""The `autumn` Textual application."""

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
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
import prompt_optimization_intake
import queue_store
import runner
import subagent
from anthropic_runner import AnthropicRunner, AnthropicRuntimeError
from cli import LaunchSpec, LaunchSpecError, PromptOptimizationDraft, is_explicit_gepa_command, parse_command_line
from curated_models import CuratedModel
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
    PromptOptimizationRunSpec,
    PromptRoutingPolicy,
    RunKind,
    RunStatus,
    ToolCitation,
)
from openai_runner import OpenAIRunner, OpenAIRuntimeError
from prompt_optimization_contracts import PromptOptimizationSpec
from prompt_optimization_runtime import run as run_prompt_optimization
from runner import PromptOptimizationRuntime
from screens.confirm_screen import ConfirmScreen
from screens.dashboard_screen import DashboardScreen
from screens.help_screen import HelpScreen
from screens.input_screen import InputScreen
from screens.model_picker_screen import ModelPickerScreen
from screens.prompt_optimization_confirm_screen import PromptOptimizationConfirmScreen

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


def _format_capability_list(capabilities: tuple[str, ...]) -> str:
    return ", ".join(capabilities) if capabilities else "none"


def _subagent_fallback_warning_text(
    subagent_name: str,
    warning: subagent.SubagentFallbackWarning,
) -> str:
    source = (
        f"from {warning.original_model} to {warning.fallback_model}"
        if warning.original_model is not None
        else f"to {warning.fallback_model}"
    )
    return (
        f"Warning: {subagent_name} fell back {source}. "
        f"Reason: {warning.reason}. "
        "Capability impact: "
        f"expected [{_format_capability_list(warning.expected_capabilities)}]; "
        f"fallback provides [{_format_capability_list(warning.fallback_capabilities)}]; "
        f"lost [{_format_capability_list(warning.lost_capabilities)}]."
    )


def _download_status_text(
    entry: CuratedModel, index: int, total: int, percent: int | None = None
) -> str:
    """Command-bar status line for a background model download (#20) --
    e.g. "Downloading Llama 3.2 3B Instruct (2 of 3)... 42%"."""
    label = f"Downloading {entry.name}"
    if total > 1:
        label += f" ({index + 1} of {total})"
    label += f"... {percent}%" if percent is not None else "..."
    return label


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

_SUBAGENT_PREFIX = "/subagent "
_SUBAGENT_WORKING_TEXT = "working..."
_SUBAGENT_QUEUED_TEXT = "queued..."
_MAX_RUNNING_SUBAGENTS = 4


@dataclass
class _DashboardSubagent:
    name: str
    prompt: str
    message: ChatMessage
    cancelled: bool = False


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
        anthropic_runner: AnthropicRunner | None = None,
        openai_runner: OpenAIRunner | None = None,
        is_model_runtime_available: model_router.RuntimeAvailability | None = None,
        prompt_routing_policy: PromptRoutingPolicy | None = None,
        model_download_fn: model_downloader.DownloadFile | None = None,
        initial_queue: list[queue_store.Item] | None = None,
        eval_assets_root: Path | None = None,
        prompt_optimization_runtime: PromptOptimizationRuntime | None = None,
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
        self._eval_assets_root = eval_assets_root or paths.eval_assets_root()
        # Defaults to the real GEPA-driving implementation (#48); tests substitute
        # a fake via the constructor param instead of exercising real GEPA/model calls.
        self._prompt_optimization_runtime = prompt_optimization_runtime or run_prompt_optimization
        # Non-None only while a GEPA-specific implicit chat phrase (#50) is
        # gathering fields via follow-up questions -- explicit `gepa optimize`
        # never touches this, it goes straight to the confirmation screen.
        self._prompt_optimization_intake: prompt_optimization_intake.IntakeState | None = None
        self._local_model_runner = local_model_runner or LocalModelRunner()
        # Unlike `_local_model_runner`, not eagerly defaulted here:
        # `GroqRunner()`'s real client construction raises if `GROQ_API_KEY`
        # is unset (the common case for most installs/tests), so the real
        # default is only constructed lazily in `_answer_prompt_async`, at
        # the point a Groq choice is actually reached -- which, per
        # `groq_policy.build_policy`'s catalog gating, only happens when the
        # key is already set.
        self._groq_runner = groq_runner
        # Same lazy-default reasoning as `_groq_runner` above: `AnthropicRunner()`/
        # `OpenAIRunner()`'s real client construction reads
        # `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` (via `credentials.resolve_key`)
        # only when actually needed, so these stay `None` here and are only
        # constructed lazily in `_run_provider_reply`, at the point an
        # Anthropic/OpenAI choice is actually reached -- which, per
        # `anthropic_policy.build_policy`/`openai_policy.build_policy`'s
        # catalog gating, only happens when the corresponding key is already
        # set.
        self._anthropic_runner = anthropic_runner
        self._openai_runner = openai_runner
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
        # start_background_model_download's background thread.
        self._model_download_fn = model_download_fn
        # Command-bar status text for whichever background model download(s)
        # start_background_model_download (#20) currently has in flight, or
        # None when nothing is downloading -- threaded through every
        # DashboardScreen construction (same pattern as _chat_model_status)
        # so it survives screen swaps, and pushed to an already-mounted
        # screen via _refresh_download_status_display.
        self._download_status: str | None = None

        # Launch mode iff both run_name and run_dir are given (cli.py's `run`
        # subcommand always supplies both together); otherwise this is
        # browse-only, with no live state and no background thread.
        if run_name is not None and run_dir is not None:
            self.state: DashboardState | None = DashboardState(run_name=run_name, run_dir=run_dir)
            self._dashboard_callback: DashboardCallback | None = DashboardCallback(self, self.state)
        else:
            self.state = None
            self._dashboard_callback = None

        # In-memory queue of commands submitted via CommandBar while a run was
        # already live (see submit_command below). Entries are typed queue
        # items; queue_store still accepts/returns legacy LaunchSpec | str at
        # compatibility boundaries.
        # `_queue_watch_state` is whichever DashboardState
        # `_poll_queue_advance` is currently watching for a RUNNING -> terminal
        # transition -- always `self.state` as of the last time a live run was
        # (re)started, so the poll never fires twice for the same run.
        self.pending_queue: list[queue_store.TypedItem] = [
            queue_store.typed_item(item) for item in (initial_queue or [])
        ]
        self._queued_prompt_refs: list[ChatMessage | None] = []
        self._queue_watch_state: DashboardState | None = self.state
        self._next_subagent_number = 1
        self._running_subagents: dict[str, _DashboardSubagent] = {}
        self._queued_subagents: list[_DashboardSubagent] = []

    def on_mount(self) -> None:
        # Launch mode (both run_name/run_dir given, cli.py's `run` subcommand)
        # goes straight to the dashboard, unaffected by InputScreen below.
        # Browse-only construction (cli.py's `_browse`, i.e. bare `autumn`)
        # lands on InputScreen first instead of jumping straight into browse
        # mode -- InputScreen itself decides whether to fall through to browse
        # (empty Enter) or promote into a live run (`gepa run ...`, see
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
                download_status=self._download_status,
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

        merged = queue_store.load_and_merge_typed(leftover)
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
                    download_status=self._download_status,
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
        `launch_gepa_run` (InputScreen's `gepa run ...` path), so the two ways of
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
            runner.launch(self._dashboard_callback, spec, prompt_optimization_runtime=run_prompt_optimization)

    def enter_browse_mode(self, *, initial_tab: str | None = None) -> None:
        """InputScreen's empty-Enter path: the same bare-browse DashboardScreen
        `_browse()`'s launch-mode-free `AutumnApp` construction produces (no
        live_state), swapped in for InputScreen with `switch_screen` rather
        than pushed on top of it -- there's nothing to go "back" to.

        `initial_tab` lets a caller land on a specific tab instead of the
        default first one -- used by `start_background_model_download` to
        open on the Models tab when more than one model was just picked and
        there's no unambiguous default to show the user yet."""
        self._dashboard_screen = DashboardScreen(
            self.runs_root,
            chat_messages=self.chat_messages,
            chat_model_status=self._chat_model_status,
            model_catalog_root=self._model_catalog_root,
            prompt_routing_policy=self._prompt_routing_policy,
            initial_tab=initial_tab,
            download_status=self._download_status,
        )
        self.switch_screen(self._dashboard_screen)

    def _persist_chat(self) -> None:
        chat_store.persist_chat(self._chat_session_path, self.chat_messages, pid=os.getpid())

    @property
    def model_catalog_root(self) -> Path:
        return self._model_catalog_root

    @property
    def eval_assets_root(self) -> Path:
        return self._eval_assets_root

    @property
    def prompt_routing_policy(self) -> PromptRoutingPolicy:
        return self._prompt_routing_policy

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

    def start_background_model_download(self, entries: list[CuratedModel]) -> None:
        """ModelPickerScreen's confirm path (#20): registers every picked
        entry as a "downloading" placeholder in the local catalog
        synchronously (see `local_models.mark_downloading`) -- so
        `model_router.choose_model` already knows there's a default chosen
        but not yet runnable, and a relaunch mid-download won't re-show the
        picker (`list_models` is already non-empty) -- then enters the
        dashboard immediately, exactly like InputScreen's empty-Enter path
        (`enter_browse_mode`), and downloads/installs `entries` one at a
        time on a background thread. No blocking progress screen in
        between: the command bar's download-status line (see
        `_set_download_status`) is the only visible sign a download is
        still in flight.

        More than one entry lands on the Models tab (same rationale
        `enter_browse_mode`'s docstring gives): the first picked becomes the
        eventual default (mirrors `local_models.install_model`'s
        empty-catalog rule, applied here to pick order since nothing is
        actually installed yet), but with several in flight, landing on the
        Models tab lets the user see/confirm that instead of it being
        silently implicit."""
        for entry in entries:
            local_models.mark_downloading(
                self._model_catalog_root, name=entry.name, context_window=entry.context_window
            )
        self._download_status = _download_status_text(entries[0], 0, len(entries))
        self.enter_browse_mode(initial_tab="models-tab" if len(entries) > 1 else None)
        # DashboardScreen mounts asynchronously -- the initial download
        # status is already baked into its compose (via the download_status
        # constructor arg above), but the background thread's later
        # call_from_thread updates go through _dashboard_screen.
        # refresh_download_status, which needs CommandBar already mounted
        # (query_one), so starting the thread waits for that mount to land
        # rather than racing it (same idiom on_result uses for
        # _advance_queue above).
        self.call_after_refresh(self._start_model_download_thread, entries)

    def _start_model_download_thread(self, entries: list[CuratedModel]) -> None:
        threading.Thread(
            target=self._download_models_in_background, args=(entries,), daemon=True
        ).start()

    def _set_download_status(self, text: str | None) -> None:
        self._download_status = text
        self._refresh_download_status_display()

    def _refresh_download_status_display(self) -> None:
        if hasattr(self, "_dashboard_screen"):
            self._dashboard_screen.refresh_download_status(self._download_status)

    def _refresh_models_display(self) -> None:
        if hasattr(self, "_dashboard_screen"):
            self._dashboard_screen.refresh_models()

    def _on_model_download_progress(
        self,
        entry: CuratedModel,
        index: int,
        total: int,
        downloaded: int,
        total_bytes: int | None,
    ) -> None:
        if not total_bytes:
            return
        percent = min(100, int(downloaded * 100 / total_bytes))
        self.call_from_thread(
            self._set_download_status, _download_status_text(entry, index, total, percent)
        )

    def _on_model_download_failure(
        self, entry: CuratedModel, detail: str, remaining: list[CuratedModel]
    ) -> None:
        """A background download failed (#20): mirrors the message the old
        blocking ModelDownloadScreen showed, but notifies instead of falling
        back to a screen -- chat/other interaction was never blocked in the
        first place. Removes the failed entry's "downloading" placeholder
        (and any not-yet-attempted entries queued after it, since this
        entry's failure stops the rest of the chain, same as the old
        blocking flow) so no permanently-stuck "downloading" ghost entry is
        left behind; whatever installed successfully before this failure
        stays installed untouched."""
        local_models.remove_model(self._model_catalog_root, entry.name)
        for queued in remaining:
            local_models.remove_model(self._model_catalog_root, queued.name)
        self.notify(f"Couldn't install {entry.name}: {detail}", severity="error")
        self._set_download_status(None)
        self._refresh_models_display()

    def _on_model_downloads_complete(self) -> None:
        self._set_download_status(None)
        self._refresh_models_display()

    def _download_models_in_background(self, entries: list[CuratedModel]) -> None:
        total = len(entries)
        for index, entry in enumerate(entries):
            self.call_from_thread(
                self._set_download_status, _download_status_text(entry, index, total)
            )
            try:
                model_downloader.download_and_install(
                    self._model_catalog_root,
                    entry,
                    on_progress=lambda downloaded, total_bytes, e=entry, i=index: (
                        self._on_model_download_progress(e, i, total, downloaded, total_bytes)
                    ),
                    download_file_fn=self._model_download_fn,
                )
            except (httpx.HTTPError, OSError) as exc:
                self.call_from_thread(
                    self._on_model_download_failure, entry, str(exc), entries[index + 1 :]
                )
                return
        self.call_from_thread(self._on_model_downloads_complete)

    def _append_user_prompt(self, text: str) -> ChatMessage:
        message = ChatMessage(role="user", text=text)
        self.chat_messages.append(message)
        self._persist_chat()
        if hasattr(self, "_dashboard_screen"):
            self._dashboard_screen.refresh_chat(self.chat_messages, self._chat_model_status)
        return message

    def _append_assistant_message(self, text: str) -> ChatMessage:
        """A plain, non-streamed assistant chat line -- used by prompt
        optimization intake's (#50) follow-up questions/cancellation
        acknowledgements, which aren't model-generated replies."""
        message = ChatMessage(role="assistant", text=text)
        self.chat_messages.append(message)
        self._persist_chat()
        self._refresh_chat_display()
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

    def _next_subagent_name(self) -> str:
        existing = {
            message.participant_name
            for message in self.chat_messages
            if message.participant_name is not None
        }
        while True:
            name = f"Autumn Sub Agent {self._next_subagent_number}"
            self._next_subagent_number += 1
            if name not in existing:
                return name

    def _parse_subagent_cancel_command(self, text: str) -> str | None:
        prefix = f"{_SUBAGENT_PREFIX}cancel "
        if text == "/subagent cancel":
            return ""
        if not text.startswith(prefix):
            return None
        return text[len(prefix) :].strip()

    def _parse_subagent_command(self, text: str) -> str | None:
        if text == "/subagent":
            return ""
        if not text.startswith(_SUBAGENT_PREFIX):
            return None
        return text[len(_SUBAGENT_PREFIX) :].strip()

    def _launch_subagent(self, *, raw_command: str, prompt: str) -> None:
        self._append_user_prompt(raw_command)
        name = self._next_subagent_name()
        message = ChatMessage(
            role="assistant",
            participant_name=name,
            text=_SUBAGENT_WORKING_TEXT
            if len(self._running_subagents) < _MAX_RUNNING_SUBAGENTS
            else _SUBAGENT_QUEUED_TEXT,
        )
        self.chat_messages.append(message)
        self._persist_chat()
        self._refresh_chat_display()
        job = _DashboardSubagent(name=name, prompt=prompt, message=message)
        if len(self._running_subagents) >= _MAX_RUNNING_SUBAGENTS:
            self._queued_subagents.append(job)
            return
        self._start_subagent_job(job)

    def _start_subagent_job(self, job: _DashboardSubagent) -> None:
        self._running_subagents[job.name] = job
        job.message.text = _SUBAGENT_WORKING_TEXT
        self._persist_chat()
        self._refresh_chat_display()

        def run() -> None:
            try:
                result = subagent.run_subagent(
                    job.prompt,
                    catalog_root=self._model_catalog_root,
                    is_runtime_available=self._is_model_runtime_available,
                    policy=self._prompt_routing_policy,
                    local_model_runner=self._local_model_runner,
                    groq_runner=self._groq_runner,
                )
            except Exception as exc:
                self.call_from_thread(
                    self._finish_subagent_message,
                    job,
                    f"Error: subagent failed: {exc}",
                    None,
                    None,
                )
                return
            self.call_from_thread(
                self._finish_subagent_message,
                job,
                result.answer,
                result.choice.name,
                result.fallback_warning,
            )

        threading.Thread(target=run, daemon=True).start()

    def _finish_subagent_message(
        self,
        job: _DashboardSubagent,
        answer: str,
        model: str | None,
        fallback_warning: subagent.SubagentFallbackWarning | None,
    ) -> None:
        if job.cancelled:
            return
        self._running_subagents.pop(job.name, None)
        job.message.text = answer
        job.message.model = model
        if fallback_warning is not None:
            self.chat_messages.append(
                ChatMessage(
                    role="assistant",
                    text=_subagent_fallback_warning_text(job.name, fallback_warning),
                )
            )
        self._persist_chat()
        self._refresh_chat_display()
        self._start_next_queued_subagent()

    def _start_next_queued_subagent(self) -> None:
        while self._queued_subagents and len(self._running_subagents) < _MAX_RUNNING_SUBAGENTS:
            self._start_subagent_job(self._queued_subagents.pop(0))

    def _cancel_subagent(self, *, raw_command: str, name: str) -> None:
        self._append_user_prompt(raw_command)
        running = self._running_subagents.pop(name, None)
        if running is not None:
            running.cancelled = True
            running.message.text = "cancelled."
            self.chat_messages.append(
                ChatMessage(role="assistant", text=f"Cancelled running subagent {name}.")
            )
            self._persist_chat()
            self._refresh_chat_display()
            self._start_next_queued_subagent()
            return

        for index, queued in enumerate(self._queued_subagents):
            if queued.name == name:
                self._queued_subagents.pop(index)
                queued.cancelled = True
                queued.message.text = "cancelled before start."
                self.chat_messages.append(
                    ChatMessage(role="assistant", text=f"Cancelled queued subagent {name}.")
                )
                self._persist_chat()
                self._refresh_chat_display()
                return

        self.chat_messages.append(
            ChatMessage(role="assistant", text=f"No running or queued subagent named {name}.")
        )
        self._persist_chat()
        self._refresh_chat_display()

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

    def _confirm_from_thread(self, message: str, initial_delay: float) -> bool:
        """Bridges a mutating system tool's confirmation request (see
        `system_tools.py`, `docs/prd/chat-cli-parity-tools.md`) from the
        background thread `GroqRunner.generate_stream` runs on (via
        `_run_groq_stream`) to the main Textual thread: pushes
        `ConfirmScreen` there and blocks the calling (background) thread
        until the user answers, then returns the result. Passed to
        `GroqRunner` as its `confirm` callback.

        Blocking a background thread on a `threading.Event` while the main
        thread handles the modal is safe here specifically because this
        method is never called from the main thread itself -- doing so
        would deadlock, since `call_from_thread` requires a *different*
        thread than the one running the Textual event loop.
        """
        result_event = threading.Event()
        result: dict[str, bool] = {}

        def _on_result(confirmed: bool) -> None:
            result["confirmed"] = confirmed
            result_event.set()

        def _push() -> None:
            self.push_screen(
                ConfirmScreen(message, title="Confirm action", initial_delay=initial_delay),
                _on_result,
            )

        self.call_from_thread(_push)
        result_event.wait()
        return result["confirmed"]

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
        groq_runner = self._groq_runner or GroqRunner(
            catalog_root=self._model_catalog_root,
            runs_root=self.runs_root,
            confirm=self._confirm_from_thread,
        )
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

    def _run_provider_reply(
        self,
        *,
        choice: ModelChoice,
        snapshot: list[ChatMessage],
        insert_after: ChatMessage | None,
        on_complete: Callable[[], None] | None,
    ) -> None:
        """Runs on `_answer_prompt_async`'s background thread for an
        Anthropic or OpenAI choice: a single blocking `generate()` call (no
        streaming -- see `anthropic_runner.py`/`openai_runner.py`'s module
        docstrings, and docs/prd/multi-provider-models.md "Future work"
        #21), producing the whole reply before it's placed into
        `self.chat_messages` at all.

        Unlike `_run_groq_stream`/`_run_local_stream` (both of which keep
        whatever text already streamed in on a mid-stream error -- see
        `_run_stream_reply`), there is no partial output to preserve here:
        a `generate()` call either returns a complete reply or raises before
        returning anything. On a runtime error, this discards the failed
        provider `choice` and regenerates a whole new offline-tiny reply,
        exactly like the catch-all `elif choice.backend == "provider"`
        branch in `_answer_prompt_async` already does for provider stubs
        that have no runner at all."""
        if choice.provider == "anthropic":
            runner_obj: AnthropicRunner | OpenAIRunner = self._anthropic_runner or AnthropicRunner()
            error_type: type[Exception] = AnthropicRuntimeError
        else:
            runner_obj = self._openai_runner or OpenAIRunner()
            error_type = OpenAIRuntimeError

        try:
            message = runner_obj.generate(snapshot, choice.name)
        except error_type as exc:
            fallback_choice = ModelChoice(
                name=local_llm.OFFLINE_TINY_MODEL,
                backend="builtin",
                path=None,
                reason=f"provider {choice.name} failed: {exc}",
            )
            message = local_llm.generate_response(snapshot, fallback_choice)
            choice = fallback_choice

        self.call_from_thread(
            self._append_assistant_reply,
            message,
            choice,
            insert_after=insert_after,
            on_complete=on_complete,
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
            elif choice.backend == "provider" and choice.provider in ("anthropic", "openai"):
                self._run_provider_reply(
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
            download_status=self._download_status,
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
        """InputScreen's `gepa run <script> ...` path: promotes this already-mounted,
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
            download_status=self._download_status,
        )
        self.switch_screen(self._dashboard_screen)
        self._start_live_run()

    def launch_gepa_runs(self, specs: list[LaunchSpec]) -> None:
        """Launches the first parsed GEPA run and queues the rest in order."""
        if not specs:
            return
        self.pending_queue.extend(queue_store.ScriptQueueItem(spec) for spec in specs[1:])
        if len(specs) > 1:
            self._persist_queue()
        self.launch_gepa_run(specs[0])
        if len(specs) > 1:
            self.call_after_refresh(self._refresh_queue_panel)

    def start_prompt_optimization_draft(self, draft: PromptOptimizationDraft) -> None:
        """Handoff point for first-class prompt optimization commands: opens
        the editable final review screen (#47) rather than launching
        directly, so prompt/models/eval asset/metric/budget can be inspected
        and corrected before an expensive or long-running optimization
        actually starts."""
        self.push_screen(
            PromptOptimizationConfirmScreen(
                draft,
                runs_root=self.runs_root,
                catalog_root=self._model_catalog_root,
                prompt_routing_policy=self._prompt_routing_policy,
                eval_assets_root=self._eval_assets_root,
            )
        )

    def start_prompt_optimization_intake(self, draft: PromptOptimizationDraft, raw_text: str) -> None:
        """Handoff point for a GEPA-specific *implicit* chat phrase (#50),
        e.g. "run GEPA on this prompt" -- unlike explicit `gepa optimize`
        (`start_prompt_optimization_draft`, above), which jumps straight to
        the confirmation screen even when incomplete, an implicit trigger
        starts a short chat-style back-and-forth (see
        `_handle_prompt_optimization_intake_answer`) to fill in whatever the
        one-line trigger didn't already supply, before handing off to that
        same confirmation screen. Shared by InputScreen's pre-dashboard
        `on_input_submitted` and `submit_command`'s CommandBar path (a
        DashboardScreen is already mounted there)."""
        self._append_user_prompt(raw_text)
        if not hasattr(self, "_dashboard_screen"):
            self._dashboard_screen = DashboardScreen(
                self.runs_root,
                chat_messages=self.chat_messages,
                chat_model_status=self._chat_model_status,
                model_catalog_root=self._model_catalog_root,
                prompt_routing_policy=self._prompt_routing_policy,
                initial_tab="chat-tab",
                download_status=self._download_status,
            )
            self.switch_screen(self._dashboard_screen)

        state = prompt_optimization_intake.start_intake(draft, assets_root=self._eval_assets_root)
        if state.asking_field is None:
            self.start_prompt_optimization_draft(state.draft)
            return
        self._prompt_optimization_intake = state
        self._append_assistant_message(prompt_optimization_intake.question_for(state.asking_field))

    def _handle_prompt_optimization_intake_answer(self, text: str) -> None:
        """`submit_command`'s handler for every chat line submitted while
        `self._prompt_optimization_intake` is active -- every such line is
        treated as an intake answer or cancellation, never as a fresh command/
        chat prompt, until intake ends (completed or cancelled)."""
        state = self._prompt_optimization_intake
        assert state is not None
        self._append_user_prompt(text)

        if prompt_optimization_intake.is_cancel_phrase(text):
            self._prompt_optimization_intake = None
            self._append_assistant_message("Prompt optimization intake cancelled.")
            return

        new_state = prompt_optimization_intake.answer_intake(
            state,
            text,
            catalog_root=self._model_catalog_root,
            assets_root=self._eval_assets_root,
            prompt_routing_policy=self._prompt_routing_policy,
        )
        if new_state.asking_field is None:
            self._prompt_optimization_intake = None
            self.start_prompt_optimization_draft(new_state.draft)
            return

        self._prompt_optimization_intake = new_state
        question = prompt_optimization_intake.question_for(new_state.asking_field)
        if new_state.last_errors:
            question = "; ".join(new_state.last_errors) + f". {question}"
        self._append_assistant_message(question)

    def is_run_live(self) -> bool:
        """Public counterpart to `_run_is_live`, for callers outside this
        class (e.g. PromptOptimizationConfirmScreen deciding whether Launch
        should start a run immediately or queue behind the active one)."""
        return self._run_is_live()

    def _adopt_live_prompt_optimization_spec(self, spec: PromptOptimizationRunSpec) -> DashboardState:
        """Prompt-optimization counterpart to `_adopt_live_spec`: promotes
        this AutumnApp into a live prompt optimization run. There's no
        script_path/dry_run to track here -- `spec.optimization_spec` is the
        durable input `runner.launch` actually consumes."""
        self.run_name = spec.run_name
        self.run_dir = spec.run_dir
        self.script_path = None
        self.dry_run = False

        state = DashboardState(run_name=spec.run_name, run_dir=spec.run_dir, run_kind=RunKind.PROMPT_OPTIMIZATION)
        self.state = state
        self._dashboard_callback = DashboardCallback(self, state)
        self._queue_watch_state = state
        return state

    def _launch_runner_prompt_optimization(self, spec: PromptOptimizationRunSpec) -> None:
        runner.launch(
            self._dashboard_callback,
            spec,
            prompt_optimization_runtime=self._prompt_optimization_runtime,
            catalog_root=self._model_catalog_root,
            assets_root=self._eval_assets_root,
        )

    def launch_prompt_optimization_run(self, spec: PromptOptimizationRunSpec) -> None:
        """PromptOptimizationConfirmScreen's Launch action when no run is
        currently live: promotes into a live prompt-optimization run,
        mirroring `launch_gepa_run` for script `LaunchSpec`s."""
        state = self._adopt_live_prompt_optimization_spec(spec)
        self._dashboard_screen = DashboardScreen(
            self.runs_root,
            live_state=state,
            chat_messages=self.chat_messages,
            chat_model_status=self._chat_model_status,
            model_catalog_root=self._model_catalog_root,
            prompt_routing_policy=self._prompt_routing_policy,
            download_status=self._download_status,
        )
        self.switch_screen(self._dashboard_screen)
        self._launch_runner_prompt_optimization(spec)

    def queue_prompt_optimization_run(self, spec: PromptOptimizationSpec) -> None:
        """PromptOptimizationConfirmScreen's Launch action when a run is
        already live: enqueues behind it, same as a script run submitted
        while busy (see `submit_command`)."""
        self.pending_queue.append(queue_store.PromptOptimizationQueueItem(spec))
        self._persist_queue()
        self._refresh_queue_panel()

    def _launch_prompt_optimization_now(self, spec: PromptOptimizationSpec) -> None:
        """Launches `spec` against the already-mounted DashboardScreen
        (mirrors `_launch_spec_now`) -- used when a queued prompt
        optimization run reaches the front of the queue."""
        run_spec = PromptOptimizationRunSpec(optimization_spec=spec, run_dir=self.runs_root / spec.run_name)
        state = self._adopt_live_prompt_optimization_spec(run_spec)
        self._dashboard_screen.promote_to_live(state)
        self._launch_runner_prompt_optimization(run_spec)

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
        if not self._dashboard_screen.is_mounted:
            return
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
        prompts = [item.text for item in self.pending_queue if isinstance(item, queue_store.ChatQueueItem)]
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

        A `gepa run <script> ...` command launches immediately if no run is live,
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

        if self._prompt_optimization_intake is not None:
            self._handle_prompt_optimization_intake_answer(text)
            return

        subagent_cancel_name = self._parse_subagent_cancel_command(text)
        if subagent_cancel_name is not None:
            if not subagent_cancel_name:
                self.notify("Usage: /subagent cancel <name>", severity="error")
                return
            self._cancel_subagent(raw_command=text, name=subagent_cancel_name)
            return

        subagent_prompt = self._parse_subagent_command(text)
        if subagent_prompt is not None:
            if not subagent_prompt:
                self.notify("Usage: /subagent <prompt>", severity="error")
                return
            self._launch_subagent(raw_command=text, prompt=subagent_prompt)
            return

        try:
            specs = parse_command_line(
                text,
                catalog_root=self._model_catalog_root,
                prompt_routing_policy=self._prompt_routing_policy,
            )
        except LaunchSpecError as exc:
            self.notify(str(exc), severity="error")
            return

        if isinstance(specs, PromptOptimizationDraft):
            if is_explicit_gepa_command(text):
                self.start_prompt_optimization_draft(specs)
            else:
                self.start_prompt_optimization_intake(specs, text)
            return

        if self._run_is_live():
            if specs is None:
                self._queued_prompt_refs.append(self._append_user_prompt(text))
                self.pending_queue.append(queue_store.ChatQueueItem(text))
            else:
                self.pending_queue.extend(queue_store.ScriptQueueItem(spec) for spec in specs)
            self._persist_queue()
            self._refresh_queue_panel()
            return

        if specs is not None:
            self.pending_queue.extend(queue_store.ScriptQueueItem(spec) for spec in specs[1:])
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
            if isinstance(item, queue_store.ChatQueueItem):
                snapshot, user_message = self._queued_prompt_context(item.text)
                self._answer_prompt_async(
                    prompt=item.text,
                    snapshot=snapshot,
                    insert_after=user_message,
                    on_complete=self._advance_queue,
                )
                return
            if isinstance(item, queue_store.ScriptQueueItem):
                self._launch_spec_now(item.spec)
                return
            if isinstance(item, queue_store.PromptOptimizationQueueItem):
                self._launch_prompt_optimization_now(item.spec)
                return
            self.pending_queue.insert(0, item)
            self._persist_queue()
            self._refresh_queue_panel()
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
        runner.launch(callback, spec, prompt_optimization_runtime=run_prompt_optimization)
        screen.promote_to_live(state)
