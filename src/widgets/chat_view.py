"""Dashboard chat transcript view."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Static

from models import ChatMessage, ToolCitation

_TOOL_STATUS_INTERVAL_SECONDS = 0.4
_TOOL_STATUS_SUFFIXES = (".", "..", "...")


def _render_model_status(model_status: str | None) -> str:
    return model_status or "Model: ready to choose a local model or offline fallback"


def _render_tool_status(tool_status: str | None, frame: int = 0) -> str:
    """Transient "Searching docs for '...'"-style status shown while a Groq
    tool-call round is in flight (see docs/prd/chat-search-tools.md,
    "Tool-call round"). Purely UI-layer state -- `tool_status` is never part
    of a persisted `ChatMessage` (see models.py), it's passed alongside the
    message list from app.py each refresh and cleared back to `None` once
    the final answer starts streaming in. Renders as an empty line rather
    than disappearing entirely so the transcript below it doesn't jump."""
    if not tool_status:
        return ""
    return f"{tool_status}{_TOOL_STATUS_SUFFIXES[frame % len(_TOOL_STATUS_SUFFIXES)]}"


def _render_citation(citation: ToolCitation | None) -> str:
    """Renders a grounded assistant reply's source attribution, generically
    across whatever tool produced it -- keyed only off `citation.sources`
    (a list of human-readable labels), never off `citation.tool` by name, so
    a later `search_web` citation (URLs instead of doc path/section labels)
    renders through this same function with no changes here."""
    if citation is None or not citation.sources:
        return ""
    label = "Source" if len(citation.sources) == 1 else "Sources"
    return f"\n{label}: {'; '.join(citation.sources)}"


def _render_messages(messages: list[ChatMessage]) -> str:
    if not messages:
        return "Ask Autumn about your runs from the landing input or command bar."
    lines = []
    for message in messages:
        speaker = message.participant_name or ("You" if message.role == "user" else "Autumn")
        suffix = f" [{message.model}]" if message.model else ""
        block = f"{speaker}{suffix}: {message.text}"
        block += _render_citation(message.citation)
        lines.append(block)
    return "\n\n".join(lines)


class ChatView(Static):
    """Renders the shared dashboard chat conversation."""

    can_focus = True

    BINDINGS = [
        Binding("ctrl+a", "select_chat_transcript", "Select chat", show=False),
        Binding("ctrl+c", "copy_chat_selection", "Copy chat", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    ChatView {
        height: 1fr;
        padding: 1 2;
    }
    ChatView #chat-transcript-scroll {
        height: 1fr;
    }
    """

    def __init__(
        self,
        messages: list[ChatMessage],
        model_status: str | None = None,
        tool_status: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        kwargs["markup"] = False
        super().__init__(_render_messages(messages), *args, **kwargs)
        self._messages = messages
        self._model_status = model_status
        self._tool_status = tool_status
        self._tool_status_frame = 0

    def compose(self) -> ComposeResult:
        yield Static(_render_model_status(self._model_status), id="chat-model-status", markup=False)
        yield Static(
            _render_tool_status(self._tool_status, self._tool_status_frame),
            id="chat-tool-status",
            markup=False,
        )
        with VerticalScroll(id="chat-transcript-scroll"):
            yield Static(_render_messages(self._messages), id="chat-transcript", markup=False)

    def on_mount(self) -> None:
        self.set_interval(_TOOL_STATUS_INTERVAL_SECONDS, self._advance_tool_status)
        self.call_after_refresh(self._scroll_transcript_to_end)

    def refresh_from_messages(
        self,
        messages: list[ChatMessage],
        model_status: str | None = None,
        tool_status: str | None = None,
    ) -> None:
        self._messages = messages
        self._model_status = model_status
        if tool_status != self._tool_status:
            self._tool_status_frame = 0
        self._tool_status = tool_status
        should_follow_transcript = self._should_follow_transcript()
        self.query_one("#chat-model-status", Static).update(_render_model_status(model_status))
        self._update_tool_status()
        self.query_one("#chat-transcript", Static).update(_render_messages(messages))
        if should_follow_transcript:
            self.call_after_refresh(self._scroll_transcript_to_end)

    def _advance_tool_status(self) -> None:
        if self._tool_status is None:
            return
        self._tool_status_frame = (self._tool_status_frame + 1) % len(_TOOL_STATUS_SUFFIXES)
        self._update_tool_status()

    def _update_tool_status(self) -> None:
        self.query_one("#chat-tool-status", Static).update(
            _render_tool_status(self._tool_status, self._tool_status_frame)
        )

    def _scroll_transcript_to_end(self) -> None:
        self.query_one("#chat-transcript-scroll", VerticalScroll).scroll_end(animate=False, immediate=True)

    def _should_follow_transcript(self) -> bool:
        scroller = self.query_one("#chat-transcript-scroll", VerticalScroll)
        return scroller.max_scroll_y == 0 or scroller.is_vertical_scroll_end

    def action_select_chat_transcript(self) -> None:
        self.query_one("#chat-transcript", Static).text_select_all()

    def action_copy_chat_selection(self) -> None:
        selected_text = self.screen.get_selected_text()
        self.app.copy_to_clipboard(selected_text or _render_messages(self._messages))
        self.notify("Copied chat transcript", title="Chat")
