"""Dashboard chat transcript view."""

from textual.app import ComposeResult
from textual.widgets import Static

from models import ChatMessage, ToolCitation


def _render_model_status(model_status: str | None) -> str:
    return model_status or "Model: ready to choose a local model or offline fallback"


def _render_tool_status(tool_status: str | None) -> str:
    """Transient "Searching docs for '...'"-style status shown while a Groq
    tool-call round is in flight (see docs/prd/chat-search-tools.md,
    "Tool-call round"). Purely UI-layer state -- `tool_status` is never part
    of a persisted `ChatMessage` (see models.py), it's passed alongside the
    message list from app.py each refresh and cleared back to `None` once
    the final answer starts streaming in. Renders as an empty line rather
    than disappearing entirely so the transcript below it doesn't jump."""
    return tool_status or ""


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

    DEFAULT_CSS = """
    ChatView {
        padding: 1 2;
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
        super().__init__(_render_messages(messages), *args, **kwargs)
        self._messages = messages
        self._model_status = model_status
        self._tool_status = tool_status

    def compose(self) -> ComposeResult:
        yield Static(_render_model_status(self._model_status), id="chat-model-status")
        yield Static(_render_tool_status(self._tool_status), id="chat-tool-status")
        yield Static(_render_messages(self._messages), id="chat-transcript")

    def refresh_from_messages(
        self,
        messages: list[ChatMessage],
        model_status: str | None = None,
        tool_status: str | None = None,
    ) -> None:
        self._messages = messages
        self._model_status = model_status
        self._tool_status = tool_status
        self.query_one("#chat-model-status", Static).update(_render_model_status(model_status))
        self.query_one("#chat-tool-status", Static).update(_render_tool_status(tool_status))
        self.query_one("#chat-transcript", Static).update(_render_messages(messages))
