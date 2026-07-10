"""Dashboard chat transcript view."""

from textual.app import ComposeResult
from textual.widgets import Static

from autumn.models import ChatMessage


def _render_model_status(model_status: str | None) -> str:
    return model_status or "Model: not selected yet"


def _render_messages(messages: list[ChatMessage]) -> str:
    if not messages:
        return "No chat messages yet."
    lines = []
    for message in messages:
        speaker = "You" if message.role == "user" else "Autumn"
        suffix = f" [{message.model}]" if message.model else ""
        lines.append(f"{speaker}{suffix}: {message.text}")
    return "\n\n".join(lines)


class ChatView(Static):
    """Renders the dashboard-scoped LLM conversation."""

    DEFAULT_CSS = """
    ChatView {
        padding: 1 2;
    }
    """

    def __init__(
        self,
        messages: list[ChatMessage],
        model_status: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(_render_messages(messages), *args, **kwargs)
        self._messages = messages
        self._model_status = model_status

    def compose(self) -> ComposeResult:
        yield Static(_render_model_status(self._model_status), id="chat-model-status")
        yield Static(_render_messages(self._messages), id="chat-transcript")

    def refresh_from_messages(
        self,
        messages: list[ChatMessage],
        model_status: str | None = None,
    ) -> None:
        self._messages = messages
        self._model_status = model_status
        self.query_one("#chat-model-status", Static).update(_render_model_status(model_status))
        self.query_one("#chat-transcript", Static).update(_render_messages(messages))
