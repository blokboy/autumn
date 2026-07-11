"""Runtime boundary for calling Anthropic's hosted Messages API.

Direct mirror of `groq_runner.py`'s `generate()` method: an injectable
client (for tests, so no real SDK/network calls are ever needed), the same
`*RuntimeError` shape (`AnthropicRuntimeError` here), and the same
non-streaming-first approach. Deliberately does NOT replicate
`GroqRunner.generate_stream`'s Groq-specific tool-calling machinery (see
`groq_runner.py`'s module docstring and docs/prd/chat-search-tools.md) --
streaming is explicitly out of scope for Anthropic/OpenAI this round (see
docs/prd/multi-provider-models.md, "Future work" #21); `generate` alone is
enough to make Anthropic a real, working chat-completion provider.
"""

from typing import Any, Protocol

import anthropic

import credentials
from models import ChatMessage

# A short, fixed reply ceiling for a single (non-streaming, blocking) chat
# turn -- `max_tokens` is a required parameter on Anthropic's Messages API,
# unlike Groq/OpenAI's chat completions endpoint. Not user-configurable this
# round; revisit if real usage needs longer replies.
_MAX_TOKENS = 1024


class AnthropicRuntimeError(RuntimeError):
    """Raised when the Anthropic API cannot produce a reply."""


class _Messages(Protocol):
    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
    ) -> Any: ...


class AnthropicClient(Protocol):
    """The subset of `anthropic.Anthropic` that `AnthropicRunner` depends on."""

    messages: _Messages


def _to_payload(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [{"role": message.role, "content": message.text} for message in messages]


def _extract_text(response: Any) -> str:
    """Joins every text content block in `response.content` -- a plain,
    tool-free reply is normally just one block, but this tolerates more than
    one without crashing (and silently skips any non-text block type, e.g.
    if thinking were ever enabled)."""
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def _wrap_anthropic_error(model_name: str, exc: anthropic.AnthropicError) -> AnthropicRuntimeError:
    detail = getattr(exc, "message", None) or str(exc)
    return AnthropicRuntimeError(f"{model_name} failed: {detail}")


class AnthropicRunner:
    """Calls Anthropic's hosted Messages API for hosted models, as a single
    blocking response (`generate`). No streaming variant -- see the module
    docstring."""

    def __init__(self, *, client: AnthropicClient | None = None) -> None:
        self._client = (
            client
            if client is not None
            else anthropic.Anthropic(api_key=credentials.resolve_key("anthropic", "ANTHROPIC_API_KEY"))
        )

    def generate(self, messages: list[ChatMessage], model_name: str) -> ChatMessage:
        payload = _to_payload(messages)

        try:
            response = self._client.messages.create(
                model=model_name, max_tokens=_MAX_TOKENS, messages=payload
            )
        except anthropic.AnthropicError as exc:
            raise _wrap_anthropic_error(model_name, exc) from exc

        text = _extract_text(response)
        return ChatMessage(role="assistant", text=text.strip(), model=model_name)
