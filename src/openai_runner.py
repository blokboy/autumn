"""Runtime boundary for calling OpenAI's hosted chat completions API.

Direct mirror of `groq_runner.py`'s `generate()` method: an injectable
client (for tests, so no real SDK/network calls are ever needed), the same
`*RuntimeError` shape (`OpenAIRuntimeError` here), and the same
non-streaming-first approach. Deliberately does NOT replicate
`GroqRunner.generate_stream`'s Groq-specific tool-calling machinery (see
`groq_runner.py`'s module docstring and docs/prd/chat-search-tools.md) --
streaming is explicitly out of scope for Anthropic/OpenAI this round (see
docs/prd/multi-provider-models.md, "Future work" #21); `generate` alone is
enough to make OpenAI a real, working chat-completion provider.
"""

from typing import Any, Protocol

import openai

import credentials
from models import ChatMessage


class OpenAIRuntimeError(RuntimeError):
    """Raised when the OpenAI API cannot produce a reply."""


class _ChatCompletions(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
    ) -> Any: ...


class _Chat(Protocol):
    completions: _ChatCompletions


class OpenAIClient(Protocol):
    """The subset of `openai.OpenAI` that `OpenAIRunner` depends on."""

    chat: _Chat


def _to_payload(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [{"role": message.role, "content": message.text} for message in messages]


def _wrap_openai_error(model_name: str, exc: openai.OpenAIError) -> OpenAIRuntimeError:
    detail = getattr(exc, "message", None) or str(exc)
    return OpenAIRuntimeError(f"{model_name} failed: {detail}")


class OpenAIRunner:
    """Calls OpenAI's hosted chat completions endpoint for hosted models, as
    a single blocking response (`generate`). No streaming variant -- see the
    module docstring."""

    def __init__(self, *, client: OpenAIClient | None = None) -> None:
        self._client = (
            client
            if client is not None
            else openai.OpenAI(api_key=credentials.resolve_key("openai", "OPENAI_API_KEY"))
        )

    def generate(self, messages: list[ChatMessage], model_name: str) -> ChatMessage:
        payload = _to_payload(messages)

        try:
            completion = self._client.chat.completions.create(model=model_name, messages=payload)
        except openai.OpenAIError as exc:
            raise _wrap_openai_error(model_name, exc) from exc

        text = completion.choices[0].message.content or ""
        return ChatMessage(role="assistant", text=text.strip(), model=model_name)
