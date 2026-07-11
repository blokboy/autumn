"""Runtime boundary for calling Groq's hosted chat completions API."""

import os
from typing import Any, Protocol

import groq

from autumn.models import ChatMessage

GROQ_MODELS = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it")


class GroqRuntimeError(RuntimeError):
    """Raised when the Groq API cannot produce a reply."""


class _ChatCompletions(Protocol):
    def create(self, *, model: str, messages: list[dict[str, str]]) -> Any: ...


class _Chat(Protocol):
    completions: _ChatCompletions


class GroqClient(Protocol):
    """The subset of `groq.Groq` that `GroqRunner` depends on."""

    chat: _Chat


class GroqRunner:
    """Calls Groq's non-streaming chat completions endpoint for hosted models."""

    def __init__(self, *, client: GroqClient | None = None) -> None:
        self._client = client if client is not None else groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def generate(self, messages: list[ChatMessage], model_name: str) -> ChatMessage:
        payload = [{"role": message.role, "content": message.text} for message in messages]

        try:
            completion = self._client.chat.completions.create(model=model_name, messages=payload)
        except groq.GroqError as exc:
            detail = getattr(exc, "message", None) or str(exc)
            raise GroqRuntimeError(f"{model_name} failed: {detail}") from exc

        text = completion.choices[0].message.content or ""
        return ChatMessage(role="assistant", text=text.strip(), model=model_name)
