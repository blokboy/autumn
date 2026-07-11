"""Runtime boundary for calling Groq's hosted chat completions API."""

import os
import threading
from typing import Any, Callable, Protocol

import groq

from autumn.models import ChatMessage

GROQ_MODELS = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it")


class GroqRuntimeError(RuntimeError):
    """Raised when the Groq API cannot produce a reply."""


class _ChatCompletions(Protocol):
    def create(self, *, model: str, messages: list[dict[str, str]], stream: bool = False) -> Any: ...


class _Chat(Protocol):
    completions: _ChatCompletions


class GroqClient(Protocol):
    """The subset of `groq.Groq` that `GroqRunner` depends on."""

    chat: _Chat


class GroqRunner:
    """Calls Groq's hosted chat completions endpoint for hosted models, either
    as a single blocking response (`generate`) or token-by-token
    (`generate_stream`)."""

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

    def generate_stream(
        self,
        messages: list[ChatMessage],
        model_name: str,
        *,
        on_chunk: Callable[[str], None],
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Streams a reply token-by-token, calling `on_chunk` with each
        non-empty piece of text as it arrives.

        Has no return value -- the caller's `on_chunk` closure is the only
        place accumulated text lives, so on a mid-stream cancellation or
        error, whatever text already reached `on_chunk` is exactly what the
        caller already has; there's nothing else to hand back.

        `cancel_event` (if given) is checked once per received chunk. Once
        set, iteration stops immediately without processing that chunk's
        content, and the stream is closed client-side rather than left to
        keep pulling further chunks off the wire. This is not treated as an
        error -- cancellation returns normally.

        Errors (including ones that surface mid-iteration, not just at call
        time) are re-raised as `GroqRuntimeError`, same as `generate`.
        """
        payload = [{"role": message.role, "content": message.text} for message in messages]

        try:
            stream = self._client.chat.completions.create(model=model_name, messages=payload, stream=True)
            for chunk in stream:
                if cancel_event is not None and cancel_event.is_set():
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                    return
                delta = chunk.choices[0].delta.content
                if delta:
                    on_chunk(delta)
        except groq.GroqError as exc:
            detail = getattr(exc, "message", None) or str(exc)
            raise GroqRuntimeError(f"{model_name} failed: {detail}") from exc
