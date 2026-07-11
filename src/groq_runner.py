"""Runtime boundary for calling Groq's hosted chat completions API."""

import json
import threading
from pathlib import Path
from typing import Any, Callable, Protocol

import groq

import credentials
import list_keys
import list_models
import list_runs
import paths
import search_docs
import system_tools
from models import ChatMessage

GROQ_MODELS = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it")

# A synchronous confirmation prompt: given a human-readable message and a
# delay (seconds the confirm action should stay disabled -- 0 for none),
# returns whether the user confirmed. Called from whatever thread
# `generate_stream` runs on (see `app.py`'s `_run_groq_stream`, which runs on
# a background thread) -- the real implementation bridges to the Textual
# main thread and blocks until `ConfirmScreen` resolves; see
# `AutumnApp._confirm_from_thread`. `None` (the default) means no
# confirmation surface is wired up, in which case every mutating tool call
# is treated as declined -- silently allowing a mutation with no way to ask
# the user would be the wrong default.
ConfirmFn = Callable[[str, float], bool]

# Tools offered to Groq on every chat turn (see docs/prd/chat-search-tools.md,
# "Tool-call round", and docs/prd/chat-cli-parity-tools.md for the read-only
# and mutating rows). The four read-only tools (`search_docs`, `list_models`,
# `list_keys`, `list_runs`) need no confirmation and are always offered. The
# five mutating tools from `system_tools.py` are always offered too, but each
# one is confirmation-gated (see `_execute_tool_call`/`_execute_mutating_tool`)
# rather than executing immediately like the read-only ones.
_TOOLS: list[dict[str, Any]] = [
    search_docs.TOOL_SCHEMA,
    list_models.TOOL_SCHEMA,
    list_keys.TOOL_SCHEMA,
    list_runs.TOOL_SCHEMA,
    *[tool.schema for tool in system_tools.MUTATING_TOOLS],
]

_MUTATING_TOOLS_BY_NAME: dict[str, system_tools.MutatingTool] = {
    tool.name: tool for tool in system_tools.MUTATING_TOOLS
}


class GroqRuntimeError(RuntimeError):
    """Raised when the Groq API cannot produce a reply."""


class _ChatCompletions(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> Any: ...


class _Chat(Protocol):
    completions: _ChatCompletions


class GroqClient(Protocol):
    """The subset of `groq.Groq` that `GroqRunner` depends on."""

    chat: _Chat


def _to_payload(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [{"role": message.role, "content": message.text} for message in messages]


def _wrap_groq_error(model_name: str, exc: groq.GroqError) -> GroqRuntimeError:
    detail = getattr(exc, "message", None) or str(exc)
    return GroqRuntimeError(f"{model_name} failed: {detail}")


class GroqRunner:
    """Calls Groq's hosted chat completions endpoint for hosted models, either
    as a single blocking response (`generate`) or token-by-token
    (`generate_stream`)."""

    def __init__(
        self,
        *,
        client: GroqClient | None = None,
        docs_root: Path | None = None,
        catalog_root: Path | None = None,
        runs_root: Path | None = None,
        confirm: ConfirmFn | None = None,
    ) -> None:
        self._client = (
            client if client is not None else groq.Groq(api_key=credentials.resolve_key("groq", "GROQ_API_KEY"))
        )
        # Override point for tests; `None` means `search_docs` resolves the
        # real repo root itself (see search_docs._repo_root).
        self._docs_root = docs_root
        # Unlike `docs_root`, the system tools (`list_models`/`list_runs`/the
        # mutating catalog tools) have no self-resolving fallback of their
        # own -- so the real defaults are resolved here, once, rather than
        # threaded through as `None` on every call.
        self._catalog_root = catalog_root if catalog_root is not None else paths.models_root()
        self._runs_root = runs_root if runs_root is not None else paths.default_runs_root()
        self._confirm = confirm

        # name -> callable executing that tool's arguments and returning the
        # result text to hand back to Groq as a tool-role message. Built here
        # (not module-level) so each executor closes over exactly the context
        # it needs -- e.g. `list_models`'s `catalog_root` -- rather than every
        # tool executor sharing one fixed extra-parameter signature. Only
        # read-only tools live here; mutating tools go through
        # `_MUTATING_TOOLS_BY_NAME`/`_execute_mutating_tool` instead, since
        # they need a confirmation step this dict alone can't express.
        self._tool_executors: dict[str, Callable[[dict[str, Any]], str]] = {
            search_docs.TOOL_NAME: lambda arguments: search_docs.run_tool(arguments, root=self._docs_root),
            list_models.TOOL_NAME: lambda arguments: list_models.run_tool(
                arguments, catalog_root=self._catalog_root
            ),
            list_keys.TOOL_NAME: list_keys.run_tool,
            list_runs.TOOL_NAME: lambda arguments: list_runs.run_tool(arguments, runs_root=self._runs_root),
        }

    def generate(self, messages: list[ChatMessage], model_name: str) -> ChatMessage:
        payload = _to_payload(messages)

        try:
            completion = self._client.chat.completions.create(model=model_name, messages=payload)
        except groq.GroqError as exc:
            raise _wrap_groq_error(model_name, exc) from exc

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

        Runs the single-round tool-calling mechanism from
        docs/prd/chat-search-tools.md ("Tool-call round") ahead of streaming:

        1. A non-streaming "decide" call is sent with every tool's schema
           (`_TOOLS`: the read-only `search_docs`/`list_models`/`list_keys`/
           `list_runs`, plus the mutating system tools from `system_tools.py`)
           in `tools`.
        2. If that response has no `tool_calls`, its text is already the
           final answer -- it's handed to `on_chunk` in one piece (no
           further streaming needed) and this returns.
        3. If it has one (or more; only the first is used) `tool_calls`, that
           tool is executed and a second call is made with the tool result
           appended as a tool-role message. A mutating tool is confirmed
           first (see `_execute_mutating_tool`); a read-only tool executes
           immediately. A tool failure doesn't abort the turn: it's surfaced
           to `on_chunk` as a visible inline notice, and the second call
           still proceeds with a tool-role message describing the failure,
           so the model can still attempt a best-effort, caveated answer.
        4. The second call's response streams via the same chunked mechanism
           `generate_stream` already used before tool-calling existed. No
           `tools` are offered on this second call -- there is no provision
           for the model requesting a further tool call within one turn.

        Has no return value -- the caller's `on_chunk` closure is the only
        place accumulated text lives, so on a mid-stream cancellation or
        error, whatever text already reached `on_chunk` is exactly what the
        caller already has; there's nothing else to hand back.

        `cancel_event` (if given) is checked once per received chunk during
        the streaming portion (step 4, and step 2's local `llama.cpp`-style
        streaming doesn't apply here since step 2 has no stream to cancel).
        Once set, iteration stops immediately without processing that
        chunk's content, and the stream is closed client-side rather than
        left to keep pulling further chunks off the wire. This is not
        treated as an error -- cancellation returns normally.

        Errors (including ones that surface mid-iteration, not just at call
        time) are re-raised as `GroqRuntimeError`, same as `generate`.
        """
        payload = _to_payload(messages)

        try:
            decision = self._client.chat.completions.create(model=model_name, messages=payload, tools=_TOOLS)
        except groq.GroqError as exc:
            raise _wrap_groq_error(model_name, exc) from exc

        decision_message = decision.choices[0].message
        tool_calls = getattr(decision_message, "tool_calls", None) or []

        if not tool_calls:
            text = decision_message.content or ""
            if text:
                on_chunk(text)
            return

        tool_call = tool_calls[0]
        tool_round_messages, failure_notice = self._execute_tool_call(tool_call, decision_message)
        if failure_notice:
            on_chunk(failure_notice)

        second_payload = payload + tool_round_messages
        self._stream_completion(second_payload, model_name, on_chunk=on_chunk, cancel_event=cancel_event)

    def _execute_tool_call(self, tool_call: Any, decision_message: Any) -> tuple[list[dict[str, Any]], str | None]:
        """Executes a single decided tool call and builds the
        assistant/tool-role message pair the second Groq call needs.

        Returns `(messages_to_append, failure_notice)`. `failure_notice` is
        `None` on success, or a human-readable string (meant for `on_chunk`)
        if the tool raised -- the tool-role message content always describes
        the failure too, so the model's second call sees it regardless of
        whether the caller surfaces `failure_notice`.
        """
        name = tool_call.function.name
        raw_arguments = tool_call.function.arguments or "{}"

        assistant_entry = {
            "role": "assistant",
            "content": decision_message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": name, "arguments": raw_arguments},
                }
            ],
        }

        failure_notice: str | None = None
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError(f"expected a JSON object, got {type(arguments).__name__}")
        except (json.JSONDecodeError, ValueError) as exc:
            result_text = f"Error: could not parse arguments for {name}: {exc}"
            failure_notice = f"[{name} failed: invalid arguments]\n\n"
        else:
            mutating_tool = _MUTATING_TOOLS_BY_NAME.get(name)
            if mutating_tool is not None:
                result_text, failure_notice = self._execute_mutating_tool(mutating_tool, arguments)
            else:
                executor = self._tool_executors.get(name)
                if executor is None:
                    result_text = f"Error: unknown tool {name!r}"
                    failure_notice = f"[{name} failed: unknown tool]\n\n"
                else:
                    try:
                        result_text = executor(arguments)
                    except Exception as exc:
                        result_text = f"Error: {name} failed: {exc}"
                        failure_notice = f"[{name} failed: {exc}]\n\n"

        tool_entry = {"role": "tool", "tool_call_id": tool_call.id, "content": result_text}
        return [assistant_entry, tool_entry], failure_notice

    def _execute_mutating_tool(
        self, tool: system_tools.MutatingTool, arguments: dict[str, Any]
    ) -> tuple[str, str | None]:
        """Confirms, then executes, a mutating system tool -- always
        confirmation-gated (see docs/prd/chat-cli-parity-tools.md; there is
        no autonomous-mode bypass yet, that's a later ticket). Returns
        `(result_text, failure_notice)`, same shape `_execute_tool_call`
        uses for the read-only tools.

        A decline produces no `failure_notice` -- it isn't an error, it's
        the user's choice, and the model's final answer is expected to
        acknowledge it (via the tool-result content) rather than retry or
        proceed as if it happened. An actual execution failure (invalid
        target, `SystemToolError`, etc.) does produce a `failure_notice`,
        same visible-degrade contract the read-only tools' failures already
        use.

        No confirmation surface wired up (`self._confirm is None`) is
        treated the same as an explicit decline -- silently allowing a
        mutation with no way to ask the user would be the wrong default.
        """
        message = tool.confirmation_message(arguments)
        delay = 1.5 if tool.destructive else 0.0
        confirmed = self._confirm(message, delay) if self._confirm is not None else False
        if not confirmed:
            return f"Declined by user: {tool.name} was not performed.", None
        try:
            return system_tools.run_tool(tool.name, arguments, catalog_root=self._catalog_root), None
        except Exception as exc:
            return f"Error: {tool.name} failed: {exc}", f"[{tool.name} failed: {exc}]\n\n"

    def _stream_completion(
        self,
        payload: list[dict[str, Any]],
        model_name: str,
        *,
        on_chunk: Callable[[str], None],
        cancel_event: threading.Event | None,
    ) -> None:
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
            raise _wrap_groq_error(model_name, exc) from exc
