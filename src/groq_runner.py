"""Runtime boundary for calling Groq's hosted chat completions API."""

import json
import threading
from pathlib import Path
from typing import Any, Callable, Protocol

import groq

import credentials
import search_docs
from models import ChatMessage, ToolCitation

GROQ_MODELS = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it")

# Tools offered to Groq on every chat turn (see docs/prd/chat-search-tools.md,
# "Tool-call round"). `search_docs` is always offered -- it's local, free, and
# has no key/opt-in gate. This is the only tool this ticket wires up; later
# tools (search_web, system tools) extend this list.
_TOOLS: list[dict[str, Any]] = [search_docs.TOOL_SCHEMA]

# name -> callable executing that tool's arguments and returning the result
# text to hand back to Groq as a tool-role message.
_TOOL_EXECUTORS: dict[str, Callable[[dict[str, Any], Path | None], str]] = {
    search_docs.TOOL_NAME: lambda arguments, docs_root: search_docs.run_tool(arguments, root=docs_root),
}

# name -> verb phrase for the "<verb> for '<query>'" status text `on_status`
# surfaces while that tool executes (see `_status_text_for_tool_call`).
# Keyed generically by tool name so a later `search_web` tool only needs to
# add one entry here -- no chat_view.py change, since chat_view.py just
# renders whatever string it's given.
_TOOL_STATUS_LABELS: dict[str, str] = {
    search_docs.TOOL_NAME: "Searching docs",
}

# name -> callable returning the citation source labels for a *successful*
# call to that tool, used to build the `ToolCitation` attached to the
# eventual assistant reply. Only consulted on success (see
# `_execute_tool_call`) -- a failed tool call has nothing to cite.
_TOOL_CITATION_EXTRACTORS: dict[str, Callable[[dict[str, Any], Path | None], list[str]]] = {
    search_docs.TOOL_NAME: lambda arguments, docs_root: search_docs.citations_for_tool_call(
        arguments, root=docs_root
    ),
}


def _status_text_for_tool_call(tool_call: Any) -> str | None:
    """Builds the "Searching docs for '...'"-style status text `on_status`
    surfaces while a decided tool call executes (see
    docs/prd/chat-search-tools.md, "Tool-call round"). Generic across tools:
    keyed only off `_TOOL_STATUS_LABELS` (tool name -> verb phrase) and
    whatever `query` argument the model passed, if any -- a later
    `search_web` tool reuses this unchanged by adding one entry to that map,
    with no chat_view.py change required, since chat_view.py only ever
    renders whatever string this function (or `None`, for "no status")
    produces. Returns `None` for a tool with no registered label -- nothing
    offered today falls into that case, but a future tool that doesn't want
    a status line can simply not be added to the map.
    """
    name = tool_call.function.name
    label = _TOOL_STATUS_LABELS.get(name)
    if label is None:
        return None
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        arguments = {}
    query = arguments.get("query") if isinstance(arguments, dict) else None
    if isinstance(query, str) and query:
        return f"{label} for '{query}'"
    return label


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

    def __init__(self, *, client: GroqClient | None = None, docs_root: Path | None = None) -> None:
        self._client = (
            client if client is not None else groq.Groq(api_key=credentials.resolve_key("groq", "GROQ_API_KEY"))
        )
        # Override point for tests; `None` means `search_docs` resolves the
        # real repo root itself (see search_docs._repo_root).
        self._docs_root = docs_root

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
        on_status: Callable[[str], None] | None = None,
        on_citation: Callable[[ToolCitation], None] | None = None,
    ) -> None:
        """Streams a reply token-by-token, calling `on_chunk` with each
        non-empty piece of text as it arrives.

        Runs the single-round tool-calling mechanism from
        docs/prd/chat-search-tools.md ("Tool-call round") ahead of streaming:

        1. A non-streaming "decide" call is sent with `search_docs`'s schema
           in `tools`.
        2. If that response has no `tool_calls`, its text is already the
           final answer -- it's handed to `on_chunk` in one piece (no
           further streaming needed) and this returns. `on_status` and
           `on_citation` are never called on this path -- an answer that
           didn't use a tool gets no status and no citation.
        3. If it has one (or more; only the first is used) `tool_calls`,
           `on_status` (if given) is called once with a human-readable
           "Searching docs for '...'"-style status string (see
           `_status_text_for_tool_call`) before that tool is executed, so a
           caller can show it's not frozen during this non-streaming
           round-trip. The tool is then executed and a second call is made
           with the tool result appended as a tool-role message. A tool
           failure doesn't abort the turn: it's surfaced to `on_chunk` as a
           visible inline notice, and the second call still proceeds with a
           tool-role message describing the failure, so the model can still
           attempt a best-effort, caveated answer -- a failed tool call
           never triggers `on_citation` (nothing was actually grounded).
        4. The second call's response streams via the same chunked mechanism
           `generate_stream` already used before tool-calling existed. No
           `tools` are offered on this second call -- there is no provision
           for the model requesting a further tool call within one turn.
        5. If the tool call succeeded, `on_citation` (if given) is called
           once, after streaming finishes, with a `ToolCitation` describing
           where the answer was grounded (empty `sources` are treated the
           same as "nothing to cite" and don't trigger the call at all).

        Has no return value -- the caller's `on_chunk`/`on_status`/
        `on_citation` closures are the only place accumulated
        text/status/citation live, so on a mid-stream cancellation or error,
        whatever already reached those callbacks is exactly what the caller
        already has; there's nothing else to hand back.

        `cancel_event` (if given) is checked once per received chunk during
        the streaming portion (step 4, and step 2's local `llama.cpp`-style
        streaming doesn't apply here since step 2 has no stream to cancel).
        Once set, iteration stops immediately without processing that
        chunk's content, and the stream is closed client-side rather than
        left to keep pulling further chunks off the wire. This is not
        treated as an error -- cancellation returns normally.

        Errors (including ones that surface mid-iteration, not just at call
        time) are re-raised as `GroqRuntimeError`, same as `generate`.

        `on_status`/`on_citation` are both optional and keyword-only,
        additive to the contract `generate_stream` already had before this
        ticket -- `LocalModelRunner.generate_stream` implements the same
        `on_chunk`/`cancel_event`-only signature it always has and is
        unaffected; only Groq's tool-calling path has anything to report
        through the two new callbacks.
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
        if on_status is not None:
            status_text = _status_text_for_tool_call(tool_call)
            if status_text is not None:
                on_status(status_text)

        tool_round_messages, failure_notice, citation = self._execute_tool_call(tool_call, decision_message)
        if failure_notice:
            on_chunk(failure_notice)

        second_payload = payload + tool_round_messages
        self._stream_completion(second_payload, model_name, on_chunk=on_chunk, cancel_event=cancel_event)

        if citation is not None and on_citation is not None:
            on_citation(citation)

    def _execute_tool_call(
        self, tool_call: Any, decision_message: Any
    ) -> tuple[list[dict[str, Any]], str | None, ToolCitation | None]:
        """Executes a single decided tool call and builds the
        assistant/tool-role message pair the second Groq call needs.

        Returns `(messages_to_append, failure_notice, citation)`.
        `failure_notice` is `None` on success, or a human-readable string
        (meant for `on_chunk`) if the tool raised -- the tool-role message
        content always describes the failure too, so the model's second
        call sees it regardless of whether the caller surfaces
        `failure_notice`. `citation` is a `ToolCitation` when the tool
        succeeded *and* `_TOOL_CITATION_EXTRACTORS` has an entry for it that
        actually returned at least one source; `None` on any failure (a
        failed tool call grounded nothing) or when no citation extractor is
        registered for the tool.
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
        citation: ToolCitation | None = None
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError(f"expected a JSON object, got {type(arguments).__name__}")
        except (json.JSONDecodeError, ValueError) as exc:
            result_text = f"Error: could not parse arguments for {name}: {exc}"
            failure_notice = f"[{name} failed: invalid arguments]\n\n"
        else:
            executor = _TOOL_EXECUTORS.get(name)
            if executor is None:
                result_text = f"Error: unknown tool {name!r}"
                failure_notice = f"[{name} failed: unknown tool]\n\n"
            else:
                try:
                    result_text = executor(arguments, self._docs_root)
                except Exception as exc:
                    result_text = f"Error: {name} failed: {exc}"
                    failure_notice = f"[{name} failed: {exc}]\n\n"
                else:
                    citation = self._citation_for_successful_call(name, arguments)

        tool_entry = {"role": "tool", "tool_call_id": tool_call.id, "content": result_text}
        return [assistant_entry, tool_entry], failure_notice, citation

    def _citation_for_successful_call(self, name: str, arguments: dict[str, Any]) -> ToolCitation | None:
        """Best-effort citation lookup for a tool call that already
        succeeded. Failure here (a bug in an extractor, an unreadable file
        that changed between the executor call and now, etc.) shouldn't
        turn an otherwise-successful tool round into a visible failure --
        it just means no citation gets attached, same as when no extractor
        is registered for `name` at all."""
        extractor = _TOOL_CITATION_EXTRACTORS.get(name)
        if extractor is None:
            return None
        try:
            sources = extractor(arguments, self._docs_root)
        except Exception:
            return None
        if not sources:
            return None
        return ToolCitation(tool=name, sources=sources)

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
