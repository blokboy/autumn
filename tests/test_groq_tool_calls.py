"""Behavioral tests for GroqRunner's single-round tool-calling mechanism
(docs/prd/chat-search-tools.md, "Tool-call round"): a non-streaming
decide call with `search_docs`'s schema in `tools`, then either treating a
tool-call-free response as the final answer, or executing exactly one tool
call and streaming a second call's response -- with a tool failure surfaced
visibly rather than aborting the turn.

Follows the same Protocol-based fake `GroqClient`/`_ChatCompletions`
convention as test_groq_runner.py/test_groq_streaming.py instead of mocking
the real SDK.
"""

import json
import threading
from dataclasses import dataclass

import groq
import httpx
import pytest
from textual.widgets import Input, Static

import search_docs
import search_web
from app import AutumnApp
from groq_runner import _TOOLS, GroqRunner, GroqRuntimeError
from models import ChatMessage, PromptRoutingPolicy, ProviderAccount, ProviderModel, ToolCitation
from widgets.command_bar import CommandBar


# --- Fakes ---------------------------------------------------------------


@dataclass
class _FakeFunctionCall:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunctionCall
    type: str = "function"


@dataclass
class _FakeDecisionMessage:
    content: str | None
    tool_calls: list[_FakeToolCall] | None = None


@dataclass
class _FakeDecisionChoice:
    message: _FakeDecisionMessage


@dataclass
class _FakeDecision:
    choices: list[_FakeDecisionChoice]


@dataclass
class _FakeDelta:
    content: str | None


@dataclass
class _FakeStreamChoice:
    delta: _FakeDelta


@dataclass
class _FakeChunk:
    choices: list[_FakeStreamChoice]


def _chunk(content: str | None) -> _FakeChunk:
    return _FakeChunk(choices=[_FakeStreamChoice(delta=_FakeDelta(content=content))])


class _GatedFakeStream:
    """Like `_FakeStream`, but blocks before yielding each chunk until
    `release_next()` is called -- lets a test pause mid-stream and inspect
    UI state (e.g. the tool-call status widget) deterministically instead
    of racing real wall-clock timing. Same idiom as test_groq_streaming.py's
    `_ScriptedGroqRunner`, one level lower (a fake stream rather than a
    fake whole runner) so it works through a real `GroqRunner`."""

    def __init__(self, chunks: list[_FakeChunk]):
        self._chunks = chunks
        self._gates = [threading.Event() for _ in chunks]
        self._released = 0
        self.closed = False

    def release_next(self) -> None:
        self._gates[self._released].set()
        self._released += 1

    def __iter__(self):
        for chunk, gate in zip(self._chunks, self._gates):
            gate.wait(timeout=5)
            yield chunk

    def close(self) -> None:
        self.closed = True


class _FakeStream:
    """Mimics the iterable `groq.Stream` returned by
    `client.chat.completions.create(..., stream=True)`."""

    def __init__(self, chunks: list[_FakeChunk], error: Exception | None = None):
        self._chunks = chunks
        self._error = error
        self.closed = False

    def __iter__(self):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.closed = True


class _ToolCallingCompletions:
    """A fake `chat.completions` that answers a non-streaming "decide" call
    with `decision` and a `stream=True` call with `stream_response`, so a
    single fake client can play both roles a tool-calling round needs."""

    def __init__(self, decision: _FakeDecision, stream_response: _FakeStream | None, seen_calls: list[dict]):
        self._decision = decision
        self._stream_response = stream_response
        self._seen_calls = seen_calls

    def create(self, *, model: str, messages: list[dict], tools=None, stream: bool = False):
        self._seen_calls.append({"model": model, "messages": messages, "tools": tools, "stream": stream})
        if stream:
            assert self._stream_response is not None, "no stream_response configured for a stream=True call"
            return self._stream_response
        return self._decision


class _FakeChat:
    def __init__(self, completions: _ToolCallingCompletions):
        self.completions = completions


class _ToolCallingGroqClient:
    def __init__(self, decision: _FakeDecision, stream_response: _FakeStream | None = None):
        self.seen_calls: list[dict] = []
        self.chat = _FakeChat(_ToolCallingCompletions(decision, stream_response, self.seen_calls))


class _ToolUseFailedThenStreamingCompletions:
    def __init__(self, failed_generation: str, stream_response: _FakeStream, seen_calls: list[dict]):
        self._failed_generation = failed_generation
        self._stream_response = stream_response
        self._seen_calls = seen_calls

    def create(self, *, model: str, messages: list[dict], tools=None, stream: bool = False):
        self._seen_calls.append({"model": model, "messages": messages, "tools": tools, "stream": stream})
        if stream:
            return self._stream_response
        body = {
            "error": {
                "message": "Failed to call a function. Please adjust your prompt.",
                "type": "invalid_request_error",
                "code": "tool_use_failed",
                "failed_generation": self._failed_generation,
            }
        }
        response = httpx.Response(
            400,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
            json=body,
        )
        raise groq.BadRequestError(f"Error code: 400 - {body!r}", response=response, body=body)


class _ToolUseFailedThenStreamingGroqClient:
    def __init__(self, failed_generation: str, stream_response: _FakeStream):
        self.seen_calls: list[dict] = []
        self.chat = _FakeChat(
            _ToolUseFailedThenStreamingCompletions(failed_generation, stream_response, self.seen_calls)
        )


def _no_tool_call_decision(text: str) -> _FakeDecision:
    return _FakeDecision(choices=[_FakeDecisionChoice(message=_FakeDecisionMessage(content=text))])


def _search_docs_tool_call_decision(query: str, *, call_id: str = "call_1") -> _FakeDecision:
    return _FakeDecision(
        choices=[
            _FakeDecisionChoice(
                message=_FakeDecisionMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id=call_id,
                            function=_FakeFunctionCall(
                                name=search_docs.TOOL_NAME, arguments=json.dumps({"query": query})
                            ),
                        )
                    ],
                )
            )
        ]
    )


def _write_docs_corpus(root) -> None:
    (root / "docs").mkdir()
    (root / "README.md").write_text(
        "# autumn\n"
        "\n"
        "## Usage\n"
        "\n"
        "```bash\n"
        "autumn run <script.py> --dry-run\n"
        "```\n"
        "`--dry-run` replays a scripted, dependency-free sequence of "
        "optimization events instead of running the script for real.\n"
    )


# --- No tool call: response is final, no streaming needed ----------------


def test_generate_stream_with_no_tool_call_renders_full_text_in_one_chunk(tmp_path):
    decision = _no_tool_call_decision("Autumn is a GEPA dashboard CLI.")
    client = _ToolCallingGroqClient(decision)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="what is autumn")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
    )

    assert received == ["Autumn is a GEPA dashboard CLI."]
    # Exactly one call was made -- no second (streaming) call when there's no
    # tool call to execute first.
    assert len(client.seen_calls) == 1
    assert client.seen_calls[0]["stream"] is False
    assert client.seen_calls[0]["tools"] == _TOOLS
    assert search_docs.TOOL_SCHEMA in _TOOLS
    assert client.seen_calls[0]["messages"] == [{"role": "user", "content": "what is autumn"}]


def test_generate_stream_sends_search_docs_schema_on_every_decide_call(tmp_path):
    decision = _no_tool_call_decision("hi")
    client = _ToolCallingGroqClient(decision)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    runner.generate_stream([ChatMessage(role="user", text="hi")], "llama-3.1-8b-instant", on_chunk=lambda _: None)

    [call] = client.seen_calls
    assert call["tools"] == _TOOLS
    assert search_docs.TOOL_SCHEMA in _TOOLS


def test_generate_stream_recovers_search_docs_call_from_groq_tool_use_failed(tmp_path):
    _write_docs_corpus(tmp_path)
    stream_response = _FakeStream([_chunk("Autumn is the dashboard CLI for GEPA runs.")])
    client = _ToolUseFailedThenStreamingGroqClient(
        '<function=search_docs{"query": "Autumn"}</function>',
        stream_response,
    )
    runner = GroqRunner(client=client, docs_root=tmp_path)

    received: list[str] = []
    statuses: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="Can you explain Autumn?")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
        on_status=statuses.append,
    )

    assert received == ["Autumn is the dashboard CLI for GEPA runs."]
    assert statuses == ["Searching docs for 'Autumn'"]
    assert len(client.seen_calls) == 2
    assert client.seen_calls[0]["tools"] == _TOOLS
    assert client.seen_calls[1]["stream"] is True
    second_messages = client.seen_calls[1]["messages"]
    assert second_messages[1]["tool_calls"][0]["function"] == {
        "name": search_docs.TOOL_NAME,
        "arguments": '{"query": "Autumn"}',
    }
    assert second_messages[2]["role"] == "tool"
    assert "README.md" in second_messages[2]["content"]


def test_generate_stream_treats_unknown_failed_generation_tool_as_plain_response(tmp_path):
    stream_response = _FakeStream([_chunk("Autumn is a CLI for working with GEPA runs.")])
    client = _ToolUseFailedThenStreamingGroqClient(
        '<function=explain_autumn{"topic": "Autumn"}</function>',
        stream_response,
    )
    runner = GroqRunner(client=client, docs_root=tmp_path)

    received: list[str] = []
    statuses: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="Can you explain Autumn?")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
        on_status=statuses.append,
    )

    assert received == ["Autumn is a CLI for working with GEPA runs."]
    assert statuses == []
    assert len(client.seen_calls) == 2
    assert client.seen_calls[1]["stream"] is True
    assert client.seen_calls[1]["tools"] is None
    assert client.seen_calls[1]["messages"] == [{"role": "user", "content": "Can you explain Autumn?"}]


def test_search_web_schema_is_not_offered_without_tavily_key_even_with_trigger(tmp_path, monkeypatch):
    monkeypatch.setattr("groq_runner.credentials.resolve_key", lambda provider, env_var: None)
    decision = _no_tool_call_decision("hi")
    client = _ToolCallingGroqClient(decision)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    runner.generate_stream(
        [ChatMessage(role="user", text="/search latest autumn news")],
        "llama-3.1-8b-instant",
        on_chunk=lambda _: None,
    )

    [call] = client.seen_calls
    offered_names = {tool["function"]["name"] for tool in call["tools"]}
    assert search_web.TOOL_NAME not in offered_names
    assert call["messages"] == [{"role": "user", "content": "/search latest autumn news"}]


def test_search_web_schema_is_not_offered_for_plain_chat_in_explicit_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "groq_runner.credentials.resolve_key",
        lambda provider, env_var: "tvly-test-key" if provider == "tavily" else None,
    )
    decision = _no_tool_call_decision("hi")
    client = _ToolCallingGroqClient(decision)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    runner.generate_stream(
        [ChatMessage(role="user", text="latest autumn news")],
        "llama-3.1-8b-instant",
        on_chunk=lambda _: None,
    )

    [call] = client.seen_calls
    offered_names = {tool["function"]["name"] for tool in call["tools"]}
    assert search_web.TOOL_NAME not in offered_names


def test_search_web_schema_is_offered_for_explicit_trigger_and_payload_strips_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "groq_runner.credentials.resolve_key",
        lambda provider, env_var: "tvly-test-key" if provider == "tavily" else None,
    )
    decision = _no_tool_call_decision("hi")
    client = _ToolCallingGroqClient(decision)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    runner.generate_stream(
        [ChatMessage(role="user", text="/search latest autumn news")],
        "llama-3.1-8b-instant",
        on_chunk=lambda _: None,
    )

    [call] = client.seen_calls
    offered_names = {tool["function"]["name"] for tool in call["tools"]}
    assert search_web.TOOL_NAME in offered_names
    assert call["messages"] == [{"role": "user", "content": "latest autumn news"}]


def test_search_web_schema_is_offered_for_plain_chat_in_autonomous_mode(tmp_path, monkeypatch):
    import config

    config.set_search_mode("autonomous")
    monkeypatch.setattr(
        "groq_runner.credentials.resolve_key",
        lambda provider, env_var: "tvly-test-key" if provider == "tavily" else None,
    )
    decision = _no_tool_call_decision("hi")
    client = _ToolCallingGroqClient(decision)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    runner.generate_stream(
        [ChatMessage(role="user", text="latest autumn news")],
        "llama-3.1-8b-instant",
        on_chunk=lambda _: None,
    )

    [call] = client.seen_calls
    offered_names = {tool["function"]["name"] for tool in call["tools"]}
    assert search_web.TOOL_NAME in offered_names


# --- One tool call: search_docs executes, second call streams ------------


def test_generate_stream_with_one_tool_call_executes_search_docs_and_streams_second_call(tmp_path):
    _write_docs_corpus(tmp_path)
    decision = _search_docs_tool_call_decision("what does --dry-run do")
    stream_response = _FakeStream([_chunk("`--dry-run`"), _chunk(" skips the real run.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="what does --dry-run do")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
    )

    # Final answer streamed in, no failure notice prepended.
    assert received == ["`--dry-run`", " skips the real run."]

    assert len(client.seen_calls) == 2
    decide_call, second_call = client.seen_calls
    assert decide_call["stream"] is False
    assert second_call["stream"] is True
    # No tools offered on the second call -- no provision for a further tool
    # call within one turn.
    assert second_call["tools"] is None

    # The second call's messages carry the assistant's tool-call and the
    # tool's (real, grounded) result.
    assistant_entry, tool_entry = second_call["messages"][1], second_call["messages"][2]
    assert assistant_entry["role"] == "assistant"
    assert assistant_entry["tool_calls"][0]["id"] == "call_1"
    assert assistant_entry["tool_calls"][0]["function"]["name"] == "search_docs"
    assert tool_entry["role"] == "tool"
    assert tool_entry["tool_call_id"] == "call_1"
    assert "--dry-run" in tool_entry["content"]
    assert "README.md" in tool_entry["content"]


def test_generate_stream_with_search_web_tool_call_streams_second_call_and_cites_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "groq_runner.credentials.resolve_key",
        lambda provider, env_var: "tvly-test-key" if provider == "tavily" else None,
    )
    monkeypatch.setattr(search_web, "run_tool", lambda arguments: "1. Result\nURL: https://example.com")
    monkeypatch.setattr(search_web, "citations_for_tool_call", lambda arguments: ["https://example.com"])

    decision = _FakeDecision(
        choices=[
            _FakeDecisionChoice(
                message=_FakeDecisionMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id="call_web",
                            function=_FakeFunctionCall(
                                name=search_web.TOOL_NAME,
                                arguments=json.dumps({"query": "latest autumn news"}),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    stream_response = _FakeStream([_chunk("According to the result.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    received: list[str] = []
    citations: list[ToolCitation] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="/search latest autumn news")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
        on_citation=citations.append,
    )

    assert received == ["According to the result."]
    tool_entry = client.seen_calls[1]["messages"][2]
    assert tool_entry["role"] == "tool"
    assert tool_entry["content"] == "1. Result\nURL: https://example.com"
    assert citations == [ToolCitation(tool=search_web.TOOL_NAME, sources=["https://example.com"])]


def test_generate_stream_search_web_failure_is_surfaced_and_turn_still_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "groq_runner.credentials.resolve_key",
        lambda provider, env_var: "tvly-test-key" if provider == "tavily" else None,
    )

    def _boom(arguments):
        raise search_web.SearchWebError("Tavily search timed out after 10 seconds.")

    monkeypatch.setattr(search_web, "run_tool", _boom)
    decision = _FakeDecision(
        choices=[
            _FakeDecisionChoice(
                message=_FakeDecisionMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id="call_web",
                            function=_FakeFunctionCall(
                                name=search_web.TOOL_NAME,
                                arguments=json.dumps({"query": "latest autumn news"}),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    stream_response = _FakeStream([_chunk("I can still answer cautiously.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="/search latest autumn news")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
    )

    assert received[0] == "[search_web failed: Tavily search timed out after 10 seconds.]\n\n"
    assert received[1:] == ["I can still answer cautiously."]


def test_generate_stream_tool_call_respects_cancel_event_on_second_call(tmp_path):
    import threading

    _write_docs_corpus(tmp_path)
    decision = _search_docs_tool_call_decision("dry-run")
    stream_response = _FakeStream([_chunk("Hello"), _chunk(" world"), _chunk("!")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    cancel_event = threading.Event()
    received: list[str] = []

    def on_chunk(text: str) -> None:
        received.append(text)
        if text == "Hello":
            cancel_event.set()

    runner.generate_stream(
        [ChatMessage(role="user", text="dry-run")],
        "llama-3.1-8b-instant",
        on_chunk=on_chunk,
        cancel_event=cancel_event,
    )

    assert received == ["Hello"]
    assert stream_response.closed is True


def test_generate_stream_tool_call_second_call_mid_stream_error_raises_runtime_error(tmp_path):
    _write_docs_corpus(tmp_path)
    decision = _search_docs_tool_call_decision("dry-run")
    request = groq._base_client.httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error = groq.APIConnectionError(message="Connection error.", request=request)
    stream_response = _FakeStream([_chunk("Hello")], error=error)
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    received: list[str] = []
    with pytest.raises(GroqRuntimeError, match="llama-3.1-8b-instant failed: Connection error."):
        runner.generate_stream(
            [ChatMessage(role="user", text="dry-run")],
            "llama-3.1-8b-instant",
            on_chunk=received.append,
        )

    assert received == ["Hello"]


def test_generate_stream_decide_call_failure_raises_runtime_error_without_executing_any_tool(tmp_path):
    class _FailingDecideCompletions:
        def create(self, *, model, messages, tools=None, stream=False):
            request = groq._base_client.httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            raise groq.APIConnectionError(message="Connection error.", request=request)

    class _FailingChat:
        completions = _FailingDecideCompletions()

    class _FailingClient:
        chat = _FailingChat()

    runner = GroqRunner(client=_FailingClient(), docs_root=tmp_path)

    with pytest.raises(GroqRuntimeError, match="llama-3.1-8b-instant failed: Connection error."):
        runner.generate_stream(
            [ChatMessage(role="user", text="hi")], "llama-3.1-8b-instant", on_chunk=lambda _: None
        )


# --- Tool failure: visible in transcript, turn still completes -----------


def test_generate_stream_tool_failure_is_surfaced_and_turn_still_completes(tmp_path, monkeypatch):
    """docs unreadable (or any other search_docs failure): the failure is
    handed to on_chunk as a visible notice, but the second Groq call still
    happens and its streamed answer still reaches on_chunk -- no hard abort,
    no silent swallow."""

    def _boom(arguments, root=None):
        raise search_docs.SearchDocsError("could not read docs/README.md: [Errno 2] No such file or directory")

    monkeypatch.setattr(search_docs, "run_tool", _boom)

    decision = _search_docs_tool_call_decision("dry-run")
    stream_response = _FakeStream([_chunk("I don't have the docs, but "), _chunk("here's what I know.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="dry-run")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
    )

    # First chunk is the visible failure notice; the model's best-effort,
    # caveated answer still streams in right after it.
    assert received[0].startswith("[search_docs failed:")
    assert "could not read docs/README.md" in received[0]
    assert received[1:] == ["I don't have the docs, but ", "here's what I know."]

    # The second call still happened -- turn wasn't aborted -- and its
    # tool-role message tells the model the tool failed, so its answer can
    # be genuinely caveated rather than pretending nothing went wrong.
    [_decide_call, second_call] = client.seen_calls
    tool_entry = second_call["messages"][2]
    assert tool_entry["role"] == "tool"
    assert "Error" in tool_entry["content"]
    assert "could not read docs/README.md" in tool_entry["content"]


def test_generate_stream_unknown_tool_name_is_surfaced_as_a_failure_too(tmp_path):
    """Defensive: only search_docs is offered, but if a model somehow
    returns a tool name we don't recognize, that's handled the same
    visible-failure way rather than crashing the turn."""
    decision = _FakeDecision(
        choices=[
            _FakeDecisionChoice(
                message=_FakeDecisionMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id="call_9", function=_FakeFunctionCall(name="made_up_tool", arguments="{}")
                        )
                    ],
                )
            )
        ]
    )
    stream_response = _FakeStream([_chunk("best effort answer")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="hi")], "llama-3.3-70b-versatile", on_chunk=received.append
    )

    assert received[0] == "[made_up_tool failed: unknown tool]\n\n"
    assert received[1:] == ["best effort answer"]


# --- on_status / on_citation: the #26 status + citation contract ---------


def test_generate_stream_calls_on_status_before_the_tool_executes(tmp_path):
    """`on_status` fires once, with a "Searching docs for '<query>'" string
    built from the decided tool call's own query argument, before the
    second (streaming) call is even made -- so a caller has something to
    show for the whole non-streaming decide/execute gap, not just the tail
    end of it."""
    _write_docs_corpus(tmp_path)
    decision = _search_docs_tool_call_decision("what does --dry-run do")
    stream_response = _FakeStream([_chunk("answer")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    events: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="what does --dry-run do")],
        "llama-3.3-70b-versatile",
        on_chunk=lambda text: events.append(f"chunk:{text}"),
        on_status=lambda text: events.append(f"status:{text}"),
    )

    assert events[0] == "status:Searching docs for 'what does --dry-run do'"
    assert events[1:] == ["chunk:answer"]


def test_generate_stream_no_tool_call_never_invokes_on_status_or_on_citation(tmp_path):
    """An answer that didn't use a tool gets neither a status nor a
    citation -- both callbacks stay untouched on this path."""
    decision = _no_tool_call_decision("Autumn is a GEPA dashboard CLI.")
    client = _ToolCallingGroqClient(decision)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    status_calls: list[str] = []
    citation_calls: list[ToolCitation] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="what is autumn")],
        "llama-3.3-70b-versatile",
        on_chunk=lambda _: None,
        on_status=status_calls.append,
        on_citation=citation_calls.append,
    )

    assert status_calls == []
    assert citation_calls == []


def test_generate_stream_tool_success_invokes_on_citation_after_streaming_completes(tmp_path):
    """A successful search_docs call attaches a ToolCitation naming the
    grounding doc/section, delivered via `on_citation` only after the
    second call's stream has already fully delivered its chunks -- so a
    caller can render it "under" the finished answer."""
    _write_docs_corpus(tmp_path)
    decision = _search_docs_tool_call_decision("what does --dry-run do")
    stream_response = _FakeStream([_chunk("`--dry-run`"), _chunk(" skips the real run.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    events: list[str] = []
    citations: list[ToolCitation] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="what does --dry-run do")],
        "llama-3.3-70b-versatile",
        on_chunk=lambda text: events.append(text),
        on_citation=citations.append,
    )

    assert events == ["`--dry-run`", " skips the real run."]
    assert citations == [ToolCitation(tool="search_docs", sources=["README.md — Usage"])]


def test_generate_stream_tool_failure_never_invokes_on_citation(tmp_path, monkeypatch):
    """A failed tool call grounded nothing -- `on_citation` is never called,
    even though the second call still streams a best-effort answer."""

    def _boom(arguments, root=None):
        raise search_docs.SearchDocsError("could not read docs/README.md: [Errno 2] No such file or directory")

    monkeypatch.setattr(search_docs, "run_tool", _boom)

    decision = _search_docs_tool_call_decision("dry-run")
    stream_response = _FakeStream([_chunk("best effort answer")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, docs_root=tmp_path)

    citations: list[ToolCitation] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="dry-run")],
        "llama-3.3-70b-versatile",
        on_chunk=lambda _: None,
        on_citation=citations.append,
    )

    assert citations == []


# --- End-to-end: real chat call site, through AutumnApp -------------------


def _groq_policy_with_default(model_name: str = "llama-3.3-70b-versatile") -> PromptRoutingPolicy:
    return PromptRoutingPolicy(
        provider_accounts=[ProviderAccount(provider="groq", account_id="default", is_signed_in=True)],
        provider_models=[
            ProviderModel(
                name=model_name,
                provider="groq",
                account_id="default",
                priority=10,
                is_default=True,
            )
        ],
    )


async def _land_on_dashboard_and_submit(pilot, prompt: str) -> None:
    await pilot.pause()
    # Empty catalog -> ModelPickerScreen lands first; skip it.
    await pilot.press("escape")
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()

    await pilot.press(":")
    await pilot.pause()
    pilot.app.screen.query_one(CommandBar).query_one(Input).insert_text_at_cursor(prompt)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(0.05)


async def test_dashboard_chat_asks_a_docs_question_and_gets_a_grounded_answer_end_to_end(tmp_path):
    """The acceptance-criteria scenario: asking chat something answerable
    from the docs (here, `--dry-run`) reaches a real `GroqRunner` (with a
    fake Groq client, not a fake GroqRunner) through the exact same call
    site `app.py` already used before tool-calling existed -- no wiring
    change to app.py was needed for the tool round to become reachable."""
    docs_root = tmp_path / "corpus"
    docs_root.mkdir()
    _write_docs_corpus(docs_root)

    decision = _search_docs_tool_call_decision("what does --dry-run do")
    stream_response = _FakeStream(
        [_chunk("`--dry-run`"), _chunk(" replays scripted events instead of running the script for real.")]
    )
    client = _ToolCallingGroqClient(decision, stream_response)
    real_runner_with_fake_client = GroqRunner(client=client, docs_root=docs_root)

    app = AutumnApp(
        runs_root=tmp_path / "runs",
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        groq_runner=real_runner_with_fake_client,
        prompt_routing_policy=_groq_policy_with_default(),
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "what does --dry-run do")
        await pilot.pause(0.2)

        assert app.chat_messages == [
            ChatMessage(role="user", text="what does --dry-run do"),
            ChatMessage(
                role="assistant",
                text="`--dry-run` replays scripted events instead of running the script for real.",
                model="llama-3.3-70b-versatile",
                citation=ToolCitation(tool="search_docs", sources=["README.md — Usage"]),
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: llama-3.3-70b-versatile (provider available)" in str(status_text)
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "--dry-run" in str(chat_text)
        # The citation shows up under the grounded answer in the transcript...
        assert "Source: README.md — Usage" in str(chat_text)
        # ...and the transient "Searching..." status is gone once the turn
        # (and the citation it fed) has fully landed.
        tool_status_text = app.screen.query_one("#chat-tool-status", Static).content
        assert str(tool_status_text) == ""

        # The decide call really did see the grounded doc content.
        [_decide_call, second_call] = client.seen_calls
        tool_entry = second_call["messages"][2]
        assert "README.md" in tool_entry["content"]
        assert "--dry-run" in tool_entry["content"]


async def test_dashboard_chat_shows_search_status_while_tool_runs_and_clears_once_streaming_starts(tmp_path):
    """Acceptance criteria: a visible status names the tool and query as
    soon as the tool call is dispatched, and is gone by the time the final
    answer's text has started rendering -- exercised with a gated fake
    stream so the test can deterministically observe the mid-round-trip
    state instead of racing real timing."""
    docs_root = tmp_path / "corpus"
    docs_root.mkdir()
    _write_docs_corpus(docs_root)

    decision = _search_docs_tool_call_decision("what does --dry-run do")
    stream_response = _GatedFakeStream([_chunk("`--dry-run`"), _chunk(" skips the real run.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    real_runner_with_fake_client = GroqRunner(client=client, docs_root=docs_root)

    app = AutumnApp(
        runs_root=tmp_path / "runs",
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        groq_runner=real_runner_with_fake_client,
        prompt_routing_policy=_groq_policy_with_default(),
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "what does --dry-run do")

        # The decide call and the (fast, synchronous) tool execution have
        # already happened by now; the background thread is blocked inside
        # the gated stream waiting for its first chunk to be released, so
        # the status should already be visible.
        status_text = ""
        for _ in range(40):
            await pilot.pause(0.05)
            status_text = str(app.screen.query_one("#chat-tool-status", Static).content)
            if status_text:
                break
        assert status_text.startswith("Searching docs for 'what does --dry-run do'")
        assert status_text.endswith((".", "..", "..."))
        # Nothing has streamed into the transcript yet.
        chat_text = str(app.screen.query_one("#chat-transcript", Static).content)
        assert "--dry-run` skips" not in chat_text

        stream_response.release_next()
        stream_response.release_next()
        await pilot.pause(0.2)

        # The status is cleared now that the final answer has streamed in.
        status_text = str(app.screen.query_one("#chat-tool-status", Static).content)
        assert status_text == ""
        chat_text = str(app.screen.query_one("#chat-transcript", Static).content)
        assert "`--dry-run` skips the real run." in chat_text


async def test_dashboard_chat_answer_without_a_tool_call_shows_no_status_or_citation(tmp_path):
    """Acceptance criteria: an answer that didn't use a tool renders exactly
    as it did before this ticket -- no status ever appears, and no citation
    line gets added under it."""
    decision = _no_tool_call_decision("Autumn is a GEPA dashboard CLI.")
    client = _ToolCallingGroqClient(decision)
    real_runner_with_fake_client = GroqRunner(client=client, docs_root=tmp_path)

    app = AutumnApp(
        runs_root=tmp_path / "runs",
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        groq_runner=real_runner_with_fake_client,
        prompt_routing_policy=_groq_policy_with_default(),
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "what is autumn")
        await pilot.pause(0.2)

        assert app.chat_messages == [
            ChatMessage(role="user", text="what is autumn"),
            ChatMessage(
                role="assistant",
                text="Autumn is a GEPA dashboard CLI.",
                model="llama-3.3-70b-versatile",
            ),
        ]
        chat_text = str(app.screen.query_one("#chat-transcript", Static).content)
        assert "Autumn is a GEPA dashboard CLI." in chat_text
        assert "Source" not in chat_text
        tool_status_text = str(app.screen.query_one("#chat-tool-status", Static).content)
        assert tool_status_text == ""
