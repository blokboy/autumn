"""Behavioral tests for GroqRunner's confirmation-gated mutating system
tools (docs/prd/chat-cli-parity-tools.md): install_model, set_default_model,
add_key, remove_model, remove_key. Builds on the single-round tool-calling
mechanism #25 introduced (see test_groq_tool_calls.py) -- these tests focus
specifically on the confirm-then-execute/confirm-then-decline branch that
only mutating tools take.

Follows the same Protocol-based fake `GroqClient` convention as
test_groq_tool_calls.py instead of mocking the real SDK.
"""

import json
from dataclasses import dataclass

import local_models
import pytest

from groq_runner import GroqRunner
from models import ChatMessage


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


class _FakeStream:
    def __init__(self, chunks: list[_FakeChunk]):
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks

    def close(self) -> None:
        pass


class _ToolCallingCompletions:
    def __init__(self, decision: _FakeDecision, stream_response: _FakeStream | None, seen_calls: list[dict]):
        self._decision = decision
        self._stream_response = stream_response
        self._seen_calls = seen_calls

    def create(self, *, model: str, messages: list[dict], tools=None, stream: bool = False):
        self._seen_calls.append({"model": model, "messages": messages, "tools": tools, "stream": stream})
        if stream:
            assert self._stream_response is not None
            return self._stream_response
        return self._decision


class _FakeChat:
    def __init__(self, completions: _ToolCallingCompletions):
        self.completions = completions


class _ToolCallingGroqClient:
    def __init__(self, decision: _FakeDecision, stream_response: _FakeStream | None = None):
        self.seen_calls: list[dict] = []
        self.chat = _FakeChat(_ToolCallingCompletions(decision, stream_response, self.seen_calls))


def _tool_call_decision(name: str, arguments: dict, *, call_id: str = "call_1") -> _FakeDecision:
    return _FakeDecision(
        choices=[
            _FakeDecisionChoice(
                message=_FakeDecisionMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(id=call_id, function=_FakeFunctionCall(name=name, arguments=json.dumps(arguments)))
                    ],
                )
            )
        ]
    )


def _chunk(content: str) -> _FakeChunk:
    return _FakeChunk(choices=[_FakeStreamChoice(delta=_FakeDelta(content=content))])


class _RecordingConfirm:
    """A fake `confirm` callback: records every `(message, delay)` it's
    called with, and returns whatever `answer` says (default: always
    confirm)."""

    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.calls: list[tuple[str, float]] = []

    def __call__(self, message: str, delay: float) -> bool:
        self.calls.append((message, delay))
        return self.answer


def _seed_model(catalog_root, name="my-model"):
    source = catalog_root / f"{name}-source.gguf"
    source.write_bytes(b"fake gguf")
    return local_models.install_model(catalog_root, name=name, source_path=source)


# --- Confirmed: tool executes ---------------------------------------------


def test_confirmed_non_destructive_tool_executes_with_zero_delay(tmp_path):
    _seed_model(tmp_path, "model-a")
    _seed_model(tmp_path, "model-b")
    decision = _tool_call_decision("set_default_model", {"model_name": "model-b"})
    stream_response = _FakeStream([_chunk("Done.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    confirm = _RecordingConfirm(answer=True)
    runner = GroqRunner(client=client, catalog_root=tmp_path, confirm=confirm)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="make model-b the default")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
    )

    assert confirm.calls == [(confirm.calls[0][0], 0.0)]
    assert "model-b" in confirm.calls[0][0]
    assert local_models.get_default(tmp_path).name == "model-b"
    assert received == ["Done."]
    # The tool-role message handed to the second call reports success.
    second_call_messages = client.seen_calls[1]["messages"]
    tool_message = next(m for m in second_call_messages if m["role"] == "tool")
    assert "model-b" in tool_message["content"]


def test_confirmed_destructive_tool_executes_with_1_5s_delay(tmp_path):
    _seed_model(tmp_path, "gone-soon")
    decision = _tool_call_decision("remove_model", {"model_name": "gone-soon"})
    stream_response = _FakeStream([_chunk("Removed.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    confirm = _RecordingConfirm(answer=True)
    runner = GroqRunner(client=client, catalog_root=tmp_path, confirm=confirm)

    runner.generate_stream(
        [ChatMessage(role="user", text="remove gone-soon")],
        "llama-3.3-70b-versatile",
        on_chunk=lambda _: None,
    )

    assert confirm.calls[0][1] == 1.5
    assert local_models.list_models(tmp_path) == []


# --- Declined: tool does not execute, model still gets an answer ----------


def test_declined_tool_does_not_execute_and_reports_decline(tmp_path):
    _seed_model(tmp_path, "should-survive")
    decision = _tool_call_decision("remove_model", {"model_name": "should-survive"})
    stream_response = _FakeStream([_chunk("Understood, I did not remove it.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    confirm = _RecordingConfirm(answer=False)
    runner = GroqRunner(client=client, catalog_root=tmp_path, confirm=confirm)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="remove should-survive")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
    )

    # Model still gets a real answer -- decline is not a hard turn-abort.
    assert received == ["Understood, I did not remove it."]
    assert [m.name for m in local_models.list_models(tmp_path)] == ["should-survive"]
    second_call_messages = client.seen_calls[1]["messages"]
    tool_message = next(m for m in second_call_messages if m["role"] == "tool")
    assert "declined" in tool_message["content"].lower()


def test_no_confirm_callback_wired_treats_mutating_call_as_declined(tmp_path):
    _seed_model(tmp_path, "should-survive")
    decision = _tool_call_decision("remove_model", {"model_name": "should-survive"})
    stream_response = _FakeStream([_chunk("ok")])
    client = _ToolCallingGroqClient(decision, stream_response)
    # No `confirm=` passed at all.
    runner = GroqRunner(client=client, catalog_root=tmp_path)

    runner.generate_stream(
        [ChatMessage(role="user", text="remove should-survive")],
        "llama-3.3-70b-versatile",
        on_chunk=lambda _: None,
    )

    assert [m.name for m in local_models.list_models(tmp_path)] == ["should-survive"]


# --- Execution failure after confirmation: visible degrade ----------------


def test_confirmed_tool_execution_failure_is_surfaced_visibly(tmp_path):
    decision = _tool_call_decision("remove_model", {"model_name": "never-installed"})
    stream_response = _FakeStream([_chunk("I couldn't remove it.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    confirm = _RecordingConfirm(answer=True)
    runner = GroqRunner(client=client, catalog_root=tmp_path, confirm=confirm)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="remove never-installed")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
    )

    # A visible failure notice arrives before the model's own follow-up text.
    assert any("remove_model failed" in chunk for chunk in received)
    assert "I couldn't remove it." in received


# --- All five mutating tools are offered on every decide call -------------


def test_all_five_mutating_tools_are_offered(tmp_path):
    decision = _FakeDecision(choices=[_FakeDecisionChoice(message=_FakeDecisionMessage(content="hi"))])
    client = _ToolCallingGroqClient(decision)
    runner = GroqRunner(client=client, catalog_root=tmp_path)

    runner.generate_stream([ChatMessage(role="user", text="hi")], "llama-3.1-8b-instant", on_chunk=lambda _: None)

    [call] = client.seen_calls
    offered_names = {tool["function"]["name"] for tool in call["tools"]}
    assert offered_names == {
        "search_docs",
        "install_model",
        "set_default_model",
        "add_key",
        "remove_model",
        "remove_key",
    }
