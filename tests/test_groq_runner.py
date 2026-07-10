"""Behavioral tests for calling Groq's hosted chat completions API."""

from dataclasses import dataclass

import groq
import pytest

from autumn.groq_runner import GroqRunner, GroqRuntimeError
from autumn.models import ChatMessage


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]


class _FakeCompletions:
    def __init__(self, reply_text: str, seen_calls: list[dict]):
        self._reply_text = reply_text
        self._seen_calls = seen_calls

    def create(self, *, model: str, messages: list[dict[str, str]]):
        self._seen_calls.append({"model": model, "messages": messages})
        return _FakeCompletion(choices=[_FakeChoice(message=_FakeMessage(content=self._reply_text))])


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class _FakeGroqClient:
    def __init__(self, reply_text: str = "A real-ish hosted answer.\n"):
        self.seen_calls: list[dict] = []
        self.chat = _FakeChat(_FakeCompletions(reply_text, self.seen_calls))


class _FailingCompletions:
    def __init__(self, error: Exception):
        self._error = error

    def create(self, *, model: str, messages: list[dict[str, str]]):
        raise self._error


class _FailingGroqClient:
    def __init__(self, error: Exception):
        self.chat = _FakeChat(_FailingCompletions(error))


def test_groq_runner_returns_assistant_message_from_api_response():
    client = _FakeGroqClient()
    runner = GroqRunner(client=client)

    message = runner.generate(
        [ChatMessage(role="user", text="hello")],
        "llama-3.3-70b-versatile",
    )

    assert message == ChatMessage(
        role="assistant",
        text="A real-ish hosted answer.",
        model="llama-3.3-70b-versatile",
    )
    assert client.seen_calls == [
        {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "hello"}],
        }
    ]


def test_groq_runner_converts_multi_turn_transcript():
    client = _FakeGroqClient()
    runner = GroqRunner(client=client)

    runner.generate(
        [
            ChatMessage(role="user", text="hi"),
            ChatMessage(role="assistant", text="hello there", model="llama-3.1-8b-instant"),
            ChatMessage(role="user", text="how are you"),
        ],
        "llama-3.1-8b-instant",
    )

    assert client.seen_calls == [
        {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there"},
                {"role": "user", "content": "how are you"},
            ],
        }
    ]


def test_groq_runner_reports_authentication_failure():
    request = groq._base_client.httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error = groq.AuthenticationError(
        message="Invalid API Key",
        response=groq._base_client.httpx.Response(401, request=request),
        body=None,
    )
    client = _FailingGroqClient(error)
    runner = GroqRunner(client=client)

    with pytest.raises(GroqRuntimeError, match="gemma2-9b-it failed: Invalid API Key"):
        runner.generate([ChatMessage(role="user", text="hello")], "gemma2-9b-it")


def test_groq_runner_reports_connection_failure():
    request = groq._base_client.httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error = groq.APIConnectionError(message="Connection error.", request=request)
    client = _FailingGroqClient(error)
    runner = GroqRunner(client=client)

    with pytest.raises(GroqRuntimeError, match="llama-3.3-70b-versatile failed: Connection error."):
        runner.generate([ChatMessage(role="user", text="hello")], "llama-3.3-70b-versatile")


def test_groq_runner_defaults_to_a_real_groq_client_when_none_injected(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    runner = GroqRunner()

    assert isinstance(runner._client, groq.Groq)
