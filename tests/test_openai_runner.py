"""Behavioral tests for calling OpenAI's hosted chat completions API."""

from dataclasses import dataclass

import openai
import pytest

from models import ChatMessage
from openai_runner import OpenAIRunner, OpenAIRuntimeError


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


class _FakeOpenAIClient:
    def __init__(self, reply_text: str = "A real-ish hosted answer.\n"):
        self.seen_calls: list[dict] = []
        self.chat = _FakeChat(_FakeCompletions(reply_text, self.seen_calls))


class _FailingCompletions:
    def __init__(self, error: Exception):
        self._error = error

    def create(self, *, model: str, messages: list[dict[str, str]]):
        raise self._error


class _FailingOpenAIClient:
    def __init__(self, error: Exception):
        self.chat = _FakeChat(_FailingCompletions(error))


def test_openai_runner_returns_assistant_message_from_api_response():
    client = _FakeOpenAIClient()
    runner = OpenAIRunner(client=client)

    message = runner.generate(
        [ChatMessage(role="user", text="hello")],
        "gpt-4o",
    )

    assert message == ChatMessage(
        role="assistant",
        text="A real-ish hosted answer.",
        model="gpt-4o",
    )
    assert client.seen_calls == [
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        }
    ]


def test_openai_runner_converts_multi_turn_transcript():
    client = _FakeOpenAIClient()
    runner = OpenAIRunner(client=client)

    runner.generate(
        [
            ChatMessage(role="user", text="hi"),
            ChatMessage(role="assistant", text="hello there", model="gpt-4o-mini"),
            ChatMessage(role="user", text="how are you"),
        ],
        "gpt-4o-mini",
    )

    assert client.seen_calls == [
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there"},
                {"role": "user", "content": "how are you"},
            ],
        }
    ]


def test_openai_runner_reports_authentication_failure():
    request = openai._base_client.httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = openai.AuthenticationError(
        message="Invalid API Key",
        response=openai._base_client.httpx.Response(401, request=request),
        body=None,
    )
    client = _FailingOpenAIClient(error)
    runner = OpenAIRunner(client=client)

    with pytest.raises(OpenAIRuntimeError, match="gpt-4.1-mini failed: Invalid API Key"):
        runner.generate([ChatMessage(role="user", text="hello")], "gpt-4.1-mini")


def test_openai_runner_reports_connection_failure():
    request = openai._base_client.httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = openai.APIConnectionError(message="Connection error.", request=request)
    client = _FailingOpenAIClient(error)
    runner = OpenAIRunner(client=client)

    with pytest.raises(OpenAIRuntimeError, match="gpt-4o failed: Connection error."):
        runner.generate([ChatMessage(role="user", text="hello")], "gpt-4o")


def test_openai_runner_defaults_to_a_real_openai_client_when_none_injected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    runner = OpenAIRunner()

    assert isinstance(runner._client, openai.OpenAI)


def test_openai_runner_prefers_a_stored_key_over_the_env_var(monkeypatch):
    import credentials

    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    credentials.set_key("openai", "from-store")

    runner = OpenAIRunner()

    assert runner._client.api_key == "from-store"
