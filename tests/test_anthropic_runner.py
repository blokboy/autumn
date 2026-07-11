"""Behavioral tests for calling Anthropic's hosted Messages API."""

from dataclasses import dataclass

import anthropic
import pytest

from anthropic_runner import AnthropicRunner, AnthropicRuntimeError
from models import ChatMessage


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeMessageResponse:
    content: list


class _FakeMessages:
    def __init__(self, reply_text: str, seen_calls: list[dict]):
        self._reply_text = reply_text
        self._seen_calls = seen_calls

    def create(self, *, model: str, max_tokens: int, messages: list[dict[str, str]]):
        self._seen_calls.append({"model": model, "max_tokens": max_tokens, "messages": messages})
        return _FakeMessageResponse(content=[_FakeTextBlock(text=self._reply_text)])


class _FakeAnthropicClient:
    def __init__(self, reply_text: str = "A real-ish hosted answer.\n"):
        self.seen_calls: list[dict] = []
        self.messages = _FakeMessages(reply_text, self.seen_calls)


class _FailingMessages:
    def __init__(self, error: Exception):
        self._error = error

    def create(self, *, model: str, max_tokens: int, messages: list[dict[str, str]]):
        raise self._error


class _FailingAnthropicClient:
    def __init__(self, error: Exception):
        self.messages = _FailingMessages(error)


def test_anthropic_runner_returns_assistant_message_from_api_response():
    client = _FakeAnthropicClient()
    runner = AnthropicRunner(client=client)

    message = runner.generate(
        [ChatMessage(role="user", text="hello")],
        "claude-sonnet-5",
    )

    assert message == ChatMessage(
        role="assistant",
        text="A real-ish hosted answer.",
        model="claude-sonnet-5",
    )
    assert client.seen_calls == [
        {
            "model": "claude-sonnet-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
        }
    ]


def test_anthropic_runner_converts_multi_turn_transcript():
    client = _FakeAnthropicClient()
    runner = AnthropicRunner(client=client)

    runner.generate(
        [
            ChatMessage(role="user", text="hi"),
            ChatMessage(role="assistant", text="hello there", model="claude-haiku-4-5"),
            ChatMessage(role="user", text="how are you"),
        ],
        "claude-haiku-4-5",
    )

    assert client.seen_calls == [
        {
            "model": "claude-haiku-4-5",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there"},
                {"role": "user", "content": "how are you"},
            ],
        }
    ]


class _MultiBlockMessages:
    def create(self, *, model: str, max_tokens: int, messages: list[dict[str, str]]):
        return _FakeMessageResponse(content=[_FakeTextBlock(text="Hello, "), _FakeTextBlock(text="world!")])


class _MultiBlockClient:
    def __init__(self):
        self.messages = _MultiBlockMessages()


def test_anthropic_runner_joins_multiple_text_blocks():
    runner = AnthropicRunner(client=_MultiBlockClient())

    message = runner.generate([ChatMessage(role="user", text="hi")], "claude-sonnet-5")

    assert message.text == "Hello, world!"


def test_anthropic_runner_reports_authentication_failure():
    request = anthropic._base_client.httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.AuthenticationError(
        message="Invalid API Key",
        response=anthropic._base_client.httpx.Response(401, request=request),
        body=None,
    )
    client = _FailingAnthropicClient(error)
    runner = AnthropicRunner(client=client)

    with pytest.raises(AnthropicRuntimeError, match="claude-haiku-4-5 failed: Invalid API Key"):
        runner.generate([ChatMessage(role="user", text="hello")], "claude-haiku-4-5")


def test_anthropic_runner_reports_connection_failure():
    request = anthropic._base_client.httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(message="Connection error.", request=request)
    client = _FailingAnthropicClient(error)
    runner = AnthropicRunner(client=client)

    with pytest.raises(AnthropicRuntimeError, match="claude-sonnet-5 failed: Connection error."):
        runner.generate([ChatMessage(role="user", text="hello")], "claude-sonnet-5")


def test_anthropic_runner_defaults_to_a_real_anthropic_client_when_none_injected(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    runner = AnthropicRunner()

    assert isinstance(runner._client, anthropic.Anthropic)


def test_anthropic_runner_prefers_a_stored_key_over_the_env_var(monkeypatch):
    import credentials

    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    credentials.set_key("anthropic", "from-store")

    runner = AnthropicRunner()

    assert runner._client.api_key == "from-store"
