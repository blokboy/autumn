"""Behavioral tests proving `GroqRunner` actually offers and executes the
three read-only system tools from docs/prd/chat-cli-parity-tools.md
(`list_models`, `list_keys`, `list_runs`) end-to-end -- same single-round
tool-calling mechanism `test_groq_tool_calls.py` (#25) already covers for
`search_docs`, extended to the new tools this ticket (#29) adds.

Follows the same Protocol-based fake `GroqClient`/`_ChatCompletions`
convention as test_groq_tool_calls.py instead of mocking the real SDK.
"""

import json
from dataclasses import dataclass

from textual.widgets import Input

import credentials
import list_keys
import list_models
import list_runs
import local_models
from app import AutumnApp
from groq_runner import _TOOLS, GroqRunner
from models import ChatMessage, PromptRoutingPolicy, ProviderAccount, ProviderModel
from screens.dashboard_screen import DashboardScreen
from widgets.command_bar import CommandBar


# --- Fakes (same shape as test_groq_tool_calls.py) ------------------------


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


class _FakeStream:
    def __init__(self, chunks: list[_FakeChunk]):
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class _ToolCallingCompletions:
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


def _tool_call_decision(tool_name: str, arguments: dict, *, call_id: str = "call_1") -> _FakeDecision:
    return _FakeDecision(
        choices=[
            _FakeDecisionChoice(
                message=_FakeDecisionMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id=call_id,
                            function=_FakeFunctionCall(name=tool_name, arguments=json.dumps(arguments)),
                        )
                    ],
                )
            )
        ]
    )


def _install_model(catalog_root, *, name="llama-3.2-3b-instruct"):
    source = catalog_root / f"{name}.gguf"
    catalog_root.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fake gguf")
    return local_models.install_model(catalog_root, name=name, source_path=source, context_window=8192)


def _write_run(runs_root, name="my-run-20260101T000000"):
    run_dir = runs_root / name
    run_dir.mkdir(parents=True)
    (run_dir / "candidates.json").write_text(json.dumps([{"val_score": 0.75}]))
    (run_dir / "run_log.json").write_text(json.dumps([{"event": "on_optimization_end"}]))
    return run_dir


# --- All four read-only tools are offered on every decide call ------------


def test_all_read_only_tools_are_offered_on_every_decide_call(tmp_path):
    decision = _FakeDecision(choices=[_FakeDecisionChoice(message=_FakeDecisionMessage(content="hi"))])
    client = _ToolCallingGroqClient(decision)
    runner = GroqRunner(client=client, docs_root=tmp_path, catalog_root=tmp_path, runs_root=tmp_path)

    runner.generate_stream([ChatMessage(role="user", text="hi")], "llama-3.3-70b-versatile", on_chunk=lambda _: None)

    [call] = client.seen_calls
    offered_names = {schema["function"]["name"] for schema in call["tools"]}
    # Subset, not exact-equality -- the mutating system tools (#30) are
    # offered alongside these four read-only ones; this test only cares that
    # the read-only tools are all present.
    assert {"search_docs", "list_models", "list_keys", "list_runs"}.issubset(offered_names)
    assert call["tools"] == _TOOLS


# --- list_models ------------------------------------------------------------


def test_generate_stream_executes_list_models_tool_call_and_streams_second_call(tmp_path):
    catalog_root = tmp_path / "models"
    installed = _install_model(catalog_root)
    decision = _tool_call_decision(list_models.TOOL_NAME, {})
    stream_response = _FakeStream([_chunk("You have "), _chunk("llama-3.2-3b-instruct installed.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, catalog_root=catalog_root)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="what models do I have installed")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
    )

    assert received == ["You have ", "llama-3.2-3b-instruct installed."]
    [_decide_call, second_call] = client.seen_calls
    tool_entry = second_call["messages"][2]
    assert tool_entry["role"] == "tool"
    result_rows = json.loads(tool_entry["content"])
    assert result_rows == [
        {
            "name": installed.name,
            "backend": installed.backend,
            "path": str(installed.path),
            "context_window": installed.context_window,
            "is_default": installed.is_default,
        }
    ]


# --- list_keys ---------------------------------------------------------------


def test_generate_stream_executes_list_keys_tool_call_and_streams_second_call():
    credentials.set_key("groq", "gsk_super_secret")
    decision = _tool_call_decision(list_keys.TOOL_NAME, {})
    stream_response = _FakeStream([_chunk("Groq is configured; the others aren't.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="which providers have a key configured")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
    )

    assert received == ["Groq is configured; the others aren't."]
    [_decide_call, second_call] = client.seen_calls
    tool_entry = second_call["messages"][2]
    result_rows = json.loads(tool_entry["content"])
    by_provider = {row["provider"]: row["configured"] for row in result_rows}
    assert by_provider["groq"] is True
    assert by_provider["anthropic"] is False
    # Never leaks the actual stored key value.
    assert "gsk_super_secret" not in tool_entry["content"]


# --- list_runs ---------------------------------------------------------------


def test_generate_stream_executes_list_runs_tool_call_and_streams_second_call(tmp_path):
    runs_root = tmp_path / "runs"
    _write_run(runs_root)
    decision = _tool_call_decision(list_runs.TOOL_NAME, {})
    stream_response = _FakeStream([_chunk("You've done one run, currently completed.")])
    client = _ToolCallingGroqClient(decision, stream_response)
    runner = GroqRunner(client=client, runs_root=runs_root)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="what runs have I done")],
        "llama-3.3-70b-versatile",
        on_chunk=received.append,
    )

    assert received == ["You've done one run, currently completed."]
    [_decide_call, second_call] = client.seen_calls
    tool_entry = second_call["messages"][2]
    result_rows = json.loads(tool_entry["content"])
    assert len(result_rows) == 1
    assert result_rows[0]["name"] == "my-run-20260101T000000"
    assert result_rows[0]["status"] == "completed"


# --- End-to-end through AutumnApp, no confirmation modal -------------------


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


async def test_dashboard_chat_asks_a_models_question_and_gets_a_real_answer_with_no_confirmation(tmp_path):
    """The acceptance-criteria scenario: asking chat "what models do I have
    installed" reaches a real `GroqRunner` (with a fake Groq client) through
    the exact same call site tool-calling already used for search_docs, and
    no confirmation modal is shown -- these are read-only, same trust level
    as search_docs."""
    catalog_root = tmp_path / "models"
    installed = _install_model(catalog_root)

    decision = _tool_call_decision(list_models.TOOL_NAME, {})
    stream_response = _FakeStream(
        [_chunk(f"You have {installed.name} installed"), _chunk(" and it's your default.")]
    )
    client = _ToolCallingGroqClient(decision, stream_response)
    real_runner_with_fake_client = GroqRunner(client=client, catalog_root=catalog_root)

    app = AutumnApp(
        runs_root=tmp_path / "runs",
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        groq_runner=real_runner_with_fake_client,
        prompt_routing_policy=_groq_policy_with_default(),
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "what models do I have installed")
        await pilot.pause(0.2)

        assert app.chat_messages == [
            ChatMessage(role="user", text="what models do I have installed"),
            ChatMessage(
                role="assistant",
                text=f"You have {installed.name} installed and it's your default.",
                model="llama-3.3-70b-versatile",
            ),
        ]
        # No confirmation modal was ever pushed -- the dashboard screen is
        # still the topmost (and only) screen on the stack.
        assert isinstance(pilot.app.screen, DashboardScreen)

        [_decide_call, second_call] = client.seen_calls
        tool_entry = second_call["messages"][2]
        assert installed.name in tool_entry["content"]
