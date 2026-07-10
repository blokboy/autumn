"""Behavioral tests for dashboard-scoped chat prompts."""

import json

from textual.widgets import Static
from textual.widgets import Input

from autumn import local_models
from autumn.app import AutumnApp
from autumn.local_model_runner import LocalModelRuntimeError
from autumn.models import (
    ChatMessage,
    LocalModel,
    PromptRoutingPolicy,
    ProviderAccount,
    ProviderModel,
    RunStatus,
)
from autumn.screens.dashboard_screen import DashboardScreen
from autumn.widgets.command_bar import CommandBar


async def test_non_gepa_landing_prompt_opens_dashboard_chat(tmp_path):
    app = AutumnApp(runs_root=tmp_path, chat_sessions_root=tmp_path / "chats")

    async with app.run_test() as pilot:
        await pilot.pause()
        pilot.app.screen.query_one("Input").insert_text_at_cursor("summarize my last run")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, DashboardScreen)
        assert app.chat_messages[0] == ChatMessage(role="user", text="summarize my last run")
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "You: summarize my last run" in str(chat_text)


async def test_empty_chat_explains_shared_prompt_surface(tmp_path):
    app = AutumnApp(runs_root=tmp_path, chat_sessions_root=tmp_path / "chats")

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        status_text = app.screen.query_one("#chat-model-status", Static).content
        chat_text = app.screen.query_one("#chat-transcript", Static).content

        assert "Model: ready to choose a local model or offline fallback" in str(status_text)
        assert "Ask Autumn about your runs from the landing input or command bar." in str(chat_text)
        assert "stub" not in str(chat_text).lower()


async def test_non_gepa_command_bar_prompt_updates_same_dashboard_chat(tmp_path):
    app = AutumnApp(runs_root=tmp_path, chat_sessions_root=tmp_path / "chats")

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press(":")
        await pilot.pause()
        app.screen.query_one(CommandBar).query_one(Input).insert_text_at_cursor("what happened?")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert app.chat_messages[0] == ChatMessage(role="user", text="what happened?")
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "You: what happened?" in str(chat_text)


async def test_prompt_receives_async_local_model_reply(tmp_path):
    app = AutumnApp(runs_root=tmp_path, chat_sessions_root=tmp_path / "chats")

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press(":")
        await pilot.pause()
        app.screen.query_one(CommandBar).query_one(Input).insert_text_at_cursor("hello")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="Offline local response: hello",
                model="autumn/offline-tiny",
            ),
        ]
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "Autumn [autumn/offline-tiny]: Offline local response: hello" in str(chat_text)


async def test_prompt_receives_async_installed_model_reply(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)

    class FakeLocalModelRunner:
        def generate(self, messages: list[ChatMessage], model: LocalModel) -> ChatMessage:
            assert messages == [ChatMessage(role="user", text="hello")]
            assert model.name == "tiny"
            return ChatMessage(role="assistant", text="Installed model answered.", model=model.name)

    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=FakeLocalModelRunner(),
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press(":")
        await pilot.pause()
        app.screen.query_one(CommandBar).query_one(Input).insert_text_at_cursor("hello")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(role="assistant", text="Installed model answered.", model="tiny"),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: tiny (installed default)" in str(status_text)
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "Autumn [tiny]: Installed model answered." in str(chat_text)


async def test_prompt_falls_back_when_installed_model_runtime_fails(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)

    class FailingLocalModelRunner:
        def generate(self, messages: list[ChatMessage], model: LocalModel) -> ChatMessage:
            raise LocalModelRuntimeError("tiny failed: bad model file")

    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=FailingLocalModelRunner(),
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press(":")
        await pilot.pause()
        app.screen.query_one(CommandBar).query_one(Input).insert_text_at_cursor("hello")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="Offline local response: hello",
                model="autumn/offline-tiny",
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Fallback: autumn/offline-tiny (tiny failed: bad model file)" in str(status_text)


async def test_prompt_preserves_selected_model_error_response_without_fallback(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)

    class ErrorResponseLocalModelRunner:
        def generate(self, messages: list[ChatMessage], model: LocalModel) -> ChatMessage:
            return ChatMessage(
                role="assistant",
                text="Error: context window exceeded",
                model=model.name,
            )

    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=ErrorResponseLocalModelRunner(),
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press(":")
        await pilot.pause()
        app.screen.query_one(CommandBar).query_one(Input).insert_text_at_cursor("hello")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="Error: context window exceeded",
                model="tiny",
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model error: tiny (installed default): Error: context window exceeded" in str(
            status_text
        )
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "Autumn [tiny]: Error: context window exceeded" in str(chat_text)


async def test_prompt_shows_fallback_reason_when_installed_model_runtime_is_missing(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        is_model_runtime_available=lambda model: False,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press(":")
        await pilot.pause()
        app.screen.query_one(CommandBar).query_one(Input).insert_text_at_cursor("hello")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="Offline local response: hello",
                model="autumn/offline-tiny",
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Fallback: autumn/offline-tiny (runtime missing for tiny)" in str(status_text)


async def test_prompt_with_provider_policy_falls_back_until_provider_execution_exists(tmp_path):
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        # Explicit (empty) catalog root, bypassing conftest's seeded local
        # model -- this test asserts the exact "offline fallback" reason,
        # which only holds when there's truly nothing local to try first.
        model_catalog_root=tmp_path / "models",
        prompt_routing_policy=PromptRoutingPolicy(
            provider_accounts=[
                ProviderAccount(provider="claude", account_id="personal", is_signed_in=True)
            ],
            provider_models=[
                ProviderModel(
                    name="claude/sonnet",
                    provider="claude",
                    account_id="personal",
                    priority=10,
                )
            ],
        ),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        # Empty catalog -> ModelPickerScreen lands first; skip it to reach
        # InputScreen's empty-Enter -> browse-mode path this test cares about.
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press(":")
        await pilot.pause()
        app.screen.query_one(CommandBar).query_one(Input).insert_text_at_cursor("hello")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="Offline local response: hello",
                model="autumn/offline-tiny",
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Fallback: autumn/offline-tiny (provider claude/sonnet not executable yet)" in str(status_text)


async def test_live_run_queues_prompt_replies_in_order_without_duplicate_users(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(0.3)\n")
    app = AutumnApp(
        runs_root=tmp_path,
        run_name="live",
        run_dir=tmp_path / "live",
        script_path=script,
        dry_run=False,
        queue_sessions_root=tmp_path / "queues",
        chat_sessions_root=tmp_path / "chats",
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.state.status is RunStatus.RUNNING

        app.submit_command("first queued prompt")
        app.submit_command("second queued prompt")
        await pilot.pause(0.1)

        assert app.chat_messages == [
            ChatMessage(role="user", text="first queued prompt"),
            ChatMessage(role="user", text="second queued prompt"),
        ]
        assert json.loads(app._queue_session_path.read_text())["items"] == [
            {"kind": "prompt", "text": "first queued prompt"},
            {"kind": "prompt", "text": "second queued prompt"},
        ]
        assert json.loads(app._chat_session_path.read_text())["messages"] == [
            {"role": "user", "text": "first queued prompt"},
            {"role": "user", "text": "second queued prompt"},
        ]

        await pilot.pause(0.7)

        assert app.pending_queue == []
        assert app.chat_messages == [
            ChatMessage(role="user", text="first queued prompt"),
            ChatMessage(
                role="assistant",
                text="Offline local response: first queued prompt",
                model="autumn/offline-tiny",
            ),
            ChatMessage(role="user", text="second queued prompt"),
            ChatMessage(
                role="assistant",
                text="Offline local response: second queued prompt",
                model="autumn/offline-tiny",
            ),
        ]
