"""Behavioral tests for dashboard-scoped chat prompts."""

from textual.widgets import Static
from textual.widgets import Input

from autumn import local_models
from autumn.app import AutumnApp
from autumn.models import ChatMessage, LocalModel, PromptRoutingPolicy, ProviderAccount, ProviderModel
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
        assert "Fallback: runtime missing for tiny" in str(status_text)


async def test_prompt_with_provider_policy_falls_back_until_provider_execution_exists(tmp_path):
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
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
        assert "Fallback: provider claude/sonnet not executable yet" in str(status_text)
