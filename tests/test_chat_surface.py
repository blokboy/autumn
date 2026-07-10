"""Behavioral tests for dashboard-scoped chat prompts."""

from textual.widgets import Static
from textual.widgets import Input

from autumn.app import AutumnApp
from autumn.models import ChatMessage
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
