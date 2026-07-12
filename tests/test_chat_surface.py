"""Behavioral tests for dashboard-scoped chat prompts."""

import json
import threading

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.widgets import Input

import groq_policy, local_models
import subagent
from app import AutumnApp
from local_model_runner import LocalModelRuntimeError
from models import (
    ChatMessage,
    PromptRoutingPolicy,
    ProviderAccount,
    ProviderModel,
    RunStatus,
)
from screens.dashboard_screen import DashboardScreen
from widgets.chat_view import ChatView, _render_messages, _render_tool_status
from widgets.command_bar import CommandBar


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


def test_chat_transcript_renders_named_participant_with_model_metadata():
    rendered = _render_messages(
        [
            ChatMessage(role="user", text="hello"),
            ChatMessage(role="assistant", text="hi", model="local/tiny"),
            ChatMessage(
                role="assistant",
                text="I checked the docs.",
                model="local/tiny",
                participant_name="Autumn Sub Agent 1",
            ),
        ]
    )

    assert "You: hello" in rendered
    assert "Autumn [local/tiny]: hi" in rendered
    assert "Autumn Sub Agent 1 [local/tiny]: I checked the docs." in rendered


def test_chat_tool_status_renders_animated_ellipsis_frames():
    assert _render_tool_status(None) == ""
    assert _render_tool_status("Searching docs for 'Autumn'", frame=0) == "Searching docs for 'Autumn'."
    assert _render_tool_status("Searching docs for 'Autumn'", frame=1) == "Searching docs for 'Autumn'.."
    assert _render_tool_status("Searching docs for 'Autumn'", frame=2) == "Searching docs for 'Autumn'..."


async def test_chat_tool_status_animates_while_search_is_active():
    class ChatHarness(App):
        def compose(self) -> ComposeResult:
            yield ChatView([], tool_status="Searching docs for 'Autumn'")

    async with ChatHarness().run_test() as pilot:
        await pilot.pause()
        first = str(pilot.app.query_one("#chat-tool-status", Static).content)
        await pilot.pause(0.45)
        second = str(pilot.app.query_one("#chat-tool-status", Static).content)

        assert first == "Searching docs for 'Autumn'."
        assert second == "Searching docs for 'Autumn'.."


async def test_chat_transcript_treats_model_output_as_plain_text():
    class ChatHarness(App):
        def compose(self) -> ComposeResult:
            yield ChatView(
                [
                    ChatMessage(role="user", text="Can you explain Autumn?"),
                    ChatMessage(
                        role="assistant",
                        text='[{"type":"function","function":{"name":"search_docs","arguments": "what is autumn"}}]',
                        model="llama-3.3-70b-versatile",
                    ),
                ]
            )

    async with ChatHarness().run_test() as pilot:
        await pilot.pause()
        replacement = [
            ChatMessage(role="user", text="Can you explain Autumn?"),
            ChatMessage(
                role="assistant",
                text='[bad-markup: "what is autumn"}</function>\'}]',
                model="llama-3.3-70b-versatile",
            ),
        ]

        pilot.app.query_one(ChatView).refresh_from_messages(replacement)
        await pilot.pause()

        chat_text = pilot.app.query_one("#chat-transcript", Static).content
        assert "[bad-markup" in str(chat_text)


async def test_chat_transcript_is_scrollable_and_tracks_latest_message():
    messages = [
        ChatMessage(role="user" if index % 2 == 0 else "assistant", text=f"message {index}\n" + ("detail\n" * 3))
        for index in range(40)
    ]

    class ChatHarness(App):
        def compose(self) -> ComposeResult:
            yield ChatView(messages)

    async with ChatHarness().run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        scroller = pilot.app.query_one("#chat-transcript-scroll", VerticalScroll)

        assert scroller.max_scroll_y > 0

        updated = messages + [ChatMessage(role="assistant", text="the latest reply")]
        pilot.app.query_one(ChatView).refresh_from_messages(updated)
        await pilot.pause()

        assert scroller.scroll_y == scroller.max_scroll_y
        chat_text = pilot.app.query_one("#chat-transcript", Static).content
        assert "the latest reply" in str(chat_text)


async def test_chat_transcript_does_not_force_scroll_when_user_reads_history():
    messages = [
        ChatMessage(role="user" if index % 2 == 0 else "assistant", text=f"message {index}\n" + ("detail\n" * 3))
        for index in range(40)
    ]

    class ChatHarness(App):
        def compose(self) -> ComposeResult:
            yield ChatView(messages)

    async with ChatHarness().run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        scroller = pilot.app.query_one("#chat-transcript-scroll", VerticalScroll)
        scroller.scroll_home(animate=False, immediate=True)
        await pilot.pause()

        updated = messages + [ChatMessage(role="assistant", text="new reply while reading history")]
        pilot.app.query_one(ChatView).refresh_from_messages(updated)
        await pilot.pause()

        assert scroller.scroll_y == 0
        chat_text = pilot.app.query_one("#chat-transcript", Static).content
        assert "new reply while reading history" in str(chat_text)


async def test_chat_transcript_can_be_selected_and_copied_with_keyboard():
    messages = [
        ChatMessage(role="user", text="copy this question"),
        ChatMessage(role="assistant", text="copy this answer", model="tiny"),
    ]

    class ChatHarness(App):
        def compose(self) -> ComposeResult:
            yield ChatView(messages)

    async with ChatHarness().run_test() as pilot:
        await pilot.pause()
        pilot.app.query_one(ChatView).focus()
        await pilot.press("ctrl+a")

        selected_text = pilot.app.screen.get_selected_text()
        assert selected_text is not None
        assert "You: copy this question" in selected_text
        assert "Autumn [tiny]: copy this answer" in selected_text

        await pilot.press("ctrl+c")

        assert "You: copy this question" in pilot.app.clipboard
        assert "Autumn [tiny]: copy this answer" in pilot.app.clipboard


async def test_chat_copy_shortcut_falls_back_to_full_transcript_without_selection():
    messages = [
        ChatMessage(role="user", text="copy all"),
        ChatMessage(role="assistant", text="without selecting first"),
    ]

    class ChatHarness(App):
        def compose(self) -> ComposeResult:
            yield ChatView(messages)

    async with ChatHarness().run_test() as pilot:
        await pilot.pause()
        pilot.app.query_one(ChatView).focus()
        await pilot.press("ctrl+c")

        assert pilot.app.clipboard == _render_messages(messages)


async def test_subagent_command_preserves_raw_command_and_posts_named_result(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)
    release = threading.Event()

    class BlockingSubagentRunner:
        def __init__(self):
            self.messages = None

        def generate(self, messages, model):
            self.messages = messages
            release.wait(timeout=2)
            return ChatMessage(role="assistant", text="subagent answer", model=model.name)

    runner = BlockingSubagentRunner()
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=runner,
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.submit_command("/subagent inspect the run")
        await pilot.pause(0.1)

        assert app.chat_messages == [
            ChatMessage(role="user", text="/subagent inspect the run"),
            ChatMessage(
                role="assistant",
                text="working...",
                participant_name="Autumn Sub Agent 1",
            ),
        ]

        release.set()
        await pilot.pause(0.2)

        assert runner.messages == [
            ChatMessage(role="system", text=subagent.SUBAGENT_SYSTEM_INSTRUCTION),
            ChatMessage(role="user", text="inspect the run"),
        ]
        assert app.chat_messages == [
            ChatMessage(role="user", text="/subagent inspect the run"),
            ChatMessage(
                role="assistant",
                text="subagent answer",
                model="tiny",
                participant_name="Autumn Sub Agent 1",
            ),
        ]
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "Autumn Sub Agent 1 [tiny]: subagent answer" in str(chat_text)


async def test_subagent_command_runs_while_live_run_without_queueing(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(0.4)\n")
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)

    class FastSubagentRunner:
        def generate(self, messages, model):
            return ChatMessage(role="assistant", text="parallel answer", model=model.name)

    app = AutumnApp(
        runs_root=tmp_path,
        run_name="live",
        run_dir=tmp_path / "live",
        script_path=script,
        dry_run=False,
        queue_sessions_root=tmp_path / "queues",
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=FastSubagentRunner(),
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.state.status is RunStatus.RUNNING

        app.submit_command("/subagent review this")
        await pilot.pause(0.2)

        assert app.pending_queue == []
        assert app.chat_messages == [
            ChatMessage(role="user", text="/subagent review this"),
            ChatMessage(
                role="assistant",
                text="parallel answer",
                model="tiny",
                participant_name="Autumn Sub Agent 1",
            ),
        ]


async def test_subagent_commands_cap_running_at_four_and_queue_the_fifth(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)
    releases = {f"task {index}": threading.Event() for index in range(1, 6)}
    started: list[str] = []

    class BlockingSubagentRunner:
        def generate(self, messages, model):
            prompt = messages[-1].text
            started.append(prompt)
            releases[prompt].wait(timeout=2)
            return ChatMessage(role="assistant", text=f"answer {prompt}", model=model.name)

    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=BlockingSubagentRunner(),
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        for index in range(1, 6):
            app.submit_command(f"/subagent task {index}")
        await pilot.pause(0.2)

        assert started == ["task 1", "task 2", "task 3", "task 4"]
        assert app.chat_messages[-1] == ChatMessage(
            role="assistant",
            text="queued...",
            participant_name="Autumn Sub Agent 5",
        )

        releases["task 1"].set()
        await pilot.pause(0.2)

        assert started == ["task 1", "task 2", "task 3", "task 4", "task 5"]
        assert app.chat_messages[1] == ChatMessage(
            role="assistant",
            text="answer task 1",
            model="tiny",
            participant_name="Autumn Sub Agent 1",
        )
        assert app.chat_messages[-1] == ChatMessage(
            role="assistant",
            text="working...",
            participant_name="Autumn Sub Agent 5",
        )

        for prompt in ["task 2", "task 3", "task 4", "task 5"]:
            releases[prompt].set()
        await pilot.pause(0.2)

        assert app.pending_queue == []


async def test_subagent_fallback_posts_warning_and_still_completes_reply(tmp_path):
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

        app.submit_command("/subagent inspect fallback")
        await pilot.pause(0.2)

        assert app.chat_messages == [
            ChatMessage(role="user", text="/subagent inspect fallback"),
            ChatMessage(
                role="assistant",
                text="Offline local response: inspect fallback",
                model="autumn/offline-tiny",
                participant_name="Autumn Sub Agent 1",
            ),
            ChatMessage(
                role="assistant",
                text=(
                    "Warning: Autumn Sub Agent 1 fell back to autumn/offline-tiny. "
                    "Reason: runtime missing for tiny. "
                    "Capability impact: expected [model_reply]; "
                    "fallback provides [model_reply]; lost [none]."
                ),
            ),
        ]


async def test_subagent_cancel_removes_queued_subagent_by_name(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)
    releases = {f"task {index}": threading.Event() for index in range(1, 6)}
    started: list[str] = []

    class BlockingSubagentRunner:
        def generate(self, messages, model):
            prompt = messages[-1].text
            started.append(prompt)
            releases[prompt].wait(timeout=2)
            return ChatMessage(role="assistant", text=f"answer {prompt}", model=model.name)

    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=BlockingSubagentRunner(),
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        for index in range(1, 6):
            app.submit_command(f"/subagent task {index}")
        await pilot.pause(0.2)

        app.submit_command("/subagent cancel Autumn Sub Agent 5")
        await pilot.pause(0.1)

        assert started == ["task 1", "task 2", "task 3", "task 4"]
        assert app.chat_messages[-3:] == [
            ChatMessage(role="assistant", text="cancelled before start.", participant_name="Autumn Sub Agent 5"),
            ChatMessage(role="user", text="/subagent cancel Autumn Sub Agent 5"),
            ChatMessage(role="assistant", text="Cancelled queued subagent Autumn Sub Agent 5."),
        ]

        for prompt in ["task 1", "task 2", "task 3", "task 4", "task 5"]:
            releases[prompt].set()
        await pilot.pause(0.2)

        assert "task 5" not in started


async def test_subagent_cancel_marks_running_subagent_and_ignores_late_result(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)
    release = threading.Event()

    class BlockingSubagentRunner:
        def generate(self, messages, model):
            release.wait(timeout=2)
            return ChatMessage(role="assistant", text="late answer", model=model.name)

    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=BlockingSubagentRunner(),
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.submit_command("/subagent slow work")
        await pilot.pause(0.1)
        app.submit_command("/subagent cancel Autumn Sub Agent 1")
        await pilot.pause(0.1)

        assert app.chat_messages == [
            ChatMessage(role="user", text="/subagent slow work"),
            ChatMessage(role="assistant", text="cancelled.", participant_name="Autumn Sub Agent 1"),
            ChatMessage(role="user", text="/subagent cancel Autumn Sub Agent 1"),
            ChatMessage(role="assistant", text="Cancelled running subagent Autumn Sub Agent 1."),
        ]

        release.set()
        await pilot.pause(0.2)

        assert app.chat_messages[1] == ChatMessage(
            role="assistant",
            text="cancelled.",
            participant_name="Autumn Sub Agent 1",
        )


async def test_subagent_cancel_unknown_reports_error_without_changing_completed_history(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)

    class FastSubagentRunner:
        def generate(self, messages, model):
            return ChatMessage(role="assistant", text="done", model=model.name)

    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=FastSubagentRunner(),
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.submit_command("/subagent quick work")
        await pilot.pause(0.2)
        completed = app.chat_messages[1]

        app.submit_command("/subagent cancel Autumn Sub Agent 1")
        await pilot.pause(0.1)

        assert app.chat_messages[1] is completed
        assert app.chat_messages[1] == ChatMessage(
            role="assistant",
            text="done",
            model="tiny",
            participant_name="Autumn Sub Agent 1",
        )
        assert app.chat_messages[-2:] == [
            ChatMessage(role="user", text="/subagent cancel Autumn Sub Agent 1"),
            ChatMessage(
                role="assistant",
                text="No running or queued subagent named Autumn Sub Agent 1.",
            ),
        ]


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
        def generate_stream(self, messages, model, *, on_chunk, cancel_event=None):
            assert messages == [ChatMessage(role="user", text="hello")]
            assert model.name == "tiny"
            on_chunk("Installed model answered.")

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


async def test_prompt_local_model_runtime_error_preserves_partial_text_with_interrupted_marker(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)

    class FailingLocalModelRunner:
        def generate_stream(self, messages, model, *, on_chunk, cancel_event=None):
            on_chunk("partial answer")
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

        # No silent fallback substitution (matching #13's Groq behavior): the
        # model is still the real installed local model, and the partial
        # text that streamed in before the crash is kept, not discarded.
        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="partial answer [interrupted: tiny failed: bad model file]",
                model="tiny",
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: tiny (installed default)" in str(status_text)
        assert "autumn/offline-tiny" not in str(status_text)


async def test_prompt_preserves_selected_model_error_response_without_fallback(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)

    class ErrorResponseLocalModelRunner:
        def generate_stream(self, messages, model, *, on_chunk, cancel_event=None):
            on_chunk("Error: context window exceeded")

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


async def test_prompt_receives_async_groq_reply(tmp_path):
    class FakeGroqRunner:
        def generate_stream(
            self, messages, model_name, *, on_chunk, cancel_event=None, on_status=None, on_citation=None
        ):
            assert messages == [ChatMessage(role="user", text="hello")]
            assert model_name == "llama-3.3-70b-versatile"
            for chunk in ["Hi ", "from ", "Groq."]:
                on_chunk(chunk)

    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        groq_runner=FakeGroqRunner(),
        prompt_routing_policy=_groq_policy_with_default(),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        # Empty catalog -> ModelPickerScreen lands first; skip it.
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
            ChatMessage(role="assistant", text="Hi from Groq.", model="llama-3.3-70b-versatile"),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: llama-3.3-70b-versatile (provider available)" in str(status_text)
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "Autumn [llama-3.3-70b-versatile]: Hi from Groq." in str(chat_text)


async def test_prompt_falls_through_when_groq_key_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        # No groq_runner injected: if the (unsigned-in) Groq catalog entries
        # were ever reached, this would fall through to a real GroqRunner()
        # and raise (no GROQ_API_KEY) rather than quietly succeeding -- so
        # this test also proves the fallback never touches Groq at all.
        prompt_routing_policy=groq_policy.build_policy(),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
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
        assert "Fallback: autumn/offline-tiny (offline fallback)" in str(status_text)


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
            {"kind": "chat", "text": "first queued prompt"},
            {"kind": "chat", "text": "second queued prompt"},
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
