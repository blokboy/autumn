"""Behavioral tests for the dashboard-level streaming/cancel/mid-stream-error
behavior wired up in AutumnApp, using a scripted fake `GroqRunner`.

`GroqRunner.generate_stream`'s own unit-level mechanics (chunk delivery,
cancel, mid-stream error) now live in test_groq_tool_calls.py, since
generate_stream's real streaming only happens on the tool round's second
call -- see docs/prd/chat-search-tools.md, "Tool-call round"."""

import threading

import pytest
from textual.widgets import Input, Static

from app import AutumnApp
from groq_runner import GroqRuntimeError
from models import ChatMessage, PromptRoutingPolicy, ProviderAccount, ProviderModel
from widgets.command_bar import CommandBar

# --- AutumnApp-level streaming/cancel/error tests -----------------------------


class _ScriptedGroqRunner:
    """A fake GroqRunner that emits one chunk per `release_next()` call,
    blocking in between -- lets tests deterministically observe partial
    streamed state and interleave app-level actions (like pressing escape)
    between chunks, instead of racing real wall-clock timing."""

    def __init__(self, chunks: list[str], error: Exception | None = None):
        self._chunks = chunks
        self._error = error
        self._gates = [threading.Event() for _ in chunks]
        self._released = 0

    def release_next(self) -> None:
        self._gates[self._released].set()
        self._released += 1

    def generate_stream(self, messages, model_name, *, on_chunk, cancel_event=None, on_status=None, on_citation=None):
        for chunk, gate in zip(self._chunks, self._gates):
            gate.wait(timeout=5)
            if cancel_event is not None and cancel_event.is_set():
                return
            on_chunk(chunk)
        if self._error is not None:
            raise self._error


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


async def test_groq_stream_updates_message_incrementally(tmp_path):
    runner = _ScriptedGroqRunner(["Hello", " world", "!"])
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        groq_runner=runner,
        prompt_routing_policy=_groq_policy_with_default(),
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "hello")

        # Placeholder assistant message inserted before any chunk streams in.
        assert app.chat_messages[-1] == ChatMessage(
            role="assistant", text="", model="llama-3.3-70b-versatile"
        )

        runner.release_next()
        await pilot.pause(0.05)
        assert app.chat_messages[-1].text == "Hello"

        runner.release_next()
        await pilot.pause(0.05)
        assert app.chat_messages[-1].text == "Hello world"

        runner.release_next()
        await pilot.pause(0.15)  # let the final (always-happens) refresh land

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(role="assistant", text="Hello world!", model="llama-3.3-70b-versatile"),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: llama-3.3-70b-versatile (provider available)" in str(status_text)
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "Autumn [llama-3.3-70b-versatile]: Hello world!" in str(chat_text)


async def test_groq_stream_cancel_preserves_partial_text_with_stopped_marker(tmp_path):
    runner = _ScriptedGroqRunner(["Hello", " world"])
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        groq_runner=runner,
        prompt_routing_policy=_groq_policy_with_default(),
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "hello")

        runner.release_next()
        await pilot.pause(0.05)
        assert app.chat_messages[-1].text == "Hello"

        # Cancel the in-flight stream (DashboardScreen's escape binding).
        await pilot.press("escape")
        await pilot.pause(0.05)

        # Unblock the runner's loop so it observes the now-set cancel_event
        # and returns instead of emitting the second chunk.
        runner.release_next()
        await pilot.pause(0.15)

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant", text="Hello [stopped]", model="llama-3.3-70b-versatile"
            ),
        ]
        # Status reflects the real Groq choice, not a builtin/offline-tiny
        # fallback -- cancellation isn't treated as an error to fall back from.
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: llama-3.3-70b-versatile (provider available)" in str(status_text)
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "Autumn [llama-3.3-70b-versatile]: Hello [stopped]" in str(chat_text)

        # The cancel event is cleared once its generation finishes, so a
        # later cancel press with nothing in flight is a harmless no-op.
        assert app._active_cancel_event is None


async def test_groq_stream_mid_stream_error_preserves_partial_text_with_interrupted_marker(tmp_path):
    error = GroqRuntimeError("llama-3.3-70b-versatile failed: connection error")
    runner = _ScriptedGroqRunner(["Hello"], error=error)
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        groq_runner=runner,
        prompt_routing_policy=_groq_policy_with_default(),
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "hello")

        runner.release_next()
        await pilot.pause(0.15)

        # No silent fallback substitution: the model is still the real Groq
        # model, and the partial text that streamed in is kept, not discarded.
        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="Hello [interrupted: llama-3.3-70b-versatile failed: connection error]",
                model="llama-3.3-70b-versatile",
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: llama-3.3-70b-versatile (provider available)" in str(status_text)
        assert "autumn/offline-tiny" not in str(status_text)


async def test_groq_stream_error_with_no_text_yet_is_just_the_marker(tmp_path):
    error = GroqRuntimeError("llama-3.3-70b-versatile failed: connection error")
    runner = _ScriptedGroqRunner([], error=error)
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        groq_runner=runner,
        prompt_routing_policy=_groq_policy_with_default(),
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "hello")
        await pilot.pause(0.15)

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="[interrupted: llama-3.3-70b-versatile failed: connection error]",
                model="llama-3.3-70b-versatile",
            ),
        ]
