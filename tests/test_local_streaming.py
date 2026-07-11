"""Behavioral tests for the local llama.cpp streaming path: incremental
`LocalModelRunner.generate_stream` chunk delivery (against a fake `Popen`),
plus the dashboard-level streaming/cancel/mid-stream-error behavior wired up
in AutumnApp. Mirrors tests/test_groq_streaming.py's structure -- #14 applies
the same chunk/throttle/cancel plumbing #13 established for Groq to local
models, swapping an SDK stream for incremental `Popen` reads."""

import threading

import pytest
from textual.widgets import Input, Static

from autumn import local_models
from autumn.app import AutumnApp
from autumn.local_model_runner import LocalModelRunner, LocalModelRuntimeError
from autumn.models import ChatMessage, LocalModel
from autumn.widgets.command_bar import CommandBar

# --- LocalModelRunner.generate_stream unit tests -----------------------------


class _FakeStdout:
    """Mimics `Popen.stdout`: serves fixed byte chunks one `read1()` call at
    a time (an empty final "chunk" isn't needed -- running out simply
    returns `b""`, the real EOF signal), so tests can assert exactly what
    `generate_stream` does with each incremental read rather than one big
    blocking read."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self.closed = False

    def read1(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeStderr:
    def __init__(self, data: bytes = b""):
        self._data = data
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        data, self._data = self._data, b""
        return data

    def close(self) -> None:
        self.closed = True


class _FakePopen:
    """Mimics the subset of `subprocess.Popen` `generate_stream` depends on:
    `.stdout`/`.stderr` readables, `.wait()` returning the exit code, and
    `.kill()` for the cancel path."""

    def __init__(self, stdout_chunks: list[bytes], stderr: bytes = b"", returncode: int = 0):
        self.stdout = _FakeStdout(stdout_chunks)
        self.stderr = _FakeStderr(stderr)
        self.returncode = returncode
        self.killed = False

    def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True


def _model(tmp_path, context_window: int | None = None) -> LocalModel:
    model_path = tmp_path / "tiny.gguf"
    model_path.write_bytes(b"fake model")
    return LocalModel(name="tiny", backend="llama.cpp", path=model_path, context_window=context_window)


def test_generate_stream_calls_on_chunk_for_each_stdout_read(tmp_path):
    model = _model(tmp_path, context_window=2048)
    process = _FakePopen([b"Hello", b" world"])
    seen_commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        seen_commands.append(command)
        return process

    runner = LocalModelRunner(llama_cli_path="/usr/local/bin/llama-cli", popen=fake_popen)

    received: list[str] = []
    runner.generate_stream(
        [ChatMessage(role="user", text="hello")],
        model,
        on_chunk=received.append,
    )

    assert received == ["Hello", " world"]
    assert seen_commands == [
        [
            "/usr/local/bin/llama-cli",
            "-m",
            str(model.path),
            "-c",
            "2048",
            "-p",
            "User: hello\nAssistant:",
        ]
    ]


def test_generate_stream_decodes_safely_across_multi_byte_utf8_chunk_boundary(tmp_path):
    model = _model(tmp_path)
    # "café" -> b"caf\xc3\xa9"; split the 2-byte é across two reads to prove
    # the incremental decoder buffers the incomplete trailing byte instead of
    # emitting a mangled/replacement character.
    text = "café"
    encoded = text.encode("utf-8")
    split_at = len(encoded) - 1
    process = _FakePopen([encoded[:split_at], encoded[split_at:]])

    runner = LocalModelRunner(popen=lambda command, **kwargs: process)

    received: list[str] = []
    runner.generate_stream([ChatMessage(role="user", text="hi")], model, on_chunk=received.append)

    assert "".join(received) == text
    assert "�" not in "".join(received)


def test_generate_stream_stops_reading_and_kills_process_on_cancel(tmp_path):
    model = _model(tmp_path)
    process = _FakePopen([b"Hello", b" world", b"!"])
    cancel_event = threading.Event()

    runner = LocalModelRunner(popen=lambda command, **kwargs: process)

    received: list[str] = []

    def on_chunk(text: str) -> None:
        received.append(text)
        if text == "Hello":
            cancel_event.set()

    runner.generate_stream(
        [ChatMessage(role="user", text="hi")],
        model,
        on_chunk=on_chunk,
        cancel_event=cancel_event,
    )

    assert received == ["Hello"]
    assert process.killed is True
    assert process.stdout.closed is True


def test_generate_stream_raises_local_model_runtime_error_on_non_zero_exit(tmp_path):
    model = _model(tmp_path)
    process = _FakePopen([b"partial output"], stderr=b"bad model file", returncode=2)

    runner = LocalModelRunner(popen=lambda command, **kwargs: process)

    received: list[str] = []
    with pytest.raises(LocalModelRuntimeError, match="tiny failed: bad model file"):
        runner.generate_stream([ChatMessage(role="user", text="hi")], model, on_chunk=received.append)

    # Chunks that arrived before the crash still reached on_chunk -- the
    # caller's closure is the only place that partial text lives.
    assert received == ["partial output"]


def test_generate_stream_raises_local_model_runtime_error_when_llama_cli_missing(tmp_path):
    model = _model(tmp_path)

    def fake_popen(command, **kwargs):
        raise FileNotFoundError()

    runner = LocalModelRunner(popen=fake_popen)

    with pytest.raises(LocalModelRuntimeError, match="tiny failed: llama-cli not found"):
        runner.generate_stream([ChatMessage(role="user", text="hi")], model, on_chunk=lambda text: None)


# --- AutumnApp-level streaming/cancel/error tests -----------------------------


class _ScriptedLocalModelRunner:
    """A fake LocalModelRunner that emits one chunk per `release_next()`
    call, blocking in between -- lets tests deterministically observe
    partial streamed state and interleave app-level actions (like pressing
    escape) between chunks, instead of racing real wall-clock timing. Mirrors
    test_groq_streaming.py's `_ScriptedGroqRunner`."""

    def __init__(self, chunks: list[str], error: Exception | None = None):
        self._chunks = chunks
        self._error = error
        self._gates = [threading.Event() for _ in chunks]
        self._released = 0

    def release_next(self) -> None:
        self._gates[self._released].set()
        self._released += 1

    def generate_stream(self, messages, model, *, on_chunk, cancel_event=None):
        for chunk, gate in zip(self._chunks, self._gates):
            gate.wait(timeout=5)
            if cancel_event is not None and cancel_event.is_set():
                return
            on_chunk(chunk)
        if self._error is not None:
            raise self._error


def _install_local_model(tmp_path):
    source_model = tmp_path / "source.gguf"
    source_model.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source_model)
    return catalog_root


async def _land_on_dashboard_and_submit(pilot, prompt: str) -> None:
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()

    await pilot.press(":")
    await pilot.pause()
    pilot.app.screen.query_one(CommandBar).query_one(Input).insert_text_at_cursor(prompt)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(0.05)


async def test_local_stream_updates_message_incrementally(tmp_path):
    catalog_root = _install_local_model(tmp_path)
    runner = _ScriptedLocalModelRunner(["Hello", " world", "!"])
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=runner,
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "hello")

        # Placeholder assistant message inserted before any chunk streams in.
        assert app.chat_messages[-1] == ChatMessage(role="assistant", text="", model="tiny")

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
            ChatMessage(role="assistant", text="Hello world!", model="tiny"),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: tiny (installed default)" in str(status_text)
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "Autumn [tiny]: Hello world!" in str(chat_text)


async def test_local_stream_cancel_kills_subprocess_and_preserves_partial_text(tmp_path):
    catalog_root = _install_local_model(tmp_path)
    runner = _ScriptedLocalModelRunner(["Hello", " world"])
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=runner,
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "hello")

        runner.release_next()
        await pilot.pause(0.05)
        assert app.chat_messages[-1].text == "Hello"

        # Cancel the in-flight stream (DashboardScreen's escape binding) --
        # this is the same generic cancel path #13 wired up for Groq; it
        # works unchanged here because _run_local_stream sets
        # `_active_cancel_event` the same way _run_groq_stream does.
        await pilot.press("escape")
        await pilot.pause(0.05)

        # Unblock the runner's loop so it observes the now-set cancel_event
        # and returns instead of emitting the second chunk.
        runner.release_next()
        await pilot.pause(0.15)

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(role="assistant", text="Hello [stopped]", model="tiny"),
        ]
        # Status reflects the real installed local choice, not a builtin
        # offline-tiny fallback -- cancellation isn't treated as an error to
        # fall back from.
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: tiny (installed default)" in str(status_text)
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "Autumn [tiny]: Hello [stopped]" in str(chat_text)

        # The cancel event is cleared once its generation finishes, so a
        # later cancel press with nothing in flight is a harmless no-op.
        assert app._active_cancel_event is None


async def test_local_stream_mid_stream_crash_preserves_partial_text_with_interrupted_marker(tmp_path):
    catalog_root = _install_local_model(tmp_path)
    error = LocalModelRuntimeError("tiny failed: exit code 1")
    runner = _ScriptedLocalModelRunner(["Hello"], error=error)
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=runner,
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "hello")

        runner.release_next()
        await pilot.pause(0.15)

        # No silent fallback substitution (matching #13's Groq behavior): the
        # model is still the real installed local model, and the partial
        # text that streamed in is kept, not discarded for a fresh
        # offline-tiny reply.
        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="Hello [interrupted: tiny failed: exit code 1]",
                model="tiny",
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: tiny (installed default)" in str(status_text)
        assert "autumn/offline-tiny" not in str(status_text)


async def test_local_stream_error_with_no_text_yet_is_just_the_marker(tmp_path):
    catalog_root = _install_local_model(tmp_path)
    error = LocalModelRuntimeError("tiny failed: llama-cli not found")
    runner = _ScriptedLocalModelRunner([], error=error)
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=catalog_root,
        local_model_runner=runner,
        is_model_runtime_available=lambda model: True,
    )

    async with app.run_test() as pilot:
        await _land_on_dashboard_and_submit(pilot, "hello")
        await pilot.pause(0.15)

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="[interrupted: tiny failed: llama-cli not found]",
                model="tiny",
            ),
        ]
