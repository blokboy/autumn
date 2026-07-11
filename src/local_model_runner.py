"""Runtime boundary for executing installed local model files."""

import codecs
import os
import subprocess
import threading
from pathlib import Path
from typing import Callable

from models import ChatMessage, LocalModel

RunCommand = Callable[..., subprocess.CompletedProcess[str]]
Popen = Callable[..., subprocess.Popen]

# Size (in bytes) of each raw read off the subprocess's stdout pipe while
# streaming. Small enough to keep replies feeling responsive chunk-to-chunk,
# large enough to avoid a syscall per byte. Decoding happens through an
# incremental UTF-8 decoder (see `generate_stream`) specifically so a read
# landing mid-multi-byte-character never produces mangled output -- the
# decoder buffers incomplete trailing bytes until the next read completes
# them.
_STREAM_READ_CHUNK_BYTES = 4096


class LocalModelRuntimeError(RuntimeError):
    """Raised when an installed local model runtime cannot produce a reply."""


class LocalModelRunner:
    """Executes installed local models through their configured backend."""

    def __init__(
        self,
        *,
        llama_cli_path: Path | str | None = None,
        run_command: RunCommand = subprocess.run,
        popen: Popen = subprocess.Popen,
    ) -> None:
        self._llama_cli_path = str(llama_cli_path or os.environ.get("AUTUMN_LLAMA_CLI", "llama-cli"))
        self._run_command = run_command
        self._popen = popen

    def generate(self, messages: list[ChatMessage], model: LocalModel) -> ChatMessage:
        command = self._command_for(messages, model)

        try:
            completed = self._run_command(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = _error_detail(exc.stderr) or f"exit code {exc.returncode}"
            raise LocalModelRuntimeError(f"{model.name} failed: {detail}") from exc
        except FileNotFoundError as exc:
            raise LocalModelRuntimeError(f"{model.name} failed: llama-cli not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise LocalModelRuntimeError(f"{model.name} failed: runtime timed out") from exc
        return ChatMessage(role="assistant", text=completed.stdout.strip(), model=model.name)

    def generate_stream(
        self,
        messages: list[ChatMessage],
        model: LocalModel,
        *,
        on_chunk: Callable[[str], None],
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Streams a reply incrementally from the `llama-cli` subprocess,
        calling `on_chunk` with each decoded piece of text as it arrives.

        Has no return value -- mirrors `GroqRunner.generate_stream` exactly:
        the caller's `on_chunk` closure is the only place accumulated text
        lives, so on a mid-stream cancellation or crash, whatever text
        already reached `on_chunk` is exactly what the caller already has.

        `cancel_event` (if given) is checked between reads. Once set, the
        subprocess is killed and reading stops immediately -- this is not
        treated as an error, cancellation returns normally, same contract as
        the Groq path.

        A non-zero exit or a missing `llama-cli` binary is re-raised as
        `LocalModelRuntimeError`, same message shape as `generate`.
        """
        if model.backend != "llama.cpp":
            raise ValueError(f"Unsupported local model backend: {model.backend}")

        command = self._command_for(messages, model)

        try:
            process = self._popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise LocalModelRuntimeError(f"{model.name} failed: llama-cli not found") from exc

        # Drained on a background thread rather than read at the end: if
        # llama-cli writes enough diagnostic/progress output to stderr to
        # fill the OS pipe buffer while this method is only reading stdout,
        # the child would block on its stderr write and the stdout read loop
        # below would hang forever waiting for tokens that will never come.
        stderr_chunks: list[bytes] = []
        stderr = process.stderr
        stderr_thread: threading.Thread | None = None
        if stderr is not None:
            stderr_thread = threading.Thread(target=_drain, args=(stderr, stderr_chunks), daemon=True)
            stderr_thread.start()

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        stdout = process.stdout
        cancelled = False
        if stdout is not None:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                raw = stdout.read1(_STREAM_READ_CHUNK_BYTES) if hasattr(stdout, "read1") else stdout.read(
                    _STREAM_READ_CHUNK_BYTES
                )
                if not raw:
                    break
                text = decoder.decode(raw)
                if text:
                    on_chunk(text)

        if cancelled:
            process.kill()
            stdout.close()
            process.wait()
            if stderr_thread is not None:
                stderr_thread.join(timeout=5)
            return

        if stdout is not None:
            stdout.close()

        # Flush any buffered partial multi-byte sequence left in the
        # decoder once the stream is known to have ended (EOF).
        tail = decoder.decode(b"", final=True)
        if tail:
            on_chunk(tail)

        returncode = process.wait()
        if stderr_thread is not None:
            stderr_thread.join(timeout=5)

        if returncode != 0:
            detail = _error_detail(b"".join(stderr_chunks)) or f"exit code {returncode}"
            raise LocalModelRuntimeError(f"{model.name} failed: {detail}")

    def _command_for(self, messages: list[ChatMessage], model: LocalModel) -> list[str]:
        if model.backend != "llama.cpp":
            raise ValueError(f"Unsupported local model backend: {model.backend}")

        command = [
            self._llama_cli_path,
            "-m",
            str(model.path),
        ]
        if model.context_window is not None:
            command.extend(["-c", str(model.context_window)])
        command.extend(["-p", _prompt_from_messages(messages)])
        return command


def _drain(stream, chunks: list[bytes]) -> None:
    """Reads `stream` to EOF in the background, appending each chunk read to
    `chunks`. Used to drain the subprocess's stderr pipe concurrently with
    the stdout read loop in `generate_stream`, so a chatty subprocess can
    never deadlock it (see the comment at its call site)."""
    while True:
        chunk = stream.read(_STREAM_READ_CHUNK_BYTES)
        if not chunk:
            break
        chunks.append(chunk)
    stream.close()


def _prompt_from_messages(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        speaker = "User" if message.role == "user" else "Assistant"
        lines.append(f"{speaker}: {message.text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _error_detail(stderr: str | bytes | None) -> str:
    if stderr is None:
        return ""
    if isinstance(stderr, bytes):
        return stderr.decode(errors="replace").strip()
    return stderr.strip()
