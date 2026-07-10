"""Runtime boundary for executing installed local model files."""

import os
import subprocess
from pathlib import Path
from typing import Callable

from autumn.models import ChatMessage, LocalModel

RunCommand = Callable[..., subprocess.CompletedProcess[str]]


class LocalModelRuntimeError(RuntimeError):
    """Raised when an installed local model runtime cannot produce a reply."""


class LocalModelRunner:
    """Executes installed local models through their configured backend."""

    def __init__(
        self,
        *,
        llama_cli_path: Path | str | None = None,
        run_command: RunCommand = subprocess.run,
    ) -> None:
        self._llama_cli_path = str(llama_cli_path or os.environ.get("AUTUMN_LLAMA_CLI", "llama-cli"))
        self._run_command = run_command

    def generate(self, messages: list[ChatMessage], model: LocalModel) -> ChatMessage:
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
