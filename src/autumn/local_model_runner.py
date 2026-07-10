"""Runtime boundary for executing installed local model files."""

import subprocess
from pathlib import Path
from typing import Callable

from autumn.models import ChatMessage, LocalModel

RunCommand = Callable[..., subprocess.CompletedProcess[str]]


class LocalModelRunner:
    """Executes installed local models through their configured backend."""

    def __init__(
        self,
        *,
        llama_cli_path: Path | str = "llama-cli",
        run_command: RunCommand = subprocess.run,
    ) -> None:
        self._llama_cli_path = str(llama_cli_path)
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

        completed = self._run_command(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return ChatMessage(role="assistant", text=completed.stdout.strip(), model=model.name)


def _prompt_from_messages(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        speaker = "User" if message.role == "user" else "Assistant"
        lines.append(f"{speaker}: {message.text}")
    lines.append("Assistant:")
    return "\n".join(lines)
