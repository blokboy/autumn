"""Behavioral tests for executing installed local models."""

import subprocess
from pathlib import Path

from autumn.local_model_runner import LocalModelRunner
from autumn.models import ChatMessage, LocalModel


def test_llama_cpp_runner_returns_assistant_message_from_runtime_output(tmp_path):
    model_path = tmp_path / "tiny.gguf"
    model_path.write_bytes(b"fake model")
    model = LocalModel(
        name="tiny",
        backend="llama.cpp",
        path=model_path,
        context_window=2048,
        is_default=True,
    )
    seen_commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        seen_commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="A real-ish local answer.\n", stderr="")

    runner = LocalModelRunner(llama_cli_path=Path("/usr/local/bin/llama-cli"), run_command=fake_run)

    message = runner.generate(
        [
            ChatMessage(role="user", text="hello"),
        ],
        model,
    )

    assert message == ChatMessage(
        role="assistant",
        text="A real-ish local answer.",
        model="tiny",
    )
    assert seen_commands == [
        [
            "/usr/local/bin/llama-cli",
            "-m",
            str(model_path),
            "-c",
            "2048",
            "-p",
            "User: hello\nAssistant:",
        ]
    ]
