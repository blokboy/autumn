"""Behavioral tests for executing installed local models."""

import subprocess
from pathlib import Path

import pytest

from autumn.local_model_runner import LocalModelRunner, LocalModelRuntimeError
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


def test_llama_cpp_runner_reports_runtime_failure(tmp_path):
    model_path = tmp_path / "tiny.gguf"
    model_path.write_bytes(b"fake model")
    model = LocalModel(name="tiny", backend="llama.cpp", path=model_path)

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=command,
            stderr="bad model file",
        )

    runner = LocalModelRunner(run_command=fake_run)

    with pytest.raises(LocalModelRuntimeError, match="tiny failed: bad model file"):
        runner.generate([ChatMessage(role="user", text="hello")], model)


def test_llama_cpp_runner_uses_env_configured_cli_path(tmp_path, monkeypatch):
    model_path = tmp_path / "tiny.gguf"
    model_path.write_bytes(b"fake model")
    model = LocalModel(name="tiny", backend="llama.cpp", path=model_path)
    seen_commands: list[list[str]] = []
    monkeypatch.setenv("AUTUMN_LLAMA_CLI", "/opt/llama/bin/llama-cli")

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        seen_commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="hello\n", stderr="")

    runner = LocalModelRunner(run_command=fake_run)

    runner.generate([ChatMessage(role="user", text="hello")], model)

    assert seen_commands[0][0] == "/opt/llama/bin/llama-cli"
