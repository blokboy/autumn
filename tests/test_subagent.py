from pathlib import Path

import cli
import local_models
import subagent
from models import ChatMessage, ModelChoice, PromptRoutingPolicy, ProviderAccount, ProviderModel


class _FakeLocalRunner:
    def __init__(self):
        self.messages = None
        self.model = None

    def generate(self, messages, model):
        self.messages = messages
        self.model = model
        return ChatMessage(role="assistant", text="local answer", model=model.name)


class _FakeGroqRunner:
    def __init__(self):
        self.messages = None
        self.model_name = None

    def generate(self, messages, model_name):
        self.messages = messages
        self.model_name = model_name
        return ChatMessage(role="assistant", text="groq answer", model=model_name)


def _install_model(tmp_path: Path) -> Path:
    source = tmp_path / "source.gguf"
    source.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source, context_window=4096)
    return catalog_root


def test_subagent_uses_explicit_prompt_and_system_instruction_only(tmp_path):
    catalog_root = _install_model(tmp_path)
    runner = _FakeLocalRunner()

    result = subagent.run_subagent(
        "summarize this",
        catalog_root=catalog_root,
        is_runtime_available=lambda model: True,
        local_model_runner=runner,
    )

    assert result.answer == "local answer"
    assert result.choice.name == "tiny"
    assert result.choice.path is not None
    assert result.messages == [
        ChatMessage(role="system", text=subagent.SUBAGENT_SYSTEM_INSTRUCTION),
        ChatMessage(role="user", text="summarize this"),
    ]
    assert runner.messages == result.messages
    assert runner.model.name == "tiny"
    assert runner.model.context_window == 4096


def test_subagent_snapshots_selected_model_for_invocation(tmp_path):
    catalog_root = _install_model(tmp_path)
    runner = _FakeLocalRunner()
    seen_paths: list[Path] = []

    def runtime_available(model):
        seen_paths.append(model.path)
        local_models.remove_model(catalog_root, "tiny")
        return True

    result = subagent.run_subagent(
        "answer once",
        catalog_root=catalog_root,
        is_runtime_available=runtime_available,
        local_model_runner=runner,
    )

    assert result.choice.name == "tiny"
    assert result.choice.path == seen_paths[0]
    assert runner.model.path == seen_paths[0]
    assert result.answer == "local answer"


def test_subagent_uses_same_provider_policy_routing_as_chat(tmp_path):
    runner = _FakeGroqRunner()
    policy = PromptRoutingPolicy(
        provider_accounts=[ProviderAccount(provider="groq", account_id="default", is_signed_in=True)],
        provider_models=[
            ProviderModel(
                name="llama-3.3-70b-versatile",
                provider="groq",
                account_id="default",
                is_default=True,
            )
        ],
    )

    result = subagent.run_subagent(
        "hello",
        catalog_root=tmp_path / "models",
        policy=policy,
        groq_runner=runner,
    )

    assert result.answer == "groq answer"
    assert result.choice.name == "llama-3.3-70b-versatile"
    assert result.choice.provider == "groq"
    assert runner.messages == [
        ChatMessage(role="system", text=subagent.SUBAGENT_SYSTEM_INSTRUCTION),
        ChatMessage(role="user", text="hello"),
    ]
    assert runner.model_name == "llama-3.3-70b-versatile"


def test_subagent_falls_back_for_provider_without_execution_path(tmp_path):
    policy = PromptRoutingPolicy(
        provider_accounts=[ProviderAccount(provider="claude", account_id="personal", is_signed_in=True)],
        provider_models=[
            ProviderModel(
                name="claude/sonnet",
                provider="claude",
                account_id="personal",
                is_default=True,
            )
        ],
    )

    result = subagent.run_subagent("hello", catalog_root=tmp_path / "models", policy=policy)

    assert result.routed_choice is not None
    assert result.routed_choice.name == "claude/sonnet"
    assert result.choice.name == "autumn/offline-tiny"
    assert result.answer == "Offline local response: hello"
    assert result.fallback_warning == subagent.SubagentFallbackWarning(
        original_model="claude/sonnet",
        fallback_model="autumn/offline-tiny",
        reason="provider claude/sonnet not executable yet",
        expected_capabilities=("model_reply",),
        fallback_capabilities=("model_reply",),
        lost_capabilities=(),
    )


def test_subagent_fallback_warning_tracks_future_capability_loss(tmp_path):
    policy = PromptRoutingPolicy(
        provider_accounts=[ProviderAccount(provider="claude", account_id="personal", is_signed_in=True)],
        provider_models=[
            ProviderModel(
                name="claude/sonnet",
                provider="claude",
                account_id="personal",
                is_default=True,
            )
        ],
    )

    result = subagent.run_subagent(
        "hello",
        catalog_root=tmp_path / "models",
        policy=policy,
        expected_capabilities=subagent.SubagentCapabilities(
            model_reply=True,
            tools=True,
            filesystem_access=True,
            shell_access=True,
            recursive_subagents=True,
            gepa_run_launching=True,
        ),
    )

    assert result.fallback_warning is not None
    assert result.fallback_warning.expected_capabilities == (
        "model_reply",
        "tools",
        "filesystem_access",
        "shell_access",
        "recursive_subagents",
        "gepa_run_launching",
    )
    assert result.fallback_warning.fallback_capabilities == ("model_reply",)
    assert result.fallback_warning.lost_capabilities == (
        "tools",
        "filesystem_access",
        "shell_access",
        "recursive_subagents",
        "gepa_run_launching",
    )


def test_subagent_cli_prints_only_final_answer(monkeypatch, capsys):
    def fake_run_subagent(prompt, *, catalog_root, policy):
        assert prompt == "do the thing"
        return subagent.SubagentResult(
            prompt=prompt,
            answer="final answer",
            choice=ModelChoice(
                name="autumn/offline-tiny",
                backend="builtin",
                path=None,
                reason="offline fallback",
            ),
            messages=[ChatMessage(role="user", text=prompt)],
        )

    monkeypatch.setattr(subagent, "run_subagent", fake_run_subagent)

    result = cli.main(["subagent", "do the thing"])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "final answer\n"
    assert captured.err == ""


def test_subagent_subcommand_is_registered():
    parser = cli._build_parser()

    args = parser.parse_args(["subagent", "hello"])

    assert args.command == "subagent"
    assert args.prompt == "hello"
