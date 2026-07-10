"""Tests for prompt model routing policy."""

from autumn import model_router
from autumn.models import LocalModel, PromptRoutingPolicy, ProviderAccount, ProviderModel


def test_router_uses_builtin_offline_model_when_no_catalog_default_exists(tmp_path):
    choice = model_router.choose_model(prompt="hello", catalog_root=tmp_path / "models")

    assert choice.name == "autumn/offline-tiny"
    assert choice.backend == "builtin"
    assert choice.path is None
    assert choice.reason == "offline fallback"


def test_router_prefers_installed_default_local_model(tmp_path):
    model_path = tmp_path / "models" / "tiny" / "tiny.gguf"
    choice = model_router.choose_model(
        prompt="hello",
        catalog_root=tmp_path / "models",
        available_models=[
            LocalModel(
                name="tiny",
                backend="llama.cpp",
                path=model_path,
                context_window=2048,
                is_default=True,
            )
        ],
        is_runtime_available=lambda model: True,
    )

    assert choice.name == "tiny"
    assert choice.backend == "llama.cpp"
    assert choice.path == model_path
    assert choice.reason == "installed default"


def test_router_falls_back_when_default_model_runtime_is_missing(tmp_path):
    model_path = tmp_path / "models" / "tiny" / "tiny.gguf"
    choice = model_router.choose_model(
        prompt="hello",
        catalog_root=tmp_path / "models",
        available_models=[
            LocalModel(
                name="tiny",
                backend="llama.cpp",
                path=model_path,
                context_window=2048,
                is_default=True,
            )
        ],
        is_runtime_available=lambda model: False,
    )

    assert choice.name == "autumn/offline-tiny"
    assert choice.backend == "builtin"
    assert choice.path is None
    assert choice.reason == "runtime missing for tiny"


def test_router_uses_available_provider_candidate_when_no_local_model_exists(tmp_path):
    choice = model_router.choose_model(
        prompt="hello",
        catalog_root=tmp_path / "models",
        policy=PromptRoutingPolicy(
            provider_accounts=[
                ProviderAccount(
                    provider="claude",
                    account_id="personal",
                    display_name="Personal Claude",
                    is_signed_in=True,
                )
            ],
            provider_models=[
                ProviderModel(
                    name="claude/sonnet",
                    provider="claude",
                    account_id="personal",
                    is_enabled=True,
                    priority=10,
                )
            ],
        ),
    )

    assert choice.name == "claude/sonnet"
    assert choice.backend == "provider"
    assert choice.path is None
    assert choice.provider == "claude"
    assert choice.account_id == "personal"
    assert choice.reason == "provider available"


def test_router_prefers_runnable_local_default_over_provider_candidate(tmp_path):
    model_path = tmp_path / "models" / "tiny" / "tiny.gguf"
    choice = model_router.choose_model(
        prompt="hello",
        catalog_root=tmp_path / "models",
        available_models=[
            LocalModel(
                name="tiny",
                backend="llama.cpp",
                path=model_path,
                context_window=2048,
                is_default=True,
            )
        ],
        is_runtime_available=lambda model: True,
        policy=PromptRoutingPolicy(
            provider_accounts=[
                ProviderAccount(provider="claude", account_id="personal", is_signed_in=True)
            ],
            provider_models=[
                ProviderModel(
                    name="claude/sonnet",
                    provider="claude",
                    account_id="personal",
                    priority=1,
                )
            ],
        ),
    )

    assert choice.name == "tiny"
    assert choice.backend == "llama.cpp"
    assert choice.provider is None
    assert choice.account_id is None
