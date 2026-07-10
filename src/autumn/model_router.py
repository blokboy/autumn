"""Policy-based prompt routing for dashboard chat replies."""

from pathlib import Path
import shutil
from typing import Callable

from autumn import local_llm, local_models
from autumn.models import LocalModel, ModelChoice, PromptRoutingPolicy, ProviderModel

RuntimeAvailability = Callable[[LocalModel], bool]


def choose_model(
    *,
    prompt: str,
    catalog_root: Path,
    available_models: list[LocalModel] | None = None,
    is_runtime_available: RuntimeAvailability | None = None,
    policy: PromptRoutingPolicy | None = None,
) -> ModelChoice:
    models = available_models if available_models is not None else local_models.list_models(catalog_root)
    runtime_available = is_runtime_available or _is_runtime_available
    unavailable_reasons: list[str] = []

    for model in _local_candidates(models):
        if runtime_available(model):
            return _local_choice(model, "installed default" if model.is_default else "installed local")
        unavailable_reasons.append(f"runtime missing for {model.name}")

    return ModelChoice(
        name=local_llm.OFFLINE_TINY_MODEL,
        backend="builtin",
        path=None,
        reason=unavailable_reasons[0] if unavailable_reasons else "offline fallback",
    )


def _local_candidates(models: list[LocalModel]) -> list[LocalModel]:
    installed_default = next((model for model in models if model.is_default), None)
    if installed_default is None:
        return models
    return [installed_default, *[model for model in models if model is not installed_default]]


def _local_choice(model: LocalModel, reason: str) -> ModelChoice:
    return ModelChoice(
        name=model.name,
        backend=model.backend,
        path=model.path,
        reason=reason,
        context_window=model.context_window,
    )


def _is_runtime_available(model: LocalModel) -> bool:
    if model.backend == "llama.cpp":
        return model.path.exists() and shutil.which("llama-cli") is not None
    return False


def _provider_choice(provider_model: ProviderModel) -> ModelChoice:
    return ModelChoice(
        name=provider_model.name,
        backend="provider",
        path=None,
        reason="provider available",
        provider=provider_model.provider,
        account_id=provider_model.account_id,
    )


def _choose_provider_model(policy: PromptRoutingPolicy | None) -> ProviderModel | None:
    if policy is None:
        return None
    signed_in_accounts = {
        (account.provider, account.account_id)
        for account in policy.provider_accounts
        if account.is_signed_in
    }
    candidates = [
        model
        for model in policy.provider_models
        if model.is_enabled and (model.provider, model.account_id) in signed_in_accounts
    ]
    return min(candidates, key=lambda model: model.priority, default=None)
