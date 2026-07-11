"""Policy-based prompt routing for dashboard chat replies."""

from pathlib import Path
import shutil
from typing import Callable

import catalog, local_llm
from models import CatalogEntry, LocalModel, ModelChoice, PromptRoutingPolicy

RuntimeAvailability = Callable[[LocalModel], bool]


def choose_model(
    *,
    prompt: str,
    catalog_root: Path,
    available_models: list[LocalModel] | None = None,
    is_runtime_available: RuntimeAvailability | None = None,
    policy: PromptRoutingPolicy | None = None,
) -> ModelChoice:
    entries = catalog.build_entries(catalog_root, local_override=available_models, policy=policy)
    runtime_available = is_runtime_available or _is_runtime_available
    unavailable_reasons: list[str] = []

    for entry in _default_first(entries):
        available, reason = _availability(entry, runtime_available)
        if available:
            return _choice(entry, reason)
        unavailable_reasons.append(reason)

    return ModelChoice(
        name=local_llm.OFFLINE_TINY_MODEL,
        backend="builtin",
        path=None,
        reason=unavailable_reasons[0] if unavailable_reasons else "offline fallback",
    )


def _default_first(entries: list[CatalogEntry]) -> list[CatalogEntry]:
    """Floats whichever entry is marked `is_default` -- local or provider --
    to the front of the candidate order, so an explicit default always wins
    regardless of catalog group or provider priority."""
    default = catalog.default_entry(entries)
    if default is None:
        return entries
    return [default, *[entry for entry in entries if entry is not default]]


def _availability(entry: CatalogEntry, runtime_available: RuntimeAvailability) -> tuple[bool, str]:
    if entry.backend == "provider":
        # Inclusion in the catalog already means catalog.build_entries found
        # this provider model enabled and its account signed in.
        return True, "provider available"

    model = LocalModel(
        name=entry.name,
        backend=entry.backend,
        path=entry.path,
        context_window=entry.context_window,
        is_default=entry.is_default,
    )
    if runtime_available(model):
        return True, "installed default" if entry.is_default else "installed local"
    return False, f"runtime missing for {entry.name}"


def _choice(entry: CatalogEntry, reason: str) -> ModelChoice:
    return ModelChoice(
        name=entry.name,
        backend=entry.backend,
        path=entry.path,
        reason=reason,
        context_window=entry.context_window,
        provider=entry.provider,
        account_id=entry.account_id,
    )


def _is_runtime_available(model: LocalModel) -> bool:
    if model.backend == "llama.cpp":
        return model.path.exists() and shutil.which("llama-cli") is not None
    return False
