"""Unified view over Autumn's model catalog: installed local models plus any
eligible provider-backed candidates from a `PromptRoutingPolicy`, as one flat
list of `CatalogEntry` addressed by (group, name). `model_router.choose_model`
and the Models tab (`ModelCatalogView`) both build their view of "what's in
the catalog" from `build_entries` here, so they can never drift apart.

Only the "Local" group has real persisted entries today -- `set_default` only
knows how to persist a local model's default flag (via `local_models.py`'s
manifest). Provider groups are populated straight from whatever
`PromptRoutingPolicy` is passed in (no persistence of their own yet), so a
future ticket wiring up a real provider (e.g. Groq) can mark one of its
`ProviderModel` entries `is_default=True` itself and have it participate in
the same fallback chain, without needing catalog.py to change shape again.
"""

from pathlib import Path

from autumn import local_models
from autumn.models import CatalogEntry, LocalModel, PromptRoutingPolicy

LOCAL_GROUP = "Local"


def build_entries(
    catalog_root: Path,
    *,
    local_override: list[LocalModel] | None = None,
    policy: PromptRoutingPolicy | None = None,
) -> list[CatalogEntry]:
    """Local entries first (in `local_models.list_models` order), then
    eligible provider entries ordered by priority -- callers needing a
    specific entry to be tried first (an explicit default) reorder this
    themselves rather than relying on this base ordering."""
    models = local_override if local_override is not None else local_models.list_models(catalog_root)
    entries = [_local_entry(model) for model in models]
    entries.extend(_eligible_provider_entries(policy))
    return entries


def _local_entry(model: LocalModel) -> CatalogEntry:
    return CatalogEntry(
        group=LOCAL_GROUP,
        name=model.name,
        backend=model.backend,
        path=model.path,
        context_window=model.context_window,
        is_default=model.is_default,
    )


def _eligible_provider_entries(policy: PromptRoutingPolicy | None) -> list[CatalogEntry]:
    if policy is None:
        return []
    signed_in_accounts = {
        (account.provider, account.account_id)
        for account in policy.provider_accounts
        if account.is_signed_in
    }
    eligible = [
        model
        for model in policy.provider_models
        if model.is_enabled and (model.provider, model.account_id) in signed_in_accounts
    ]
    eligible.sort(key=lambda model: model.priority)
    return [
        CatalogEntry(
            group=model.provider,
            name=model.name,
            backend="provider",
            provider=model.provider,
            account_id=model.account_id,
            is_default=model.is_default,
        )
        for model in eligible
    ]


def default_entry(entries: list[CatalogEntry]) -> CatalogEntry | None:
    return next((entry for entry in entries if entry.is_default), None)


def set_default(catalog_root: Path, group: str, name: str) -> CatalogEntry:
    """Persists `name` (within `group`) as the catalog's active default.

    Only the Local group persists today -- there's nowhere yet to durably
    store a provider entry's default flag, since provider entries aren't
    installed/tracked by Autumn itself, just advertised transiently by
    whatever `PromptRoutingPolicy` the caller supplies. Raises for any other
    group so callers (e.g. a future disabled-provider-row UI) can catch this
    and surface "not available yet" rather than silently no-op'ing.
    """
    if group != LOCAL_GROUP:
        raise ValueError(f"Setting the default isn't supported yet for catalog group {group!r}")
    model = local_models.set_default(catalog_root, name)
    return _local_entry(model)
