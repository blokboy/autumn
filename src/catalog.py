"""Unified view over Autumn's model catalog: installed local models plus any
eligible provider-backed candidates from a `PromptRoutingPolicy`, as one flat
list of `CatalogEntry` addressed by (group, name). `model_router.choose_model`
and the Models tab (`ModelCatalogView`) both build their view of "what's in
the catalog" from `build_entries` here, so they can never drift apart.

Local models persist their own default flag in `local_models.py`'s manifest.
Provider entries have nowhere of their own to persist one (they aren't
installed/tracked by Autumn, just advertised transiently by whatever
`PromptRoutingPolicy` the caller supplies each call), so `set_default`
additionally persists a small `default_override.json` alongside the
manifest when a *currently-eligible* provider entry is picked. `build_entries`
applies that override on top of the local manifest's own flags, so at most
one entry is ever `is_default=True` at a time regardless of which kind it
is. An override pointing at something no longer eligible (key revoked, entry
disabled) is silently ignored -- falls back to the local manifest, same
"falls through" spirit as the rest of the fallback chain.
"""

import json
from dataclasses import replace
from pathlib import Path

import local_models
from models import CatalogEntry, LocalModel, PromptRoutingPolicy

LOCAL_GROUP = "Local"
_DEFAULT_OVERRIDE = "default_override.json"


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
    return _apply_default_override(catalog_root, entries)


def _local_entry(model: LocalModel) -> CatalogEntry:
    return CatalogEntry(
        group=LOCAL_GROUP,
        name=model.name,
        backend=model.backend,
        path=model.path,
        context_window=model.context_window,
        is_default=model.is_default,
        status=model.status,
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


def set_default(
    catalog_root: Path,
    group: str,
    name: str,
    *,
    policy: PromptRoutingPolicy | None = None,
) -> CatalogEntry:
    """Persists `name` (within `group`) as the catalog's active default.

    The Local group persists via `local_models.py`'s manifest, same as
    always. Any other group persists via a `default_override.json` file
    alongside it, but only if `(group, name)` matches a currently-eligible
    provider entry from `policy` -- raises `ValueError` otherwise (same
    contract as before), so callers (e.g. a disabled-provider-row UI) can
    catch this and surface "not available yet" rather than silently
    persisting an override for something that was never selectable.
    """
    if group == LOCAL_GROUP:
        model = local_models.set_default(catalog_root, name)
        _clear_override(catalog_root)
        return _local_entry(model)

    eligible = _eligible_provider_entries(policy)
    match = next((entry for entry in eligible if entry.group == group and entry.name == name), None)
    if match is None:
        raise ValueError(f"Setting the default isn't supported yet for catalog group {group!r}")
    _save_override(catalog_root, group, name)
    return replace(match, is_default=True)


def _override_path(catalog_root: Path) -> Path:
    return catalog_root / _DEFAULT_OVERRIDE


def _load_override(catalog_root: Path) -> tuple[str, str] | None:
    try:
        payload = json.loads(_override_path(catalog_root).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    group = payload.get("group") if isinstance(payload, dict) else None
    name = payload.get("name") if isinstance(payload, dict) else None
    if isinstance(group, str) and isinstance(name, str):
        return group, name
    return None


def _save_override(catalog_root: Path, group: str, name: str) -> None:
    catalog_root.mkdir(parents=True, exist_ok=True)
    _override_path(catalog_root).write_text(json.dumps({"group": group, "name": name}))


def _clear_override(catalog_root: Path) -> None:
    _override_path(catalog_root).unlink(missing_ok=True)


def _apply_default_override(catalog_root: Path, entries: list[CatalogEntry]) -> list[CatalogEntry]:
    override = _load_override(catalog_root)
    if override is None:
        return entries
    override_group, override_name = override
    if not any(entry.group == override_group and entry.name == override_name for entry in entries):
        return entries
    return [
        replace(entry, is_default=(entry.group == override_group and entry.name == override_name))
        for entry in entries
    ]
