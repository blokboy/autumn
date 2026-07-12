"""Draft parsing for first-class GEPA prompt optimization commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import catalog
from models import CatalogEntry, PromptRoutingPolicy
from prompt_optimization_contracts import (
    BuiltInMetricRef,
    EvalAssetRef,
    MetricRef,
    ModelIdentity,
    PromptOptimizationSpecBudget,
)
from prompt_optimization_metrics import list_builtin_metrics


@dataclass(frozen=True)
class PromptOptimizationDraft:
    """Recoverably validated prompt optimization handoff.

    Missing or unavailable fields live in `validation_errors` so the UI can
    keep the draft visible and let later confirmation work repair it.
    """

    raw_text: str
    prompt: str | None = None
    system_prompt: str | None = None
    task_model: ModelIdentity | None = None
    optimizer_model: ModelIdentity | None = None
    eval_asset: EvalAssetRef | None = None
    metric: MetricRef | None = None
    run_name: str | None = None
    budget: PromptOptimizationSpecBudget = field(default_factory=PromptOptimizationSpecBudget)
    validation_errors: tuple[str, ...] = ()
    tokens: tuple[str, ...] = field(default=(), compare=False)


@dataclass
class _DraftFields:
    prompt: str | None = None
    system_prompt: str | None = None
    model: str | None = None
    task_model: str | None = None
    optimizer_model: str | None = None
    eval_asset: str | None = None
    metric: str | None = None
    run_name: str | None = None
    max_metric_calls: str | None = None


_FLAG_FIELDS = {
    "--prompt": "prompt",
    "--system-prompt": "system_prompt",
    "--system": "system_prompt",
    "--model": "model",
    "--task-model": "task_model",
    "--optimizer-model": "optimizer_model",
    "--eval-asset": "eval_asset",
    "--eval": "eval_asset",
    "--metric": "metric",
    "--name": "run_name",
    "--max-metric-calls": "max_metric_calls",
    "--budget": "max_metric_calls",
}

_FIELD_HINTS = (
    (("optimizer", "model"), "optimizer_model"),
    (("task", "model"), "task_model"),
    (("system", "prompt"), "system_prompt"),
    (("eval", "asset"), "eval_asset"),
    (("max", "metric", "calls"), "max_metric_calls"),
    (("prompt",), "prompt"),
    (("model",), "model"),
    (("metric",), "metric"),
    (("name",), "run_name"),
    (("budget",), "max_metric_calls"),
)


def parse_prompt_optimization_draft(
    tokens: list[str],
    *,
    raw_text: str,
    catalog_root: Path,
    prompt_routing_policy: PromptRoutingPolicy | None = None,
) -> PromptOptimizationDraft:
    fields, errors = _parse_fields(tokens)
    entries = [
        entry
        for entry in catalog.build_entries(catalog_root, policy=prompt_routing_policy)
        if entry.group != catalog.LOCAL_GROUP or entry.status == "installed"
    ]

    if fields.model is not None and fields.task_model is None and fields.optimizer_model is None:
        shared_model = _resolve_model(fields.model, entries, errors=errors)
        task_model = shared_model
        optimizer_model = shared_model
    else:
        task_model = _resolve_model(fields.task_model, entries, errors=errors)
        optimizer_model = _resolve_model(fields.optimizer_model, entries, errors=errors)

    eval_asset = _parse_eval_asset(fields.eval_asset, errors)
    metric = _parse_metric(fields.metric, errors)
    budget = _parse_budget(fields.max_metric_calls, errors)

    if fields.prompt is None:
        errors.append("prompt is required")

    return PromptOptimizationDraft(
        raw_text=raw_text,
        tokens=tuple(tokens),
        prompt=fields.prompt,
        system_prompt=fields.system_prompt,
        task_model=task_model,
        optimizer_model=optimizer_model,
        eval_asset=eval_asset,
        metric=metric,
        run_name=fields.run_name,
        budget=budget,
        validation_errors=tuple(errors),
    )


def _parse_fields(tokens: list[str]) -> tuple[_DraftFields, list[str]]:
    fields = _DraftFields()
    errors: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        field_name = _FLAG_FIELDS.get(token)
        if field_name is None:
            hint = _field_hint_at(tokens, index)
            if hint is not None:
                hint_length, hinted_field_name = hint
                value_index = index + hint_length
                if value_index >= len(tokens) or _is_field_start(tokens, value_index):
                    errors.append(f"{' '.join(tokens[index:value_index])} requires a value")
                    index = value_index
                    continue
                setattr(fields, hinted_field_name, tokens[value_index])
                index = value_index + 1
                continue
            if token.startswith("--"):
                errors.append(f"unknown prompt optimization option: {token}")
            index += 1
            continue
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
            errors.append(f"{token} requires a value")
            index += 1
            continue
        setattr(fields, field_name, tokens[index + 1])
        index += 2
    return fields, errors


def _field_hint_at(tokens: list[str], index: int) -> tuple[int, str] | None:
    remaining = [token.lower() for token in tokens[index:]]
    for label_tokens, field_name in _FIELD_HINTS:
        if tuple(remaining[: len(label_tokens)]) == label_tokens:
            return len(label_tokens), field_name
    return None


def _is_field_start(tokens: list[str], index: int) -> bool:
    if tokens[index].startswith("--"):
        return True
    return _field_hint_at(tokens, index) is not None


def _resolve_model(
    model_ref: str | None,
    entries: list[CatalogEntry],
    *,
    errors: list[str],
) -> ModelIdentity | None:
    if model_ref is None:
        return None
    match = _find_model(model_ref, entries)
    if match is None:
        errors.append(f"model is not available in the unified catalog: {model_ref}")
        return None
    return ModelIdentity(
        name=match.name,
        backend=match.backend,
        provider=match.provider,
        account_id=match.account_id,
    )


def _find_model(model_ref: str, entries: list[CatalogEntry]) -> CatalogEntry | None:
    if "/" in model_ref:
        group, name = model_ref.split("/", 1)
        return next(
            (
                entry
                for entry in entries
                if entry.group.lower() == group.lower() and entry.name == name
            ),
            None,
        )

    matches = [entry for entry in entries if entry.name == model_ref]
    if len(matches) == 1:
        return matches[0]
    return None


def _parse_eval_asset(value: str | None, errors: list[str]) -> EvalAssetRef | None:
    if value is None:
        return None
    separator = "@" if "@" in value else ":"
    if separator not in value:
        errors.append("eval asset must be written as asset_id@version")
        return None
    asset_id, version = value.split(separator, 1)
    if not asset_id or not version:
        errors.append("eval asset must be written as asset_id@version")
        return None
    return EvalAssetRef(asset_id=asset_id, version=version)


def _parse_metric(value: str | None, errors: list[str]) -> MetricRef | None:
    if value is None:
        return None
    if value not in list_builtin_metrics():
        errors.append(f"unknown built-in metric: {value}")
        return None
    return BuiltInMetricRef(metric_id=value)


def _parse_budget(value: str | None, errors: list[str]) -> PromptOptimizationSpecBudget:
    if value is None:
        return PromptOptimizationSpecBudget()
    try:
        max_metric_calls = int(value)
    except ValueError:
        errors.append("max metric calls must be a positive integer")
        return PromptOptimizationSpecBudget()
    if max_metric_calls <= 0:
        errors.append("max metric calls must be a positive integer")
        return PromptOptimizationSpecBudget()
    return PromptOptimizationSpecBudget(max_metric_calls=max_metric_calls)
