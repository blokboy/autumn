"""Metric resolution for first-class prompt optimization runs."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from prompt_optimization_contracts import BuiltInMetricRef, CustomLocalMetricRef, MetricRef

MetricFunction = Callable[[Mapping[str, Any], str, Mapping[str, Any]], object]


class MetricResolutionError(ValueError):
    """Raised when a metric reference cannot be resolved before execution."""


class MetricExecutionError(RuntimeError):
    """Raised when a resolved metric fails or returns an invalid result."""


@dataclass(frozen=True)
class MetricResult:
    """Normalized metric output consumed by prompt optimization runtime code."""

    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalAssetMetricSupport:
    """Built-in metric declarations read from an installed eval asset manifest."""

    asset_id: str
    default_metric_id: str
    supported_metric_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id is required")
        if not self.default_metric_id:
            raise ValueError("default_metric_id is required")
        if not self.supported_metric_ids:
            raise ValueError("supported_metric_ids is required")
        if self.default_metric_id not in self.supported_metric_ids:
            raise ValueError(
                f"default metric '{self.default_metric_id}' must be listed in supported metrics"
            )


@dataclass(frozen=True)
class ResolvedMetric:
    """Callable metric with metadata about its source and trust boundary."""

    metric_id: str
    label: str
    trust_boundary: Literal["built_in", "custom_local"]
    _evaluate: MetricFunction
    source_path: Path | None = None

    def evaluate(
        self,
        *,
        example: Mapping[str, Any],
        final_response: str,
        candidate: Mapping[str, Any] | None = None,
    ) -> MetricResult:
        context = _sanitize_candidate_context(candidate or {})
        try:
            raw_result = self._evaluate(example, final_response, context)
        except Exception as exc:  # noqa: BLE001 - metric failures need wrapping for callers
            source = f" at {self.source_path}" if self.source_path is not None else ""
            raise MetricExecutionError(f"metric '{self.metric_id}'{source} failed: {exc}") from exc
        return _coerce_metric_result(raw_result, metric_id=self.metric_id)


def resolve_metric(
    metric_ref: MetricRef,
    eval_asset: EvalAssetMetricSupport | None = None,
) -> ResolvedMetric:
    """Resolve a durable metric reference into an executable metric."""

    if isinstance(metric_ref, BuiltInMetricRef):
        return _resolve_builtin_metric(metric_ref.metric_id, eval_asset)
    if isinstance(metric_ref, CustomLocalMetricRef):
        return _resolve_custom_local_metric(metric_ref)
    raise MetricResolutionError("unsupported metric reference")


def list_builtin_metrics() -> tuple[str, ...]:
    """Return built-in metric IDs shipped with Autumn."""

    return tuple(_BUILT_IN_METRICS)


def default_metric_for_eval(eval_asset: EvalAssetMetricSupport) -> BuiltInMetricRef:
    """Return the built-in default metric declared by an eval asset."""

    return BuiltInMetricRef(metric_id=eval_asset.default_metric_id)


def _resolve_builtin_metric(metric_id: str, eval_asset: EvalAssetMetricSupport | None) -> ResolvedMetric:
    metric = _BUILT_IN_METRICS.get(metric_id)
    if metric is None:
        raise MetricResolutionError(f"unknown built-in metric '{metric_id}'")
    if eval_asset is not None and metric_id not in eval_asset.supported_metric_ids:
        raise MetricResolutionError(
            f"metric '{metric_id}' is not supported by eval asset '{eval_asset.asset_id}'"
        )
    return ResolvedMetric(
        metric_id=metric_id,
        label=metric["label"],
        trust_boundary="built_in",
        _evaluate=metric["function"],
    )


def _resolve_custom_local_metric(metric_ref: CustomLocalMetricRef) -> ResolvedMetric:
    path = Path(metric_ref.path)
    if not path.exists():
        raise MetricResolutionError(f"custom local metric file does not exist: {path}")
    if not path.is_file():
        raise MetricResolutionError(f"custom local metric path is not a file: {path}")

    module = _load_module_from_path(path)
    function = getattr(module, metric_ref.function, None)
    if not callable(function):
        raise MetricResolutionError(
            f"custom local metric function '{metric_ref.function}' was not found in {path}"
        )

    return ResolvedMetric(
        metric_id=metric_ref.function,
        label=f"{metric_ref.function} ({path.name})",
        trust_boundary="custom_local",
        _evaluate=function,
        source_path=path,
    )


def _load_module_from_path(path: Path) -> ModuleType:
    module_name = f"_autumn_custom_metric_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise MetricResolutionError(f"custom local metric file cannot be loaded: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - load failures should be resolution errors
        raise MetricResolutionError(f"custom local metric file cannot be loaded: {path}: {exc}") from exc
    return module


def _sanitize_candidate_context(candidate: Mapping[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key in ("candidate_idx", "prompt", "system_prompt", "metadata"):
        if key in candidate:
            context[key] = candidate[key]
    return context


def _coerce_metric_result(raw_result: object, *, metric_id: str) -> MetricResult:
    if _is_number(raw_result):
        return MetricResult(score=float(raw_result))
    if isinstance(raw_result, Mapping):
        score = raw_result.get("score")
        details = raw_result.get("details", {})
        if _is_number(score) and isinstance(details, dict):
            return MetricResult(score=float(score), details=dict(details))
    raise MetricExecutionError(
        f"metric '{metric_id}' must return a number or an object with a numeric score"
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _exact_match(
    example: Mapping[str, Any],
    final_response: str,
    context: Mapping[str, Any],
) -> float:
    expected = _expected_answer(example)
    return 1.0 if final_response.strip() == expected.strip() else 0.0


def _contains(
    example: Mapping[str, Any],
    final_response: str,
    context: Mapping[str, Any],
) -> float:
    expected = _expected_answer(example)
    return 1.0 if expected.strip() in final_response else 0.0


def _expected_answer(example: Mapping[str, Any]) -> str:
    for key in ("answer", "expected", "label", "target"):
        value = example.get(key)
        if isinstance(value, str):
            return value
    raise MetricExecutionError("built-in metrics require an example answer")


_BUILT_IN_METRICS: dict[str, dict[str, Any]] = {
    "exact_match": {"label": "Exact match", "function": _exact_match},
    "contains": {"label": "Contains", "function": _contains},
}
