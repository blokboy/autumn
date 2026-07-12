"""Durable contracts for first-class GEPA prompt optimization runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PROMPT_OPTIMIZATION_SPEC_VERSION = 1
PROMPT_OPTIMIZATION_RUN_KIND = "prompt_optimization"
BEST_RESULT_ARTIFACTS_VERSION = 1

# Filenames `prompt_optimization_runtime.py` writes into a run directory on
# completion, and `registry.py` reads back (for both a live run that just
# finished and any historical run reopened later) -- defined here rather than
# in `prompt_optimization_runtime.py` so `registry.py` can read them back
# without importing that module's `gepa` dependency (see `registry.py`'s own
# module docstring on staying import-light for the non-Textual `autumn runs`
# CLI path).
BEST_CANDIDATE_FILENAME = "autumn_best_candidate.json"
BEST_PROMPT_FILENAME = "autumn_best_prompt.md"
BEST_RESULT_FILENAME = "autumn_best_result.json"


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} is required")
    return value


@dataclass(frozen=True)
class ModelIdentity:
    """Catalog-backed identity for a model selected into an optimization spec."""

    name: str
    backend: str
    provider: str | None = None
    account_id: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {"name": self.name, "backend": self.backend}
        if self.provider is not None:
            payload["provider"] = self.provider
        if self.account_id is not None:
            payload["account_id"] = self.account_id
        return payload


def model_identity_from_dict(payload: object) -> ModelIdentity:
    if not isinstance(payload, dict):
        raise ValueError("model identity must be an object")
    provider = payload.get("provider")
    account_id = payload.get("account_id")
    if provider is not None and not isinstance(provider, str):
        raise ValueError("provider must be a string")
    if account_id is not None and not isinstance(account_id, str):
        raise ValueError("account_id must be a string")
    return ModelIdentity(
        name=_required_str(payload, "name"),
        backend=_required_str(payload, "backend"),
        provider=provider,
        account_id=account_id,
    )


@dataclass(frozen=True)
class EvalAssetRef:
    """Exact immutable eval asset version used by a prompt optimization run."""

    asset_id: str
    version: str

    def to_dict(self) -> dict[str, str]:
        return {"asset_id": self.asset_id, "version": self.version}


def eval_asset_ref_from_dict(payload: object) -> EvalAssetRef:
    if not isinstance(payload, dict):
        raise ValueError("eval_asset must be an object")
    return EvalAssetRef(
        asset_id=_required_str(payload, "asset_id"),
        version=_required_str(payload, "version"),
    )


@dataclass(frozen=True)
class BuiltInMetricRef:
    """Metric implemented and shipped by Autumn."""

    metric_id: str
    kind: Literal["built_in"] = "built_in"

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "metric_id": self.metric_id}


@dataclass(frozen=True)
class CustomLocalMetricRef:
    """Trusted local metric code selected explicitly by path and function."""

    path: Path
    function: str
    kind: Literal["custom_local"] = "custom_local"

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "path": str(self.path), "function": self.function}


MetricRef = BuiltInMetricRef | CustomLocalMetricRef


def metric_ref_from_dict(payload: object) -> MetricRef:
    if not isinstance(payload, dict):
        raise ValueError("metric must be an object")
    kind = payload.get("kind")
    if kind == "built_in":
        return BuiltInMetricRef(metric_id=_required_str(payload, "metric_id"))
    if kind == "custom_local":
        return CustomLocalMetricRef(
            path=Path(_required_str(payload, "path")),
            function=_required_str(payload, "function"),
        )
    raise ValueError("metric kind must be built_in or custom_local")


@dataclass(frozen=True)
class PromptOptimizationSpecBudget:
    """Runtime budget defaults for a prompt optimization run."""

    max_metric_calls: int | None = None

    def to_dict(self) -> dict[str, int]:
        payload = {}
        if self.max_metric_calls is not None:
            payload["max_metric_calls"] = self.max_metric_calls
        return payload


def prompt_optimization_budget_from_dict(payload: object) -> PromptOptimizationSpecBudget:
    if not isinstance(payload, dict):
        raise ValueError("budget must be an object")
    max_metric_calls = payload.get("max_metric_calls")
    if max_metric_calls is not None:
        if not isinstance(max_metric_calls, int) or isinstance(max_metric_calls, bool) or max_metric_calls <= 0:
            raise ValueError("max_metric_calls must be a positive integer")
    return PromptOptimizationSpecBudget(max_metric_calls=max_metric_calls)


@dataclass(frozen=True)
class PromptOptimizationSpec:
    """Versioned durable input for a first-class prompt optimization run."""

    prompt: str
    task_model: ModelIdentity
    optimizer_model: ModelIdentity
    eval_asset: EvalAssetRef
    metric: MetricRef
    run_name: str
    budget: PromptOptimizationSpecBudget
    system_prompt: str | None = None
    schema_version: int = PROMPT_OPTIMIZATION_SPEC_VERSION
    run_kind: Literal["prompt_optimization"] = PROMPT_OPTIMIZATION_RUN_KIND

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "run_kind": self.run_kind,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "task_model": self.task_model.to_dict(),
            "optimizer_model": self.optimizer_model.to_dict(),
            "eval_asset": self.eval_asset.to_dict(),
            "metric": self.metric.to_dict(),
            "run_name": self.run_name,
            "budget": self.budget.to_dict(),
        }
        return payload


def prompt_optimization_spec_from_dict(payload: object) -> PromptOptimizationSpec:
    if not isinstance(payload, dict):
        raise ValueError("prompt optimization spec must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != PROMPT_OPTIMIZATION_SPEC_VERSION:
        raise ValueError("unsupported prompt optimization spec schema_version")
    if payload.get("run_kind") != PROMPT_OPTIMIZATION_RUN_KIND:
        raise ValueError("run_kind must be prompt_optimization")
    system_prompt = payload.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string or null")
    return PromptOptimizationSpec(
        prompt=_required_str(payload, "prompt"),
        system_prompt=system_prompt,
        task_model=model_identity_from_dict(payload.get("task_model")),
        optimizer_model=model_identity_from_dict(payload.get("optimizer_model")),
        eval_asset=eval_asset_ref_from_dict(payload.get("eval_asset")),
        metric=metric_ref_from_dict(payload.get("metric")),
        run_name=_required_str(payload, "run_name"),
        budget=prompt_optimization_budget_from_dict(payload.get("budget")),
    )


@dataclass(frozen=True)
class BestPromptArtifact:
    """Primary user-facing result of a prompt optimization run."""

    prompt: str
    system_prompt: str | None
    score: float | None
    candidate_idx: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "score": self.score,
            "candidate_idx": self.candidate_idx,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "BestPromptArtifact":
        if not isinstance(payload, dict):
            raise ValueError("best_prompt must be an object")
        score = payload.get("score")
        candidate_idx = payload.get("candidate_idx")
        system_prompt = payload.get("system_prompt")
        if score is not None and (not isinstance(score, (int, float)) or isinstance(score, bool)):
            raise ValueError("score must be numeric or null")
        if candidate_idx is not None and (not isinstance(candidate_idx, int) or isinstance(candidate_idx, bool)):
            raise ValueError("candidate_idx must be an integer or null")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string or null")
        return cls(
            prompt=_required_str(payload, "prompt"),
            system_prompt=system_prompt,
            score=float(score) if score is not None else None,
            candidate_idx=candidate_idx,
        )


@dataclass(frozen=True)
class BestResultArtifacts:
    """File contract for best-result artifacts written into a run directory."""

    candidate_json: str
    prompt_markdown: str
    best_prompt: BestPromptArtifact
    schema_version: int = BEST_RESULT_ARTIFACTS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_json": self.candidate_json,
            "prompt_markdown": self.prompt_markdown,
            "best_prompt": self.best_prompt.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "BestResultArtifacts":
        if not isinstance(payload, dict):
            raise ValueError("best result artifacts must be an object")
        if payload.get("schema_version") != BEST_RESULT_ARTIFACTS_VERSION:
            raise ValueError("unsupported best result artifacts schema_version")
        return cls(
            candidate_json=_required_str(payload, "candidate_json"),
            prompt_markdown=_required_str(payload, "prompt_markdown"),
            best_prompt=BestPromptArtifact.from_dict(payload.get("best_prompt")),
        )
