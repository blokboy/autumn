"""Executes a `PromptOptimizationSpec` as a real GEPA optimization.

Unlike `runner.py`'s script path (which monkeypatches GEPA's entry points via
`patch.py` so an *unmodified user script*'s own `import gepa` picks up the
dashboard callback), a prompt optimization run has no user script at all --
the spec is the only input. This module therefore calls `gepa.optimize`
directly, passing `dashboard` through `callbacks=[dashboard]` exactly the way
`patch.py` injects it for script runs, so `DashboardCallback` never needs to
know or care which path produced the events it's receiving.

`run()` matches `runner.PromptOptimizationRuntime`'s Protocol shape and is
meant to be passed as `runner.launch(..., prompt_optimization_runtime=run)` --
see `app.py`'s `_launch_runner_prompt_optimization`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

import eval_assets
import local_models
import paths
from anthropic_runner import AnthropicRunner, AnthropicRuntimeError
from dashboard_callback import DashboardCallback
from groq_runner import GroqRunner, GroqRuntimeError
from local_model_runner import LocalModelRunner, LocalModelRuntimeError
from models import ChatMessage
from openai_runner import OpenAIRunner, OpenAIRuntimeError
from prompt_optimization_contracts import (
    BEST_CANDIDATE_FILENAME,
    BEST_PROMPT_FILENAME,
    BEST_RESULT_FILENAME,
    BestPromptArtifact,
    BestResultArtifacts,
    ModelIdentity,
    PromptOptimizationSpec,
)
from prompt_optimization_metrics import (
    EvalAssetMetricSupport,
    MetricExecutionError,
    MetricResolutionError,
    ResolvedMetric,
    resolve_metric,
)

# GEPA requires either `max_metric_calls` or `stop_callbacks`. A prompt
# optimization spec's budget is optional (see `PromptOptimizationSpecBudget`),
# so this is the floor applied when the user didn't set one -- small enough to
# finish in reasonable time against a hosted model, generous enough to let a
# handful of reflective mutations happen.
_DEFAULT_MAX_METRIC_CALLS = 30

_ANSWER_KEYS = ("answer", "expected", "label", "target")
_INPUT_KEYS = ("input", "prompt")
_RESERVED_EXAMPLE_KEYS = frozenset(_ANSWER_KEYS + _INPUT_KEYS)

ChatCompletionFn = Callable[[Sequence[dict[str, str]]], str]


class PromptOptimizationRuntimeError(RuntimeError):
    """Raised when a prompt optimization spec cannot be executed to completion.

    Wraps every distinct failure mode called out by issue #48 -- a missing/
    uninstalled eval asset, an unsupported metric, a model resolution/call
    error, or a custom metric exception -- behind one exception type so
    `runner.py`'s existing `except BaseException` -> `dashboard.mark_script_finished(exc)`
    path (shared with script runs) surfaces a clear, single-line message
    without needing prompt-optimization-specific handling of its own.
    """


def run(
    *,
    dashboard: DashboardCallback,
    optimization_spec: PromptOptimizationSpec,
    run_dir: Path,
    catalog_root: Path | None = None,
    assets_root: Path | None = None,
) -> None:
    # Callers that already track an AutumnApp-scoped catalog/assets root (e.g.
    # a test fixture, or a future --data-root override) pass it through so this
    # runtime never silently reads/writes the real global XDG paths instead of
    # the roots the rest of the app is actually using.
    catalog_root = catalog_root or paths.models_root()
    assets_root = assets_root or paths.eval_assets_root()

    installed_asset = _load_installed_asset(assets_root, optimization_spec)
    resolved_metric = _resolve_spec_metric(optimization_spec, installed_asset)
    trainset, valset = _load_eval_dataset(installed_asset)

    task_completion = _completion_fn(optimization_spec.task_model, catalog_root)
    optimizer_completion = _completion_fn(optimization_spec.optimizer_model, catalog_root)

    adapter = _PromptOptimizationAdapter(task_completion=task_completion, resolved_metric=resolved_metric)
    seed_candidate = _seed_candidate(optimization_spec)
    max_metric_calls = optimization_spec.budget.max_metric_calls or _DEFAULT_MAX_METRIC_CALLS

    try:
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=_reflection_fn(optimizer_completion),
            max_metric_calls=max_metric_calls,
            run_dir=str(run_dir),
            callbacks=[dashboard],
            raise_on_exception=True,
        )
    except PromptOptimizationRuntimeError:
        raise
    except MetricExecutionError as exc:
        raise PromptOptimizationRuntimeError(f"metric failed during optimization: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - GEPA's own engine exceptions are unpredictable; wrap them all
        raise PromptOptimizationRuntimeError(f"prompt optimization failed: {exc}") from exc

    _write_best_result_artifacts(run_dir, result)


def _load_installed_asset(
    assets_root: Path, optimization_spec: PromptOptimizationSpec
) -> eval_assets.InstalledEvalAsset:
    ref = optimization_spec.eval_asset
    try:
        return eval_assets.select_installed_asset(assets_root, f"{ref.asset_id}@{ref.version}")
    except eval_assets.EvalAssetError as exc:
        raise PromptOptimizationRuntimeError(f"eval asset unavailable: {exc}") from exc


def _resolve_spec_metric(
    optimization_spec: PromptOptimizationSpec, installed_asset: eval_assets.InstalledEvalAsset
) -> ResolvedMetric:
    metric_support = EvalAssetMetricSupport(
        asset_id=installed_asset.asset_id,
        default_metric_id=installed_asset.default_metric,
        supported_metric_ids=installed_asset.supported_metrics,
    )
    try:
        return resolve_metric(optimization_spec.metric, metric_support)
    except MetricResolutionError as exc:
        raise PromptOptimizationRuntimeError(f"metric unavailable: {exc}") from exc


def _seed_candidate(optimization_spec: PromptOptimizationSpec) -> dict[str, str]:
    """`prompt` is always an optimizable GEPA component. `system_prompt` only
    becomes one when the spec actually set it -- an empty/absent system
    prompt means "no system message", not "optimize an empty string"."""
    seed_candidate = {"prompt": optimization_spec.prompt}
    if optimization_spec.system_prompt:
        seed_candidate["system_prompt"] = optimization_spec.system_prompt
    return seed_candidate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise PromptOptimizationRuntimeError(f"eval data record in {path} must be a JSON object")
        input_text = next((record[key] for key in _INPUT_KEYS if isinstance(record.get(key), str)), "")
        answer = next((record[key] for key in _ANSWER_KEYS if isinstance(record.get(key), str)), "")
        additional_context = {
            key: value
            for key, value in record.items()
            if key not in _RESERVED_EXAMPLE_KEYS and isinstance(value, str)
        }
        examples.append({"input": input_text, "answer": answer, "additional_context": additional_context})
    if not examples:
        raise PromptOptimizationRuntimeError(f"eval data file has no examples: {path}")
    return examples


def _load_eval_dataset(installed_asset: eval_assets.InstalledEvalAsset) -> tuple[list[dict], list[dict]]:
    """Reads the installed asset's bundled data files.

    v1 eval asset bundles (see `eval_assets.py`/#42) don't yet carry an
    explicit train/validation/input-field/answer-field schema in their
    installed manifest -- only a data-only file bundle. This adopts the
    convention already established by #42's own test fixtures: a
    `train.jsonl` + optional `validation.jsonl` pair when present, otherwise
    a single `examples.jsonl` reused for both train and validation. Revisit
    once eval asset manifests grow explicit file/field metadata.
    """
    train_path = installed_asset.path / "train.jsonl"
    if train_path.exists():
        trainset = _read_jsonl(train_path)
        validation_path = installed_asset.path / "validation.jsonl"
        valset = _read_jsonl(validation_path) if validation_path.exists() else trainset
        return trainset, valset

    examples_path = installed_asset.path / "examples.jsonl"
    if examples_path.exists():
        examples = _read_jsonl(examples_path)
        return examples, examples

    raise PromptOptimizationRuntimeError(
        f"eval asset {installed_asset.asset_id}@{installed_asset.version} has no "
        "train.jsonl/validation.jsonl or examples.jsonl data file"
    )


def _find_local_model(name: str, catalog_root: Path) -> local_models.LocalModel:
    for model in local_models.list_models(catalog_root):
        if model.name == name and model.status == "installed":
            return model
    raise PromptOptimizationRuntimeError(f"local model is not installed: {name}")


def _provider_runner(provider: str | None) -> AnthropicRunner | OpenAIRunner | GroqRunner:
    if provider == "anthropic":
        return AnthropicRunner()
    if provider == "openai":
        return OpenAIRunner()
    if provider == "groq":
        return GroqRunner()
    raise PromptOptimizationRuntimeError(f"unsupported provider for prompt optimization: {provider}")


def _completion_fn(identity: ModelIdentity, catalog_root: Path) -> ChatCompletionFn:
    """Builds a `messages -> response text` callable for `identity`, routed
    through the same catalog-backed runners (`LocalModelRunner`/
    `AnthropicRunner`/`OpenAIRunner`/`GroqRunner`) chat replies already use --
    never a bespoke call path -- so a prompt optimization run is
    indistinguishable, from the model's point of view, from any other Autumn
    request to that model."""
    if identity.backend == "llama.cpp":
        model = _find_local_model(identity.name, catalog_root)

        def call_local(messages: Sequence[dict[str, str]]) -> str:
            chat_messages = [ChatMessage(role=m["role"], text=m["content"]) for m in messages]
            try:
                return LocalModelRunner().generate(chat_messages, model).text
            except LocalModelRuntimeError as exc:
                raise PromptOptimizationRuntimeError(f"model call failed for {identity.name}: {exc}") from exc

        return call_local

    if identity.backend == "provider":
        runner = _provider_runner(identity.provider)

        def call_provider(messages: Sequence[dict[str, str]]) -> str:
            chat_messages = [ChatMessage(role=m["role"], text=m["content"]) for m in messages]
            try:
                return runner.generate(chat_messages, identity.name).text
            except (AnthropicRuntimeError, OpenAIRuntimeError, GroqRuntimeError) as exc:
                raise PromptOptimizationRuntimeError(f"model call failed for {identity.name}: {exc}") from exc

        return call_provider

    raise PromptOptimizationRuntimeError(f"unsupported model backend: {identity.backend}")


def _reflection_fn(completion: ChatCompletionFn) -> Callable[[str | list[dict[str, str]]], str]:
    """Adapts a `ChatCompletionFn` to GEPA's `LanguageModel` protocol
    (`__call__(prompt: str | list[dict]) -> str`), used for `reflection_lm` --
    GEPA calls its reflection model with a single rendered prompt string far
    more often than a pre-built messages list."""

    def call(prompt: str | list[dict[str, str]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        return completion(messages)

    return call


def _feedback_text(data: dict[str, Any], score: float, details: dict[str, Any]) -> str:
    parts = [f"score: {score}"]
    answer = data.get("answer")
    if answer:
        parts.append(f"expected: {answer}")
    if details:
        parts.append(f"details: {details}")
    return " | ".join(parts)


class _PromptOptimizationAdapter(GEPAAdapter[dict, dict, dict]):
    """Bridges GEPA's optimization loop to Autumn's catalog-backed task model
    and resolved metric. A custom adapter (rather than GEPA's own
    `DefaultAdapter`) is used because a prompt optimization spec optimizes up
    to *two* named components (`prompt` and optionally `system_prompt`) and
    must enforce Autumn's own built-in/custom-local metric contract
    (`ResolvedMetric.evaluate`) rather than GEPA's bundled evaluator.

    Model-call and metric exceptions are deliberately NOT caught per-example
    here (contrary to `GEPAAdapter`'s general per-example-failure guidance) --
    issue #48 requires model errors and custom metric exceptions to "surface
    clearly" as run failures, not be silently scored 0.0 and continue, so
    they're left to propagate as the systemic failures they actually are (a
    broken model connection or a buggy custom metric fails every subsequent
    call too).
    """

    def __init__(self, *, task_completion: ChatCompletionFn, resolved_metric: ResolvedMetric) -> None:
        self._task_completion = task_completion
        self._resolved_metric = resolved_metric

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict, dict]:
        task_prompt = candidate.get("prompt", "")
        system_prompt = candidate.get("system_prompt")

        outputs: list[dict] = []
        scores: list[float] = []
        trajectories: list[dict] | None = [] if capture_traces else None

        for data in batch:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            user_content = f"{task_prompt}\n\n{data['input']}" if task_prompt else data["input"]
            messages.append({"role": "user", "content": user_content})

            response = self._task_completion(messages)
            result = self._resolved_metric.evaluate(
                example=data,
                final_response=response,
                candidate={"prompt": task_prompt, "system_prompt": system_prompt},
            )

            outputs.append({"response": response})
            scores.append(result.score)
            if trajectories is not None:
                trajectories.append(
                    {
                        "input": data["input"],
                        "response": response,
                        "feedback": _feedback_text(data, result.score, result.details),
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict, dict],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        trajectories = eval_batch.trajectories or []
        records = [
            {
                "Inputs": traj["input"],
                "Generated Outputs": traj["response"],
                "Feedback": traj["feedback"],
            }
            for traj in trajectories
        ]
        return {component: records for component in components_to_update}


def _write_best_result_artifacts(run_dir: Path, result: Any) -> BestResultArtifacts:
    best_idx = result.best_idx
    best_candidate = result.candidates[best_idx]
    best_score = result.val_aggregate_scores[best_idx] if result.val_aggregate_scores else None

    (run_dir / BEST_CANDIDATE_FILENAME).write_text(
        json.dumps({"candidate_idx": best_idx, "score": best_score, "candidate": best_candidate}, indent=2)
    )

    prompt_text = best_candidate.get("prompt", "")
    system_prompt_text = best_candidate.get("system_prompt")
    markdown_lines = ["# Best optimized prompt", "", f"Score: {best_score}", ""]
    if system_prompt_text:
        markdown_lines += ["## System prompt", "", "```", system_prompt_text, "```", ""]
    markdown_lines += ["## Prompt", "", "```", prompt_text, "```", ""]
    (run_dir / BEST_PROMPT_FILENAME).write_text("\n".join(markdown_lines))

    artifacts = BestResultArtifacts(
        candidate_json=BEST_CANDIDATE_FILENAME,
        prompt_markdown=BEST_PROMPT_FILENAME,
        best_prompt=BestPromptArtifact(
            prompt=prompt_text,
            system_prompt=system_prompt_text,
            score=float(best_score) if best_score is not None else None,
            candidate_idx=best_idx,
        ),
    )
    (run_dir / BEST_RESULT_FILENAME).write_text(json.dumps(artifacts.to_dict(), indent=2))
    return artifacts
