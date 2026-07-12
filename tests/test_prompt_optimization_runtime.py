import json
from pathlib import Path

import pytest

import local_models
import prompt_optimization_runtime as runtime
from models import ChatMessage
from prompt_optimization_contracts import (
    BestResultArtifacts,
    BuiltInMetricRef,
    CustomLocalMetricRef,
    EvalAssetRef,
    ModelIdentity,
    PromptOptimizationSpec,
    PromptOptimizationSpecBudget,
)


class _NoOpDashboard:
    """Accepts any GEPACallback-shaped method call as a no-op.

    A real `DashboardCallback` needs a live Textual `App` (`call_from_thread`)
    -- far heavier than these runtime-level tests need. What's under test here
    is prompt_optimization_runtime driving a real `gepa.optimize()` call
    correctly, not DashboardCallback's own event handling (covered by
    dashboard_callback's own tests and exercised end-to-end by test_runner.py
    with a stubbed prompt_optimization_runtime)."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


def _install_tiny_eval_asset(
    assets_root: Path,
    *,
    asset_id: str = "tiny-arithmetic",
    version: str = "2026.07.12",
    examples: list[dict] | None = None,
    supported_metrics: tuple[str, ...] = ("exact_match", "contains"),
    default_metric: str = "exact_match",
) -> None:
    asset_dir = assets_root / asset_id / version
    asset_dir.mkdir(parents=True)
    examples = examples if examples is not None else [{"input": "2+2", "answer": "4"}]
    (asset_dir / "examples.jsonl").write_text("\n".join(json.dumps(example) for example in examples) + "\n")
    metadata = {
        "schema_version": 1,
        "asset_id": asset_id,
        "version": version,
        "label": "Tiny arithmetic",
        "description": "Test fixture eval asset.",
        "supported_metrics": list(supported_metrics),
        "default_metric": default_metric,
        "size_bytes": 12,
        "license": "CC-BY-4.0",
        "provenance": "Synthetic examples generated for Autumn tests.",
        "sha256": "0" * 64,
        "path": str(asset_dir),
    }
    (asset_dir / "autumn_eval_asset.json").write_text(json.dumps(metadata))


def _install_local_model(catalog_root: Path, tmp_path: Path, name: str) -> None:
    model_file = tmp_path / f"{name}.gguf"
    model_file.write_bytes(b"fake model weights")
    local_models.install_model(catalog_root, name=name, source_path=model_file)


def _spec(
    tmp_path: Path,
    *,
    eval_asset: EvalAssetRef,
    metric=BuiltInMetricRef(metric_id="exact_match"),
    task_model_name: str = "tiny-task",
    optimizer_model_name: str = "tiny-optimizer",
    budget: PromptOptimizationSpecBudget = PromptOptimizationSpecBudget(max_metric_calls=1),
) -> PromptOptimizationSpec:
    return PromptOptimizationSpec(
        prompt="Answer the arithmetic question.",
        task_model=ModelIdentity(name=task_model_name, backend="llama.cpp"),
        optimizer_model=ModelIdentity(name=optimizer_model_name, backend="llama.cpp"),
        eval_asset=eval_asset,
        metric=metric,
        run_name="tiny-run",
        budget=budget,
    )


def test_run_completes_a_small_successful_optimization_and_writes_artifacts(tmp_path, monkeypatch):
    catalog_root = tmp_path / "models"
    assets_root = tmp_path / "eval-assets"
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _install_local_model(catalog_root, tmp_path, "tiny-task")
    _install_local_model(catalog_root, tmp_path, "tiny-optimizer")
    _install_tiny_eval_asset(assets_root)

    monkeypatch.setattr(runtime.paths, "models_root", lambda: catalog_root)
    monkeypatch.setattr(runtime.paths, "eval_assets_root", lambda: assets_root)
    monkeypatch.setattr(
        runtime.LocalModelRunner,
        "generate",
        lambda self, messages, model: ChatMessage(role="assistant", text="4", model=model.name),
    )

    spec = _spec(tmp_path, eval_asset=EvalAssetRef(asset_id="tiny-arithmetic", version="2026.07.12"))

    runtime.run(dashboard=_NoOpDashboard(), optimization_spec=spec, run_dir=run_dir)

    candidate_payload = json.loads((run_dir / runtime.BEST_CANDIDATE_FILENAME).read_text())
    assert candidate_payload["score"] == 1.0
    assert candidate_payload["candidate"]["prompt"] == "Answer the arithmetic question."

    prompt_markdown = (run_dir / runtime.BEST_PROMPT_FILENAME).read_text()
    assert "Answer the arithmetic question." in prompt_markdown

    artifacts = BestResultArtifacts.from_dict(json.loads((run_dir / runtime.BEST_RESULT_FILENAME).read_text()))
    assert artifacts.best_prompt.score == 1.0
    assert artifacts.best_prompt.prompt == "Answer the arithmetic question."
    assert artifacts.candidate_json == runtime.BEST_CANDIDATE_FILENAME
    assert artifacts.prompt_markdown == runtime.BEST_PROMPT_FILENAME


def test_run_raises_clearly_when_eval_asset_is_not_installed(tmp_path, monkeypatch):
    catalog_root = tmp_path / "models"
    assets_root = tmp_path / "eval-assets"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(runtime.paths, "models_root", lambda: catalog_root)
    monkeypatch.setattr(runtime.paths, "eval_assets_root", lambda: assets_root)

    spec = _spec(tmp_path, eval_asset=EvalAssetRef(asset_id="does-not-exist", version="1"))

    with pytest.raises(runtime.PromptOptimizationRuntimeError, match="eval asset unavailable"):
        runtime.run(dashboard=_NoOpDashboard(), optimization_spec=spec, run_dir=run_dir)


def test_run_raises_clearly_when_metric_is_unsupported_by_the_eval_asset(tmp_path, monkeypatch):
    catalog_root = tmp_path / "models"
    assets_root = tmp_path / "eval-assets"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _install_tiny_eval_asset(assets_root, supported_metrics=("contains",), default_metric="contains")
    monkeypatch.setattr(runtime.paths, "models_root", lambda: catalog_root)
    monkeypatch.setattr(runtime.paths, "eval_assets_root", lambda: assets_root)

    spec = _spec(
        tmp_path,
        eval_asset=EvalAssetRef(asset_id="tiny-arithmetic", version="2026.07.12"),
        metric=BuiltInMetricRef(metric_id="exact_match"),
    )

    with pytest.raises(runtime.PromptOptimizationRuntimeError, match="metric unavailable"):
        runtime.run(dashboard=_NoOpDashboard(), optimization_spec=spec, run_dir=run_dir)


def test_run_raises_clearly_when_task_model_is_not_installed(tmp_path, monkeypatch):
    catalog_root = tmp_path / "models"
    assets_root = tmp_path / "eval-assets"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _install_tiny_eval_asset(assets_root)
    monkeypatch.setattr(runtime.paths, "models_root", lambda: catalog_root)
    monkeypatch.setattr(runtime.paths, "eval_assets_root", lambda: assets_root)

    spec = _spec(tmp_path, eval_asset=EvalAssetRef(asset_id="tiny-arithmetic", version="2026.07.12"))

    with pytest.raises(runtime.PromptOptimizationRuntimeError, match="tiny-task"):
        runtime.run(dashboard=_NoOpDashboard(), optimization_spec=spec, run_dir=run_dir)


def test_run_raises_clearly_when_task_model_call_fails(tmp_path, monkeypatch):
    catalog_root = tmp_path / "models"
    assets_root = tmp_path / "eval-assets"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _install_local_model(catalog_root, tmp_path, "tiny-task")
    _install_local_model(catalog_root, tmp_path, "tiny-optimizer")
    _install_tiny_eval_asset(assets_root)
    monkeypatch.setattr(runtime.paths, "models_root", lambda: catalog_root)
    monkeypatch.setattr(runtime.paths, "eval_assets_root", lambda: assets_root)

    def _boom(self, messages, model):
        raise runtime.LocalModelRuntimeError(f"{model.name} failed: llama-cli not found")

    monkeypatch.setattr(runtime.LocalModelRunner, "generate", _boom)

    spec = _spec(tmp_path, eval_asset=EvalAssetRef(asset_id="tiny-arithmetic", version="2026.07.12"))

    with pytest.raises(runtime.PromptOptimizationRuntimeError, match="model call failed"):
        runtime.run(dashboard=_NoOpDashboard(), optimization_spec=spec, run_dir=run_dir)


def test_run_raises_clearly_when_custom_metric_raises(tmp_path, monkeypatch):
    catalog_root = tmp_path / "models"
    assets_root = tmp_path / "eval-assets"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _install_local_model(catalog_root, tmp_path, "tiny-task")
    _install_local_model(catalog_root, tmp_path, "tiny-optimizer")
    _install_tiny_eval_asset(assets_root, supported_metrics=("exact_match",))
    monkeypatch.setattr(runtime.paths, "models_root", lambda: catalog_root)
    monkeypatch.setattr(runtime.paths, "eval_assets_root", lambda: assets_root)
    monkeypatch.setattr(
        runtime.LocalModelRunner,
        "generate",
        lambda self, messages, model: ChatMessage(role="assistant", text="4", model=model.name),
    )

    metric_path = tmp_path / "custom_metric.py"
    metric_path.write_text(
        "def broken_metric(example, final_response, context):\n"
        "    raise ValueError('custom metric is broken')\n"
    )

    spec = _spec(
        tmp_path,
        eval_asset=EvalAssetRef(asset_id="tiny-arithmetic", version="2026.07.12"),
        metric=CustomLocalMetricRef(path=metric_path, function="broken_metric"),
    )

    with pytest.raises(runtime.PromptOptimizationRuntimeError, match="metric failed during optimization"):
        runtime.run(dashboard=_NoOpDashboard(), optimization_spec=spec, run_dir=run_dir)
