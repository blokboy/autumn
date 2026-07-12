import json
import time
from pathlib import Path

from dashboard_callback import DashboardCallback
from models import (
    DashboardState,
    LiveRunSpec,
    PromptOptimizationRunSpec,
    RunKind,
    RunStatus,
)
from prompt_optimization_contracts import (
    BuiltInMetricRef,
    EvalAssetRef,
    ModelIdentity,
    PromptOptimizationSpec,
    PromptOptimizationSpecBudget,
)
from runner import launch, recover_launch_spec


def _write_meta(run_dir: Path, meta: dict) -> None:
    (run_dir / "autumn_meta.json").write_text(json.dumps(meta))


def test_recover_launch_spec_reconstructs_spec_from_valid_meta(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_meta(
        run_dir,
        {
            "script_path": "/home/user/my_script.py",
            "run_name": "run1",
            "launched_at": "2026-07-10T12:00:00",
        },
    )

    spec = recover_launch_spec(run_dir)

    assert spec == LiveRunSpec(
        script_path=Path("/home/user/my_script.py"),
        run_dir=run_dir,
        run_name="run1",
    )


def test_recover_launch_spec_returns_none_when_meta_file_missing(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()

    assert recover_launch_spec(run_dir) is None


def test_recover_launch_spec_returns_none_on_invalid_json(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "autumn_meta.json").write_text("not valid json")

    assert recover_launch_spec(run_dir) is None


def test_recover_launch_spec_returns_none_when_script_path_missing(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_meta(run_dir, {"run_name": "run1", "launched_at": "2026-07-10T12:00:00"})

    assert recover_launch_spec(run_dir) is None


def test_recover_launch_spec_returns_none_when_script_path_empty(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_meta(run_dir, {"script_path": "", "run_name": "run1"})

    assert recover_launch_spec(run_dir) is None


def test_recover_launch_spec_returns_none_when_run_name_missing(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_meta(run_dir, {"script_path": "/home/user/my_script.py"})

    assert recover_launch_spec(run_dir) is None


def test_recover_launch_spec_returns_none_when_meta_is_not_a_dict(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "autumn_meta.json").write_text(json.dumps(["not", "a", "dict"]))

    assert recover_launch_spec(run_dir) is None


def test_recover_launch_spec_does_not_require_script_path_to_exist_on_disk(tmp_path):
    """Validation is purely about the JSON's shape -- whether the recovered
    script still exists on disk is a launch-time concern for runner.launch()/
    runpy, not this function's job."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_meta(run_dir, {"script_path": "/nonexistent/path/script.py", "run_name": "run1"})

    spec = recover_launch_spec(run_dir)

    assert spec is not None
    assert spec.script_path == Path("/nonexistent/path/script.py")


class _StubDashboard:
    """Minimal stand-in for DashboardCallback -- launch() only ever calls
    mark_script_finished on it (the script under test never calls
    gepa.optimize, so the patched functions launch() wires up are never
    invoked)."""

    def __init__(self) -> None:
        self.finished_with: list[BaseException | None] = []

    def mark_script_finished(self, exc: BaseException | None) -> None:
        self.finished_with.append(exc)


class _ImmediateApp:
    def call_from_thread(self, fn):
        fn()


def _prompt_optimization_spec(run_name: str = "prompt-run") -> PromptOptimizationSpec:
    model = ModelIdentity(name="local-test", backend="llama.cpp")
    return PromptOptimizationSpec(
        prompt="Summarize the ticket",
        task_model=model,
        optimizer_model=model,
        eval_asset=EvalAssetRef(asset_id="support-emails", version="v1"),
        metric=BuiltInMetricRef(metric_id="exact_match"),
        run_name=run_name,
        budget=PromptOptimizationSpecBudget(max_metric_calls=10),
    )


def test_launch_clears_a_stale_gepa_stop_file_before_starting(tmp_path):
    """A prior graceful stop leaves gepa.stop behind (GEPA's FileStopper never
    removes it). If launch() didn't clear it, resuming would have GEPA see the
    stale file on its very first check and halt again immediately instead of
    actually resuming -- see the comment in runner.launch()."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "gepa.stop").write_text("")
    script = tmp_path / "script.py"
    script.write_text("")  # never calls gepa.optimize -- just needs to exist and run cleanly

    dashboard = _StubDashboard()
    spec = LiveRunSpec(script_path=script, run_dir=run_dir, run_name="run1")

    thread = launch(dashboard, spec)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert not (run_dir / "gepa.stop").exists()
    assert dashboard.finished_with == [None]


def test_launch_writes_script_run_kind_to_meta(tmp_path):
    run_dir = tmp_path / "run1"
    script = tmp_path / "script.py"
    script.write_text("")

    dashboard = _StubDashboard()
    spec = LiveRunSpec(script_path=script, run_dir=run_dir, run_name="run1")

    thread = launch(dashboard, spec)
    thread.join(timeout=5)

    meta = json.loads((run_dir / "autumn_meta.json").read_text())
    assert meta["run_kind"] == "script"


def test_launch_runs_prompt_optimization_spec_target_without_script_file(tmp_path):
    run_dir = tmp_path / "prompt-run"
    target = PromptOptimizationRunSpec(
        optimization_spec=_prompt_optimization_spec(),
        run_dir=run_dir,
    )
    dashboard = _StubDashboard()
    observed = []

    def runtime(*, dashboard, optimization_spec, run_dir, catalog_root=None, assets_root=None):
        observed.append((dashboard, optimization_spec, run_dir))

    thread = launch(dashboard, target, prompt_optimization_runtime=runtime)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert observed == [(dashboard, target.optimization_spec, run_dir)]
    assert dashboard.finished_with == [None]
    assert not hasattr(target, "script_path")
    meta = json.loads((run_dir / "autumn_meta.json").read_text())
    assert meta["run_kind"] == "prompt_optimization"
    assert meta["run_name"] == "prompt-run"
    assert meta["prompt_optimization_spec"] == target.optimization_spec.to_dict()


def test_launch_reports_unsupported_target_through_dashboard():
    dashboard = _StubDashboard()

    thread = launch(dashboard, object())
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(dashboard.finished_with) == 1
    assert isinstance(dashboard.finished_with[0], TypeError)
    assert str(dashboard.finished_with[0]) == "unsupported run target: object"


def test_unsupported_target_fails_dashboard_state(tmp_path):
    run_dir = tmp_path / "bad-target"
    run_dir.mkdir()
    state = DashboardState(run_name="bad-target", run_dir=run_dir)
    dashboard = DashboardCallback(_ImmediateApp(), state)

    thread = launch(dashboard, object())
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert state.status is RunStatus.FAILED
    assert state.error == "unsupported run target: object"


def test_prompt_optimization_target_drives_dashboard_lifecycle(tmp_path):
    run_dir = tmp_path / "prompt-run"
    target = PromptOptimizationRunSpec(
        optimization_spec=_prompt_optimization_spec(),
        run_dir=run_dir,
    )
    state = DashboardState(
        run_name=target.run_name,
        run_dir=run_dir,
        run_kind=RunKind.PROMPT_OPTIMIZATION,
    )
    dashboard = DashboardCallback(_ImmediateApp(), state)

    def runtime(*, dashboard, optimization_spec, run_dir, catalog_root=None, assets_root=None):
        dashboard.on_optimization_start(
            {
                "trainset_size": 3,
                "valset_size": 2,
                "config": {"engine": {"max_metric_calls": optimization_spec.budget.max_metric_calls}},
            }
        )
        dashboard.on_optimization_end({"best_candidate_idx": 0, "total_iterations": 1})

    thread = launch(dashboard, target, prompt_optimization_runtime=runtime)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert state.run_kind is RunKind.PROMPT_OPTIMIZATION
    assert state.status is RunStatus.COMPLETED
    assert state.trainset_size == 3
    assert state.valset_size == 2
    assert state.max_metric_calls == 10
    assert state.best_idx == 0
    assert state.total_iterations == 1


def test_prompt_optimization_target_respects_stop_file_at_completion(tmp_path):
    run_dir = tmp_path / "prompt-run"
    target = PromptOptimizationRunSpec(
        optimization_spec=_prompt_optimization_spec(),
        run_dir=run_dir,
    )
    state = DashboardState(
        run_name=target.run_name,
        run_dir=run_dir,
        run_kind=RunKind.PROMPT_OPTIMIZATION,
    )
    dashboard = DashboardCallback(_ImmediateApp(), state)

    def runtime(*, dashboard, optimization_spec, run_dir, catalog_root=None, assets_root=None):
        (run_dir / "gepa.stop").touch()

    thread = launch(dashboard, target, prompt_optimization_runtime=runtime)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert state.status is RunStatus.STOPPED


def test_prompt_optimization_runtime_error_fails_dashboard_state(tmp_path):
    run_dir = tmp_path / "prompt-run"
    target = PromptOptimizationRunSpec(
        optimization_spec=_prompt_optimization_spec(),
        run_dir=run_dir,
    )
    state = DashboardState(
        run_name=target.run_name,
        run_dir=run_dir,
        run_kind=RunKind.PROMPT_OPTIMIZATION,
    )
    dashboard = DashboardCallback(_ImmediateApp(), state)

    def runtime(*, dashboard, optimization_spec, run_dir, catalog_root=None, assets_root=None):
        raise ValueError("bad prompt optimization target")

    thread = launch(dashboard, target, prompt_optimization_runtime=runtime)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert state.status is RunStatus.FAILED
    assert state.error == "bad prompt optimization target"
