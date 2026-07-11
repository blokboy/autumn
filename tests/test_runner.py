import json
import time
from pathlib import Path

from models import LiveRunSpec
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
