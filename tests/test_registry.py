import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from models import DashboardState, RunKind, RunStatus, RunSummary
from registry import infer_status, load_dashboard_state, merge_live, scan


def _dead_pid() -> int:
    """A PID guaranteed to be dead: spawn a subprocess and wait for it to exit."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _write_pid(run_dir: Path, pid: int) -> None:
    (run_dir / "autumn.pid").write_text(str(pid))


def _write_run_log(run_dir: Path, entries: list[dict]) -> None:
    (run_dir / "run_log.json").write_text(json.dumps(entries))


def _write_candidates(run_dir: Path, entries: list[dict]) -> None:
    (run_dir / "candidates.json").write_text(json.dumps(entries))


def _write_status_marker(run_dir: Path, status: str) -> None:
    (run_dir / "autumn_status.json").write_text(json.dumps({"status": status}))


def _write_meta(run_dir: Path, meta: dict) -> None:
    (run_dir / "autumn_meta.json").write_text(json.dumps(meta))


def _write_candidates_snapshot(
    run_dir: Path, *, best_idx, best_score, pareto_front: list[int], candidates: list[dict]
) -> None:
    (run_dir / "autumn_candidates.json").write_text(
        json.dumps(
            {
                "best_idx": best_idx,
                "best_score": best_score,
                "pareto_front": pareto_front,
                "candidates": candidates,
            }
        )
    )


# A real GEPA run_log.json entry (captured verbatim from an actual `gepa.optimize`
# run against src/fixtures/demo_script.py) -- GEPA's own per-iteration trace
# schema, with no "event"/"terminal" key of any kind, whether the run finished
# cleanly or was killed mid-iteration. The old terminal-event heuristic alone
# always reports FAILED for genuinely completed real runs; this is what
# autumn_status.json exists to fix. See test_infer_status_completed_via_status_marker_even_with_real_gepa_run_log.
_REAL_GEPA_RUN_LOG_ENTRY = {
    "i": 0,
    "selected_program_candidate": 0,
    "subsample_ids": [4, 2, 1],
    "subsample_scores": [0.0, 0.0, 0.0],
    "new_subsample_scores": [1.0, 1.0, 1.0],
    "new_program_idx": 1,
    "evaluated_val_indices": [0, 1, 2, 3, 4],
}


# --- infer_status -----------------------------------------------------------


def test_infer_status_running_when_pid_alive(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_pid(run_dir, os.getpid())

    assert infer_status(run_dir) == RunStatus.RUNNING


def test_infer_status_stopped_when_pid_dead_and_stop_file_present(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_pid(run_dir, _dead_pid())
    (run_dir / "gepa.stop").touch()

    assert infer_status(run_dir) == RunStatus.STOPPED


def test_infer_status_completed_when_dead_and_terminal_event_in_log(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_pid(run_dir, _dead_pid())
    _write_run_log(run_dir, [{"event": "on_optimization_end"}])

    assert infer_status(run_dir) == RunStatus.COMPLETED


def test_infer_status_completed_with_missing_pid_file_and_terminal_event(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_run_log(run_dir, [{"event": "optimization_end"}])

    assert infer_status(run_dir) == RunStatus.COMPLETED


def test_infer_status_completed_via_truthy_terminal_key(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_run_log(run_dir, [{"terminal": True}])

    assert infer_status(run_dir) == RunStatus.COMPLETED


def test_infer_status_failed_when_dead_no_stop_no_terminal_event(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_pid(run_dir, _dead_pid())
    _write_run_log(run_dir, [{"event": "on_iteration"}])

    assert infer_status(run_dir) == RunStatus.FAILED


def test_infer_status_failed_when_no_pid_file_and_no_run_log(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "candidates.json").write_text("[]")

    assert infer_status(run_dir) == RunStatus.FAILED


def test_infer_status_malformed_pid_file_treated_as_no_live_pid(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "autumn.pid").write_text("not-a-pid")
    _write_run_log(run_dir, [{"event": "on_optimization_end"}])

    assert infer_status(run_dir) == RunStatus.COMPLETED


def test_infer_status_failed_via_old_heuristic_against_real_gepa_run_log_shape(tmp_path):
    """Regression guard: GEPA's real run_log.json (see _REAL_GEPA_RUN_LOG_ENTRY)
    has no terminal marker at all, so a completed real run with no
    autumn_status.json (e.g. one from before this marker existed) falls through
    the old heuristic to FAILED. This is the exact bug a real end-to-end run
    surfaced -- documented here as the known limitation the marker below fixes,
    not something to "fix" by reinterpreting run_log.json further (there is
    nothing in it to distinguish clean completion from a mid-iteration kill)."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_pid(run_dir, _dead_pid())
    _write_run_log(run_dir, [_REAL_GEPA_RUN_LOG_ENTRY])

    assert infer_status(run_dir) == RunStatus.FAILED


def test_infer_status_completed_via_status_marker_even_with_real_gepa_run_log(tmp_path):
    """The actual fix: autumn_status.json (written by
    DashboardCallback.mark_script_finished) is authoritative and doesn't depend
    on run_log.json's shape at all."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_pid(run_dir, _dead_pid())
    _write_run_log(run_dir, [_REAL_GEPA_RUN_LOG_ENTRY])
    _write_status_marker(run_dir, "completed")

    assert infer_status(run_dir) == RunStatus.COMPLETED


def test_infer_status_failed_via_status_marker(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_pid(run_dir, _dead_pid())
    _write_status_marker(run_dir, "failed")

    assert infer_status(run_dir) == RunStatus.FAILED


def test_infer_status_gepa_stop_file_takes_priority_over_completed_marker(tmp_path):
    """A graceful stop still runs GEPA's engine loop to a normal exit (it just
    stops accepting new iterations), so mark_script_finished ends up writing
    status="completed" even for a gracefully-stopped run -- it can't tell the
    difference on its own. The gepa.stop file (GEPA's own FileStopper
    convention) is the only unambiguous signal that this was a deliberate early
    stop, so it must be checked, and win, before the marker."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_pid(run_dir, _dead_pid())
    (run_dir / "gepa.stop").touch()
    _write_status_marker(run_dir, "completed")

    assert infer_status(run_dir) == RunStatus.STOPPED


def test_infer_status_malformed_status_marker_falls_back_to_old_heuristic(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_pid(run_dir, _dead_pid())
    (run_dir / "autumn_status.json").write_text("not valid json")
    _write_run_log(run_dir, [{"event": "on_optimization_end"}])

    assert infer_status(run_dir) == RunStatus.COMPLETED


# --- merge_live --------------------------------------------------------------


def _summary(name: str, last_modified: datetime, run_dir: Path | None = None) -> RunSummary:
    return RunSummary(
        name=name,
        run_dir=run_dir or Path(f"/runs/{name}"),
        status=RunStatus.COMPLETED,
        best_score=None,
        num_candidates=0,
        last_modified=last_modified,
    )


def test_merge_live_with_no_live_state_returns_historical_sorted_newest_first():
    now = datetime.now()
    older = _summary("older", now - timedelta(hours=2))
    newer = _summary("newer", now)
    middle = _summary("middle", now - timedelta(hours=1))

    result = merge_live([older, newer, middle], None)

    assert [r.name for r in result] == ["newer", "middle", "older"]


def test_merge_live_prepends_new_live_run_not_in_historical():
    now = datetime.now()
    older = _summary("older", now - timedelta(hours=2))
    newer = _summary("newer", now)
    historical = [older, newer]

    live_state = DashboardState(run_name="live-run", run_dir=Path("/runs/live-run"))

    result = merge_live(historical, live_state)

    assert result[0].name == "live-run"
    assert result[0].is_live is True
    assert [r.name for r in result[1:]] == ["newer", "older"]


def test_merge_live_drops_historical_duplicate_of_live_run():
    now = datetime.now()
    older = _summary("older", now - timedelta(hours=2))
    stale_live_entry = _summary("live-run", now - timedelta(hours=1), run_dir=Path("/runs/live-run"))
    historical = [older, stale_live_entry]

    live_state = DashboardState(run_name="live-run", run_dir=Path("/runs/live-run"))

    result = merge_live(historical, live_state)

    assert [r.name for r in result] == ["live-run", "older"]
    assert result[0].is_live is True


def test_merge_live_does_not_mutate_input_list_or_elements():
    now = datetime.now()
    older = _summary("older", now - timedelta(hours=2))
    stale_live_entry = _summary("live-run", now - timedelta(hours=1), run_dir=Path("/runs/live-run"))
    historical = [stale_live_entry, older]
    historical_copy = list(historical)

    live_state = DashboardState(run_name="live-run", run_dir=Path("/runs/live-run"))

    merge_live(historical, live_state)

    assert historical == historical_copy
    assert historical[0] is stale_live_entry
    assert historical[0].is_live is False
    assert historical[0].last_modified == now - timedelta(hours=1)


# --- scan ----------------------------------------------------------------------


def _set_mtime(path: Path, when: datetime) -> None:
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def test_scan_returns_summaries_sorted_newest_first_with_correct_fields(tmp_path):
    now = datetime.now()

    run_a = tmp_path / "run_a"
    run_a.mkdir()
    _write_candidates(run_a, [{"component": "p1"}, {"component": "p2"}])
    _write_run_log(run_a, [{"event": "on_optimization_end"}])
    _set_mtime(run_a, now - timedelta(hours=2))

    run_b = tmp_path / "run_b"
    run_b.mkdir()
    _write_candidates(run_b, [{"component": "p1"}])
    _write_pid(run_b, os.getpid())
    _set_mtime(run_b, now)

    run_c = tmp_path / "run_c"
    run_c.mkdir()
    _write_candidates(run_c, [])
    _write_run_log(run_c, [{"event": "on_iteration"}])
    _set_mtime(run_c, now - timedelta(hours=1))

    stray = tmp_path / "not_a_run"
    stray.mkdir()
    (stray / "readme.txt").write_text("nothing to see here")

    results = scan(tmp_path)

    assert [r.name for r in results] == ["run_b", "run_c", "run_a"]

    by_name = {r.name: r for r in results}
    assert by_name["run_a"].status == RunStatus.COMPLETED
    assert by_name["run_a"].run_kind == RunKind.SCRIPT
    assert by_name["run_a"].num_candidates == 2
    assert by_name["run_b"].status == RunStatus.RUNNING
    assert by_name["run_b"].num_candidates == 1
    assert by_name["run_c"].status == RunStatus.FAILED
    assert by_name["run_c"].num_candidates == 0

    assert "not_a_run" not in by_name


def test_scan_reads_prompt_optimization_run_kind_from_meta(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_candidates(run_dir, [])
    _write_run_log(run_dir, [{"event": "on_optimization_end"}])
    _write_meta(run_dir, {"run_kind": "prompt_optimization", "run_name": "run1"})

    results = scan(tmp_path)

    assert results[0].run_kind == RunKind.PROMPT_OPTIMIZATION


def test_scan_skips_non_run_subdirectories(tmp_path):
    stray = tmp_path / "unrelated"
    stray.mkdir()
    (stray / "notes.txt").write_text("hello")

    results = scan(tmp_path)

    assert results == []


# --- _extract_best_score (via scan()'s best_score field) -----------------------


def test_scan_best_score_picked_up_from_candidates_val_score(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_candidates(run_dir, [{"val_score": 0.5}, {"val_score": 0.9}])
    _write_run_log(run_dir, [{"event": "on_optimization_end"}])

    results = scan(tmp_path)

    assert results[0].best_score == 0.9


def test_scan_best_score_picked_up_from_run_log_score_key(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_candidates(run_dir, [{"component": "p1"}])
    _write_run_log(run_dir, [{"score": 0.3}, {"score": 0.8}, {"event": "on_optimization_end"}])

    results = scan(tmp_path)

    assert results[0].best_score == 0.8


def test_scan_best_score_is_none_when_no_score_key_present(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_candidates(run_dir, [{"component": "p1"}])
    _write_run_log(run_dir, [{"event": "on_optimization_end"}])

    results = scan(tmp_path)

    assert results[0].best_score is None


def test_scan_best_score_prefers_candidates_snapshot_over_inline_keys(tmp_path):
    """Regression guard for the real-run bug: real candidates.json/run_log.json
    never carry inline score keys (see _REAL_GEPA_RUN_LOG_ENTRY), so
    autumn_candidates.json must be checked first."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_candidates(run_dir, [{"system_prompt": "seed"}, {"system_prompt": "better"}])
    _write_run_log(run_dir, [_REAL_GEPA_RUN_LOG_ENTRY])
    _write_status_marker(run_dir, "completed")
    _write_candidates_snapshot(
        run_dir,
        best_idx=1,
        best_score=1.0,
        pareto_front=[1],
        candidates=[
            {"idx": 0, "val_score": 0.0, "discovered_iteration": None, "parent_ids": [], "is_pareto_member": False},
            {"idx": 1, "val_score": 1.0, "discovered_iteration": 1, "parent_ids": [0], "is_pareto_member": True},
        ],
    )

    results = scan(tmp_path)

    assert results[0].best_score == 1.0


# --- load_dashboard_state -------------------------------------------------------


def test_load_dashboard_state_real_gepa_files_get_scores_from_snapshot(tmp_path):
    """The actual end-to-end bug this was written to fix: browsing a real,
    successfully completed GEPA run showed no best candidate, no score, and an
    empty Pareto front, purely because GEPA's own plain-JSON files never carry
    that information. autumn_candidates.json is what makes it show up."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_candidates(
        run_dir,
        [
            {"system_prompt": "Repeat the input back unchanged."},
            {"system_prompt": "Convert to uppercase."},
        ],
    )
    _write_run_log(run_dir, [_REAL_GEPA_RUN_LOG_ENTRY])
    _write_status_marker(run_dir, "completed")
    _write_candidates_snapshot(
        run_dir,
        best_idx=1,
        best_score=1.0,
        pareto_front=[1],
        candidates=[
            {"idx": 0, "val_score": 0.0, "discovered_iteration": None, "parent_ids": [], "is_pareto_member": False},
            {"idx": 1, "val_score": 1.0, "discovered_iteration": 1, "parent_ids": [0], "is_pareto_member": True},
        ],
    )

    state = load_dashboard_state(run_dir)

    assert state.run_kind == RunKind.SCRIPT
    assert state.status == RunStatus.COMPLETED
    assert state.best_idx == 1
    assert state.best_score == 1.0
    assert state.pareto_front == {1}
    assert state.candidates[1].val_score == 1.0
    assert state.candidates[1].discovered_iteration == 1
    assert state.candidates[1].parent_ids == [0]
    assert state.candidates[1].is_pareto_member is True
    # text still comes from GEPA's own candidates.json, not the snapshot.
    assert state.candidates[1].text == {"system_prompt": "Convert to uppercase."}
    assert state.candidates[0].is_pareto_member is False


def test_load_dashboard_state_reads_prompt_optimization_run_kind_from_meta(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_candidates(run_dir, [])
    _write_run_log(run_dir, [{"event": "on_optimization_end"}])
    _write_meta(run_dir, {"run_kind": "prompt_optimization", "run_name": "run1"})

    state = load_dashboard_state(run_dir)

    assert state.run_kind == RunKind.PROMPT_OPTIMIZATION


def test_load_dashboard_state_falls_back_to_inline_score_keys_without_snapshot(tmp_path):
    """No autumn_candidates.json (older run, or a hand-crafted fixture) -- the
    pre-existing best-effort behavior must keep working unchanged."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_candidates(run_dir, [{"val_score": 0.5}, {"val_score": 0.9}])
    _write_run_log(run_dir, [{"event": "on_optimization_end"}])

    state = load_dashboard_state(run_dir)

    assert state.best_score == 0.9
    assert state.best_idx == 1
    assert state.pareto_front == set()
    assert state.candidates[1].val_score == 0.9
