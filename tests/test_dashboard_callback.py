"""Tests for DashboardCallback: throttling of high-frequency events, on_state_saved,
mark_script_finished, and the multi-objective detection heuristic."""

import time
from pathlib import Path

from dashboard_callback import DashboardCallback
from models import DashboardState, RunStatus


class FakeApp:
    """Stand-in for the Textual App. `call_from_thread` normally hops onto the
    main thread; in tests we just invoke synchronously since there's no real
    event loop / background thread involved."""

    def call_from_thread(self, fn):
        return fn()


def make_callback(run_dir: Path | None = None) -> tuple[DashboardCallback, DashboardState]:
    # A fake, non-existent path is fine for tests that never reach
    # mark_script_finished (it's the only method that touches disk, via
    # _write_status_marker); tests that do must pass a real tmp_path.
    state = DashboardState(run_name="test-run", run_dir=run_dir or Path("/tmp/test-run"))
    callback = DashboardCallback(FakeApp(), state)
    return callback, state


# ---------------------------------------------------------------------------
# Throttle: on_budget_updated / on_valset_evaluated
# ---------------------------------------------------------------------------


def test_budget_updated_throttles_rapid_calls_but_keeps_latest_value():
    callback, state = make_callback()

    callback.on_budget_updated({"metric_calls_used": 1, "metric_calls_remaining": 99})
    version_after_first = state.version
    assert state.metric_calls_used == 1

    # Fired immediately again, well within the 100ms throttle window -- should
    # not push (version unchanged), even though newer data was recorded.
    callback.on_budget_updated({"metric_calls_used": 2, "metric_calls_remaining": 98})
    assert state.version == version_after_first
    assert state.metric_calls_used == 1  # unchanged: throttled away

    # Wait past the throttle window, then fire again -- should push, and reflect
    # the latest data from this call (not the first call's data).
    time.sleep(0.11)
    callback.on_budget_updated({"metric_calls_used": 3, "metric_calls_remaining": 97})
    assert state.version > version_after_first
    assert state.metric_calls_used == 3
    assert state.metric_calls_remaining == 97


def test_valset_evaluated_throttles_rapid_calls_but_keeps_latest_value():
    callback, state = make_callback()

    callback.on_valset_evaluated(
        {
            "candidate_idx": 1,
            "average_score": 0.5,
            "is_best_program": True,
            "candidate": {"system_prompt": "v1"},
        }
    )
    version_after_first = state.version
    assert state.best_score == 0.5

    # Within throttle window -- should be dropped.
    callback.on_valset_evaluated(
        {
            "candidate_idx": 2,
            "average_score": 0.9,
            "is_best_program": True,
            "candidate": {"system_prompt": "v2"},
        }
    )
    assert state.version == version_after_first
    assert state.best_score == 0.5

    # Past throttle window -- should push with the latest data.
    time.sleep(0.11)
    callback.on_valset_evaluated(
        {
            "candidate_idx": 3,
            "average_score": 0.95,
            "is_best_program": True,
            "candidate": {"system_prompt": "v3"},
        }
    )
    assert state.version > version_after_first
    assert state.best_idx == 3
    assert state.best_score == 0.95


def test_other_events_are_not_throttled():
    """Sanity check: state-defining events (e.g. on_iteration_start) still bump
    version on every single call, unlike the two throttled events above."""
    callback, state = make_callback()

    callback.on_iteration_start({"iteration": 1})
    v1 = state.version
    callback.on_iteration_start({"iteration": 2})
    v2 = state.version
    assert v2 > v1


# ---------------------------------------------------------------------------
# on_state_saved
# ---------------------------------------------------------------------------


def test_on_state_saved_logs_and_bumps_version(tmp_path):
    callback, state = make_callback(tmp_path)
    version_before = state.version

    callback.on_state_saved({"run_dir": str(tmp_path)})

    assert state.version > version_before
    assert len(state.log_lines) == 1
    assert "saved" in state.log_lines[-1].text.lower()


def test_on_state_saved_writes_candidates_snapshot(tmp_path):
    """on_state_saved fires on GEPA's own checkpoint cadence -- the right
    moment to refresh autumn_candidates.json so a second `autumn` process
    browsing this run while it's still live sees reasonably fresh data."""
    callback, state = make_callback(tmp_path)
    callback.on_candidate_accepted({"new_candidate_idx": 1, "new_score": 3.0, "parent_ids": [0]})
    callback.on_pareto_front_updated({"new_front": [1], "displaced_candidates": []})
    callback.on_valset_evaluated(
        {"candidate_idx": 1, "average_score": 1.0, "is_best_program": True, "candidate": None}
    )

    callback.on_state_saved({"run_dir": str(tmp_path)})

    import json

    snapshot = json.loads((tmp_path / "autumn_candidates.json").read_text())
    assert snapshot["best_idx"] == 1
    assert snapshot["best_score"] == 1.0
    assert snapshot["pareto_front"] == [1]
    assert snapshot["candidates"] == [
        {
            "idx": 1,
            "val_score": 1.0,
            "discovered_iteration": 0,
            "parent_ids": [0],
            "is_pareto_member": True,
        }
    ]


# ---------------------------------------------------------------------------
# mark_script_finished
# ---------------------------------------------------------------------------


def test_mark_script_finished_none_completes_running_status(tmp_path):
    callback, state = make_callback(tmp_path)
    assert state.status is RunStatus.RUNNING
    version_before = state.version

    callback.mark_script_finished(None)

    assert state.status is RunStatus.COMPLETED
    assert state.version > version_before


def test_mark_script_finished_none_does_not_clobber_existing_terminal_status(tmp_path):
    callback, state = make_callback(tmp_path)
    callback.on_optimization_end({"best_candidate_idx": 0, "total_iterations": 1})
    assert state.status is RunStatus.COMPLETED
    version_before = state.version

    callback.mark_script_finished(None)

    assert state.status is RunStatus.COMPLETED
    assert state.version > version_before  # still logs/bumps, just doesn't reassign status


def test_mark_script_finished_with_exception_sets_failed(tmp_path):
    callback, state = make_callback(tmp_path)
    version_before = state.version
    exc = RuntimeError("boom")

    callback.mark_script_finished(exc)

    assert state.status is RunStatus.FAILED
    assert state.error == "boom"
    assert state.version > version_before
    assert state.log_lines[-1].level == "error"


def test_mark_script_finished_writes_status_marker_for_registry_to_read(tmp_path):
    """mark_script_finished is the only place a real run's on-disk directory
    ever records whether it actually completed -- GEPA's own run_log.json has
    no terminal marker of its own (confirmed against a real run's output), so
    registry.py's status inference depends entirely on this file existing."""
    callback, state = make_callback(tmp_path)

    callback.mark_script_finished(None)

    import json

    marker = json.loads((tmp_path / "autumn_status.json").read_text())
    assert marker == {"status": "completed"}


def test_mark_script_finished_failure_marker_includes_error(tmp_path):
    callback, state = make_callback(tmp_path)

    callback.mark_script_finished(RuntimeError("boom"))

    import json

    marker = json.loads((tmp_path / "autumn_status.json").read_text())
    assert marker == {"status": "failed", "error": "boom"}


# ---------------------------------------------------------------------------
# Multi-objective detection heuristic
# ---------------------------------------------------------------------------


def test_single_objective_fixtures_do_not_flip_multi_objective_flag():
    callback, state = make_callback()

    callback.on_optimization_start({"trainset_size": 20, "valset_size": 10, "config": {}})
    callback.on_valset_evaluated(
        {
            "candidate_idx": 1,
            "average_score": 0.5,
            "is_best_program": True,
            "candidate": {"system_prompt": "v1"},
        }
    )

    assert state.is_multi_objective is False


def test_multi_objective_shaped_event_flips_flag_on_optimization_start():
    callback, state = make_callback()

    callback.on_optimization_start(
        {
            "trainset_size": 20,
            "valset_size": 10,
            "config": {},
            "objectives": {"accuracy": 0.5, "latency": 0.8},
        }
    )

    assert state.is_multi_objective is True


def test_multi_objective_shaped_event_flips_flag_on_valset_evaluated_and_stays_true():
    callback, state = make_callback()

    callback.on_valset_evaluated(
        {
            "candidate_idx": 1,
            "average_score": 0.5,
            "is_best_program": True,
            "candidate": {"system_prompt": "v1"},
            "objective_scores": {"accuracy": 0.5, "latency": 0.8},
        }
    )
    assert state.is_multi_objective is True

    # Later, a plain single-objective-shaped event fires -- the flag must not
    # unset once flipped.
    time.sleep(0.11)
    callback.on_valset_evaluated(
        {
            "candidate_idx": 2,
            "average_score": 0.6,
            "is_best_program": False,
            "candidate": None,
        }
    )
    assert state.is_multi_objective is True
