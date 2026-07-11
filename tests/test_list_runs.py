"""Behavioral tests for the `list_runs` chat tool -- a thin wrapper around
`registry.scan` that must serialize the exact same JSON row shape `autumn
runs --json` prints (cli._runs_as_rows)."""

import json

import list_runs
import registry


def _write_completed_run(runs_root, name, *, num_candidates=2):
    run_dir = runs_root / name
    run_dir.mkdir(parents=True)
    (run_dir / "candidates.json").write_text(
        json.dumps([{"val_score": 0.5 + i * 0.1} for i in range(num_candidates)])
    )
    (run_dir / "run_log.json").write_text(json.dumps([{"event": "on_optimization_end"}]))
    return run_dir


def test_list_runs_returns_empty_array_when_runs_root_does_not_exist(tmp_path):
    result = list_runs.run_tool({}, runs_root=tmp_path / "does-not-exist")

    assert json.loads(result) == []


def test_list_runs_returns_empty_array_for_empty_runs_root(tmp_path):
    result = list_runs.run_tool({}, runs_root=tmp_path)

    assert json.loads(result) == []


def test_list_runs_matches_registry_scan_output(tmp_path):
    _write_completed_run(tmp_path, "my-run-20260101T000000")

    result = json.loads(list_runs.run_tool({}, runs_root=tmp_path))
    [summary] = registry.scan(tmp_path)

    assert result == [
        {
            "name": summary.name,
            "status": summary.status.value,
            "best_score": summary.best_score,
            "num_candidates": summary.num_candidates,
            "last_modified": summary.last_modified.isoformat(),
            "is_live": summary.is_live,
        }
    ]
    assert result[0]["name"] == "my-run-20260101T000000"
    assert result[0]["status"] == "completed"
    assert result[0]["num_candidates"] == 2
    assert result[0]["best_score"] == 0.6
    assert result[0]["is_live"] is False


def test_list_runs_reports_multiple_runs_newest_first(tmp_path):
    import os
    import time

    _write_completed_run(tmp_path, "older-run")
    time.sleep(0.01)
    _write_completed_run(tmp_path, "newer-run")
    # last_modified is read from directory mtime; make the ordering
    # unambiguous regardless of filesystem timestamp resolution.
    older_time = time.time() - 100
    os.utime(tmp_path / "older-run", (older_time, older_time))

    result = json.loads(list_runs.run_tool({}, runs_root=tmp_path))

    assert [row["name"] for row in result] == ["newer-run", "older-run"]


def test_list_runs_ignores_arguments_since_schema_declares_none(tmp_path):
    _write_completed_run(tmp_path, "a-run")

    result = list_runs.run_tool({"anything": "goes here"}, runs_root=tmp_path)

    assert json.loads(result) != []


def test_list_runs_tool_schema_declares_no_required_parameters():
    assert list_runs.TOOL_SCHEMA["function"]["name"] == "list_runs"
    assert list_runs.TOOL_SCHEMA["function"]["parameters"]["properties"] == {}
