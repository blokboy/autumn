"""Unit tests for queue_store.py's session-file persistence: serialization
round-tripping, empty-queue-deletes-file semantics, and -- the core
concurrency guarantee -- that a leftover session file is only ever surfaced
as resumable once its writer PID is confirmed dead."""

import json
import os
import subprocess
import sys
from pathlib import Path

import queue_store
from cli import LaunchSpec
from prompt_optimization_contracts import (
    BuiltInMetricRef,
    EvalAssetRef,
    ModelIdentity,
    PromptOptimizationSpec,
    PromptOptimizationSpecBudget,
)


def _dead_pid() -> int:
    """A PID guaranteed to be dead: spawn a subprocess and wait for it to exit."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _spec(name: str, tmp_path: Path) -> LaunchSpec:
    return LaunchSpec(
        run_name=name,
        run_dir=tmp_path / name,
        script_path=tmp_path / f"{name}.py",
        dry_run=True,
    )


def _prompt_optimization_spec(name: str) -> PromptOptimizationSpec:
    return PromptOptimizationSpec(
        prompt="Classify the support ticket.",
        system_prompt=None,
        task_model=ModelIdentity(name="llama-3.1-8b", backend="llama.cpp"),
        optimizer_model=ModelIdentity(name="gpt-4.1", backend="openai", provider="openai"),
        eval_asset=EvalAssetRef(asset_id="support-tickets", version="2026-07-12"),
        metric=BuiltInMetricRef(metric_id="exact_match"),
        run_name=name,
        budget=PromptOptimizationSpecBudget(max_metric_calls=20),
    )


def test_persist_queue_writes_items_and_pid(tmp_path):
    path = tmp_path / "session.json"
    spec = _spec("alpha", tmp_path)
    queue_store.persist_queue(path, [spec, "a queued prompt"], pid=4242)

    payload = json.loads(path.read_text())
    assert payload["pid"] == 4242
    assert payload["items"] == [
        {
            "kind": "gepa",
            "run_name": "alpha",
            "run_dir": str(tmp_path / "alpha"),
            "script_path": str(tmp_path / "alpha.py"),
            "dry_run": True,
        },
        {"kind": "prompt", "text": "a queued prompt"},
    ]


def test_persist_queue_empty_deletes_file(tmp_path):
    path = tmp_path / "session.json"
    queue_store.persist_queue(path, [_spec("alpha", tmp_path)], pid=1)
    assert path.exists()

    queue_store.persist_queue(path, [], pid=1)
    assert not path.exists()


def test_persist_queue_empty_is_a_noop_when_no_file_exists(tmp_path):
    path = tmp_path / "session.json"
    queue_store.persist_queue(path, [], pid=1)
    assert not path.exists()


def test_load_queue_roundtrips_launchspec_and_prompt(tmp_path):
    path = tmp_path / "session.json"
    spec = _spec("beta", tmp_path)
    queue_store.persist_queue(path, [spec, "prompt text"], pid=os.getpid())

    loaded = queue_store.load_queue(path)
    assert loaded == [spec, "prompt text"]


def test_load_typed_queue_roundtrips_script_chat_and_prompt_optimization(tmp_path):
    path = tmp_path / "session.json"
    script = queue_store.ScriptQueueItem(_spec("script-run", tmp_path))
    chat = queue_store.ChatQueueItem("explain the latest run")
    optimization = queue_store.PromptOptimizationQueueItem(_prompt_optimization_spec("optimize-priority"))

    queue_store.persist_queue(path, [script, chat, optimization], pid=4242)

    assert queue_store.load_typed_queue(path) == [script, chat, optimization]
    assert json.loads(path.read_text())["items"] == [
        {
            "kind": "script",
            "run_name": "script-run",
            "run_dir": str(tmp_path / "script-run"),
            "script_path": str(tmp_path / "script-run.py"),
            "dry_run": True,
        },
        {"kind": "chat", "text": "explain the latest run"},
        {
            "kind": "prompt_optimization",
            "spec": _prompt_optimization_spec("optimize-priority").to_dict(),
        },
    ]


def test_load_typed_queue_recovers_legacy_launchspec_and_prompt_items(tmp_path):
    path = tmp_path / "session.json"
    spec = _spec("legacy-run", tmp_path)
    queue_store.persist_queue(path, [spec, "legacy prompt"], pid=4242)

    assert queue_store.load_typed_queue(path) == [
        queue_store.ScriptQueueItem(spec),
        queue_store.ChatQueueItem("legacy prompt"),
    ]


def test_load_queue_missing_file_returns_empty(tmp_path):
    assert queue_store.load_queue(tmp_path / "does-not-exist.json") == []


def test_load_queue_corrupt_json_returns_empty(tmp_path):
    path = tmp_path / "session.json"
    path.write_text("{not valid json")
    assert queue_store.load_queue(path) == []


def test_load_queue_drops_malformed_entries_but_keeps_the_rest(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps(
            {
                "pid": 1,
                "items": [
                    {"kind": "prompt", "text": "keep me"},
                    {"kind": "gepa", "run_name": "missing-fields"},
                    {"kind": "prompt_optimization", "spec": {"schema_version": 1}},
                    {"kind": "unknown-kind"},
                    "not even a dict",
                ],
            }
        )
    )
    assert queue_store.load_queue(path) == ["keep me"]


def test_discover_resumable_excludes_own_path(tmp_path):
    sessions_root = tmp_path / "sessions"
    own = queue_store.session_path(sessions_root, "own")
    queue_store.persist_queue(own, [_spec("alpha", tmp_path)], pid=_dead_pid())

    assert queue_store.discover_resumable(sessions_root, own) == []


def test_discover_resumable_excludes_live_pid(tmp_path):
    sessions_root = tmp_path / "sessions"
    own = queue_store.session_path(sessions_root, "own")
    other = queue_store.session_path(sessions_root, "other")
    queue_store.persist_queue(other, [_spec("alpha", tmp_path)], pid=os.getpid())

    assert queue_store.discover_resumable(sessions_root, own) == []


def test_discover_resumable_includes_dead_pid_with_items(tmp_path):
    sessions_root = tmp_path / "sessions"
    own = queue_store.session_path(sessions_root, "own")
    other = queue_store.session_path(sessions_root, "other")
    queue_store.persist_queue(other, [_spec("alpha", tmp_path)], pid=_dead_pid())

    assert queue_store.discover_resumable(sessions_root, own) == [other]


def test_discover_resumable_excludes_empty_queue(tmp_path):
    sessions_root = tmp_path / "sessions"
    own = queue_store.session_path(sessions_root, "own")
    other = queue_store.session_path(sessions_root, "other")
    sessions_root.mkdir(parents=True)
    other.write_text(json.dumps({"pid": _dead_pid(), "items": []}))

    assert queue_store.discover_resumable(sessions_root, own) == []


def test_discover_resumable_missing_directory_returns_empty(tmp_path):
    own = tmp_path / "sessions" / "own.json"
    assert queue_store.discover_resumable(tmp_path / "sessions", own) == []


def test_load_and_merge_concatenates_in_order(tmp_path):
    sessions_root = tmp_path / "sessions"
    first = queue_store.session_path(sessions_root, "first")
    second = queue_store.session_path(sessions_root, "second")
    queue_store.persist_queue(first, ["one", "two"], pid=1)
    queue_store.persist_queue(second, [_spec("three", tmp_path)], pid=2)

    merged = queue_store.load_and_merge([first, second])
    assert merged == ["one", "two", _spec("three", tmp_path)]


def test_load_and_merge_typed_queue_concatenates_in_order(tmp_path):
    sessions_root = tmp_path / "sessions"
    first = queue_store.session_path(sessions_root, "first")
    second = queue_store.session_path(sessions_root, "second")
    optimization = queue_store.PromptOptimizationQueueItem(_prompt_optimization_spec("optimize"))
    queue_store.persist_queue(first, [queue_store.ChatQueueItem("one"), optimization], pid=1)
    queue_store.persist_queue(second, [queue_store.ScriptQueueItem(_spec("three", tmp_path))], pid=2)

    merged = queue_store.load_and_merge_typed([first, second])
    assert merged == [
        queue_store.ChatQueueItem("one"),
        optimization,
        queue_store.ScriptQueueItem(_spec("three", tmp_path)),
    ]


def test_delete_files_removes_existing_and_tolerates_missing(tmp_path):
    path = tmp_path / "session.json"
    queue_store.persist_queue(path, ["x"], pid=1)
    assert path.exists()

    queue_store.delete_files([path, tmp_path / "already-gone.json"])
    assert not path.exists()
