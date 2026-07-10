"""Unit tests for queue_store.py's session-file persistence: serialization
round-tripping, empty-queue-deletes-file semantics, and -- the core
concurrency guarantee -- that a leftover session file is only ever surfaced
as resumable once its writer PID is confirmed dead."""

import json
import os
import subprocess
import sys
from pathlib import Path

from autumn import queue_store
from autumn.cli import LaunchSpec


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


def test_delete_files_removes_existing_and_tolerates_missing(tmp_path):
    path = tmp_path / "session.json"
    queue_store.persist_queue(path, ["x"], pid=1)
    assert path.exists()

    queue_store.delete_files([path, tmp_path / "already-gone.json"])
    assert not path.exists()
