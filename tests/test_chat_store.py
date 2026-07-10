"""Session-file persistence tests for dashboard chat transcripts."""

import json
import os
import subprocess
import sys

from autumn import chat_store
from autumn.models import ChatMessage


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_persist_chat_writes_messages_and_pid(tmp_path):
    path = tmp_path / "session.json"
    messages = [
        ChatMessage(role="user", text="summarize my last run"),
        ChatMessage(role="assistant", text="No runs found yet.", model="local/tiny"),
    ]

    chat_store.persist_chat(path, messages, pid=4242)

    payload = json.loads(path.read_text())
    assert payload == {
        "pid": 4242,
        "messages": [
            {"role": "user", "text": "summarize my last run"},
            {"role": "assistant", "text": "No runs found yet.", "model": "local/tiny"},
        ],
    }


def test_persist_chat_empty_deletes_file(tmp_path):
    path = tmp_path / "session.json"
    chat_store.persist_chat(path, [ChatMessage(role="user", text="hello")], pid=1)
    assert path.exists()

    chat_store.persist_chat(path, [], pid=1)
    assert not path.exists()


def test_load_chat_roundtrips_messages(tmp_path):
    path = tmp_path / "session.json"
    messages = [
        ChatMessage(role="user", text="hello"),
        ChatMessage(role="assistant", text="hi", model="local/tiny"),
    ]
    chat_store.persist_chat(path, messages, pid=os.getpid())

    assert chat_store.load_chat(path) == messages


def test_load_chat_drops_malformed_messages_but_keeps_the_rest(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps(
            {
                "pid": 1,
                "messages": [
                    {"role": "user", "text": "keep me"},
                    {"role": "system", "text": "drop me"},
                    {"role": "assistant"},
                    "not even a dict",
                ],
            }
        )
    )

    assert chat_store.load_chat(path) == [ChatMessage(role="user", text="keep me")]


def test_discover_resumable_includes_dead_pid_with_messages(tmp_path):
    sessions_root = tmp_path / "sessions"
    own = chat_store.session_path(sessions_root, "own")
    other = chat_store.session_path(sessions_root, "other")
    chat_store.persist_chat(other, [ChatMessage(role="user", text="leftover")], pid=_dead_pid())

    assert chat_store.discover_resumable(sessions_root, own) == [other]


def test_discover_resumable_excludes_dead_pid_without_valid_messages(tmp_path):
    sessions_root = tmp_path / "sessions"
    own = chat_store.session_path(sessions_root, "own")
    malformed = chat_store.session_path(sessions_root, "malformed")
    malformed.parent.mkdir(parents=True)
    malformed.write_text(
        json.dumps(
            {
                "pid": _dead_pid(),
                "messages": [
                    {"role": "system", "text": "drop me"},
                    {"role": "assistant"},
                    "not even a dict",
                ],
            }
        )
    )

    assert chat_store.discover_resumable(sessions_root, own) == []


def test_discover_resumable_excludes_own_path(tmp_path):
    sessions_root = tmp_path / "sessions"
    own = chat_store.session_path(sessions_root, "own")
    chat_store.persist_chat(own, [ChatMessage(role="user", text="mine")], pid=_dead_pid())

    assert chat_store.discover_resumable(sessions_root, own) == []


def test_discover_resumable_excludes_live_pid(tmp_path):
    sessions_root = tmp_path / "sessions"
    own = chat_store.session_path(sessions_root, "own")
    live = chat_store.session_path(sessions_root, "live")
    chat_store.persist_chat(live, [ChatMessage(role="user", text="still open")], pid=os.getpid())

    assert chat_store.discover_resumable(sessions_root, own) == []


def test_load_and_merge_concatenates_transcripts_in_order(tmp_path):
    sessions_root = tmp_path / "sessions"
    first = chat_store.session_path(sessions_root, "first")
    second = chat_store.session_path(sessions_root, "second")
    chat_store.persist_chat(first, [ChatMessage(role="user", text="one")], pid=1)
    chat_store.persist_chat(second, [ChatMessage(role="assistant", text="two")], pid=2)

    assert chat_store.load_and_merge([first, second]) == [
        ChatMessage(role="user", text="one"),
        ChatMessage(role="assistant", text="two"),
    ]


def test_delete_files_removes_existing_and_tolerates_missing(tmp_path):
    path = tmp_path / "session.json"
    chat_store.persist_chat(path, [ChatMessage(role="user", text="x")], pid=1)

    chat_store.delete_files([path, tmp_path / "already-gone.json"])

    assert not path.exists()
