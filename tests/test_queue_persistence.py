"""Integration tests (via Textual's Pilot harness) for issue #9: persisting
AutumnApp.pending_queue to a per-session file on every change, and bare
`autumn`'s Resume/Start-fresh prompt for leftover sessions on the next
launch."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from textual.widgets import Static

import chat_store, queue_store
from app import AutumnApp
from cli import LaunchSpec
from models import ChatMessage, RunStatus
from screens.confirm_screen import ConfirmScreen
from screens.dashboard_screen import DashboardScreen
from screens.input_screen import InputScreen


def _dead_pid() -> int:
    """A PID guaranteed to be dead: spawn a subprocess and wait for it to exit."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _write_sleepy_script(path: Path, seconds: float) -> None:
    path.write_text(f"import time\ntime.sleep({seconds})\n")


async def test_append_persists_to_this_sessions_file(tmp_path):
    sessions_root = tmp_path / "sessions"
    first_script = tmp_path / "first.py"
    _write_sleepy_script(first_script, 1.0)
    second_script = tmp_path / "second.py"
    second_script.write_text("pass\n")

    app = AutumnApp(
        runs_root=tmp_path,
        run_name="first",
        run_dir=tmp_path / "first",
        script_path=first_script,
        dry_run=False,
        queue_sessions_root=sessions_root,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.state.status is RunStatus.RUNNING

        app.submit_command(f"gepa {second_script} --name second --run-dir {tmp_path / 'second'}")
        await pilot.pause()

        assert app._queue_session_path.exists()
        payload = json.loads(app._queue_session_path.read_text())
        assert payload["items"] == [
            {
                "kind": "gepa",
                "run_name": "second",
                "run_dir": str(tmp_path / "second"),
                "script_path": str(second_script),
                "dry_run": False,
            }
        ]

        await asyncio.sleep(1.1)  # let "first" finish cleanly before teardown


async def test_pop_deletes_session_file_when_queue_drains_to_empty(tmp_path):
    sessions_root = tmp_path / "sessions"
    quick_script = tmp_path / "quick.py"
    _write_sleepy_script(quick_script, 0.6)

    app = AutumnApp(
        runs_root=tmp_path,
        run_name="first",
        run_dir=tmp_path / "first",
        script_path=quick_script,
        dry_run=False,
        queue_sessions_root=sessions_root,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.submit_command("a queued prompt")
        await pilot.pause()
        assert app._queue_session_path.exists()

        await asyncio.sleep(0.9)
        await pilot.pause()

        assert app.pending_queue == []
        assert not app._queue_session_path.exists()


async def test_resume_merges_leftover_sessions_and_auto_launches_first_item(tmp_path):
    sessions_root = tmp_path / "sessions"
    script = tmp_path / "queued_script.py"
    _write_sleepy_script(script, 0.3)

    leftover_a = queue_store.session_path(sessions_root, "leftover-a")
    leftover_b = queue_store.session_path(sessions_root, "leftover-b")
    spec = LaunchSpec(run_name="resumed", run_dir=tmp_path / "resumed", script_path=script, dry_run=False)
    queue_store.persist_queue(leftover_a, [spec], pid=_dead_pid())
    queue_store.persist_queue(leftover_b, ["a queued prompt"], pid=_dead_pid())
    # Force leftover_a to sort first (oldest-touched) so the merge order is deterministic.
    older = leftover_a.stat().st_mtime - 10
    os.utime(leftover_a, (older, older))

    app = AutumnApp(runs_root=tmp_path, queue_sessions_root=sessions_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        assert "2 pending items" in app.screen.message
        assert "2 previous sessions" in app.screen.message

        await pilot.press("y")
        await pilot.pause()
        await pilot.pause()

        assert isinstance(app.screen, DashboardScreen)
        assert app.state is not None
        assert app.state.run_name == "resumed"
        assert app.state.status is RunStatus.RUNNING
        # The gepa item launched immediately; the chat prompt stays queued behind it.
        assert app.pending_queue == ["a queued prompt"]
        assert app._queue_session_path.exists()
        assert json.loads(app._queue_session_path.read_text())["items"] == [
            {"kind": "prompt", "text": "a queued prompt"}
        ]

        # The old leftover files were consumed by the merge either way.
        assert not leftover_a.exists()
        assert not leftover_b.exists()

        await asyncio.sleep(0.5)  # let "resumed" finish cleanly before teardown


async def test_resume_restores_leftover_chat_session(tmp_path):
    chat_sessions_root = tmp_path / "chat-sessions"
    leftover = chat_store.session_path(chat_sessions_root, "leftover")
    chat_store.persist_chat(
        leftover,
        [
            ChatMessage(role="user", text="what happened?"),
            ChatMessage(
                role="assistant",
                text="Offline local response: what happened?",
                model="autumn/offline-tiny",
            ),
        ],
        pid=_dead_pid(),
    )

    app = AutumnApp(
        runs_root=tmp_path,
        queue_sessions_root=tmp_path / "queue-sessions",
        chat_sessions_root=chat_sessions_root,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        assert "2 pending chat messages" in app.screen.message

        await pilot.press("y")
        await pilot.pause()

        assert isinstance(app.screen, DashboardScreen)
        assert app.chat_messages == [
            ChatMessage(role="user", text="what happened?"),
            ChatMessage(
                role="assistant",
                text="Offline local response: what happened?",
                model="autumn/offline-tiny",
            ),
        ]
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "You: what happened?" in str(chat_text)
        assert not leftover.exists()


async def test_resume_merges_leftover_queue_and_chat_into_current_sessions(tmp_path):
    queue_sessions_root = tmp_path / "queue-sessions"
    chat_sessions_root = tmp_path / "chat-sessions"
    queue_leftover = queue_store.session_path(queue_sessions_root, "queue-leftover")
    chat_leftover = chat_store.session_path(chat_sessions_root, "chat-leftover")
    queue_store.persist_queue(queue_leftover, ["queued prompt"], pid=_dead_pid())
    chat_store.persist_chat(
        chat_leftover,
        [
            ChatMessage(role="user", text="old question"),
            ChatMessage(role="assistant", text="old answer", model="autumn/offline-tiny"),
            ChatMessage(role="user", text="queued prompt"),
        ],
        pid=_dead_pid(),
    )

    app = AutumnApp(
        runs_root=tmp_path,
        queue_sessions_root=queue_sessions_root,
        chat_sessions_root=chat_sessions_root,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        assert "1 pending item" in app.screen.message
        assert "3 pending chat messages" in app.screen.message

        await pilot.press("y")
        await pilot.pause()
        await pilot.pause(0.2)

        assert isinstance(app.screen, DashboardScreen)
        assert app.pending_queue == []
        assert app.chat_messages == [
            ChatMessage(role="user", text="old question"),
            ChatMessage(role="assistant", text="old answer", model="autumn/offline-tiny"),
            ChatMessage(role="user", text="queued prompt"),
            ChatMessage(
                role="assistant",
                text="Offline local response: queued prompt",
                model="autumn/offline-tiny",
            ),
        ]
        assert chat_store.load_chat(app._chat_session_path) == app.chat_messages
        assert not queue_leftover.exists()
        assert not chat_leftover.exists()


async def test_start_fresh_deletes_leftover_chat_and_does_not_seed_current_session(tmp_path):
    chat_sessions_root = tmp_path / "chat-sessions"
    leftover = chat_store.session_path(chat_sessions_root, "leftover")
    chat_store.persist_chat(leftover, [ChatMessage(role="user", text="old question")], pid=_dead_pid())

    app = AutumnApp(
        runs_root=tmp_path,
        queue_sessions_root=tmp_path / "queue-sessions",
        chat_sessions_root=chat_sessions_root,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)

        await pilot.press("n")
        await pilot.pause()

        assert isinstance(app.screen, InputScreen)
        assert app.chat_messages == []
        assert not leftover.exists()
        assert not app._chat_session_path.exists()


async def test_invalid_leftover_chat_session_does_not_trigger_resume_prompt(tmp_path):
    chat_sessions_root = tmp_path / "chat-sessions"
    malformed = chat_store.session_path(chat_sessions_root, "malformed")
    malformed.parent.mkdir(parents=True)
    malformed.write_text(
        json.dumps(
            {
                "pid": _dead_pid(),
                "messages": [
                    {"role": "system", "text": "drop me"},
                    {"role": "assistant"},
                ],
            }
        )
    )

    app = AutumnApp(
        runs_root=tmp_path,
        queue_sessions_root=tmp_path / "queue-sessions",
        chat_sessions_root=chat_sessions_root,
    )
    async with app.run_test() as pilot:
        await pilot.pause()

        assert isinstance(app.screen, InputScreen)
        assert malformed.exists()


async def test_start_fresh_deletes_leftover_files_and_shows_input_screen(tmp_path):
    sessions_root = tmp_path / "sessions"
    leftover = queue_store.session_path(sessions_root, "leftover")
    queue_store.persist_queue(leftover, ["queued prompt"], pid=_dead_pid())

    app = AutumnApp(runs_root=tmp_path, queue_sessions_root=sessions_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)

        await pilot.press("n")
        await pilot.pause()

        assert isinstance(app.screen, InputScreen)
        assert app.pending_queue == []
        assert not leftover.exists()


async def test_no_leftover_sessions_skips_prompt(tmp_path):
    sessions_root = tmp_path / "sessions"  # never created -- no session files at all

    app = AutumnApp(runs_root=tmp_path, queue_sessions_root=sessions_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, InputScreen)


async def test_leftover_session_with_live_pid_is_never_offered_or_touched(tmp_path):
    sessions_root = tmp_path / "sessions"
    live = queue_store.session_path(sessions_root, "live")
    queue_store.persist_queue(live, ["queued prompt"], pid=os.getpid())

    app = AutumnApp(runs_root=tmp_path, queue_sessions_root=sessions_root)
    async with app.run_test() as pilot:
        await pilot.pause()

        assert isinstance(app.screen, InputScreen)  # no prompt for a still-live session
        assert live.exists()
        assert json.loads(live.read_text())["items"] == [{"kind": "prompt", "text": "queued prompt"}]
