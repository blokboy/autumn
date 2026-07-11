"""Headless behavioral tests for AutumnApp's Q (graceful stop) / q (quit,
confirm-if-active) / r (resume) keybindings, via Textual's Pilot harness.
These exercise the real app wiring (bindings -> actions -> ConfirmScreen ->
callback), as close to a human pressing keys in a terminal as a test can get
without a real TTY.
"""

import asyncio
import json
import threading
from pathlib import Path

from app import AutumnApp
from models import RunStatus
from screens.confirm_screen import ConfirmScreen


def _write_meta(run_dir: Path, script_path: Path, run_name: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "autumn_meta.json").write_text(
        json.dumps({"script_path": str(script_path), "run_name": run_name, "launched_at": "2026-07-10T00:00:00"})
    )


async def _drain_dry_run_replay() -> None:
    """dry_run_events.replay's scripted sequence (5 iterations x 0.5s) has no
    way to be cancelled early -- it ignores gepa.stop entirely, unlike a real
    run. Waiting it out here keeps its background thread from calling back
    into this test's event loop after run_test()'s context (and its loop)
    have already torn down."""
    await asyncio.sleep(3.0)


async def test_stop_run_shows_confirm_then_touches_stop_file(tmp_path):
    run_dir = tmp_path / "q-test"
    app = AutumnApp(
        runs_root=tmp_path, run_name="q-test", run_dir=run_dir, script_path=Path("unused.py"), dry_run=True
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.state.status is RunStatus.RUNNING

        await pilot.press("Q")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)

        await pilot.press("y")
        await pilot.pause()
        assert (run_dir / "gepa.stop").exists()

        await _drain_dry_run_replay()


async def test_stop_run_confirm_creates_missing_run_dir(tmp_path):
    """--dry-run never creates run_dir on disk (unlike a real run, where
    runner.launch's mkdir already ran) -- confirming Q must not crash on a
    missing directory."""
    run_dir = tmp_path / "does-not-exist-yet"
    app = AutumnApp(
        runs_root=tmp_path, run_name="q-test", run_dir=run_dir, script_path=Path("unused.py"), dry_run=True
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not run_dir.exists()

        await pilot.press("Q")
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
        assert (run_dir / "gepa.stop").exists()

        await _drain_dry_run_replay()


async def test_stop_run_noop_without_an_active_run(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")  # bare construction lands on InputScreen; empty Enter -> browse
        await pilot.pause()
        assert app.state is None
        depth_before = len(app.screen_stack)

        await pilot.press("Q")
        await pilot.pause()
        assert len(app.screen_stack) == depth_before


async def test_quit_confirms_while_a_run_is_active(tmp_path):
    run_dir = tmp_path / "quit-test"
    app = AutumnApp(
        runs_root=tmp_path, run_name="quit-test", run_dir=run_dir, script_path=Path("unused.py"), dry_run=True
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.state.status is RunStatus.RUNNING

        await pilot.press("q")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        assert "Press Q instead" in app.screen.message

        await pilot.press("n")
        await pilot.pause()
        assert not app._exit

        await _drain_dry_run_replay()


async def test_quit_exits_immediately_with_no_active_run(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")  # bare construction lands on InputScreen; empty Enter -> browse
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        assert app._exit


async def test_confirm_from_thread_blocks_background_thread_until_confirmed(tmp_path):
    """`_confirm_from_thread` (used by GroqRunner's mutating system tools --
    see docs/prd/chat-cli-parity-tools.md) bridges a confirmation request
    from a background thread to the main Textual thread and blocks the
    background thread until the modal resolves. Simulated here the same way
    test_groq_streaming.py's `_ScriptedGroqRunner` interleaves a real
    background thread with Pilot actions, rather than racing wall-clock
    timing."""
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        result: dict[str, bool] = {}

        def call_from_background() -> None:
            result["confirmed"] = app._confirm_from_thread("Remove `thing`?", 0.0)

        thread = threading.Thread(target=call_from_background, daemon=True)
        thread.start()

        await pilot.pause(0.05)
        assert isinstance(app.screen, ConfirmScreen)
        assert "confirmed" not in result, "background thread must still be blocked"

        await pilot.click("#confirm")
        await pilot.pause()
        thread.join(timeout=2)

        assert result["confirmed"] is True


async def test_confirm_from_thread_returns_false_on_cancel(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        result: dict[str, bool] = {}

        def call_from_background() -> None:
            result["confirmed"] = app._confirm_from_thread("Remove `thing`?", 0.0)

        thread = threading.Thread(target=call_from_background, daemon=True)
        thread.start()

        await pilot.pause(0.05)
        await pilot.press("escape")
        await pilot.pause()
        thread.join(timeout=2)

        assert result["confirmed"] is False


async def test_confirm_from_thread_respects_initial_delay(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        result: dict[str, bool] = {}

        def call_from_background() -> None:
            result["confirmed"] = app._confirm_from_thread("Remove `thing`?", 5.0)

        thread = threading.Thread(target=call_from_background, daemon=True)
        thread.start()

        await pilot.pause(0.05)
        assert app.screen.query_one("#confirm").disabled is True

        await pilot.press("escape")
        await pilot.pause()
        thread.join(timeout=2)

        assert result["confirmed"] is False


async def test_resume_relaunches_a_selected_stopped_run(tmp_path):
    """`r` on a STOPPED historical run selected in the sidebar re-launches it
    live via runner.launch, recovering script_path from autumn_meta.json --
    the DashboardScreen lookup must survive even when it's not the top of the
    screen stack (regression: action_resume previously used
    self.query_one(DashboardScreen), which only searches the active screen
    and raised NoMatches on every real invocation)."""
    run_dir = tmp_path / "stopped-run"
    script = tmp_path / "trivial_script.py"
    # A brief sleep -- long enough that the relaunched thread is still
    # RUNNING when we check right after pressing 'r', but short enough that
    # explicitly waiting it out below (so its daemon thread doesn't outlive
    # this test's event loop) stays fast.
    script.write_text("import time\ntime.sleep(0.3)\n")
    _write_meta(run_dir, script, "stopped-run")
    (run_dir / "run_log.json").write_text("[]")
    (run_dir / "candidates.json").write_text("[]")
    (run_dir / "gepa.stop").write_text("")  # dead PID (none written) + gepa.stop -> STOPPED

    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")  # bare construction lands on InputScreen; empty Enter -> browse
        await pilot.pause()
        assert app.state is None
        assert app._dashboard_screen.selected_run_dir == run_dir
        assert app._dashboard_screen.selected_run_status() is RunStatus.STOPPED

        await pilot.press("r")
        await pilot.pause()
        assert app.state is not None
        assert app.state.run_dir == run_dir
        assert app.state.status is RunStatus.RUNNING

        # Wait out the relaunched script's thread so it can't try to call
        # back into this test's event loop after teardown.
        await asyncio.sleep(0.6)
        await pilot.pause()
        assert app.state.status is RunStatus.COMPLETED
