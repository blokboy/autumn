"""Behavioral tests for DashboardScreen's CommandBar and AutumnApp's
in-memory run queue (`:` to focus, `gepa ...` launch-vs-queue branching,
auto-advance on completion, queued prompt auto-advance), via Textual's Pilot
harness against a real AutumnApp.

Live runs here use trivial real scripts (`runpy`-executed, no GEPA
dependency) with a short `time.sleep` rather than `--dry-run`:
`fixtures.dry_run_events.replay` runs a fixed ~2.5s scripted sequence that
can't be waited on precisely (see test_app.py's own comments on this), which
would make the timing-sensitive assertions below (still-RUNNING-until-a-
specific-moment, auto-advance-after-completion) unnecessarily flaky. A short
sleep gives the same "a run is live for a little while" precondition much
faster and more predictably.
"""

import asyncio

from textual.widgets import Input

from autumn.app import AutumnApp
from autumn.models import ChatMessage, RunStatus
from autumn.screens.dashboard_screen import DashboardScreen
from autumn.screens.help_screen import HelpScreen
from autumn.widgets.command_bar import CommandBar


def _write_sleepy_script(path, seconds: float) -> None:
    path.write_text(f"import time\ntime.sleep({seconds})\n")


async def _type_and_submit(pilot, text: str) -> None:
    await pilot.press(":")
    await pilot.pause()
    pilot.app.screen.query_one(CommandBar).query_one(Input).insert_text_at_cursor(text)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


async def test_colon_focuses_command_bar(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")  # empty Enter on InputScreen -> browse DashboardScreen
        await pilot.pause()

        bar_input = app.screen.query_one(CommandBar).query_one(Input)
        assert not bar_input.has_focus

        await pilot.press(":")
        await pilot.pause()
        assert bar_input.has_focus


async def test_command_bar_copy_invites_chat_or_gepa(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")  # empty Enter on InputScreen -> browse DashboardScreen
        await pilot.pause()

        bar_input = app.screen.query_one(CommandBar).query_one(Input)

        assert bar_input.placeholder == ": ask Autumn a question, gepa my_script.py, or gepa --run-dir examples/"
        assert "stub" not in bar_input.placeholder.lower()


async def test_question_mark_still_opens_help_when_bar_unfocused(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert not app.screen.query_one(CommandBar).query_one(Input).has_focus
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)


async def test_gepa_command_via_bar_launches_immediately_when_no_run_live(tmp_path):
    script = tmp_path / "demo_script.py"
    _write_sleepy_script(script, 0.3)
    run_dir = tmp_path / "bar-launch"

    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")  # -> browse DashboardScreen, no run live
        await pilot.pause()

        await _type_and_submit(
            pilot, f"gepa {script} --name bar-launch --run-dir {run_dir}"
        )

        assert isinstance(app.screen, DashboardScreen)
        assert app.state is not None
        assert app.state.run_name == "bar-launch"
        assert app.state.status is RunStatus.RUNNING
        assert app.pending_queue == []

        await asyncio.sleep(0.4)  # let the script's thread finish cleanly


async def test_gepa_command_via_bar_queues_while_a_run_is_live(tmp_path):
    first_script = tmp_path / "first.py"
    _write_sleepy_script(first_script, 1.0)
    second_script = tmp_path / "second.py"
    _write_sleepy_script(second_script, 0.1)

    app = AutumnApp(
        runs_root=tmp_path,
        run_name="first",
        run_dir=tmp_path / "first",
        script_path=first_script,
        dry_run=False,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.state.status is RunStatus.RUNNING

        await _type_and_submit(
            pilot, f"gepa {second_script} --name second --run-dir {tmp_path / 'second'}"
        )

        # Queued, not launched: the live run is still "first".
        assert app.state.run_name == "first"
        assert len(app.pending_queue) == 1
        assert app.pending_queue[0].run_name == "second"

        await asyncio.sleep(1.1)  # let "first" finish cleanly before teardown


async def test_queue_auto_advances_when_live_run_finishes(tmp_path):
    # "first" sleeps long enough to comfortably outlast the Pilot overhead of
    # typing + submitting the queued command (a handful of `pilot.pause()`
    # message-pump round-trips, which can add up to noticeable wall-clock time
    # under load) -- otherwise "first" could finish before the queue command
    # is even submitted, and it'd launch immediately instead of queuing.
    quick_script = tmp_path / "quick.py"
    _write_sleepy_script(quick_script, 0.8)
    queued_script = tmp_path / "queued.py"
    _write_sleepy_script(queued_script, 1.0)

    app = AutumnApp(
        runs_root=tmp_path,
        run_name="first",
        run_dir=tmp_path / "first",
        script_path=quick_script,
        dry_run=False,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.state.status is RunStatus.RUNNING

        await _type_and_submit(
            pilot, f"gepa {queued_script} --name second --run-dir {tmp_path / 'second'}"
        )
        assert len(app.pending_queue) == 1

        # Let "first" finish and the queue poll notice, however much of its
        # 0.8s was already spent during the typing/submit above.
        await asyncio.sleep(1.2)
        await pilot.pause()

        assert app.pending_queue == []
        assert app.state.run_name == "second"
        assert app.state.status is RunStatus.RUNNING

        await asyncio.sleep(1.1)  # let "second" finish cleanly before teardown


async def test_prompt_queued_while_live_auto_advances_to_chat_reply(tmp_path):
    quick_script = tmp_path / "quick.py"
    _write_sleepy_script(quick_script, 0.8)

    app = AutumnApp(
        runs_root=tmp_path,
        run_name="first",
        run_dir=tmp_path / "first",
        script_path=quick_script,
        dry_run=False,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.state.status is RunStatus.RUNNING

        await _type_and_submit(pilot, "summarize my last run")
        assert len(app.pending_queue) == 1

        await asyncio.sleep(1.2)
        await pilot.pause()

        assert app.pending_queue == []
        assert app.chat_messages == [
            ChatMessage(role="user", text="summarize my last run"),
            ChatMessage(
                role="assistant",
                text="Offline local response: summarize my last run",
                model="autumn/offline-tiny",
            ),
        ]
        # No new run was launched by the prompt entry -- the queue just emptied.
        assert app.state.run_name == "first"


async def test_bad_gepa_syntax_via_bar_shows_error_and_does_not_queue(tmp_path):
    script = tmp_path / "demo_script.py"
    script.write_text("pass\n")

    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")  # -> browse DashboardScreen, no run live
        await pilot.pause()

        await _type_and_submit(pilot, f"gepa {script} --bogus-flag")

        assert app.pending_queue == []
        assert app.state is None
        notifications = list(app._notifications)
        assert any(n.severity == "error" for n in notifications)
