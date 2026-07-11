"""Behavioral tests for InputScreen: the bare-`autumn` landing screen and its
three submit paths (empty Enter -> browse, `gepa ...` -> live launch, anything
else -> shared chat prompt), via Textual's Pilot harness against a real
AutumnApp constructed the same way cli.py's `_browse()` does (no
run_name/run_dir).
"""

import asyncio

from textual.widgets import Input, Label, TabbedContent

from app import AutumnApp
from models import ChatMessage, RunStatus
from screens.dashboard_screen import DashboardScreen
from screens.help_screen import HelpScreen
from screens.input_screen import InputScreen


async def _drain_dry_run_replay() -> None:
    """See tests/test_app.py -- dry_run_events.replay's scripted sequence
    can't be cancelled early, so outlast its background thread before the
    test's event loop tears down."""
    await asyncio.sleep(3.0)


async def _submit(pilot, text: str) -> None:
    if text:
        pilot.app.screen.query_one(Input).insert_text_at_cursor(text)
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _write_sleepy_script(path, seconds: float) -> None:
    path.write_text(f"import time\ntime.sleep({seconds})\n")


async def test_bare_autumn_opens_input_screen_first(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, InputScreen)
        assert app.state is None


async def test_landing_copy_invites_autumn_questions(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()

        hint = app.screen.query_one(".input-hint", Label)
        command_input = app.screen.query_one("#command-input", Input)

        assert str(hint.content) == "A configurable CLI for LLMs"
        assert command_input.placeholder == "Ask Autumn anything..."


async def test_empty_enter_transitions_to_browse_dashboard(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        depth_before = len(app.screen_stack)
        await _submit(pilot, "")

        assert isinstance(app.screen, DashboardScreen)
        assert app.state is None
        # InputScreen was replaced in place (switch_screen), not stacked under
        # the dashboard.
        assert len(app.screen_stack) == depth_before
        assert not any(isinstance(s, InputScreen) for s in app.screen_stack)


async def test_gepa_command_launches_live_run(tmp_path):
    script = tmp_path / "demo_script.py"
    script.write_text("pass\n")

    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        depth_before = len(app.screen_stack)
        await _submit(pilot, f"gepa {script} --dry-run --name my-run")

        assert isinstance(app.screen, DashboardScreen)
        assert app.state is not None
        assert app.state.run_name == "my-run"
        assert app.state.status is RunStatus.RUNNING
        assert len(app.screen_stack) == depth_before
        assert not any(isinstance(s, InputScreen) for s in app.screen_stack)

        await _drain_dry_run_replay()


async def test_gepa_directory_command_launches_first_script_and_queues_rest(tmp_path):
    script_dir = tmp_path / "examples"
    script_dir.mkdir()
    first_script = script_dir / "a_first.py"
    second_script = script_dir / "b_second.py"
    _write_sleepy_script(first_script, 0.5)
    _write_sleepy_script(second_script, 0.1)
    (script_dir / "README.md").write_text("# ignored\n")

    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, f"gepa --run-dir {script_dir}")

        assert isinstance(app.screen, DashboardScreen)
        assert app.state is not None
        assert app.script_path == first_script
        assert app.state.status is RunStatus.RUNNING
        assert len(app.pending_queue) == 1
        assert app.pending_queue[0].script_path == second_script

        await asyncio.sleep(0.7)


async def test_gepa_command_missing_script_shows_error_and_stays(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, "gepa --dry-run")

        assert isinstance(app.screen, InputScreen)
        assert app.state is None
        notifications = list(app._notifications)
        assert len(notifications) == 1
        assert notifications[0].severity == "error"


async def test_gepa_command_nonexistent_script_shows_error_and_stays(tmp_path):
    missing = tmp_path / "does-not-exist.py"

    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, f"gepa {missing}")

        assert isinstance(app.screen, InputScreen)
        assert app.state is None
        notifications = list(app._notifications)
        assert len(notifications) == 1
        assert notifications[0].severity == "error"


async def test_gepa_command_bad_flag_shows_error_and_stays(tmp_path):
    script = tmp_path / "demo_script.py"
    script.write_text("pass\n")

    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, f"gepa {script} --bogus-flag")

        assert isinstance(app.screen, InputScreen)
        assert app.state is None
        notifications = list(app._notifications)
        assert len(notifications) == 1
        assert notifications[0].severity == "error"


async def test_q_quits_when_input_is_empty(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, InputScreen)

        await pilot.press("q")
        await pilot.pause()

        assert app._exit


async def test_question_mark_opens_help_when_input_is_empty(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, InputScreen)

        await pilot.press("question_mark")
        await pilot.pause()

        assert isinstance(app.screen, HelpScreen)


async def test_typed_q_and_question_mark_are_not_intercepted_once_input_is_non_empty(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()

        # Real keystrokes (not insert_text_at_cursor, which bypasses
        # _CommandInput._on_key entirely) -- the leading "gepa" keystrokes
        # make the field non-empty before the "q" and "question_mark" presses
        # that matter for this test.
        for key in "gepa":
            await pilot.press(key)
        await pilot.press("q")
        await pilot.press("question_mark")
        await pilot.pause()

        assert isinstance(app.screen, InputScreen)
        assert app.screen.query_one(Input).value == "gepaq?"
        assert not app._exit


async def test_non_gepa_prompt_opens_dashboard_chat(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, "summarize my last run")

        assert isinstance(app.screen, DashboardScreen)
        assert app.state is None
        assert app.screen.query_one(TabbedContent).active == "chat-tab"
        assert app.chat_messages[0] == ChatMessage(role="user", text="summarize my last run")
