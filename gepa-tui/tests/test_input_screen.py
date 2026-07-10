"""Behavioral tests for InputScreen: the bare-`autumn` landing screen and its
three submit paths (empty Enter -> browse, `gepa ...` -> live launch, anything
else -> stub notice), via Textual's Pilot harness against a real AutumnApp
constructed the same way cli.py's `_browse()` does (no run_name/run_dir).
"""

import asyncio

from textual.widgets import Input

from autumn.app import AutumnApp
from autumn.models import RunStatus
from autumn.screens.dashboard_screen import DashboardScreen
from autumn.screens.input_screen import InputScreen


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


async def test_bare_autumn_opens_input_screen_first(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, InputScreen)
        assert app.state is None


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


async def test_non_gepa_prompt_shows_stub_notice_and_stays(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, "summarize my last run")

        assert isinstance(app.screen, InputScreen)
        assert app.state is None
        notifications = list(app._notifications)
        assert len(notifications) == 1
        assert notifications[0].severity == "warning"
        assert "not implemented" in notifications[0].message
