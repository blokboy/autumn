"""Dashboard layout smoke tests."""

from textual.app import App

from screens.dashboard_screen import DashboardScreen
from widgets.run_sidebar import RunSidebar


class _DashboardLayoutTestApp(App):
    def __init__(self, **screen_kwargs) -> None:
        super().__init__()
        self._screen_kwargs = screen_kwargs

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen(**self._screen_kwargs))


async def test_left_sidebar_is_split_into_agents_and_jobs_panels(tmp_path):
    app = _DashboardLayoutTestApp(runs_root=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen

        assert screen.query_one("#left-sidebar").styles.width.value == 32
        assert screen.query_one("#agents-panel").border_title == "Agents"
        assert screen.query_one(RunSidebar).border_title == "Jobs"
