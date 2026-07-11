"""Tests for ConfirmScreen: push/dismiss result semantics via Textual's Pilot
test harness (confirm button, cancel button, y/n keys, and escape)."""

from textual.app import App, ComposeResult
from textual.widgets import Label

from screens.confirm_screen import ConfirmScreen


class ConfirmHostApp(App):
    """Trivial host app: records the bool result of the last ConfirmScreen
    it pushed, so tests can assert on it after simulating input."""

    def compose(self) -> ComposeResult:
        yield Label("host")

    def __init__(self) -> None:
        super().__init__()
        self.result: bool | None = None

    def ask(self, **kwargs) -> None:
        self.result = None
        self.push_screen(ConfirmScreen("Are you sure?", **kwargs), self._on_result)

    def _on_result(self, confirmed: bool) -> None:
        self.result = confirmed


async def test_confirm_button_dismisses_true():
    app = ConfirmHostApp()
    async with app.run_test() as pilot:
        app.ask()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)

        await pilot.click("#confirm")
        await pilot.pause()

        assert app.result is True
        assert not isinstance(app.screen, ConfirmScreen)


async def test_cancel_button_dismisses_false():
    app = ConfirmHostApp()
    async with app.run_test() as pilot:
        app.ask()
        await pilot.pause()

        await pilot.click("#cancel")
        await pilot.pause()

        assert app.result is False


async def test_escape_key_dismisses_false():
    app = ConfirmHostApp()
    async with app.run_test() as pilot:
        app.ask()
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        assert app.result is False


async def test_y_key_dismisses_true():
    app = ConfirmHostApp()
    async with app.run_test() as pilot:
        app.ask()
        await pilot.pause()

        await pilot.press("y")
        await pilot.pause()

        assert app.result is True


async def test_n_key_dismisses_false():
    app = ConfirmHostApp()
    async with app.run_test() as pilot:
        app.ask()
        await pilot.pause()

        await pilot.press("n")
        await pilot.pause()

        assert app.result is False


async def test_confirm_button_disabled_during_initial_delay():
    app = ConfirmHostApp()
    async with app.run_test() as pilot:
        app.ask(initial_delay=5.0)
        await pilot.pause()

        assert app.screen.query_one("#confirm").disabled is True

        await pilot.click("#confirm")
        await pilot.pause()
        assert app.result is None, "clicking a disabled button must not confirm"
        assert isinstance(app.screen, ConfirmScreen)


async def test_confirm_button_enables_after_initial_delay():
    app = ConfirmHostApp()
    async with app.run_test() as pilot:
        app.ask(initial_delay=0.05)
        await pilot.pause()
        await pilot.pause(0.1)

        assert app.screen.query_one("#confirm").disabled is False

        await pilot.click("#confirm")
        await pilot.pause()
        assert app.result is True


async def test_y_key_does_nothing_during_initial_delay():
    app = ConfirmHostApp()
    async with app.run_test() as pilot:
        app.ask(initial_delay=5.0)
        await pilot.pause()

        await pilot.press("y")
        await pilot.pause()

        assert app.result is None
        assert isinstance(app.screen, ConfirmScreen)


async def test_cancel_and_escape_are_never_delayed():
    app = ConfirmHostApp()
    async with app.run_test() as pilot:
        app.ask(initial_delay=5.0)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        assert app.result is False


async def test_no_delay_by_default_confirm_button_is_immediately_enabled():
    app = ConfirmHostApp()
    async with app.run_test() as pilot:
        app.ask()
        await pilot.pause()

        assert app.screen.query_one("#confirm").disabled is False


async def test_custom_labels_and_title_are_used():
    app = ConfirmHostApp()
    async with app.run_test() as pilot:
        app.ask(title="Quit?", confirm_label="Quit anyway", cancel_label="Stay")
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ConfirmScreen)
        assert screen.query_one("#confirm").label == "Quit anyway"
        assert screen.query_one("#cancel").label == "Stay"

        await pilot.click("#confirm")
        await pilot.pause()
        assert app.result is True
