"""Dashboard settings controls backed by the shared config API."""

from textual.widgets import Static, Switch, TabbedContent

import config
from app import AutumnApp


async def test_settings_tab_reads_initial_config_modes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "get_search_mode", lambda: "autonomous")
    monkeypatch.setattr(config, "get_action_mode", lambda: "autonomous")
    app = AutumnApp(runs_root=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one(TabbedContent).active = "settings-tab"
        await pilot.pause()

        assert app.screen.query_one("#search-mode-switch", Switch).value is True
        assert app.screen.query_one("#action-mode-switch", Switch).value is True
        assert str(app.screen.query_one("#search-mode-value", Static).content) == "autonomous"
        assert str(app.screen.query_one("#action-mode-value", Static).content) == "autonomous"


async def test_settings_tab_persists_search_mode_changes(tmp_path, monkeypatch):
    search_mode = "explicit"
    calls: list[tuple[str, str]] = []

    def set_search_mode(value: str) -> None:
        nonlocal search_mode
        calls.append(("search", value))
        search_mode = value

    monkeypatch.setattr(config, "get_search_mode", lambda: search_mode)
    monkeypatch.setattr(config, "set_search_mode", set_search_mode)
    monkeypatch.setattr(config, "get_action_mode", lambda: "confirm")
    app = AutumnApp(runs_root=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one(TabbedContent).active = "settings-tab"
        await pilot.pause()
        app.screen.query_one("#search-mode-switch", Switch).toggle()
        await pilot.pause()

        assert search_mode == "autonomous"
        assert ("search", "autonomous") in calls
        assert str(app.screen.query_one("#search-mode-value", Static).content) == "autonomous"


async def test_settings_tab_persists_action_mode_changes(tmp_path, monkeypatch):
    action_mode = "confirm"
    calls: list[tuple[str, str]] = []

    def set_action_mode(value: str) -> None:
        nonlocal action_mode
        calls.append(("action", value))
        action_mode = value

    monkeypatch.setattr(config, "get_search_mode", lambda: "explicit")
    monkeypatch.setattr(config, "get_action_mode", lambda: action_mode)
    monkeypatch.setattr(config, "set_action_mode", set_action_mode)
    app = AutumnApp(runs_root=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one(TabbedContent).active = "settings-tab"
        await pilot.pause()
        app.screen.query_one("#action-mode-switch", Switch).toggle()
        await pilot.pause()

        assert action_mode == "autonomous"
        assert ("action", "autonomous") in calls
        assert str(app.screen.query_one("#action-mode-value", Static).content) == "autonomous"
