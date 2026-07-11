"""Dashboard settings controls backed by Autumn's shared config module."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Switch

import config


def _mode_label(mode: str) -> str:
    return config.ACTION_MODE_AUTONOMOUS if mode == config.ACTION_MODE_AUTONOMOUS else mode


class SettingsView(Vertical):
    """Small operational settings surface for shared CLI/dashboard state."""

    DEFAULT_CSS = """
    SettingsView {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
    }
    SettingsView .settings-row {
        height: auto;
        margin-bottom: 1;
    }
    SettingsView .settings-label {
        width: 20;
    }
    SettingsView .settings-value {
        width: 14;
        margin-left: 2;
    }
    SettingsView Switch {
        width: 10;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        self._search_mode = config.get_search_mode()
        self._action_mode = config.get_action_mode()
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Static("Agent behavior", id="settings-heading")
        with Horizontal(classes="settings-row"):
            yield Static("Search web", classes="settings-label")
            yield Switch(
                value=self._search_mode == config.SEARCH_MODE_AUTONOMOUS,
                id="search-mode-switch",
                tooltip="Allow Autumn to search the web without an explicit /search command.",
            )
            yield Static(_mode_label(self._search_mode), id="search-mode-value", classes="settings-value")
        with Horizontal(classes="settings-row"):
            yield Static("System actions", classes="settings-label")
            yield Switch(
                value=self._action_mode == config.ACTION_MODE_AUTONOMOUS,
                id="action-mode-switch",
                tooltip="Allow Autumn to run system actions without confirmation prompts.",
            )
            yield Static(_mode_label(self._action_mode), id="action-mode-value", classes="settings-value")

    def on_mount(self) -> None:
        self.border_title = "Settings"

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "search-mode-switch":
            value = config.SEARCH_MODE_AUTONOMOUS if event.value else config.SEARCH_MODE_EXPLICIT
            config.set_search_mode(value)
            self.query_one("#search-mode-value", Static).update(_mode_label(value))
            return
        if event.switch.id == "action-mode-switch":
            value = config.ACTION_MODE_AUTONOMOUS if event.value else config.ACTION_MODE_CONFIRM
            config.set_action_mode(value)
            self.query_one("#action-mode-value", Static).update(_mode_label(value))
