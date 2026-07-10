"""Modal help screen: a static reference for all dashboard keybindings."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label


class HelpScreen(ModalScreen):
    """Read-only overlay listing every keybinding grouped by area."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("question_mark", "dismiss", "Close"),
    ]

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    HelpScreen VerticalScroll {
        width: auto;
        height: auto;
        max-width: 70%;
        max-height: 80%;
        padding: 1 2;
        border: round #a78bfa;
        background: #0a0810;
    }
    HelpScreen .help-heading {
        text-style: bold;
        margin-top: 1;
    }
    HelpScreen .help-heading:first-of-type {
        margin-top: 0;
    }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label("Navigation", classes="help-heading")
            yield Label("↑ / ↓ / j / k    Move selection (sidebar and Candidates table)")
            yield Label("Tab    Switch between Overview / Candidates / Log tabs")

            yield Label("Candidates", classes="help-heading")
            yield Label("s    Cycle sort order (idx / val score / iteration)")
            yield Label("v / Enter    View full candidate detail")
            yield Label("y    Copy candidate text to clipboard")

            yield Label("Run control", classes="help-heading")
            yield Label("Q (shift+q)    Send graceful stop signal to the current run")
            yield Label("r    Resume a stopped/failed run")
            yield Label(":    Focus the command bar (launch or queue a gepa run)")
            yield Label("q    Quit (confirms first if a run is active)")

            yield Label("General", classes="help-heading")
            yield Label("?    Toggle this help screen")

    def on_mount(self) -> None:
        self.query_one(VerticalScroll).border_title = "Help"
