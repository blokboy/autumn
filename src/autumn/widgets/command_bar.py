"""Persistent bottom command bar for DashboardScreen.

Unfocused by default so DashboardScreen's existing single-key bindings
(q/Q/r/j/k/?/arrows) keep working exactly as before -- Textual delivers key
presses to the focused widget first, so as long as nothing has focused the
bar's Input, those keys never reach it and bubble straight up to the screen's
own bindings. `:` (bound on DashboardScreen) calls `focus_input()` to enter
"command mode"; from then on keystrokes go to the Input like any text field,
which is the whole point -- typing `gepa myscript.py` shouldn't also fire `q`
for every letter that happens to collide with a binding.

Submitting a line hands the raw text to `AutumnApp.submit_command`, which owns
the launch-vs-queue decision for `gepa ...` commands and the shared chat
prompt flow for everything else -- this widget only renders input and an
ordered preview of whatever queue AutumnApp currently holds, via
`refresh_queue`.
"""

from autumn.cli import LaunchSpec
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Static


def _describe(item) -> str:
    if isinstance(item, LaunchSpec):
        return f"gepa {item.script_path.name}" + (" --dry-run" if item.dry_run else "")
    return item


class CommandBar(Vertical):
    """Docked-bottom composite widget: a queue preview line above the input."""

    DEFAULT_CSS = """
    CommandBar {
        dock: bottom;
        height: auto;
        border-top: round #a9906f;
    }
    CommandBar:focus-within {
        border-top: round #d98e4a;
    }
    CommandBar #queue-preview {
        color: #c2a880;
        text-opacity: 55%;
        padding: 0 1;
        display: none;
    }
    CommandBar Input {
        border: none;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="queue-preview")
        yield Input(
            placeholder=": ask Autumn a question, gepa my_script.py, or gepa --run-dir examples/",
            id="command-bar-input",
        )

    def focus_input(self) -> None:
        self.query_one("#command-bar-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.input.value = ""
        self.app.submit_command(event.value)

    def refresh_queue(self, items: list) -> None:
        preview = self.query_one("#queue-preview", Static)
        if not items:
            preview.update("")
            preview.display = False
            return
        preview.update(" | ".join(f"{i}. {_describe(item)}" for i, item in enumerate(items, start=1)))
        preview.display = True
