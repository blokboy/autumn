"""Landing screen for bare `autumn`: a single command-line-style input.

Three behaviors, all decided from one submitted line of text:

- Empty Enter -> browse mode (`AutumnApp.enter_browse_mode`), same as bare
  `autumn` used to land on directly before this screen existed.
- `gepa <script> [--dry-run] [--name ...] [--run-dir ...]` -> parsed via
  `autumn.cli.parse_command_line` (shared with `DashboardScreen`'s CommandBar,
  so the two surfaces can't drift) and handed to `AutumnApp.launch_gepa_run`
  to launch identically to `autumn run <script>`.
- Anything else non-empty -> a stub "not implemented yet" notice. No
  LLM/agent invocation is wired up in this pass; this is just the landing
  spot for that future command language.

Bad `gepa ...` syntax (unknown flags, missing/nonexistent script) shows an
error notification and leaves the user on this screen rather than crashing or
navigating away.
"""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Input, Label

from autumn.cli import LaunchSpecError, parse_command_line


class InputScreen(Screen):
    """Landing screen shown for browse-only `AutumnApp` construction (bare `autumn`)."""

    DEFAULT_CSS = """
    InputScreen {
        align: center middle;
    }
    InputScreen Vertical {
        width: 80%;
        max-width: 100;
        height: auto;
        border: round #a78bfa;
        padding: 1 2;
    }
    InputScreen .input-hint {
        color: #6b6577;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(
                "Enter to browse runs, or `gepa <script.py> [--dry-run] [--name ...] "
                "[--run-dir ...]` to launch one.",
                classes="input-hint",
            )
            yield Input(placeholder="gepa my_script.py --dry-run", id="command-input")

    def on_mount(self) -> None:
        self.query_one(Vertical).border_title = "autumn"
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            self.app.enter_browse_mode()
            return

        try:
            spec = parse_command_line(text)
        except LaunchSpecError as exc:
            self.notify(str(exc), severity="error")
            return

        if spec is None:
            self.notify("Prompt execution not implemented yet", severity="warning")
            return

        self.app.launch_gepa_run(spec)
