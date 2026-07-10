"""Landing screen for bare `autumn`: a single command-line-style input.

Three behaviors, all decided from one submitted line of text:

- Empty Enter -> browse mode (`AutumnApp.enter_browse_mode`), same as bare
  `autumn` used to land on directly before this screen existed.
- `gepa <script> [--dry-run] [--name ...] [--run-dir ...]` -> parsed via
  `autumn.cli.parse_command_line` (shared with `DashboardScreen`'s CommandBar,
  so the two surfaces can't drift) and handed to `AutumnApp.launch_gepa_run`
  to launch identically to `autumn run <script>`.
- Anything else non-empty -> a shared Autumn chat prompt, shown on the
  dashboard and answered by the available local/fallback model path.

Bad `gepa ...` syntax (unknown flags, missing/nonexistent script) shows an
error notification and leaves the user on this screen rather than crashing or
navigating away.
"""

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Input, Label

from autumn.cli import LaunchSpecError, parse_command_line
from autumn.widgets.deer_sprite import DeerSprite

# Keys that should act as the app-level shortcuts advertised in the landing
# screen's footer, but only while the field is still empty -- keyed by the
# character Input._on_key reports, mapped to the AutumnApp action to call.
_EMPTY_INPUT_SHORTCUTS = {"q": "action_request_quit", "?": "action_toggle_help"}


class _CommandInput(Input):
    """Input variant used only by InputScreen.

    `Input._on_key` unconditionally consumes and stops every printable key
    before it can reach any binding -- Textual's `check_consume_key`-based
    filtering strips even `priority=True` Screen/App bindings for a
    printable key out of the binding chain whenever an Input has focus (see
    `textual.screen.Screen._binding_chain`), so a plain `BINDINGS` entry for
    `q`/`question_mark` can never fire here the way it does on DashboardScreen
    (whose CommandBar Input stays unfocused until `:`). This subclass
    special-cases those two keys itself, before the base class's insertion
    logic runs, and only while the field is empty -- so a real `q` or `?`
    typed as part of a command or chat prompt (e.g. "what's my score?") is
    unaffected.
    """

    async def _on_key(self, event: events.Key) -> None:
        action = _EMPTY_INPUT_SHORTCUTS.get(event.character) if not self.value else None
        if action is not None:
            event.stop()
            event.prevent_default()
            getattr(self.app, action)()
            return
        await super()._on_key(event)


# Mirrors torlink's Splash (src/ui/views/Splash.tsx): centered borderless
# column -- big block-letter logo, dim descriptive line, input, dot-separated
# keybind footer in the accent-secondary "keybinding hints" shade (see
# palette.ACCENT_SECONDARY). Logo lines are a hardcoded 5-row block font
# (torlink's own LOGO_LINES in src/ui/logo.ts is likewise a literal, not a
# generic font engine -- only "autumn" needs rendering here).
_LOGO_LINES = (
    " ███  █   █ █████ █   █ █   █ █   █",
    "█   █ █   █   █   █   █ ██ ██ ██  █",
    "█████ █   █   █   █   █ █ █ █ █ █ █",
    "█   █ █   █   █   █   █ █   █ █  ██",
    "█   █  ███    █    ███  █   █ █   █",
)
_HINT_TEXT = "Ask Autumn anything..."
_FOOTER_HINT = "[#f0c17a]enter[/] browse   •   [#f0c17a]q[/] quit   •   [#f0c17a]?[/] help"


class InputScreen(Screen):
    """Landing screen shown for browse-only `AutumnApp` construction (bare `autumn`)."""

    DEFAULT_CSS = """
    InputScreen {
        align: center middle;
    }
    InputScreen #input-screen-frame {
        width: auto;
        height: auto;
        align: center middle;
    }
    InputScreen .app-title {
        width: 100%;
        height: auto;
        content-align: center middle;
        text-style: bold;
        color: #d98e4a;
        margin-bottom: 1;
    }
    InputScreen DeerSprite {
        width: 100%;
        height: auto;
        content-align: center middle;
        margin-bottom: 1;
    }
    InputScreen .input-hint {
        width: 100%;
        content-align: center middle;
        text-opacity: 55%;
        margin-bottom: 1;
    }
    InputScreen #command-input {
        width: 62;
    }
    InputScreen .app-footer-hint {
        width: 100%;
        content-align: center middle;
        text-opacity: 70%;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="input-screen-frame"):
            yield Label("\n".join(_LOGO_LINES), classes="app-title")
            yield DeerSprite()
            yield Label(_HINT_TEXT, classes="input-hint")
            yield _CommandInput(placeholder="Ask Autumn about your runs...", id="command-input")
            yield Label(_FOOTER_HINT, classes="app-footer-hint")

    def on_mount(self) -> None:
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
            self.app.open_chat_prompt(text)
            return

        self.app.launch_gepa_run(spec)
