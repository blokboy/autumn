"""Generic reusable yes/no confirmation modal.

``ConfirmScreen`` is a ``ModalScreen[bool]``: push it with
``app.push_screen(ConfirmScreen(...), callback)`` (Textual's standard
modal-with-result pattern) and the ``callback`` will be invoked with ``True``
if the user confirmed, or ``False`` if they cancelled -- via the Cancel
button, the Escape key, or the "n" key.

Constructor parameters:
    message: str
        The question/warning text shown in the modal body. Required.
    title: str
        Border title for the modal frame. Defaults to "Confirm".
    confirm_label: str
        Label for the affirmative button (e.g. "Confirm", "Quit anyway").
        Defaults to "Confirm".
    cancel_label: str
        Label for the negative/safe-default button. Defaults to "Cancel".

Result semantics:
    The screen dismisses with a ``bool``: ``True`` when the confirm button
    (or "y") is pressed, ``False`` when the cancel button, "n", or Escape is
    pressed. Cancel is always the safe default for Escape, matching the
    convention used elsewhere in this app (see HelpScreen).

Example:
    def _on_result(confirmed: bool) -> None:
        if confirmed:
            ...

    self.app.push_screen(
        ConfirmScreen(
            "Send graceful stop signal to this run?",
            confirm_label="Stop",
        ),
        _on_result,
    )
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ConfirmScreen(ModalScreen[bool]):
    """A compact modal asking the user to confirm or cancel an action."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("n", "cancel", "Cancel"),
        ("y", "confirm", "Confirm"),
    ]

    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    ConfirmScreen Vertical {
        width: auto;
        height: auto;
        max-width: 70%;
        padding: 1 2;
        border: round #a78bfa;
        background: #0a0810;
    }
    ConfirmScreen .confirm-message {
        margin-bottom: 1;
        max-width: 60;
    }
    ConfirmScreen Horizontal {
        width: auto;
        height: auto;
        align: center middle;
    }
    ConfirmScreen Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        message: str,
        *,
        title: str = "Confirm",
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
    ) -> None:
        super().__init__()
        self.message = message
        self.confirm_title = title
        self.confirm_label = confirm_label
        self.cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.message, classes="confirm-message")
            with Horizontal():
                yield Button(self.confirm_label, variant="error", id="confirm")
                yield Button(self.cancel_label, variant="primary", id="cancel")

    def on_mount(self) -> None:
        self.query_one(Vertical).border_title = self.confirm_title
        self.query_one("#cancel", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
