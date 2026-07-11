"""Modal screen showing full metadata and prompt text for a single candidate."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, Static

from models import CandidateRow


class CandidateDetailScreen(ModalScreen):
    """Displays a candidate's score/lineage metadata plus its full prompt text, if captured."""

    DEFAULT_CSS = """
    CandidateDetailScreen {
        align: center middle;
    }
    CandidateDetailScreen > VerticalScroll {
        width: 90%;
        height: 90%;
        border: round #d98e4a;
        padding: 1 2;
    }
    CandidateDetailScreen Label.component-name {
        text-style: bold;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_modal", "Close"),
        Binding("v", "dismiss_modal", "Close"),
        Binding("y", "copy", "Copy text"),
    ]

    def __init__(self, candidate_row: CandidateRow, *args, **kwargs) -> None:
        self.candidate_row = candidate_row
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        row = self.candidate_row
        score_text = f"{row.val_score:.4f}" if row.val_score is not None else "—"
        iteration_text = (
            str(row.discovered_iteration) if row.discovered_iteration is not None else "—"
        )
        parents_text = ", ".join(
            str(parent_id) if parent_id is not None else "—" for parent_id in row.parent_ids
        ) or "—"
        pareto_text = "yes" if row.is_pareto_member else "no"

        with VerticalScroll():
            yield Static(
                f"Score: {score_text}    Discovered: iteration {iteration_text}    "
                f"Parents: {parents_text}    Pareto front member: {pareto_text}",
                id="cd-summary",
            )
            if row.text is None:
                yield Static("No candidate text captured yet for this entry.", id="cd-empty")
            else:
                for component_name, prompt_text in row.text.items():
                    yield Label(component_name, classes="component-name")
                    yield Static(prompt_text)

    def on_mount(self) -> None:
        self.query_one(VerticalScroll).border_title = f"Candidate #{self.candidate_row.idx}"

    def action_dismiss_modal(self) -> None:
        self.dismiss()

    def action_copy(self) -> None:
        if self.candidate_row.text is None:
            return
        flattened = "\n\n".join(
            f"{component_name}:\n{prompt_text}"
            for component_name, prompt_text in self.candidate_row.text.items()
        )
        self.app.copy_to_clipboard(flattened)
