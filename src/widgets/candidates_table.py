"""Candidates table: sortable list of GEPA candidates with Pareto/rejection styling."""

from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.widgets import DataTable

from models import CandidateRow, DashboardState
from screens.candidate_detail_screen import CandidateDetailScreen

_PARETO_COLOR = "#d98e4a"
_REJECTED_COLOR = "#c2a880"
_PARETO_MARKER = "●"  # filled circle, marks a Pareto-front member


def _format_parents(parent_ids: list[int | None]) -> str:
    return ", ".join("—" if parent_id is None else str(parent_id) for parent_id in parent_ids)


class CandidatesTable(DataTable):
    """DataTable listing candidates; call refresh_from_state() to rebuild rows."""

    BINDINGS = [
        Binding("s", "cycle_sort", "Sort"),
        Binding("v", "view_candidate", "View"),
        Binding("enter", "view_candidate", "View"),
        Binding("y", "copy_candidate", "Copy"),
        Binding("j", "cursor_down", "Cursor down", show=False),
        Binding("k", "cursor_up", "Cursor up", show=False),
    ]

    def __init__(self, state: DashboardState, *args, **kwargs) -> None:
        self.state = state
        self._sort_keys = ["idx", "val_score", "iteration"]
        self._sort_index = 0
        self._row_to_idx: list[int] = []
        super().__init__(*args, **kwargs)

    def on_mount(self) -> None:
        self.border_title = "Candidates"
        self.add_columns("Idx", "Val Score", "Iter Found", "Parents", "Front")
        self.refresh_from_state(self.state)

    def _sorted_rows(self) -> list[CandidateRow]:
        sort_key = self._sort_keys[self._sort_index]
        rows = list(self.state.candidates.values())
        if sort_key == "idx":
            rows.sort(key=lambda row: row.idx)
        elif sort_key == "val_score":
            rows.sort(key=lambda row: (row.val_score is None, -(row.val_score or 0)))
        elif sort_key == "iteration":
            rows.sort(
                key=lambda row: (
                    row.discovered_iteration is None,
                    row.discovered_iteration if row.discovered_iteration is not None else 0,
                )
            )
        return rows

    def refresh_from_state(self, state: DashboardState) -> None:
        self.state = state
        self.clear()
        self._row_to_idx = []

        for row in self._sorted_rows():
            val_score_text = "—" if row.val_score is None else f"{row.val_score:.4f}"
            iter_found_text = (
                "—" if row.discovered_iteration is None else str(row.discovered_iteration)
            )
            parents_text = _format_parents(row.parent_ids)

            # NOTE: DataTable has no per-row CSS selector, so Pareto/rejected
            # styling is applied here directly via Rich Text/Style objects
            # rather than through styles/autumn.tcss like the rest of the app.
            if row.is_pareto_member:
                style = Style(color=_PARETO_COLOR, bold=True)
                idx_cell = Text(str(row.idx), style=style)
                val_score_cell = Text(val_score_text, style=style)
                iter_found_cell = Text(iter_found_text, style=style)
                parents_cell = Text(parents_text, style=style)
                front_cell = Text(_PARETO_MARKER, style=style)
            elif row.was_rejected:
                style = Style(color=_REJECTED_COLOR, dim=True)
                idx_cell = Text(str(row.idx), style=style)
                val_score_cell = Text(val_score_text, style=style)
                iter_found_cell = Text(iter_found_text, style=style)
                parents_cell = Text(parents_text, style=style)
                front_cell = Text("", style=style)
            else:
                idx_cell = row.idx
                val_score_cell = val_score_text
                iter_found_cell = iter_found_text
                parents_cell = parents_text
                front_cell = ""

            self.add_row(idx_cell, val_score_cell, iter_found_cell, parents_cell, front_cell)
            self._row_to_idx.append(row.idx)

    def cycle_sort(self) -> None:
        """Advance to the next sort key and re-render rows in that order."""
        self._sort_index = (self._sort_index + 1) % len(self._sort_keys)
        self.border_subtitle = f"sort: {self._sort_keys[self._sort_index]}"
        self.refresh_from_state(self.state)

    def current_candidate_idx(self) -> int | None:
        """Return the candidate idx backing the currently highlighted row, if any."""
        row_index = self.cursor_row
        if 0 <= row_index < len(self._row_to_idx):
            return self._row_to_idx[row_index]
        return None

    def action_cycle_sort(self) -> None:
        self.cycle_sort()

    def action_view_candidate(self) -> None:
        idx = self.current_candidate_idx()
        if idx is None:
            return
        row = self.state.candidates.get(idx)
        if row is None:
            return
        self.app.push_screen(CandidateDetailScreen(row))

    def action_copy_candidate(self) -> None:
        idx = self.current_candidate_idx()
        if idx is None:
            return
        row = self.state.candidates.get(idx)
        if row is None or row.text is None:
            return
        flattened = "\n\n".join(f"{name}:\n{text}" for name, text in row.text.items())
        self.app.copy_to_clipboard(flattened)
