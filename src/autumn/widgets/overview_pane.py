"""Overview pane: at-a-glance status, progress, and best-candidate summary."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, ProgressBar

from autumn.models import DashboardState


class OverviewPane(Vertical):
    """Renders a snapshot of a DashboardState. Call refresh_from_state() to update."""

    DEFAULT_CSS = """
    OverviewPane {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
    }
    OverviewPane Label {
        width: 1fr;
        margin-bottom: 1;
    }
    OverviewPane #ov-error {
        display: none;
    }
    OverviewPane #ov-multi-objective-notice {
        display: none;
    }
    """

    def __init__(self, state: DashboardState, *args, **kwargs) -> None:
        self._state = state
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        # Multi-objective notice sits at the top so it's the first thing seen,
        # but the rest of the (objective-agnostic) fields below still render
        # normally underneath it -- V1 is single-objective only, so we show a
        # notice instead of building multi-objective UI, not instead of the
        # existing summary fields.
        yield Label("", id="ov-multi-objective-notice", classes="status-warning")
        yield Label("", id="ov-status")
        yield Label("", id="ov-iteration")
        yield Label("", id="ov-budget-label")
        yield ProgressBar(id="ov-budget-bar", show_eta=False)
        yield Label("", id="ov-best")
        yield Label("", id="ov-pareto")
        yield Label("", id="ov-error", classes="status-error")

    def on_mount(self) -> None:
        self.border_title = "Overview"
        self.refresh_from_state(self._state)

    def refresh_from_state(self, state: DashboardState) -> None:
        self._state = state

        notice_label = self.query_one("#ov-multi-objective-notice", Label)
        if state.is_multi_objective:
            notice_label.update(
                "Multi-objective runs are not yet supported — showing limited info."
            )
            notice_label.display = True
        else:
            notice_label.update("")
            notice_label.display = False

        self.query_one("#ov-status", Label).update(f"Status: {state.status.value}")

        total_iterations = (
            state.total_iterations if state.total_iterations is not None else "?"
        )
        self.query_one("#ov-iteration", Label).update(
            f"Iteration: {state.current_iteration} / {total_iterations}"
        )

        budget_label = self.query_one("#ov-budget-label", Label)
        bar = self.query_one("#ov-budget-bar", ProgressBar)
        if state.max_metric_calls is None:
            budget_label.update(f"Metric calls used: {state.metric_calls_used} (unbounded)")
            bar.display = False
        else:
            budget_label.update(
                f"Metric calls: {state.metric_calls_used} / {state.max_metric_calls}"
            )
            bar.display = True
            bar.update(total=state.max_metric_calls, progress=state.metric_calls_used)

        best_label = self.query_one("#ov-best", Label)
        if state.best_idx is None:
            best_label.update("Best candidate: --")
        else:
            score_text = (
                f"{state.best_score:.4f}" if state.best_score is not None else "?"
            )
            best_label.update(f"Best candidate: #{state.best_idx} (score={score_text})")

        self.query_one("#ov-pareto", Label).update(
            f"Pareto front size: {len(state.pareto_front)}"
        )

        error_label = self.query_one("#ov-error", Label)
        if state.error:
            error_label.update(f"Error: {state.error}")
            error_label.display = True
        else:
            error_label.update("")
            error_label.display = False
