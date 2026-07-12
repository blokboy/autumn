"""Best-result pane: the dedicated result surface for prompt optimization
runs (#49) -- shows the best optimized prompt, optional system prompt, score,
and artifact status. DashboardScreen only shows this pane's tab for
`RunKind.PROMPT_OPTIMIZATION` runs (see its `_sync_best_result_tab`); script
runs never see it.

Result data always comes from `registry.load_best_result_artifacts`, reading
`autumn_best_result.json` straight off disk -- the same file whether the run
is still live or long finished, since `prompt_optimization_runtime.py`
writes it before the run is ever marked finished. `state` itself is only
consulted for status/error messaging while no artifacts exist yet.
"""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label

import registry
from models import DashboardState, RunStatus


class BestResultPane(Vertical):
    """Renders the best-result artifacts for a prompt optimization DashboardState.
    Call refresh_from_state() to update."""

    DEFAULT_CSS = """
    BestResultPane {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
    }
    BestResultPane Label {
        width: 1fr;
        margin-bottom: 1;
    }
    BestResultPane #br-system-prompt-label,
    BestResultPane #br-system-prompt {
        display: none;
    }
    """

    def __init__(self, state: DashboardState, *args, **kwargs) -> None:
        self._state = state
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Label("", id="br-status")
        yield Label("System prompt:", id="br-system-prompt-label", classes="field-label")
        yield Label("", id="br-system-prompt")
        yield Label("Prompt:", id="br-prompt-label", classes="field-label")
        yield Label("", id="br-prompt")
        yield Label("", id="br-score")
        yield Label("", id="br-artifacts")

    def on_mount(self) -> None:
        self.border_title = "Best Result"
        self.refresh_from_state(self._state)

    def refresh_from_state(self, state: DashboardState) -> None:
        self._state = state
        artifacts = registry.load_best_result_artifacts(state.run_dir)

        system_prompt_label = self.query_one("#br-system-prompt-label", Label)
        system_prompt_value = self.query_one("#br-system-prompt", Label)
        prompt_value = self.query_one("#br-prompt", Label)
        score_label = self.query_one("#br-score", Label)
        artifacts_label = self.query_one("#br-artifacts", Label)
        status_label = self.query_one("#br-status", Label)

        if artifacts is None:
            status_label.update(self._not_available_message(state))
            prompt_value.update("")
            system_prompt_label.display = False
            system_prompt_value.display = False
            score_label.update("")
            artifacts_label.update("")
            return

        status_label.update("Status: best prompt available")
        prompt_value.update(artifacts.best_prompt.prompt)

        if artifacts.best_prompt.system_prompt:
            system_prompt_label.display = True
            system_prompt_value.display = True
            system_prompt_value.update(artifacts.best_prompt.system_prompt)
        else:
            system_prompt_label.display = False
            system_prompt_value.display = False
            system_prompt_value.update("")

        score = artifacts.best_prompt.score
        score_label.update(f"Score: {score:.4f}" if score is not None else "Score: ?")

        artifacts_label.update(
            f"Artifacts: {artifacts.candidate_json}, {artifacts.prompt_markdown} "
            f"(in {state.run_dir})"
        )

    def _not_available_message(self, state: DashboardState) -> str:
        if state.status == RunStatus.RUNNING:
            return "Status: optimization in progress — best prompt not yet available."
        if state.status == RunStatus.FAILED:
            suffix = f" ({state.error})" if state.error else ""
            return f"Status: run failed before a best prompt was written.{suffix}"
        if state.status == RunStatus.STOPPED:
            return "Status: run was stopped before a best prompt was written."
        return "Status: no result artifacts found for this run."
