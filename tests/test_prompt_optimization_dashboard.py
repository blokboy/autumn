"""Behavioral tests for #49 -- the dedicated prompt optimization result
surface on DashboardScreen: run-kind-based tab visibility, live vs.
historical result loading, and missing-artifact messaging."""

import json
from pathlib import Path

from textual.app import App
from textual.widgets import TabbedContent

from models import DashboardState, RunKind, RunStatus
from prompt_optimization_contracts import BEST_RESULT_FILENAME
from screens.dashboard_screen import DashboardScreen
from widgets.best_result_pane import BestResultPane


class _DashboardTestApp(App):
    """Mounts DashboardScreen directly, without AutumnApp's full launch/input
    wiring -- #49 only needs a populated runs_root and/or a live DashboardState,
    both of which DashboardScreen already accepts standalone."""

    def __init__(self, **screen_kwargs) -> None:
        super().__init__()
        self._screen_kwargs = screen_kwargs

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen(**self._screen_kwargs))


def _write_script_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_log.json").write_text(json.dumps([{"event": "on_optimization_end"}]))
    (run_dir / "autumn_meta.json").write_text(json.dumps({"run_kind": "script", "run_name": run_dir.name}))


def _write_prompt_optimization_run(
    run_dir: Path, *, status_marker: str | None = "completed", with_artifacts: bool = True
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_log.json").write_text(json.dumps([{"event": "on_optimization_end"}]))
    (run_dir / "autumn_meta.json").write_text(
        json.dumps({"run_kind": "prompt_optimization", "run_name": run_dir.name})
    )
    if status_marker is not None:
        (run_dir / "autumn_status.json").write_text(json.dumps({"status": status_marker}))
    if with_artifacts:
        (run_dir / BEST_RESULT_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate_json": "autumn_best_candidate.json",
                    "prompt_markdown": "autumn_best_prompt.md",
                    "best_prompt": {
                        "prompt": "Classify the ticket by priority.",
                        "system_prompt": "Be terse.",
                        "score": 0.875,
                        "candidate_idx": 2,
                    },
                }
            )
        )


async def test_script_run_hides_best_result_tab(tmp_path):
    _write_script_run(tmp_path / "script-run")
    app = _DashboardTestApp(runs_root=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, DashboardScreen)
        tabbed_content = screen.query_one(TabbedContent)
        tab = tabbed_content.get_tab("best-result-tab")
        assert tab.display is False


async def test_historical_prompt_optimization_run_shows_best_result(tmp_path):
    run_dir = tmp_path / "po-run"
    _write_prompt_optimization_run(run_dir, with_artifacts=True)
    app = _DashboardTestApp(runs_root=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        tabbed_content = screen.query_one(TabbedContent)
        tab = tabbed_content.get_tab("best-result-tab")
        assert tab.display is True

        pane = screen.query_one("#best-result", BestResultPane)
        assert "Classify the ticket by priority." in pane.query_one("#br-prompt").content
        assert "Be terse." in pane.query_one("#br-system-prompt").content
        assert "0.8750" in pane.query_one("#br-score").content


async def test_prompt_optimization_run_without_artifacts_shows_failed_message(tmp_path):
    run_dir = tmp_path / "po-run-failed"
    _write_prompt_optimization_run(run_dir, status_marker="failed", with_artifacts=False)
    app = _DashboardTestApp(runs_root=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        tabbed_content = screen.query_one(TabbedContent)
        assert tabbed_content.get_tab("best-result-tab").display is True

        pane = screen.query_one("#best-result", BestResultPane)
        status_text = str(pane.query_one("#br-status").content)
        assert "run failed" in status_text


async def test_live_prompt_optimization_run_shows_in_progress_before_artifacts(tmp_path):
    run_dir = tmp_path / "po-run-live"
    live_state = DashboardState(
        run_name="po-run-live",
        run_dir=run_dir,
        run_kind=RunKind.PROMPT_OPTIMIZATION,
        status=RunStatus.RUNNING,
    )
    app = _DashboardTestApp(runs_root=tmp_path, live_state=live_state)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        tabbed_content = screen.query_one(TabbedContent)
        assert tabbed_content.get_tab("best-result-tab").display is True

        pane = screen.query_one("#best-result", BestResultPane)
        status_text = str(pane.query_one("#br-status").content)
        assert "in progress" in status_text


async def test_switching_between_script_and_prompt_optimization_runs_toggles_tab(tmp_path):
    script_dir = tmp_path / "script-run"
    po_dir = tmp_path / "po-run"
    _write_script_run(script_dir)
    _write_prompt_optimization_run(po_dir, with_artifacts=True)
    app = _DashboardTestApp(runs_root=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        tabbed_content = screen.query_one(TabbedContent)

        # Newest-first sort means whichever was written last loads first;
        # explicitly select each run_dir to avoid depending on mtime ordering.
        screen._select_run(po_dir)
        await pilot.pause()
        assert tabbed_content.get_tab("best-result-tab").display is True

        screen._select_run(script_dir)
        await pilot.pause()
        assert tabbed_content.get_tab("best-result-tab").display is False

        screen._select_run(po_dir)
        await pilot.pause()
        assert tabbed_content.get_tab("best-result-tab").display is True
