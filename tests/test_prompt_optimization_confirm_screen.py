"""Behavioral tests for PromptOptimizationConfirmScreen (#47): the editable
final review screen `gepa optimize ...` opens instead of launching directly.

Covers prefilled fields, required-field validation blocking Launch, the
hosted-provider data-flow warning, Launch/Cancel behavior (including
queueing behind an already-live run), and eval-asset/metric selection
(single-installed-eval defaulting, built-in vs custom-local metric
visibility).
"""

import json
from pathlib import Path

import local_models
from app import AutumnApp
from models import (
    PromptRoutingPolicy,
    ProviderAccount,
    ProviderModel,
    RunKind,
    RunStatus,
)
from prompt_optimization_contracts import (
    BuiltInMetricRef,
    EvalAssetRef,
    ModelIdentity,
    PromptOptimizationSpecBudget,
)
from prompt_optimization_drafts import PromptOptimizationDraft
from screens.dashboard_screen import DashboardScreen
from screens.prompt_optimization_confirm_screen import PromptOptimizationConfirmScreen
from textual.widgets import Input, Select, Static


def _model_file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.gguf"
    path.write_bytes(b"fake model")
    return path


def _catalog_with_local_model(tmp_path: Path, name: str) -> Path:
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name=name, source_path=_model_file(tmp_path, name))
    return catalog_root


def _install_fake_eval_asset(
    assets_root: Path,
    *,
    asset_id: str,
    version: str = "2026.07.12",
    supported_metrics: tuple[str, ...] = ("exact_match",),
    default_metric: str = "exact_match",
) -> None:
    directory = assets_root / asset_id / version
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "autumn_eval_asset.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "asset_id": asset_id,
                "version": version,
                "label": asset_id,
                "description": "fixture eval asset",
                "supported_metrics": list(supported_metrics),
                "default_metric": default_metric,
                "size_bytes": 1,
                "license": "MIT",
                "provenance": "fixture",
                "sha256": "0" * 64,
            }
        )
    )


def _make_app(tmp_path: Path, *, catalog_root: Path, assets_root: Path, policy=None) -> AutumnApp:
    return AutumnApp(
        runs_root=tmp_path / "runs",
        model_catalog_root=catalog_root,
        eval_assets_root=assets_root,
        prompt_routing_policy=policy,
    )


async def _mount_screen(pilot, app, screen) -> None:
    app.push_screen(screen)
    await pilot.pause()


def _valid_draft(*, task_model: ModelIdentity, run_name="a-run") -> PromptOptimizationDraft:
    return PromptOptimizationDraft(
        raw_text="gepa optimize ...",
        prompt="Classify tickets.",
        system_prompt="You are a careful assistant.",
        task_model=task_model,
        optimizer_model=task_model,
        eval_asset=None,
        metric=None,
        run_name=run_name,
        budget=PromptOptimizationSpecBudget(max_metric_calls=10),
    )


async def test_prefills_fields_from_draft(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    assets_root = tmp_path / "eval-assets"
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval")
    identity = ModelIdentity(name="tiny", backend="llama.cpp")

    draft = PromptOptimizationDraft(
        raw_text="gepa optimize ...",
        prompt="Classify support tickets.",
        system_prompt="Be concise.",
        task_model=identity,
        optimizer_model=identity,
        eval_asset=EvalAssetRef(asset_id="tiny-eval", version="2026.07.12"),
        metric=BuiltInMetricRef(metric_id="exact_match"),
        run_name="priority-run",
        budget=PromptOptimizationSpecBudget(max_metric_calls=25),
    )

    app = _make_app(tmp_path, catalog_root=catalog_root, assets_root=assets_root)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        screen = PromptOptimizationConfirmScreen(
            draft,
            runs_root=app.runs_root,
            catalog_root=catalog_root,
            eval_assets_root=assets_root,
        )
        await _mount_screen(pilot, app, screen)

        assert screen.query_one("#po-prompt-input", Input).value == "Classify support tickets."
        assert screen.query_one("#po-system-prompt-input", Input).value == "Be concise."
        assert screen.query_one("#po-task-model-select", Select).value == "Local/tiny"
        assert screen.query_one("#po-optimizer-model-select", Select).value == "Local/tiny"
        assert screen.query_one("#po-eval-asset-select", Select).value == "tiny-eval@2026.07.12"
        assert screen.query_one("#po-metric-select", Select).value == "builtin:exact_match"
        assert screen.query_one("#po-run-name-input", Input).value == "priority-run"
        assert screen.query_one("#po-budget-input", Input).value == "25"


async def test_prefills_validation_errors_already_on_the_draft(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    assets_root = tmp_path / "eval-assets"

    draft = PromptOptimizationDraft(
        raw_text="gepa optimize ...",
        prompt="Classify tickets.",
        validation_errors=("model is not available in the unified catalog: openai/gpt-4.1-mini",),
    )

    app = _make_app(tmp_path, catalog_root=catalog_root, assets_root=assets_root)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        screen = PromptOptimizationConfirmScreen(
            draft, runs_root=app.runs_root, catalog_root=catalog_root, eval_assets_root=assets_root
        )
        await _mount_screen(pilot, app, screen)

        errors_text = str(screen.query_one("#po-errors", Static).content)
        assert "model is not available in the unified catalog" in errors_text


async def test_launch_blocked_until_required_fields_are_valid(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    assets_root = tmp_path / "eval-assets"
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval")
    identity = ModelIdentity(name="tiny", backend="llama.cpp")

    draft = _valid_draft(task_model=identity)
    app = _make_app(tmp_path, catalog_root=catalog_root, assets_root=assets_root)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        screen = PromptOptimizationConfirmScreen(
            draft, runs_root=app.runs_root, catalog_root=catalog_root, eval_assets_root=assets_root
        )
        await _mount_screen(pilot, app, screen)

        # Clear the required prompt field.
        prompt_input = screen.query_one("#po-prompt-input", Input)
        prompt_input.value = ""
        await pilot.pause()

        await pilot.click("#po-launch-button")
        await pilot.pause()

        assert app.screen is screen
        assert app.state is None
        assert app.pending_queue == []
        errors_text = str(screen.query_one("#po-errors", Static).content)
        assert "prompt is required" in errors_text


async def test_hosted_model_warning_displays_for_provider_backed_model(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    assets_root = tmp_path / "eval-assets"
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval")
    policy = PromptRoutingPolicy(
        provider_accounts=[ProviderAccount(provider="openai", account_id="default", is_signed_in=True)],
        provider_models=[ProviderModel(name="gpt-4.1-mini", provider="openai", account_id="default")],
    )
    hosted_identity = ModelIdentity(name="gpt-4.1-mini", backend="provider", provider="openai", account_id="default")
    draft = _valid_draft(task_model=hosted_identity)

    app = _make_app(tmp_path, catalog_root=catalog_root, assets_root=assets_root, policy=policy)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        screen = PromptOptimizationConfirmScreen(
            draft,
            runs_root=app.runs_root,
            catalog_root=catalog_root,
            eval_assets_root=assets_root,
            prompt_routing_policy=policy,
        )
        await _mount_screen(pilot, app, screen)

        warning_text = str(screen.query_one("#po-hosted-warning", Static).content)
        assert "Hosted provider model" in warning_text
        assert "gpt-4.1-mini" in warning_text
        assert "leave your machine" in warning_text


async def test_launch_starts_a_live_run_when_none_is_active(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    assets_root = tmp_path / "eval-assets"
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval")
    identity = ModelIdentity(name="tiny", backend="llama.cpp")
    draft = _valid_draft(task_model=identity, run_name="launch-now")

    app = _make_app(tmp_path, catalog_root=catalog_root, assets_root=assets_root)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        screen = PromptOptimizationConfirmScreen(
            draft, runs_root=app.runs_root, catalog_root=catalog_root, eval_assets_root=assets_root
        )
        await _mount_screen(pilot, app, screen)

        await pilot.click("#po-launch-button")
        await pilot.pause()

        assert isinstance(app.screen, DashboardScreen)
        assert app.state is not None
        assert app.state.run_name == "launch-now"
        assert app.state.run_kind == RunKind.PROMPT_OPTIMIZATION
        assert app.pending_queue == []

        # #48 hasn't wired a real runtime in yet -- runner.launch's default
        # unsupported-runtime stub fails the run cleanly on its background
        # thread rather than actually optimizing anything.
        for _ in range(20):
            if app.state.status is not RunStatus.RUNNING:
                break
            await pilot.pause()
        assert app.state.status is RunStatus.FAILED
        assert "not configured" in (app.state.error or "")


async def test_launch_queues_behind_an_already_live_run(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    assets_root = tmp_path / "eval-assets"
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval")
    identity = ModelIdentity(name="tiny", backend="llama.cpp")
    draft = _valid_draft(task_model=identity, run_name="queued-run")

    app = _make_app(tmp_path, catalog_root=catalog_root, assets_root=assets_root)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        # A live run always implies an already-mounted DashboardScreen (see
        # AutumnApp._refresh_queue_panel) -- enter browse mode first so
        # `_dashboard_screen` exists, then fake the run as RUNNING without
        # actually starting a background thread for it (is_run_live() only
        # inspects app.state.status).
        from models import DashboardState

        app.enter_browse_mode()
        await pilot.pause()
        app.state = DashboardState(run_name="already-live", run_dir=tmp_path / "already-live", status=RunStatus.RUNNING)

        screen = PromptOptimizationConfirmScreen(
            draft, runs_root=app.runs_root, catalog_root=catalog_root, eval_assets_root=assets_root
        )
        await _mount_screen(pilot, app, screen)

        screen.query_one("#po-launch-button").scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#po-launch-button")
        await pilot.pause()

        assert app.state.run_name == "already-live"
        assert len(app.pending_queue) == 1
        queued = app.pending_queue[0]
        assert queued.spec.run_name == "queued-run"
        assert screen not in app.screen_stack


async def test_cancel_returns_without_queueing_a_run(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    assets_root = tmp_path / "eval-assets"
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval")
    identity = ModelIdentity(name="tiny", backend="llama.cpp")
    draft = _valid_draft(task_model=identity)

    app = _make_app(tmp_path, catalog_root=catalog_root, assets_root=assets_root)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        screen = PromptOptimizationConfirmScreen(
            draft, runs_root=app.runs_root, catalog_root=catalog_root, eval_assets_root=assets_root
        )
        await _mount_screen(pilot, app, screen)

        await pilot.click("#po-cancel-button")
        await pilot.pause()

        assert app.state is None
        assert app.pending_queue == []
        assert screen not in app.screen_stack


async def test_eval_asset_defaults_only_when_exactly_one_installed(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    identity = ModelIdentity(name="tiny", backend="llama.cpp")
    draft = _valid_draft(task_model=identity)

    # Exactly one installed eval asset -> defaulted.
    single_assets_root = tmp_path / "single-eval-assets"
    _install_fake_eval_asset(single_assets_root, asset_id="only-eval")
    app = _make_app(tmp_path, catalog_root=catalog_root, assets_root=single_assets_root)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        screen = PromptOptimizationConfirmScreen(
            draft, runs_root=app.runs_root, catalog_root=catalog_root, eval_assets_root=single_assets_root
        )
        await _mount_screen(pilot, app, screen)
        select = screen.query_one("#po-eval-asset-select", Select)
        assert not select.is_blank()
        assert select.value == "only-eval@2026.07.12"

    # Two installed eval assets -> left for the user to choose.
    multi_assets_root = tmp_path / "multi-eval-assets"
    _install_fake_eval_asset(multi_assets_root, asset_id="eval-a")
    _install_fake_eval_asset(multi_assets_root, asset_id="eval-b")
    app2 = _make_app(tmp_path, catalog_root=catalog_root, assets_root=multi_assets_root)
    async with app2.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        screen2 = PromptOptimizationConfirmScreen(
            draft, runs_root=app2.runs_root, catalog_root=catalog_root, eval_assets_root=multi_assets_root
        )
        await _mount_screen(pilot, app2, screen2)
        select2 = screen2.query_one("#po-eval-asset-select", Select)
        assert select2.is_blank()


async def test_custom_local_metric_is_visually_distinguished_and_reveals_fields(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    assets_root = tmp_path / "eval-assets"
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval", supported_metrics=("exact_match",))
    identity = ModelIdentity(name="tiny", backend="llama.cpp")
    draft = _valid_draft(task_model=identity)

    app = _make_app(tmp_path, catalog_root=catalog_root, assets_root=assets_root)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        screen = PromptOptimizationConfirmScreen(
            draft, runs_root=app.runs_root, catalog_root=catalog_root, eval_assets_root=assets_root
        )
        await _mount_screen(pilot, app, screen)

        metric_select = screen.query_one("#po-metric-select", Select)
        # Single supported metric -> defaulted, labeled as built-in.
        assert metric_select.value == "builtin:exact_match"
        option_labels = [str(label) for label, _value in metric_select._options]
        assert any("built-in" in label for label in option_labels)
        assert any("trusted local code" in label for label in option_labels)

        path_input = screen.query_one("#po-custom-metric-path-input")
        assert path_input.display is False

        metric_select.value = "custom_local"
        await pilot.pause()

        assert screen.query_one("#po-custom-metric-path-input").display is True
        assert screen.query_one("#po-custom-metric-function-input").display is True

        screen.query_one("#po-custom-metric-path-input", Input).value = "/tmp/metric.py"
        screen.query_one("#po-custom-metric-function-input", Input).value = "score"
        await pilot.pause()

        screen.query_one("#po-launch-button").scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#po-launch-button")
        await pilot.pause()

        assert isinstance(app.screen, DashboardScreen)
        assert app.state is not None
