"""Editable final review screen for first-class GEPA prompt optimization
commands (`gepa optimize ...` and, later, GEPA-specific chat intake).

`AutumnApp.start_prompt_optimization_draft` pushes this screen with whatever
`PromptOptimizationDraft` was parsed from the user's command -- fields that
resolved cleanly are prefilled, fields that didn't (or weren't given at all)
are left blank, and any of `draft.validation_errors` are shown immediately
so the user knows what to fix before ever touching Launch.

Launch re-validates from the *current* widget values (not the original
draft) into a durable `PromptOptimizationSpec`, so edits made on this screen
always win. If a run is already live, Launch queues behind it (matching how
a script run submitted while busy is queued rather than started
immediately); otherwise it promotes this AutumnApp straight into a live
prompt optimization run. Cancel (button, "n", or Escape) always just pops
back to whatever screen was active before, without queueing anything.

Actually executing a queued/launched spec against real GEPA is out of scope
here -- that's issue #48. `runner.launch` already accepts a
`PromptOptimizationRunSpec` and simply fails the run cleanly (via its
`_unsupported_prompt_optimization_runtime` default) until #48 wires a real
runtime in.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static

import catalog
import eval_assets
from eval_assets import InstalledEvalAsset
from models import CatalogEntry, PromptOptimizationRunSpec, PromptRoutingPolicy
from prompt_optimization_contracts import (
    BuiltInMetricRef,
    CustomLocalMetricRef,
    EvalAssetRef,
    MetricRef,
    ModelIdentity,
    PromptOptimizationSpec,
    PromptOptimizationSpecBudget,
)
from prompt_optimization_drafts import PromptOptimizationDraft

# Applied to the Budget field whenever the draft carries no explicit budget --
# "Required optimization fields before launch are prompt, task model,
# optimizer model, eval asset, metric, and run name/budget defaults" (see
# issue #39's Implementation Decisions): a default must always be available,
# not left for the user to have to know a good number for.
_DEFAULT_MAX_METRIC_CALLS = 100

_CUSTOM_LOCAL_METRIC_VALUE = "custom_local"


def _model_key(entry: CatalogEntry) -> str:
    return f"{entry.group}/{entry.name}"


def _model_option_label(entry: CatalogEntry) -> str:
    suffix = " (hosted provider)" if entry.backend == "provider" else ""
    return f"{_model_key(entry)}{suffix}"


def _entry_matches_identity(entry: CatalogEntry, identity: ModelIdentity) -> bool:
    return (
        entry.name == identity.name
        and entry.backend == identity.backend
        and entry.provider == identity.provider
        and entry.account_id == identity.account_id
    )


def _builtin_metric_value(metric_id: str) -> str:
    return f"builtin:{metric_id}"


def _same_eval_asset(left: InstalledEvalAsset | None, right: InstalledEvalAsset | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.asset_id == right.asset_id and left.version == right.version


class PromptOptimizationConfirmScreen(Screen):
    """Prefilled, editable review of a `PromptOptimizationDraft` before it
    becomes a durable `PromptOptimizationSpec` and is queued/launched."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    PromptOptimizationConfirmScreen {
        align: center middle;
    }
    PromptOptimizationConfirmScreen #po-confirm-frame {
        width: 84;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round #d98e4a;
        background: #241a12;
    }
    PromptOptimizationConfirmScreen .po-title {
        text-style: bold;
        color: #d98e4a;
        margin-bottom: 1;
    }
    PromptOptimizationConfirmScreen .po-field-label {
        margin-top: 1;
        text-opacity: 85%;
    }
    PromptOptimizationConfirmScreen .po-hint {
        text-opacity: 60%;
    }
    PromptOptimizationConfirmScreen .po-hosted-warning {
        color: #d9a04a;
        margin-top: 1;
    }
    PromptOptimizationConfirmScreen .po-errors {
        color: #d94a4a;
        margin-top: 1;
    }
    PromptOptimizationConfirmScreen .po-button-row {
        width: auto;
        height: auto;
        margin-top: 1;
    }
    PromptOptimizationConfirmScreen .po-button-row Button {
        margin-right: 1;
    }
    """

    def __init__(
        self,
        draft: PromptOptimizationDraft,
        *,
        runs_root: Path,
        catalog_root: Path,
        eval_assets_root: Path,
        prompt_routing_policy: PromptRoutingPolicy | None = None,
        default_max_metric_calls: int = _DEFAULT_MAX_METRIC_CALLS,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.draft = draft
        self.runs_root = Path(runs_root)
        self._default_max_metric_calls = default_max_metric_calls

        self._catalog_entries = [
            entry
            for entry in catalog.build_entries(catalog_root, policy=prompt_routing_policy)
            if entry.group != catalog.LOCAL_GROUP or entry.status == "installed"
        ]
        self._entries_by_key = {_model_key(entry): entry for entry in self._catalog_entries}
        self._model_options = [(_model_option_label(entry), _model_key(entry)) for entry in self._catalog_entries]

        self._installed_eval_assets = eval_assets.list_installed_assets(Path(eval_assets_root))
        self._assets_by_key = {
            f"{asset.asset_id}@{asset.version}": asset for asset in self._installed_eval_assets
        }
        self._eval_options = [(key, key) for key in self._assets_by_key]

        self._initial_task_value = self._identity_value(draft.task_model)
        self._initial_optimizer_value = self._identity_value(draft.optimizer_model)
        self._initial_eval_asset = self._default_eval_asset()
        self._initial_eval_value = (
            f"{self._initial_eval_asset.asset_id}@{self._initial_eval_asset.version}"
            if self._initial_eval_asset is not None
            else Select.NULL
        )
        self._initial_metric_value = self._initial_metric_key()
        # Which eval asset the metric Select's *current* options were built
        # for -- lets on_select_changed tell a genuine user-driven eval-asset
        # change (which should reset the metric selection) apart from the
        # eval-asset Select's own constructor-supplied initial value firing
        # its first Changed message on mount (which must NOT clobber the
        # metric Select's separately-computed initial value via set_options,
        # see on_select_changed below).
        self._metric_options_asset = self._initial_eval_asset

    # -- initial-value resolution -------------------------------------------------

    def _identity_value(self, identity: ModelIdentity | None):
        if identity is None:
            return Select.NULL
        match = next((entry for entry in self._catalog_entries if _entry_matches_identity(entry, identity)), None)
        return _model_key(match) if match is not None else Select.NULL

    def _default_eval_asset(self) -> InstalledEvalAsset | None:
        if self.draft.eval_asset is not None:
            match = next(
                (
                    asset
                    for asset in self._installed_eval_assets
                    if asset.asset_id == self.draft.eval_asset.asset_id
                    and asset.version == self.draft.eval_asset.version
                ),
                None,
            )
            if match is not None:
                return match
        # "Eval selection defaults only when exactly one installed eval is
        # available; otherwise the user is asked to choose" (#47).
        if len(self._installed_eval_assets) == 1:
            return self._installed_eval_assets[0]
        return None

    def _initial_metric_key(self):
        asset = self._initial_eval_asset
        if asset is None:
            return Select.NULL
        metric = self.draft.metric
        if isinstance(metric, BuiltInMetricRef) and metric.metric_id in asset.supported_metrics:
            return _builtin_metric_value(metric.metric_id)
        if len(asset.supported_metrics) == 1:
            return _builtin_metric_value(asset.supported_metrics[0])
        return Select.NULL

    def _initial_run_name(self) -> str:
        if self.draft.run_name:
            return self.draft.run_name
        return f"prompt-optimize-{datetime.now():%Y%m%d-%H%M%S}"

    def _metric_options_for(self, asset: InstalledEvalAsset | None) -> list[tuple[str, str]]:
        options = []
        if asset is not None:
            options.extend(
                (f"{metric_id} (built-in)", _builtin_metric_value(metric_id))
                for metric_id in asset.supported_metrics
            )
        # Always offered, distinct from built-ins by both label and value --
        # "Built-in and custom local metrics are visually distinguishable,
        # with custom metrics marked as trusted local code" (#47).
        options.append(("Custom local metric... (trusted local code)", _CUSTOM_LOCAL_METRIC_VALUE))
        return options

    # -- compose --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="po-confirm-frame"):
            yield Label("Confirm prompt optimization", classes="po-title")

            yield Label("Prompt", classes="po-field-label")
            yield Input(value=self.draft.prompt or "", placeholder="Prompt to optimize", id="po-prompt-input")

            yield Label("System prompt (optional)", classes="po-field-label")
            yield Input(
                value=self.draft.system_prompt or "",
                placeholder="Optional system prompt",
                id="po-system-prompt-input",
            )

            yield Label("Task model", classes="po-field-label")
            yield Select(
                self._model_options,
                value=self._initial_task_value,
                allow_blank=True,
                id="po-task-model-select",
            )

            yield Label("Optimizer model", classes="po-field-label")
            yield Select(
                self._model_options,
                value=self._initial_optimizer_value,
                allow_blank=True,
                id="po-optimizer-model-select",
            )

            yield Static("", id="po-hosted-warning", classes="po-hosted-warning")

            yield Label("Eval asset", classes="po-field-label")
            yield Select(
                self._eval_options,
                value=self._initial_eval_value,
                allow_blank=True,
                disabled=not self._eval_options,
                id="po-eval-asset-select",
            )
            if not self._installed_eval_assets:
                yield Static(
                    "No eval assets installed. Run `autumn evals install <asset[@version]>`.",
                    classes="po-hint",
                )

            yield Label("Metric", classes="po-field-label")
            yield Select(
                self._metric_options_for(self._initial_eval_asset),
                value=self._initial_metric_value,
                allow_blank=True,
                id="po-metric-select",
            )
            yield Label("Custom metric file path", classes="po-field-label", id="po-custom-metric-path-label")
            yield Input(placeholder="/path/to/metric.py", id="po-custom-metric-path-input")
            yield Label(
                "Custom metric function", classes="po-field-label", id="po-custom-metric-function-label"
            )
            yield Input(placeholder="score", id="po-custom-metric-function-input")

            yield Label("Run name", classes="po-field-label")
            yield Input(value=self._initial_run_name(), id="po-run-name-input")

            yield Label("Budget (max metric calls)", classes="po-field-label")
            yield Input(
                value=str(self.draft.budget.max_metric_calls or self._default_max_metric_calls),
                id="po-budget-input",
            )

            yield Static("", id="po-errors", classes="po-errors")

            with Horizontal(classes="po-button-row"):
                yield Button("Launch", variant="success", id="po-launch-button")
                yield Button("Cancel", variant="default", id="po-cancel-button")

    def on_mount(self) -> None:
        self._sync_custom_metric_visibility()
        self._refresh_hosted_warning()
        if self.draft.validation_errors:
            self._show_errors(list(self.draft.validation_errors))

    # -- reactive field wiring --------------------------------------------------

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "po-eval-asset-select":
            asset = None if event.select.is_blank() else self._assets_by_key.get(event.value)
            if _same_eval_asset(asset, self._metric_options_asset):
                # The eval-asset Select's own constructor-supplied initial
                # value fires a Changed message once mounted, same as a real
                # user pick would -- but the metric Select was already given
                # matching initial options/value directly in compose(), and
                # Select.set_options() unconditionally clears the current
                # selection, so calling it again here for a no-op "change"
                # would wipe out that prefilled metric for nothing.
                return
            self._metric_options_asset = asset
            metric_select = self.query_one("#po-metric-select", Select)
            metric_select.set_options(self._metric_options_for(asset))
            self._sync_custom_metric_visibility()
            return
        if event.select.id == "po-metric-select":
            self._sync_custom_metric_visibility()
            return
        if event.select.id in ("po-task-model-select", "po-optimizer-model-select"):
            self._refresh_hosted_warning()

    def _sync_custom_metric_visibility(self) -> None:
        metric_select = self.query_one("#po-metric-select", Select)
        is_custom = not metric_select.is_blank() and metric_select.value == _CUSTOM_LOCAL_METRIC_VALUE
        for selector in (
            "#po-custom-metric-path-label",
            "#po-custom-metric-path-input",
            "#po-custom-metric-function-label",
            "#po-custom-metric-function-input",
        ):
            self.query_one(selector).display = is_custom

    def _refresh_hosted_warning(self) -> None:
        warning = self.query_one("#po-hosted-warning", Static)
        hosted_names = []
        for select_id in ("po-task-model-select", "po-optimizer-model-select"):
            select = self.query_one(f"#{select_id}", Select)
            if select.is_blank():
                continue
            entry = self._entries_by_key.get(select.value)
            if entry is not None and entry.backend == "provider":
                hosted_names.append(entry.name)
        if hosted_names:
            warning.update(
                "Hosted provider model(s) selected ("
                + ", ".join(hosted_names)
                + "). Prompts, eval inputs, candidates, and responses may leave your machine."
            )
        else:
            warning.update("")

    def _show_errors(self, errors: list[str]) -> None:
        self.query_one("#po-errors", Static).update("\n".join(f"- {error}" for error in errors))

    # -- launch/cancel ----------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "po-cancel-button":
            self.action_cancel()
            return
        if event.button.id == "po-launch-button":
            self._attempt_launch()

    def action_cancel(self) -> None:
        self.app.pop_screen()

    def _attempt_launch(self) -> None:
        spec, errors = self._build_spec()
        if errors:
            self._show_errors(errors)
            return
        self._show_errors([])
        if self.app.is_run_live():
            self.app.queue_prompt_optimization_run(spec)
            self.app.pop_screen()
            return
        run_spec = PromptOptimizationRunSpec(optimization_spec=spec, run_dir=self.runs_root / spec.run_name)
        self.app.launch_prompt_optimization_run(run_spec)

    def _build_spec(self) -> tuple[PromptOptimizationSpec | None, list[str]]:
        errors: list[str] = []

        prompt = self.query_one("#po-prompt-input", Input).value.strip()
        if not prompt:
            errors.append("prompt is required")

        system_prompt = self.query_one("#po-system-prompt-input", Input).value.strip() or None

        task_model = self._selected_model("po-task-model-select", "task model", errors)
        optimizer_model = self._selected_model("po-optimizer-model-select", "optimizer model", errors)
        eval_asset = self._selected_eval_asset(errors)
        metric = self._selected_metric(eval_asset, errors)

        run_name = self.query_one("#po-run-name-input", Input).value.strip()
        if not run_name:
            errors.append("run name is required")

        budget = self._selected_budget(errors)

        if errors:
            return None, errors

        spec = PromptOptimizationSpec(
            prompt=prompt,
            system_prompt=system_prompt,
            task_model=task_model,
            optimizer_model=optimizer_model,
            eval_asset=EvalAssetRef(asset_id=eval_asset.asset_id, version=eval_asset.version),
            metric=metric,
            run_name=run_name,
            budget=budget,
        )
        return spec, []

    def _selected_model(self, select_id: str, label: str, errors: list[str]) -> ModelIdentity | None:
        select = self.query_one(f"#{select_id}", Select)
        if select.is_blank():
            errors.append(f"{label} is required")
            return None
        entry = self._entries_by_key.get(select.value)
        if entry is None:
            errors.append(f"{label} is not available in the unified catalog")
            return None
        return ModelIdentity(
            name=entry.name, backend=entry.backend, provider=entry.provider, account_id=entry.account_id
        )

    def _selected_eval_asset(self, errors: list[str]) -> InstalledEvalAsset | None:
        if not self._installed_eval_assets:
            errors.append("no eval assets installed; run `autumn evals install <asset[@version]>`")
            return None
        select = self.query_one("#po-eval-asset-select", Select)
        if select.is_blank():
            errors.append("eval asset is required")
            return None
        asset = self._assets_by_key.get(select.value)
        if asset is None:
            errors.append("selected eval asset is no longer installed")
        return asset

    def _selected_metric(self, eval_asset: InstalledEvalAsset | None, errors: list[str]) -> MetricRef | None:
        select = self.query_one("#po-metric-select", Select)
        if select.is_blank():
            errors.append("metric is required")
            return None
        if select.value == _CUSTOM_LOCAL_METRIC_VALUE:
            path_value = self.query_one("#po-custom-metric-path-input", Input).value.strip()
            function_value = self.query_one("#po-custom-metric-function-input", Input).value.strip()
            if not path_value or not function_value:
                errors.append("custom local metric requires both a file path and a function name")
                return None
            return CustomLocalMetricRef(path=Path(path_value), function=function_value)
        metric_id = select.value.removeprefix("builtin:")
        if eval_asset is not None and metric_id not in eval_asset.supported_metrics:
            errors.append(f"metric '{metric_id}' is not supported by the selected eval asset")
            return None
        return BuiltInMetricRef(metric_id=metric_id)

    def _selected_budget(self, errors: list[str]) -> PromptOptimizationSpecBudget:
        raw = self.query_one("#po-budget-input", Input).value.strip()
        if not raw:
            errors.append("budget is required")
            return PromptOptimizationSpecBudget()
        try:
            max_metric_calls = int(raw)
        except ValueError:
            errors.append("budget must be a positive integer")
            return PromptOptimizationSpecBudget()
        if max_metric_calls <= 0:
            errors.append("budget must be a positive integer")
            return PromptOptimizationSpecBudget()
        return PromptOptimizationSpecBudget(max_metric_calls=max_metric_calls)
