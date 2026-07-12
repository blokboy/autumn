"""Chat-style intake for GEPA-specific implicit prompt optimization requests.

`cli.parse_command_line`'s implicit-trigger detection (#44) already turns a
narrow set of GEPA-specific phrases (e.g. "run GEPA on ...") into a
`PromptOptimizationDraft`, exactly as explicit `gepa optimize ...` does.
Explicit `gepa optimize` skips straight to `PromptOptimizationConfirmScreen`
(#47) even when incomplete -- typing a command is already a deliberate act.
An implicit chat phrase is different: the user was just talking, so instead
of dropping them straight into an edit form, `AutumnApp` drives a short,
explicit back-and-forth in chat (see `start_prompt_optimization_intake`/
`_handle_prompt_optimization_intake_answer`) to fill in whatever the one-line
trigger didn't already supply, before handing off to that same confirmation
screen.

`IntakeState` is deliberately its own small object, never conflated with
normal chat reply generation/streaming -- `AutumnApp` only consults it when
routing a submitted chat line, and clears it the moment intake ends
(completed or cancelled). Questions/answers are still recorded as ordinary
`ChatMessage`s so the conversation reads naturally.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, replace
from pathlib import Path

import eval_assets
from models import PromptRoutingPolicy
from prompt_optimization_contracts import BuiltInMetricRef, EvalAssetRef
from prompt_optimization_drafts import PromptOptimizationDraft, parse_prompt_optimization_draft

# Deliberately narrow and explicit rather than reusing chat's own cancel/stop
# vocabulary -- intake is the only flow that treats a whole line as a magic
# word, so it should be unambiguous which line did it.
CANCEL_PHRASES = frozenset({"cancel", "stop", "nevermind", "never mind", "cancel optimization", "quit"})

# Fixed so follow-up ordering is deterministic and testable, and so earlier
# answers are already resolved by the time later, dependent questions (e.g.
# metric, which is validated against the resolved eval asset) are asked.
_FIELD_ORDER = ("prompt", "task_model", "optimizer_model", "eval_asset", "metric")

_FLAG_FOR_FIELD = {
    "prompt": "--prompt",
    "task_model": "--task-model",
    "optimizer_model": "--optimizer-model",
    "eval_asset": "--eval-asset",
    "metric": "--metric",
}

_QUESTION_FOR_FIELD = {
    "prompt": "What prompt would you like to optimize?",
    "task_model": "Which model should run the task? (Give one model to use it for both task and optimizer.)",
    "optimizer_model": "Which model should act as the optimizer?",
    "eval_asset": "Which eval asset should I use? (`autumn evals list` shows what's installed -- answer as asset_id@version.)",
    "metric": "Which metric should score responses?",
}

_MERGEABLE_FIELDS = ("prompt", "system_prompt", "task_model", "optimizer_model", "eval_asset", "metric", "run_name")
_FREE_TEXT_FIELDS = frozenset({"prompt", "system_prompt"})


@dataclass(frozen=True)
class IntakeState:
    """One in-progress prompt-optimization chat intake."""

    draft: PromptOptimizationDraft
    asking_field: str | None
    last_errors: tuple[str, ...] = ()


def start_intake(draft: PromptOptimizationDraft, *, assets_root: Path) -> IntakeState:
    """Applies the same safe defaults #47's confirmation screen would (an
    eval asset when exactly one is installed, a metric when the resolved eval
    asset only supports one), then computes the first field to ask about, if
    any."""
    draft = _apply_defaults(draft, assets_root=assets_root)
    return IntakeState(draft=draft, asking_field=next_missing_field(draft))


def next_missing_field(draft: PromptOptimizationDraft) -> str | None:
    """Run name is deliberately not asked about: #47's confirmation screen
    already generates a timestamped default when the field is blank, so
    intake shouldn't interrupt the user for something that already has a
    sensible fallback."""
    if not draft.prompt:
        return "prompt"
    if draft.task_model is None:
        return "task_model"
    if draft.optimizer_model is None:
        return "optimizer_model"
    if draft.eval_asset is None:
        return "eval_asset"
    if draft.metric is None:
        return "metric"
    return None


def is_complete(draft: PromptOptimizationDraft) -> bool:
    return next_missing_field(draft) is None


def question_for(field: str) -> str:
    return _QUESTION_FOR_FIELD[field]


def is_cancel_phrase(text: str) -> bool:
    return text.strip().lower() in CANCEL_PHRASES


def answer_intake(
    state: IntakeState,
    answer_text: str,
    *,
    catalog_root: Path,
    assets_root: Path,
    prompt_routing_policy: PromptRoutingPolicy | None,
) -> IntakeState:
    """Parses `answer_text` via #44's deterministic flag/hint grammar first,
    so a user can answer several fields in one response (e.g. "task model
    tiny optimizer model bigger") -- except while asking about a free-text
    field (prompt/system_prompt), where a plain-English answer could easily
    contain an incidental word like "model" or "name" that would otherwise be
    misread as a hint; free-text answers only go through the full grammar
    when the user opts in with an explicit `--flag`. Otherwise (or when the
    full grammar recognizes nothing at all), retries once treating the whole
    trimmed answer as a single value for whichever field was just asked
    about, so a bare "tiny" answers "which task model?" without the user
    needing to type `--task-model tiny`, and a multi-word prompt answer isn't
    word-split the way shlex-based flag parsing would split it."""
    stripped = answer_text.strip()
    attempt_full_grammar = state.asking_field not in _FREE_TEXT_FIELDS or stripped.startswith("--")

    merged = state.draft
    errors: tuple[str, ...] = ()
    if attempt_full_grammar:
        parsed = _parse_answer_tokens(answer_text, catalog_root=catalog_root, prompt_routing_policy=prompt_routing_policy)
        merged = _merge(state.draft, parsed)
        errors = parsed.validation_errors

    if merged == state.draft and state.asking_field is not None:
        flag = _FLAG_FOR_FIELD.get(state.asking_field)
        if flag is not None:
            # `[flag, stripped]` is handed to the parser as a single
            # already-tokenized pair -- deliberately not shlex.split() over a
            # combined "flag value" string, which would word-split a
            # multi-word answer (e.g. a whole prompt) instead of keeping it
            # as one value.
            targeted = parse_prompt_optimization_draft(
                [flag, stripped],
                raw_text=answer_text,
                catalog_root=catalog_root,
                prompt_routing_policy=prompt_routing_policy,
            )
            merged = _merge(state.draft, targeted)
            errors = targeted.validation_errors

    merged = _apply_defaults(merged, assets_root=assets_root)
    remaining_field = next_missing_field(merged)
    return IntakeState(draft=merged, asking_field=remaining_field, last_errors=_filter_stale_errors(merged, errors))


def _filter_stale_errors(draft: PromptOptimizationDraft, errors: tuple[str, ...]) -> tuple[str, ...]:
    still_missing = {
        "prompt": not draft.prompt,
        "task model": draft.task_model is None,
        "optimizer model": draft.optimizer_model is None,
        "eval asset": draft.eval_asset is None,
        "metric": draft.metric is None,
    }
    kept = []
    for error in errors:
        # "<field> is required" errors about a field the merge just resolved
        # are stale noise; anything else (unknown model, bad eval asset
        # format, unknown metric, ...) is a real problem worth surfacing.
        stale = any(
            error == f"{field_label} is required" and not missing
            for field_label, missing in still_missing.items()
        )
        if not stale:
            kept.append(error)
    return tuple(kept)


def _parse_answer_tokens(
    text: str, *, catalog_root: Path, prompt_routing_policy: PromptRoutingPolicy | None
) -> PromptOptimizationDraft:
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = [text]
    return parse_prompt_optimization_draft(
        tokens, raw_text=text, catalog_root=catalog_root, prompt_routing_policy=prompt_routing_policy
    )


def _merge(draft: PromptOptimizationDraft, parsed: PromptOptimizationDraft) -> PromptOptimizationDraft:
    updates = {}
    for field_name in _MERGEABLE_FIELDS:
        value = getattr(parsed, field_name)
        if value is not None and getattr(draft, field_name) is None:
            updates[field_name] = value
    if parsed.budget.max_metric_calls is not None and draft.budget.max_metric_calls is None:
        updates["budget"] = parsed.budget
    if not updates:
        return draft
    return replace(draft, **updates)


def _apply_defaults(draft: PromptOptimizationDraft, *, assets_root: Path) -> PromptOptimizationDraft:
    """Mirrors `PromptOptimizationConfirmScreen`'s own defaulting: an eval
    asset is only assumed when exactly one is installed (never silently pick
    among several), and a metric is only assumed when the resolved eval asset
    supports exactly one."""
    updates: dict[str, object] = {}
    installed = eval_assets.list_installed_assets(assets_root)
    resolved_installed = None

    if draft.eval_asset is None and len(installed) == 1:
        resolved_installed = installed[0]
        updates["eval_asset"] = EvalAssetRef(asset_id=resolved_installed.asset_id, version=resolved_installed.version)
    elif draft.eval_asset is not None:
        resolved_installed = next(
            (
                asset
                for asset in installed
                if asset.asset_id == draft.eval_asset.asset_id and asset.version == draft.eval_asset.version
            ),
            None,
        )

    if draft.metric is None and resolved_installed is not None and len(resolved_installed.supported_metrics) == 1:
        updates["metric"] = BuiltInMetricRef(metric_id=resolved_installed.supported_metrics[0])

    if not updates:
        return draft
    return replace(draft, **updates)
