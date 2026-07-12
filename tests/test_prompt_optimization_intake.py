"""Behavioral tests for GEPA-specific chat intake (#50): a narrow implicit
trigger phrase (e.g. "run GEPA on ...") starts a short chat-style
back-and-forth to fill in missing prompt-optimization fields, rather than
jumping straight to the confirmation screen the way explicit
`gepa optimize ...` does (#47).

Covers: pure intake-state logic (field ordering, defaulting, multi-field
merge, bare single-field answers, cancel-phrase detection) and end-to-end
CommandBar-driven flows (trigger entry, non-trigger regression, multi-field
answers, follow-up ordering, cancellation/reset, confirmation handoff).
"""

import json
from pathlib import Path

import prompt_optimization_intake as intake
from app import AutumnApp
from models import ChatMessage
from prompt_optimization_contracts import BuiltInMetricRef, EvalAssetRef, ModelIdentity
from prompt_optimization_drafts import PromptOptimizationDraft
from screens.prompt_optimization_confirm_screen import PromptOptimizationConfirmScreen
from textual.widgets import Input
from widgets.command_bar import CommandBar

_TINY = ModelIdentity(name="conftest-seed-model", backend="llama.cpp")


async def _type_and_submit(pilot, text: str) -> None:
    bar_input = pilot.app.screen.query_one(CommandBar).query_one(Input)
    # CommandBar stays focused after a submit (only clears its value, see
    # CommandBar.on_input_submitted) -- ":" is only the global focus-toggle
    # shortcut while unfocused, so pressing it again on a second/third
    # submission in the same test would type a literal ":" character instead.
    if not bar_input.has_focus:
        await pilot.press(":")
        await pilot.pause()
    bar_input.insert_text_at_cursor(text)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


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


# -- pure intake-state logic --------------------------------------------------


def test_next_missing_field_orders_prompt_before_models_before_eval_before_metric():
    assert intake.next_missing_field(PromptOptimizationDraft(raw_text="")) == "prompt"
    assert (
        intake.next_missing_field(PromptOptimizationDraft(raw_text="", prompt="p")) == "task_model"
    )
    assert (
        intake.next_missing_field(
            PromptOptimizationDraft(raw_text="", prompt="p", task_model=_TINY)
        )
        == "optimizer_model"
    )
    assert (
        intake.next_missing_field(
            PromptOptimizationDraft(raw_text="", prompt="p", task_model=_TINY, optimizer_model=_TINY)
        )
        == "eval_asset"
    )
    assert (
        intake.next_missing_field(
            PromptOptimizationDraft(
                raw_text="",
                prompt="p",
                task_model=_TINY,
                optimizer_model=_TINY,
                eval_asset=EvalAssetRef(asset_id="a", version="1"),
            )
        )
        == "metric"
    )
    assert (
        intake.next_missing_field(
            PromptOptimizationDraft(
                raw_text="",
                prompt="p",
                task_model=_TINY,
                optimizer_model=_TINY,
                eval_asset=EvalAssetRef(asset_id="a", version="1"),
                metric=BuiltInMetricRef(metric_id="exact_match"),
            )
        )
        is None
    )


def test_next_missing_field_never_asks_about_run_name():
    complete_but_unnamed = PromptOptimizationDraft(
        raw_text="",
        prompt="p",
        task_model=_TINY,
        optimizer_model=_TINY,
        eval_asset=EvalAssetRef(asset_id="a", version="1"),
        metric=BuiltInMetricRef(metric_id="exact_match"),
        run_name=None,
    )
    assert intake.is_complete(complete_but_unnamed)


def test_start_intake_defaults_eval_asset_and_metric_when_unambiguous(tmp_path):
    assets_root = tmp_path / "eval-assets"
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval")
    draft = PromptOptimizationDraft(raw_text="", prompt="p", task_model=_TINY, optimizer_model=_TINY)

    state = intake.start_intake(draft, assets_root=assets_root)

    assert state.draft.eval_asset == EvalAssetRef(asset_id="tiny-eval", version="2026.07.12")
    assert state.draft.metric == BuiltInMetricRef(metric_id="exact_match")
    assert state.asking_field is None


def test_start_intake_does_not_default_eval_asset_when_multiple_installed(tmp_path):
    assets_root = tmp_path / "eval-assets"
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval", version="2026.07.12")
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval", version="2026.07.13")
    draft = PromptOptimizationDraft(raw_text="", prompt="p", task_model=_TINY, optimizer_model=_TINY)

    state = intake.start_intake(draft, assets_root=assets_root)

    assert state.draft.eval_asset is None
    assert state.asking_field == "eval_asset"


def test_answer_intake_merges_multiple_fields_from_one_response(tmp_path):
    assets_root = tmp_path / "eval-assets"
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval")
    catalog_root = tmp_path / "models"
    import local_models

    model_file = tmp_path / "tiny.gguf"
    model_file.write_bytes(b"fake")
    local_models.install_model(catalog_root, name="tiny", source_path=model_file)

    state = intake.IntakeState(
        draft=PromptOptimizationDraft(raw_text="", prompt="Classify tickets."),
        asking_field="task_model",
    )

    new_state = intake.answer_intake(
        state,
        "task model tiny optimizer model tiny",
        catalog_root=catalog_root,
        assets_root=assets_root,
        prompt_routing_policy=None,
    )

    assert new_state.draft.task_model == ModelIdentity(name="tiny", backend="llama.cpp")
    assert new_state.draft.optimizer_model == ModelIdentity(name="tiny", backend="llama.cpp")
    # Eval asset/metric auto-defaulted (single installed asset, single metric).
    assert new_state.asking_field is None


def test_answer_intake_treats_bare_reply_as_value_for_the_field_just_asked(tmp_path):
    assets_root = tmp_path / "eval-assets"
    catalog_root = tmp_path / "models"
    import local_models

    model_file = tmp_path / "tiny.gguf"
    model_file.write_bytes(b"fake")
    local_models.install_model(catalog_root, name="tiny", source_path=model_file)

    state = intake.IntakeState(
        draft=PromptOptimizationDraft(raw_text="", prompt="Classify tickets."),
        asking_field="task_model",
    )

    new_state = intake.answer_intake(
        state, "tiny", catalog_root=catalog_root, assets_root=assets_root, prompt_routing_policy=None
    )

    assert new_state.draft.task_model == ModelIdentity(name="tiny", backend="llama.cpp")
    assert new_state.asking_field == "optimizer_model"


def test_answer_intake_never_overwrites_an_already_resolved_field(tmp_path):
    assets_root = tmp_path / "eval-assets"
    catalog_root = tmp_path / "models"
    import local_models

    for name in ("tiny", "bigger"):
        model_file = tmp_path / f"{name}.gguf"
        model_file.write_bytes(b"fake")
        local_models.install_model(catalog_root, name=name, source_path=model_file)

    state = intake.IntakeState(
        draft=PromptOptimizationDraft(raw_text="", prompt="Classify tickets.", task_model=_TINY),
        asking_field="optimizer_model",
    )

    new_state = intake.answer_intake(
        state,
        "task model bigger optimizer model bigger",
        catalog_root=catalog_root,
        assets_root=assets_root,
        prompt_routing_policy=None,
    )

    # task_model was already resolved (tiny) before this answer -- a later
    # answer's own "task model bigger" must not silently overwrite it.
    assert new_state.draft.task_model == _TINY
    assert new_state.draft.optimizer_model == ModelIdentity(name="bigger", backend="llama.cpp")


def test_is_cancel_phrase():
    assert intake.is_cancel_phrase("cancel")
    assert intake.is_cancel_phrase("  Never Mind  ")
    assert not intake.is_cancel_phrase("cancel the eval asset named tiny")
    assert not intake.is_cancel_phrase("tiny")


# -- end-to-end CommandBar-driven flows ---------------------------------------


async def test_implicit_gepa_phrase_starts_chat_intake_with_followup_question(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await _type_and_submit(pilot, "run GEPA on this prompt")

        assert app.chat_messages[0] == ChatMessage(role="user", text="run GEPA on this prompt")
        assert app.chat_messages[-1].role == "assistant"
        assert app.chat_messages[-1].text == intake.question_for("prompt")
        # Still on the dashboard's chat surface, not the confirmation screen --
        # the one-line trigger alone didn't supply a prompt.
        assert not isinstance(app.screen, PromptOptimizationConfirmScreen)


async def test_non_trigger_optimize_phrase_stays_normal_chat(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await _type_and_submit(pilot, "optimize this prompt for me")
        await pilot.pause()

        assert app.chat_messages == [
            ChatMessage(role="user", text="optimize this prompt for me"),
            ChatMessage(
                role="assistant",
                text="Offline local response: optimize this prompt for me",
                model="autumn/offline-tiny",
            ),
        ]


async def test_cancelling_intake_returns_to_normal_chat(tmp_path):
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await _type_and_submit(pilot, "run GEPA on this prompt")
        assert app._prompt_optimization_intake is not None

        await _type_and_submit(pilot, "cancel")
        await pilot.pause()

        assert app._prompt_optimization_intake is None
        assert app.chat_messages[-1] == ChatMessage(
            role="assistant", text="Prompt optimization intake cancelled."
        )

        # Intake is gone -- the next line is normal chat again, not another
        # intake answer.
        await _type_and_submit(pilot, "just chatting now")
        await pilot.pause()
        assert app.chat_messages[-1] == ChatMessage(
            role="assistant",
            text="Offline local response: just chatting now",
            model="autumn/offline-tiny",
        )


async def test_followup_questions_are_asked_one_field_at_a_time_in_order(tmp_path):
    assets_root = tmp_path / "eval-assets"
    # Two installed assets -- eval asset can't auto-default (only exactly-one
    # installed does), so this field must actually be asked about. The
    # chosen one has two supported metrics, so metric can't auto-default
    # either once it's picked.
    _install_fake_eval_asset(
        assets_root, asset_id="tiny-eval-a", supported_metrics=("exact_match", "contains")
    )
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval-b", supported_metrics=("exact_match",))
    app = AutumnApp(runs_root=tmp_path, eval_assets_root=assets_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await _type_and_submit(pilot, "run GEPA on this")
        assert app.chat_messages[-1].text == intake.question_for("prompt")

        await _type_and_submit(pilot, "Classify support tickets by priority.")
        assert app.chat_messages[-1].text == intake.question_for("task_model")

        await _type_and_submit(pilot, "conftest-seed-model")
        assert app.chat_messages[-1].text == intake.question_for("optimizer_model")

        await _type_and_submit(pilot, "conftest-seed-model")
        assert app.chat_messages[-1].text == intake.question_for("eval_asset")

        await _type_and_submit(pilot, "tiny-eval-a@2026.07.12")
        # Two supported metrics on this asset -- can't auto-default, must ask.
        assert app.chat_messages[-1].text == intake.question_for("metric")


async def test_completed_intake_opens_confirmation_screen_prefilled(tmp_path):
    assets_root = tmp_path / "eval-assets"
    _install_fake_eval_asset(assets_root, asset_id="tiny-eval")
    app = AutumnApp(runs_root=tmp_path, eval_assets_root=assets_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await _type_and_submit(pilot, "run GEPA on this")
        await _type_and_submit(pilot, "Classify support tickets by priority.")
        # One line answering both remaining required models -- eval
        # asset/metric auto-default (single installed asset, single metric).
        await _type_and_submit(pilot, "task model conftest-seed-model optimizer model conftest-seed-model")
        await pilot.pause()

        assert app._prompt_optimization_intake is None
        assert isinstance(app.screen, PromptOptimizationConfirmScreen)
        draft = app.screen.draft
        assert draft.prompt == "Classify support tickets by priority."
        assert draft.task_model == ModelIdentity(name="conftest-seed-model", backend="llama.cpp")
        assert draft.optimizer_model == ModelIdentity(name="conftest-seed-model", backend="llama.cpp")
        assert draft.eval_asset == EvalAssetRef(asset_id="tiny-eval", version="2026.07.12")
        assert draft.metric == BuiltInMetricRef(metric_id="exact_match")


async def test_explicit_gepa_optimize_still_skips_intake_entirely(tmp_path):
    """No regression: explicit `gepa optimize ...` opens the confirmation
    screen directly, exactly as #47 already tests -- it never touches intake
    state or the chat transcript."""
    app = AutumnApp(runs_root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await _type_and_submit(pilot, "gepa optimize --prompt 'Write a summary'")

        assert app._prompt_optimization_intake is None
        assert app.chat_messages == []
        assert isinstance(app.screen, PromptOptimizationConfirmScreen)
