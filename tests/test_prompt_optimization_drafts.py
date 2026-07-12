from pathlib import Path

import local_models
from cli import PromptOptimizationDraft, parse_command_line
from models import PromptRoutingPolicy, ProviderAccount, ProviderModel
from prompt_optimization_contracts import (
    BuiltInMetricRef,
    EvalAssetRef,
    ModelIdentity,
    PromptOptimizationSpecBudget,
)


def test_explicit_optimize_flags_populate_a_validated_draft(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    policy = _provider_policy("openai", "gpt-4.1-mini")

    draft = parse_command_line(
        "gepa optimize "
        "--prompt 'Classify support tickets by priority.' "
        "--system-prompt 'You are a careful triage assistant.' "
        "--task-model tiny "
        "--optimizer-model openai/gpt-4.1-mini "
        "--eval-asset ticket-priority@2026-07-12 "
        "--metric exact_match "
        "--name priority-optimizer "
        "--max-metric-calls 40",
        catalog_root=catalog_root,
        prompt_routing_policy=policy,
    )

    assert draft == PromptOptimizationDraft(
        raw_text=(
            "gepa optimize "
            "--prompt 'Classify support tickets by priority.' "
            "--system-prompt 'You are a careful triage assistant.' "
            "--task-model tiny "
            "--optimizer-model openai/gpt-4.1-mini "
            "--eval-asset ticket-priority@2026-07-12 "
            "--metric exact_match "
            "--name priority-optimizer "
            "--max-metric-calls 40"
        ),
        prompt="Classify support tickets by priority.",
        system_prompt="You are a careful triage assistant.",
        task_model=ModelIdentity(name="tiny", backend="llama.cpp"),
        optimizer_model=ModelIdentity(
            name="gpt-4.1-mini",
            backend="provider",
            provider="openai",
            account_id="default",
        ),
        eval_asset=EvalAssetRef(asset_id="ticket-priority", version="2026-07-12"),
        metric=BuiltInMetricRef(metric_id="exact_match"),
        run_name="priority-optimizer",
        budget=PromptOptimizationSpecBudget(max_metric_calls=40),
        validation_errors=(),
    )


def test_single_model_flag_fills_task_and_optimizer_models(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")

    draft = parse_command_line(
        "gepa optimize --prompt 'Answer briefly.' --model tiny",
        catalog_root=catalog_root,
    )

    assert draft.task_model == ModelIdentity(name="tiny", backend="llama.cpp")
    assert draft.optimizer_model == ModelIdentity(name="tiny", backend="llama.cpp")
    assert draft.validation_errors == ()


def test_task_and_optimizer_model_flags_split_model_fields(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    local_models.install_model(catalog_root, name="bigger", source_path=_model_file(tmp_path, "bigger"))

    draft = parse_command_line(
        "gepa optimize --prompt 'Answer carefully.' --task-model tiny --optimizer-model bigger",
        catalog_root=catalog_root,
    )

    assert draft.task_model == ModelIdentity(name="tiny", backend="llama.cpp")
    assert draft.optimizer_model == ModelIdentity(name="bigger", backend="llama.cpp")
    assert draft.validation_errors == ()


def test_deterministic_field_hints_populate_multiple_fields(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    local_models.install_model(catalog_root, name="bigger", source_path=_model_file(tmp_path, "bigger"))

    draft = parse_command_line(
        "gepa optimize prompt 'Classify tickets.' "
        "task model tiny optimizer model bigger "
        "eval asset ticket-priority@v1 metric contains name ticket-run budget 25",
        catalog_root=catalog_root,
    )

    assert draft.prompt == "Classify tickets."
    assert draft.task_model == ModelIdentity(name="tiny", backend="llama.cpp")
    assert draft.optimizer_model == ModelIdentity(name="bigger", backend="llama.cpp")
    assert draft.eval_asset == EvalAssetRef(asset_id="ticket-priority", version="v1")
    assert draft.metric == BuiltInMetricRef(metric_id="contains")
    assert draft.run_name == "ticket-run"
    assert draft.budget == PromptOptimizationSpecBudget(max_metric_calls=25)
    assert draft.validation_errors == ()


def test_unknown_or_unavailable_models_are_recoverable_validation_errors(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")
    policy = PromptRoutingPolicy(
        provider_accounts=[
            ProviderAccount(provider="openai", account_id="default", is_signed_in=False)
        ],
        provider_models=[
            ProviderModel(name="gpt-4.1-mini", provider="openai", account_id="default")
        ],
    )

    draft = parse_command_line(
        "gepa optimize --prompt 'Answer carefully.' --model openai/gpt-4.1-mini",
        catalog_root=catalog_root,
        prompt_routing_policy=policy,
    )

    assert draft.task_model is None
    assert draft.optimizer_model is None
    assert draft.validation_errors == (
        "model is not available in the unified catalog: openai/gpt-4.1-mini",
    )


def test_gepa_specific_implicit_phrase_creates_a_draft(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")

    draft = parse_command_line(
        "Use GEPA to optimize --prompt 'Classify tickets.' --model tiny",
        catalog_root=catalog_root,
    )

    assert isinstance(draft, PromptOptimizationDraft)
    assert draft.prompt == "Classify tickets."
    assert draft.task_model == ModelIdentity(name="tiny", backend="llama.cpp")
    assert draft.optimizer_model == ModelIdentity(name="tiny", backend="llama.cpp")


def test_run_gepa_on_phrase_creates_a_draft(tmp_path):
    catalog_root = _catalog_with_local_model(tmp_path, "tiny")

    draft = parse_command_line(
        "Run GEPA on --prompt 'Summarize the ticket.' --model tiny",
        catalog_root=catalog_root,
    )

    assert isinstance(draft, PromptOptimizationDraft)
    assert draft.prompt == "Summarize the ticket."
    assert draft.task_model == ModelIdentity(name="tiny", backend="llama.cpp")
    assert draft.optimizer_model == ModelIdentity(name="tiny", backend="llama.cpp")


def test_generic_optimize_phrasing_stays_chat():
    assert parse_command_line("optimize this prompt for me") is None
    assert parse_command_line("make this better: classify tickets") is None


def test_gepa_mention_without_lead_verb_or_connector_stays_chat():
    assert parse_command_line("I really like gepa as a tool") is None
    assert parse_command_line("what is gepa") is None


def _catalog_with_local_model(tmp_path: Path, name: str) -> Path:
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name=name, source_path=_model_file(tmp_path, name))
    return catalog_root


def _model_file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.gguf"
    path.write_bytes(b"fake model")
    return path


def _provider_policy(provider: str, model: str) -> PromptRoutingPolicy:
    return PromptRoutingPolicy(
        provider_accounts=[
            ProviderAccount(provider=provider, account_id="default", is_signed_in=True)
        ],
        provider_models=[
            ProviderModel(name=model, provider=provider, account_id="default")
        ],
    )
