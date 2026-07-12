from pathlib import Path

from prompt_optimization_contracts import (
    BestPromptArtifact,
    BestResultArtifacts,
    BuiltInMetricRef,
    EvalAssetRef,
    ModelIdentity,
    PromptOptimizationSpec,
    PromptOptimizationSpecBudget,
    metric_ref_from_dict,
    prompt_optimization_spec_from_dict,
)


def test_prompt_optimization_spec_round_trips_through_plain_dict():
    spec = PromptOptimizationSpec(
        prompt="Classify the ticket by priority.",
        system_prompt="You are a careful triage assistant.",
        task_model=ModelIdentity(name="llama-3.1-8b", backend="llama.cpp"),
        optimizer_model=ModelIdentity(name="gpt-4.1", backend="openai", provider="openai", account_id="acct_123"),
        eval_asset=EvalAssetRef(asset_id="ticket-priority", version="2026-07-12"),
        metric=BuiltInMetricRef(metric_id="exact_match"),
        run_name="priority-optimizer",
        budget=PromptOptimizationSpecBudget(max_metric_calls=40),
    )

    payload = spec.to_dict()

    assert payload == {
        "schema_version": 1,
        "run_kind": "prompt_optimization",
        "prompt": "Classify the ticket by priority.",
        "system_prompt": "You are a careful triage assistant.",
        "task_model": {"name": "llama-3.1-8b", "backend": "llama.cpp"},
        "optimizer_model": {
            "name": "gpt-4.1",
            "backend": "openai",
            "provider": "openai",
            "account_id": "acct_123",
        },
        "eval_asset": {"asset_id": "ticket-priority", "version": "2026-07-12"},
        "metric": {"kind": "built_in", "metric_id": "exact_match"},
        "run_name": "priority-optimizer",
        "budget": {"max_metric_calls": 40},
    }
    assert prompt_optimization_spec_from_dict(payload) == spec


def test_prompt_optimization_spec_validation_rejects_missing_prompt():
    payload = {
        "schema_version": 1,
        "run_kind": "prompt_optimization",
        "prompt": "",
        "system_prompt": None,
        "task_model": {"name": "llama-3.1-8b", "backend": "llama.cpp"},
        "optimizer_model": {"name": "llama-3.1-8b", "backend": "llama.cpp"},
        "eval_asset": {"asset_id": "ticket-priority", "version": "2026-07-12"},
        "metric": {"kind": "built_in", "metric_id": "exact_match"},
        "run_name": "priority-optimizer",
        "budget": {"max_metric_calls": 40},
    }

    try:
        prompt_optimization_spec_from_dict(payload)
    except ValueError as exc:
        assert "prompt is required" in str(exc)
    else:
        raise AssertionError("expected prompt validation to fail")


def test_metric_ref_from_dict_supports_custom_local_metric():
    metric = metric_ref_from_dict(
        {
            "kind": "custom_local",
            "path": "/Users/me/autumn-metrics.py",
            "function": "score_answer",
        }
    )

    assert metric.to_dict() == {
        "kind": "custom_local",
        "path": str(Path("/Users/me/autumn-metrics.py")),
        "function": "score_answer",
    }


def test_best_result_artifacts_define_candidate_json_and_prompt_markdown():
    artifacts = BestResultArtifacts(
        candidate_json="best_candidate.json",
        prompt_markdown="best_prompt.md",
        best_prompt=BestPromptArtifact(
            prompt="Use short labels and quote the evidence.",
            system_prompt="You classify support tickets.",
            score=0.92,
            candidate_idx=3,
        ),
    )

    payload = artifacts.to_dict()

    assert payload == {
        "schema_version": 1,
        "candidate_json": "best_candidate.json",
        "prompt_markdown": "best_prompt.md",
        "best_prompt": {
            "prompt": "Use short labels and quote the evidence.",
            "system_prompt": "You classify support tickets.",
            "score": 0.92,
            "candidate_idx": 3,
        },
    }
    assert BestResultArtifacts.from_dict(payload) == artifacts
