from pathlib import Path

import pytest

from prompt_optimization_contracts import BuiltInMetricRef, CustomLocalMetricRef
from prompt_optimization_metrics import (
    EvalAssetMetricSupport,
    MetricExecutionError,
    MetricResolutionError,
    default_metric_for_eval,
    list_builtin_metrics,
    resolve_metric,
)


def test_resolves_builtin_exact_match_metric_for_supported_eval_asset():
    metric = resolve_metric(
        BuiltInMetricRef(metric_id="exact_match"),
        EvalAssetMetricSupport(
            asset_id="ticket-priority",
            default_metric_id="exact_match",
            supported_metric_ids=("exact_match",),
        ),
    )

    result = metric.evaluate(
        example={"answer": "high"},
        final_response=" high ",
        candidate={"candidate_idx": 2, "full_transcript": ["hidden"]},
    )

    assert result.score == 1.0


def test_builtin_metric_registry_lists_shipped_metrics_and_eval_defaults():
    eval_asset = EvalAssetMetricSupport(
        asset_id="ticket-priority",
        default_metric_id="exact_match",
        supported_metric_ids=("exact_match", "contains"),
    )

    assert list_builtin_metrics() == ("exact_match", "contains")
    assert default_metric_for_eval(eval_asset) == BuiltInMetricRef(metric_id="exact_match")


def test_unknown_builtin_metric_id_is_rejected_clearly():
    with pytest.raises(MetricResolutionError, match="unknown built-in metric 'rouge_l'"):
        resolve_metric(BuiltInMetricRef(metric_id="rouge_l"))


def test_unsupported_builtin_metric_for_eval_asset_is_rejected_clearly():
    with pytest.raises(
        MetricResolutionError,
        match="metric 'contains' is not supported by eval asset 'ticket-priority'",
    ):
        resolve_metric(
            BuiltInMetricRef(metric_id="contains"),
            EvalAssetMetricSupport(
                asset_id="ticket-priority",
                default_metric_id="exact_match",
                supported_metric_ids=("exact_match",),
            ),
        )


def test_eval_asset_metric_support_rejects_default_outside_supported_set():
    with pytest.raises(
        ValueError,
        match="default metric 'contains' must be listed in supported metrics",
    ):
        EvalAssetMetricSupport(
            asset_id="ticket-priority",
            default_metric_id="contains",
            supported_metric_ids=("exact_match",),
        )


def test_custom_local_metric_loads_explicit_path_and_function(tmp_path):
    metric_file = _write_metric(
        tmp_path,
        """
def score_answer(example, final_response, context):
    return 0.75
""",
    )

    metric = resolve_metric(CustomLocalMetricRef(path=metric_file, function="score_answer"))

    result = metric.evaluate(
        example={"answer": "high"},
        final_response="medium",
        candidate={"candidate_idx": 9},
    )

    assert result.score == 0.75
    assert metric.trust_boundary == "custom_local"


def test_custom_local_metric_remains_distinct_from_eval_asset_builtin_support(tmp_path):
    metric_file = _write_metric(
        tmp_path,
        """
def score_answer(example, final_response, context):
    return 1.0
""",
    )

    metric = resolve_metric(
        CustomLocalMetricRef(path=metric_file, function="score_answer"),
        EvalAssetMetricSupport(
            asset_id="ticket-priority",
            default_metric_id="exact_match",
            supported_metric_ids=("exact_match",),
        ),
    )

    assert metric.trust_boundary == "custom_local"


def test_custom_local_metric_receives_sanitized_candidate_context(tmp_path):
    metric_file = _write_metric(
        tmp_path,
        """
def score_answer(example, final_response, context):
    assert example == {"answer": "high"}
    assert final_response == "medium"
    assert context == {
        "candidate_idx": 7,
        "prompt": "Classify by priority.",
        "system_prompt": "You are careful.",
        "metadata": {"source": "val"},
    }
    return {"score": 0.25, "details": {"seen": True}}
""",
    )
    metric = resolve_metric(CustomLocalMetricRef(path=metric_file, function="score_answer"))

    result = metric.evaluate(
        example={"answer": "high"},
        final_response="medium",
        candidate={
            "candidate_idx": 7,
            "prompt": "Classify by priority.",
            "system_prompt": "You are careful.",
            "metadata": {"source": "val"},
            "full_transcript": ["do not pass"],
            "request_messages": ["do not pass"],
            "response_messages": ["do not pass"],
        },
    )

    assert result.score == 0.25
    assert result.details == {"seen": True}


def test_custom_local_metric_invalid_return_is_rejected(tmp_path):
    metric_file = _write_metric(
        tmp_path,
        """
def score_answer(example, final_response, context):
    return {"value": 1.0}
""",
    )
    metric = resolve_metric(CustomLocalMetricRef(path=metric_file, function="score_answer"))

    with pytest.raises(
        MetricExecutionError,
        match="must return a number or an object with a numeric score",
    ):
        metric.evaluate(example={}, final_response="", candidate={})


def test_custom_local_metric_exception_reports_function_and_path(tmp_path):
    metric_file = _write_metric(
        tmp_path,
        """
def score_answer(example, final_response, context):
    raise RuntimeError("boom")
""",
    )
    metric = resolve_metric(CustomLocalMetricRef(path=metric_file, function="score_answer"))

    with pytest.raises(MetricExecutionError) as exc_info:
        metric.evaluate(example={}, final_response="", candidate={})

    message = str(exc_info.value)
    assert "score_answer" in message
    assert str(metric_file) in message
    assert "boom" in message


def _write_metric(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "custom_metric.py"
    path.write_text(source.strip())
    return path
