"""Offline GEPA example: route terse support tickets to JSON queues.

The seed prompt returns a plain queue name. GEPA should discover that the task
needs a JSON object with a `queue` field and one of the allowed labels.
"""

from __future__ import annotations

import json
from typing import Any

import gepa
from gepa.adapters.default_adapter.default_adapter import EvaluationResult

TRAINSET: list[dict[str, str]] = [
    {"input": "card charged twice after upgrade", "answer": "billing"},
    {"input": "cannot log in with magic link", "answer": "auth"},
    {"input": "export button spins forever", "answer": "product"},
    {"input": "invoice has wrong company address", "answer": "billing"},
    {"input": "password reset email never arrives", "answer": "auth"},
    {"input": "dashboard chart is missing yesterday", "answer": "product"},
]

SEED_CANDIDATE: dict[str, str] = {
    "system_prompt": "Read the support ticket and name the team that should handle it.",
}


def task_lm(messages: list[dict[str, Any]]) -> str:
    system_text = ""
    user_text = ""
    for message in messages:
        if message.get("role") == "system":
            system_text = message.get("content", "")
        elif message.get("role") == "user":
            user_text = message.get("content", "")

    text = user_text.lower()
    if "invoice" in text or "card" in text or "charged" in text:
        queue = "billing"
    elif "login" in text or "log in" in text or "password" in text or "magic link" in text:
        queue = "auth"
    else:
        queue = "product"

    instruction = system_text.lower()
    if "json" in instruction and "queue" in instruction:
        return json.dumps({"queue": queue}, separators=(",", ":"))
    return queue


def evaluator(data: dict[str, str], response: str) -> EvaluationResult:
    expected = data["answer"]
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return EvaluationResult(
            score=0.0,
            feedback=(
                "The response must be valid JSON, for example "
                '{"queue":"billing"}, with no surrounding prose.'
            ),
        )

    actual = parsed.get("queue") if isinstance(parsed, dict) else None
    if actual == expected:
        return EvaluationResult(score=1.0, feedback=f"Correctly routed ticket to {expected}.")
    return EvaluationResult(
        score=0.0,
        feedback=f"Expected queue {expected!r}, but the JSON response contained {actual!r}.",
    )


def reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
    return (
        "```\n"
        "Classify the support ticket into exactly one queue: billing, auth, or product. "
        "Return only compact JSON with this shape: {\"queue\":\"<queue>\"}. "
        "Use billing for payments or invoices, auth for login or password problems, "
        "and product for broken features or data issues.\n"
        "```"
    )


def main() -> None:
    result = gepa.optimize(
        seed_candidate=SEED_CANDIDATE,
        trainset=TRAINSET,
        task_lm=task_lm,
        evaluator=evaluator,
        reflection_lm=reflection_lm,
        max_metric_calls=18,
        display_progress_bar=False,
    )
    print("Best ticket router:")
    for component, text in result.best_candidate.items():
        print(f"  {component}: {text!r}")


if __name__ == "__main__":
    main()
