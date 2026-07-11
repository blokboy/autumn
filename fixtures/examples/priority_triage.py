"""Offline GEPA example: triage incidents into priority labels.

The seed prompt gives a prose summary. GEPA should discover that the evaluator
expects one strict label from a small priority vocabulary.
"""

from __future__ import annotations

from typing import Any

import gepa
from gepa.adapters.default_adapter.default_adapter import EvaluationResult

TRAINSET: list[dict[str, str]] = [
    {"input": "all users cannot complete checkout", "answer": "p0"},
    {"input": "admin export is slow but still finishes", "answer": "p2"},
    {"input": "signup is down for new accounts", "answer": "p0"},
    {"input": "typo in a settings tooltip", "answer": "p3"},
    {"input": "webhook delivery is delayed by ten minutes", "answer": "p1"},
    {"input": "one customer's avatar is not loading", "answer": "p2"},
]

SEED_CANDIDATE: dict[str, str] = {
    "system_prompt": "Explain how serious the incident sounds.",
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
    if "all users" in text or "down" in text or "cannot complete" in text:
        priority = "p0"
    elif "delayed" in text or "webhook" in text:
        priority = "p1"
    elif "slow" in text or "one customer" in text or "avatar" in text:
        priority = "p2"
    else:
        priority = "p3"

    instruction = system_text.lower()
    if "p0" in instruction and "p1" in instruction and "p2" in instruction and "p3" in instruction:
        return priority
    return f"This looks like a {priority.upper()} incident."


def evaluator(data: dict[str, str], response: str) -> EvaluationResult:
    expected = data["answer"]
    actual = response.strip().lower()
    if actual == expected:
        return EvaluationResult(score=1.0, feedback=f"Correct priority: {expected}.")
    return EvaluationResult(
        score=0.0,
        feedback=f"Expected exactly {expected!r}, but got {actual!r}. Return only p0, p1, p2, or p3.",
    )


def reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
    return (
        "```\n"
        "Classify the incident and return exactly one lowercase priority label: "
        "p0, p1, p2, or p3. Use p0 for outages or blocked core flows, p1 for "
        "important delayed systems, p2 for degraded or single-customer issues, "
        "and p3 for cosmetic or text-only problems. Return no prose.\n"
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
    print("Best priority triage prompt:")
    for component, text in result.best_candidate.items():
        print(f"  {component}: {text!r}")


if __name__ == "__main__":
    main()
