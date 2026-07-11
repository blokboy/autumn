"""Offline GEPA example: produce strict sentiment labels.

The seed prompt writes a sentence. The improved prompt should constrain the
program to one of three labels, which gives Autumn a compact but real run to
display.
"""

from __future__ import annotations

from typing import Any

import gepa
from gepa.adapters.default_adapter.default_adapter import EvaluationResult

TRAINSET: list[dict[str, str]] = [
    {"input": "The deploy finished cleanly and customers are happy.", "answer": "positive"},
    {"input": "The checkout flow is broken for every mobile user.", "answer": "negative"},
    {"input": "The weekly usage report is attached for review.", "answer": "neutral"},
    {"input": "Latency dropped and support tickets are down.", "answer": "positive"},
    {"input": "The import job failed again overnight.", "answer": "negative"},
    {"input": "The account was created on July 10.", "answer": "neutral"},
]

SEED_CANDIDATE: dict[str, str] = {
    "system_prompt": "Briefly describe the user's message.",
}


def task_lm(messages: list[dict[str, Any]]) -> str:
    system_text = ""
    user_text = ""
    for message in messages:
        if message.get("role") == "system":
            system_text = message.get("content", "")
        elif message.get("role") == "user":
            user_text = message.get("content", "")

    lowered = user_text.lower()
    if any(word in lowered for word in ["happy", "cleanly", "dropped", "down"]):
        label = "positive"
    elif any(word in lowered for word in ["broken", "failed", "again"]):
        label = "negative"
    else:
        label = "neutral"

    instruction = system_text.lower()
    if "positive" in instruction and "negative" in instruction and "neutral" in instruction:
        return label
    return f"This message sounds {label}."


def evaluator(data: dict[str, str], response: str) -> EvaluationResult:
    expected = data["answer"]
    actual = response.strip().lower()
    if actual == expected:
        return EvaluationResult(score=1.0, feedback=f"Correct label: {expected}.")
    return EvaluationResult(
        score=0.0,
        feedback=(
            f"Expected exactly {expected!r}, but got {actual!r}. "
            "Return only one label: positive, negative, or neutral."
        ),
    )


def reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
    return (
        "```\n"
        "Classify the user's message and return exactly one lowercase label: "
        "positive, negative, or neutral. Do not include punctuation, prose, "
        "or explanations.\n"
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
    print("Best sentiment classifier:")
    for component, text in result.best_candidate.items():
        print(f"  {component}: {text!r}")


if __name__ == "__main__":
    main()
