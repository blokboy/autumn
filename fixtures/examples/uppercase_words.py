"""Offline GEPA example: transform words into uppercase.

This mirrors the smallest possible prompt-optimization loop: a seed prompt,
a deterministic task model, an exact-match evaluator, and a deterministic
reflection model. It is intentionally tiny so a demo run finishes quickly.
"""

from __future__ import annotations

from typing import Any

import gepa
from gepa.adapters.default_adapter.default_adapter import EvaluationResult

TRAINSET: list[dict[str, str]] = [
    {"input": "autumn", "answer": "AUTUMN"},
    {"input": "gepa", "answer": "GEPA"},
    {"input": "dashboard", "answer": "DASHBOARD"},
    {"input": "prompt", "answer": "PROMPT"},
    {"input": "optimize", "answer": "OPTIMIZE"},
]

SEED_CANDIDATE: dict[str, str] = {
    "system_prompt": "Repeat the user's word exactly as written.",
}


def task_lm(messages: list[dict[str, Any]]) -> str:
    system_text = ""
    user_text = ""
    for message in messages:
        if message.get("role") == "system":
            system_text = message.get("content", "")
        elif message.get("role") == "user":
            user_text = message.get("content", "")

    instruction = system_text.lower()
    if "uppercase" in instruction or "upper case" in instruction or "capital" in instruction:
        return user_text.upper()
    return user_text


def evaluator(data: dict[str, str], response: str) -> EvaluationResult:
    expected = data["answer"]
    actual = response.strip()
    if actual == expected:
        return EvaluationResult(score=1.0, feedback=f"Correctly returned {expected}.")
    return EvaluationResult(
        score=0.0,
        feedback=f"Expected {expected!r}, but got {actual!r}. Convert the word to uppercase.",
    )


def reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
    return (
        "```\n"
        "Convert the user's word to uppercase letters and return only the transformed word. "
        "Do not add punctuation or explanations.\n"
        "```"
    )


def main() -> None:
    result = gepa.optimize(
        seed_candidate=SEED_CANDIDATE,
        trainset=TRAINSET,
        task_lm=task_lm,
        evaluator=evaluator,
        reflection_lm=reflection_lm,
        max_metric_calls=15,
        display_progress_bar=False,
    )
    print("Best uppercase prompt:")
    for component, text in result.best_candidate.items():
        print(f"  {component}: {text!r}")


if __name__ == "__main__":
    main()
