"""Offline GEPA example: normalize task updates into CSV rows.

The seed prompt writes human-readable prose. GEPA should discover that the
consumer expects exactly `owner,status` with normalized status values.
"""

from __future__ import annotations

from typing import Any

import gepa
from gepa.adapters.default_adapter.default_adapter import EvaluationResult

TRAINSET: list[dict[str, str]] = [
    {"input": "Mina finished the importer", "answer": "mina,done"},
    {"input": "Jon is blocked on API credentials", "answer": "jon,blocked"},
    {"input": "Ari is still wiring the chart", "answer": "ari,in_progress"},
    {"input": "Leah completed the docs pass", "answer": "leah,done"},
    {"input": "Noor is waiting on legal review", "answer": "noor,blocked"},
    {"input": "Sam continues the browser tests", "answer": "sam,in_progress"},
]

SEED_CANDIDATE: dict[str, str] = {
    "system_prompt": "Summarize the task update in a short sentence.",
}


def task_lm(messages: list[dict[str, Any]]) -> str:
    system_text = ""
    user_text = ""
    for message in messages:
        if message.get("role") == "system":
            system_text = message.get("content", "")
        elif message.get("role") == "user":
            user_text = message.get("content", "")

    words = user_text.lower().split()
    owner = words[0] if words else "unknown"
    if "finished" in words or "completed" in words:
        status = "done"
    elif "blocked" in words or "waiting" in words:
        status = "blocked"
    else:
        status = "in_progress"

    instruction = system_text.lower()
    if "csv" in instruction and "owner" in instruction and "status" in instruction:
        return f"{owner},{status}"
    return f"{owner.title()} is {status.replace('_', ' ')}."


def evaluator(data: dict[str, str], response: str) -> EvaluationResult:
    expected = data["answer"]
    actual = response.strip().lower()
    if actual == expected:
        return EvaluationResult(score=1.0, feedback=f"Correct CSV row: {expected}.")
    return EvaluationResult(
        score=0.0,
        feedback=(
            f"Expected exactly {expected!r}, but got {actual!r}. "
            "Return owner,status with no header or prose."
        ),
    )


def reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
    return (
        "```\n"
        "Parse the task update and return exactly one CSV row in the format "
        "owner,status. Lowercase the owner name. Use status done for finished "
        "or completed work, blocked for blocked or waiting work, and in_progress "
        "for work still being done. Do not include a header.\n"
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
    print("Best status CSV prompt:")
    for component, text in result.best_candidate.items():
        print(f"  {component}: {text!r}")


if __name__ == "__main__":
    main()
