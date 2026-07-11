"""Offline GEPA example: create URL-safe product slugs.

The seed prompt preserves title casing and spaces. GEPA should discover the
expected lowercase hyphenated slug format.
"""

from __future__ import annotations

import re
from typing import Any

import gepa
from gepa.adapters.default_adapter.default_adapter import EvaluationResult

TRAINSET: list[dict[str, str]] = [
    {"input": "Autumn Dashboard Pro", "answer": "autumn-dashboard-pro"},
    {"input": "GEPA Run Monitor", "answer": "gepa-run-monitor"},
    {"input": "Prompt QA Toolkit", "answer": "prompt-qa-toolkit"},
    {"input": "Model Router 2", "answer": "model-router-2"},
    {"input": "Live Trace Viewer", "answer": "live-trace-viewer"},
]

SEED_CANDIDATE: dict[str, str] = {
    "system_prompt": "Rewrite the product name so it is easy to read.",
}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-+", "-", slug)


def task_lm(messages: list[dict[str, Any]]) -> str:
    system_text = ""
    user_text = ""
    for message in messages:
        if message.get("role") == "system":
            system_text = message.get("content", "")
        elif message.get("role") == "user":
            user_text = message.get("content", "")

    instruction = system_text.lower()
    if "slug" in instruction or "hyphen" in instruction or "url" in instruction:
        return _slugify(user_text)
    return user_text


def evaluator(data: dict[str, str], response: str) -> EvaluationResult:
    expected = data["answer"]
    actual = response.strip()
    if actual == expected:
        return EvaluationResult(score=1.0, feedback=f"Correct slug: {expected}.")
    return EvaluationResult(
        score=0.0,
        feedback=(
            f"Expected {expected!r}, but got {actual!r}. "
            "Use lowercase words separated by single hyphens."
        ),
    )


def reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
    return (
        "```\n"
        "Convert the product name into a URL-safe slug: lowercase all letters, "
        "replace spaces and punctuation with single hyphens, keep digits, trim "
        "leading or trailing hyphens, and return only the slug.\n"
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
    print("Best slug prompt:")
    for component, text in result.best_candidate.items():
        print(f"  {component}: {text!r}")


if __name__ == "__main__":
    main()
