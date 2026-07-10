"""A minimal, fully offline GEPA optimization script.

This is a completely ordinary GEPA user script. It does not import or know
anything about the `autumn` package -- it is meant to be executed as-is
(e.g. via `runpy`) by autumn's runner, exactly like it could be executed by
any GEPA user on their own machine with `python demo_script.py`.

The task being optimized is a toy "shout" transformation: given a word,
return its uppercase form. The seed instruction deliberately does *not*
mention uppercasing, so it starts out scoring poorly; GEPA's reflective
mutation loop is expected to discover and propose an instruction that does.

Everything here -- the "task" model being optimized (`task_lm`), the scoring
function (`evaluator`), and the reflection model (`reflection_lm`) -- is a
small deterministic Python callable. None of them make network calls or
require API keys, so the whole script runs offline in a couple of seconds.
"""

from __future__ import annotations

from typing import Any

import gepa
from gepa.adapters.default_adapter.default_adapter import EvaluationResult

# ---------------------------------------------------------------------------
# Tiny in-memory dataset: word -> its uppercase form.
# ---------------------------------------------------------------------------

TRAINSET: list[dict[str, str]] = [
    {"input": "hello", "answer": "HELLO"},
    {"input": "world", "answer": "WORLD"},
    {"input": "gepa", "answer": "GEPA"},
    {"input": "textual", "answer": "TEXTUAL"},
    {"input": "dashboard", "answer": "DASHBOARD"},
]

# The seed instruction is deliberately unhelpful: it tells the "model" to
# just repeat the input back, so it will fail almost every example and give
# GEPA's reflective mutation something obvious to fix.
SEED_CANDIDATE: dict[str, str] = {
    "system_prompt": "Repeat the user's input back to me exactly as given, unchanged.",
}


def task_lm(messages: list[dict[str, Any]]) -> str:
    """Deterministic stand-in for the system/program being optimized.

    Plays the role of the "task" model: it reads the system instruction
    (the candidate text GEPA is evolving) and the user's input, and produces
    a response by following simple, literal keyword rules instead of an
    actual LLM call. This keeps the whole pipeline offline while still
    letting the candidate's instruction text meaningfully affect the score,
    so GEPA's optimization loop has something real to do.
    """
    system_text = ""
    user_text = ""
    for message in messages:
        if message.get("role") == "system":
            system_text = message.get("content", "")
        elif message.get("role") == "user":
            user_text = message.get("content", "")

    instruction = system_text.lower()
    if "uppercase" in instruction or "upper case" in instruction or "shout" in instruction:
        return user_text.upper()
    if "lowercase" in instruction or "lower case" in instruction:
        return user_text.lower()
    if "reverse" in instruction:
        return user_text[::-1]
    # Default behavior matches the (unhelpful) seed instruction: echo input.
    return user_text


def evaluator(data: dict[str, str], response: str) -> EvaluationResult:
    """Trivial deterministic scorer: exact string match, no LLM involved."""
    expected = data["answer"]
    actual = response.strip()
    if actual == expected:
        return EvaluationResult(
            score=1.0,
            feedback=f"Correct. Input '{data['input']}' produced the expected output '{expected}'.",
        )
    return EvaluationResult(
        score=0.0,
        feedback=(
            f"Incorrect. For input '{data['input']}', expected exactly '{expected}' "
            f"but got '{actual}'. The task is to convert the input word to uppercase."
        ),
    )


def reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
    """Deterministic stand-in for the reflection LM.

    Confirmed: `reflection_lm` accepts any plain callable matching
    `def __call__(self, prompt: str | list[dict[str, Any]]) -> str`, so no
    real LLM is required here either. Ignoring the prompt content entirely
    and returning a fixed, plausible "improved" instruction is enough to
    keep this fully offline and deterministic while still exercising GEPA's
    real reflective-mutation code path.
    """
    return (
        "```\n"
        "Convert the user's input word to uppercase (all capital letters) "
        "and return only the uppercased word, with no extra words, "
        "punctuation, or explanation.\n"
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
    best = result.best_candidate
    print("Best candidate found:")
    for component, text in best.items():
        print(f"  {component}: {text!r}")


if __name__ == "__main__":
    main()
