"""Shared one-shot subagent execution service."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import local_llm, model_router
from groq_runner import GroqRunner, GroqRuntimeError
from local_model_runner import LocalModelRunner, LocalModelRuntimeError
from models import ChatMessage, LocalModel, ModelChoice, PromptRoutingPolicy

SUBAGENT_SYSTEM_INSTRUCTION = (
    "You are an Autumn subagent. Answer the user's prompt directly and concisely. "
    "Use only the prompt in this invocation; do not assume parent chat context."
)


class _LocalModelRunner(Protocol):
    def generate(self, messages: list[ChatMessage], model: LocalModel) -> ChatMessage: ...


class _GroqRunner(Protocol):
    def generate(self, messages: list[ChatMessage], model_name: str) -> ChatMessage: ...


@dataclass
class SubagentCapabilities:
    """Capability profile for a subagent execution path."""

    model_reply: bool = True
    tools: bool = False
    filesystem_access: bool = False
    shell_access: bool = False
    recursive_subagents: bool = False
    gepa_run_launching: bool = False

    def enabled(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "model_reply",
                "tools",
                "filesystem_access",
                "shell_access",
                "recursive_subagents",
                "gepa_run_launching",
            )
            if getattr(self, name)
        )

    def missing_from(self, other: "SubagentCapabilities") -> tuple[str, ...]:
        return tuple(name for name in self.enabled() if name not in other.enabled())


@dataclass
class SubagentFallbackWarning:
    """Metadata the dashboard can render as a regular Autumn warning message."""

    original_model: str | None
    fallback_model: str
    reason: str
    expected_capabilities: tuple[str, ...]
    fallback_capabilities: tuple[str, ...]
    lost_capabilities: tuple[str, ...]


@dataclass
class SubagentResult:
    """Structured result for one completed subagent invocation."""

    prompt: str
    answer: str
    choice: ModelChoice
    messages: list[ChatMessage]
    interrupted: bool = False
    routed_choice: ModelChoice | None = None
    fallback_warning: SubagentFallbackWarning | None = None


def run_subagent(
    prompt: str,
    *,
    catalog_root: Path,
    is_runtime_available: model_router.RuntimeAvailability | None = None,
    policy: PromptRoutingPolicy | None = None,
    local_model_runner: _LocalModelRunner | None = None,
    groq_runner: _GroqRunner | None = None,
    expected_capabilities: SubagentCapabilities | None = None,
) -> SubagentResult:
    """Routes and runs a single blocking subagent reply.

    The chosen model is captured once before execution and reused for the
    invocation, so later catalog/default changes cannot affect this response.
    """
    choice = model_router.choose_model(
        prompt=prompt,
        catalog_root=catalog_root,
        is_runtime_available=is_runtime_available,
        policy=policy,
    )
    messages = [
        ChatMessage(role="system", text=SUBAGENT_SYSTEM_INSTRUCTION),
        ChatMessage(role="user", text=prompt),
    ]
    answer, interrupted, executed_choice = _answer_with_choice(
        choice,
        messages,
        local_model_runner=local_model_runner,
        groq_runner=groq_runner,
    )
    return SubagentResult(
        prompt=prompt,
        answer=answer,
        choice=executed_choice,
        messages=messages,
        interrupted=interrupted,
        routed_choice=choice,
        fallback_warning=_fallback_warning(
            routed_choice=choice,
            executed_choice=executed_choice,
            expected_capabilities=expected_capabilities or SubagentCapabilities(),
        ),
    )


def _fallback_warning(
    *,
    routed_choice: ModelChoice,
    executed_choice: ModelChoice,
    expected_capabilities: SubagentCapabilities,
) -> SubagentFallbackWarning | None:
    fallback_capabilities = _capabilities_for_choice(executed_choice)
    model_changed = (
        routed_choice.name != executed_choice.name or routed_choice.backend != executed_choice.backend
    )
    router_fallback = (
        executed_choice.backend == "builtin"
        and executed_choice.reason != "offline fallback"
    )
    if not model_changed and not router_fallback:
        return None

    original_model = routed_choice.name if model_changed else None
    return SubagentFallbackWarning(
        original_model=original_model,
        fallback_model=executed_choice.name,
        reason=executed_choice.reason,
        expected_capabilities=expected_capabilities.enabled(),
        fallback_capabilities=fallback_capabilities.enabled(),
        lost_capabilities=expected_capabilities.missing_from(fallback_capabilities),
    )


def _capabilities_for_choice(choice: ModelChoice) -> SubagentCapabilities:
    return SubagentCapabilities(model_reply=True)


def _answer_with_choice(
    choice: ModelChoice,
    messages: list[ChatMessage],
    *,
    local_model_runner: _LocalModelRunner | None,
    groq_runner: _GroqRunner | None,
) -> tuple[str, bool, ModelChoice]:
    if choice.backend == "llama.cpp" and choice.path is not None:
        model = LocalModel(
            name=choice.name,
            backend=choice.backend,
            path=choice.path,
            context_window=choice.context_window,
            is_default=True,
        )
        try:
            return (local_model_runner or LocalModelRunner()).generate(messages, model).text, False, choice
        except LocalModelRuntimeError as exc:
            return f"Error: {exc}", True, choice

    if choice.backend == "provider" and choice.provider == "groq":
        try:
            return (groq_runner or GroqRunner()).generate(messages, choice.name).text, False, choice
        except GroqRuntimeError as exc:
            return f"Error: {exc}", True, choice

    if choice.backend == "provider":
        fallback = ModelChoice(
            name=local_llm.OFFLINE_TINY_MODEL,
            backend="builtin",
            path=None,
            reason=f"provider {choice.name} not executable yet",
        )
        return local_llm.generate_response(messages, fallback).text, False, fallback

    return local_llm.generate_response(messages, choice).text, False, choice
