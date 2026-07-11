"""Tests for the always-visible, never-eligible Anthropic/OpenAI stub catalog
rows (#15)."""

import stub_providers


def test_disabled_provider_entries_covers_anthropic_and_openai_in_order():
    entries = stub_providers.disabled_provider_entries()

    groups = [entry.group for entry in entries]
    assert groups == ["Anthropic", "Anthropic", "OpenAI", "OpenAI"]


def test_disabled_provider_entries_are_all_marked_disabled_and_never_default():
    entries = stub_providers.disabled_provider_entries()

    assert entries, "expected at least one representative row per provider"
    assert all(entry.disabled is True for entry in entries)
    assert all(entry.is_default is False for entry in entries)


def test_disabled_provider_entries_has_representative_models_per_group():
    entries = stub_providers.disabled_provider_entries()

    anthropic_names = [entry.name for entry in entries if entry.group == "Anthropic"]
    openai_names = [entry.name for entry in entries if entry.group == "OpenAI"]
    assert anthropic_names
    assert openai_names
    assert all(name.startswith("claude") for name in anthropic_names)
    assert all(name.startswith("gpt") for name in openai_names)


def test_disabled_provider_entries_is_deterministic_and_side_effect_free(monkeypatch):
    """No env var is read and no network call is made -- calling this twice
    with different (irrelevant) environment state must yield identical
    results."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    first_call = stub_providers.disabled_provider_entries()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-actually-read")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-actually-read")
    second_call = stub_providers.disabled_provider_entries()

    assert first_call == second_call
