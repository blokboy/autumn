"""Tests for the Anthropic catalog-visibility policy (gated on ANTHROPIC_API_KEY)."""

import anthropic_policy, catalog, credentials


def test_build_policy_signs_in_when_anthropic_api_key_is_set(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    policy = anthropic_policy.build_policy()

    assert len(policy.provider_accounts) == 1
    account = policy.provider_accounts[0]
    assert account.provider == "anthropic"
    assert account.is_signed_in is True

    names = {model.name for model in policy.provider_models}
    assert names == {"claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5"}
    for model in policy.provider_models:
        assert model.provider == "anthropic"
        assert model.account_id == account.account_id
        assert model.is_default is False

    entries = catalog.build_entries(tmp_path / "models", policy=policy)
    assert {entry.name for entry in entries} == names
    assert all(entry.group == "anthropic" for entry in entries)


def test_build_policy_is_not_signed_in_when_anthropic_api_key_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    policy = anthropic_policy.build_policy()

    assert policy.provider_accounts[0].is_signed_in is False

    entries = catalog.build_entries(tmp_path / "models", policy=policy)
    assert entries == []


def test_build_policy_is_not_signed_in_when_anthropic_api_key_is_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    policy = anthropic_policy.build_policy()

    assert policy.provider_accounts[0].is_signed_in is False

    entries = catalog.build_entries(tmp_path / "models", policy=policy)
    assert entries == []


def test_build_policy_signs_in_from_a_stored_key_with_no_env_var(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    credentials.set_key("anthropic", "stored-key")

    policy = anthropic_policy.build_policy()

    assert policy.provider_accounts[0].is_signed_in is True
