"""Tests for the OpenAI catalog-visibility policy (gated on OPENAI_API_KEY)."""

import catalog, credentials, openai_policy


def test_build_policy_signs_in_when_openai_api_key_is_set(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    policy = openai_policy.build_policy()

    assert len(policy.provider_accounts) == 1
    account = policy.provider_accounts[0]
    assert account.provider == "openai"
    assert account.is_signed_in is True

    names = {model.name for model in policy.provider_models}
    assert names == {"gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"}
    for model in policy.provider_models:
        assert model.provider == "openai"
        assert model.account_id == account.account_id
        assert model.is_default is False

    entries = catalog.build_entries(tmp_path / "models", policy=policy)
    assert {entry.name for entry in entries} == names
    assert all(entry.group == "openai" for entry in entries)


def test_build_policy_is_not_signed_in_when_openai_api_key_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    policy = openai_policy.build_policy()

    assert policy.provider_accounts[0].is_signed_in is False

    entries = catalog.build_entries(tmp_path / "models", policy=policy)
    assert entries == []


def test_build_policy_is_not_signed_in_when_openai_api_key_is_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "")

    policy = openai_policy.build_policy()

    assert policy.provider_accounts[0].is_signed_in is False

    entries = catalog.build_entries(tmp_path / "models", policy=policy)
    assert entries == []


def test_build_policy_signs_in_from_a_stored_key_with_no_env_var(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    credentials.set_key("openai", "stored-key")

    policy = openai_policy.build_policy()

    assert policy.provider_accounts[0].is_signed_in is True
