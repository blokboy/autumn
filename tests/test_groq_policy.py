"""Tests for the Groq catalog-visibility policy (gated on GROQ_API_KEY)."""

import catalog, credentials, groq_policy


def test_build_policy_signs_in_when_groq_api_key_is_set(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    policy = groq_policy.build_policy()

    assert len(policy.provider_accounts) == 1
    account = policy.provider_accounts[0]
    assert account.provider == "groq"
    assert account.is_signed_in is True

    names = {model.name for model in policy.provider_models}
    assert names == {"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"}
    for model in policy.provider_models:
        assert model.provider == "groq"
        assert model.account_id == account.account_id
        assert model.is_default is False

    entries = catalog.build_entries(tmp_path / "models", policy=policy)
    assert {entry.name for entry in entries} == names
    assert all(entry.group == "groq" for entry in entries)


def test_build_policy_is_not_signed_in_when_groq_api_key_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    policy = groq_policy.build_policy()

    assert policy.provider_accounts[0].is_signed_in is False

    entries = catalog.build_entries(tmp_path / "models", policy=policy)
    assert entries == []


def test_build_policy_is_not_signed_in_when_groq_api_key_is_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "")

    policy = groq_policy.build_policy()

    assert policy.provider_accounts[0].is_signed_in is False

    entries = catalog.build_entries(tmp_path / "models", policy=policy)
    assert entries == []


def test_build_policy_signs_in_from_a_stored_key_with_no_env_var(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    credentials.set_key("groq", "stored-key")

    policy = groq_policy.build_policy()

    assert policy.provider_accounts[0].is_signed_in is True
