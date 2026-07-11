"""Tests for local provider API key storage."""

import credentials


def test_set_and_get_key_round_trips():
    credentials.set_key("groq", "gsk_abc123")

    assert credentials.get_key("groq") == "gsk_abc123"


def test_get_key_returns_none_when_nothing_stored():
    assert credentials.get_key("groq") is None


def test_set_key_overwrites_existing_value():
    credentials.set_key("groq", "gsk_first")
    credentials.set_key("groq", "gsk_second")

    assert credentials.get_key("groq") == "gsk_second"


def test_set_key_persists_multiple_providers_independently():
    credentials.set_key("groq", "gsk_abc")
    credentials.set_key("anthropic", "sk-ant-xyz")

    assert credentials.get_key("groq") == "gsk_abc"
    assert credentials.get_key("anthropic") == "sk-ant-xyz"
    assert credentials.configured_providers() == {"groq", "anthropic"}


def test_remove_key_returns_true_and_deletes_when_present():
    credentials.set_key("groq", "gsk_abc")

    assert credentials.remove_key("groq") is True
    assert credentials.get_key("groq") is None


def test_remove_key_returns_false_when_nothing_stored():
    assert credentials.remove_key("groq") is False


def test_credentials_file_is_owner_only_permissions():
    import stat

    credentials.set_key("groq", "gsk_abc")

    mode = stat.S_IMODE(credentials._path().stat().st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_resolve_key_prefers_stored_over_env_var(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    credentials.set_key("groq", "from-store")

    assert credentials.resolve_key("groq", "GROQ_API_KEY") == "from-store"


def test_resolve_key_falls_back_to_env_var_when_nothing_stored(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-env")

    assert credentials.resolve_key("groq", "GROQ_API_KEY") == "from-env"


def test_resolve_key_returns_none_when_neither_source_has_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert credentials.resolve_key("groq", "GROQ_API_KEY") is None


def test_resolve_key_prefers_stored_over_env_var_for_tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    credentials.set_key("tavily", "from-store")

    assert credentials.resolve_key("tavily", "TAVILY_API_KEY") == "from-store"
