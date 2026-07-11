"""Behavioral tests for the `list_keys` chat tool -- mirrors `autumn keys
list`'s exact semantics (cli._keys's `list` branch): one row per known
provider, configured/not-configured only, **never** a key value."""

import json

import credentials
import list_keys


def test_list_keys_reports_not_configured_for_every_known_provider_by_default():
    result = json.loads(list_keys.run_tool({}))

    assert result == [{"provider": provider, "configured": False} for provider in credentials.KNOWN_PROVIDERS]


def test_list_keys_reports_configured_for_a_stored_key():
    credentials.set_key("groq", "gsk_abc123")

    result = json.loads(list_keys.run_tool({}))

    by_provider = {row["provider"]: row["configured"] for row in result}
    assert by_provider["groq"] is True
    assert by_provider["anthropic"] is False


def test_list_keys_reports_configured_for_an_env_var_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")

    result = json.loads(list_keys.run_tool({}))

    by_provider = {row["provider"]: row["configured"] for row in result}
    assert by_provider["tavily"] is True


def test_list_keys_never_includes_the_actual_key_value():
    credentials.set_key("groq", "gsk_super_secret_value")
    import os

    os.environ["OPENAI_API_KEY"] = "sk-also-secret"
    try:
        result = list_keys.run_tool({})
    finally:
        del os.environ["OPENAI_API_KEY"]

    assert "gsk_super_secret_value" not in result
    assert "sk-also-secret" not in result


def test_list_keys_ignores_arguments_since_schema_declares_none():
    result = json.loads(list_keys.run_tool({"anything": "goes here"}))

    assert len(result) == len(credentials.KNOWN_PROVIDERS)


def test_list_keys_tool_schema_declares_no_required_parameters():
    assert list_keys.TOOL_SCHEMA["function"]["name"] == "list_keys"
    assert list_keys.TOOL_SCHEMA["function"]["parameters"]["properties"] == {}
