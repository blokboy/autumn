"""Behavioral tests for `system_tools.py`'s mutating tool implementations
(install_model, set_default_model, add_key, remove_model, remove_key) --
confirmation is not this module's concern (see `groq_runner.py`), only the
mutation itself and its human-readable confirmation message."""

import credentials
import local_models
import pytest
from curated_models import CURATED_MODELS
from system_tools import MUTATING_TOOLS, SystemToolError, run_tool


def _seed_model(catalog_root, name="my-model"):
    source = catalog_root / f"{name}-source.gguf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fake gguf")
    return local_models.install_model(catalog_root, name=name, source_path=source)


# --- install_model -----------------------------------------------------------


def test_install_model_message_names_model_and_size():
    entry = CURATED_MODELS[0]
    tool = next(t for t in MUTATING_TOOLS if t.name == "install_model")

    message = tool.confirmation_message({"model_name": entry.name})

    assert entry.name in message
    assert entry.vendor in message


def test_install_model_rejects_non_curated_name(tmp_path):
    with pytest.raises(SystemToolError):
        run_tool("install_model", {"model_name": "not-a-real-model"}, catalog_root=tmp_path)


def test_install_model_downloads_and_installs_curated_entry(tmp_path, monkeypatch):
    import model_downloader

    entry = CURATED_MODELS[0]
    installed_calls = []

    def fake_download_and_install(catalog_root, curated_entry, **kwargs):
        installed = local_models.install_model(
            catalog_root, name=curated_entry.name, source_path=_fake_source(tmp_path)
        )
        installed_calls.append(curated_entry.name)
        return installed

    monkeypatch.setattr(model_downloader, "download_and_install", fake_download_and_install)

    result = run_tool("install_model", {"model_name": entry.name}, catalog_root=tmp_path)

    assert installed_calls == [entry.name]
    assert entry.name in result
    assert local_models.get_default(tmp_path).name == entry.name


def _fake_source(tmp_path):
    source = tmp_path / "scratch-source.gguf"
    source.write_bytes(b"fake gguf")
    return source


# --- set_default_model ---------------------------------------------------


def test_set_default_model_message_names_model():
    tool = next(t for t in MUTATING_TOOLS if t.name == "set_default_model")

    assert "my-model" in tool.confirmation_message({"model_name": "my-model"})


def test_set_default_model_sets_default(tmp_path):
    _seed_model(tmp_path, "model-a")
    _seed_model(tmp_path, "model-b")

    result = run_tool("set_default_model", {"model_name": "model-b"}, catalog_root=tmp_path)

    assert "model-b" in result
    assert local_models.get_default(tmp_path).name == "model-b"


def test_set_default_model_raises_for_unknown_model(tmp_path):
    with pytest.raises(SystemToolError):
        run_tool("set_default_model", {"model_name": "does-not-exist"}, catalog_root=tmp_path)


# --- add_key ---------------------------------------------------------------


def test_add_key_message_never_includes_the_key_value():
    tool = next(t for t in MUTATING_TOOLS if t.name == "add_key")

    message = tool.confirmation_message({"provider": "groq", "api_key": "gsk_super_secret"})

    assert "groq" in message
    assert "gsk_super_secret" not in message


def test_add_key_stores_the_key(tmp_path):
    run_tool("add_key", {"provider": "tavily", "api_key": "tvly-abc"}, catalog_root=tmp_path)

    assert credentials.get_key("tavily") == "tvly-abc"


def test_add_key_rejects_unknown_provider(tmp_path):
    with pytest.raises(SystemToolError):
        run_tool("add_key", {"provider": "made-up", "api_key": "x"}, catalog_root=tmp_path)


# --- remove_model (destructive) -------------------------------------------


def test_remove_model_is_marked_destructive():
    tool = next(t for t in MUTATING_TOOLS if t.name == "remove_model")

    assert tool.destructive is True


def test_remove_model_removes_an_installed_model(tmp_path):
    _seed_model(tmp_path, "gone-soon")

    result = run_tool("remove_model", {"model_name": "gone-soon"}, catalog_root=tmp_path)

    assert "gone-soon" in result
    assert local_models.list_models(tmp_path) == []


def test_remove_model_raises_for_unknown_model(tmp_path):
    with pytest.raises(SystemToolError):
        run_tool("remove_model", {"model_name": "never-installed"}, catalog_root=tmp_path)


# --- remove_key (destructive) ---------------------------------------------


def test_remove_key_is_marked_destructive():
    tool = next(t for t in MUTATING_TOOLS if t.name == "remove_key")

    assert tool.destructive is True


def test_remove_key_removes_a_stored_key(tmp_path):
    credentials.set_key("groq", "gsk_abc")

    result = run_tool("remove_key", {"provider": "groq"}, catalog_root=tmp_path)

    assert "groq" in result
    assert credentials.get_key("groq") is None


def test_remove_key_reports_when_nothing_was_stored(tmp_path):
    result = run_tool("remove_key", {"provider": "groq"}, catalog_root=tmp_path)

    assert "no stored key" in result.lower()


def test_remove_key_rejects_unknown_provider(tmp_path):
    with pytest.raises(SystemToolError):
        run_tool("remove_key", {"provider": "made-up"}, catalog_root=tmp_path)


# --- shared -----------------------------------------------------------------


def test_run_tool_raises_for_unknown_tool_name(tmp_path):
    with pytest.raises(SystemToolError):
        run_tool("not_a_real_tool", {}, catalog_root=tmp_path)


def test_only_remove_tools_are_destructive():
    destructive_names = {tool.name for tool in MUTATING_TOOLS if tool.destructive}

    assert destructive_names == {"remove_model", "remove_key"}
