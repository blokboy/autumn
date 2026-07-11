"""Behavioral tests for the `list_models` chat tool -- a thin wrapper around
`local_models.list_models` that must serialize the exact same JSON row shape
`autumn models list --json` prints (cli._models_as_rows)."""

import json

import local_models
import list_models


def _install(catalog_root, *, name, backend="llama.cpp", context_window=4096, source_bytes=b"fake gguf"):
    source = catalog_root / f"{name}-source.gguf"
    source.write_bytes(source_bytes)
    return local_models.install_model(
        catalog_root, name=name, source_path=source, backend=backend, context_window=context_window
    )


def test_list_models_returns_empty_array_for_empty_catalog(tmp_path):
    result = list_models.run_tool({}, catalog_root=tmp_path)

    assert json.loads(result) == []


def test_list_models_matches_local_models_list_models_output(tmp_path):
    installed = _install(tmp_path, name="llama-3.2-3b-instruct")

    result = json.loads(list_models.run_tool({}, catalog_root=tmp_path))

    assert result == [
        {
            "name": installed.name,
            "backend": installed.backend,
            "path": str(installed.path),
            "context_window": installed.context_window,
            "is_default": installed.is_default,
        }
    ]


def test_list_models_reports_multiple_models_and_default_flag(tmp_path):
    first = _install(tmp_path, name="model-a")
    _install(tmp_path, name="model-b")

    result = json.loads(list_models.run_tool({}, catalog_root=tmp_path))

    assert [row["name"] for row in result] == ["model-a", "model-b"]
    # First-installed model becomes the default (local_models.install_model's
    # behavior); list_models must reflect that, not invent its own notion.
    assert [row["is_default"] for row in result] == [True, False]
    assert first.is_default is True


def test_list_models_ignores_arguments_since_schema_declares_none(tmp_path):
    _install(tmp_path, name="model-a")

    result = list_models.run_tool({"anything": "goes here"}, catalog_root=tmp_path)

    assert json.loads(result) != []


def test_list_models_tool_schema_declares_no_required_parameters():
    assert list_models.TOOL_SCHEMA["function"]["name"] == "list_models"
    assert list_models.TOOL_SCHEMA["function"]["parameters"]["properties"] == {}
