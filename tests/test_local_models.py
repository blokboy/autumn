"""Behavioral tests for Autumn's managed local model catalog."""

import json

from autumn import local_models
from autumn.models import LocalModel


def test_install_model_copies_file_and_records_metadata(tmp_path):
    source = tmp_path / "tiny.gguf"
    source.write_bytes(b"fake gguf")
    catalog_root = tmp_path / "models"

    installed = local_models.install_model(
        catalog_root,
        name="tiny",
        source_path=source,
        backend="llama.cpp",
        context_window=2048,
    )

    assert installed == LocalModel(
        name="tiny",
        backend="llama.cpp",
        path=catalog_root / "tiny" / "tiny.gguf",
        context_window=2048,
        is_default=True,
    )
    assert installed.path.read_bytes() == b"fake gguf"
    manifest = json.loads((catalog_root / "manifest.json").read_text())
    assert manifest["models"][0]["name"] == "tiny"


def test_installing_second_model_keeps_existing_default(tmp_path):
    first = tmp_path / "first.gguf"
    first.write_bytes(b"first")
    second = tmp_path / "second.gguf"
    second.write_bytes(b"second")
    catalog_root = tmp_path / "models"

    local_models.install_model(catalog_root, name="first", source_path=first)
    local_models.install_model(catalog_root, name="second", source_path=second)

    assert local_models.list_models(catalog_root) == [
        LocalModel(
            name="first",
            backend="llama.cpp",
            path=catalog_root / "first" / "first.gguf",
            context_window=None,
            is_default=True,
        ),
        LocalModel(
            name="second",
            backend="llama.cpp",
            path=catalog_root / "second" / "second.gguf",
            context_window=None,
            is_default=False,
        ),
    ]


def test_set_default_model_selects_installed_model(tmp_path):
    first = tmp_path / "first.gguf"
    first.write_bytes(b"first")
    second = tmp_path / "second.gguf"
    second.write_bytes(b"second")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="first", source_path=first)
    local_models.install_model(catalog_root, name="second", source_path=second)

    selected = local_models.set_default(catalog_root, "second")

    assert selected.name == "second"
    assert selected.is_default
    assert [model.name for model in local_models.list_models(catalog_root) if model.is_default] == [
        "second"
    ]


def test_remove_model_deletes_file_and_promotes_next_default(tmp_path):
    first = tmp_path / "first.gguf"
    first.write_bytes(b"first")
    second = tmp_path / "second.gguf"
    second.write_bytes(b"second")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="first", source_path=first)
    local_models.install_model(catalog_root, name="second", source_path=second)

    local_models.remove_model(catalog_root, "first")

    models = local_models.list_models(catalog_root)
    assert models == [
        LocalModel(
            name="second",
            backend="llama.cpp",
            path=catalog_root / "second" / "second.gguf",
            context_window=None,
            is_default=True,
        )
    ]
    assert not (catalog_root / "first").exists()


def test_get_default_model_returns_none_when_catalog_is_empty(tmp_path):
    assert local_models.get_default(tmp_path / "models") is None
