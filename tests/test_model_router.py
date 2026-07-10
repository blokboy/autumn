"""Tests for prompt model routing policy."""

from autumn import model_router
from autumn.models import LocalModel


def test_router_uses_builtin_offline_model_when_no_catalog_default_exists(tmp_path):
    choice = model_router.choose_model(prompt="hello", catalog_root=tmp_path / "models")

    assert choice.name == "autumn/offline-tiny"
    assert choice.backend == "builtin"
    assert choice.path is None
    assert choice.reason == "offline fallback"


def test_router_prefers_installed_default_local_model(tmp_path):
    model_path = tmp_path / "models" / "tiny" / "tiny.gguf"
    choice = model_router.choose_model(
        prompt="hello",
        catalog_root=tmp_path / "models",
        available_models=[
            LocalModel(
                name="tiny",
                backend="llama.cpp",
                path=model_path,
                context_window=2048,
                is_default=True,
            )
        ],
    )

    assert choice.name == "tiny"
    assert choice.backend == "llama.cpp"
    assert choice.path == model_path
    assert choice.reason == "installed default"
