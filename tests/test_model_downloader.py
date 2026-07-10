"""Behavioral tests for downloading curated models into the local catalog."""

from pathlib import Path

from autumn import local_models, model_downloader
from autumn.curated_models import CuratedModel

_ENTRY = CuratedModel(
    name="Tiny Test Model",
    vendor="test",
    repo="test-org/tiny-model-GGUF",
    filename="tiny-model.Q4_K_M.gguf",
    approx_size_gb=0.1,
    context_window=2048,
)


def _fake_download_file(url: str, destination: Path, on_progress) -> None:
    assert url == "https://huggingface.co/test-org/tiny-model-GGUF/resolve/main/tiny-model.Q4_K_M.gguf"
    destination.write_bytes(b"fake gguf bytes")
    if on_progress is not None:
        on_progress(len(b"fake gguf bytes"), len(b"fake gguf bytes"))


def test_download_and_install_records_model_in_catalog(tmp_path):
    catalog_root = tmp_path / "models"

    installed = model_downloader.download_and_install(
        catalog_root, _ENTRY, download_file_fn=_fake_download_file
    )

    assert installed.name == "Tiny Test Model"
    assert installed.backend == "llama.cpp"
    assert installed.context_window == 2048
    assert installed.is_default
    assert installed.path.read_bytes() == b"fake gguf bytes"
    assert local_models.list_models(catalog_root) == [installed]


def test_download_and_install_cleans_up_scratch_file(tmp_path):
    catalog_root = tmp_path / "models"

    model_downloader.download_and_install(catalog_root, _ENTRY, download_file_fn=_fake_download_file)

    scratch_path = catalog_root / ".downloads" / _ENTRY.filename
    assert not scratch_path.exists()


def test_download_and_install_reports_progress(tmp_path):
    catalog_root = tmp_path / "models"
    progress_calls = []

    model_downloader.download_and_install(
        catalog_root,
        _ENTRY,
        on_progress=lambda downloaded, total: progress_calls.append((downloaded, total)),
        download_file_fn=_fake_download_file,
    )

    assert progress_calls == [(len(b"fake gguf bytes"), len(b"fake gguf bytes"))]


def test_download_and_install_cleans_up_scratch_file_even_on_failure(tmp_path):
    catalog_root = tmp_path / "models"

    def _failing_download(url: str, destination: Path, on_progress) -> None:
        destination.write_bytes(b"partial")
        raise OSError("connection reset")

    try:
        model_downloader.download_and_install(catalog_root, _ENTRY, download_file_fn=_failing_download)
    except OSError:
        pass

    scratch_path = catalog_root / ".downloads" / _ENTRY.filename
    assert not scratch_path.exists()
    assert local_models.list_models(catalog_root) == []
