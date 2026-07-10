"""Downloads curated GGUF model files from Hugging Face straight into
Autumn's managed local model catalog."""

from pathlib import Path
from typing import Callable

import httpx

from autumn import local_models
from autumn.curated_models import CuratedModel
from autumn.models import LocalModel

ProgressCallback = Callable[[int, int | None], None]
DownloadFile = Callable[[str, Path, ProgressCallback | None], None]


def _hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def download_file(url: str, destination: Path, on_progress: ProgressCallback | None = None) -> None:
    """Streams `url` to `destination` over HTTPS, calling
    `on_progress(downloaded_bytes, total_bytes)` after each chunk
    (`total_bytes` is `None` if the server omits Content-Length)."""
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        total_bytes = int(content_length) if content_length is not None else None
        downloaded = 0
        with destination.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
                downloaded += len(chunk)
                if on_progress is not None:
                    on_progress(downloaded, total_bytes)


def download_and_install(
    catalog_root: Path,
    entry: CuratedModel,
    *,
    on_progress: ProgressCallback | None = None,
    download_file_fn: DownloadFile | None = None,
) -> LocalModel:
    """Downloads `entry`'s GGUF file to a scratch location under
    `catalog_root`, then installs it through the same
    `local_models.install_model` path `autumn models install` uses, so a
    picker-downloaded model is indistinguishable from a manually-installed
    one afterward. The scratch copy is removed once installed (installing
    copies rather than moves, matching `install_model`'s existing contract
    for a manually-supplied source file).

    `download_file_fn` defaults to `download_file` (looked up here rather
    than bound as a parameter default) so tests can monkeypatch this
    module's `download_file` to avoid real network calls.
    """
    download = download_file_fn or download_file
    scratch_dir = catalog_root / ".downloads"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_path = scratch_dir / entry.filename

    try:
        download(_hf_url(entry.repo, entry.filename), scratch_path, on_progress)
        return local_models.install_model(
            catalog_root,
            name=entry.name,
            source_path=scratch_path,
            context_window=entry.context_window,
        )
    finally:
        scratch_path.unlink(missing_ok=True)
