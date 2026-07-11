"""Downloads curated GGUF model files from Hugging Face straight into
Autumn's managed local model catalog."""

import hashlib
from pathlib import Path
from typing import Callable

import httpx

import local_models
from curated_models import CuratedModel
from models import LocalModel

ProgressCallback = Callable[[int, int | None], None]
DownloadFile = Callable[[str, Path, ProgressCallback | None], None]

_CHECKSUM_CHUNK_BYTES = 1024 * 1024


class ChecksumMismatchError(OSError):
    """Raised by `download_and_install` when a downloaded file's SHA-256
    doesn't match the pinned `CuratedModel.sha256`. Subclasses `OSError`
    (rather than introducing a new type callers must know about) so it's
    caught by the same `except (httpx.HTTPError, OSError)` clause
    `ModelDownloadScreen._on_failure` already uses for every other download
    failure -- no screen-level change needed."""


def _hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _sha256_file(path: Path) -> str:
    """Streams `path` through SHA-256 in fixed-size chunks so verifying a
    multi-GB model file doesn't require holding it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHECKSUM_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, on_progress: ProgressCallback | None = None) -> None:
    """Streams `url` to `destination` over HTTPS, calling
    `on_progress(downloaded_bytes, total_bytes)` after each chunk
    (`total_bytes` is `None` if the server omits Content-Length).

    Resumable: if `destination` already exists (e.g. left over from an
    interrupted previous attempt), issues a `Range: bytes=<size>-` request
    for the remaining bytes and appends to the existing file instead of
    restarting from byte 0. If the server doesn't honor the Range request
    (responds with a full `200` instead of a partial `206`), falls back to
    overwriting `destination` and downloading the whole file again. If the
    existing file already covers the whole remote object, the server
    replies `416 Range Not Satisfiable`, which is treated as "nothing left
    to fetch" rather than an error.
    """
    resume_from = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else None

    with httpx.stream("GET", url, follow_redirects=True, timeout=None, headers=headers) as response:
        if resume_from and response.status_code == 416:
            # We already have every byte the server has to offer.
            return

        resuming = bool(resume_from) and response.status_code == 206
        if resume_from and not resuming:
            # Server ignored (or can't honor) the Range request -- the only
            # correct move is a full restart, since we can't trust that the
            # bytes it's about to send line up with what's already on disk.
            resume_from = 0
        response.raise_for_status()

        content_length = response.headers.get("content-length")
        remaining_bytes = int(content_length) if content_length is not None else None
        total_bytes = (resume_from + remaining_bytes) if remaining_bytes is not None else None

        downloaded = resume_from
        with destination.open("ab" if resuming else "wb") as f:
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
    `catalog_root`, verifies it against `entry.sha256`, then installs it
    through the same `local_models.install_model` path `autumn models
    install` uses, so a picker-downloaded model is indistinguishable from a
    manually-installed one afterward.

    Failure handling is deliberately asymmetric so a multi-GB download
    doesn't have to restart from zero after a transient hiccup:

    - A network/IO error raised by `download` (e.g. connection reset)
      propagates as-is and leaves whatever was written at the scratch path
      in place, so a later retry can resume it via `download_file`'s Range
      support instead of re-downloading everything.
    - A checksum mismatch means the bytes on disk are simply wrong -- there
      is nothing to resume -- so the scratch file is deleted and
      `ChecksumMismatchError` is raised, forcing a clean full re-download on
      the next attempt.
    - The scratch file is only removed after `install_model` succeeds
      (installing copies rather than moves, matching `install_model`'s
      existing contract for a manually-supplied source file); if install
      itself fails, the verified scratch file survives so a retry can skip
      straight to re-installing without re-downloading or re-verifying the
      remote fetch.

    `download_file_fn` defaults to `download_file` (looked up here rather
    than bound as a parameter default) so tests can monkeypatch this
    module's `download_file` to avoid real network calls.
    """
    download = download_file_fn or download_file
    scratch_dir = catalog_root / ".downloads"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_path = scratch_dir / entry.filename

    download(_hf_url(entry.repo, entry.filename), scratch_path, on_progress)

    digest = _sha256_file(scratch_path)
    if digest != entry.sha256:
        scratch_path.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"{entry.filename}: checksum mismatch "
            f"(expected {entry.sha256}, got {digest}) -- download is corrupt"
        )

    installed = local_models.install_model(
        catalog_root,
        name=entry.name,
        source_path=scratch_path,
        context_window=entry.context_window,
    )
    scratch_path.unlink(missing_ok=True)
    return installed
