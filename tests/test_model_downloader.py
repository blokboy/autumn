"""Behavioral tests for downloading curated models into the local catalog,
including checksum verification and resumable downloads."""

import hashlib
from pathlib import Path

import httpx
import pytest

import local_models, model_downloader
from curated_models import CuratedModel

_FAKE_BYTES = b"fake gguf bytes"
_FAKE_SHA256 = hashlib.sha256(_FAKE_BYTES).hexdigest()

_ENTRY = CuratedModel(
    name="Tiny Test Model",
    vendor="test",
    repo="test-org/tiny-model-GGUF",
    filename="tiny-model.Q4_K_M.gguf",
    approx_size_gb=0.1,
    sha256=_FAKE_SHA256,
    context_window=2048,
)


def _fake_download_file(url: str, destination: Path, on_progress) -> None:
    assert url == "https://huggingface.co/test-org/tiny-model-GGUF/resolve/main/tiny-model.Q4_K_M.gguf"
    destination.write_bytes(_FAKE_BYTES)
    if on_progress is not None:
        on_progress(len(_FAKE_BYTES), len(_FAKE_BYTES))


# ---------------------------------------------------------------------------
# download_and_install: happy path + existing scratch-file-cleanup behavior
# ---------------------------------------------------------------------------


def test_download_and_install_records_model_in_catalog(tmp_path):
    catalog_root = tmp_path / "models"

    installed = model_downloader.download_and_install(
        catalog_root, _ENTRY, download_file_fn=_fake_download_file
    )

    assert installed.name == "Tiny Test Model"
    assert installed.backend == "llama.cpp"
    assert installed.context_window == 2048
    assert installed.is_default
    assert installed.path.read_bytes() == _FAKE_BYTES
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

    assert progress_calls == [(len(_FAKE_BYTES), len(_FAKE_BYTES))]


def test_download_and_install_leaves_partial_scratch_file_on_download_failure(tmp_path):
    """A transient download error (network reset, etc.) should NOT wipe out
    whatever was already written to the scratch path -- a later retry needs
    it there to resume from, rather than restarting a multi-GB download
    from byte 0."""
    catalog_root = tmp_path / "models"

    def _failing_download(url: str, destination: Path, on_progress) -> None:
        destination.write_bytes(b"partial")
        raise OSError("connection reset")

    with pytest.raises(OSError):
        model_downloader.download_and_install(catalog_root, _ENTRY, download_file_fn=_failing_download)

    scratch_path = catalog_root / ".downloads" / _ENTRY.filename
    assert scratch_path.exists()
    assert scratch_path.read_bytes() == b"partial"
    assert local_models.list_models(catalog_root) == []


# ---------------------------------------------------------------------------
# download_and_install: checksum verification
# ---------------------------------------------------------------------------


def test_checksum_mismatch_error_is_an_os_error():
    # ModelDownloadScreen._on_failure only catches (httpx.HTTPError, OSError);
    # this must hold for the screen's error handling to cover it too.
    assert issubclass(model_downloader.ChecksumMismatchError, OSError)


def test_download_and_install_checksum_match_installs_normally(tmp_path):
    catalog_root = tmp_path / "models"

    installed = model_downloader.download_and_install(
        catalog_root, _ENTRY, download_file_fn=_fake_download_file
    )

    assert installed.path.read_bytes() == _FAKE_BYTES
    assert local_models.list_models(catalog_root) == [installed]


def test_download_and_install_checksum_mismatch_fails_without_installing(tmp_path):
    catalog_root = tmp_path / "models"

    def _corrupt_download(url: str, destination: Path, on_progress) -> None:
        destination.write_bytes(b"corrupted bytes, not what we expected")

    with pytest.raises(model_downloader.ChecksumMismatchError):
        model_downloader.download_and_install(catalog_root, _ENTRY, download_file_fn=_corrupt_download)

    assert local_models.list_models(catalog_root) == []
    # A checksum mismatch means the bytes on disk are simply wrong -- there's
    # nothing worth resuming, so the corrupt scratch file is discarded to
    # force a clean re-download next time.
    scratch_path = catalog_root / ".downloads" / _ENTRY.filename
    assert not scratch_path.exists()


def test_download_and_install_checksum_mismatch_message_names_the_file(tmp_path):
    catalog_root = tmp_path / "models"

    def _corrupt_download(url: str, destination: Path, on_progress) -> None:
        destination.write_bytes(b"nope")

    with pytest.raises(model_downloader.ChecksumMismatchError, match=_ENTRY.filename):
        model_downloader.download_and_install(catalog_root, _ENTRY, download_file_fn=_corrupt_download)


def test_download_and_install_survives_install_failure_for_a_verified_scratch_file(tmp_path, monkeypatch):
    """If install_model itself fails after a successful, checksum-verified
    download, the scratch file should stick around -- a retry shouldn't have
    to re-download or re-verify a file we already know is good."""
    catalog_root = tmp_path / "models"

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(local_models, "install_model", _boom)

    with pytest.raises(OSError):
        model_downloader.download_and_install(catalog_root, _ENTRY, download_file_fn=_fake_download_file)

    scratch_path = catalog_root / ".downloads" / _ENTRY.filename
    assert scratch_path.exists()
    assert scratch_path.read_bytes() == _FAKE_BYTES


# ---------------------------------------------------------------------------
# download_file: resumable downloads over HTTP Range requests
#
# These monkeypatch httpx.stream (rather than injecting a fake
# download_file_fn) so the real Range/resume logic inside download_file
# itself gets exercised, without making a real network call.
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    def __init__(self, status_code: int, headers: dict, body: bytes):
        self.status_code = status_code
        self.headers = headers
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://huggingface.co/fake")
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )

    def iter_bytes(self):
        chunk_size = 4
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse):
        self._response = response

    def __enter__(self) -> _FakeStreamResponse:
        return self._response

    def __exit__(self, *exc_info) -> bool:
        return False


def _range_start(headers: dict | None) -> int | None:
    range_header = (headers or {}).get("Range")
    if range_header is None:
        return None
    # download_file always sends "bytes=<n>-"
    return int(range_header.removeprefix("bytes=").rstrip("-"))


def _make_fake_stream(full_content: bytes, *, support_range: bool = True, requests: list | None = None):
    """Simulates a range-aware (or not) HTTP server serving `full_content`,
    recording each request's Range header (or None) into `requests` if
    given so tests can assert on what download_file actually sent."""

    def _fake_stream(method: str, url: str, *, follow_redirects: bool, timeout, headers=None):
        assert method == "GET"
        start = _range_start(headers)
        if requests is not None:
            requests.append(start)

        if start is not None and support_range:
            if start >= len(full_content):
                return _FakeStreamContext(_FakeStreamResponse(416, {}, b""))
            body = full_content[start:]
            return _FakeStreamContext(
                _FakeStreamResponse(206, {"content-length": str(len(body))}, body)
            )

        return _FakeStreamContext(
            _FakeStreamResponse(200, {"content-length": str(len(full_content))}, full_content)
        )

    return _fake_stream


def test_download_file_fresh_download_sends_no_range_header(tmp_path, monkeypatch):
    full_content = b"0123456789" * 5
    requests: list = []
    monkeypatch.setattr(httpx, "stream", _make_fake_stream(full_content, requests=requests))

    destination = tmp_path / "model.gguf"
    model_downloader.download_file("https://huggingface.co/fake", destination)

    assert requests == [None]
    assert destination.read_bytes() == full_content


def test_download_file_resumes_partial_file_via_range_request(tmp_path, monkeypatch):
    full_content = b"0123456789" * 5
    split = 17
    destination = tmp_path / "model.gguf"
    destination.write_bytes(full_content[:split])

    requests: list = []
    monkeypatch.setattr(httpx, "stream", _make_fake_stream(full_content, requests=requests))

    progress_calls = []
    model_downloader.download_file(
        "https://huggingface.co/fake",
        destination,
        on_progress=lambda downloaded, total: progress_calls.append((downloaded, total)),
    )

    # A Range request was issued for exactly the missing tail...
    assert requests == [split]
    # ...and the file ends up byte-identical to a fresh full download.
    assert destination.read_bytes() == full_content
    # Progress accounting includes the bytes that were already on disk.
    assert progress_calls[-1] == (len(full_content), len(full_content))
    assert all(downloaded >= split for downloaded, _ in progress_calls)


def test_download_file_falls_back_to_full_restart_when_server_ignores_range(tmp_path, monkeypatch):
    full_content = b"0123456789" * 5
    destination = tmp_path / "model.gguf"
    destination.write_bytes(b"stale wrong bytes")

    monkeypatch.setattr(httpx, "stream", _make_fake_stream(full_content, support_range=False))

    model_downloader.download_file("https://huggingface.co/fake", destination)

    # Restarted cleanly -- no leftover stale bytes prepended/mixed in.
    assert destination.read_bytes() == full_content


def test_download_file_treats_already_complete_range_as_done(tmp_path, monkeypatch):
    full_content = b"0123456789" * 5
    destination = tmp_path / "model.gguf"
    destination.write_bytes(full_content)

    requests: list = []
    monkeypatch.setattr(httpx, "stream", _make_fake_stream(full_content, requests=requests))

    model_downloader.download_file("https://huggingface.co/fake", destination)

    assert requests == [len(full_content)]
    assert destination.read_bytes() == full_content


# ---------------------------------------------------------------------------
# Resume + checksum verification composing correctly end-to-end
# ---------------------------------------------------------------------------


def test_resumed_download_still_passes_checksum_verification(tmp_path, monkeypatch):
    """A download interrupted partway through, then resumed on a later call
    via download_file's real Range logic, must still checksum-match the
    same as an uninterrupted single-shot download would."""
    full_content = b"the quick brown fox jumps over the lazy dog " * 200
    digest = hashlib.sha256(full_content).hexdigest()
    entry = CuratedModel(
        name="Resumable Test Model",
        vendor="test",
        repo="test-org/resumable-model-GGUF",
        filename="resumable-model.Q4_K_M.gguf",
        approx_size_gb=0.1,
        sha256=digest,
        context_window=4096,
    )
    catalog_root = tmp_path / "models"
    scratch_path = catalog_root / ".downloads" / entry.filename

    # First attempt: interrupted partway through, leaving a partial file.
    split = 123

    def _interrupted_download(url: str, destination: Path, on_progress) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(full_content[:split])
        raise OSError("connection reset")

    with pytest.raises(OSError):
        model_downloader.download_and_install(catalog_root, entry, download_file_fn=_interrupted_download)

    assert scratch_path.read_bytes() == full_content[:split]

    # Second attempt: uses the real download_file (not a fake), so its
    # Range-resume logic actually runs against the partial file left above.
    requests: list = []
    monkeypatch.setattr(httpx, "stream", _make_fake_stream(full_content, requests=requests))

    installed = model_downloader.download_and_install(catalog_root, entry)

    assert requests == [split]  # resumed, did not restart from byte 0
    assert installed.path.read_bytes() == full_content
    assert local_models.list_models(catalog_root) == [installed]
    assert not scratch_path.exists()
