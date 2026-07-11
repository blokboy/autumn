"""Behavioral tests for issue #20: entering the dashboard immediately after
picking a first-run model, with the download completing in the background
instead of blocking on a full-screen progress bar.

`_make_blocking_download` gives tests control over exactly when a "download"
finishes -- the fake `model_download_fn` blocks on a `threading.Event` the
test controls, so assertions about "dashboard entered / prompt answered
*before* the download completes" are deterministic rather than racing a real
background thread."""

import dataclasses
import hashlib
import threading
from pathlib import Path

import pytest
from textual.widgets import Static

import local_models
import model_router
import screens.model_picker_screen as model_picker_screen
from app import AutumnApp
from curated_models import CURATED_MODELS
from screens.dashboard_screen import DashboardScreen
from screens.model_picker_screen import ModelPickerScreen

_FAKE_CONTENT = b"fake gguf bytes"
_FAKE_CONTENT_SHA256 = hashlib.sha256(_FAKE_CONTENT).hexdigest()


@pytest.fixture(autouse=True)
def _patch_curated_models_checksum(monkeypatch):
    """These tests' fake downloads always write `_FAKE_CONTENT` regardless of
    which curated model was picked (see test_model_picker.py, which
    originated this pattern) -- point each curated model's pinned checksum
    at that fixed payload's checksum so #23's real checksum verification in
    `download_and_install` doesn't reject it here; this file exercises the
    background-download lifecycle, not checksum verification itself."""
    patched = [dataclasses.replace(entry, sha256=_FAKE_CONTENT_SHA256) for entry in CURATED_MODELS]
    monkeypatch.setattr(model_picker_screen, "CURATED_MODELS", patched)


def _make_blocking_download(content: bytes, release: threading.Event):
    def _download(url: str, destination: Path, on_progress) -> None:
        release.wait()
        destination.write_bytes(content)
        if on_progress is not None:
            on_progress(len(content), len(content))

    return _download


def _fake_download_file(url: str, destination: Path, on_progress) -> None:
    destination.write_bytes(b"fake gguf bytes")
    if on_progress is not None:
        on_progress(len(b"fake gguf bytes"), len(b"fake gguf bytes"))


async def test_picking_a_model_enters_dashboard_without_waiting_for_download(tmp_path):
    catalog_root = tmp_path / "models"
    release = threading.Event()
    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=catalog_root,
        model_download_fn=_make_blocking_download(b"fake gguf bytes", release),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ModelPickerScreen)

        await pilot.press("space")
        await pilot.press("enter")
        await pilot.pause()

        # Straight into the dashboard -- the fake download is still blocked
        # on `release`, so this only passes if nothing waited for it.
        assert isinstance(app.screen, DashboardScreen)

        # A "downloading" placeholder already occupies the catalog slot: the
        # picked model is registered (and marked default, matching the
        # empty-catalog rule) before a single byte has actually downloaded.
        models = local_models.list_models(catalog_root)
        assert len(models) == 1
        assert models[0].name == CURATED_MODELS[0].name
        assert models[0].status == "downloading"
        assert models[0].is_default

        # A visible, non-blocking progress indicator is showing in the
        # command bar.
        status = app.screen.query_one("#download-status", Static).content
        assert CURATED_MODELS[0].name in str(status)

        release.set()
        await pilot.pause(0.2)

        # Once the download finishes, it becomes a normal fully-installed
        # catalog entry -- no relaunch required.
        installed = local_models.get_default(catalog_root)
        assert installed is not None
        assert installed.status == "installed"
        assert installed.name == CURATED_MODELS[0].name
        assert installed.path.read_bytes() == b"fake gguf bytes"

        # And model_router now treats it as a normal, routable default.
        choice = model_router.choose_model(
            prompt="hello",
            catalog_root=catalog_root,
            is_runtime_available=lambda model: True,
        )
        assert choice.name == CURATED_MODELS[0].name
        assert choice.backend == "llama.cpp"
        assert choice.reason == "installed default"

        # The command bar's progress line clears once done.
        status = app.screen.query_one("#download-status", Static).content
        assert str(status) == ""


async def test_prompt_answered_before_download_completes_falls_through(tmp_path):
    catalog_root = tmp_path / "models"
    release = threading.Event()
    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=catalog_root,
        chat_sessions_root=tmp_path / "chats",
        model_download_fn=_make_blocking_download(b"fake gguf bytes", release),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, DashboardScreen)

        # Answer a prompt while the picked default is still downloading --
        # choose_model must fall through to the offline fallback rather than
        # trying (and failing) to run a model with no file on disk yet.
        app.submit_command("hello there")
        await pilot.pause(0.3)

        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Fallback: autumn/offline-tiny" in str(status_text)
        assert f"{CURATED_MODELS[0].name} is still downloading" in str(status_text)

        release.set()
        await pilot.pause(0.2)
        assert local_models.get_default(catalog_root).status == "installed"


async def test_failed_background_download_notifies_without_crashing(tmp_path):
    catalog_root = tmp_path / "models"

    def _failing_download(url: str, destination: Path, on_progress) -> None:
        raise OSError("connection reset")

    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=catalog_root,
        model_download_fn=_failing_download,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")
        await pilot.press("enter")
        await pilot.pause(0.3)

        # No crash: still on the dashboard, chat/other interaction usable.
        assert isinstance(app.screen, DashboardScreen)

        notifications = list(app._notifications)
        assert any(
            f"Couldn't install {CURATED_MODELS[0].name}" in notification.message
            and notification.severity == "error"
            for notification in notifications
        )

        # The failed placeholder doesn't linger as a permanently-stuck
        # "downloading" ghost entry -- the catalog reverts to empty since
        # nothing actually finished installing.
        assert local_models.list_models(catalog_root) == []

        # The command bar's progress line clears rather than getting stuck.
        status = app.screen.query_one("#download-status", Static).content
        assert str(status) == ""


async def test_failed_download_in_a_multi_pick_stops_remaining_queue_and_keeps_prior_installs(
    tmp_path,
):
    catalog_root = tmp_path / "models"
    attempts: list[str] = []

    def _fails_on_second(url: str, destination: Path, on_progress) -> None:
        attempts.append(url)
        if len(attempts) == 1:
            destination.write_bytes(b"fake gguf bytes")
            if on_progress is not None:
                on_progress(len(b"fake gguf bytes"), len(b"fake gguf bytes"))
            return
        raise OSError("connection reset")

    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=catalog_root,
        model_download_fn=_fails_on_second,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        # Check the first two curated rows.
        await pilot.press("space")
        await pilot.press("down")
        await pilot.press("space")
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert isinstance(app.screen, DashboardScreen)

        notifications = list(app._notifications)
        assert any(
            f"Couldn't install {CURATED_MODELS[1].name}" in notification.message
            for notification in notifications
        )

        # First entry installed successfully and stays installed; the
        # second's placeholder is gone rather than stuck "downloading"
        # forever.
        remaining = local_models.list_models(catalog_root)
        assert [model.name for model in remaining] == [CURATED_MODELS[0].name]
        assert remaining[0].status == "installed"
        assert remaining[0].is_default


async def test_picking_multiple_models_focuses_models_tab_immediately(tmp_path):
    """Landing on the Models tab for an ambiguous multi-pick (#20's
    background version of the old blocking flow's same rule) doesn't need to
    wait for any download to finish -- it's decided at pick time."""
    catalog_root = tmp_path / "models"
    release = threading.Event()
    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=catalog_root,
        model_download_fn=_make_blocking_download(b"fake gguf bytes", release),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")
        await pilot.press("down")
        await pilot.press("space")
        await pilot.press("enter")
        await pilot.pause()

        from textual.widgets import TabbedContent

        assert isinstance(app.screen, DashboardScreen)
        assert app.screen.query_one(TabbedContent).active == "models-tab"

        release.set()
        await pilot.pause(0.2)
