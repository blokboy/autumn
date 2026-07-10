"""Blocking progress screen shown after ModelPickerScreen: downloads the
chosen curated model from Hugging Face and installs it into Autumn's local
model catalog before continuing into the dashboard."""

import threading

import httpx
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label, ProgressBar

from autumn import model_downloader
from autumn.curated_models import CuratedModel


class ModelDownloadScreen(Screen):
    """Downloads and installs `entry` on a background thread (so the
    Textual event loop keeps rendering the progress bar), then hands off to
    `AutumnApp.finish_model_download` to enter the dashboard -- the same
    transition InputScreen's empty-Enter path uses. A failed download shows
    a notification and falls back to InputScreen rather than getting stuck
    on an unusable progress screen."""

    def __init__(self, entry: CuratedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._entry = entry

    DEFAULT_CSS = """
    ModelDownloadScreen {
        align: center middle;
    }
    ModelDownloadScreen #model-download-frame {
        width: auto;
        height: auto;
        align: center middle;
    }
    ModelDownloadScreen .download-title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        margin-bottom: 1;
    }
    ModelDownloadScreen ProgressBar {
        width: 50;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="model-download-frame"):
            yield Label(f"Downloading {self._entry.name}...", classes="download-title")
            yield ProgressBar(total=100, id="download-progress")

    def on_mount(self) -> None:
        threading.Thread(target=self._download, daemon=True).start()

    def _on_progress(self, downloaded: int, total: int | None) -> None:
        if not total:
            return
        percent = min(100, int(downloaded * 100 / total))
        self.app.call_from_thread(self._set_progress, percent)

    def _set_progress(self, percent: int) -> None:
        self.query_one("#download-progress", ProgressBar).update(progress=percent)

    def _download(self) -> None:
        try:
            model_downloader.download_and_install(
                self.app.model_catalog_root,
                self._entry,
                on_progress=self._on_progress,
                download_file_fn=self.app.model_download_fn,
            )
        except (httpx.HTTPError, OSError) as exc:
            self.app.call_from_thread(self._on_failure, str(exc))
            return
        self.app.call_from_thread(self.app.finish_model_download)

    def _on_failure(self, detail: str) -> None:
        from autumn.screens.input_screen import InputScreen

        self.app.notify(f"Couldn't install {self._entry.name}: {detail}", severity="error")
        self.app.switch_screen(InputScreen())
