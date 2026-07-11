"""Blocking progress screen shown after ModelPickerScreen: downloads each
chosen curated model from Hugging Face, one at a time, and installs it into
Autumn's local model catalog before continuing into the dashboard."""

import threading

import httpx
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label, ProgressBar

import model_downloader
from curated_models import CuratedModel


class ModelDownloadScreen(Screen):
    """Downloads and installs `entries` in order on a background thread (so
    the Textual event loop keeps rendering the progress bar), then hands off
    to `AutumnApp.finish_model_download` to enter the dashboard -- the same
    transition InputScreen's empty-Enter path uses. A failed download shows
    a notification and falls back to InputScreen rather than getting stuck
    on an unusable progress screen; any entries already installed before the
    failure stay installed."""

    def __init__(self, entries: list[CuratedModel], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._entries = entries
        self._current_index = 0

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
            yield Label(self._title_text(), classes="download-title", id="download-title")
            yield ProgressBar(total=100, id="download-progress")

    def _title_text(self) -> str:
        entry = self._entries[self._current_index]
        if len(self._entries) == 1:
            return f"Downloading {entry.name}..."
        return f"Downloading {entry.name}... ({self._current_index + 1} of {len(self._entries)})"

    def on_mount(self) -> None:
        threading.Thread(target=self._download_all, daemon=True).start()

    def _on_progress(self, downloaded: int, total: int | None) -> None:
        if not total:
            return
        percent = min(100, int(downloaded * 100 / total))
        self.app.call_from_thread(self._set_progress, percent)

    def _set_progress(self, percent: int) -> None:
        self.query_one("#download-progress", ProgressBar).update(progress=percent)

    def _set_title(self) -> None:
        self.query_one("#download-title", Label).update(self._title_text())

    def _download_all(self) -> None:
        for index, entry in enumerate(self._entries):
            self._current_index = index
            self.app.call_from_thread(self._set_title)
            self.app.call_from_thread(self._set_progress, 0)
            try:
                model_downloader.download_and_install(
                    self.app.model_catalog_root,
                    entry,
                    on_progress=self._on_progress,
                    download_file_fn=self.app.model_download_fn,
                )
            except (httpx.HTTPError, OSError) as exc:
                self.app.call_from_thread(self._on_failure, entry, str(exc))
                return
        self.app.call_from_thread(
            self.app.finish_model_download, focus_models_tab=len(self._entries) > 1
        )

    def _on_failure(self, entry: CuratedModel, detail: str) -> None:
        from screens.input_screen import InputScreen

        self.app.notify(f"Couldn't install {entry.name}: {detail}", severity="error")
        self.app.switch_screen(InputScreen())
