"""Log view: live-scrolling feed of LogLine entries, colored by level."""

from rich.text import Text
from textual.widgets import RichLog

from models import DashboardState, LogLine

_LEVEL_COLORS = {
    "info": "#a9906f",
    "success": "#8faa5a",
    "warn": "#e0a94f",
    "error": "#c1542e",
}


def color_for_level(level: str) -> str:
    """Returns the hex color associated with a LogLine level."""
    return _LEVEL_COLORS.get(level, _LEVEL_COLORS["info"])


class LogView(RichLog):
    """Thin RichLog wrapper that stays in sync with DashboardState.log_lines."""

    def __init__(self, state: DashboardState, *args, **kwargs) -> None:
        kwargs.setdefault("max_lines", 2000)
        kwargs.setdefault("auto_scroll", True)
        super().__init__(*args, **kwargs)
        self.state = state
        self._rendered_count = 0

    def on_mount(self) -> None:
        self.border_title = "Log"
        self.refresh_from_state(self.state)

    def append_line(self, line: LogLine) -> None:
        self.write(Text(line.text, style=color_for_level(line.level)))

    def refresh_from_state(self, state: DashboardState) -> None:
        self.state = state
        total = len(state.log_lines)

        if total >= self._rendered_count:
            for line in list(state.log_lines)[self._rendered_count :]:
                self.append_line(line)
        else:
            self.clear()
            for line in state.log_lines:
                self.append_line(line)

        self._rendered_count = total
