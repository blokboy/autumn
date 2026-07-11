"""Filesystem path helpers for locating and naming run directories."""

import os
from datetime import datetime
from pathlib import Path


def _data_root() -> Path:
    """XDG-style: $XDG_DATA_HOME/autumn, falling back to ~/.local/share/autumn."""
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        base = Path(xdg_data_home)
    else:
        base = Path.home() / ".local" / "share"
    return base / "autumn"


def default_runs_root() -> Path:
    """XDG-style: $XDG_DATA_HOME/autumn/runs, falling back to ~/.local/share/autumn/runs."""
    return _data_root() / "runs"


def sessions_root() -> Path:
    """XDG-style: $XDG_DATA_HOME/autumn/sessions, falling back to
    ~/.local/share/autumn/sessions -- where each AutumnApp instance persists its
    own pending-queue session file (see queue_store.py), scoped per-process so
    concurrent `autumn` instances never write the same file."""
    return _data_root() / "sessions"


def chat_sessions_root() -> Path:
    """XDG-style location for dashboard chat transcript session files."""
    return _data_root() / "chat-sessions"


def models_root() -> Path:
    """XDG-style location for Autumn-managed local model files."""
    return _data_root() / "models"


def credentials_path() -> Path:
    """XDG-style location for provider API keys stored via `autumn keys add`."""
    return _data_root() / "credentials.json"


def config_path() -> Path:
    """XDG-style location for user preferences stored via `autumn config`."""
    return _data_root() / "config.json"


def derive_run_name(script_path: Path) -> str:
    """<script-stem>-<timestamp>, e.g. 'my_script-20260709T143200'. Timestamp format: %Y%m%dT%H%M%S (filesystem-safe, no colons)."""
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{script_path.stem}-{timestamp}"
