"""Filesystem path helpers for locating and naming run directories."""

import os
from datetime import datetime
from pathlib import Path


def default_runs_root() -> Path:
    """XDG-style: $XDG_DATA_HOME/autumn/runs, falling back to ~/.local/share/autumn/runs."""
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        base = Path(xdg_data_home)
    else:
        base = Path.home() / ".local" / "share"
    return base / "autumn" / "runs"


def derive_run_name(script_path: Path) -> str:
    """<script-stem>-<timestamp>, e.g. 'my_script-20260709T143200'. Timestamp format: %Y%m%dT%H%M%S (filesystem-safe, no colons)."""
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{script_path.stem}-{timestamp}"
