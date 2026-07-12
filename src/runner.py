"""Launches a user's unmodified GEPA script on a background thread.

autumn never asks the user to change their script. Instead it monkeypatches
GEPA's own entry points (`patch.apply`) so that whatever the script's
`import gepa` / `from gepa import optimize` lines pull in is already wrapped,
then runs the script itself via `runpy` so it executes exactly as it would
under `python script.py`. This module owns that sequencing plus the on-disk
bookkeeping (`autumn.pid`, `autumn_meta.json`) that `registry.py` depends on
for status inference, and that `recover_launch_spec()` reads back to resume a
STOPPED/FAILED run via the `r` keybinding.
"""

import json
import os
import runpy
import threading
from datetime import datetime
from pathlib import Path

import patch
from dashboard_callback import DashboardCallback
from models import LiveRunSpec, RunKind


def launch(dashboard: DashboardCallback, spec: LiveRunSpec) -> threading.Thread:
    """Starts `spec.script_path` on a daemon thread with GEPA patched in first.

    Writes `autumn.pid` and `autumn_meta.json` into `spec.run_dir` before the
    thread's work begins, so a run directory is immediately recognizable by
    `registry.py`'s status inference even if the script fails instantly.

    Returns the already-started `Thread` -- a function named `launch` should
    leave nothing for the caller to remember to kick off.
    """
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    # A prior graceful stop (the `Q` keybinding) leaves `gepa.stop` behind --
    # GEPA's FileStopper checks for its existence but never removes it itself
    # (confirmed against source: `FileStopper.remove_stop_file` exists but is
    # never called by gepa.optimize/optimize_anything). Without clearing it
    # here, resuming (`r`) would have GEPA see the stale file on its very
    # first check and halt again immediately instead of actually resuming.
    stop_file = spec.run_dir / "gepa.stop"
    if stop_file.exists():
        stop_file.unlink()
    (spec.run_dir / "autumn.pid").write_text(str(os.getpid()))
    meta = {
        "run_kind": RunKind.SCRIPT.value,
        "script_path": str(spec.script_path),
        "run_name": spec.run_name,
        "launched_at": datetime.now().isoformat(),
    }
    (spec.run_dir / "autumn_meta.json").write_text(json.dumps(meta))

    def run() -> None:
        exc: BaseException | None = None
        try:
            # patch.apply() must complete before runpy.run_path() starts: the
            # user's script's own `import gepa` / `from gepa import optimize`
            # lines execute *during* run_path, so as long as the patch is
            # already in place before that call, the unmodified script picks
            # up the patched functions with zero changes on its end.
            patch.apply(dashboard, spec.run_dir)
            runpy.run_path(str(spec.script_path), run_name="__main__")
        except BaseException as caught:  # noqa: BLE001 - must catch SystemExit/KeyboardInterrupt too
            exc = caught
        finally:
            dashboard.mark_script_finished(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()  # "launch" implies started, not merely constructed
    return thread


def recover_launch_spec(run_dir: Path) -> LiveRunSpec | None:
    """Reconstructs the `LiveRunSpec` that originally produced `run_dir`, by
    reading back the `autumn_meta.json` this module's own `launch()` writes at
    the start of every run (`{"script_path", "run_name", "launched_at"}`).

    Meant for the resume (`r`) flow: resuming a STOPPED/FAILED historical run
    means calling `launch()` again with the same `run_dir`, but the caller only
    has the run directory, not the original script path -- this recovers it.

    Returns None if `autumn_meta.json` is missing, isn't valid JSON, or lacks a
    non-empty `script_path`, so callers can show an error instead of crashing.
    Does not check whether `script_path` still exists on disk -- that's a
    launch-time concern for `runpy`, not this function's job.
    """
    run_dir = Path(run_dir)
    try:
        with (run_dir / "autumn_meta.json").open() as f:
            meta = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    if not isinstance(meta, dict):
        return None

    script_path = meta.get("script_path")
    if not isinstance(script_path, str) or not script_path:
        return None

    run_name = meta.get("run_name")
    if not isinstance(run_name, str) or not run_name:
        return None

    return LiveRunSpec(script_path=Path(script_path), run_dir=run_dir, run_name=run_name)
