"""Launches a user's unmodified GEPA script on a background thread.

autumn never asks the user to change their script. Instead it monkeypatches
GEPA's own entry points (`autumn.patch.apply`) so that whatever the script's
`import gepa` / `from gepa import optimize` lines pull in is already wrapped,
then runs the script itself via `runpy` so it executes exactly as it would
under `python script.py`. This module owns that sequencing plus the on-disk
bookkeeping (`autumn.pid`, `autumn_meta.json`) that `registry.py` and the
future resume keybinding depend on.
"""

import json
import os
import runpy
import threading
from datetime import datetime

from autumn import patch
from autumn.dashboard_callback import DashboardCallback
from autumn.models import LiveRunSpec


def launch(dashboard: DashboardCallback, spec: LiveRunSpec) -> threading.Thread:
    """Starts `spec.script_path` on a daemon thread with GEPA patched in first.

    Writes `autumn.pid` and `autumn_meta.json` into `spec.run_dir` before the
    thread's work begins, so a run directory is immediately recognizable by
    `registry.py`'s status inference even if the script fails instantly.

    Returns the already-started `Thread` -- a function named `launch` should
    leave nothing for the caller to remember to kick off.
    """
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "autumn.pid").write_text(str(os.getpid()))
    meta = {
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
