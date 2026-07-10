"""Process-liveness check shared by `registry.py` (`autumn.pid` -> `is_live`)
and `queue_store.py` (deciding whether a persisted queue session belongs to a
still-running `autumn` process before ever offering to resume or delete it).
"""

import os


def pid_alive(pid: int) -> bool:
    """True if `pid` refers to a live process this user can see (or owns)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by someone else -- still alive.
        return True
    except OSError:
        return False
    return True
