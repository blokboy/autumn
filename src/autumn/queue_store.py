"""Crash-safe, per-session persistence for `AutumnApp.pending_queue`.

Each `AutumnApp` instance owns one session file under `paths.sessions_root()`,
named by a session id generated once at construction. The file is rewritten on
every queue mutation (append or auto-advance pop) -- not just at quit -- so a
normal quit, a crash, or `kill -9` never loses queued commands. Content shape:

    {"pid": <int>, "items": [<item>, ...]}

Each item is either `{"kind": "gepa", "run_name", "run_dir", "script_path",
"dry_run"}` (a `LaunchSpec`) or `{"kind": "prompt", "text": ...}` (a stub
prompt string), mirroring the `LaunchSpec | str` shape of `pending_queue`
itself.

Scoping the file per session (rather than one shared file) is what lets two
`autumn` processes run at once without clobbering each other's queue. The
`pid` field is what makes that safe on the *read* side too: `discover_resumable`
only ever treats another session's file as leftover -- eligible to be counted,
merged, or deleted -- once `procutil.pid_alive` confirms that PID is no longer
running. A file belonging to a still-live process (this process included) is
always left untouched.

Known accepted edge case: if a process exits and its PID is later reused by an
unrelated program before the next scan, that old session file will look
"still alive" and be permanently skipped -- an orphan left on disk, never
merged or cleaned up, but never wrongly resumed or double-launched either.
Same tradeoff `registry.py` already accepts for `autumn.pid`/`is_live`.
"""

import json
import os
import uuid
from pathlib import Path

from autumn.cli import LaunchSpec
from autumn.procutil import pid_alive

Item = LaunchSpec | str


def new_session_id() -> str:
    return uuid.uuid4().hex


def session_path(sessions_root: Path, session_id: str) -> Path:
    return sessions_root / f"{session_id}.json"


def _item_to_dict(item: Item) -> dict:
    if isinstance(item, LaunchSpec):
        return {
            "kind": "gepa",
            "run_name": item.run_name,
            "run_dir": str(item.run_dir),
            "script_path": str(item.script_path),
            "dry_run": item.dry_run,
        }
    return {"kind": "prompt", "text": item}


def _item_from_dict(raw: object) -> Item | None:
    """Returns `None` for any entry that isn't a well-formed item dict, so a
    partially-corrupt file degrades to dropping just that entry rather than
    the whole session."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if kind == "gepa":
        run_name, run_dir, script_path, dry_run = (
            raw.get("run_name"),
            raw.get("run_dir"),
            raw.get("script_path"),
            raw.get("dry_run"),
        )
        if not isinstance(run_name, str) or not run_name:
            return None
        if not isinstance(run_dir, str) or not run_dir:
            return None
        if not isinstance(script_path, str) or not script_path:
            return None
        return LaunchSpec(
            run_name=run_name,
            run_dir=Path(run_dir),
            script_path=Path(script_path),
            dry_run=bool(dry_run),
        )
    if kind == "prompt":
        text = raw.get("text")
        return text if isinstance(text, str) and text else None
    return None


def persist_queue(path: Path, items: list[Item], pid: int | None = None) -> None:
    """Rewrites `path` to reflect `items` exactly, or deletes it if `items` is
    empty -- an empty queue and "no session file" are treated as equivalent
    everywhere else in this module, so callers never need to special-case
    which one they're looking at."""
    if not items:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": pid if pid is not None else os.getpid(), "items": [_item_to_dict(i) for i in items]}
    path.write_text(json.dumps(payload))


def _load_session(path: Path) -> dict | None:
    try:
        with path.open() as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    pid = payload.get("pid")
    items = payload.get("items")
    if not isinstance(pid, int) or not isinstance(items, list):
        return None
    return {"pid": pid, "items": items}


def load_queue(path: Path) -> list[Item]:
    """Reads back `path`'s items, tolerating a missing or corrupt file (empty
    list) and dropping any individual malformed entries."""
    session = _load_session(path)
    if session is None:
        return []
    parsed = [_item_from_dict(raw) for raw in session["items"]]
    return [item for item in parsed if item is not None]


def discover_resumable(sessions_root: Path, own_path: Path) -> list[Path]:
    """Other sessions' files with a non-empty queue whose writer process is no
    longer alive -- i.e. safe to offer for Resume/Start-fresh. Never includes
    `own_path`, and never includes a file whose `pid` is still running (that
    session, live or another concurrent `autumn` instance, is left completely
    untouched: not counted, not merged, not deleted)."""
    if not sessions_root.is_dir():
        return []

    resumable = []
    for path in sessions_root.glob("*.json"):
        if path == own_path:
            continue
        session = _load_session(path)
        if session is None or not session["items"]:
            continue
        if pid_alive(session["pid"]):
            continue
        resumable.append(path)
    # Oldest-touched session first, so a merged queue launches in roughly the
    # order the leftover work was originally queued.
    resumable.sort(key=lambda p: p.stat().st_mtime)
    return resumable


def load_and_merge(paths: list[Path]) -> list[Item]:
    """Concatenates the queues of `paths` (assumed already ordered by the
    caller, see `discover_resumable`) into one combined list."""
    merged: list[Item] = []
    for path in paths:
        merged.extend(load_queue(path))
    return merged


def delete_files(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
