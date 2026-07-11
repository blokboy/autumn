"""Run discovery: scans the runs root for past GEPA run directories, infers each
run's status from the `autumn.pid` convention, and merges in the in-process live
run (if any) -- all reading plain JSON, never unpickling `gepa_state.bin`.

On-disk files this module reads (written by GEPA's `GEPAState.save()`, or by
hand-crafted fixtures in tests):

- `candidates.json` -- a plain JSON list mirroring GEPA's `program_candidates`
  (one dict per candidate, component name -> program text). `num_candidates` is
  simply `len()` of this list. GEPA does not persist per-candidate scores here
  (those live only in the pickled `gepa_state.bin`), so this module also accepts
  an optional numeric score under any of `_SCORE_KEYS` on each candidate entry,
  for fixtures/future GEPA versions that do include one.
- `run_log.json` -- a plain JSON list mirroring GEPA's `full_program_trace`
  (one dict per iteration). `best_score` is the max numeric value found under
  any of `_SCORE_KEYS` across every entry in `run_log.json` and `candidates.json`
  combined, or `None` if neither carries one. A "terminal event" (used for
  status inference) is any entry with `event` in `_TERMINAL_EVENTS`, or a truthy
  `terminal` key.
- `autumn.pid` -- plain text, the PID of the process that launched the run
  (written by `runner.py`, introduced in a later ticket). Its absence means the
  run was never launched under autumn's PID convention.
- `gepa.stop` -- GEPA's own `FileStopper` convention: its presence signals the
  run was asked to stop gracefully.
- `autumn_status.json` -- autumn's own authoritative completion marker
  (`{"status": ..., "error"?: ...}`), written by
  `DashboardCallback.mark_script_finished()` as its last act. Needed because
  GEPA's own `run_log.json` carries no terminal marker of any kind -- the same
  per-iteration trace shape is written whether a run finishes cleanly or is
  killed mid-iteration, so without this file a completed run and a crashed one
  are indistinguishable from plain JSON alone (confirmed against a real run).
- `autumn_candidates.json` -- autumn's own per-candidate score snapshot
  (`{"best_idx", "best_score", "pareto_front": [...], "candidates": [{"idx",
  "val_score", "discovered_iteration", "parent_ids", "is_pareto_member"}]}`),
  written by `DashboardCallback` at every GEPA checkpoint (`on_state_saved`)
  and finally at `mark_script_finished()`. Needed because GEPA only persists
  per-candidate *scores* in the pickled `gepa_state.bin`, which this module
  deliberately never unpickles -- without this file, every score/best-candidate/
  Pareto-front field in the historical browser reads as empty for a real run.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from models import CandidateRow, DashboardState, RunStatus, RunSummary
from procutil import pid_alive

_SCORE_KEYS = ("val_score", "valset_score", "average_score", "best_score", "score")
_TERMINAL_EVENTS = {"on_optimization_end", "optimization_end"}


def _read_json(path: Path) -> Any:
    try:
        with path.open() as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _read_pid(run_dir: Path) -> int | None:
    pid_file = run_dir / "autumn.pid"
    try:
        text = pid_file.read_text().strip()
    except (FileNotFoundError, OSError):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _has_terminal_event(run_log: Any) -> bool:
    if not isinstance(run_log, list):
        return False
    for entry in run_log:
        if not isinstance(entry, dict):
            continue
        if entry.get("event") in _TERMINAL_EVENTS:
            return True
        if entry.get("terminal"):
            return True
    return False


def _read_candidates_snapshot(run_dir: Path) -> dict | None:
    """Reads `autumn_candidates.json` if present and roughly well-formed.
    Returns None for older runs that predate this marker (or a run that never
    reached its first checkpoint), so callers fall back to the best-effort
    `_extract_best_score`/no-metadata behavior below."""
    snapshot = _read_json(run_dir / "autumn_candidates.json")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("candidates"), list):
        return None
    return snapshot


def _extract_best_score(candidates: Any, run_log: Any) -> float | None:
    best: float | None = None
    for collection in (run_log, candidates):
        if not isinstance(collection, list):
            continue
        for entry in collection:
            if not isinstance(entry, dict):
                continue
            for key in _SCORE_KEYS:
                value = entry.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if best is None or value > best:
                        best = float(value)
    return best


_MARKER_STATUS_VALUES = {status.value: status for status in RunStatus}


def _read_status_marker(run_dir: Path) -> RunStatus | None:
    """Reads the `autumn_status.json` marker dashboard_callback.py's
    mark_script_finished() writes as its very last act. This is the
    authoritative completion signal for real runs: GEPA's own run_log.json
    carries no terminal marker of its own (the same per-iteration trace shape
    is written whether a run finishes cleanly or is killed mid-iteration), so
    without this file a completed run and a crashed one are indistinguishable
    from plain JSON alone. Returns None if the marker is absent (older runs, or
    a process killed before mark_script_finished could run) so callers can fall
    back to the best-effort heuristic below."""
    marker = _read_json(run_dir / "autumn_status.json")
    if not isinstance(marker, dict):
        return None
    return _MARKER_STATUS_VALUES.get(marker.get("status"))


def infer_status(run_dir: Path) -> RunStatus:
    """Status inference:

    alive PID -> RUNNING; dead + `gepa.stop` present -> STOPPED (checked first
    since a graceful stop still runs GEPA's engine loop to a normal exit, so it
    would otherwise look identical to a natural COMPLETED); dead + the
    `autumn_status.json` marker present -> whatever status it records; dead +
    no marker (older runs / abnormal kill) -> FAILED unless `run_log.json` shows
    a terminal event -> COMPLETED, the old best-effort fallback.
    """
    pid = _read_pid(run_dir)
    if pid is not None and pid_alive(pid):
        return RunStatus.RUNNING

    if (run_dir / "gepa.stop").exists():
        return RunStatus.STOPPED

    marker_status = _read_status_marker(run_dir)
    if marker_status is not None:
        return marker_status

    run_log = _read_json(run_dir / "run_log.json")
    if _has_terminal_event(run_log):
        return RunStatus.COMPLETED

    return RunStatus.FAILED


def _is_run_dir(run_dir: Path) -> bool:
    """A directory counts as a run directory if it has any of the files this
    module knows how to read; guards `scan()` against unrelated clutter under
    the runs root."""
    return (
        (run_dir / "candidates.json").exists()
        or (run_dir / "run_log.json").exists()
        or (run_dir / "autumn.pid").exists()
    )


def scan(runs_root: Path) -> list[RunSummary]:
    """Reads every run directory directly under `runs_root` into a `RunSummary`,
    newest-first. Reads only plain JSON -- never unpickles `gepa_state.bin`.
    """
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []

    summaries = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir() or not _is_run_dir(run_dir):
            continue

        candidates = _read_json(run_dir / "candidates.json")
        run_log = _read_json(run_dir / "run_log.json")
        num_candidates = len(candidates) if isinstance(candidates, list) else None

        snapshot = _read_candidates_snapshot(run_dir)
        best_score = snapshot["best_score"] if snapshot is not None else None
        if best_score is None:
            best_score = _extract_best_score(candidates, run_log)

        summaries.append(
            RunSummary(
                name=run_dir.name,
                run_dir=run_dir,
                status=infer_status(run_dir),
                best_score=best_score,
                num_candidates=num_candidates,
                last_modified=datetime.fromtimestamp(run_dir.stat().st_mtime),
                is_live=False,
            )
        )

    summaries.sort(key=lambda r: r.last_modified, reverse=True)
    return summaries


def merge_live(
    historical: list[RunSummary], live_state: DashboardState | None
) -> list[RunSummary]:
    """Pure merge: the live run (if any) always sorts first, then the rest of
    `historical` newest-first. If `live_state`'s run_dir also appears in
    `historical` (e.g. it has already written JSON to disk), the disk entry is
    dropped in favor of the live, always-fresher in-memory one.
    """
    rest = sorted(
        (r for r in historical if live_state is None or r.run_dir != live_state.run_dir),
        key=lambda r: r.last_modified,
        reverse=True,
    )
    if live_state is None:
        return rest

    live_summary = RunSummary(
        name=live_state.run_name,
        run_dir=live_state.run_dir,
        status=live_state.status,
        best_score=live_state.best_score,
        num_candidates=len(live_state.candidates),
        last_modified=datetime.now(),
        is_live=True,
    )
    return [live_summary] + rest


def load_dashboard_state(run_dir: Path) -> DashboardState:
    """Constructs a throwaway `DashboardState` for a historical run directly
    from its on-disk JSON, so it can be rendered through the same
    Overview/Candidates/Log widgets as a live run via `refresh_from_state`
    (no live/historical fork in the widgets themselves).
    """
    run_dir = Path(run_dir)
    candidates = _read_json(run_dir / "candidates.json") or []
    run_log = _read_json(run_dir / "run_log.json") or []
    snapshot = _read_candidates_snapshot(run_dir)
    snapshot_by_idx = (
        {row["idx"]: row for row in snapshot["candidates"] if isinstance(row, dict) and "idx" in row}
        if snapshot is not None
        else {}
    )

    state = DashboardState(run_name=run_dir.name, run_dir=run_dir)
    state.status = infer_status(run_dir)

    if isinstance(candidates, list):
        for idx, program in enumerate(candidates):
            program = program if isinstance(program, dict) else {}
            snapshot_row = snapshot_by_idx.get(idx)
            if snapshot_row is not None:
                # Authoritative: autumn's own snapshot, captured live from the
                # run that produced this candidate.
                score = snapshot_row.get("val_score")
                discovered_iteration = snapshot_row.get("discovered_iteration")
                parent_ids = snapshot_row.get("parent_ids") or []
                is_pareto_member = bool(snapshot_row.get("is_pareto_member"))
            else:
                # No snapshot for this candidate (older run predating
                # autumn_candidates.json, or a run that never reached its first
                # checkpoint) -- same best-effort inline-score fallback as before
                # this snapshot existed.
                score = None
                for key in _SCORE_KEYS:
                    value = program.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        score = float(value)
                        break
                discovered_iteration = None
                parent_ids = []
                is_pareto_member = False
            state.candidates[idx] = CandidateRow(
                idx=idx,
                val_score=score,
                discovered_iteration=discovered_iteration,
                parent_ids=parent_ids,
                is_pareto_member=is_pareto_member,
                text=program or None,
            )

    if isinstance(run_log, list) and run_log:
        state.total_iterations = len(run_log)
        state.current_iteration = len(run_log)

    if snapshot is not None:
        state.best_idx = snapshot.get("best_idx")
        state.best_score = snapshot.get("best_score")
        state.pareto_front = set(snapshot.get("pareto_front") or [])
    else:
        state.best_score = _extract_best_score(candidates, run_log)
        if state.best_score is not None:
            for row in state.candidates.values():
                if row.val_score == state.best_score:
                    state.best_idx = row.idx
                    break

    for entry in run_log if isinstance(run_log, list) else []:
        if not isinstance(entry, dict):
            continue
        level = "error" if entry.get("event") == "on_error" else "info"
        state.append_log(level, json.dumps(entry, default=str))

    return state
