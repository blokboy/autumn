"""Chat tool wrapper around `registry.scan` -- lets dashboard chat answer
questions like "what runs have I done" without leaving chat for a shell.
Read-only, no confirmation required (see docs/prd/chat-cli-parity-tools.md,
"Tools exposed").

Reuses `registry.scan` outright (no reimplemented run-directory scanning)
and serializes results with the exact same JSON row shape `autumn runs
--json` already prints -- see `cli._runs_as_rows` -- so a chat answer and
the CLI's own output can never drift apart. Deliberately mirrors only the
historical (on-disk) scan, not `registry.merge_live`'s in-process live run
-- a chat tool call has no access to the dashboard's live `DashboardState`,
same as `autumn runs` run from a separate shell wouldn't either.
"""

import json
from pathlib import Path
from typing import Any

import paths
import registry

TOOL_NAME = "list_runs"

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "List past and in-progress GEPA optimization runs Autumn knows about, including "
            "each run's name, status, best score so far, candidate count, and when it was "
            "last modified. Use this for questions about what runs the user has done or their "
            "results -- not for live details of the run currently open in the dashboard."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def _run_to_row(summary: registry.RunSummary) -> dict[str, Any]:
    """Same JSON shape as `autumn runs --json` (cli._runs_as_rows)."""
    return {
        "name": summary.name,
        "status": summary.status.value,
        "best_score": summary.best_score,
        "num_candidates": summary.num_candidates,
        "last_modified": summary.last_modified.isoformat(),
        "is_live": summary.is_live,
    }


def run_tool(arguments: dict[str, Any], *, runs_root: Path | None = None) -> str:
    """Entry point `GroqRunner` calls to execute a `list_runs` tool call.
    Takes no arguments beyond the (ignored) `arguments` dict -- the schema
    declares none -- and returns a JSON array of run rows, or `"[]"` if no
    runs exist. `runs_root` defaults to the real runs root
    (`paths.default_runs_root()`) when not given; tests override it the
    same way `search_docs.run_tool`'s `root` param does."""
    resolved_root = runs_root if runs_root is not None else paths.default_runs_root()
    summaries = registry.scan(resolved_root)
    return json.dumps([_run_to_row(summary) for summary in summaries])
