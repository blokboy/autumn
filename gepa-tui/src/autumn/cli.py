"""Command-line entry point for the `autumn` console script.

Three modes:

- `autumn run <script.py> [--dry-run]` launches the dashboard. Without
  `--dry-run`, it runs `<script.py>` for real: `autumn.runner.launch` patches
  GEPA's entry points (`autumn.patch.apply`) and executes the script via
  `runpy` on a background thread, streaming its real `GEPACallback` events
  into the dashboard live. `--dry-run` instead replays a scripted,
  dependency-free sequence of GEPA callback events
  (`autumn.fixtures.dry_run_events.replay`) so the UI can be exercised without
  a real GEPA optimization run -- `<script.py>` is still accepted and used to
  derive the run name in that mode, but is never imported or executed.
- `autumn` (no subcommand) opens the dashboard in browse mode: no live run, just
  `registry.scan(paths.default_runs_root())` rendered through the same sidebar +
  tabs as launch mode, fully navigable.
- `autumn runs [--json]` is the non-interactive counterpart to bare `autumn`:
  it prints the same registry scan as a plain-text table (or `--json` for a
  JSON array) and exits, without ever opening Textual -- so it must not import
  `autumn.app` (that import pulls in Textual and the whole dashboard stack).
  `registry.py` itself has no Textual dependency, so it's safe to import at
  module level here.
"""

import argparse
import json
from pathlib import Path

from autumn import paths, registry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autumn",
        description="A Textual dashboard for GEPA prompt-optimization runs.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Launch the dashboard for a GEPA optimization script.",
    )
    run_parser.add_argument(
        "script",
        type=Path,
        help="Path to the GEPA optimization script to run (see --dry-run to instead "
        "replay a scripted demo without executing it).",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Replay a scripted demo run instead of executing <script.py>.",
    )
    run_parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory to store/read run state (defaults under the autumn data "
        "root). Accepted now; not yet wired up for real runs.",
    )
    run_parser.add_argument(
        "--name",
        default=None,
        help="Override the derived run name.",
    )

    subparsers.add_parser(
        "runs",
        help="List known runs without opening the dashboard.",
    ).add_argument(
        "--json",
        action="store_true",
        help="Print the run list as a JSON array instead of a plain-text table.",
    )

    return parser


def _run(args: argparse.Namespace) -> int:
    # Imported lazily so that `autumn` (no subcommand) and `autumn --help`
    # don't require Textual to be importable just to print a message.
    from autumn.app import AutumnApp

    script_path = Path(args.script)
    run_name = args.name or paths.derive_run_name(script_path)
    run_dir = Path(args.run_dir) if args.run_dir else paths.default_runs_root() / run_name

    app = AutumnApp(
        runs_root=paths.default_runs_root(),
        run_name=run_name,
        run_dir=run_dir,
        script_path=script_path,
        dry_run=args.dry_run,
    )
    app.run()
    return 0


def _browse(args: argparse.Namespace) -> int:
    from autumn.app import AutumnApp

    app = AutumnApp(runs_root=paths.default_runs_root())
    app.run()
    return 0


def _runs_as_rows(summaries: list[registry.RunSummary]) -> list[dict]:
    return [
        {
            "name": summary.name,
            "status": summary.status.value,
            "best_score": summary.best_score,
            "num_candidates": summary.num_candidates,
            "last_modified": summary.last_modified.isoformat(),
            "is_live": summary.is_live,
        }
        for summary in summaries
    ]


def _runs(args: argparse.Namespace) -> int:
    summaries = registry.scan(paths.default_runs_root())

    if args.json:
        print(json.dumps(_runs_as_rows(summaries), indent=2))
        return 0

    if not summaries:
        print("no runs found")
        return 0

    print(f"{'NAME':<40} {'STATUS':<10} {'SCORE':<8} {'CANDIDATES':<11} LAST MODIFIED")
    for summary in summaries:
        score = "-" if summary.best_score is None else f"{summary.best_score:.4f}"
        candidates = "-" if summary.num_candidates is None else str(summary.num_candidates)
        modified = summary.last_modified.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{summary.name:<40} {summary.status.value:<10} {score:<8} {candidates:<11} {modified}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        return _browse(args)

    if args.command == "run":
        return _run(args)

    if args.command == "runs":
        return _runs(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
