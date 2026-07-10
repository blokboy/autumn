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
- `autumn` (no subcommand) opens the dashboard on the InputScreen landing
  screen: an empty Enter drops into browse mode (no live run, just
  `registry.scan(paths.default_runs_root())` rendered through the same
  sidebar + tabs as launch mode, fully navigable); a `gepa <script> ...`
  command parses with the exact same grammar as `autumn run` (see
  `_add_run_arguments`/`parse_gepa_command` below) and launches identically.
- `autumn runs [--json]` is the non-interactive counterpart to bare `autumn`:
  it prints the same registry scan as a plain-text table (or `--json` for a
  JSON array) and exits, without ever opening Textual -- so it must not import
  `autumn.app` (that import pulls in Textual and the whole dashboard stack).
  `registry.py` itself has no Textual dependency, so it's safe to import at
  module level here.
"""

import argparse
import json
import shlex
from dataclasses import dataclass
from pathlib import Path

from autumn import local_models, paths, registry

_GEPA_PREFIX = "gepa "


class LaunchSpecError(ValueError):
    """Raised by `parse_gepa_command` on invalid `gepa ...` input (bad flags,
    missing/nonexistent script path) -- a plain exception rather than
    argparse's default SystemExit, since InputScreen needs to catch this and
    show a notification instead of taking down the Textual event loop."""


class _RaisingArgumentParser(argparse.ArgumentParser):
    """argparse.ArgumentParser.error() prints usage to stderr and calls
    sys.exit(2) -- correct for the `autumn run` CLI subcommand, fatal if it
    ever escaped from inside InputScreen's `gepa ...` parsing. This subclass
    is used only for that in-app parsing path; raising LaunchSpecError lets
    the screen catch it and stay put."""

    def error(self, message: str) -> None:
        raise LaunchSpecError(message)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Grammar shared between `autumn run` (via `_build_parser`) and
    InputScreen's `gepa <script> ...` parsing (via `parse_gepa_command`), so
    the two entry points can never drift apart."""
    parser.add_argument(
        "script",
        type=Path,
        help="Path to the GEPA optimization script to run (see --dry-run to instead "
        "replay a scripted demo without executing it).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Replay a scripted demo run instead of executing <script.py>.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory to store/read run state (defaults under the autumn data "
        "root). Accepted now; not yet wired up for real runs.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override the derived run name.",
    )


@dataclass
class LaunchSpec:
    """Fully-resolved parameters for launching a live GEPA run: the shared
    output of both `autumn run`'s argparse namespace and InputScreen's
    `gepa ...` parsing, ready to hand to `AutumnApp`/`AutumnApp.launch_gepa_run`."""

    run_name: str
    run_dir: Path
    script_path: Path
    dry_run: bool


def build_launch_spec(args: argparse.Namespace) -> LaunchSpec:
    """Resolves an argparse Namespace produced by `_add_run_arguments`
    (deriving the run name and default run_dir when not explicitly given) into
    a `LaunchSpec` -- the same derivation `_run` has always done, now shared
    with InputScreen's `gepa ...` parsing."""
    script_path = Path(args.script)
    run_name = args.name or paths.derive_run_name(script_path)
    run_dir = Path(args.run_dir) if args.run_dir else paths.default_runs_root() / run_name
    return LaunchSpec(run_name=run_name, run_dir=run_dir, script_path=script_path, dry_run=args.dry_run)


def parse_gepa_command(tokens: list[str]) -> LaunchSpec:
    """Parses the tokens following a `gepa ` prefix typed into InputScreen
    (e.g. `["myscript.py", "--dry-run"]`) with the exact same grammar as
    `autumn run` (`_add_run_arguments`), then resolves them into a
    `LaunchSpec` exactly as `build_launch_spec` does for the CLI.

    Raises `LaunchSpecError` -- never `SystemExit`, unlike the CLI's own
    argparse invocation -- on bad flags, a missing positional, or a script
    path that doesn't exist on disk, so InputScreen can show a notification
    and stay on screen instead of crashing or navigating away.
    """
    parser = _RaisingArgumentParser(prog="gepa", add_help=False)
    _add_run_arguments(parser)
    args = parser.parse_args(tokens)

    spec = build_launch_spec(args)
    if not spec.script_path.exists():
        raise LaunchSpecError(f"script not found: {spec.script_path}")
    return spec


def parse_command_line(text: str) -> LaunchSpec | None:
    """Parses one submitted line of free text from either InputScreen or
    CommandBar into a `LaunchSpec`, or `None` if it isn't a `gepa ...` command
    at all (a stub/non-`gepa` prompt) -- the shared classification both
    surfaces use so they can't drift apart on what counts as a launch command.

    `text` is assumed already stripped and non-empty (both callers handle the
    empty-Enter case themselves before reaching this point, since it means
    different things to each: browse-mode navigation for InputScreen, a no-op
    for CommandBar).

    Raises `LaunchSpecError` on malformed `gepa ...` syntax -- unbalanced
    quotes, unknown flags, a missing/nonexistent script path -- exactly as
    `parse_gepa_command` does, so callers only need one except clause.
    """
    if not text.startswith(_GEPA_PREFIX):
        return None
    remainder = text[len(_GEPA_PREFIX) :]
    try:
        tokens = shlex.split(remainder)
    except ValueError as exc:  # unbalanced quotes, e.g. `gepa "foo`
        raise LaunchSpecError(f"Couldn't parse command: {exc}") from exc
    return parse_gepa_command(tokens)


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
    _add_run_arguments(run_parser)

    subparsers.add_parser(
        "runs",
        help="List known runs without opening the dashboard.",
    ).add_argument(
        "--json",
        action="store_true",
        help="Print the run list as a JSON array instead of a plain-text table.",
    )

    models_parser = subparsers.add_parser(
        "models",
        help="Manage Autumn's local model catalog.",
    )
    model_subparsers = models_parser.add_subparsers(dest="models_command")

    install_parser = model_subparsers.add_parser(
        "install",
        help="Install a local model file into Autumn's managed catalog.",
    )
    install_parser.add_argument("name", help="Name to give this model in Autumn.")
    install_parser.add_argument("source", type=Path, help="Path to a local model file.")
    install_parser.add_argument(
        "--backend",
        default="llama.cpp",
        help="Runtime backend for this model (default: llama.cpp).",
    )
    install_parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Optional context window metadata for this model.",
    )

    list_parser = model_subparsers.add_parser(
        "list",
        help="List installed local models.",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the model catalog as JSON instead of a plain-text table.",
    )

    default_parser = model_subparsers.add_parser(
        "default",
        help="Select the default local model.",
    )
    default_parser.add_argument("name", help="Installed model name to use by default.")

    remove_parser = model_subparsers.add_parser(
        "remove",
        help="Remove an installed local model.",
    )
    remove_parser.add_argument("name", help="Installed model name to remove.")

    return parser


def _run(args: argparse.Namespace) -> int:
    # Imported lazily so that `autumn` (no subcommand) and `autumn --help`
    # don't require Textual to be importable just to print a message.
    from autumn.app import AutumnApp

    spec = build_launch_spec(args)

    app = AutumnApp(
        runs_root=paths.default_runs_root(),
        run_name=spec.run_name,
        run_dir=spec.run_dir,
        script_path=spec.script_path,
        dry_run=spec.dry_run,
        queue_sessions_root=paths.sessions_root(),
    )
    app.run()
    return 0


def _browse(args: argparse.Namespace) -> int:
    from autumn.app import AutumnApp

    app = AutumnApp(runs_root=paths.default_runs_root(), queue_sessions_root=paths.sessions_root())
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


def _models_as_rows(models: list[local_models.LocalModel]) -> list[dict]:
    return [
        {
            "name": model.name,
            "backend": model.backend,
            "path": str(model.path),
            "context_window": model.context_window,
            "is_default": model.is_default,
        }
        for model in models
    ]


def _models(args: argparse.Namespace) -> int:
    catalog_root = paths.models_root()
    command = args.models_command

    if command == "install":
        installed = local_models.install_model(
            catalog_root,
            name=args.name,
            source_path=args.source,
            backend=args.backend,
            context_window=args.context_window,
        )
        default_note = " (default)" if installed.is_default else ""
        print(f"installed {installed.name}{default_note}: {installed.path}")
        return 0

    if command == "list":
        models = local_models.list_models(catalog_root)
        if args.json:
            print(json.dumps(_models_as_rows(models), indent=2))
            return 0
        if not models:
            print("no models installed")
            return 0
        print(f"{'NAME':<24} {'BACKEND':<12} {'DEFAULT':<8} PATH")
        for model in models:
            default = "default" if model.is_default else "-"
            print(f"{model.name:<24} {model.backend:<12} {default:<8} {model.path}")
        return 0

    if command == "default":
        try:
            selected = local_models.set_default(catalog_root, args.name)
        except ValueError as exc:
            print(str(exc))
            return 1
        print(f"default model: {selected.name}")
        return 0

    if command == "remove":
        local_models.remove_model(catalog_root, args.name)
        print(f"removed {args.name}")
        return 0

    print("models command required")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        return _browse(args)

    if args.command == "run":
        return _run(args)

    if args.command == "runs":
        return _runs(args)

    if args.command == "models":
        return _models(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
