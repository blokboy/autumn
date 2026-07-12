"""Command-line entry point for the `autumn` console script.

Three modes:

- `autumn run <script.py> [--dry-run]` launches the dashboard. Without
  `--dry-run`, it runs `<script.py>` for real: `runner.launch` patches
  GEPA's entry points (`patch.apply`) and executes the script via
  `runpy` on a background thread, streaming its real `GEPACallback` events
  into the dashboard live. `--dry-run` instead replays a scripted,
  dependency-free sequence of GEPA callback events
  (`fixtures.dry_run_events.replay`) so the UI can be exercised without
  a real GEPA optimization run -- `<script.py>` is still accepted and used to
  derive the run name in that mode, but is never imported or executed.
- `autumn` (no subcommand) opens the dashboard on the InputScreen landing
  screen: an empty Enter drops into browse mode (no live run, just
  `registry.scan(paths.default_runs_root())` rendered through the same
  sidebar + tabs as launch mode, fully navigable); `gepa run <script> ...`
  parses with the exact same grammar as `autumn run` (see
  `_add_run_arguments`/`parse_gepa_command` below) and launches identically,
  while `gepa optimize ...` is routed to a prompt-optimization draft handoff.
- `autumn runs [--json]` is the non-interactive counterpart to bare `autumn`:
  it prints the same registry scan as a plain-text table (or `--json` for a
  JSON array) and exits, without ever opening Textual -- so it must not import
  `app` (that import pulls in Textual and the whole dashboard stack).
  `registry.py` itself has no Textual dependency, so it's safe to import at
  module level here.
"""

import argparse
import json
import shlex
from dataclasses import dataclass
from pathlib import Path

import anthropic_policy, config, credentials, eval_assets, groq_policy, local_models, openai_policy, paths, registry
from models import PromptRoutingPolicy
from prompt_optimization_drafts import PromptOptimizationDraft, parse_prompt_optimization_draft

_GEPA_COMMAND = "gepa"
_GEPA_MODE_GUIDANCE = "Use `gepa run <script.py>` to launch a script run, or `gepa optimize ...` to start prompt optimization."

# `credentials.KNOWN_PROVIDERS` is the single source of truth for the known
# provider set -- shared with the `list_keys` chat tool (see
# `list_keys.py`) so both surfaces show the same expected set.
_KNOWN_PROVIDERS = credentials.KNOWN_PROVIDERS


class LaunchSpecError(ValueError):
    """Raised by `parse_gepa_command` on invalid `gepa run ...` input (bad flags,
    missing/nonexistent script path) -- a plain exception rather than
    argparse's default SystemExit, since InputScreen needs to catch this and
    show a notification instead of taking down the Textual event loop."""


class _RaisingArgumentParser(argparse.ArgumentParser):
    """argparse.ArgumentParser.error() prints usage to stderr and calls
    sys.exit(2) -- correct for the `autumn run` CLI subcommand, fatal if it
    ever escaped from inside InputScreen's GEPA command parsing. This subclass
    is used only for that in-app parsing path; raising LaunchSpecError lets
    the screen catch it and stay put."""

    def error(self, message: str) -> None:
        raise LaunchSpecError(message)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Grammar shared between `autumn run` (via `_build_parser`) and
    InputScreen's `gepa run <script> ...` parsing (via `parse_gepa_command`), so
    the two entry points can never drift apart."""
    parser.add_argument(
        "script",
        nargs="?",
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
        "root). If no script is supplied, discover and run direct child *.py "
        "scripts from this directory.",
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
    `gepa run ...` parsing, ready to hand to `AutumnApp`/`AutumnApp.launch_gepa_run`."""

    run_name: str
    run_dir: Path
    script_path: Path
    dry_run: bool


def build_launch_spec(args: argparse.Namespace) -> LaunchSpec:
    """Resolves an argparse Namespace produced by `_add_run_arguments`
    (deriving the run name and default run_dir when not explicitly given) into
    a `LaunchSpec` -- the same derivation `_run` has always done, now shared
    with InputScreen's `gepa run ...` parsing."""
    script_path = Path(args.script)
    run_name = args.name or paths.derive_run_name(script_path)
    run_dir = Path(args.run_dir) if args.run_dir else paths.default_runs_root() / run_name
    return LaunchSpec(run_name=run_name, run_dir=run_dir, script_path=script_path, dry_run=args.dry_run)


def _discover_directory_specs(script_dir: Path, *, dry_run: bool) -> list[LaunchSpec]:
    if not script_dir.exists():
        raise LaunchSpecError(f"run directory not found: {script_dir}")
    if not script_dir.is_dir():
        raise LaunchSpecError(f"run directory is not a directory: {script_dir}")

    script_paths = sorted(path for path in script_dir.iterdir() if path.is_file() and path.suffix == ".py")
    if not script_paths:
        raise LaunchSpecError(f"no Python scripts found in run directory: {script_dir}")

    specs = []
    for script_path in script_paths:
        run_name = paths.derive_run_name(script_path)
        specs.append(
            LaunchSpec(
                run_name=run_name,
                run_dir=paths.default_runs_root() / run_name,
                script_path=script_path,
                dry_run=dry_run,
            )
        )
    return specs


def build_launch_specs(args: argparse.Namespace) -> list[LaunchSpec]:
    """Resolves one parsed `gepa`/`autumn run` command into concrete runs.

    The normal form still launches exactly one script. Directory mode is only
    selected when no positional script is present and `--run-dir` points at the
    directory of scripts to discover.
    """
    if args.script is not None:
        return [build_launch_spec(args)]
    if args.run_dir is None:
        raise LaunchSpecError("script is required unless --run-dir points to a directory of scripts")
    if args.name is not None:
        raise LaunchSpecError("--name can't be used with directory run mode")
    return _discover_directory_specs(Path(args.run_dir), dry_run=args.dry_run)


def parse_gepa_command(tokens: list[str]) -> list[LaunchSpec]:
    """Parses the tokens following a `gepa run` prefix typed into InputScreen
    (e.g. `["myscript.py", "--dry-run"]`) with the exact same grammar as
    `autumn run` (`_add_run_arguments`), then resolves them into one or more
    `LaunchSpec`s exactly as `build_launch_specs` does for the CLI.

    Raises `LaunchSpecError` -- never `SystemExit`, unlike the CLI's own
    argparse invocation -- on bad flags, a missing positional, or a script
    path that doesn't exist on disk, so InputScreen can show a notification
    and stay on screen instead of crashing or navigating away.
    """
    parser = _RaisingArgumentParser(prog="gepa", add_help=False)
    _add_run_arguments(parser)
    args = parser.parse_args(tokens)

    specs = build_launch_specs(args)
    for spec in specs:
        if not spec.script_path.exists():
            raise LaunchSpecError(f"script not found: {spec.script_path}")
    return specs


def parse_command_line(
    text: str,
    *,
    catalog_root: Path | None = None,
    prompt_routing_policy: PromptRoutingPolicy | None = None,
) -> list[LaunchSpec] | PromptOptimizationDraft | None:
    """Parses one submitted line of free text from either InputScreen or
    CommandBar into one or more `LaunchSpec`s, or `None` if it isn't a
    `gepa` command at all (a chat prompt) -- the shared classification both
    surfaces use so they can't drift apart on what counts as a launch command.

    `text` is assumed already stripped and non-empty (both callers handle the
    empty-Enter case themselves before reaching this point, since it means
    different things to each: browse-mode navigation for InputScreen, a no-op
    for CommandBar).

    Raises `LaunchSpecError` on malformed `gepa ...` syntax -- unbalanced
    quotes, unknown flags, a missing/nonexistent script path -- exactly as
    `parse_gepa_command` does, so callers only need one except clause.
    """
    try:
        command_tokens = shlex.split(text)
    except ValueError as exc:  # unbalanced quotes, e.g. `gepa run "foo`
        if text.lstrip().startswith(_GEPA_COMMAND):
            raise LaunchSpecError(f"Couldn't parse command: {exc}") from exc
        return None

    if not command_tokens:
        return None
    if command_tokens[0] != _GEPA_COMMAND:
        implicit_tokens = _implicit_prompt_optimization_tokens(command_tokens)
        if implicit_tokens is None:
            return None
        return parse_prompt_optimization_draft(
            implicit_tokens,
            raw_text=text,
            catalog_root=catalog_root or paths.models_root(),
            prompt_routing_policy=prompt_routing_policy or _build_prompt_routing_policy(),
        )
    if len(command_tokens) == 1:
        raise LaunchSpecError(_GEPA_MODE_GUIDANCE)

    mode = command_tokens[1]
    mode_tokens = command_tokens[2:]
    if mode == "run":
        return parse_gepa_command(mode_tokens)
    if mode == "optimize":
        if not mode_tokens:
            raise LaunchSpecError("Prompt optimization details are required after `gepa optimize`.")
        return parse_prompt_optimization_draft(
            mode_tokens,
            raw_text=text,
            catalog_root=catalog_root or paths.models_root(),
            prompt_routing_policy=prompt_routing_policy or _build_prompt_routing_policy(),
        )
    raise LaunchSpecError(_GEPA_MODE_GUIDANCE)


def is_explicit_gepa_command(text: str) -> bool:
    """True when `text` starts with the literal `gepa` command word.

    `parse_command_line` returns the same `PromptOptimizationDraft` type for
    both explicit `gepa optimize ...` and implicit GEPA-specific chat phrases
    (e.g. "run GEPA on ..."), so callers that need to treat the two
    differently -- #50's chat intake skips straight to the confirmation
    screen only for the explicit form -- use this to tell them apart.
    """
    try:
        tokens = shlex.split(text)
    except ValueError:
        return False
    return bool(tokens) and tokens[0] == _GEPA_COMMAND


_IMPLICIT_GEPA_LEAD_VERBS = {"run", "use", "try", "launch"}
_IMPLICIT_GEPA_CONNECTORS = {"on", "to", "for", "with", "against", "over"}


def _implicit_prompt_optimization_tokens(tokens: list[str]) -> list[str] | None:
    """Detects a narrow set of GEPA-specific implicit phrases, e.g. `run GEPA
    on ...` or `use GEPA to optimize ...`, per the spec in issue #39. Mentioning
    "gepa" is necessary but not sufficient -- generic "optimize"/"improve"/
    "make better" phrasing that never names GEPA must stay normal chat, so this
    only fires when GEPA is paired with an adjacent action verb or connector
    rather than merely appearing somewhere in the sentence.
    """
    lowered = [token.lower() for token in tokens]
    if _GEPA_COMMAND not in lowered:
        return None
    gepa_index = lowered.index(_GEPA_COMMAND)

    if "optimize" in lowered:
        optimize_index = lowered.index("optimize")
        if gepa_index <= optimize_index:
            tail = tokens[optimize_index + 1 :]
            return tail or None

    lead_ok = gepa_index == 0 or lowered[gepa_index - 1] in _IMPLICIT_GEPA_LEAD_VERBS
    following = lowered[gepa_index + 1 :]
    if lead_ok and following and following[0] in _IMPLICIT_GEPA_CONNECTORS:
        tail = tokens[gepa_index + 2 :]
        return tail or None

    return None


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

    subagent_parser = subparsers.add_parser(
        "subagent",
        help="Run a one-shot subagent prompt and print only the final answer.",
    )
    subagent_parser.add_argument("prompt", help="Prompt for the one-shot subagent.")

    evals_parser = subparsers.add_parser(
        "evals",
        help="Discover and manage immutable eval assets.",
    )
    evals_subparsers = evals_parser.add_subparsers(dest="evals_command")

    evals_subparsers.add_parser(
        "list",
        help="List installed eval asset versions.",
    )

    evals_subparsers.add_parser(
        "available",
        help="List eval asset versions available from the static manifest.",
    )

    evals_install_parser = evals_subparsers.add_parser(
        "install",
        help="Install an immutable eval asset version.",
    )
    evals_install_parser.add_argument("asset", help="Eval asset selector, e.g. tiny-smoke@2026.07.12.")

    evals_remove_parser = evals_subparsers.add_parser(
        "remove",
        help="Remove an installed eval asset version.",
    )
    evals_remove_parser.add_argument("asset", help="Eval asset selector, e.g. tiny-smoke@2026.07.12.")

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

    keys_parser = subparsers.add_parser(
        "keys",
        help="Manage stored provider API keys (Groq, etc).",
    )
    keys_subparsers = keys_parser.add_subparsers(dest="keys_command")

    keys_add_parser = keys_subparsers.add_parser(
        "add",
        help="Store an API key for a provider, overriding any env var of the same purpose.",
    )
    keys_add_parser.add_argument("provider", choices=_KNOWN_PROVIDERS, help="Provider to store a key for.")
    keys_add_parser.add_argument("api_key", help="The provider's API key.")

    keys_subparsers.add_parser(
        "list",
        help="Show which providers have a key configured (stored or via env var) -- never prints key values.",
    )

    keys_remove_parser = keys_subparsers.add_parser(
        "remove",
        help="Remove a stored API key for a provider (an env var, if set, still applies afterward).",
    )
    keys_remove_parser.add_argument("provider", choices=_KNOWN_PROVIDERS, help="Provider to remove the stored key for.")

    config_parser = subparsers.add_parser(
        "config",
        help="Manage Autumn user preferences.",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    config_set_parser = config_subparsers.add_parser(
        "set",
        help="Persist a user preference.",
    )
    config_set_parser.add_argument("name", choices=config.CONFIG_KEYS, help="Preference to update.")
    config_set_parser.add_argument("value", help="Preference value.")

    config_subparsers.add_parser(
        "show",
        help="Show current user preferences.",
    )

    return parser


def _build_prompt_routing_policy() -> PromptRoutingPolicy:
    """Combines every real (non-stub) provider's policy into the single
    `PromptRoutingPolicy` `catalog.build_entries` accepts: Groq (#13),
    Anthropic and OpenAI (#21) each contribute their own `ProviderAccount`
    and `ProviderModel`s via their own `build_policy()`, gated independently
    on their own env var/stored key (see `groq_policy.py`,
    `anthropic_policy.py`, `openai_policy.py`). Plain list concatenation is
    enough -- no dedup needed, since every model is already namespaced by
    its `provider` field, so entries from different providers' policies can
    never collide."""
    policies = [groq_policy.build_policy(), anthropic_policy.build_policy(), openai_policy.build_policy()]
    return PromptRoutingPolicy(
        provider_accounts=[account for policy in policies for account in policy.provider_accounts],
        provider_models=[model for policy in policies for model in policy.provider_models],
    )


def _run(args: argparse.Namespace) -> int:
    try:
        specs = build_launch_specs(args)
    except LaunchSpecError as exc:
        print(str(exc))
        return 2
    spec = specs[0]

    # Imported lazily so that `autumn` (no subcommand) and `autumn --help`
    # don't require Textual to be importable just to print a message.
    from app import AutumnApp

    app = AutumnApp(
        runs_root=paths.default_runs_root(),
        run_name=spec.run_name,
        run_dir=spec.run_dir,
        script_path=spec.script_path,
        dry_run=spec.dry_run,
        initial_queue=specs[1:],
        queue_sessions_root=paths.sessions_root(),
        prompt_routing_policy=_build_prompt_routing_policy(),
    )
    app.run()
    return 0


def _browse(args: argparse.Namespace) -> int:
    from app import AutumnApp

    app = AutumnApp(
        runs_root=paths.default_runs_root(),
        queue_sessions_root=paths.sessions_root(),
        prompt_routing_policy=_build_prompt_routing_policy(),
    )
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


def _subagent(args: argparse.Namespace) -> int:
    import subagent

    result = subagent.run_subagent(
        args.prompt,
        catalog_root=paths.models_root(),
        policy=groq_policy.build_policy(),
    )
    print(result.answer)
    return 0


def _print_eval_assets_table(assets: list[eval_assets.InstalledEvalAsset] | tuple[eval_assets.RemoteEvalAsset, ...]) -> None:
    print(f"{'ASSET':<24} {'VERSION':<12} {'METRICS':<22} {'SIZE':<10} {'LICENSE':<12} PROVENANCE")
    for asset in assets:
        metrics = ", ".join(asset.supported_metrics)
        print(
            f"{asset.asset_id:<24} {asset.version:<12} {metrics:<22} "
            f"{asset.size_bytes:<10} {asset.license:<12} {asset.provenance}"
        )


def _evals(args: argparse.Namespace) -> int:
    command = args.evals_command
    assets_root = paths.eval_assets_root()

    if command == "list":
        installed = eval_assets.list_installed_assets(assets_root)
        if not installed:
            print("no eval assets installed")
            return 0
        _print_eval_assets_table(installed)
        return 0

    if command == "available":
        try:
            manifest = eval_assets.load_remote_manifest(eval_assets.DEFAULT_REMOTE_MANIFEST_PATH)
        except eval_assets.EvalAssetError as exc:
            print(str(exc))
            return 1
        if not manifest.assets:
            print("no eval assets available")
            return 0
        _print_eval_assets_table(manifest.assets)
        return 0

    if command == "install":
        try:
            manifest = eval_assets.load_remote_manifest(eval_assets.DEFAULT_REMOTE_MANIFEST_PATH)
            asset = eval_assets.select_remote_asset(manifest, args.asset)
            installed = eval_assets.install_asset(assets_root, asset)
        except eval_assets.EvalAssetError as exc:
            print(str(exc))
            return 1
        print(f"installed {installed.asset_id}@{installed.version}: {installed.path}")
        return 0

    if command == "remove":
        try:
            asset = eval_assets.select_installed_asset(assets_root, args.asset)
        except eval_assets.EvalAssetError as exc:
            print(str(exc))
            return 1
        eval_assets.remove_asset(assets_root, asset.asset_id, asset.version)
        print(f"removed {asset.asset_id}@{asset.version}")
        return 0

    print("evals command required")
    return 1


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


def _keys(args: argparse.Namespace) -> int:
    command = args.keys_command

    if command == "add":
        credentials.set_key(args.provider, args.api_key)
        print(f"stored a key for {args.provider}")
        return 0

    if command == "list":
        for provider in _KNOWN_PROVIDERS:
            configured = credentials.resolve_key(provider, credentials.env_var_for_provider(provider)) is not None
            print(f"{provider:<10} {'configured' if configured else 'not configured'}")
        return 0

    if command == "remove":
        removed = credentials.remove_key(args.provider)
        if removed:
            print(f"removed the stored key for {args.provider}")
        else:
            print(f"no stored key for {args.provider}")
        return 0

    print("keys command required")
    return 1


def _config(args: argparse.Namespace) -> int:
    command = args.config_command

    if command == "set":
        try:
            if args.name == config.SEARCH_MODE_KEY:
                config.set_search_mode(args.value)
            elif args.name == config.ACTION_MODE_KEY:
                config.set_action_mode(args.value)
            else:
                print(f"unknown config key: {args.name}")
                return 1
        except config.ConfigError as exc:
            print(str(exc))
            return 1
        print(f"{args.name}: {args.value}")
        return 0

    if command == "show":
        values = config.as_dict()
        for name in config.CONFIG_KEYS:
            print(f"{name}: {values[name]}")
        return 0

    print("config command required")
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

    if args.command == "subagent":
        return _subagent(args)

    if args.command == "evals":
        return _evals(args)

    if args.command == "models":
        return _models(args)

    if args.command == "keys":
        return _keys(args)

    if args.command == "config":
        return _config(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
