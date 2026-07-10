"""Tests for the launch-spec parsing shared between `autumn run` and
InputScreen's `gepa ...` command (autumn.cli.parse_gepa_command /
build_launch_spec), plus a regression check that `autumn run`/`autumn runs`
keep behaving exactly as before the InputScreen refactor.
"""

from pathlib import Path

import pytest

from autumn import cli
from autumn.cli import LaunchSpecError, parse_gepa_command


def test_parse_gepa_command_valid_minimal(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("pass\n")

    spec = parse_gepa_command([str(script)])

    assert spec.script_path == script
    assert spec.dry_run is False
    assert spec.run_name.startswith("script-")
    assert spec.run_dir == cli.paths.default_runs_root() / spec.run_name


def test_parse_gepa_command_valid_with_all_flags(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("pass\n")
    run_dir = tmp_path / "custom-run-dir"

    spec = parse_gepa_command(
        [str(script), "--dry-run", "--name", "my-run", "--run-dir", str(run_dir)]
    )

    assert spec.script_path == script
    assert spec.dry_run is True
    assert spec.run_name == "my-run"
    assert spec.run_dir == run_dir


def test_parse_gepa_command_missing_script_raises():
    with pytest.raises(LaunchSpecError):
        parse_gepa_command(["--dry-run"])


def test_parse_gepa_command_unknown_flag_raises(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("pass\n")

    with pytest.raises(LaunchSpecError):
        parse_gepa_command([str(script), "--bogus-flag"])


def test_parse_gepa_command_nonexistent_script_raises(tmp_path):
    missing = tmp_path / "does-not-exist.py"

    with pytest.raises(LaunchSpecError):
        parse_gepa_command([str(missing)])


def test_run_subcommand_grammar_unchanged(tmp_path):
    """`autumn run` still parses the same flags via the shared grammar --
    a regression guard that extracting `_add_run_arguments` didn't change the
    CLI's own argparse behavior."""
    script = tmp_path / "script.py"
    parser = cli._build_parser()

    args = parser.parse_args(["run", str(script), "--dry-run", "--name", "foo"])

    assert args.command == "run"
    assert Path(args.script) == script
    assert args.dry_run is True
    assert args.name == "foo"
    assert args.run_dir is None


def test_run_subcommand_rejects_missing_script(tmp_path):
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--dry-run"])
