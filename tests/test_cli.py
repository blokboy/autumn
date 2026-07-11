"""Tests for the launch-spec parsing shared between `autumn run` and
InputScreen's `gepa ...` command (autumn.cli.parse_gepa_command /
build_launch_spec), plus a regression check that `autumn run`/`autumn runs`
keep behaving exactly as before the InputScreen refactor.
"""

from pathlib import Path

import pytest

from autumn import cli
from autumn import local_models
from autumn.cli import LaunchSpecError, parse_gepa_command


def test_parse_gepa_command_valid_minimal(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("pass\n")

    specs = parse_gepa_command([str(script)])
    spec = specs[0]

    assert len(specs) == 1
    assert spec.script_path == script
    assert spec.dry_run is False
    assert spec.run_name.startswith("script-")
    assert spec.run_dir == cli.paths.default_runs_root() / spec.run_name


def test_parse_gepa_command_valid_with_all_flags(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("pass\n")
    run_dir = tmp_path / "custom-run-dir"

    specs = parse_gepa_command(
        [str(script), "--dry-run", "--name", "my-run", "--run-dir", str(run_dir)]
    )
    spec = specs[0]

    assert len(specs) == 1
    assert spec.script_path == script
    assert spec.dry_run is True
    assert spec.run_name == "my-run"
    assert spec.run_dir == run_dir


def test_parse_gepa_command_missing_script_raises():
    with pytest.raises(LaunchSpecError):
        parse_gepa_command(["--dry-run"])


def test_parse_gepa_command_directory_mode_discovers_only_python_files(tmp_path):
    script_dir = tmp_path / "examples"
    script_dir.mkdir()
    first = script_dir / "a_first.py"
    first.write_text("pass\n")
    second = script_dir / "b_second.py"
    second.write_text("pass\n")
    (script_dir / "README.md").write_text("# docs\n")
    (script_dir / "notes.txt").write_text("not a script\n")
    (script_dir / "nested.py").mkdir()

    specs = parse_gepa_command(["--run-dir", str(script_dir), "--dry-run"])

    assert [spec.script_path for spec in specs] == [first, second]
    assert [spec.run_name.split("-")[0] for spec in specs] == ["a_first", "b_second"]
    assert all(spec.run_dir.parent == cli.paths.default_runs_root() for spec in specs)
    assert all(spec.dry_run is True for spec in specs)


def test_parse_gepa_command_directory_mode_rejects_name_override(tmp_path):
    script_dir = tmp_path / "examples"
    script_dir.mkdir()
    (script_dir / "script.py").write_text("pass\n")

    with pytest.raises(LaunchSpecError):
        parse_gepa_command(["--run-dir", str(script_dir), "--name", "custom"])


def test_parse_gepa_command_directory_mode_requires_python_scripts(tmp_path):
    script_dir = tmp_path / "examples"
    script_dir.mkdir()
    (script_dir / "README.md").write_text("# docs\n")

    with pytest.raises(LaunchSpecError):
        parse_gepa_command(["--run-dir", str(script_dir)])


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
    result = cli.main(["run", "--dry-run"])

    assert result == 2


def test_models_install_adds_model_to_catalog(tmp_path, monkeypatch, capsys):
    model_file = tmp_path / "tiny.gguf"
    model_file.write_bytes(b"fake model")
    catalog_root = tmp_path / "models"
    monkeypatch.setattr(cli.paths, "models_root", lambda: catalog_root)

    result = cli.main(
        [
            "models",
            "install",
            "tiny",
            str(model_file),
            "--backend",
            "llama.cpp",
            "--context-window",
            "2048",
        ]
    )

    assert result == 0
    assert "installed tiny" in capsys.readouterr().out
    assert local_models.get_default(catalog_root).name == "tiny"


def test_models_list_prints_installed_models(tmp_path, monkeypatch, capsys):
    model_file = tmp_path / "tiny.gguf"
    model_file.write_bytes(b"fake model")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=model_file)
    monkeypatch.setattr(cli.paths, "models_root", lambda: catalog_root)

    result = cli.main(["models", "list"])

    assert result == 0
    output = capsys.readouterr().out
    assert "tiny" in output
    assert "llama.cpp" in output
    assert "default" in output


def test_models_default_selects_installed_model(tmp_path, monkeypatch, capsys):
    first = tmp_path / "first.gguf"
    first.write_bytes(b"first")
    second = tmp_path / "second.gguf"
    second.write_bytes(b"second")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="first", source_path=first)
    local_models.install_model(catalog_root, name="second", source_path=second)
    monkeypatch.setattr(cli.paths, "models_root", lambda: catalog_root)

    result = cli.main(["models", "default", "second"])

    assert result == 0
    assert "default model: second" in capsys.readouterr().out
    assert local_models.get_default(catalog_root).name == "second"


def test_models_remove_deletes_installed_model(tmp_path, monkeypatch, capsys):
    model_file = tmp_path / "tiny.gguf"
    model_file.write_bytes(b"fake model")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=model_file)
    monkeypatch.setattr(cli.paths, "models_root", lambda: catalog_root)

    result = cli.main(["models", "remove", "tiny"])

    assert result == 0
    assert "removed tiny" in capsys.readouterr().out
    assert local_models.list_models(catalog_root) == []
