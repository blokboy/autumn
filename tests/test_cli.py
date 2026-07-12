"""Tests for the launch-spec parsing shared between `autumn run` and
InputScreen's `gepa run ...` command (cli.parse_gepa_command /
build_launch_spec), plus a regression check that `autumn run`/`autumn runs`
keep behaving exactly as before the InputScreen refactor.
"""

from pathlib import Path

import pytest

import cli
import local_models
from cli import LaunchSpecError, PromptOptimizationDraft, parse_command_line, parse_gepa_command


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


def test_parse_command_line_gepa_run_launches_script(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("pass\n")

    specs = parse_command_line(f"gepa run {script} --dry-run --name split-run")

    assert specs is not None
    assert len(specs) == 1
    assert specs[0].script_path == script
    assert specs[0].dry_run is True
    assert specs[0].run_name == "split-run"


def test_parse_command_line_gepa_optimize_returns_prompt_draft():
    draft = parse_command_line("gepa optimize --prompt 'Write a better summary' --name summary")

    assert draft == PromptOptimizationDraft(
        raw_text="gepa optimize --prompt 'Write a better summary' --name summary",
        prompt="Write a better summary",
        run_name="summary",
    )


def test_parse_command_line_rejects_bare_gepa_script_with_guidance(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("pass\n")

    with pytest.raises(LaunchSpecError, match="gepa run <script.py>"):
        parse_command_line(f"gepa {script}")


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


def test_keys_add_stores_a_key(capsys):
    import credentials

    result = cli.main(["keys", "add", "groq", "gsk_abc123"])

    assert result == 0
    assert "stored a key for groq" in capsys.readouterr().out
    assert credentials.get_key("groq") == "gsk_abc123"


def test_keys_add_rejects_unknown_provider(capsys):
    with pytest.raises(SystemExit):
        cli.main(["keys", "add", "made-up-provider", "some-key"])


def test_keys_list_shows_configured_and_not_configured(monkeypatch, capsys):
    import credentials

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    credentials.set_key("groq", "gsk_abc123")

    result = cli.main(["keys", "list"])

    assert result == 0
    output = capsys.readouterr().out
    assert "groq       configured" in output
    assert "anthropic  not configured" in output
    assert "openai     not configured" in output
    assert f"{'tavily':<10} not configured" in output


def test_keys_list_reflects_env_var_without_any_stored_key(monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "from-env")

    result = cli.main(["keys", "list"])

    assert result == 0
    assert "groq       configured" in capsys.readouterr().out


def test_keys_remove_deletes_a_stored_key(capsys):
    import credentials

    credentials.set_key("groq", "gsk_abc123")

    result = cli.main(["keys", "remove", "groq"])

    assert result == 0
    assert "removed the stored key for groq" in capsys.readouterr().out
    assert credentials.get_key("groq") is None


def test_keys_remove_reports_when_nothing_was_stored(capsys):
    result = cli.main(["keys", "remove", "groq"])

    assert result == 0
    assert "no stored key for groq" in capsys.readouterr().out


def test_keys_remove_does_not_affect_env_var_fallback(monkeypatch, capsys):
    import credentials

    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    credentials.set_key("groq", "from-store")

    cli.main(["keys", "remove", "groq"])
    capsys.readouterr()
    result = cli.main(["keys", "list"])

    assert result == 0
    assert "groq       configured" in capsys.readouterr().out


def test_keys_add_accepts_tavily(capsys):
    import credentials

    result = cli.main(["keys", "add", "tavily", "tvly-abc123"])

    assert result == 0
    assert "stored a key for tavily" in capsys.readouterr().out
    assert credentials.get_key("tavily") == "tvly-abc123"


def test_keys_remove_accepts_tavily(capsys):
    import credentials

    credentials.set_key("tavily", "tvly-abc123")

    result = cli.main(["keys", "remove", "tavily"])

    assert result == 0
    assert "removed the stored key for tavily" in capsys.readouterr().out
    assert credentials.get_key("tavily") is None


def test_config_show_uses_default_modes(capsys):
    result = cli.main(["config", "show"])

    assert result == 0
    output = capsys.readouterr().out
    assert "search-mode: explicit" in output
    assert "action-mode: confirm" in output


def test_config_set_search_mode_round_trip(capsys):
    import config

    result = cli.main(["config", "set", "search-mode", "autonomous"])

    assert result == 0
    assert "search-mode: autonomous" in capsys.readouterr().out
    assert config.get_search_mode() == "autonomous"

    cli.main(["config", "show"])
    assert "search-mode: autonomous" in capsys.readouterr().out


def test_config_set_action_mode_round_trip_alongside_search_mode(capsys):
    import config

    assert cli.main(["config", "set", "search-mode", "autonomous"]) == 0
    capsys.readouterr()

    result = cli.main(["config", "set", "action-mode", "autonomous"])

    assert result == 0
    assert "action-mode: autonomous" in capsys.readouterr().out
    assert config.as_dict() == {"search-mode": "autonomous", "action-mode": "autonomous"}

    cli.main(["config", "show"])
    output = capsys.readouterr().out
    assert "search-mode: autonomous" in output
    assert "action-mode: autonomous" in output


def test_config_set_rejects_invalid_value(capsys):
    result = cli.main(["config", "set", "search-mode", "always"])

    assert result == 1
    assert "search-mode must be one of: explicit, autonomous" in capsys.readouterr().out
