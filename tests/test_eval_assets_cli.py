import hashlib
import json
import zipfile
from pathlib import Path

import cli
import eval_assets


def _write_bundle(path: Path) -> str:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("examples.jsonl", '{"input":"hello","answer":"world"}\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, *, bundle: Path, version: str = "2026.07.12") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "asset_id": "tiny-smoke",
                        "version": version,
                        "label": "Tiny smoke",
                        "description": "A compact smoke-test eval.",
                        "supported_metrics": ["exact_match", "contains"],
                        "default_metric": "exact_match",
                        "size_bytes": bundle.stat().st_size,
                        "license": "MIT",
                        "provenance": "Checked-in synthetic fixture.",
                        "url": bundle.as_uri(),
                        "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )


def test_evals_available_prints_static_manifest_metadata(tmp_path, monkeypatch, capsys):
    bundle = tmp_path / "tiny.zip"
    _write_bundle(bundle)
    manifest = tmp_path / "evals.json"
    _write_manifest(manifest, bundle=bundle)
    monkeypatch.setattr(cli.eval_assets, "DEFAULT_REMOTE_MANIFEST_PATH", manifest)

    result = cli.main(["evals", "available"])

    assert result == 0
    output = capsys.readouterr().out
    assert "tiny-smoke" in output
    assert "2026.07.12" in output
    assert "exact_match, contains" in output
    assert "MIT" in output
    assert "Checked-in synthetic fixture." in output


def test_evals_install_list_and_remove_round_trip(tmp_path, monkeypatch, capsys):
    assets_root = tmp_path / "eval-assets"
    bundle = tmp_path / "tiny.zip"
    _write_bundle(bundle)
    manifest = tmp_path / "evals.json"
    _write_manifest(manifest, bundle=bundle)
    monkeypatch.setattr(cli.paths, "eval_assets_root", lambda: assets_root)
    monkeypatch.setattr(cli.eval_assets, "DEFAULT_REMOTE_MANIFEST_PATH", manifest)

    install_result = cli.main(["evals", "install", "tiny-smoke@2026.07.12"])
    assert install_result == 0
    assert "installed tiny-smoke@2026.07.12" in capsys.readouterr().out

    list_result = cli.main(["evals", "list"])
    assert list_result == 0
    output = capsys.readouterr().out
    assert "tiny-smoke" in output
    assert "2026.07.12" in output
    assert "MIT" in output
    assert "Checked-in synthetic fixture." in output

    installed = eval_assets.list_installed_assets(assets_root)
    assert installed[0].license == "MIT"
    assert installed[0].provenance == "Checked-in synthetic fixture."

    remove_result = cli.main(["evals", "remove", "tiny-smoke@2026.07.12"])
    assert remove_result == 0
    assert "removed tiny-smoke@2026.07.12" in capsys.readouterr().out
    assert eval_assets.list_installed_assets(assets_root) == []


def test_evals_remove_without_version_removes_latest_only(tmp_path, monkeypatch, capsys):
    assets_root = tmp_path / "eval-assets"
    monkeypatch.setattr(cli.paths, "eval_assets_root", lambda: assets_root)
    for version in ("2026.07.12", "2026.07.13"):
        bundle = tmp_path / f"{version}.zip"
        _write_bundle(bundle)
        manifest = eval_assets.RemoteEvalAsset(
            asset_id="tiny-smoke",
            version=version,
            label="Tiny smoke",
            description="A compact smoke-test eval.",
            supported_metrics=("exact_match",),
            default_metric="exact_match",
            size_bytes=bundle.stat().st_size,
            license="MIT",
            provenance="Synthetic",
            url=bundle.as_uri(),
            sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        )
        eval_assets.install_asset(assets_root, manifest)

    result = cli.main(["evals", "remove", "tiny-smoke"])

    assert result == 0
    assert "removed tiny-smoke@2026.07.13" in capsys.readouterr().out
    installed = eval_assets.list_installed_assets(assets_root)
    assert [(asset.asset_id, asset.version) for asset in installed] == [("tiny-smoke", "2026.07.12")]


def test_evals_install_reports_checksum_mismatch_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    assets_root = tmp_path / "eval-assets"
    bundle = tmp_path / "tiny.zip"
    _write_bundle(bundle)
    manifest = tmp_path / "evals.json"
    _write_manifest(manifest, bundle=bundle)
    # Corrupt the manifest's checksum after computing it from the real bundle.
    payload = json.loads(manifest.read_text())
    payload["assets"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    monkeypatch.setattr(cli.paths, "eval_assets_root", lambda: assets_root)
    monkeypatch.setattr(cli.eval_assets, "DEFAULT_REMOTE_MANIFEST_PATH", manifest)

    result = cli.main(["evals", "install", "tiny-smoke@2026.07.12"])

    assert result == 1
    assert "checksum" in capsys.readouterr().out
    assert eval_assets.list_installed_assets(assets_root) == []


def test_evals_install_unknown_asset_reports_error_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    assets_root = tmp_path / "eval-assets"
    bundle = tmp_path / "tiny.zip"
    _write_bundle(bundle)
    manifest = tmp_path / "evals.json"
    _write_manifest(manifest, bundle=bundle)
    monkeypatch.setattr(cli.paths, "eval_assets_root", lambda: assets_root)
    monkeypatch.setattr(cli.eval_assets, "DEFAULT_REMOTE_MANIFEST_PATH", manifest)

    result = cli.main(["evals", "install", "does-not-exist@2026.07.12"])

    assert result == 1
    assert "unknown eval asset" in capsys.readouterr().out


def test_evals_install_already_installed_reports_error_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    assets_root = tmp_path / "eval-assets"
    bundle = tmp_path / "tiny.zip"
    _write_bundle(bundle)
    manifest = tmp_path / "evals.json"
    _write_manifest(manifest, bundle=bundle)
    monkeypatch.setattr(cli.paths, "eval_assets_root", lambda: assets_root)
    monkeypatch.setattr(cli.eval_assets, "DEFAULT_REMOTE_MANIFEST_PATH", manifest)
    assert cli.main(["evals", "install", "tiny-smoke@2026.07.12"]) == 0
    capsys.readouterr()

    result = cli.main(["evals", "install", "tiny-smoke@2026.07.12"])

    assert result == 1
    assert "already installed" in capsys.readouterr().out
    installed = eval_assets.list_installed_assets(assets_root)
    assert len(installed) == 1


def test_evals_remove_unknown_version_reports_error_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    assets_root = tmp_path / "eval-assets"
    monkeypatch.setattr(cli.paths, "eval_assets_root", lambda: assets_root)

    result = cli.main(["evals", "remove", "tiny-smoke@2099.01.01"])

    assert result == 1
    assert "not installed" in capsys.readouterr().out


def test_evals_list_shows_no_assets_message_when_empty(tmp_path, monkeypatch, capsys):
    assets_root = tmp_path / "eval-assets"
    monkeypatch.setattr(cli.paths, "eval_assets_root", lambda: assets_root)

    result = cli.main(["evals", "list"])

    assert result == 0
    assert "no eval assets installed" in capsys.readouterr().out
