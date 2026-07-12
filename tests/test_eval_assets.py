import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import eval_assets


def _write_bundle(path: Path, *, executable_name: str | None = None) -> str:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("examples.jsonl", '{"input":"2+2","answer":"4"}\n')
        bundle.writestr("README.md", "# Tiny arithmetic eval\n")
        if executable_name is not None:
            bundle.writestr(executable_name, "def score():\n    return 1\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, *, checksum: str, url: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "asset_id": "tiny-arithmetic",
                        "version": "2026.07.12",
                        "label": "Tiny arithmetic",
                        "description": "A tiny arithmetic eval for smoke tests.",
                        "supported_metrics": ["exact_match", "contains"],
                        "default_metric": "exact_match",
                        "size_bytes": 1234,
                        "license": "CC-BY-4.0",
                        "provenance": "Synthetic examples generated for Autumn tests.",
                        "url": url,
                        "sha256": checksum,
                    }
                ],
            }
        )
    )


def test_load_remote_manifest_parses_asset_metadata(tmp_path):
    bundle = tmp_path / "tiny.zip"
    checksum = _write_bundle(bundle)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, checksum=checksum, url=bundle.as_uri())

    manifest = eval_assets.load_remote_manifest(manifest_path)

    asset = manifest.assets[0]
    assert asset.asset_id == "tiny-arithmetic"
    assert asset.version == "2026.07.12"
    assert asset.supported_metrics == ("exact_match", "contains")
    assert asset.license == "CC-BY-4.0"
    assert asset.provenance == "Synthetic examples generated for Autumn tests."


def test_install_asset_verifies_checksum_before_writing(tmp_path):
    bundle = tmp_path / "tiny.zip"
    _write_bundle(bundle)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, checksum="0" * 64, url=bundle.as_uri())
    manifest = eval_assets.load_remote_manifest(manifest_path)

    with pytest.raises(eval_assets.EvalAssetError, match="checksum"):
        eval_assets.install_asset(tmp_path / "assets", manifest.assets[0])

    assert eval_assets.list_installed_assets(tmp_path / "assets") == []


def test_install_asset_is_immutable_per_version_and_preserves_metadata(tmp_path):
    bundle = tmp_path / "tiny.zip"
    checksum = _write_bundle(bundle)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, checksum=checksum, url=bundle.as_uri())
    manifest = eval_assets.load_remote_manifest(manifest_path)

    installed = eval_assets.install_asset(tmp_path / "assets", manifest.assets[0])

    assert installed.asset_id == "tiny-arithmetic"
    assert installed.version == "2026.07.12"
    assert installed.license == "CC-BY-4.0"
    assert installed.provenance == "Synthetic examples generated for Autumn tests."
    assert (installed.path / "examples.jsonl").exists()

    with pytest.raises(eval_assets.EvalAssetError, match="already installed"):
        eval_assets.install_asset(tmp_path / "assets", manifest.assets[0])


def test_install_asset_rejects_executable_bundle_members(tmp_path):
    bundle = tmp_path / "tiny.zip"
    checksum = _write_bundle(bundle, executable_name="metrics/custom_metric.py")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, checksum=checksum, url=bundle.as_uri())
    manifest = eval_assets.load_remote_manifest(manifest_path)

    with pytest.raises(eval_assets.EvalAssetError, match="data-only"):
        eval_assets.install_asset(tmp_path / "assets", manifest.assets[0])

    assert eval_assets.list_installed_assets(tmp_path / "assets") == []


def test_remove_asset_deletes_only_selected_version(tmp_path):
    assets_root = tmp_path / "assets"
    first_bundle = tmp_path / "first.zip"
    first_checksum = _write_bundle(first_bundle)
    second_bundle = tmp_path / "second.zip"
    second_checksum = _write_bundle(second_bundle)
    first = eval_assets.RemoteEvalAsset(
        asset_id="tiny-arithmetic",
        version="2026.07.12",
        label="Tiny arithmetic",
        description="First version",
        supported_metrics=("exact_match",),
        default_metric="exact_match",
        size_bytes=1,
        license="CC-BY-4.0",
        provenance="Synthetic",
        url=first_bundle.as_uri(),
        sha256=first_checksum,
    )
    second = eval_assets.RemoteEvalAsset(
        asset_id="tiny-arithmetic",
        version="2026.07.13",
        label="Tiny arithmetic",
        description="Second version",
        supported_metrics=("exact_match",),
        default_metric="exact_match",
        size_bytes=1,
        license="CC-BY-4.0",
        provenance="Synthetic",
        url=second_bundle.as_uri(),
        sha256=second_checksum,
    )
    eval_assets.install_asset(assets_root, first)
    eval_assets.install_asset(assets_root, second)

    eval_assets.remove_asset(assets_root, "tiny-arithmetic", "2026.07.12")

    installed = eval_assets.list_installed_assets(assets_root)
    assert [(asset.asset_id, asset.version) for asset in installed] == [("tiny-arithmetic", "2026.07.13")]
