"""Managed catalog for immutable prompt optimization eval assets."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_optimization_metrics import list_builtin_metrics

REMOTE_MANIFEST_SCHEMA_VERSION = 1
INSTALLED_ASSET_SCHEMA_VERSION = 1
DEFAULT_REMOTE_MANIFEST_PATH = Path(__file__).with_name("eval_assets_manifest.json")
_INSTALLED_METADATA = "autumn_eval_asset.json"
_EXECUTABLE_SUFFIXES = {
    ".bat",
    ".cmd",
    ".dll",
    ".dylib",
    ".exe",
    ".js",
    ".mjs",
    ".py",
    ".pyc",
    ".pyo",
    ".sh",
    ".so",
}


class EvalAssetError(ValueError):
    """Raised when an eval asset manifest or install operation is invalid."""


@dataclass(frozen=True)
class RemoteEvalAsset:
    """One immutable eval asset version available for installation."""

    asset_id: str
    version: str
    label: str
    description: str
    supported_metrics: tuple[str, ...]
    default_metric: str
    size_bytes: int
    license: str
    provenance: str
    url: str
    sha256: str


@dataclass(frozen=True)
class RemoteEvalManifest:
    """Static manifest of remotely available eval asset versions."""

    assets: tuple[RemoteEvalAsset, ...]
    schema_version: int = REMOTE_MANIFEST_SCHEMA_VERSION


@dataclass(frozen=True)
class InstalledEvalAsset:
    """An immutable eval asset version installed into the local catalog."""

    asset_id: str
    version: str
    label: str
    description: str
    supported_metrics: tuple[str, ...]
    default_metric: str
    size_bytes: int
    license: str
    provenance: str
    sha256: str
    path: Path


def load_remote_manifest(path: Path = DEFAULT_REMOTE_MANIFEST_PATH) -> RemoteEvalManifest:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise EvalAssetError(f"eval asset manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvalAssetError(f"eval asset manifest is invalid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise EvalAssetError("eval asset manifest must be an object")
    if payload.get("schema_version") != REMOTE_MANIFEST_SCHEMA_VERSION:
        raise EvalAssetError("unsupported eval asset manifest schema_version")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise EvalAssetError("eval asset manifest assets must be a list")
    return RemoteEvalManifest(assets=tuple(_remote_asset_from_dict(asset) for asset in assets))


def list_installed_assets(assets_root: Path) -> list[InstalledEvalAsset]:
    installed = []
    if not assets_root.exists():
        return []
    for metadata_path in sorted(assets_root.glob(f"*/*/{_INSTALLED_METADATA}")):
        try:
            payload = json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        asset = _installed_asset_from_dict(payload, path=metadata_path.parent)
        if asset is not None:
            installed.append(asset)
    return installed


def install_asset(assets_root: Path, asset: RemoteEvalAsset) -> InstalledEvalAsset:
    destination = _asset_version_dir(assets_root, asset.asset_id, asset.version)
    if destination.exists():
        raise EvalAssetError(f"eval asset already installed: {asset.asset_id}@{asset.version}")

    assets_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="autumn-eval-", dir=assets_root) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_path = tmp_dir / "asset.zip"
        _download(asset.url, archive_path)
        digest = _sha256(archive_path)
        if digest.lower() != asset.sha256.lower():
            raise EvalAssetError(
                f"checksum mismatch for {asset.asset_id}@{asset.version}: expected {asset.sha256}, got {digest}"
            )
        _validate_bundle(archive_path)
        staging_dir = tmp_dir / "staged"
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(staging_dir)
        metadata = _installed_asset_to_dict(asset, path=destination)
        (staging_dir / _INSTALLED_METADATA).write_text(json.dumps(metadata, indent=2))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_dir), destination)

    installed = _installed_asset_from_dict(metadata, path=destination)
    if installed is None:
        raise EvalAssetError("installed eval asset metadata could not be read")
    return installed


def remove_asset(assets_root: Path, asset_id: str, version: str) -> bool:
    destination = _asset_version_dir(assets_root, asset_id, version)
    if not destination.exists():
        return False
    shutil.rmtree(destination)
    parent = destination.parent
    try:
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass
    return True


def parse_asset_selector(selector: str) -> tuple[str, str | None]:
    if not selector:
        raise EvalAssetError("eval asset selector is required")
    asset_id, separator, version = selector.partition("@")
    if not asset_id:
        raise EvalAssetError("eval asset id is required")
    if separator and not version:
        raise EvalAssetError("eval asset version is required after @")
    return asset_id, version or None


def select_remote_asset(manifest: RemoteEvalManifest, selector: str) -> RemoteEvalAsset:
    asset_id, version = parse_asset_selector(selector)
    matches = [asset for asset in manifest.assets if asset.asset_id == asset_id]
    if version is not None:
        for asset in matches:
            if asset.version == version:
                return asset
        raise EvalAssetError(f"unknown eval asset version: {asset_id}@{version}")
    if not matches:
        raise EvalAssetError(f"unknown eval asset: {asset_id}")
    return sorted(matches, key=lambda asset: asset.version)[-1]


def select_installed_asset(assets_root: Path, selector: str) -> InstalledEvalAsset:
    asset_id, version = parse_asset_selector(selector)
    matches = [asset for asset in list_installed_assets(assets_root) if asset.asset_id == asset_id]
    if version is not None:
        for asset in matches:
            if asset.version == version:
                return asset
        raise EvalAssetError(f"eval asset is not installed: {asset_id}@{version}")
    if not matches:
        raise EvalAssetError(f"eval asset is not installed: {asset_id}")
    return sorted(matches, key=lambda asset: asset.version)[-1]


def _remote_asset_from_dict(payload: object) -> RemoteEvalAsset:
    if not isinstance(payload, dict):
        raise EvalAssetError("eval asset manifest entry must be an object")
    supported_metrics = _required_str_tuple(payload, "supported_metrics")
    default_metric = _required_str(payload, "default_metric")
    built_ins = set(list_builtin_metrics())
    unknown_metrics = [metric for metric in supported_metrics if metric not in built_ins]
    if unknown_metrics:
        raise EvalAssetError(f"unsupported built-in metric in eval asset manifest: {unknown_metrics[0]}")
    if default_metric not in supported_metrics:
        raise EvalAssetError("default_metric must be listed in supported_metrics")
    size_bytes = payload.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise EvalAssetError("size_bytes must be a non-negative integer")
    sha256 = _required_str(payload, "sha256")
    if len(sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in sha256):
        raise EvalAssetError("sha256 must be a 64-character hex digest")
    return RemoteEvalAsset(
        asset_id=_required_str(payload, "asset_id"),
        version=_required_str(payload, "version"),
        label=_required_str(payload, "label"),
        description=_required_str(payload, "description"),
        supported_metrics=supported_metrics,
        default_metric=default_metric,
        size_bytes=size_bytes,
        license=_required_str(payload, "license"),
        provenance=_required_str(payload, "provenance"),
        url=_required_str(payload, "url"),
        sha256=sha256,
    )


def _installed_asset_from_dict(payload: object, *, path: Path) -> InstalledEvalAsset | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != INSTALLED_ASSET_SCHEMA_VERSION:
        return None
    try:
        return InstalledEvalAsset(
            asset_id=_required_str(payload, "asset_id"),
            version=_required_str(payload, "version"),
            label=_required_str(payload, "label"),
            description=_required_str(payload, "description"),
            supported_metrics=_required_str_tuple(payload, "supported_metrics"),
            default_metric=_required_str(payload, "default_metric"),
            size_bytes=_required_int(payload, "size_bytes"),
            license=_required_str(payload, "license"),
            provenance=_required_str(payload, "provenance"),
            sha256=_required_str(payload, "sha256"),
            path=path,
        )
    except EvalAssetError:
        return None


def _installed_asset_to_dict(asset: RemoteEvalAsset, *, path: Path) -> dict[str, Any]:
    return {
        "schema_version": INSTALLED_ASSET_SCHEMA_VERSION,
        "asset_id": asset.asset_id,
        "version": asset.version,
        "label": asset.label,
        "description": asset.description,
        "supported_metrics": list(asset.supported_metrics),
        "default_metric": asset.default_metric,
        "size_bytes": asset.size_bytes,
        "license": asset.license,
        "provenance": asset.provenance,
        "sha256": asset.sha256,
        "path": str(path),
    }


def _download(url: str, destination: Path) -> None:
    if url.startswith("file://"):
        shutil.copy2(urllib.request.url2pathname(url.removeprefix("file://")), destination)
        return
    with urllib.request.urlopen(url) as response:  # noqa: S310 - URLs come from the static eval manifest
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def _validate_bundle(archive_path: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if not names:
                raise EvalAssetError("eval asset bundle is empty")
            for name in names:
                parts = Path(name).parts
                if Path(name).is_absolute() or ".." in parts:
                    raise EvalAssetError("eval asset bundle contains an unsafe path")
                if "__pycache__" in parts or Path(name).suffix.lower() in _EXECUTABLE_SUFFIXES:
                    raise EvalAssetError("eval asset bundles must be data-only")
    except zipfile.BadZipFile as exc:
        raise EvalAssetError("eval asset bundle must be a zip archive") from exc


def _asset_version_dir(assets_root: Path, asset_id: str, version: str) -> Path:
    return assets_root / asset_id / version


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise EvalAssetError(f"{key} is required")
    return value


def _required_str_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise EvalAssetError(f"{key} must be a non-empty list of strings")
    return tuple(value)


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise EvalAssetError(f"{key} must be an integer")
    return value
