"""Managed disk catalog for local LLM model files."""

import json
import shutil
from pathlib import Path

from models import LocalModel

_MANIFEST = "manifest.json"
_DEFAULT_BACKEND = "llama.cpp"


def _manifest_path(catalog_root: Path) -> Path:
    return catalog_root / _MANIFEST


def _load_manifest(catalog_root: Path) -> dict:
    try:
        with _manifest_path(catalog_root).open() as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"models": []}
    if not isinstance(payload, dict):
        return {"models": []}
    models = payload.get("models")
    if not isinstance(models, list):
        return {"models": []}
    return {"models": [model for model in models if isinstance(model, dict)]}


def _save_manifest(catalog_root: Path, manifest: dict) -> None:
    catalog_root.mkdir(parents=True, exist_ok=True)
    _manifest_path(catalog_root).write_text(json.dumps(manifest))


def _model_from_dict(raw: dict) -> LocalModel | None:
    name = raw.get("name")
    backend = raw.get("backend")
    path = raw.get("path")
    context_window = raw.get("context_window")
    is_default = raw.get("is_default")
    status = raw.get("status")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(backend, str) or not backend:
        return None
    if not isinstance(path, str) or not path:
        return None
    if context_window is not None and not isinstance(context_window, int):
        return None
    if status not in ("installed", "downloading"):
        # Missing (older manifests written before #20) or malformed -> a
        # fully-installed model is by far the common case, so that's the
        # safe default rather than silently treating an old manifest's
        # models as unavailable.
        status = "installed"
    return LocalModel(
        name=name,
        backend=backend,
        path=Path(path),
        context_window=context_window,
        is_default=bool(is_default),
        status=status,
    )


def _model_to_dict(model: LocalModel) -> dict:
    return {
        "name": model.name,
        "backend": model.backend,
        "path": str(model.path),
        "context_window": model.context_window,
        "is_default": model.is_default,
        "status": model.status,
    }


def list_models(catalog_root: Path) -> list[LocalModel]:
    manifest = _load_manifest(catalog_root)
    parsed = [_model_from_dict(raw) for raw in manifest["models"]]
    return [model for model in parsed if model is not None]


def install_model(
    catalog_root: Path,
    *,
    name: str,
    source_path: Path,
    backend: str = _DEFAULT_BACKEND,
    context_window: int | None = None,
) -> LocalModel:
    models = [model for model in list_models(catalog_root) if model.name != name]
    destination_dir = catalog_root / name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source_path.name
    shutil.copy2(source_path, destination)

    installed = LocalModel(
        name=name,
        backend=backend,
        path=destination,
        context_window=context_window,
        is_default=not any(model.is_default for model in models),
    )
    models.append(installed)
    _save_manifest(catalog_root, {"models": [_model_to_dict(model) for model in models]})
    return installed


def mark_downloading(
    catalog_root: Path,
    *,
    name: str,
    backend: str = _DEFAULT_BACKEND,
    context_window: int | None = None,
) -> LocalModel:
    """Registers `name` in the manifest as "downloading" -- occupying a
    catalog slot (so `list_models` is non-empty, meaning the first-run
    picker won't reappear, and a default can already be recorded) before any
    bytes have actually landed on disk. `is_default` is computed with the
    exact same "first one in an otherwise-default-less catalog" rule
    `install_model` uses below, so a picked model that becomes the eventual
    default is already marked as such while it's still downloading -- which
    is the "default chosen but not yet runnable" state `model_router.
    choose_model` (see `_availability`'s `status == "downloading"` check)
    falls through on, mirroring how a missing local runtime already falls
    through today.

    `model_downloader.download_and_install`'s call into `install_model` once
    the download finishes overwrites this placeholder outright (same name ->
    replaced), flipping it back to `status="installed"` with a real path."""
    models = [model for model in list_models(catalog_root) if model.name != name]
    placeholder = LocalModel(
        name=name,
        backend=backend,
        path=catalog_root / name / ".downloading",
        context_window=context_window,
        is_default=not any(model.is_default for model in models),
        status="downloading",
    )
    models.append(placeholder)
    _save_manifest(catalog_root, {"models": [_model_to_dict(model) for model in models]})
    return placeholder


def set_default(catalog_root: Path, name: str) -> LocalModel:
    models = list_models(catalog_root)
    selected: LocalModel | None = None
    updated = []
    for model in models:
        is_default = model.name == name
        candidate = LocalModel(
            name=model.name,
            backend=model.backend,
            path=model.path,
            context_window=model.context_window,
            is_default=is_default,
            status=model.status,
        )
        if is_default:
            selected = candidate
        updated.append(candidate)
    if selected is None:
        raise ValueError(f"Unknown local model: {name}")
    _save_manifest(catalog_root, {"models": [_model_to_dict(model) for model in updated]})
    return selected


def get_default(catalog_root: Path) -> LocalModel | None:
    for model in list_models(catalog_root):
        if model.is_default:
            return model
    return None


def remove_model(catalog_root: Path, name: str) -> None:
    models = [model for model in list_models(catalog_root) if model.name != name]
    shutil.rmtree(catalog_root / name, ignore_errors=True)
    if models and not any(model.is_default for model in models):
        first = models[0]
        models[0] = LocalModel(
            name=first.name,
            backend=first.backend,
            path=first.path,
            context_window=first.context_window,
            is_default=True,
            status=first.status,
        )
    _save_manifest(catalog_root, {"models": [_model_to_dict(model) for model in models]})
