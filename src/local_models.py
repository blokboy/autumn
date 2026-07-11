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
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(backend, str) or not backend:
        return None
    if not isinstance(path, str) or not path:
        return None
    if context_window is not None and not isinstance(context_window, int):
        return None
    return LocalModel(
        name=name,
        backend=backend,
        path=Path(path),
        context_window=context_window,
        is_default=bool(is_default),
    )


def _model_to_dict(model: LocalModel) -> dict:
    return {
        "name": model.name,
        "backend": model.backend,
        "path": str(model.path),
        "context_window": model.context_window,
        "is_default": model.is_default,
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
        )
    _save_manifest(catalog_root, {"models": [_model_to_dict(model) for model in models]})
