"""Policy-based prompt routing for dashboard chat replies."""

from pathlib import Path

from autumn import local_llm, local_models
from autumn.models import LocalModel, ModelChoice


def choose_model(
    *,
    prompt: str,
    catalog_root: Path,
    available_models: list[LocalModel] | None = None,
) -> ModelChoice:
    models = available_models if available_models is not None else local_models.list_models(catalog_root)
    installed_default = next((model for model in models if model.is_default), None)
    if installed_default is not None:
        return ModelChoice(
            name=installed_default.name,
            backend=installed_default.backend,
            path=installed_default.path,
            reason="installed default",
        )
    return ModelChoice(
        name=local_llm.OFFLINE_TINY_MODEL,
        backend="builtin",
        path=None,
        reason="offline fallback",
    )
