"""Chat tool wrapper around `local_models.list_models` -- lets dashboard chat
answer questions like "what models do I have installed" without leaving
chat for a shell. Read-only, no confirmation required (see
docs/prd/chat-cli-parity-tools.md, "Tools exposed").

Reuses `local_models.list_models` outright (no reimplemented catalog
reading) and serializes results with the exact same JSON row shape
`autumn models list --json` already prints -- see `cli._models_as_rows` --
so a chat answer and the CLI's own output can never drift apart.
"""

import json
from pathlib import Path
from typing import Any

import local_models
import paths

TOOL_NAME = "list_models"

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "List the local LLM models Autumn has installed on this machine (its managed "
            "model catalog), including each model's backend, on-disk path, context window, "
            "and whether it's the current default. Use this for questions about what models "
            "are installed locally -- not Groq's hosted models, which aren't installed on "
            "disk at all."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def _model_to_row(model: local_models.LocalModel) -> dict[str, Any]:
    """Same JSON shape as `autumn models list --json` (cli._models_as_rows)."""
    return {
        "name": model.name,
        "backend": model.backend,
        "path": str(model.path),
        "context_window": model.context_window,
        "is_default": model.is_default,
    }


def run_tool(arguments: dict[str, Any], *, catalog_root: Path | None = None) -> str:
    """Entry point `GroqRunner` calls to execute a `list_models` tool call.
    Takes no arguments beyond the (ignored) `arguments` dict -- the schema
    declares none -- and returns a JSON array of model rows, or `"[]"` if
    the catalog is empty. `catalog_root` defaults to the real managed
    catalog (`paths.models_root()`) when not given; tests override it the
    same way `search_docs.run_tool`'s `root` param does."""
    resolved_root = catalog_root if catalog_root is not None else paths.models_root()
    models = local_models.list_models(resolved_root)
    return json.dumps([_model_to_row(model) for model in models])
