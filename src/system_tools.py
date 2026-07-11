"""Tool equivalents of `autumn models`/`autumn keys` mutating subcommands
(see docs/prd/chat-cli-parity-tools.md) -- install/set-default/remove a
local model, add/remove a stored provider API key. Every tool here is
mutating, so `GroqRunner` gates each one behind a confirmation before
calling `execute` (see `groq_runner.py`'s `_MUTATING_TOOLS` wiring); this
module itself has no confirmation logic of its own, only the mutation and
the human-readable confirmation message describing it.

Read-only equivalents (`list_models`, `list_keys`, `list_runs`) live
elsewhere (#29) -- they need no confirmation and no target-description
machinery, so folding them in here would blur what actually needs gating.

`install_model` is scoped to the five curated models
(`curated_models.CURATED_MODELS`) only, not an arbitrary local file path --
the model has no visibility into what's actually on the user's disk (see
docs/prd/chat-cli-parity-tools.md, "Scope this round").
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import credentials
import local_models
import model_downloader
import paths
from curated_models import CURATED_MODELS

# Mirrors cli.py's `_KNOWN_PROVIDERS` -- duplicated rather than imported
# since that name is private-by-convention to the CLI module and this is a
# small, closed, rarely-changing list; promoting it to a shared public
# constant is a reasonable follow-up if a third consumer ever needs it.
_KNOWN_PROVIDERS = ("groq", "anthropic", "openai", "tavily")


class SystemToolError(RuntimeError):
    """Raised when a mutating tool's arguments are invalid or its target
    doesn't exist -- distinct from a successful-but-declined confirmation,
    which is not an error (see `groq_runner.py`)."""


@dataclass(frozen=True)
class MutatingTool:
    """One chat-callable mutating tool: its Groq function-calling schema,
    whether it's destructive (gates the confirmation modal's `initial_delay`
    -- see `screens/confirm_screen.py`), how to render a specific
    confirmation message from its arguments, and how to actually perform
    the mutation once confirmed."""

    name: str
    schema: dict[str, Any]
    destructive: bool
    confirmation_message: Callable[[dict[str, Any]], str]
    execute: Callable[[dict[str, Any], Path], str]


def _require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise SystemToolError(f"missing or invalid {key!r} argument")
    return value


# --- install_model ---------------------------------------------------------

INSTALL_MODEL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "install_model",
        "description": (
            "Download and install one of Autumn's curated local models by name. "
            "Only the curated catalog can be installed this way -- not an "
            "arbitrary file path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": f"One of: {', '.join(entry.name for entry in CURATED_MODELS)}.",
                }
            },
            "required": ["model_name"],
        },
    },
}


def _find_curated(model_name: str):
    for entry in CURATED_MODELS:
        if entry.name == model_name:
            return entry
    return None


def _install_model_message(arguments: dict[str, Any]) -> str:
    model_name = arguments.get("model_name", "?")
    entry = _find_curated(model_name) if isinstance(model_name, str) else None
    if entry is None:
        return f"Install `{model_name}`?"
    return f"Install `{entry.name}` ({entry.vendor}, ~{entry.approx_size_gb:.1f}GB) from Hugging Face?"


def _install_model_execute(arguments: dict[str, Any], catalog_root: Path) -> str:
    model_name = _require_str(arguments, "model_name")
    entry = _find_curated(model_name)
    if entry is None:
        available = ", ".join(m.name for m in CURATED_MODELS)
        raise SystemToolError(f"{model_name!r} is not in the curated catalog. Available: {available}")
    installed = model_downloader.download_and_install(catalog_root, entry)
    default_note = " (set as default)" if installed.is_default else ""
    return f"Installed {installed.name}{default_note}."


# --- set_default_model -------------------------------------------------------

SET_DEFAULT_MODEL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "set_default_model",
        "description": "Set an already-installed local model as the default.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "An installed local model's name."}
            },
            "required": ["model_name"],
        },
    },
}


def _set_default_model_message(arguments: dict[str, Any]) -> str:
    return f"Set `{arguments.get('model_name', '?')}` as the default model?"


def _set_default_model_execute(arguments: dict[str, Any], catalog_root: Path) -> str:
    model_name = _require_str(arguments, "model_name")
    try:
        selected = local_models.set_default(catalog_root, model_name)
    except ValueError as exc:
        raise SystemToolError(str(exc)) from exc
    return f"{selected.name} is now the default model."


# --- add_key -----------------------------------------------------------------

ADD_KEY_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add_key",
        "description": "Store an API key for a provider (groq, anthropic, openai, tavily).",
        "parameters": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": list(_KNOWN_PROVIDERS)},
                "api_key": {"type": "string", "description": "The provider's API key."},
            },
            "required": ["provider", "api_key"],
        },
    },
}


def _add_key_message(arguments: dict[str, Any]) -> str:
    # Deliberately never includes the key value itself in the confirmation
    # text -- only the provider name.
    return f"Store an API key for {arguments.get('provider', '?')}?"


def _add_key_execute(arguments: dict[str, Any], _catalog_root: Path) -> str:
    provider = _require_str(arguments, "provider")
    api_key = _require_str(arguments, "api_key")
    if provider not in _KNOWN_PROVIDERS:
        raise SystemToolError(f"unknown provider {provider!r}. Known providers: {', '.join(_KNOWN_PROVIDERS)}")
    credentials.set_key(provider, api_key)
    return f"Stored a key for {provider}."


# --- remove_model (destructive) ----------------------------------------------

REMOVE_MODEL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "remove_model",
        "description": "Uninstall an installed local model.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "An installed local model's name."}
            },
            "required": ["model_name"],
        },
    },
}


def _remove_model_message(arguments: dict[str, Any]) -> str:
    return f"Uninstall `{arguments.get('model_name', '?')}`? This cannot be undone."


def _remove_model_execute(arguments: dict[str, Any], catalog_root: Path) -> str:
    model_name = _require_str(arguments, "model_name")
    installed_names = {model.name for model in local_models.list_models(catalog_root)}
    if model_name not in installed_names:
        raise SystemToolError(f"no installed model named {model_name!r}")
    local_models.remove_model(catalog_root, model_name)
    return f"Removed {model_name}."


# --- remove_key (destructive) -------------------------------------------------

REMOVE_KEY_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "remove_key",
        "description": "Remove a stored API key for a provider.",
        "parameters": {
            "type": "object",
            "properties": {"provider": {"type": "string", "enum": list(_KNOWN_PROVIDERS)}},
            "required": ["provider"],
        },
    },
}


def _remove_key_message(arguments: dict[str, Any]) -> str:
    return f"Remove the locally-stored {arguments.get('provider', '?')} API key? This cannot be undone."


def _remove_key_execute(arguments: dict[str, Any], _catalog_root: Path) -> str:
    provider = _require_str(arguments, "provider")
    if provider not in _KNOWN_PROVIDERS:
        raise SystemToolError(f"unknown provider {provider!r}. Known providers: {', '.join(_KNOWN_PROVIDERS)}")
    removed = credentials.remove_key(provider)
    if removed:
        return f"Removed the stored key for {provider}."
    return f"No stored key for {provider} (an env var, if set, still applies)."


MUTATING_TOOLS: list[MutatingTool] = [
    MutatingTool(
        name="install_model",
        schema=INSTALL_MODEL_SCHEMA,
        destructive=False,
        confirmation_message=_install_model_message,
        execute=_install_model_execute,
    ),
    MutatingTool(
        name="set_default_model",
        schema=SET_DEFAULT_MODEL_SCHEMA,
        destructive=False,
        confirmation_message=_set_default_model_message,
        execute=_set_default_model_execute,
    ),
    MutatingTool(
        name="add_key",
        schema=ADD_KEY_SCHEMA,
        destructive=False,
        confirmation_message=_add_key_message,
        execute=_add_key_execute,
    ),
    MutatingTool(
        name="remove_model",
        schema=REMOVE_MODEL_SCHEMA,
        destructive=True,
        confirmation_message=_remove_model_message,
        execute=_remove_model_execute,
    ),
    MutatingTool(
        name="remove_key",
        schema=REMOVE_KEY_SCHEMA,
        destructive=True,
        confirmation_message=_remove_key_message,
        execute=_remove_key_execute,
    ),
]


def run_tool(name: str, arguments: dict[str, Any], *, catalog_root: Path | None = None) -> str:
    """Executes an already-confirmed mutating tool by name. Propagates
    `SystemToolError` on invalid arguments or a missing target; callers
    (`groq_runner.py`) are responsible for having already confirmed the
    action before calling this."""
    resolved_root = catalog_root if catalog_root is not None else paths.models_root()
    for tool in MUTATING_TOOLS:
        if tool.name == name:
            return tool.execute(arguments, resolved_root)
    raise SystemToolError(f"unknown tool {name!r}")
