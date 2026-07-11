"""Chat tool wrapper around `credentials.resolve_key`/`configured_providers`
-- lets dashboard chat answer questions like "which providers have a key
configured" without leaving chat for a shell. Read-only, no confirmation
required (see docs/prd/chat-cli-parity-tools.md, "Tools exposed").

Mirrors `autumn keys list`'s exact semantics (`cli._keys`'s `list` branch):
one entry per provider in `credentials.KNOWN_PROVIDERS`, each reporting only
whether a key is configured (stored, or via env var) -- **never** the key
value itself. This is the same guarantee `credentials.py` already provides
everywhere else; this module doesn't touch a key's actual value at all, so
there's nothing here that could leak one.
"""

import json
from typing import Any

import credentials

TOOL_NAME = "list_keys"

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "List which LLM provider API keys are configured (stored via `autumn keys add` "
            "or set as an environment variable) versus not configured. Never returns key "
            "values -- only a configured/not-configured status per provider. Use this for "
            "questions like which providers have a key set up."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def _provider_row(provider: str) -> dict[str, Any]:
    configured = credentials.resolve_key(provider, credentials.env_var_for_provider(provider)) is not None
    return {"provider": provider, "configured": configured}


def run_tool(arguments: dict[str, Any]) -> str:
    """Entry point `GroqRunner` calls to execute a `list_keys` tool call.
    Takes no arguments beyond the (ignored) `arguments` dict -- the schema
    declares none -- and returns a JSON array of `{"provider", "configured"}`
    rows, one per `credentials.KNOWN_PROVIDERS`, in that order. Never
    includes a key value."""
    return json.dumps([_provider_row(provider) for provider in credentials.KNOWN_PROVIDERS])
