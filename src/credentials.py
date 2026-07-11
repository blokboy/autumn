"""Local storage for provider API keys entered via `autumn keys add`, so a
`GROQ_API_KEY`-style env var isn't the only way to configure a provider.

Plaintext JSON, matching the same on-disk security posture as an env var
sitting in a shell profile (both are readable by the local user account,
neither is protected by an OS keychain) -- kept readable/writable only by
the owning user via a `0600` permission set on every write.

`resolve_key` is the single source of truth both `groq_policy._is_signed_in`
(catalog visibility) and `GroqRunner`'s real-client construction (actually
authenticating) call through, so a stored key and an env var can never
disagree about whether a provider is usable."""

import json
import os
import stat
from pathlib import Path

import paths

# Providers autumn knows the *names* of, for listing purposes (`autumn keys
# list`, and the `list_keys` chat tool), even before they're wired up as real
# completion providers (see #21) -- keeps both surfaces showing the full
# expected set rather than only whatever happens to already have a stored
# key. Also doubles as the `choices=` set for `autumn keys add`/`remove`.
KNOWN_PROVIDERS = ("groq", "anthropic", "openai", "tavily")


def env_var_for_provider(provider: str) -> str:
    """`groq` -> `GROQ_API_KEY`, matching the naming convention every
    provider's own module already reads its env var by (see
    `groq_policy.ENV_VAR`)."""
    return f"{provider.upper()}_API_KEY"


def _path() -> Path:
    return paths.credentials_path()


def _load() -> dict[str, str]:
    try:
        payload = json.loads(_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items() if isinstance(key, str) and isinstance(value, str)}


def _save(credentials: dict[str, str]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(credentials))
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def set_key(provider: str, api_key: str) -> None:
    credentials = _load()
    credentials[provider] = api_key
    _save(credentials)


def remove_key(provider: str) -> bool:
    """Returns whether a key was actually removed (`False` if none was stored)."""
    credentials = _load()
    if provider not in credentials:
        return False
    del credentials[provider]
    _save(credentials)
    return True


def get_key(provider: str) -> str | None:
    return _load().get(provider)


def configured_providers() -> set[str]:
    return set(_load())


def resolve_key(provider: str, env_var: str) -> str | None:
    """The stored key for `provider`, if any; otherwise `env_var` read from
    the environment. A stored key always wins, so re-entering a key via
    `autumn keys add` overrides a stale or shared shell env var without
    requiring the user to unset it first."""
    stored = get_key(provider)
    if stored:
        return stored
    return os.environ.get(env_var) or None
