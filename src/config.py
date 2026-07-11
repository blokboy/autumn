"""Persistent user preferences shared by the CLI and dashboard.

Small plaintext JSON under Autumn's XDG-style data root, with the same
owner-only `0600` write posture as `credentials.py`. The file stores user
preferences, not secrets, but keeping both stores equally private avoids a
surprising split in local-data behavior.
"""

import json
import os
import stat

import paths

SEARCH_MODE_KEY = "search-mode"
ACTION_MODE_KEY = "action-mode"
SEARCH_MODE_EXPLICIT = "explicit"
SEARCH_MODE_AUTONOMOUS = "autonomous"
ACTION_MODE_CONFIRM = "confirm"
ACTION_MODE_AUTONOMOUS = "autonomous"

CONFIG_KEYS = (SEARCH_MODE_KEY, ACTION_MODE_KEY)
SEARCH_MODES = (SEARCH_MODE_EXPLICIT, SEARCH_MODE_AUTONOMOUS)
ACTION_MODES = (ACTION_MODE_CONFIRM, ACTION_MODE_AUTONOMOUS)

_DEFAULTS = {
    SEARCH_MODE_KEY: SEARCH_MODE_EXPLICIT,
    ACTION_MODE_KEY: ACTION_MODE_CONFIRM,
}


class ConfigError(ValueError):
    """Raised when a caller tries to persist an unsupported preference value."""


def _path() -> Path:
    return paths.config_path()


def _load() -> dict[str, str]:
    try:
        payload = json.loads(_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items() if isinstance(key, str) and isinstance(value, str)}


def _save(values: dict[str, str]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values))
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def _get(name: str) -> str:
    return _load().get(name, _DEFAULTS[name])


def _set(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{name} must be one of: {', '.join(allowed)}")
    values = _load()
    values[name] = value
    _save(values)


def get_search_mode() -> str:
    return _get(SEARCH_MODE_KEY)


def set_search_mode(value: str) -> None:
    _set(SEARCH_MODE_KEY, value, SEARCH_MODES)


def get_action_mode() -> str:
    return _get(ACTION_MODE_KEY)


def set_action_mode(value: str) -> None:
    _set(ACTION_MODE_KEY, value, ACTION_MODES)


def as_dict() -> dict[str, str]:
    """All supported preferences with defaults filled in."""
    return {key: _get(key) for key in _DEFAULTS}
