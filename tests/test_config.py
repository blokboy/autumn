import stat

import pytest

import config
import paths


def test_defaults_when_no_config_file_exists():
    assert config.get_search_mode() == "explicit"
    assert config.get_action_mode() == "confirm"
    assert config.as_dict() == {"search-mode": "explicit", "action-mode": "confirm"}


def test_set_search_mode_persists_value():
    config.set_search_mode("autonomous")

    assert config.get_search_mode() == "autonomous"


def test_set_action_mode_persists_value_alongside_search_mode():
    config.set_search_mode("autonomous")
    config.set_action_mode("autonomous")

    assert config.as_dict() == {"search-mode": "autonomous", "action-mode": "autonomous"}


def test_invalid_values_raise_without_persisting():
    with pytest.raises(config.ConfigError):
        config.set_search_mode("sometimes")

    with pytest.raises(config.ConfigError):
        config.set_action_mode("maybe")

    assert config.as_dict() == {"search-mode": "explicit", "action-mode": "confirm"}


def test_config_file_is_owner_read_write_only():
    config.set_search_mode("autonomous")

    mode = paths.config_path().stat().st_mode
    assert stat.S_IMODE(mode) == 0o600
