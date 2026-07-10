"""Tests for the unified model catalog (local models + provider entries)."""

from autumn import catalog, local_models
from autumn.models import LocalModel, PromptRoutingPolicy, ProviderAccount, ProviderModel


def test_build_entries_is_empty_when_catalog_root_has_no_models(tmp_path):
    entries = catalog.build_entries(tmp_path / "models")

    assert entries == []


def test_build_entries_includes_installed_local_models(tmp_path):
    source = tmp_path / "tiny.gguf"
    source.write_bytes(b"tiny")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="tiny", source_path=source)

    entries = catalog.build_entries(catalog_root)

    assert len(entries) == 1
    assert entries[0].group == catalog.LOCAL_GROUP
    assert entries[0].name == "tiny"
    assert entries[0].backend == "llama.cpp"
    assert entries[0].is_default is True


def test_build_entries_appends_eligible_provider_entries_after_local(tmp_path):
    entries = catalog.build_entries(
        tmp_path / "models",
        local_override=[
            LocalModel(name="tiny", backend="llama.cpp", path=tmp_path / "tiny.gguf", is_default=True)
        ],
        policy=PromptRoutingPolicy(
            provider_accounts=[
                ProviderAccount(provider="claude", account_id="personal", is_signed_in=True)
            ],
            provider_models=[
                ProviderModel(name="claude/sonnet", provider="claude", account_id="personal", priority=5)
            ],
        ),
    )

    assert [entry.name for entry in entries] == ["tiny", "claude/sonnet"]
    assert entries[1].group == "claude"
    assert entries[1].backend == "provider"
    assert entries[1].provider == "claude"
    assert entries[1].account_id == "personal"


def test_build_entries_excludes_ineligible_provider_models(tmp_path):
    entries = catalog.build_entries(
        tmp_path / "models",
        local_override=[],
        policy=PromptRoutingPolicy(
            provider_accounts=[
                ProviderAccount(provider="claude", account_id="personal", is_signed_in=False)
            ],
            provider_models=[
                ProviderModel(name="claude/sonnet", provider="claude", account_id="personal")
            ],
        ),
    )

    assert entries == []


def test_build_entries_orders_eligible_provider_models_by_priority(tmp_path):
    entries = catalog.build_entries(
        tmp_path / "models",
        local_override=[],
        policy=PromptRoutingPolicy(
            provider_accounts=[
                ProviderAccount(provider="claude", account_id="personal", is_signed_in=True)
            ],
            provider_models=[
                ProviderModel(name="slow", provider="claude", account_id="personal", priority=50),
                ProviderModel(name="fast", provider="claude", account_id="personal", priority=1),
            ],
        ),
    )

    assert [entry.name for entry in entries] == ["fast", "slow"]


def test_default_entry_returns_none_when_nothing_is_marked_default(tmp_path):
    entries = catalog.build_entries(
        tmp_path / "models",
        local_override=[
            LocalModel(name="tiny", backend="llama.cpp", path=tmp_path / "tiny.gguf", is_default=False)
        ],
    )

    assert catalog.default_entry(entries) is None


def test_set_default_persists_local_group_default(tmp_path):
    first = tmp_path / "first.gguf"
    first.write_bytes(b"first")
    second = tmp_path / "second.gguf"
    second.write_bytes(b"second")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="first", source_path=first)
    local_models.install_model(catalog_root, name="second", source_path=second)

    entry = catalog.set_default(catalog_root, catalog.LOCAL_GROUP, "second")

    assert entry.name == "second"
    assert entry.is_default is True
    assert local_models.get_default(catalog_root).name == "second"


def test_set_default_raises_for_non_local_group(tmp_path):
    try:
        catalog.set_default(tmp_path / "models", "claude", "claude/sonnet")
    except ValueError as exc:
        assert "claude" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-local catalog group")
