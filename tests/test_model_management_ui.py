"""Behavioral tests for dashboard local-model management."""

from textual.widgets import Static, TabbedContent, Tree

import local_models
from app import AutumnApp
from models import PromptRoutingPolicy, ProviderAccount, ProviderModel
from widgets.model_catalog_view import ModelCatalogView


async def test_models_tab_lists_installed_models_and_sets_default(tmp_path):
    first = tmp_path / "first.gguf"
    first.write_bytes(b"first")
    second = tmp_path / "second.gguf"
    second.write_bytes(b"second")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="first", source_path=first)
    local_models.install_model(catalog_root, name="second", source_path=second)
    app = AutumnApp(runs_root=tmp_path, model_catalog_root=catalog_root)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one(TabbedContent).active = "models-tab"
        await pilot.pause()

        view = app.screen.query_one("#models", ModelCatalogView)
        summary = view.query_one("#model-catalog-summary", Static).content
        assert "2 installed models" in str(summary)
        assert "Default: first" in str(summary)

        tree = view.query_one("#model-tree", Tree)
        # Root's children: the "Local" group node, then its two model leaves.
        second_leaf = tree.root.children[0].children[1]
        tree.move_cursor(second_leaf)
        await pilot.press("d")
        await pilot.pause()

        assert local_models.get_default(catalog_root).name == "second"
        summary = view.query_one("#model-catalog-summary", Static).content
        assert "Default: second" in str(summary)
        notifications = list(app._notifications)
        assert any("Default model set to second" in notification.message for notification in notifications)


async def test_models_tab_shows_empty_state_when_no_models_are_installed(tmp_path):
    app = AutumnApp(runs_root=tmp_path, model_catalog_root=tmp_path / "models")

    async with app.run_test() as pilot:
        await pilot.pause()
        # Empty catalog -> ModelPickerScreen lands first; skip it to reach
        # InputScreen's empty-Enter -> browse-mode path this test cares about.
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one(TabbedContent).active = "models-tab"
        await pilot.pause()

        view = app.screen.query_one("#models", ModelCatalogView)
        summary = view.query_one("#model-catalog-summary", Static).content
        assert "No local models installed" in str(summary)


async def test_models_tab_sets_eligible_provider_default_without_crashing(tmp_path):
    catalog_root = tmp_path / "models"
    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=catalog_root,
        prompt_routing_policy=PromptRoutingPolicy(
            provider_accounts=[
                ProviderAccount(provider="groq", account_id="default", is_signed_in=True)
            ],
            provider_models=[
                ProviderModel(name="llama-3.3-70b-versatile", provider="groq", account_id="default")
            ],
        ),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one(TabbedContent).active = "models-tab"
        await pilot.pause()

        view = app.screen.query_one("#models", ModelCatalogView)
        tree = view.query_one("#model-tree", Tree)
        # Root's children: the "groq" group node (no local models installed).
        groq_leaf = tree.root.children[0].children[0]
        tree.move_cursor(groq_leaf)
        await pilot.press("d")
        await pilot.pause()

        notifications = list(app._notifications)
        assert any(
            "Default model set to llama-3.3-70b-versatile" in notification.message
            for notification in notifications
        )
        summary = view.query_one("#model-catalog-summary", Static).content
        assert "Default: llama-3.3-70b-versatile" in str(summary)


async def test_models_tab_excludes_ineligible_provider_from_selection(tmp_path):
    catalog_root = tmp_path / "models"
    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=catalog_root,
        prompt_routing_policy=PromptRoutingPolicy(
            provider_accounts=[
                ProviderAccount(provider="groq", account_id="default", is_signed_in=False)
            ],
            provider_models=[
                ProviderModel(name="llama-3.3-70b-versatile", provider="groq", account_id="default")
            ],
        ),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one(TabbedContent).active = "models-tab"
        await pilot.pause()

        view = app.screen.query_one("#models", ModelCatalogView)
        summary = view.query_one("#model-catalog-summary", Static).content
        # Not signed in -> the group is excluded from the catalog entirely,
        # same as an unset GROQ_API_KEY -- nothing to select, nothing crashes.
        assert "No local models installed" in str(summary)


async def test_set_default_model_notifies_instead_of_crashing_for_ineligible_entry(tmp_path):
    """Direct method-level coverage of `AutumnApp.set_default_model`'s
    ValueError catch: the tree UI never lets you *select* an ineligible
    entry today (it's excluded from the catalog entirely), but the catch
    still matters -- e.g. a future visible-but-disabled row (#15), or an
    entry that stopped being eligible between render and keypress."""
    app = AutumnApp(runs_root=tmp_path, model_catalog_root=tmp_path / "models")

    async with app.run_test() as pilot:
        await pilot.pause()
        app.set_default_model("groq", "llama-3.3-70b-versatile")
        await pilot.pause()

        notifications = list(app._notifications)
        assert any(
            "isn't supported yet for catalog group 'groq'" in notification.message
            for notification in notifications
        )
        assert notifications[-1].severity == "warning"


async def test_models_tab_groups_entries_by_source(tmp_path):
    first = tmp_path / "first.gguf"
    first.write_bytes(b"first")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="first", source_path=first)
    app = AutumnApp(runs_root=tmp_path, model_catalog_root=catalog_root)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one(TabbedContent).active = "models-tab"
        await pilot.pause()

        view = app.screen.query_one("#models", ModelCatalogView)
        tree = view.query_one("#model-tree", Tree)
        # "Local" first, then the always-visible Anthropic/OpenAI stub
        # groups (#15) -- no provider policy was supplied, so there's no
        # eligible Groq group in between this time.
        group_labels = [str(node.label) for node in tree.root.children]
        assert group_labels == ["Local", "Anthropic", "OpenAI"]
        assert len(tree.root.children[0].children) == 1


async def test_anthropic_and_openai_always_appear_grayed_out(tmp_path):
    """#15: Anthropic/OpenAI must always show up in the Models tab -- with no
    provider policy, no env vars, and no local models installed at all --
    with a couple of representative subrows each, rendered visibly
    disabled."""
    app = AutumnApp(runs_root=tmp_path, model_catalog_root=tmp_path / "models")

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one(TabbedContent).active = "models-tab"
        await pilot.pause()

        view = app.screen.query_one("#models", ModelCatalogView)
        tree = view.query_one("#model-tree", Tree)
        group_labels = [str(node.label) for node in tree.root.children]
        assert group_labels == ["Anthropic", "OpenAI"]

        for group_node in tree.root.children:
            # Group header itself is dimmed.
            assert any(span.style == "dim" for span in group_node.label.spans)
            assert group_node.children, "expected representative model subrows"
            for leaf in group_node.children:
                assert "not available yet" in str(leaf.label)
                assert any(span.style == "dim" for span in leaf.label.spans)


async def test_setting_default_on_disabled_provider_row_is_a_no_op_with_notification(tmp_path):
    """#15: pressing `d` on an Anthropic/OpenAI row must not change the
    active default, and must surface a friendly notification instead of
    crashing."""
    first = tmp_path / "first.gguf"
    first.write_bytes(b"first")
    catalog_root = tmp_path / "models"
    local_models.install_model(catalog_root, name="first", source_path=first)
    app = AutumnApp(runs_root=tmp_path, model_catalog_root=catalog_root)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one(TabbedContent).active = "models-tab"
        await pilot.pause()

        view = app.screen.query_one("#models", ModelCatalogView)
        tree = view.query_one("#model-tree", Tree)
        # Root's children: "Local" (index 0), then "Anthropic" (index 1).
        anthropic_leaf = tree.root.children[1].children[0]
        assert anthropic_leaf.data.group == "Anthropic"
        tree.move_cursor(anthropic_leaf)
        await pilot.press("d")
        await pilot.pause()

        # Unchanged: the real local default is still "first".
        assert local_models.get_default(catalog_root).name == "first"
        summary = view.query_one("#model-catalog-summary", Static).content
        assert "Default: first" in str(summary)

        notifications = list(app._notifications)
        assert any(
            "isn't supported yet for catalog group 'Anthropic'" in notification.message
            for notification in notifications
        )
        assert notifications[-1].severity == "warning"
