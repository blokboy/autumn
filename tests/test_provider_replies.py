"""Behavioral tests for real Anthropic/OpenAI chat completions (#21):

- a signed-in default provider choice answers via the injected runner
  (non-streaming -- see `anthropic_runner.py`/`openai_runner.py`);
- an absent key falls through to the offline-tiny builtin responder, and
  the Models tab still shows that provider's grayed-out stub rows;
- a simulated runtime error (bad key, network failure) also falls through
  to offline-tiny, discarding the failed provider choice;
- none of this ever touches a real network call -- every runner here is a
  fake injected via `anthropic_runner=`/`openai_runner=`, or (for the
  key-absent cases) never constructed at all because the catalog entry
  never becomes eligible.
"""

from textual.widgets import Input, Static, TabbedContent, Tree

import anthropic_policy, openai_policy
from anthropic_runner import AnthropicRuntimeError
from app import AutumnApp
from models import ChatMessage, PromptRoutingPolicy, ProviderAccount, ProviderModel
from openai_runner import OpenAIRuntimeError
from widgets.command_bar import CommandBar
from widgets.model_catalog_view import ModelCatalogView


def _policy_with_default(provider: str, model_name: str) -> PromptRoutingPolicy:
    return PromptRoutingPolicy(
        provider_accounts=[ProviderAccount(provider=provider, account_id="default", is_signed_in=True)],
        provider_models=[
            ProviderModel(
                name=model_name,
                provider=provider,
                account_id="default",
                priority=10,
                is_default=True,
            )
        ],
    )


async def _land_and_submit_prompt(pilot, prompt: str) -> None:
    await pilot.pause()
    # Empty catalog -> ModelPickerScreen lands first; skip it.
    await pilot.press("escape")
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()

    await pilot.press(":")
    await pilot.pause()
    pilot.app.screen.query_one(CommandBar).query_one(Input).insert_text_at_cursor(prompt)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(0.2)


class _FakeAnthropicRunner:
    def __init__(self, reply_text: str = "Hi from Claude."):
        self._reply_text = reply_text
        self.seen_calls: list[tuple[list[ChatMessage], str]] = []

    def generate(self, messages: list[ChatMessage], model_name: str) -> ChatMessage:
        self.seen_calls.append((messages, model_name))
        return ChatMessage(role="assistant", text=self._reply_text, model=model_name)


class _FakeOpenAIRunner:
    def __init__(self, reply_text: str = "Hi from GPT."):
        self._reply_text = reply_text
        self.seen_calls: list[tuple[list[ChatMessage], str]] = []

    def generate(self, messages: list[ChatMessage], model_name: str) -> ChatMessage:
        self.seen_calls.append((messages, model_name))
        return ChatMessage(role="assistant", text=self._reply_text, model=model_name)


class _FailingAnthropicRunner:
    def generate(self, messages: list[ChatMessage], model_name: str) -> ChatMessage:
        raise AnthropicRuntimeError(f"{model_name} failed: connection error")


class _FailingOpenAIRunner:
    def generate(self, messages: list[ChatMessage], model_name: str) -> ChatMessage:
        raise OpenAIRuntimeError(f"{model_name} failed: connection error")


# --- Anthropic ----------------------------------------------------------


async def test_prompt_receives_anthropic_reply_when_default_and_key_present(tmp_path):
    runner = _FakeAnthropicRunner()
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        anthropic_runner=runner,
        prompt_routing_policy=_policy_with_default("anthropic", "claude-sonnet-5"),
    )

    async with app.run_test() as pilot:
        await _land_and_submit_prompt(pilot, "hello")

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(role="assistant", text="Hi from Claude.", model="claude-sonnet-5"),
        ]
        assert runner.seen_calls == [([ChatMessage(role="user", text="hello")], "claude-sonnet-5")]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: claude-sonnet-5 (provider available)" in str(status_text)
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "Autumn [claude-sonnet-5]: Hi from Claude." in str(chat_text)


async def test_prompt_falls_through_when_anthropic_key_absent_and_stub_rows_still_show(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        # No anthropic_runner injected: if the (unsigned-in) Anthropic
        # catalog entries were ever reached, this would fall through to a
        # real AnthropicRunner() -- proving the fallback never touches
        # Anthropic at all.
        prompt_routing_policy=anthropic_policy.build_policy(),
    )

    async with app.run_test() as pilot:
        await _land_and_submit_prompt(pilot, "hello")

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="Offline local response: hello",
                model="autumn/offline-tiny",
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Fallback: autumn/offline-tiny (offline fallback)" in str(status_text)

        app.screen.query_one(TabbedContent).active = "models-tab"
        await pilot.pause()
        view = app.screen.query_one("#models", ModelCatalogView)
        tree = view.query_one("#model-tree", Tree)
        group_labels = [str(node.label) for node in tree.root.children]
        # Still visible, still grayed out -- no key means no real "anthropic"
        # entry, so the stub group is never excluded.
        assert "Anthropic" in group_labels


async def test_prompt_falls_through_on_anthropic_runtime_error(tmp_path):
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        anthropic_runner=_FailingAnthropicRunner(),
        prompt_routing_policy=_policy_with_default("anthropic", "claude-sonnet-5"),
    )

    async with app.run_test() as pilot:
        await _land_and_submit_prompt(pilot, "hello")

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="Offline local response: hello",
                model="autumn/offline-tiny",
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Fallback: autumn/offline-tiny" in str(status_text)
        assert "claude-sonnet-5 failed: connection error" in str(status_text)


async def test_anthropic_real_entry_hides_its_own_stub_but_openai_stub_still_shows(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=tmp_path / "models",
        prompt_routing_policy=_policy_with_default("anthropic", "claude-sonnet-5"),
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
        group_labels = [str(node.label) for node in tree.root.children]

        # A real, eligible "anthropic" group is present (not grayed out)...
        assert "anthropic" in group_labels
        # ...and the "Anthropic" stub group is gone (never both at once)...
        assert "Anthropic" not in group_labels
        # ...while OpenAI (no policy supplied here) still shows its stub.
        assert "OpenAI" in group_labels


# --- OpenAI ---------------------------------------------------------------


async def test_prompt_receives_openai_reply_when_default_and_key_present(tmp_path):
    runner = _FakeOpenAIRunner()
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        openai_runner=runner,
        prompt_routing_policy=_policy_with_default("openai", "gpt-4o"),
    )

    async with app.run_test() as pilot:
        await _land_and_submit_prompt(pilot, "hello")

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(role="assistant", text="Hi from GPT.", model="gpt-4o"),
        ]
        assert runner.seen_calls == [([ChatMessage(role="user", text="hello")], "gpt-4o")]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Model: gpt-4o (provider available)" in str(status_text)
        chat_text = app.screen.query_one("#chat-transcript", Static).content
        assert "Autumn [gpt-4o]: Hi from GPT." in str(chat_text)


async def test_prompt_falls_through_when_openai_key_absent_and_stub_rows_still_show(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        # No openai_runner injected -- same "would raise if reached" proof
        # as the Anthropic key-absent test above.
        prompt_routing_policy=openai_policy.build_policy(),
    )

    async with app.run_test() as pilot:
        await _land_and_submit_prompt(pilot, "hello")

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="Offline local response: hello",
                model="autumn/offline-tiny",
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Fallback: autumn/offline-tiny (offline fallback)" in str(status_text)

        app.screen.query_one(TabbedContent).active = "models-tab"
        await pilot.pause()
        view = app.screen.query_one("#models", ModelCatalogView)
        tree = view.query_one("#model-tree", Tree)
        group_labels = [str(node.label) for node in tree.root.children]
        assert "OpenAI" in group_labels


async def test_prompt_falls_through_on_openai_runtime_error(tmp_path):
    app = AutumnApp(
        runs_root=tmp_path,
        chat_sessions_root=tmp_path / "chats",
        model_catalog_root=tmp_path / "models",
        openai_runner=_FailingOpenAIRunner(),
        prompt_routing_policy=_policy_with_default("openai", "gpt-4o"),
    )

    async with app.run_test() as pilot:
        await _land_and_submit_prompt(pilot, "hello")

        assert app.chat_messages == [
            ChatMessage(role="user", text="hello"),
            ChatMessage(
                role="assistant",
                text="Offline local response: hello",
                model="autumn/offline-tiny",
            ),
        ]
        status_text = app.screen.query_one("#chat-model-status", Static).content
        assert "Fallback: autumn/offline-tiny" in str(status_text)
        assert "gpt-4o failed: connection error" in str(status_text)


async def test_setting_default_on_real_anthropic_entry_persists(tmp_path):
    """Acceptance criterion: setting a real (key-present) Anthropic model as
    default via the Models tab `d` keybinding persists correctly, reusing
    `catalog.set_default`'s existing eligibility check -- no new plumbing."""
    app = AutumnApp(
        runs_root=tmp_path,
        model_catalog_root=tmp_path / "models",
        prompt_routing_policy=PromptRoutingPolicy(
            provider_accounts=[
                ProviderAccount(provider="anthropic", account_id="default", is_signed_in=True)
            ],
            provider_models=[
                ProviderModel(name="claude-sonnet-5", provider="anthropic", account_id="default")
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
        # Root's children: the real "anthropic" group (no local models
        # installed, and its own stub group is excluded -- see the test
        # above), then "OpenAI"'s stub group.
        anthropic_leaf = tree.root.children[0].children[0]
        tree.move_cursor(anthropic_leaf)
        await pilot.press("d")
        await pilot.pause()

        notifications = list(app._notifications)
        assert any(
            "Default model set to claude-sonnet-5" in notification.message for notification in notifications
        )
        summary = view.query_one("#model-catalog-summary", Static).content
        assert "Default: claude-sonnet-5" in str(summary)
