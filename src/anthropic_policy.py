"""Builds the `PromptRoutingPolicy` for Anthropic's catalog group.

Direct mirror of `groq_policy.py` -- catalog-visibility/gating only: whether
the "anthropic" group and its models appear in the unified catalog
(`catalog.build_entries`) and therefore get considered by
`model_router.choose_model`. Zero dependency on the `anthropic` package -- no
SDK import, no network call, just a credential lookup
(`credentials.resolve_key`, an in-app-stored key from `autumn keys add
anthropic` or the `ANTHROPIC_API_KEY` env var) -- actually calling Anthropic
is `anthropic_runner.py`'s job, and wiring that runner into app.py's
chat-answer path is a separate concern (see `app.py::_answer_prompt_async`).

`account.is_signed_in` gates eligibility per `catalog._eligible_provider_entries`:
a `ProviderModel` only becomes a real catalog entry if its `(provider,
account_id)` matches a signed-in `ProviderAccount`. Here that's a simple "is
there an Anthropic key available from either source" check standing in for a
real sign-in flow.

Unlike `stub_providers.py`'s always-visible, never-eligible Anthropic rows,
entries built from this policy are real and selectable -- they only exist in
the catalog once a key is actually configured, at which point
`DashboardScreen._catalog_entries` stops showing the Anthropic stub rows
alongside them (see `stub_providers.disabled_provider_entries`'s
`exclude_providers` parameter).
"""

import credentials
from models import PromptRoutingPolicy, ProviderAccount, ProviderModel

PROVIDER = "anthropic"
ACCOUNT_ID = "default"
ENV_VAR = "ANTHROPIC_API_KEY"


def _is_signed_in() -> bool:
    """True iff an Anthropic key is available -- stored via `autumn keys add
    anthropic` or set as `ANTHROPIC_API_KEY` in the environment."""
    return bool(credentials.resolve_key(PROVIDER, ENV_VAR))


def build_policy() -> PromptRoutingPolicy:
    """Builds the Anthropic `PromptRoutingPolicy`: one account gated on
    `ANTHROPIC_API_KEY`, plus its candidate models. None of the models are
    `is_default=True` -- nothing here should silently become the active
    default just because this policy exists; only an explicit user action
    (the `d` keybinding, `AutumnApp.set_default_model`) should ever do that.
    """
    account = ProviderAccount(
        provider=PROVIDER,
        account_id=ACCOUNT_ID,
        display_name="Anthropic",
        is_signed_in=_is_signed_in(),
    )
    models = [
        ProviderModel(
            name="claude-sonnet-5",
            provider=PROVIDER,
            account_id=ACCOUNT_ID,
            priority=10,
        ),
        ProviderModel(
            name="claude-opus-4-8",
            provider=PROVIDER,
            account_id=ACCOUNT_ID,
            priority=20,
        ),
        ProviderModel(
            name="claude-haiku-4-5",
            provider=PROVIDER,
            account_id=ACCOUNT_ID,
            priority=30,
        ),
    ]
    return PromptRoutingPolicy(provider_accounts=[account], provider_models=models)
