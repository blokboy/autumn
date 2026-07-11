"""Builds the `PromptRoutingPolicy` for OpenAI's catalog group.

Direct mirror of `groq_policy.py` -- catalog-visibility/gating only: whether
the "openai" group and its models appear in the unified catalog
(`catalog.build_entries`) and therefore get considered by
`model_router.choose_model`. Zero dependency on the `openai` package -- no
SDK import, no network call, just a credential lookup
(`credentials.resolve_key`, an in-app-stored key from `autumn keys add
openai` or the `OPENAI_API_KEY` env var) -- actually calling OpenAI is
`openai_runner.py`'s job, and wiring that runner into app.py's chat-answer
path is a separate concern (see `app.py::_answer_prompt_async`).

`account.is_signed_in` gates eligibility per `catalog._eligible_provider_entries`:
a `ProviderModel` only becomes a real catalog entry if its `(provider,
account_id)` matches a signed-in `ProviderAccount`. Here that's a simple "is
there an OpenAI key available from either source" check standing in for a
real sign-in flow.

Unlike `stub_providers.py`'s always-visible, never-eligible OpenAI rows,
entries built from this policy are real and selectable -- they only exist in
the catalog once a key is actually configured, at which point
`DashboardScreen._catalog_entries` stops showing the OpenAI stub rows
alongside them (see `stub_providers.disabled_provider_entries`'s
`exclude_providers` parameter).
"""

import credentials
from models import PromptRoutingPolicy, ProviderAccount, ProviderModel

PROVIDER = "openai"
ACCOUNT_ID = "default"
ENV_VAR = "OPENAI_API_KEY"


def _is_signed_in() -> bool:
    """True iff an OpenAI key is available -- stored via `autumn keys add
    openai` or set as `OPENAI_API_KEY` in the environment."""
    return bool(credentials.resolve_key(PROVIDER, ENV_VAR))


def build_policy() -> PromptRoutingPolicy:
    """Builds the OpenAI `PromptRoutingPolicy`: one account gated on
    `OPENAI_API_KEY`, plus its candidate models. None of the models are
    `is_default=True` -- nothing here should silently become the active
    default just because this policy exists; only an explicit user action
    (the `d` keybinding, `AutumnApp.set_default_model`) should ever do that.
    """
    account = ProviderAccount(
        provider=PROVIDER,
        account_id=ACCOUNT_ID,
        display_name="OpenAI",
        is_signed_in=_is_signed_in(),
    )
    models = [
        ProviderModel(
            name="gpt-4o",
            provider=PROVIDER,
            account_id=ACCOUNT_ID,
            priority=10,
        ),
        ProviderModel(
            name="gpt-4o-mini",
            provider=PROVIDER,
            account_id=ACCOUNT_ID,
            priority=20,
        ),
        ProviderModel(
            name="gpt-4.1-mini",
            provider=PROVIDER,
            account_id=ACCOUNT_ID,
            priority=30,
        ),
    ]
    return PromptRoutingPolicy(provider_accounts=[account], provider_models=models)
