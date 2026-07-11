"""Builds the `PromptRoutingPolicy` for Groq's catalog group.

This is catalog-visibility/gating only: whether the "groq" group and its
three models appear in the unified catalog (`catalog.build_entries`) and
therefore get considered by `model_router.choose_model`. It has zero
dependency on the `groq` package -- no SDK import, no network call, just a
credential lookup (`credentials.resolve_key`, an in-app-stored key from
`autumn keys add groq` or the `GROQ_API_KEY` env var) -- since actually
calling Groq is a separate ticket's job (the runtime), as is swapping
app.py's chat-answer stub for a real call (the integration).

`account.is_signed_in` gates eligibility per `catalog._eligible_provider_entries`:
a `ProviderModel` only becomes a real catalog entry if its `(provider,
account_id)` matches a signed-in `ProviderAccount`. Here that's a simple
"is there a Groq key available from either source" check standing in for a
real sign-in flow.
"""

import credentials
from models import PromptRoutingPolicy, ProviderAccount, ProviderModel

PROVIDER = "groq"
ACCOUNT_ID = "default"
ENV_VAR = "GROQ_API_KEY"


def _is_signed_in() -> bool:
    """True iff a Groq key is available -- stored via `autumn keys add groq`
    or set as `GROQ_API_KEY` in the environment."""
    return bool(credentials.resolve_key(PROVIDER, ENV_VAR))


def build_policy() -> PromptRoutingPolicy:
    """Builds the Groq `PromptRoutingPolicy`: one account gated on
    `GROQ_API_KEY`, plus its three candidate models. None of the models are
    `is_default=True` -- nothing here should silently become the active
    default just because this policy exists; only an explicit user action
    (the `d` keybinding, `AutumnApp.set_default_model`) should ever do that.
    """
    account = ProviderAccount(
        provider=PROVIDER,
        account_id=ACCOUNT_ID,
        display_name="Groq",
        is_signed_in=_is_signed_in(),
    )
    models = [
        ProviderModel(
            name="llama-3.3-70b-versatile",
            provider=PROVIDER,
            account_id=ACCOUNT_ID,
            priority=10,
        ),
        ProviderModel(
            name="llama-3.1-8b-instant",
            provider=PROVIDER,
            account_id=ACCOUNT_ID,
            priority=20,
        ),
        ProviderModel(
            name="gemma2-9b-it",
            provider=PROVIDER,
            account_id=ACCOUNT_ID,
            priority=30,
        ),
    ]
    return PromptRoutingPolicy(provider_accounts=[account], provider_models=models)
