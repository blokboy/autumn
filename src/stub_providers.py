"""Anthropic and OpenAI as always-visible, permanently non-functional catalog
groups (#15): named, recognizable rows in the Models tab that communicate
what's coming, without pretending they're selectable or wiring any real
account/auth plumbing this round (see docs/prd/multi-provider-models.md,
"Providers" table: both are "Listed, grayed out, not selectable as default",
auth "none read this round").

This is deliberately NOT routed through `PromptRoutingPolicy`/
`ProviderAccount` -- that mechanism (see `groq_policy.py`) is for real
accounts with a real sign-in state, binary eligible-or-absent, which is
exactly wrong here: these two providers must always be visible yet never
selectable, and have neither an account nor a sign-in state this round.

Also deliberately kept out of `catalog.build_entries`'s return value, which
is the same list `model_router.choose_model` walks to pick what a prompt
actually runs against -- `model_router._availability` treats *any* entry
with `backend == "provider"` as already known-enabled-and-signed-in, so a
disabled stub entry sneaking in there would get happily "chosen" and then
have nothing to actually run it. Callers building the Models tab's entry
list call `disabled_provider_entries()` directly and append it themselves
(see `screens/dashboard_screen.py::_catalog_entries`); callers building the
routing catalog (`model_router.choose_model`) never do.

No environment variable is read and no network call is ever made here --
these are static, hardcoded rows, same spirit as `GROQ_MODELS` in
`groq_runner.py` or `CURATED_MODELS` in `curated_models.py`.

As of #21, Anthropic and OpenAI can each have a *real*, policy-driven catalog
entry once their respective `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` is set (see
`anthropic_policy.py`/`openai_policy.py`, consumed the same way
`groq_policy.py` always has been). `disabled_provider_entries`'s
`exclude_providers` parameter lets a caller that already knows which
providers have a real entry this call (e.g.
`screens/dashboard_screen.py::_catalog_entries`, from
`catalog.build_entries`'s output) skip that provider's stub rows entirely --
so a signed-in provider's group is never shown both as real, selectable rows
*and* grayed-out stub rows at the same time.
"""

from models import CatalogEntry

ANTHROPIC_GROUP = "Anthropic"
OPENAI_GROUP = "OpenAI"

# Representative model names only -- cosmetic, not a real, callable catalog.
ANTHROPIC_MODELS: tuple[str, ...] = ("claude-sonnet", "claude-haiku")
OPENAI_MODELS: tuple[str, ...] = ("gpt-4o", "gpt-4o-mini")

_STUB_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (ANTHROPIC_GROUP, "anthropic", ANTHROPIC_MODELS),
    (OPENAI_GROUP, "openai", OPENAI_MODELS),
)


def disabled_provider_entries(*, exclude_providers: set[str] | None = None) -> list[CatalogEntry]:
    """Always-present, never-eligible catalog rows for Anthropic/OpenAI, in
    display order (Anthropic, then OpenAI). Otherwise unconditional -- no
    env var or network call gates these, so with `exclude_providers` empty
    (or omitted) they appear identically on every call regardless of
    environment or sign-in state.

    `exclude_providers`, if given, drops a provider's stub rows entirely --
    e.g. `{"anthropic"}` skips both Anthropic stub rows while still
    returning OpenAI's. Used when that provider already has a real,
    eligible entry elsewhere in the catalog (see module docstring)."""
    skip = exclude_providers or set()
    return [
        CatalogEntry(
            group=group,
            name=name,
            backend="provider",
            provider=provider,
            disabled=True,
        )
        for group, provider, names in _STUB_GROUPS
        if provider not in skip
        for name in names
    ]
