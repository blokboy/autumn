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
"""

from autumn.models import CatalogEntry

ANTHROPIC_GROUP = "Anthropic"
OPENAI_GROUP = "OpenAI"

# Representative model names only -- cosmetic, not a real, callable catalog.
ANTHROPIC_MODELS: tuple[str, ...] = ("claude-sonnet", "claude-haiku")
OPENAI_MODELS: tuple[str, ...] = ("gpt-4o", "gpt-4o-mini")

_STUB_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (ANTHROPIC_GROUP, "anthropic", ANTHROPIC_MODELS),
    (OPENAI_GROUP, "openai", OPENAI_MODELS),
)


def disabled_provider_entries() -> list[CatalogEntry]:
    """Always-present, never-eligible catalog rows for Anthropic/OpenAI, in
    display order (Anthropic, then OpenAI). Unconditional -- no gating of
    any kind, so these appear identically on every call regardless of
    environment or sign-in state."""
    return [
        CatalogEntry(
            group=group,
            name=name,
            backend="provider",
            provider=provider,
            disabled=True,
        )
        for group, provider, names in _STUB_GROUPS
        for name in names
    ]
