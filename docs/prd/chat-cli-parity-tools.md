# PRD: CLI-parity system-control tools for dashboard chat

Status: drafted from a grilling/ticketing session, not yet implemented. Depends
on the tool-calling mechanism introduced in
[chat-search-tools.md](./chat-search-tools.md).

## Problem

`chat-search-tools.md` lets dashboard chat *inform* the user via `search_web`
and `search_docs`, but chat still can't *act* on the user's behalf. Today,
configuring autumn means leaving chat and running `autumn models` (list,
install, default, remove) or `autumn keys` (add, list, remove) at a shell.
This PRD gives chat tool equivalents of those CLI subcommands, reusing the
same Groq-only, single-round tool-calling mechanism `chat-search-tools.md`
establishes.

## Scope this round

**In scope:**
- Read-only tools: `list_models` (→ `autumn models list`), `list_keys` (→
  `autumn keys list` — never returns key values, same guarantee
  `credentials.py` already provides), `list_runs` (→ `autumn runs`). No
  confirmation required — same trust level as `search_docs`.
- Mutating tools: `install_model`, `set_default_model`, `add_key`. Each
  requires an explicit confirmation via `ConfirmScreen` before executing,
  regardless of the autonomy toggle's default state.
- Destructive mutating tools: `remove_model`, `remove_key`. Same confirmation
  requirement as above, **plus** the modal's confirm button is forced
  disabled for 1.5 seconds after it appears, so a reflexive keypress/click
  can't confirm a deletion before the user has actually read the prompt.
- `install_model` via chat is scoped to the five curated models already named
  in `multi-provider-models.md`'s first-run picker — not an arbitrary local
  file path, since the model has no visibility into what's actually on the
  user's disk. Arbitrary-path installs stay a CLI-only/manual operation.
- A second, independent autonomy toggle (own config key, reusing the same
  `autumn config` CLI + dashboard-toggle mechanism `chat-search-tools.md`'s
  autonomy-setting ticket builds for search) that, when enabled, skips the
  confirmation modal entirely for mutating tool calls — both the standard and
  destructive tiers. Defaults to **off** (confirmation required).
- Same gating as the search tools: Groq-routed chat only, no tool-calling for
  local/offline models.

**Explicitly out of scope (deferred to a later version):**
- Launching or controlling GEPA runs (`autumn run`/`gepa ...`) via chat
  tools — a long-running process is a different, larger risk profile than a
  single reversible-ish config mutation, and the existing command-bar
  `gepa <script>` flow already covers manual launching without needing
  model-driven initiation.
- A single unified autonomy toggle shared between search and system-control
  actions. Kept independent this round — a user may reasonably want
  autonomous search but confirmed mutations, or vice versa.
- Undo/rollback for a mistakenly confirmed destructive action — relies on the
  user re-running the equivalent tool call or CLI command (e.g. `add_key`
  after an accidental `remove_key`).
- Arbitrary-HF-repo or arbitrary-local-path model installs via chat, beyond
  the five curated models.

## Tools exposed

| Tool | Maps to | Mutating? | Confirmation |
|---|---|---|---|
| `list_models` | `autumn models list` | No | None |
| `list_keys` | `autumn keys list` | No | None |
| `list_runs` | `autumn runs` | No | None |
| `install_model` | `autumn models install` (curated only) | Yes | Standard confirm |
| `set_default_model` | `autumn models default` | Yes | Standard confirm |
| `add_key` | `autumn keys add` | Yes | Standard confirm |
| `remove_model` | `autumn models remove` | Yes, destructive | Confirm + 1.5s delay |
| `remove_key` | `autumn keys remove` | Yes, destructive | Confirm + 1.5s delay |

## Confirmation flow

- Extend `ConfirmScreen` with an optional `initial_delay: float` param
  (default `0`), keeping the confirm button disabled and unfocusable for that
  many seconds after mount — set to `1.5` for `remove_model`/`remove_key`,
  left at the default for the other three mutating tools.
- The modal message names the exact action and target (e.g. "Remove the
  locally-stored Groq API key?", "Uninstall `llama-3.2-3b-instruct`?") rather
  than a generic "confirm this tool call" — the user is confirming a specific
  consequence, not a mechanism.
- On cancel, the tool call is reported back to the model as declined; the
  model's final answer should acknowledge that rather than silently retrying
  or proceeding as if it had happened.

## Autonomous mode

- New config key (name TBD at implementation, e.g. `action-mode:
  autonomous|confirm`), independent of `search-mode`, persisted via the same
  mechanism `chat-search-tools.md`'s autonomy-setting ticket introduces.
- Default: `confirm` — every mutating call shows the modal (with the 1.5s
  delay on the two destructive tools).
- `autonomous`: mutating calls execute immediately, no modal, no delay — same
  trust level implied by a user running `autumn keys add` from their own
  shell today.
- CLI: `autumn config set action-mode autonomous|confirm`. Dashboard: a
  second toggle alongside the search `search-mode` toggle.

## Future work (explicitly deferred, noted here so it isn't lost)

- Chat-driven `autumn run`/`gepa ...` launching.
- Collapsing `search-mode` and `action-mode` into a single toggle, if usage
  shows users don't actually want them independent.
- Undo/rollback for confirmed destructive actions.
- Arbitrary local-file-path or arbitrary-HF-repo model installs via chat.

## Open questions to resolve during implementation (not blocking this PRD)

- Exact `action-mode` config key name and dashboard toggle placement (likely
  paired with the `search-mode` toggle).
- Exact confirmation modal copy per action.
- Whether declined tool calls should be retryable in the same turn or require
  a fresh user message.
