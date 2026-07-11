# PRD: Real-time web search and local docs retrieval for dashboard chat

Status: drafted from a grilling session, not yet implemented.

## Problem

Dashboard chat (`app.py`, `dashboard_callback.py`, `chat_view.py`) routes every
prompt through `model_router.choose_model()` to either a Groq-hosted model
(`groq_runner.py`) or a local `llama.cpp` model (`local_model_runner.py`),
falling back to the built-in offline stub (`local_llm.py`). None of these
paths have any notion of tools: `GroqRunner.generate`/`generate_stream` call
Groq's chat completions endpoint with plain messages only, and there is no
search, retrieval, embedding, or vector-store code anywhere in the repo.

This means chat can only ever answer from whatever the underlying model
already knows — nothing about events after its training cutoff, and nothing
about this project's own docs unless that happened to be in training data.

This PRD adds two tools the model can call mid-conversation to fix both
gaps: live web search, and retrieval over this project's own documentation.

## Scope this round

**In scope:**
- A `search_web` tool backed directly by the Tavily API (chosen over Brave/Exa
  for being purpose-built for LLM tool use, and over an MCP client for not
  requiring MCP protocol plumbing to reach a single provider that still needs
  the same API key either way).
- A `search_docs` tool doing keyword/BM25-style retrieval (no embeddings, no
  vector store) over `docs/`, `README.md`, and `AGENTS.md` only — not GEPA run
  history, not chat history.
- Tool-calling wired into the **Groq path only**. Local `llama.cpp` models and
  the offline-tiny fallback are unaffected — no tools offered, no behavior
  change, since `llama-cli` has no structured tool-call output and small local
  models are unreliable at emitting parseable tool syntax.
- A single non-streaming tool-decision round per turn (not a multi-round
  agentic loop): the model either answers directly, or calls at most one tool
  before producing its final answer, which streams normally once ready.
- A visible "Searching…" status shown in the chat view during the
  non-streaming tool-decision/execution gap, so a Tavily round-trip isn't
  silent dead air.
- Source citations on any answer that used a tool result — doc filename/section
  for `search_docs`, result URL(s) for `search_web` — rendered under the
  answer in `chat_view.py`.
- Visible (not silent, not turn-aborting) degradation on tool failure: if
  Tavily errors or times out, or the docs index can't be read, that failure is
  shown in the transcript and the model still attempts a best-effort,
  caveated answer from whatever it has left (its own knowledge, or the other
  tool's results if only one failed).
- `tavily` added to `_KNOWN_PROVIDERS` in `cli.py` and wired through
  `credentials.py`/`autumn keys` exactly like `groq` is today — same
  `TAVILY_API_KEY` env var convention via `_env_var_for_provider`, same
  stored-key-wins-over-env-var resolution via `resolve_key`.
- An autonomy setting controlling whether `search_web` is offered to the
  model on every turn, or only when the user explicitly signals a search
  (e.g. a `/search` prefix, matching the existing `gepa <script>` vs. freeform
  chat convention in `command_bar.py`). Defaults to **explicit**.
  `search_docs` is always offered regardless of this setting — it's local,
  free, and has no third-party privacy exposure.
- The autonomy setting is both a new CLI subcommand (`autumn config`-shaped,
  mirroring the `autumn keys`/`autumn models` precedent) and a live toggle
  surfaced in the dashboard, matching how the Models tab already dual-surfaces
  the model catalog.

**Explicitly out of scope (deferred to a later version):**
- Embeddings/vector-store-based semantic search over docs (see Future work).
- Multi-round agentic tool-chaining, where the model can call more than one
  tool before answering (see Future work).
- Any per-session/daily cap on Tavily call volume (see Future work).
- Retrieval over GEPA run history or chat history — different retrieval shape
  (structured lookup, not prose chunking) than this PRD's docs corpus; a
  separate tool if pursued later.
- Tool-calling support for local `llama.cpp` models.

## `search_web` (Tavily)

Direct REST integration, no MCP layer — MCP is a transport standard, not a
hosting service, and Tavily's own MCP server still reads the same
`TAVILY_API_KEY` under the hood, so it would add a JSON-RPC/stdio client to
build for zero reduction in what the user has to configure.

- New `tavily_runner.py` (or similar), same shape as `groq_runner.py`: a thin
  runtime boundary around the Tavily client, resolving its key via
  `credentials.resolve_key("tavily", "TAVILY_API_KEY")`.
- The tool is only included in the `tools` payload sent to Groq when a Tavily
  key resolves to something — mirroring how a provider only appears "available"
  in `model_router.py`/`catalog.py` once its account is usable. No key means
  the model is never even offered the tool, so there's nothing to fail loudly
  about mid-conversation.
- Query text comes from whatever the model chooses to send as the tool-call
  argument (Groq's function-calling decides this, same mechanism as any
  OpenAI-compatible tool call) — not the raw user message verbatim.

## `search_docs` (local, keyword-based)

- Indexes `docs/**/*.md`, `README.md`, `AGENTS.md` — small, static-ish corpus
  where a BM25/keyword ranker over doc chunks will answer realistic
  documentation questions ("what does `--dry-run` do") as well as embeddings
  would, without a new provider dependency or a vector store to maintain.
- Built in-memory; given the corpus size, rebuilding on each `search_docs`
  call (or re-checking file mtimes) is effectively free, so there's no need
  for on-disk caching or file-watching machinery.
- Always included in the `tools` payload sent to Groq — no key, no opt-in
  gate, no autonomy-setting interaction.

## Tool-call round

1. Send the conversation plus the currently-applicable `tools` schema
   (`search_docs` always; `search_web` only if a Tavily key resolves *and*
   either autonomous mode is on or the user's message carries the explicit
   trigger) to Groq, non-streaming.
2. If the response has no `tool_calls`, treat it as the final answer — stream
   isn't needed since the full text is already in hand; render it directly.
3. If it has exactly one tool call, show a "Searching the web for '…'" /
   "Searching docs for '…'" status in `chat_view.py`, execute that one tool,
   and make a second Groq call with the tool result appended as a tool-role
   message.
4. Stream the second call's response as the final answer, same mechanism
   `GroqRunner.generate_stream` already uses today. Attach source citations
   (doc path/section or Tavily URLs) from step 3's result once the stream
   completes.
5. If step 3's tool call raises, surface that failure inline in the
   transcript (not a hard abort) and still perform step 4 — the model gets
   told the tool failed and answers with whatever it has, caveated.

No provision for the model requesting a second tool call after the first
result comes back — that's the multi-round case deferred below.

## Autonomy setting

- CLI: a new `autumn config` subcommand (`autumn config set search-mode
  autonomous|explicit`, `autumn config show`), persisted the same way
  `credentials.py` persists provider keys — small JSON file under the
  existing XDG-style `paths.py` convention, `0600` permissions.
- Dashboard: a live toggle (exact surface TBD at implementation time — most
  likely a row in the Models tab or a new lightweight settings affordance)
  that reads/writes the same persisted value, so CLI and dashboard can never
  disagree about the current mode, matching the single-source-of-truth
  pattern `resolve_key` already guarantees for credentials.
- Default: **explicit**. `search_web` is not offered to the model until the
  user's message carries the trigger (exact syntax TBD at implementation
  time — most likely a `/search` prefix). Switching to autonomous mode offers
  `search_web` on every turn without requiring the trigger.

## Future work (explicitly deferred, noted here so it isn't lost)

Carried over from the multi-provider-models PRD, still outstanding
(re-listed here only for visibility, not re-scoped by this PRD):
- Background/non-blocking model downloads
  ([#20](https://github.com/blokboy/autumn/issues/20)).
- Anthropic/OpenAI real completions via env-var auth
  ([#21](https://github.com/blokboy/autumn/issues/21)).
- In-app credential entry/storage for Groq/Anthropic/OpenAI
  ([#22](https://github.com/blokboy/autumn/issues/22)) — note `tavily` should
  join this list once it exists as a provider.
- Checksum verification and resumable downloads for the HF fetch
  ([#23](https://github.com/blokboy/autumn/issues/23)).

New, generated during this PRD's grilling session (to be ticketed once this
PRD ships):
- **Embeddings/vector-store RAG over docs.** If the `docs/` corpus grows
  past what keyword/BM25 search handles well (paraphrased queries, larger
  doc set), revisit with real chunk-and-embed retrieval — needs an embedding
  provider (Groq doesn't offer embeddings) or a local embedding model, plus a
  local vector store (e.g. `sqlite-vec`, `chromadb`).
- **Multi-round agentic tool-chaining.** Let the model call more than one
  tool (e.g. search docs, then search the web) before producing a final
  answer, instead of this PRD's one-tool-per-turn limit. Deferred because it
  adds a new failure mode (loop-bound tuning, runaway tool chains) most
  realistic doc/web lookups don't need.
- **Soft cap on `search_web` calls.** A configurable per-session limit (e.g.
  20) after which autonomous mode stops offering `search_web` for the rest of
  the session, as a backstop against runaway Tavily usage beyond what
  Tavily's own account-level limits catch. Deferred as a premature guardrail
  until real usage shows it's needed.
- **Tool-calling support for local `llama.cpp` models**, via a prompted
  ReAct-style convention, if local-model chat users end up wanting search/RAG
  too. Deferred given small local models' unreliability at emitting
  parseable tool-call syntax.

## Open questions to resolve during implementation (not blocking this PRD)

- Exact `/search`-equivalent trigger syntax for explicit-mode web search.
- Exact dashboard surface for the autonomy toggle (Models tab row vs. a new
  settings affordance).
- Exact `search_docs` chunking granularity (whole-file vs. heading-level
  sections) and BM25 library vs. hand-rolled ranking.
- Exact citation rendering in `chat_view.py` (inline footnote vs. a separate
  "Sources" block under the message).
