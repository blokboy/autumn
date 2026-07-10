# PRD: Multi-provider model interaction + first-run local model picker

Status: drafted from a grilling session, not yet implemented.

## Problem

Autumn already has a working local-model catalog (`local_models.py`) and a
`llama.cpp`-backed runner (`local_model_runner.py`), routed through
`model_router.choose_model()`. It also already has dataclasses for cloud
provider accounts/models (`ProviderAccount`, `ProviderModel`,
`PromptRoutingPolicy` in `models.py`) but nothing wires them up:
`_choose_provider_model`/`_provider_choice` in `model_router.py` are dead
code, and `app.py` hardcodes any provider choice to a
`"not executable yet"` stub reply.

Separately, there is no way to get a local model onto a fresh install short
of manually running `autumn models install` with a file path you already
downloaded yourself — there's no bundled/curated model list and no fetch
mechanism at all.

This PRD covers making a subset of that real, end to end.

## Scope this round

**In scope:**
- Real, working chat completions from **Groq** (free tier).
- **Anthropic** and **OpenAI** appear in the model catalog as named,
  recognizable options, but stay non-executable (visibly disabled) — same
  behavior as today, just now visible instead of entirely absent.
- Streaming replies for **both** Groq and local `llama.cpp` models.
- Cancelling an in-flight streaming reply (local or cloud).
- A first-run picker that downloads one of five curated small local GGUF
  models from Hugging Face when the local catalog is empty.
- Unifying local + provider models into one catalog with a single
  user-settable default, with a defined fallback chain.

**Explicitly out of scope (deferred to a later version):**
- Any in-app credential entry/storage UI (Groq, Anthropic, or OpenAI). Auth
  this round is env-var only.
- Making Anthropic or OpenAI actually callable.
- Background/non-blocking model downloads (first-run download blocks in a
  progress screen this round — see "Future work").
- Checksum pinning / resumable downloads for the HF fetch.

## Providers

| Provider  | This round                                  | Auth                    |
|-----------|----------------------------------------------|-------------------------|
| Local (llama.cpp) | Fully working (already shipped) + streaming | n/a (local file) |
| Groq      | Fully working, streaming, cancellable        | `GROQ_API_KEY` env var  |
| Anthropic | Listed, grayed out, not selectable as default | none read this round   |
| OpenAI    | Listed, grayed out, not selectable as default | none read this round   |

Rationale for punting on Anthropic/OpenAI auth: a shared bundled API key is
a non-starter (trivially extracted from a public package, shared free-tier
quota gets exhausted/banned immediately, likely violates provider ToS, no
revocation story if leaked). Real auth requires either per-user keys or an
in-app credential flow, both deliberately deferred.

Groq needs a real per-user key too, but is real infrastructure work worth
doing now since (a) the ask was for at least one fully-working provider path,
and (b) it validates the provider abstraction end-to-end for Anthropic/OpenAI
to plug into later.

### Groq models (subrows under the Groq entry)

- `llama-3.3-70b-versatile` (best quality)
- `llama-3.1-8b-instant` (fastest)
- `gemma2-9b-it` (alternate vendor)

### Dependencies

Add the official `groq` Python SDK (pulls in `httpx` transitively). Reuse
that same `httpx` client for the Hugging Face model-file downloads (streamed
download with progress), so one new dependency chain covers both needs
instead of hand-rolling SSE parsing and chunked downloads against stdlib
`urllib`.

## Unified model catalog & routing

Today: `is_default` lives only on `LocalModel`, and `choose_model()`
hardcodes "try local candidates, else offline-tiny" — it never consults
`PromptRoutingPolicy` at all despite the plumbing existing.

New behavior:
- `is_default` becomes a single flag over the *whole* catalog (local models
  + provider models), settable from the Models tab regardless of type —
  the existing `d` keybinding extends to cover provider rows.
- `choose_model()` tries the explicit default first. If it's unavailable
  (missing runtime, missing/invalid API key, network error), it falls
  through the rest of the catalog in priority order, and only reaches the
  offline-tiny builtin responder if nothing else works.
- Anthropic/OpenAI rows can never become the active default (UI disables
  selecting them), so the fallback chain never has to reason about a stub
  "becoming" a default.

### Models tab UI

Restructured from a flat `ListView` (`ModelCatalogView`) to a grouped/tree
view:

```
Local
├─ (installed local models, as today)
Groq
├─ llama-3.3-70b-versatile
├─ llama-3.1-8b-instant
└─ gemma2-9b-it
Anthropic  (grayed out)
├─ claude-...
OpenAI  (grayed out)
├─ gpt-...
```

Selecting a grayed-out row and pressing `d` shows a notification (e.g. "Not
available yet") instead of setting it as default.

## Streaming

Both Groq and local `llama.cpp` replies stream incrementally into the chat
transcript:

- **Groq**: use the SDK's streaming endpoint; append each chunk to the
  in-progress `ChatMessage.text` and re-render.
- **Local**: `local_model_runner.py` currently uses
  `subprocess.run(capture_output=True)`, which blocks until the process
  exits and reads all of stdout at once. This changes to `Popen` with
  `stdout=PIPE`, reading incrementally and decoding safely across
  multi-byte UTF-8 boundaries, invoking the same per-chunk callback path as
  Groq.
- UI updates are throttled (not one `call_from_thread` per token) to avoid
  flooding the Textual event loop.
- `ChatView.refresh_from_messages` already does a full transcript
  re-render from the message list on every call, so no new incremental-diff
  widget is needed — mutating the in-progress message's `.text` and calling
  the existing refresh path is sufficient.

### Cancellation

A cancel action (exact keybinding TBD at implementation time, likely
`escape` given it's unused at the `DashboardScreen` level) stops the
current stream: kills the `llama-cli` subprocess or closes the Groq HTTP
stream, and marks the message as stopped, keeping whatever text streamed in
so far.

### Error handling mid-stream

If a stream errors out partway (Groq API error, `llama-cli` crash), the
partial text that already rendered is kept, with an error marker appended
(e.g. `[interrupted: connection error]`). No silent fallback substitution —
this differs from today's non-streaming `LocalModelRuntimeError` handling,
which discards the failed attempt and regenerates a whole new offline-tiny
reply; that made sense when nothing had been shown yet, but throwing away
real partial output the user already saw would be worse here.

## First-run local model picker

**Trigger**: whenever `local_models.list_models(catalog_root)` is empty —
no separate "have we onboarded before" flag. Skipping just means "not now";
it reappears next launch since the catalog is still empty. Installing any
one model (via the picker or manually) makes it stop appearing.

**Selection**: pick exactly **one** of five curated models to download (not
multi-select) — GGUF files are multi-GB, and getting to a working default
fast matters more than front-loading every option.

**Curated list** (small, 1–4GB Q4_K_M GGUF, one per vendor):

| Model | Vendor | Approx. size |
|---|---|---|
| Llama 3.2 3B Instruct | Meta | ~2.0GB |
| Qwen2.5 3B Instruct | Alibaba | ~1.9GB |
| Phi-3.5-mini-instruct | Microsoft | ~2.4GB |
| Gemma 2 2B Instruct | Google | ~1.6GB |
| TinyLlama 1.1B Chat | community | ~0.7GB |

Exact HF repo/file paths to be pinned at implementation time (verify
current repo names, exact quant file names, and licenses before wiring up
URLs).

**Download source**: direct HTTPS from Hugging Face Hub
(`https://huggingface.co/<repo>/resolve/main/<file>.gguf`), no auth needed
for public repos. The downloaded file is installed through the existing
`local_models.install_model()` path (same manifest, same catalog root) so
it's indistinguishable from a manually-installed model afterward.

**Download UX**: blocking progress screen (percentage/progress bar) shown
after the user picks a model, before entering the dashboard. Once
installed, it's set as the default and the app proceeds to the dashboard
normally.

## Future work (explicitly deferred, noted here so it isn't lost)

- Background/non-blocking downloads: let the user enter the dashboard
  immediately after picking a model, with the download completing in the
  background and the model becoming available once done. Needs a defined
  "default chosen but not yet runnable" state and an in-app progress
  indicator. Deferred because it adds real state-machine complexity the
  blocking flow avoids.
- In-app credential entry/storage for Groq/Anthropic/OpenAI, replacing the
  env-var-only auth.
- Making Anthropic and OpenAI actually callable once auth exists.
- Checksum verification and resumable downloads for the HF fetch.

## Open questions to resolve during implementation (not blocking this PRD)

- Exact cancel keybinding.
- Exact pinned HF repo/file names + license check for each of the 5
  curated models.
- Whether provider "availability" (e.g. missing `GROQ_API_KEY`) is surfaced
  in the Models tab row itself (e.g. dimmed) or only when it's attempted as
  the active default and falls through.
