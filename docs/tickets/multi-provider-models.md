# Tickets: Multi-provider models & first-run picker

Vertical-slice breakdown of docs/prd/multi-provider-models.md, published as
GitHub issues on `blokboy/autumn` (labeled `ready-for-agent`). This file
mirrors those issues for local reference — the issues are the source of
truth for status/comments.

Work the frontier: any ticket whose blockers are all done can start. #10 and
#11 have no blockers and are being worked in parallel.

## First-run local model picker & download — [#10](https://github.com/blokboy/autumn/issues/10)

**What to build:** On a fresh install with an empty local model catalog, show
a picker with 5 curated small GGUF models. The user picks one, it downloads
from Hugging Face with a blocking progress screen, installs through the
existing local-model catalog machinery, and becomes the default model.

**Blocked by:** None — can start immediately.

- [ ] Picker shows the 5 curated models (name/vendor/approx size) when the
      catalog is empty
- [ ] Selected model downloads via direct HTTPS from Hugging Face
- [ ] Blocking progress screen shown during download
- [ ] Installed via existing `local_models.install_model()`, set as default
- [ ] Skipping re-shows the picker next launch; installing (any way) stops it
      from ever showing again
- [ ] HF repo/file names for all 5 models pinned + license-checked
- [ ] Tests: empty-catalog trigger, skip, install+default, no-picker-when-nonempty

## Unified model catalog with single default + fallback chain — [#11](https://github.com/blokboy/autumn/issues/11)

**What to build:** Generalize "the active default model" and the fallback
logic across the whole catalog (local + future provider entries), and
restructure the Models tab into a grouped/tree view.

**Blocked by:** None — can start immediately.

- [ ] Default applies across the unified catalog, not just `LocalModel`
- [ ] Models tab is a grouped/tree view (starts with just "Local")
- [ ] `choose_model()`: explicit default → fall through rest of catalog →
      offline-tiny
- [ ] Existing local CLI/`d`-keybinding behavior unchanged from the user's
      perspective
- [ ] `PromptRoutingPolicy`/`ProviderAccount`/`ProviderModel` wired in, no
      dead code left over
- [ ] Tests: default chosen, fallback on unavailable default, fallback to
      offline-tiny

## Groq: real chat completions (blocking) — [#12](https://github.com/blokboy/autumn/issues/12)

**What to build:** Real, working (non-streaming) chat completions from
Groq's free tier, gated on `GROQ_API_KEY`, as the first functional provider
entry in the unified catalog. Tracks the overall vertical slice; split into
three sub-tickets (#16-#18) below so the runtime and catalog-gating work can
proceed in parallel, converging into one integration step.

**Blocked by:** #11

### Groq: provider runtime (non-streaming chat completion) — [#16](https://github.com/blokboy/autumn/issues/16)

SDK-wrapping only, mirrors `LocalModelRunner`/`LocalModelRuntimeError`. No
catalog/`app.py` dependency.

**Blocked by:** None — can start immediately, in parallel with #17.

- [ ] `groq` SDK added as a dependency
- [ ] `GroqRunner.generate(messages, model) -> ChatMessage`, non-streaming,
      client injectable at construction for test mocking
- [ ] API/network errors raise a `GroqRuntimeError` (mirrors
      `LocalModelRuntimeError`)
- [ ] Tests: successful completion, `GroqRuntimeError` on API/SDK error, no
      live network calls (injected fake client)

### Groq: catalog policy gated on GROQ_API_KEY — [#17](https://github.com/blokboy/autumn/issues/17)

Catalog-visibility/gating only — builds the real `PromptRoutingPolicy` and
wires it into `AutumnApp` construction (today it's always `None`). No
SDK/network dependency.

**Blocked by:** None — can start immediately, in parallel with #16.

- [ ] Policy builder producing a `groq` `ProviderAccount` + 3
      `ProviderModel` rows: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`,
      `gemma2-9b-it`
- [ ] `is_signed_in` gated on `GROQ_API_KEY` being set
- [ ] `cli.py` builds and passes this policy so the "Groq" group actually
      appears in the Models tab
- [ ] Tests: entries present w/ key, absent/excluded w/o key, catalog
      reflects both via `catalog.build_entries`

### Groq: wire runtime + policy into chat-answer path — [#18](https://github.com/blokboy/autumn/issues/18)

Integration step — replaces `app.py`'s hardcoded provider stub with a real
call into #16's runner for Groq choices produced via #17's policy.

**Blocked by:** #16, #17

- [ ] Groq `ModelChoice` triggers a real call into #16's runner from
      `_answer_prompt_async`
- [ ] `GroqRuntimeError` falls through to offline-tiny (mirrors the existing
      `LocalModelRuntimeError` catch)
- [ ] Non-Groq provider choices (Anthropic/OpenAI) keep today's "not
      executable yet" stub unchanged
- [ ] Tests (end-to-end): chosen w/ default + key present, falls through
      w/o key, falls through on simulated API error, no live network calls

## Groq: streaming + cancel + mid-stream error handling — [#13](https://github.com/blokboy/autumn/issues/13)

**What to build:** Upgrade Groq replies to token-by-token streaming, add a
cancel action, and handle mid-stream errors without discarding partial
output.

**Blocked by:** #12

- [ ] Replies stream incrementally (throttled, not per-token) into the chat
      transcript
- [ ] Cancel action stops an in-flight stream, keeping partial text
- [ ] Mid-stream error keeps partial text + appends an `[interrupted: ...]`
      marker, no silent fallback substitution
- [ ] Tests: incremental updates, cancel preserves partial text, mid-stream
      error preserves partial text + marker

## Local llama.cpp: streaming + cancel + mid-stream error handling — [#14](https://github.com/blokboy/autumn/issues/14)

**What to build:** Apply #13's streaming/cancel/error-marker plumbing to
local llama.cpp models, reworking the runner from a blocking subprocess call
to incremental streamed reads.

**Blocked by:** #13

- [ ] `local_model_runner.py`: `Popen` + incremental `stdout` reads (safe
      multi-byte UTF-8 decoding) instead of blocking `subprocess.run`
- [ ] Local replies stream using #13's chunk/throttle/cancel plumbing
- [ ] Cancel kills the `llama-cli` subprocess, keeping partial text
- [ ] Crash/non-zero exit mid-stream keeps partial text + error marker
- [ ] Tests: incremental updates, cancel kills subprocess + preserves partial
      text, crash mid-stream preserves partial text + marker

## Anthropic & OpenAI: visible but disabled catalog entries — [#15](https://github.com/blokboy/autumn/issues/15)

**What to build:** Show Anthropic and OpenAI as recognizable, grayed-out
groups in the unified catalog — visible, not functional.

**Blocked by:** #11

- [ ] "Anthropic"/"OpenAI" groups with representative model subrows
      (`claude-*`, `gpt-*`)
- [ ] Rendered visibly disabled/grayed out
- [ ] Setting one as default shows "Not available yet", doesn't change the
      active default
- [ ] No API keys read, no network calls made
- [ ] Tests: grayed-out rendering, attempted-default-set is a no-op +
      notification
