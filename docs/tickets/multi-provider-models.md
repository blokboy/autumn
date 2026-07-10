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
entry in the unified catalog.

**Blocked by:** #11

- [ ] `groq` SDK added as a dependency
- [ ] "Groq" group with 3 subrows: `llama-3.3-70b-versatile`,
      `llama-3.1-8b-instant`, `gemma2-9b-it`
- [ ] Unavailable (falls through chain) when `GROQ_API_KEY` unset
- [ ] Real completion returned when set as default and key present
- [ ] API/network errors fall through the chain rather than crashing
- [ ] Tests: chosen w/ key, falls through w/o key, falls through on API
      error, no live network calls in test suite

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
