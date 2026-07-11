# autumn

![autumn landing screen](docs/landing-screenshot.png)

`autumn` is a CLI for agent interactions, along with monitoring [GEPA](https://github.com/gepa-ai/gepa) optimization runs live. Choose which models you would like to make available to autumn for routing and monitor activity from the dashboard. That's it. 

Inspired by [Torlink](https://github.com/baairon/torlink) and built with [Textual](https://textual.textualize.io/).

## Install

```bash
pip install -e .
```

## Features

- **Live run dashboard** — Overview (status, iteration/budget progress, best candidate), Candidates (sortable table of every candidate with Pareto-front markers), and a streaming Log tab, all updating in real time while a GEPA run is active.
- **Run browser** — every past run under your runs root is listed in a sidebar, live or historical, fully navigable without a run active.
- **One shared chat** — ask Autumn questions from the landing screen or the dashboard's command bar; both feed the same persisted conversation, visible in the Chat tab.
- **Local model routing with fallback** — prompts try your configured default model first, fall back to any other installed runnable model, and finally to a built-in offline responder — never silently swallowing a model's own error.
- **Run queueing** — launch a `gepa ...` run or ask a question while another run is live, and it queues automatically, processing everything in submission order once the run finishes.
- **Resume on relaunch** — if `autumn` exits with pending queued commands or an unfinished chat, the next launch offers to resume or start fresh.
- **Model catalog management** — install, list, default, and remove local `llama.cpp` models from the CLI or the dashboard's Models tab.
```bash
autumn models install tiny ./tiny.gguf --backend llama.cpp --context-window 2048
autumn models list
autumn models default tiny
autumn models remove tiny
```

## Release

The PyPI distribution is published as `autumn-cli` (plain `autumn` was already taken by an unrelated package), but the installed console command is still `autumn`. Releases are cut by hand -- a maintainer with PyPI access bumps `version` in `pyproject.toml` and, from the repo root, runs:

```bash
python -m build
twine upload dist/*
```

## Usage

```bash
autumn
```

Bare `autumn` opens on a landing screen with a single input field. Press Enter on an empty line to drop into browse mode: it scans the runs directory (`$XDG_DATA_HOME/autumn/runs`, falling back to `~/.local/share/autumn/runs`) and lets you browse any past run's full Overview/Candidates/Log/Chat/Models tabs, reading straight from GEPA's own on-disk JSON files. Move the sidebar cursor (arrows or `j`/`k`) to preview a different run. Typing `gepa <script.py> [--dry-run] [--name ...] [--run-dir ...]` launches a live run identically to `autumn run <script.py>` below, without leaving the TUI. Any other non-empty input becomes a dashboard chat prompt, persisted for the life of the session and offered for resume if the process exits before the transcript is cleared.

Once inside the dashboard, press `:` to focus the persistent command bar at the bottom. Submitting a `gepa <script.py> ...` command there launches it immediately if nothing is running, or appends it to a visible, in-memory queue if a run is already live -- the next queued item auto-starts as soon as the current run finishes or is stopped. Submitting any other text appends it to the shared dashboard chat immediately and answers asynchronously with the selected local model policy; if a run is live, the prompt is still recorded and shown right away, but its answer joins the same queue as pending `gepa` commands and is generated once the run reaches a terminal state, in the order everything was submitted.

```bash
autumn runs [--json]
```

Non-interactive counterpart to bare `autumn`: prints the same run list as a plain-text table (or a JSON array with `--json`) and exits, without opening the TUI.

Manages Autumn's local model catalog under `$XDG_DATA_HOME/autumn/models` (falling back to `~/.local/share/autumn/models`). Installed model files are copied into Autumn's managed catalog, one model can be marked as the default, and `autumn models list --json` prints the catalog for scripting. The dashboard's Models tab shows the installed catalog and lets you press `d` to make the highlighted model the default.

For `llama.cpp` models, Autumn tries the catalog's default model first, then any other installed, runnable model, before finally falling back to the built-in `autumn/offline-tiny`. If `llama-cli` is not on `PATH`, set `AUTUMN_LLAMA_CLI=/path/to/llama-cli`. When every installed model's runtime is unavailable or fails to start, the Chat tab falls back to `autumn/offline-tiny` and shows the original reason above the transcript; if a model's runtime does start but the model itself returns an error, that error is shown as-is in the transcript instead of being silently replaced by a fallback reply.

```bash
autumn run <script.py> --dry-run
```
Launches the live dashboard. Without `--dry-run`, `<script.py>` is run for real: autumn monkeypatches `gepa.optimize`/`gepa.optimize_anything` and executes the script unmodified via `runpy`, streaming its real `GEPACallback` events into the dashboard live. `--dry-run` instead replays a scripted, dependency-free sequence of optimization events against it (no real GEPA run required) so you can see the sidebar, status, iteration/budget progress, and best-candidate summary update live; the sidebar still shows the full run registry, with the live run pinned first. `<script.py>` is not executed in `--dry-run` mode -- it's only used to derive the run's display name.
