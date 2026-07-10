# autumn

`autumn` is a [Textual](https://textual.textualize.io/) terminal dashboard for watching [GEPA](https://github.com/gepa-ai/gepa) prompt-optimization runs live, styled after the terminal torrent client Torlink (dark violet, rounded box-drawn panels with embedded titles).

## Install

```bash
pip install -e .
```

## Release

The PyPI distribution is published as `autumn-cli` (plain `autumn` was already taken by an unrelated package), but the installed console command is still `autumn`. Releases are cut by hand -- a maintainer with PyPI access bumps `version` in `pyproject.toml` and, from `gepa-tui/`, runs:

```bash
python -m build
twine upload dist/*
```

## Usage

```bash
autumn
```

Bare `autumn` opens on a landing screen with a single input field. Press Enter on an empty line to drop into browse mode: it scans the runs directory (`$XDG_DATA_HOME/autumn/runs`, falling back to `~/.local/share/autumn/runs`) and lets you browse any past run's full Overview/Candidates/Log tabs, reading straight from GEPA's own on-disk JSON files. Move the sidebar cursor (arrows or `j`/`k`) to preview a different run. Typing `gepa <script.py> [--dry-run] [--name ...] [--run-dir ...]` instead launches a live run identically to `autumn run <script.py>` below, without leaving the TUI.

```bash
autumn runs [--json]
```

Non-interactive counterpart to bare `autumn`: prints the same run list as a plain-text table (or a JSON array with `--json`) and exits, without opening the TUI.

```bash
autumn run <script.py> --dry-run
```

Launches the live dashboard. Without `--dry-run`, `<script.py>` is run for real: autumn monkeypatches `gepa.optimize`/`gepa.optimize_anything` and executes the script unmodified via `runpy`, streaming its real `GEPACallback` events into the dashboard live. `--dry-run` instead replays a scripted, dependency-free sequence of optimization events against it (no real GEPA run required) so you can see the sidebar, status, iteration/budget progress, and best-candidate summary update live; the sidebar still shows the full run registry, with the live run pinned first. `<script.py>` is not executed in `--dry-run` mode -- it's only used to derive the run's display name.
