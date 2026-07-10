"""Monkeypatches GEPA's entry points so an unmodified user script picks up
autumn's dashboard callback and managed run_dir transparently.

Call `apply()` strictly before `runpy.run_path` executes the user's script
(see runner.py). This ordering is the entire reason zero-script-modification
works: the user's own `import gepa` / `from gepa import optimize` statements
execute *during* `runpy.run_path`, i.e. after this module has already
replaced `gepa.optimize` and `gepa.optimize_anything.optimize_anything` --
so the user's script only ever sees the patched versions, never the
originals it would otherwise have imported.

Two entry points, two shapes:

- `gepa.optimize` (flat/older API) is a real function living directly on the
  `gepa` package. It already accepts `run_dir` and `callbacks` kwargs, so
  it's patched by wrapping it and rewriting those kwargs before delegating.
- `gepa.optimize_anything` is *not* callable -- it's a submodule (see
  `gepa/__init__.py`: `from . import optimize_anything`). The real function
  lives at `gepa.optimize_anything.optimize_anything` and is configured via
  a `GEPAConfig` dataclass, so it must be patched on the submodule object
  itself (`gepa.optimize_anything.optimize_anything = patched_fn`) --
  patching the package-level `gepa.optimize_anything` name would be
  pointless since nothing calls it that way.

  GEPA's `main` branch added a `callbacks` field to `GEPAConfig` (upstream
  commit 5b07402) that wires a callback list through to `GEPAEngine`, but
  that commit landed after the currently-installed PyPI release was cut, so
  today's `GEPAConfig` has no `callbacks` field. `GEPAEngine.__init__`
  already accepts `callbacks` today regardless, so this module detects at
  runtime (via `dataclasses.fields(GEPAConfig)`) which release is installed:
  when the field exists, it patches `optimize_anything` to clone `config`
  with the dashboard appended to `config.callbacks`; when it doesn't, it
  instead patches `GEPAEngine.__init__` itself so any `callbacks` list
  passed to it gets the dashboard appended. Either way, `run_dir` is always
  threaded through `config.engine.run_dir`. This means `apply()` keeps
  working with no code changes once GEPA ships that release.

Merge semantics, identical for both entry points:
- Append `dashboard` to any existing callbacks list -- never override.
- Honor an explicit user-set run_dir that differs from the managed one
  (print a one-line notice when this happens).
- Never mutate `config` / `config.engine` in place -- always clone via
  `dataclasses.replace`.
"""

import dataclasses
import functools
from pathlib import Path

import gepa
import gepa.optimize_anything as _optimize_anything_module
from gepa.core.engine import GEPAEngine
from gepa.optimize_anything import GEPAConfig

from autumn.dashboard_callback import DashboardCallback


def apply(dashboard: DashboardCallback, run_dir: Path) -> None:
    """Patches `gepa.optimize` and `gepa.optimize_anything.optimize_anything`
    so any call the user's script makes to either one transparently gets
    `dashboard` appended to its callbacks and `run_dir` as its managed
    run_dir (unless the user already set their own).

    Must run before `runpy.run_path` executes the user's script -- see the
    module docstring for why.
    """
    _patch_optimize(dashboard, run_dir)
    _patch_optimize_anything(dashboard, run_dir)


def _notice_run_dir_honored(user_run_dir: str, managed_run_dir: Path) -> None:
    print(
        f"autumn: honoring user-provided run_dir {user_run_dir!r} "
        f"over the managed run_dir {str(managed_run_dir)!r}"
    )


def _patch_optimize(dashboard: DashboardCallback, run_dir: Path) -> None:
    """Wraps `gepa.optimize`, merging `dashboard` into `callbacks` and
    defaulting `run_dir` to the managed one unless the user set their own."""
    original = gepa.optimize

    @functools.wraps(original)
    def patched(*args, **kwargs):
        callbacks = list(kwargs.get("callbacks") or [])
        if dashboard not in callbacks:
            callbacks.append(dashboard)
        kwargs["callbacks"] = callbacks

        existing_run_dir = kwargs.get("run_dir")
        if existing_run_dir is not None and existing_run_dir != str(run_dir):
            _notice_run_dir_honored(existing_run_dir, run_dir)
        else:
            kwargs["run_dir"] = str(run_dir)

        return original(*args, **kwargs)

    gepa.optimize = patched


def _patch_optimize_anything(dashboard: DashboardCallback, run_dir: Path) -> None:
    """Wraps `gepa.optimize_anything.optimize_anything`, cloning `config`
    (never mutating it) so `config.engine.run_dir` defaults to the managed
    run_dir unless the user set their own.

    Callback injection depends on the installed GEPA release: if
    `GEPAConfig` has a `callbacks` field, `dashboard` is merged into it
    directly; otherwise `GEPAEngine.__init__` is patched as a fallback (see
    module docstring).
    """
    has_callbacks_field = "callbacks" in {f.name for f in dataclasses.fields(GEPAConfig)}
    original = _optimize_anything_module.optimize_anything

    @functools.wraps(original)
    def patched(*args, **kwargs):
        config = kwargs.get("config") or GEPAConfig()
        engine_config = config.engine

        existing_run_dir = engine_config.run_dir
        if existing_run_dir is not None and existing_run_dir != str(run_dir):
            _notice_run_dir_honored(existing_run_dir, run_dir)
            resolved_run_dir = existing_run_dir
        else:
            resolved_run_dir = str(run_dir)
        new_engine_config = dataclasses.replace(engine_config, run_dir=resolved_run_dir)

        if has_callbacks_field:
            callbacks = list(getattr(config, "callbacks", None) or [])
            if dashboard not in callbacks:
                callbacks.append(dashboard)
            config = dataclasses.replace(config, engine=new_engine_config, callbacks=callbacks)
        else:
            config = dataclasses.replace(config, engine=new_engine_config)

        kwargs["config"] = config
        return original(*args, **kwargs)

    _optimize_anything_module.optimize_anything = patched

    if not has_callbacks_field:
        _patch_engine_callbacks(dashboard)


def _patch_engine_callbacks(dashboard: DashboardCallback) -> None:
    """Fallback for GEPA releases whose `GEPAConfig` has no `callbacks`
    field: patches `GEPAEngine.__init__` once, idempotently, so whatever
    `callbacks` list gets passed to it has `dashboard` appended (or is set
    to `[dashboard]` if `None`), unless `dashboard` is already present.

    Broader than strictly necessary -- it fires for every `GEPAEngine`
    constructed after `apply()` runs, not just this one run_dir -- but
    that's acceptable for v1 since `autumn run` drives one script per
    process.
    """
    if getattr(GEPAEngine.__init__, "_autumn_patched", False):
        return

    original_init = GEPAEngine.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, callbacks=None, **kwargs):
        callbacks = list(callbacks) if callbacks else []
        if dashboard not in callbacks:
            callbacks.append(dashboard)
        return original_init(self, *args, callbacks=callbacks, **kwargs)

    patched_init._autumn_patched = True
    GEPAEngine.__init__ = patched_init
