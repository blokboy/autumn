import dataclasses
from pathlib import Path

import gepa
import gepa.optimize_anything as oa_module
import pytest
from gepa.core.engine import GEPAEngine
from gepa.optimize_anything import EngineConfig, GEPAConfig

import patch
from dashboard_callback import DashboardCallback
from models import DashboardState


def _make_dashboard(run_dir: Path) -> DashboardCallback:
    return DashboardCallback(app=None, state=DashboardState(run_name="test", run_dir=run_dir))


@pytest.fixture(autouse=True)
def _restore_gepa_state(monkeypatch):
    """`apply()` mutates module/class attributes in place rather than
    replacing `sys.modules['gepa']`. Registering the *current* value with
    `monkeypatch.setattr` (a no-op reassignment) makes monkeypatch restore
    exactly that value on teardown, regardless of what `apply()` later sets
    it to -- so patches from one test never leak into the next.
    """
    monkeypatch.setattr(gepa, "optimize", gepa.optimize)
    monkeypatch.setattr(oa_module, "optimize_anything", oa_module.optimize_anything)
    monkeypatch.setattr(GEPAEngine, "__init__", GEPAEngine.__init__)


# --- gepa.optimize -------------------------------------------------------------


def test_optimize_no_callbacks_kwarg_dashboard_becomes_callbacks_list(monkeypatch, tmp_path):
    captured = {}

    def fake_optimize(*args, **kwargs):
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(gepa, "optimize", fake_optimize)

    dashboard = _make_dashboard(tmp_path)
    run_dir = tmp_path / "run1"
    patch.apply(dashboard, run_dir)

    result = gepa.optimize(seed_candidate={}, trainset=[])

    assert result == "result"
    assert captured["callbacks"] == [dashboard]


def test_optimize_appends_dashboard_to_existing_callbacks(monkeypatch, tmp_path):
    captured = {}

    def fake_optimize(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(gepa, "optimize", fake_optimize)

    dashboard = _make_dashboard(tmp_path)
    existing_cb = object()
    run_dir = tmp_path / "run1"
    patch.apply(dashboard, run_dir)

    gepa.optimize(seed_candidate={}, trainset=[], callbacks=[existing_cb])

    assert existing_cb in captured["callbacks"]
    assert dashboard in captured["callbacks"]
    assert len(captured["callbacks"]) == 2


def test_optimize_no_run_dir_uses_managed_run_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_optimize(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(gepa, "optimize", fake_optimize)

    dashboard = _make_dashboard(tmp_path)
    run_dir = tmp_path / "run1"
    patch.apply(dashboard, run_dir)

    gepa.optimize(seed_candidate={}, trainset=[])

    assert captured["run_dir"] == str(run_dir)


def test_optimize_explicit_differing_run_dir_is_honored(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_optimize(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(gepa, "optimize", fake_optimize)

    dashboard = _make_dashboard(tmp_path)
    managed_run_dir = tmp_path / "managed"
    user_run_dir = str(tmp_path / "user_chosen")
    patch.apply(dashboard, managed_run_dir)

    gepa.optimize(seed_candidate={}, trainset=[], run_dir=user_run_dir)

    assert captured["run_dir"] == user_run_dir
    assert user_run_dir in capsys.readouterr().out


# --- gepa.optimize_anything ------------------------------------------------


def test_gepa_config_has_no_callbacks_field_in_installed_release():
    """Documents the version-gap assumption this module's runtime-detection
    branch depends on; if this ever fails, GEPA has shipped the release with
    `GEPAConfig.callbacks` and the `GEPAEngine.__init__`-patch fallback path
    (and this test) can be retired."""
    assert "callbacks" not in {f.name for f in dataclasses.fields(GEPAConfig)}


def test_optimize_anything_engine_init_gets_dashboard_when_no_callbacks_passed(
    monkeypatch, tmp_path
):
    captured = {}

    def fake_engine_init(self, *args, callbacks=None, **kwargs):
        captured["callbacks"] = callbacks

    monkeypatch.setattr(GEPAEngine, "__init__", fake_engine_init)

    dashboard = _make_dashboard(tmp_path)
    run_dir = tmp_path / "run1"
    patch.apply(dashboard, run_dir)

    instance = object.__new__(GEPAEngine)
    GEPAEngine.__init__(instance)

    assert captured["callbacks"] == [dashboard]


def test_optimize_anything_engine_init_appends_dashboard_to_existing_callbacks(
    monkeypatch, tmp_path
):
    captured = {}

    def fake_engine_init(self, *args, callbacks=None, **kwargs):
        captured["callbacks"] = callbacks

    monkeypatch.setattr(GEPAEngine, "__init__", fake_engine_init)

    dashboard = _make_dashboard(tmp_path)
    existing_cb = object()
    run_dir = tmp_path / "run1"
    patch.apply(dashboard, run_dir)

    instance = object.__new__(GEPAEngine)
    GEPAEngine.__init__(instance, callbacks=[existing_cb])

    assert existing_cb in captured["callbacks"]
    assert dashboard in captured["callbacks"]
    assert len(captured["callbacks"]) == 2


def test_apply_twice_does_not_double_append_to_engine_callbacks(monkeypatch, tmp_path):
    captured = {}

    def fake_engine_init(self, *args, callbacks=None, **kwargs):
        captured["callbacks"] = callbacks

    monkeypatch.setattr(GEPAEngine, "__init__", fake_engine_init)

    dashboard = _make_dashboard(tmp_path)
    run_dir = tmp_path / "run1"
    patch.apply(dashboard, run_dir)
    patch.apply(dashboard, run_dir)

    instance = object.__new__(GEPAEngine)
    GEPAEngine.__init__(instance)

    assert captured["callbacks"] == [dashboard]


def test_optimize_anything_no_run_dir_uses_managed_run_dir_via_replace(monkeypatch, tmp_path):
    captured = {}

    def fake_optimize_anything(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(oa_module, "optimize_anything", fake_optimize_anything)

    dashboard = _make_dashboard(tmp_path)
    run_dir = tmp_path / "run1"
    patch.apply(dashboard, run_dir)

    oa_module.optimize_anything(evaluator=lambda candidate: 1.0)

    config = captured["config"]
    assert isinstance(config, GEPAConfig)
    assert config.engine.run_dir == str(run_dir)


def test_optimize_anything_explicit_differing_run_dir_is_honored_and_config_cloned(
    monkeypatch, tmp_path, capsys
):
    captured = {}

    def fake_optimize_anything(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(oa_module, "optimize_anything", fake_optimize_anything)

    dashboard = _make_dashboard(tmp_path)
    managed_run_dir = tmp_path / "managed"
    patch.apply(dashboard, managed_run_dir)

    user_run_dir = str(tmp_path / "user_chosen")
    original_engine = EngineConfig(run_dir=user_run_dir)
    original_config = GEPAConfig(engine=original_engine)

    oa_module.optimize_anything(evaluator=lambda candidate: 1.0, config=original_config)

    result_config = captured["config"]
    assert result_config.engine.run_dir == user_run_dir
    assert user_run_dir in capsys.readouterr().out

    # never mutated in place
    assert result_config is not original_config
    assert result_config.engine is not original_engine
    assert original_config.engine.run_dir == user_run_dir
    assert original_engine.run_dir == user_run_dir


# --- smoke test against the real, installed gepa ----------------------------


def test_apply_runs_cleanly_against_installed_gepa(tmp_path):
    dashboard = _make_dashboard(tmp_path)
    run_dir = tmp_path / "run1"

    patch.apply(dashboard, run_dir)
