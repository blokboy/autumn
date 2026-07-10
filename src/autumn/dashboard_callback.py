"""Bridges GEPA's callback events onto a Textual app's main thread by mutating DashboardState."""

import json
import time
from typing import Any, Callable

from autumn.models import CandidateRow, DashboardState, RunStatus

# Minimum interval between pushes for high-frequency event types (on_budget_updated,
# on_valset_evaluated), which GEPA fires once per metric call / valset eval -- far
# too often to log or cross the call_from_thread boundary on every single call
# without bottlenecking the optimization thread.
_THROTTLE_INTERVAL_SECONDS = 0.1


class DashboardCallback:
    """Implements a subset of GEPA's `GEPACallback` protocol (structurally, not via inheritance)."""

    def __init__(self, app: Any, state: DashboardState) -> None:
        self._app = app
        self._state = state
        # Per-event-type scratch state for the high-frequency throttle below. Both
        # dicts are plain, same-thread reads/writes -- they never need
        # call_from_thread, only the state mutations pushed through self._push do.
        self._pending: dict[str, dict] = {}
        self._last_push_time: dict[str, float] = {}

    def _push(self, mutate_fn: Callable[[], None]) -> None:
        def apply_and_refresh() -> None:
            mutate_fn()

        self._app.call_from_thread(apply_and_refresh)

    def _push_throttled(self, key: str, pending_data: dict, mutate_fn: Callable[[], None]) -> None:
        """Record the latest data for a high-frequency event type, and push at
        most once per _THROTTLE_INTERVAL_SECONDS.

        Storing into self._pending is a cheap, same-thread dict write and happens
        on every call regardless of throttling. Only the actual push -- which pays
        the cross-thread call_from_thread synchronization cost -- is gated by the
        throttle, so a burst of high-frequency events doesn't bottleneck the
        optimization thread. Whichever call eventually clears the throttle window
        pushes whatever is newest in self._pending at that moment (mutate_fn reads
        self._pending[key] lazily, once it actually runs on the main thread).
        """
        self._pending[key] = pending_data
        now = time.monotonic()
        if now - self._last_push_time.get(key, 0.0) < _THROTTLE_INTERVAL_SECONDS:
            return
        self._last_push_time[key] = now
        self._push(mutate_fn)

    def _detect_multi_objective(self, event: dict) -> bool:
        """Best-effort heuristic for flagging multi-objective runs.

        No live multi-objective GEPA run/fixture was available while writing this,
        so GEPA's exact event payload shape for that case isn't pinned down here.
        We infer multi-objective-ness from the presence of an `objectives` or
        `objective_scores` key whose value is a mapping with more than one entry --
        a plausible shape given this callback's other event dicts are flat and
        descriptively keyed (e.g. `average_score` for the single-objective case).
        Revisit this once a real multi-objective fixture/run exists to verify
        against.
        """
        for candidate_key in ("objectives", "objective_scores"):
            value = event.get(candidate_key)
            if isinstance(value, dict) and len(value) > 1:
                return True
        return False

    def on_optimization_start(self, event: dict) -> None:
        trainset_size = event.get("trainset_size")
        valset_size = event.get("valset_size")
        config = event.get("config") or {}
        max_metric_calls = config.get("engine", {}).get("max_metric_calls")
        is_multi_objective = self._detect_multi_objective(event) or self._detect_multi_objective(config)

        def mutate() -> None:
            self._state.trainset_size = trainset_size
            self._state.valset_size = valset_size
            self._state.max_metric_calls = max_metric_calls
            if is_multi_objective:
                self._state.is_multi_objective = True
            self._state.append_log(
                "info",
                f"optimization started: {trainset_size} train / {valset_size} val examples",
            )

        self._push(mutate)

    def on_iteration_start(self, event: dict) -> None:
        iteration = event.get("iteration")

        def mutate() -> None:
            self._state.current_iteration = iteration
            self._state.append_log("info", f"iteration {iteration} started")

        self._push(mutate)

    def on_iteration_end(self, event: dict) -> None:
        iteration = event.get("iteration")
        proposal_accepted = event.get("proposal_accepted")

        def mutate() -> None:
            level = "success" if proposal_accepted else "info"
            outcome = "accepted" if proposal_accepted else "rejected"
            self._state.append_log(level, f"iteration {iteration} proposal {outcome}")

        self._push(mutate)

    def on_candidate_accepted(self, event: dict) -> None:
        new_candidate_idx = event.get("new_candidate_idx")
        new_score = event.get("new_score")
        parent_ids = event.get("parent_ids")

        def mutate() -> None:
            existing = self._state.candidates.get(new_candidate_idx)
            if existing is None:
                self._state.candidates[new_candidate_idx] = CandidateRow(
                    idx=new_candidate_idx,
                    val_score=new_score,
                    discovered_iteration=self._state.current_iteration,
                    parent_ids=list(parent_ids) if parent_ids is not None else [],
                    is_pareto_member=False,
                    was_rejected=False,
                )
            else:
                existing.val_score = new_score
                existing.was_rejected = False
                existing.parent_ids = list(parent_ids) if parent_ids is not None else []
                existing.discovered_iteration = self._state.current_iteration
            self._state.append_log(
                "success", f"candidate {new_candidate_idx} accepted (score={new_score})"
            )

        self._push(mutate)

    def on_candidate_rejected(self, event: dict) -> None:
        old_score = event.get("old_score")
        new_score = event.get("new_score")
        reason = event.get("reason")

        def mutate() -> None:
            self._state.append_log(
                "info",
                f"candidate rejected (reason: {reason}, old_score={old_score}, new_score={new_score})",
            )

        self._push(mutate)

    def on_pareto_front_updated(self, event: dict) -> None:
        new_front = event.get("new_front") or []
        displaced_candidates = event.get("displaced_candidates") or []

        def mutate() -> None:
            self._state.pareto_front = set(new_front)
            for idx in new_front:
                row = self._state.candidates.get(idx)
                if row is not None:
                    row.is_pareto_member = True
            for idx in displaced_candidates:
                row = self._state.candidates.get(idx)
                if row is not None:
                    row.is_pareto_member = False
            self._state.append_log(
                "info", f"pareto front updated: {sorted(new_front)}"
            )

        self._push(mutate)

    def _apply_pending_valset(self) -> None:
        """Copies whatever's newest in self._pending["valset_evaluated"] into
        DashboardState. Shared by on_valset_evaluated's throttled push and by the
        end-of-run flush below, so both paths apply identical logic."""
        data = self._pending.get("valset_evaluated")
        if data is None:
            return
        row = self._state.candidates.get(data["candidate_idx"])
        if row is not None:
            row.val_score = data["average_score"]
            if data["candidate"] is not None:
                row.text = data["candidate"]
        if data["is_best_program"]:
            self._state.best_idx = data["candidate_idx"]
            self._state.best_score = data["average_score"]
        if data["is_multi_objective"]:
            self._state.is_multi_objective = True

    def _apply_pending_budget(self) -> None:
        """Copies whatever's newest in self._pending["budget_updated"] into
        DashboardState. Shared by on_budget_updated's throttled push and by the
        end-of-run flush below."""
        data = self._pending.get("budget_updated")
        if data is None:
            return
        self._state.metric_calls_used = data["metric_calls_used"]
        self._state.metric_calls_remaining = data["metric_calls_remaining"]

    def _flush_pending(self) -> None:
        """Applies any high-frequency event data still sitting unflushed in
        self._pending because its throttle window never got hit again. A run's
        very last on_valset_evaluated/on_budget_updated call is routinely still
        inside the 100ms throttle window when on_optimization_end (or a script
        crash) follows right behind it -- without this, that last update (e.g. the
        winning candidate's real score) would be silently stranded in
        self._pending forever. Called from on_optimization_end and
        mark_script_finished, the two places a run's story ends."""
        self._apply_pending_valset()
        self._apply_pending_budget()

    def on_valset_evaluated(self, event: dict) -> None:
        # High-frequency (fires once per valset eval) -- throttled. See
        # _push_throttled for why only the push, not this data capture, is gated.
        key = "valset_evaluated"
        # Sticky-OR the multi-objective flag across throttle windows so a detection
        # made on a call that gets throttled away is never silently dropped by a
        # later, non-evidencing call overwriting self._pending before the next push.
        previously_pending_multi_objective = self._pending.get(key, {}).get("is_multi_objective", False)
        is_multi_objective = self._detect_multi_objective(event) or previously_pending_multi_objective

        def mutate() -> None:
            self._apply_pending_valset()
            self._state.version += 1

        self._push_throttled(
            key,
            {
                "candidate_idx": event.get("candidate_idx"),
                "average_score": event.get("average_score"),
                "is_best_program": event.get("is_best_program"),
                "candidate": event.get("candidate"),
                "is_multi_objective": is_multi_objective,
            },
            mutate,
        )

    def on_budget_updated(self, event: dict) -> None:
        # High-frequency (fires once per metric call) -- throttled, same reasoning
        # as on_valset_evaluated above.
        key = "budget_updated"

        def mutate() -> None:
            self._apply_pending_budget()
            self._state.version += 1

        self._push_throttled(
            key,
            {
                "metric_calls_used": event.get("metric_calls_used"),
                "metric_calls_remaining": event.get("metric_calls_remaining"),
            },
            mutate,
        )

    def on_merge_attempted(self, event: dict) -> None:
        def mutate() -> None:
            self._state.append_log("warn", "merge attempted")

        self._push(mutate)

    def on_merge_accepted(self, event: dict) -> None:
        def mutate() -> None:
            self._state.append_log("success", "merge accepted")

        self._push(mutate)

    def on_merge_rejected(self, event: dict) -> None:
        reason = event.get("reason")

        def mutate() -> None:
            self._state.append_log("info", f"merge rejected (reason: {reason})")

        self._push(mutate)

    def on_error(self, event: dict) -> None:
        exception = event.get("exception")
        will_continue = event.get("will_continue")

        def mutate() -> None:
            self._state.append_log("error", str(exception))
            if not will_continue:
                self._state.status = RunStatus.FAILED
                self._state.error = str(exception)

        self._push(mutate)

    def _stopped_or(self, default: RunStatus) -> RunStatus:
        """`default` unless `<run_dir>/gepa.stop` exists, in which case STOPPED.

        A graceful stop (the `Q` keybinding) touches `gepa.stop` and then just
        waits for GEPA's own `FileStopper` to let the current iteration finish
        naturally -- from GEPA's engine's point of view that's a normal exit,
        indistinguishable from completing on its own. Checking for the stop
        file at completion time is what tells STOPPED apart from COMPLETED
        (mirrors `registry.infer_status`'s same check for historical runs)."""
        if (self._state.run_dir / "gepa.stop").exists():
            return RunStatus.STOPPED
        return default

    def on_optimization_end(self, event: dict) -> None:
        best_candidate_idx = event.get("best_candidate_idx")
        total_iterations = event.get("total_iterations")

        def mutate() -> None:
            self._flush_pending()
            self._state.best_idx = best_candidate_idx
            self._state.total_iterations = total_iterations
            if self._state.status is RunStatus.RUNNING:
                self._state.status = self._stopped_or(RunStatus.COMPLETED)
            self._state.append_log(
                "success" if self._state.status is RunStatus.COMPLETED else "warn",
                "optimization finished" if self._state.status is RunStatus.COMPLETED else "optimization stopped",
            )

        self._push(mutate)

    def on_state_saved(self, event: dict) -> None:
        """GEPA fires this after writing gepa_state.bin/run_log.json/candidates.json
        to disk. DashboardScreen already re-scans the runs root from disk on its
        own timer (_REGISTRY_POLL_INTERVAL_SECONDS), so no explicit registry
        refresh is needed here -- but this is exactly GEPA's own checkpoint
        cadence, so it's also the right moment to refresh our own
        autumn_candidates.json snapshot (see _write_candidates_snapshot) that
        registry.py leans on for per-candidate scores GEPA's own on-disk files
        don't carry."""

        def mutate() -> None:
            self._state.append_log("info", "run state saved to disk")
            self._write_candidates_snapshot()

        self._push(mutate)

    def mark_script_finished(self, exc: BaseException | None) -> None:
        """Autumn-specific completion hook, NOT part of GEPA's GEPACallback
        protocol. Called exactly once by runner.py when the user's script finishes
        running, whether it exited cleanly (exc=None) or raised (exc=<exception>).
        May run on the same background thread runner.py runs the script on, so
        this goes through self._push like every other method here."""

        def mutate() -> None:
            self._flush_pending()
            if exc is None:
                if self._state.status is RunStatus.RUNNING:
                    self._state.status = self._stopped_or(RunStatus.COMPLETED)
                    if self._state.status is RunStatus.STOPPED:
                        self._state.append_log("warn", "script finished (graceful stop)")
                    else:
                        self._state.append_log("success", "script finished")
                else:
                    # GEPA's own on_optimization_end already set a terminal status
                    # (e.g. COMPLETED) -- don't clobber it, just note the exit.
                    self._state.append_log("info", "script exited")
            else:
                self._state.status = RunStatus.FAILED
                self._state.error = str(exc)
                self._state.append_log("error", f"script failed: {exc}")
            self._write_status_marker()
            self._write_candidates_snapshot()

        self._push(mutate)

    def _write_candidates_snapshot(self) -> None:
        """Writes run_dir/autumn_candidates.json: per-candidate val_score,
        discovered_iteration, parent_ids, is_pareto_member, plus best_idx/
        best_score/pareto_front -- everything DashboardCallback already tracks
        in memory that GEPA's own on-disk candidates.json doesn't carry (GEPA
        only persists per-candidate *scores* in the pickled gepa_state.bin,
        which registry.py deliberately never unpickles). Without this,
        registry.py's historical browser can list candidates by index but has
        no way to show which one actually won or what it scored (confirmed by
        hand against a real GEPA run: every score field came back empty).
        candidate `text` isn't duplicated here -- GEPA's own candidates.json
        already carries it and registry.py reads that directly."""
        snapshot = {
            "best_idx": self._state.best_idx,
            "best_score": self._state.best_score,
            "pareto_front": sorted(self._state.pareto_front),
            "candidates": [
                {
                    "idx": row.idx,
                    "val_score": row.val_score,
                    "discovered_iteration": row.discovered_iteration,
                    "parent_ids": row.parent_ids,
                    "is_pareto_member": row.is_pareto_member,
                }
                for row in sorted(self._state.candidates.values(), key=lambda r: r.idx)
            ],
        }
        (self._state.run_dir / "autumn_candidates.json").write_text(json.dumps(snapshot))

    def _write_status_marker(self) -> None:
        """Writes run_dir/autumn_status.json with the final in-memory status.

        GEPA's own on-disk run_log.json carries no terminal marker of any kind --
        the same per-iteration trace entries get written whether the run finishes
        cleanly or is killed mid-iteration, and registry.py deliberately never
        unpickles gepa_state.bin to check. Without this file, a historical run
        that actually completed successfully is indistinguishable on disk from one
        that crashed, and registry.py's status inference would report every real
        run as FAILED (confirmed by hand against a real GEPA run's on-disk
        output). This marker is autumn's own authoritative signal for the common
        graceful-exit path; registry.py falls back to the old best-effort
        heuristic only for runs that predate this marker or were killed before
        mark_script_finished got a chance to run (e.g. kill -9)."""
        marker = {"status": self._state.status.value}
        if self._state.error:
            marker["error"] = self._state.error
        (self._state.run_dir / "autumn_status.json").write_text(json.dumps(marker))
