"""Dataclasses and enums shared across the autumn dashboard."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal


class RunStatus(str, Enum):
    """Lifecycle state of a GEPA optimization run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass
class CandidateRow:
    """A single candidate's row in the candidates table."""

    idx: int
    val_score: float | None
    discovered_iteration: int | None
    parent_ids: list[int | None]
    is_pareto_member: bool
    was_rejected: bool = False
    text: dict[str, str] | None = None


@dataclass
class LogLine:
    """A single timestamped line in the live log feed."""

    timestamp: datetime
    level: Literal["info", "success", "warn", "error"]
    text: str


@dataclass
class ChatMessage:
    """One message in a dashboard-scoped chat transcript."""

    role: Literal["user", "assistant"]
    text: str
    model: str | None = None


@dataclass
class LocalModel:
    """A model installed into Autumn's managed local catalog."""

    name: str
    backend: str
    path: Path
    context_window: int | None = None
    is_default: bool = False


@dataclass
class ModelChoice:
    """The model selected to answer a dashboard prompt."""

    name: str
    backend: str
    path: Path | None
    reason: str


@dataclass
class DashboardState:
    """Mutable in-memory snapshot of a run, watched by Textual widgets."""

    run_name: str
    run_dir: Path
    status: RunStatus = RunStatus.RUNNING
    trainset_size: int | None = None
    valset_size: int | None = None
    max_metric_calls: int | None = None
    metric_calls_used: int = 0
    metric_calls_remaining: int | None = None
    current_iteration: int = 0
    total_iterations: int | None = None
    best_idx: int | None = None
    best_score: float | None = None
    pareto_front: set[int] = field(default_factory=set)
    candidates: dict[int, CandidateRow] = field(default_factory=dict)
    log_lines: deque = field(default_factory=lambda: deque(maxlen=2000))
    is_multi_objective: bool = False
    error: str | None = None
    version: int = 0

    def append_log(self, level: str, text: str) -> None:
        """Convenience helper: appends a LogLine with the current timestamp and bumps version."""
        self.log_lines.append(LogLine(timestamp=datetime.now(), level=level, text=text))
        self.version += 1


@dataclass
class RunSummary:
    """Lightweight summary of a run for the run-picker list."""

    name: str
    run_dir: Path
    status: RunStatus
    best_score: float | None
    num_candidates: int | None
    last_modified: datetime
    is_live: bool = False


@dataclass
class LiveRunSpec:
    """Parameters needed to launch and track a new live run."""

    script_path: Path
    run_dir: Path
    run_name: str
