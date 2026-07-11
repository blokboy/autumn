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
class ProviderAccount:
    """A signed-in external provider account available for future prompt routing."""

    provider: str
    account_id: str
    display_name: str | None = None
    is_signed_in: bool = False


@dataclass
class ProviderModel:
    """A provider-backed model candidate advertised by an account."""

    name: str
    provider: str
    account_id: str
    is_enabled: bool = True
    priority: int = 100
    is_default: bool = False


@dataclass
class PromptRoutingPolicy:
    """Provider/account candidates considered after runnable local models."""

    provider_accounts: list[ProviderAccount] = field(default_factory=list)
    provider_models: list[ProviderModel] = field(default_factory=list)


@dataclass
class CatalogEntry:
    """A single selectable row in Autumn's unified model catalog -- local
    models and any eligible provider-backed candidates from a
    `PromptRoutingPolicy`, addressed by (group, name) so the Models tab and
    `model_router.choose_model`'s fallback chain share one shape. `group` is
    "Local" for installed local models, or a provider name (e.g. "groq") for
    provider-backed entries.

    `disabled` marks a row that is visible in the Models tab but never
    selectable (see `stub_providers.py`, #15) -- always `False` for anything
    that comes out of `catalog.build_entries`, which is the only entry list
    `model_router.choose_model` ever consults, so a disabled row can never
    accidentally become the model an actual prompt gets routed to."""

    group: str
    name: str
    backend: str
    path: Path | None = None
    context_window: int | None = None
    provider: str | None = None
    account_id: str | None = None
    is_default: bool = False
    disabled: bool = False


@dataclass
class ModelChoice:
    """The model selected to answer a dashboard prompt."""

    name: str
    backend: str
    path: Path | None
    reason: str
    context_window: int | None = None
    provider: str | None = None
    account_id: str | None = None


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
