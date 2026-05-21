from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

DANGEROUS_MODE_METADATA_KEY = "dangerous_mode"
PERMISSION_MODE_METADATA_KEY = "permission_mode"
ASSIGNMENT_PROMPT_METADATA_KEY = "assignment_prompt"
ROSTER_SUMMARY_METADATA_KEY = "roster_summary"
PR_URL_METADATA_KEY = "pr_url"
PR_URLS_METADATA_KEY = "pr_urls"
EXTERNAL_SESSION_DEPENDENCY_PREFIX = "external_session:"
DEFERRED_WORK_DEPENDENCY_PREFIX = "deferred_work:"
SCHEDULED_WORK_DEPENDENCY_PREFIX = "scheduled_work:"


class Provider(StrEnum):
    CODEX = "codex"
    CLAUDE = "claude"


class PermissionMode(StrEnum):
    LOCKED = "locked"
    SAFE_AUTO = "safe-auto"
    DANGEROUS = "dangerous"


DEFAULT_PERMISSION_MODE = PermissionMode.SAFE_AUTO


class SessionStatus(StrEnum):
    ACTIVE = "active"
    IDLE = "idle"
    DONE = "done"
    UNKNOWN = "unknown"


class ControlMode(StrEnum):
    MANAGED = "managed"
    OBSERVED = "observed"
    ADOPTABLE = "adoptable"


class TeamAgentStatus(StrEnum):
    ACTIVE = "active"
    FIRED = "fired"


class TeamAgentKind(StrEnum):
    ENGINEER = "engineer"
    PM = "pm"


DEFAULT_TEAM_AGENT_KIND = TeamAgentKind.ENGINEER


class AgentTaskStatus(StrEnum):
    QUEUED = "queued"
    ACTIVE = "active"
    DONE = "done"
    CANCELLED = "cancelled"


class AgentTaskKind(StrEnum):
    WORK = "work"
    REVIEW = "review"


class PendingWorkRequestStatus(StrEnum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    CANCELLED = "cancelled"


class ScheduledTimerStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    FIRED = "fired"
    CANCELLED = "cancelled"


class ScheduledWorkStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"
    CANCELLED = "cancelled"


class ScheduledWorkKind(StrEnum):
    ONE_OFF = "one_off"
    RECURRING = "recurring"


class DeferredWorkStatus(StrEnum):
    WAITING_DEPS = "waiting_deps"
    READY = "ready"
    CLAIMED = "claimed"
    DONE = "done"
    CANCELLED = "cancelled"


class WorkDependencyKind(StrEnum):
    THREAD = "thread"
    AGENT_BUSY = "agent_busy"
    SUBTASK = "subtask"


class PmInitiativeStatus(StrEnum):
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    ACTIVE = "active"
    DONE = "done"
    CANCELLED = "cancelled"


class AssignmentMode(StrEnum):
    ANYONE = "anyone"
    SPECIFIC = "specific"


@dataclass(frozen=True)
class AgentSession:
    provider: Provider
    session_id: str
    transcript_path: Path
    cwd: Path | None = None
    started_at: datetime | None = None
    last_seen_at: datetime | None = None
    status: SessionStatus = SessionStatus.UNKNOWN
    control_mode: ControlMode = ControlMode.OBSERVED
    model: str | None = None
    git_branch: str | None = None
    permission_mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentEvent:
    provider: Provider
    session_id: str
    timestamp: datetime | None
    event_type: str
    text: str | None = None
    source_path: Path | None = None
    line_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    def plus(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_output_tokens=(self.reasoning_output_tokens + other.reasoning_output_tokens),
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True)
class RateLimitWindow:
    used_percent: float | None
    window_minutes: int | None
    resets_at: datetime | None


@dataclass(frozen=True)
class UsageSnapshot:
    provider: Provider
    as_of: datetime
    session_id: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    context_window: int | None = None
    primary_limit: RateLimitWindow | None = None
    secondary_limit: RateLimitWindow | None = None
    plan_type: str | None = None
    remaining_description: str | None = None


@dataclass(frozen=True)
class TeamAgent:
    agent_id: str
    handle: str
    full_name: str
    initials: str
    color_hex: str
    avatar_slug: str
    icon_emoji: str
    role: str
    personality: str
    voice: str
    unique_strength: str
    reaction_names: tuple[str, ...]
    sort_order: int
    provider_preference: Provider | None = None
    status: TeamAgentStatus = TeamAgentStatus.ACTIVE
    kind: TeamAgentKind = DEFAULT_TEAM_AGENT_KIND
    hired_at: datetime | None = None
    fired_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pm(self) -> bool:
        return self.kind == TeamAgentKind.PM


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    agent_id: str
    prompt: str
    channel_id: str
    kind: AgentTaskKind
    status: AgentTaskStatus
    created_at: datetime
    updated_at: datetime
    requested_by_slack_user: str | None = None
    thread_ts: str | None = None
    parent_message_ts: str | None = None
    session_provider: Provider | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkRequest:
    prompt: str
    assignment_mode: AssignmentMode
    requested_handle: str | None = None
    task_kind: AgentTaskKind = AgentTaskKind.WORK
    author_handle: str | None = None
    pr_url: str | None = None
    permission_mode: PermissionMode = DEFAULT_PERMISSION_MODE

    @property
    def dangerous_mode(self) -> bool:
        return self.permission_mode == PermissionMode.DANGEROUS


@dataclass(frozen=True)
class SlackThreadRef:
    channel_id: str
    thread_ts: str
    message_ts: str | None = None
    permalink: str | None = None


@dataclass(frozen=True)
class PendingWorkRequest:
    pending_id: str
    channel_id: str
    thread_ts: str
    request: WorkRequest
    requested_by_slack_user: str | None
    status: PendingWorkRequestStatus
    created_at: datetime
    updated_at: datetime
    message_ts: str | None = None
    author_agent_id: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    exclude_agent_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionDependency:
    blocked_session_id: str
    blocking_thread: SlackThreadRef
    created_by_slack_user: str | None = None
    reason: str | None = None
    status: str = "waiting"


@dataclass(frozen=True)
class ScheduledTimer:
    timer_id: str
    task_id: str
    agent_id: str
    channel_id: str
    thread_ts: str
    prompt: str
    due_at: datetime
    status: ScheduledTimerStatus
    created_at: datetime
    updated_at: datetime
    parent_message_ts: str | None = None


@dataclass(frozen=True)
class ScheduledWork:
    schedule_id: str
    channel_id: str
    thread_ts: str
    prompt: str
    assignment_mode: AssignmentMode
    task_kind: AgentTaskKind
    schedule_kind: ScheduledWorkKind
    status: ScheduledWorkStatus
    next_run_at: datetime
    created_at: datetime
    updated_at: datetime
    message_ts: str | None = None
    requested_handle: str | None = None
    author_handle: str | None = None
    pr_url: str | None = None
    requested_by_slack_user: str | None = None
    dangerous_mode: bool = False
    recurrence: dict[str, Any] = field(default_factory=dict)
    timezone: str | None = None
    last_run_at: datetime | None = None
    last_task_id: str | None = None


@dataclass(frozen=True)
class WorkDependency:
    kind: WorkDependencyKind
    channel_id: str | None = None
    thread_ts: str | None = None
    permalink: str | None = None
    task_id: str | None = None
    handle: str | None = None
    description: str | None = None
    initiative_id: str | None = None
    local_id: str | None = None


@dataclass(frozen=True)
class PmInitiative:
    initiative_id: str
    channel_id: str
    thread_ts: str
    title: str
    summary: str
    status: PmInitiativeStatus
    created_at: datetime
    updated_at: datetime
    message_ts: str | None = None
    requested_by_slack_user: str | None = None
    pm_agent_id: str | None = None
    pm_task_id: str | None = None
    watchdog_last_run_at: datetime | None = None
    pending_plan_json: str | None = None
    pending_plan_message_ts: str | None = None


@dataclass(frozen=True)
class PmSubtask:
    initiative_id: str
    local_id: str
    title: str
    deferred_id: str
    depends_on: tuple[str, ...]
    sort_order: int
    created_at: datetime


@dataclass(frozen=True)
class DeferredWork:
    deferred_id: str
    channel_id: str
    thread_ts: str
    prompt: str
    assignment_mode: AssignmentMode
    task_kind: AgentTaskKind
    status: DeferredWorkStatus
    depends_on: tuple[WorkDependency, ...]
    created_at: datetime
    updated_at: datetime
    message_ts: str | None = None
    requested_handle: str | None = None
    author_handle: str | None = None
    pr_url: str | None = None
    requested_by_slack_user: str | None = None
    dangerous_mode: bool = False
    after_dep_delay_seconds: int | None = None
    run_at: datetime | None = None
    fire_at: datetime | None = None
    deps_satisfied_at: datetime | None = None
    dispatched_at: datetime | None = None
    last_task_id: str | None = None
    description: str | None = None


def external_session_dependency_id(provider: Provider, session_id: str) -> str:
    return f"{EXTERNAL_SESSION_DEPENDENCY_PREFIX}{provider.value}:{session_id}"


def deferred_work_dependency_id(deferred_id: str) -> str:
    return f"{DEFERRED_WORK_DEPENDENCY_PREFIX}{deferred_id}"


def scheduled_work_dependency_id(schedule_id: str) -> str:
    return f"{SCHEDULED_WORK_DEPENDENCY_PREFIX}{schedule_id}"


def parse_external_session_dependency_id(value: str | None) -> tuple[Provider, str] | None:
    if not value or not value.startswith(EXTERNAL_SESSION_DEPENDENCY_PREFIX):
        return None
    body = value.removeprefix(EXTERNAL_SESSION_DEPENDENCY_PREFIX)
    provider_text, separator, session_id = body.partition(":")
    if not separator or not session_id:
        return None
    try:
        provider = Provider(provider_text)
    except ValueError:
        return None
    return provider, session_id


def parse_deferred_work_dependency_id(value: str | None) -> str | None:
    if not value or not value.startswith(DEFERRED_WORK_DEPENDENCY_PREFIX):
        return None
    deferred_id = value.removeprefix(DEFERRED_WORK_DEPENDENCY_PREFIX)
    return deferred_id or None


def parse_scheduled_work_dependency_id(value: str | None) -> str | None:
    if not value or not value.startswith(SCHEDULED_WORK_DEPENDENCY_PREFIX):
        return None
    schedule_id = value.removeprefix(SCHEDULED_WORK_DEPENDENCY_PREFIX)
    return schedule_id or None


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_timestamp(value: str | int | float | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    text = str(value)
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=UTC)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
