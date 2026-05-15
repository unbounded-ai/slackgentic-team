from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent_harness.models import (
    AgentTaskKind,
    AssignmentMode,
    ScheduledWorkKind,
    WorkRequest,
    parse_timestamp,
    utc_now,
)

AGENT_SCHEDULE_SIGNAL_PREFIX = "SLACKGENTIC: SCHEDULE "
SCHEDULE_RESOLUTION_METADATA_KEY = "schedule_resolution"
SCHEDULE_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY = "schedule_resolution_original_text"
SCHEDULE_RESOLUTION_ATTEMPTS_METADATA_KEY = "schedule_resolution_attempts"
MAX_SCHEDULE_RESOLUTION_ATTEMPTS = 2

SCHEDULE_VERBS = ("schedule", "remind", "reminder")
ANYONE_TARGETS = {"anyone", "any agent", "somebody", "someone", "whoever"}
WEEKDAYS = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}


@dataclass(frozen=True)
class ParsedAgentSchedule:
    request: WorkRequest
    schedule_kind: ScheduledWorkKind
    next_run_at: datetime
    recurrence: dict[str, object]
    timezone: str | None
    description: str


@dataclass(frozen=True)
class AgentScheduleParseResult:
    schedule: ParsedAgentSchedule | None = None
    error: str | None = None


def looks_like_schedule_request(text: str) -> bool:
    normalized = re.sub(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", "", text).strip().lower()
    return any(normalized.startswith(f"{verb} ") for verb in SCHEDULE_VERBS)


def is_agent_schedule_signal(signal: str) -> bool:
    return signal.strip().upper().startswith(AGENT_SCHEDULE_SIGNAL_PREFIX)


def build_schedule_resolution_prompt(
    text: str,
    agent_handles: list[str] | tuple[str, ...],
    *,
    now: datetime | None = None,
    validation_error: str | None = None,
) -> str:
    reference = now or utc_now()
    handles = ", ".join(f"@{handle}" for handle in agent_handles) or "(no active agents)"
    lines = [
        "Interpret this Slack scheduling request and create exactly one Slackgentic schedule.",
        "",
        f"Current UTC time: {reference.isoformat()}",
        f"Active Slackgentic agent handles: {handles}",
        f"User request: {text.strip()}",
        "",
        "Resolve the schedule with the LLM. Do not use local regex assumptions. If the user "
        "uses location-dependent wording such as sunset, sunrise, a city, or a local event, "
        "compute the concrete next run time before emitting the schedule. Use an IANA timezone "
        "name whenever the request implies a location.",
        "",
        "Emit exactly one hidden control line on its own final line:",
        f"{AGENT_SCHEDULE_SIGNAL_PREFIX}<json>",
        "",
        "The JSON object must have this shape:",
        json.dumps(
            {
                "task": "task for the future agent to run",
                "target": "somebody or an active handle without @",
                "task_kind": "work",
                "dangerous_mode": False,
                "schedule": {
                    "kind": "one_off",
                    "run_at": "2026-05-16T01:23:00Z",
                    "timezone": "America/Chicago",
                    "description": "tomorrow at sunset in Waco",
                },
            },
            indent=2,
        ),
        "",
        "For recurring schedules, use:",
        json.dumps(
            {
                "task": "task for the future agent to run",
                "target": "somebody",
                "task_kind": "work",
                "schedule": {
                    "kind": "recurring",
                    "frequency": "daily",
                    "time": "17:00",
                    "timezone": "America/New_York",
                    "next_run_at": "2026-05-15T21:00:00Z",
                    "description": "every day at 5pm ET",
                },
            },
            indent=2,
        ),
        "",
        "Weekly recurring schedules must set frequency to weekly and include weekday as "
        "0=Monday through 6=Sunday. The target must be either somebody/anyone or one of the "
        "active handles listed above. If the request is ambiguous enough that no schedule can "
        "be created, ask one concise Slack-visible clarification question and do not emit a "
        "control line.",
    ]
    if validation_error:
        lines.extend(
            [
                "",
                f"Your previous schedule control line was invalid: {validation_error}",
                "Try again now. Emit only the corrected control line unless you need a "
                "clarifying answer from the user.",
            ]
        )
    return "\n".join(lines)


def parse_agent_schedule_signal(
    signal: str,
    *,
    known_handles: list[str] | tuple[str, ...],
    now: datetime | None = None,
) -> AgentScheduleParseResult:
    stripped = signal.strip()
    if not stripped.upper().startswith(AGENT_SCHEDULE_SIGNAL_PREFIX):
        return AgentScheduleParseResult()
    body = stripped[len(AGENT_SCHEDULE_SIGNAL_PREFIX) :].strip()
    if not body:
        return AgentScheduleParseResult(error="missing schedule JSON")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return AgentScheduleParseResult(error=f"invalid schedule JSON: {exc.msg}")
    if not isinstance(payload, dict):
        return AgentScheduleParseResult(error="schedule JSON must be an object")
    return _parse_payload(payload, known_handles=known_handles, now=now or utc_now())


def next_run_after(recurrence: dict[str, object], *, after: datetime) -> datetime | None:
    frequency = recurrence.get("frequency")
    timezone = recurrence.get("timezone")
    time_text = recurrence.get("time")
    if not isinstance(frequency, str) or not isinstance(timezone, str):
        return None
    if not isinstance(time_text, str):
        return None
    local_time = _time_from_text(time_text)
    if local_time is None:
        return None
    zone = _zone(timezone)
    if zone is None:
        return None
    local_after = after.astimezone(zone)
    local_due = datetime.combine(local_after.date(), local_time, zone)
    if frequency == "daily":
        if local_due <= local_after:
            local_due += timedelta(days=1)
        return local_due.astimezone(_utc_zone())
    if frequency == "weekly":
        weekday = recurrence.get("weekday")
        if not isinstance(weekday, int):
            return None
        days_ahead = (weekday - local_after.weekday()) % 7
        local_due = datetime.combine(
            local_after.date() + timedelta(days=days_ahead),
            local_time,
            zone,
        )
        if local_due <= local_after:
            local_due += timedelta(days=7)
        return local_due.astimezone(_utc_zone())
    return None


def _parse_payload(
    payload: dict[str, object],
    *,
    known_handles: list[str] | tuple[str, ...],
    now: datetime,
) -> AgentScheduleParseResult:
    task = payload.get("task")
    if not isinstance(task, str) or not task.strip():
        return AgentScheduleParseResult(error="schedule JSON must include a non-empty task")
    target = payload.get("target", "somebody")
    request = _work_request(
        task.strip(),
        target,
        known_handles=known_handles,
        task_kind=payload.get("task_kind"),
        dangerous_mode=payload.get("dangerous_mode"),
    )
    if isinstance(request, str):
        return AgentScheduleParseResult(error=request)
    schedule = payload.get("schedule")
    if not isinstance(schedule, dict):
        return AgentScheduleParseResult(error="schedule JSON must include a schedule object")
    kind = schedule.get("kind")
    if kind == ScheduledWorkKind.ONE_OFF.value:
        return _parse_one_off_schedule(schedule, request, now=now)
    if kind == ScheduledWorkKind.RECURRING.value:
        return _parse_recurring_schedule(schedule, request, now=now)
    return AgentScheduleParseResult(error="schedule.kind must be one_off or recurring")


def _work_request(
    task: str,
    target,
    *,
    known_handles: list[str] | tuple[str, ...],
    task_kind,
    dangerous_mode,
) -> WorkRequest | str:
    normalized_target = str(target or "somebody").strip().lstrip("@").lower()
    if normalized_target in ANYONE_TARGETS:
        assignment_mode = AssignmentMode.ANYONE
        requested_handle = None
    else:
        normalized_handles = {handle.lower(): handle for handle in known_handles}
        requested_handle = normalized_handles.get(normalized_target)
        if requested_handle is None:
            return f"target must be somebody/anyone or an active handle: {', '.join(known_handles)}"
        assignment_mode = AssignmentMode.SPECIFIC
    try:
        parsed_kind = AgentTaskKind(str(task_kind or AgentTaskKind.WORK.value))
    except ValueError:
        return "task_kind must be work or review"
    return WorkRequest(
        prompt=task,
        assignment_mode=assignment_mode,
        requested_handle=requested_handle,
        task_kind=parsed_kind,
        dangerous_mode=bool(dangerous_mode),
    )


def _parse_one_off_schedule(
    schedule: dict[str, object],
    request: WorkRequest,
    *,
    now: datetime,
) -> AgentScheduleParseResult:
    run_at = parse_timestamp(schedule.get("run_at"))
    if run_at is None:
        return AgentScheduleParseResult(error="one_off schedule must include parseable run_at")
    if run_at <= now:
        return AgentScheduleParseResult(error="run_at must be in the future")
    timezone = _timezone(schedule.get("timezone"))
    if schedule.get("timezone") is not None and timezone is None:
        return AgentScheduleParseResult(error="timezone must be an IANA timezone name")
    description = _description(schedule, fallback=f"once at {run_at.isoformat()}")
    return AgentScheduleParseResult(
        schedule=ParsedAgentSchedule(
            request=request,
            schedule_kind=ScheduledWorkKind.ONE_OFF,
            next_run_at=run_at,
            recurrence={},
            timezone=timezone,
            description=description,
        )
    )


def _parse_recurring_schedule(
    schedule: dict[str, object],
    request: WorkRequest,
    *,
    now: datetime,
) -> AgentScheduleParseResult:
    frequency = schedule.get("frequency")
    if frequency not in {"daily", "weekly"}:
        return AgentScheduleParseResult(error="recurring frequency must be daily or weekly")
    timezone = _timezone(schedule.get("timezone"))
    if timezone is None:
        return AgentScheduleParseResult(error="recurring schedule must include IANA timezone")
    time_text = schedule.get("time")
    if not isinstance(time_text, str) or _time_from_text(time_text) is None:
        return AgentScheduleParseResult(error="recurring schedule time must be HH:MM")
    recurrence: dict[str, object] = {
        "frequency": frequency,
        "time": time_text,
        "timezone": timezone,
    }
    if frequency == "weekly":
        weekday = _weekday(schedule.get("weekday"))
        if weekday is None:
            return AgentScheduleParseResult(
                error="weekly recurring schedule must include weekday 0-6"
            )
        recurrence["weekday"] = weekday
    next_run_at = parse_timestamp(schedule.get("next_run_at"))
    if next_run_at is None:
        next_run_at = next_run_after(recurrence, after=now)
    if next_run_at is None:
        return AgentScheduleParseResult(error="could not compute recurring next_run_at")
    if next_run_at <= now:
        return AgentScheduleParseResult(error="next_run_at must be in the future")
    description = _description(schedule, fallback=f"{frequency} at {time_text} {timezone}")
    return AgentScheduleParseResult(
        schedule=ParsedAgentSchedule(
            request=request,
            schedule_kind=ScheduledWorkKind.RECURRING,
            next_run_at=next_run_at,
            recurrence=recurrence,
            timezone=timezone,
            description=description,
        )
    )


def _timezone(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip() if _zone(value.strip()) is not None else None


def _zone(value: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError:
        return None


def _time_from_text(value: str) -> time | None:
    match = re.match(r"^(?P<hour>\d{2}):(?P<minute>\d{2})$", value)
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        return None
    return time(hour=hour, minute=minute)


def _weekday(value) -> int | None:
    if isinstance(value, int) and 0 <= value <= 6:
        return value
    if isinstance(value, str):
        return WEEKDAYS.get(value.strip().lower())
    return None


def _description(schedule: dict[str, object], *, fallback: str) -> str:
    description = schedule.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return fallback


def _utc_zone() -> ZoneInfo:
    return ZoneInfo("UTC")
