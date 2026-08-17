from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent_harness.loop_icons import LoopBadgeSpec, parse_loop_badge_spec
from agent_harness.models import (
    DEFAULT_PERMISSION_MODE,
    Loop,
    LoopJournalEntry,
    LoopRun,
    LoopVisibility,
    PermissionMode,
    Provider,
    TeamAgent,
    TeamAgentKind,
    TeamAgentStatus,
    utc_now,
)
from agent_harness.schedules import (
    format_interval_seconds,
    interval_seconds_from_recurrence,
    parse_recurrence_payload,
)
from agent_harness.team import AGENT_CONTEXT_PLACEHOLDER, COLORS, normalize_handle

AGENT_LOOP_SIGNAL_PREFIX = "SLACKGENTIC: LOOP "
AGENT_LOOP_SUMMARY_SIGNAL_PREFIX = "SLACKGENTIC: LOOP_SUMMARY "
AGENT_LOOP_FETCH_SIGNAL_PREFIX = "SLACKGENTIC: LOOP_FETCH "
AGENT_LOOP_COMPACT_SIGNAL_PREFIX = "SLACKGENTIC: LOOP_COMPACT "
LOOP_SIGNAL_PREFIXES_LONGEST_FIRST = (
    AGENT_LOOP_SUMMARY_SIGNAL_PREFIX,
    AGENT_LOOP_COMPACT_SIGNAL_PREFIX,
    AGENT_LOOP_FETCH_SIGNAL_PREFIX,
    AGENT_LOOP_SIGNAL_PREFIX,
)

MAX_ACTIVE_LOOPS = 25
MAX_LOOP_RESOLUTION_ATTEMPTS = 3
LOOP_MIN_INTERVAL_SECONDS = 300
LOOP_MEMORY_CHAR_BUDGET = 24_000
LOOP_COMPACTION_TRIGGER_CHARS = 48_000
LOOP_COMPACT_SNAPSHOT_MAX_CHARS = 6_000
LOOP_FETCH_MAX_PER_RUN = 5
LOOP_FETCH_PAYLOAD_MAX_CHARS = 8_000
LOOP_SUMMARY_MAX_CHARS = 2_000
LOOP_SUMMARY_NUDGE_ATTEMPTS = 1
LOOP_MAX_CONSECUTIVE_FAILURES = 3
LOOP_IGNORED_NOTICE_INTERVAL_SECONDS = 86_400
LOOP_RUNNER_POLL_FLOOR_SECONDS = 5.0

LOOP_CREATE_VERBS = (
    "loop create",
    "create a loop",
    "create loop",
    "start a loop",
    "start loop",
    "new loop",
    "loop:",
)
LOOP_SUMMARY_STATUSES = frozenset({"ok", "found_issue", "action_taken", "failed"})
_ICON_EMOJI_RE = re.compile(r"^[a-z0-9_+-]+$")
_LEADING_MENTION_RE = re.compile(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", re.IGNORECASE)


@dataclass(frozen=True)
class LoopIconSpec:
    emoji: str
    badge: LoopBadgeSpec | None = None


@dataclass(frozen=True)
class LoopSpec:
    title: str
    bot_name: str
    channel_name: str
    mission: str
    recurrence: dict[str, object]
    timezone: str | None
    next_run_at: datetime
    schedule_description: str
    icon: LoopIconSpec


@dataclass(frozen=True)
class LoopSpecParseResult:
    spec: LoopSpec | None = None
    error: str | None = None


@dataclass(frozen=True)
class LoopSummary:
    summary: str
    status: str = "ok"
    carry: dict[str, Any] | None = None


@dataclass(frozen=True)
class LoopSummaryParseResult:
    summary: LoopSummary | None = None
    error: str | None = None


@dataclass(frozen=True)
class LoopFetchRequest:
    run_number: int | None = None
    thread_permalink: str | None = None


@dataclass(frozen=True)
class LoopFetchParseResult:
    request: LoopFetchRequest | None = None
    error: str | None = None


@dataclass(frozen=True)
class LoopCompactParseResult:
    snapshot: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class LoopCreateRequest:
    description: str
    visibility: LoopVisibility = LoopVisibility.PRIVATE
    provider: Provider | None = None
    model: str | None = None
    permission_mode: PermissionMode = DEFAULT_PERMISSION_MODE


@dataclass(frozen=True)
class LoopStatusCommand:
    pass


@dataclass(frozen=True)
class LoopPauseCommand:
    pass


@dataclass(frozen=True)
class LoopResumeCommand:
    pass


@dataclass(frozen=True)
class LoopRunNowCommand:
    pass


@dataclass(frozen=True)
class LoopScheduleCommand:
    text: str


@dataclass(frozen=True)
class LoopTaskCommand:
    text: str


@dataclass(frozen=True)
class LoopNameCommand:
    name: str


@dataclass(frozen=True)
class LoopIconCommand:
    value: str


@dataclass(frozen=True)
class LoopCwdCommand:
    path: str


@dataclass(frozen=True)
class LoopPermissionsCommand:
    permission_mode: PermissionMode


@dataclass(frozen=True)
class LoopCompactNowCommand:
    pass


@dataclass(frozen=True)
class LoopStopCommand:
    archive: bool = False


@dataclass(frozen=True)
class LoopHelpCommand:
    pass


@dataclass(frozen=True)
class LoopListCommand:
    pass


LoopCommand = (
    LoopStatusCommand
    | LoopPauseCommand
    | LoopResumeCommand
    | LoopRunNowCommand
    | LoopScheduleCommand
    | LoopTaskCommand
    | LoopNameCommand
    | LoopIconCommand
    | LoopCwdCommand
    | LoopPermissionsCommand
    | LoopCompactNowCommand
    | LoopStopCommand
    | LoopHelpCommand
    | LoopListCommand
)


def looks_like_loop_create_request(text: str) -> bool:
    normalized = _strip_leading_mention(text).lower()
    return any(
        normalized == verb or normalized.startswith(f"{verb} ") for verb in LOOP_CREATE_VERBS
    )


def parse_loop_create_request(text: str) -> LoopCreateRequest | None:
    cleaned = _strip_leading_mention(text)
    lowered = cleaned.lower()
    verb = next(
        (
            candidate
            for candidate in LOOP_CREATE_VERBS
            if lowered == candidate or lowered.startswith(f"{candidate} ")
        ),
        None,
    )
    if verb is None:
        return None
    description = cleaned[len(verb) :].strip()
    visibility = LoopVisibility.PRIVATE
    visibility_matches = list(
        re.finditer(r"(?<!\S)#(?P<value>public|private)\b", description, re.I)
    )
    if visibility_matches:
        visibility = LoopVisibility(visibility_matches[-1].group("value").lower())
    provider_match = _last_match(r"(?<!\S)provider=(codex|claude)\b", description)
    provider = Provider(provider_match.group(1).lower()) if provider_match else None
    model_match = _last_match(r"(?<!\S)model=([^\s]+)", description)
    model = model_match.group(1).strip() if model_match else None
    permission_mode = (
        PermissionMode.DANGEROUS
        if re.search(r"(?<!\S)#dangerous-mode\b", description, re.I)
        else DEFAULT_PERMISSION_MODE
    )
    description = re.sub(
        r"(?<!\S)#(?:public|private|dangerous-mode)\b", "", description, flags=re.I
    )
    description = re.sub(r"(?<!\S)provider=(?:codex|claude)\b", "", description, flags=re.I)
    description = re.sub(r"(?<!\S)model=[^\s]+", "", description, flags=re.I)
    description = re.sub(r"\s+", " ", description).strip()
    return LoopCreateRequest(
        description=description,
        visibility=visibility,
        provider=provider,
        model=model,
        permission_mode=permission_mode,
    )


def parse_loop_command(text: str) -> LoopCommand | None:
    cleaned = re.sub(r"\s+", " ", _strip_leading_mention(text)).strip()
    if re.fullmatch(r"(?:loops|loop list)", cleaned, re.I):
        return LoopListCommand()
    if re.fullmatch(r"loop status", cleaned, re.I):
        return LoopStatusCommand()
    if re.fullmatch(r"loop pause", cleaned, re.I):
        return LoopPauseCommand()
    if re.fullmatch(r"loop resume", cleaned, re.I):
        return LoopResumeCommand()
    if re.fullmatch(r"loop run now", cleaned, re.I):
        return LoopRunNowCommand()
    if match := re.fullmatch(r"loop schedule:\s*(.+)", cleaned, re.I):
        return LoopScheduleCommand(match.group(1).strip())
    if match := re.fullmatch(r"loop task:\s*(.+)", cleaned, re.I):
        return LoopTaskCommand(match.group(1).strip())
    if match := re.fullmatch(r"loop name:\s*(.+)", cleaned, re.I):
        return LoopNameCommand(match.group(1).strip())
    if match := re.fullmatch(r"loop icon:\s*(.+)", cleaned, re.I):
        return LoopIconCommand(match.group(1).strip())
    if match := re.fullmatch(r"loop cwd:\s*(.+)", cleaned, re.I):
        return LoopCwdCommand(match.group(1).strip())
    if match := re.fullmatch(r"loop permissions:\s*(locked|safe-auto|dangerous)", cleaned, re.I):
        return LoopPermissionsCommand(PermissionMode(match.group(1).lower()))
    if re.fullmatch(r"loop compact now", cleaned, re.I):
        return LoopCompactNowCommand()
    if match := re.fullmatch(r"loop stop(?:\s+(archive))?", cleaned, re.I):
        return LoopStopCommand(archive=bool(match.group(1)))
    if re.fullmatch(r"loop help", cleaned, re.I):
        return LoopHelpCommand()
    return None


def build_loop_resolution_prompt(
    text: str,
    *,
    now: datetime | None = None,
    validation_error: str | None = None,
) -> str:
    reference = now or utc_now()
    example = {
        "title": "AWS Billing Anomaly Watch",
        "bot_name": "Billing Anomaly Bot",
        "channel_name": "loop-aws-billing",
        "mission": "Every run: inspect recent cloud costs, explain anomalies, and open a PR when a concrete code fix exists.",
        "schedule": {
            "frequency": "daily",
            "time": "09:00",
            "timezone": "America/Los_Angeles",
        },
        "icon": {
            "emoji": "money_with_wings",
            "badge": {
                "background": "#0B6E4F",
                "glyph": "$",
                "glyph_color": "#F4FFF9",
                "shape": "circle",
            },
        },
    }
    lines = [
        "Resolve this recurring Slackgentic loop into one implementation-ready specification.",
        "The loop runs in a dedicated Slack channel and accepts instructions only from its owner. "
        "The harness withholds every other member's messages before the agent sees them.",
        "Loops cannot delegate to other agents or read Slack files and attachments.",
        "",
        f"Current UTC time: {reference.astimezone(UTC).isoformat()}",
        f"Owner request: {text.strip()}",
        "",
        "Rewrite the mission as a complete, self-contained standing runbook paragraph. Choose a "
        "short bot name (ending in Bot when natural), a lowercase-dash channel name prefixed "
        "with loop-, and a standard Slack emoji name without surrounding colons.",
        "The schedule must recur. Daily and weekly schedules require HH:MM and an IANA timezone; "
        "weekly schedules also use weekday 0=Monday through 6=Sunday. Interval schedules use "
        f"interval_seconds and must be at least {LOOP_MIN_INTERVAL_SECONDS} seconds.",
        "",
        "Emit no Slack-visible prose on success. Emit exactly one hidden control line:",
        f"{AGENT_LOOP_SIGNAL_PREFIX}<json>",
        "Use this shape:",
        json.dumps(example, indent=2),
    ]
    if validation_error:
        lines.extend(
            [
                "",
                f"Your previous loop control line was invalid: {validation_error}",
                "Emit only a corrected loop control line unless owner clarification is required.",
            ]
        )
    return "\n".join(lines)


def parse_agent_loop_signal(
    signal: str,
    *,
    now: datetime | None = None,
) -> LoopSpecParseResult:
    body = _signal_body(signal, AGENT_LOOP_SIGNAL_PREFIX)
    if body is None:
        return LoopSpecParseResult()
    payload = _json_object(body, label="loop")
    if isinstance(payload, str):
        return LoopSpecParseResult(error=payload)
    for key in ("title", "bot_name", "mission"):
        if not isinstance(payload.get(key), str) or not str(payload[key]).strip():
            return LoopSpecParseResult(error=f"loop JSON must include a non-empty {key}")
    schedule = payload.get("schedule")
    if not isinstance(schedule, dict):
        return LoopSpecParseResult(error="loop JSON must include a schedule object")
    if schedule.get("kind") == "one_off":
        return LoopSpecParseResult(error="loops require a recurring schedule")
    recurrence_result = parse_recurrence_payload(schedule, now=now or utc_now())
    if recurrence_result.error:
        return LoopSpecParseResult(error=recurrence_result.error)
    assert recurrence_result.recurrence is not None
    recurrence = recurrence_result.recurrence
    interval_seconds = interval_seconds_from_recurrence(recurrence.recurrence)
    if interval_seconds is not None and interval_seconds < LOOP_MIN_INTERVAL_SECONDS:
        return LoopSpecParseResult(
            error=f"loop interval must be at least {LOOP_MIN_INTERVAL_SECONDS} seconds"
        )
    bot_name = str(payload["bot_name"]).strip()
    channel_value = payload.get("channel_name")
    if isinstance(channel_value, str) and channel_value.strip():
        channel_name = normalize_loop_channel_name(channel_value)
    else:
        channel_name = normalize_loop_channel_name(f"loop-{bot_name}")
    icon_result = _parse_loop_icon(payload.get("icon"))
    if isinstance(icon_result, str):
        return LoopSpecParseResult(error=icon_result)
    return LoopSpecParseResult(
        spec=LoopSpec(
            title=str(payload["title"]).strip(),
            bot_name=bot_name,
            channel_name=channel_name,
            mission=str(payload["mission"]).strip(),
            recurrence=recurrence.recurrence,
            timezone=recurrence.timezone,
            next_run_at=recurrence.next_run_at,
            schedule_description=recurrence.description,
            icon=icon_result,
        )
    )


def parse_agent_loop_summary_signal(signal: str) -> LoopSummaryParseResult:
    body = _signal_body(signal, AGENT_LOOP_SUMMARY_SIGNAL_PREFIX)
    if body is None:
        return LoopSummaryParseResult()
    payload = _json_object(body, label="loop summary")
    if isinstance(payload, str):
        return LoopSummaryParseResult(error=payload)
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return LoopSummaryParseResult(error="loop summary must include a non-empty summary")
    summary = summary.strip()
    if len(summary) > LOOP_SUMMARY_MAX_CHARS:
        return LoopSummaryParseResult(
            error=f"loop summary must be at most {LOOP_SUMMARY_MAX_CHARS} characters"
        )
    status = payload.get("status", "ok")
    if status not in LOOP_SUMMARY_STATUSES:
        return LoopSummaryParseResult(
            error="loop summary status must be ok, found_issue, action_taken, or failed"
        )
    carry = payload.get("carry")
    if carry is not None and not isinstance(carry, dict):
        return LoopSummaryParseResult(error="loop summary carry must be an object")
    if carry is not None and len(json.dumps(carry, sort_keys=True)) > 4_000:
        return LoopSummaryParseResult(error="loop summary carry must be at most 4000 characters")
    return LoopSummaryParseResult(summary=LoopSummary(summary, str(status), carry))


def parse_agent_loop_fetch_signal(signal: str) -> LoopFetchParseResult:
    body = _signal_body(signal, AGENT_LOOP_FETCH_SIGNAL_PREFIX)
    if body is None:
        return LoopFetchParseResult()
    payload = _json_object(body, label="loop fetch")
    if isinstance(payload, str):
        return LoopFetchParseResult(error=payload)
    run_number = payload.get("run")
    thread_permalink = payload.get("thread")
    if isinstance(run_number, bool) or (
        run_number is not None and (not isinstance(run_number, int) or run_number < 1)
    ):
        return LoopFetchParseResult(error="loop fetch run must be a positive integer")
    if thread_permalink is not None and (
        not isinstance(thread_permalink, str)
        or not thread_permalink.startswith("https://")
        or not thread_permalink.strip()
    ):
        return LoopFetchParseResult(error="loop fetch thread must be an https permalink")
    if (run_number is None) == (thread_permalink is None):
        return LoopFetchParseResult(error="loop fetch must include exactly one of run or thread")
    return LoopFetchParseResult(
        request=LoopFetchRequest(
            run_number=run_number if isinstance(run_number, int) else None,
            thread_permalink=thread_permalink.strip()
            if isinstance(thread_permalink, str)
            else None,
        )
    )


def parse_agent_loop_compact_signal(signal: str) -> LoopCompactParseResult:
    body = _signal_body(signal, AGENT_LOOP_COMPACT_SIGNAL_PREFIX)
    if body is None:
        return LoopCompactParseResult()
    payload = _json_object(body, label="loop compact")
    if isinstance(payload, str):
        return LoopCompactParseResult(error=payload)
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, str) or not snapshot.strip():
        return LoopCompactParseResult(error="loop compact must include a non-empty snapshot")
    snapshot = snapshot.strip()
    if len(snapshot) > LOOP_COMPACT_SNAPSHOT_MAX_CHARS:
        return LoopCompactParseResult(
            error=f"loop compact snapshot must be at most {LOOP_COMPACT_SNAPSHOT_MAX_CHARS} characters"
        )
    return LoopCompactParseResult(snapshot=snapshot)


def loop_spec_to_json(spec: LoopSpec) -> str:
    payload = {
        "title": spec.title,
        "bot_name": spec.bot_name,
        "channel_name": spec.channel_name,
        "mission": spec.mission,
        "recurrence": spec.recurrence,
        "timezone": spec.timezone,
        "next_run_at": spec.next_run_at.isoformat(),
        "schedule_description": spec.schedule_description,
        "icon": {
            "emoji": spec.icon.emoji,
            "badge": (
                {
                    "background": spec.icon.badge.background,
                    "glyph": spec.icon.badge.glyph,
                    "glyph_color": spec.icon.badge.glyph_color,
                    "shape": spec.icon.badge.shape,
                }
                if spec.icon.badge
                else None
            ),
        },
    }
    return json.dumps(payload, sort_keys=True)


def loop_spec_from_json(value: str) -> LoopSpec | None:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    next_run_at = _parse_datetime(payload.get("next_run_at"))
    recurrence = payload.get("recurrence")
    icon = _parse_loop_icon(payload.get("icon"))
    required = ("title", "bot_name", "channel_name", "mission", "schedule_description")
    if (
        next_run_at is None
        or not isinstance(recurrence, dict)
        or isinstance(icon, str)
        or any(not isinstance(payload.get(key), str) for key in required)
    ):
        return None
    timezone = payload.get("timezone")
    if timezone is not None and not isinstance(timezone, str):
        return None
    return LoopSpec(
        title=str(payload["title"]),
        bot_name=str(payload["bot_name"]),
        channel_name=str(payload["channel_name"]),
        mission=str(payload["mission"]),
        recurrence=recurrence,
        timezone=timezone,
        next_run_at=next_run_at,
        schedule_description=str(payload["schedule_description"]),
        icon=icon,
    )


def build_loop_run_prompt(
    loop: Loop,
    run: LoopRun,
    *,
    journal_rendered: str,
    now: datetime,
    bot_name: str | None = None,
) -> str:
    del now
    identity = bot_name or str(loop.metadata.get("bot_name") or loop.title)
    due_text = format_loop_timestamp(run.due_at, loop.timezone)
    schedule = describe_loop_schedule(loop.recurrence, loop.timezone)
    return "\n".join(
        [
            "[LOOP HARNESS: state]",
            f"Loop: {loop.title} — run #{run.run_number}, scheduled for {due_text}.",
            f"You are {identity}, the dedicated automation bot for this loop's Slack channel.",
            "The owner is the only human whose words ever reach you. The harness withholds all "
            "other channel members' messages before you see anything; never speculate about or "
            "ask for them.",
            f"Schedule: {schedule}.",
            "",
            "Mission (your standing instruction):",
            loop.mission,
            "",
            "[LOOP MEMORY]",
            journal_rendered,
            "",
            "[THIS RUN]",
            "Perform the mission now. Post your findings in this thread as you go.",
            "When the run's work is finished, emit exactly one hidden line:",
            f'{AGENT_LOOP_SUMMARY_SIGNAL_PREFIX}{{"summary": "<3-5 sentences>", "status": "ok|found_issue|action_taken|failed", "carry": {{}}}}',
            "To read an earlier run, emit one line and wait:",
            f'{AGENT_LOOP_FETCH_SIGNAL_PREFIX}{{"run": <run number>}}',
            "To replace redundant long-term memory, emit:",
            f'{AGENT_LOOP_COMPACT_SIGNAL_PREFIX}{{"snapshot": "<replacement memory>"}}',
            "Then finish with SLACKGENTIC: THREAD_DONE when nothing remains for this run.",
        ]
    )


def build_loop_compaction_prompt(
    loop: Loop,
    *,
    journal_rendered: str,
    bot_name: str | None = None,
) -> str:
    identity = bot_name or str(loop.metadata.get("bot_name") or loop.title)
    return "\n".join(
        [
            "[LOOP HARNESS: memory compaction]",
            f"You are {identity}, compacting durable memory for {loop.title}.",
            "The owner is the only human whose words are included. The harness withholds every "
            "other member's messages before building this prompt.",
            "Preserve decisions, findings, unresolved work, and state needed for future runs. "
            "Standing owner instructions are shown read-only and remain stored separately; do "
            "not attempt to replace them.",
            "",
            journal_rendered,
            "",
            "Emit exactly one hidden line with a non-empty snapshot:",
            f'{AGENT_LOOP_COMPACT_SIGNAL_PREFIX}{{"snapshot": "<replacement long-term memory, at most {LOOP_COMPACT_SNAPSHOT_MAX_CHARS} characters>"}}',
            "Other channel members' messages are unavailable and must not be inferred.",
        ]
    )


def build_loop_fetch_result(
    *,
    run_number: int,
    rendered_messages: str,
    max_chars: int = LOOP_FETCH_PAYLOAD_MAX_CHARS,
) -> str:
    boundary = (
        "Only the owner and this loop bot are included; all other channel members' messages "
        "are content the harness withholds."
    )
    header = f"[LOOP FETCH RESULT run #{run_number}]\n{boundary}\n"
    footer = "\n[END LOOP FETCH RESULT]"
    available = max(0, max_chars - len(header) - len(footer))
    content = _head_tail(rendered_messages, available)
    return f"{header}{content}{footer}"


def render_loop_journal(
    entries: list[LoopJournalEntry] | tuple[LoopJournalEntry, ...],
    *,
    budget: int = LOOP_MEMORY_CHAR_BUDGET,
) -> str:
    owner_notes = [entry for entry in entries if entry.kind == "owner_note"]
    compactions = [entry for entry in entries if entry.kind == "compaction"]
    recent = [entry for entry in entries if entry.kind in {"run_summary", "system"}]
    sections = ["Standing owner instructions:"]
    sections.extend(
        f"- {entry.content}" for entry in sorted(owner_notes, key=lambda item: item.created_at)
    )
    if not owner_notes:
        sections.append("- (none)")
    sections.extend(["", "Long-term memory:"])
    if compactions:
        newest = max(compactions, key=lambda item: item.created_at)
        sections.append(newest.content)
    else:
        sections.append("(none yet)")
    sections.extend(["", "Recent runs:"])
    base = "\n".join(sections)
    rendered_recent: list[str] = []
    omitted = 0
    for entry in sorted(recent, key=lambda item: item.created_at, reverse=True):
        line = f"- {entry.content}"
        candidate = "\n".join([base, *rendered_recent, line])
        if len(candidate) <= budget:
            rendered_recent.append(line)
        else:
            omitted += 1
    if not rendered_recent:
        rendered_recent.append("- (none yet)")
    if omitted:
        marker = f"({omitted} older entries available — fetch with SLACKGENTIC: LOOP_FETCH)"
        while rendered_recent and len("\n".join([base, *rendered_recent, marker])) > budget:
            if rendered_recent == ["- (none yet)"]:
                rendered_recent.clear()
                break
            rendered_recent.pop()
            omitted += 1
            marker = f"({omitted} older entries available — fetch with SLACKGENTIC: LOOP_FETCH)"
        if len("\n".join([base, *rendered_recent, marker])) <= budget:
            rendered_recent.append(marker)
    return "\n".join([base, *rendered_recent])


def loop_visible_messages(
    loop: Loop,
    messages: list[dict],
    *,
    own_bot_id: str | None,
) -> list[dict]:
    keep = []
    for message in messages:
        if message.get("subtype") not in (None, "bot_message", "thread_broadcast"):
            continue
        from_owner = message.get("user") == loop.owner_slack_user_id and not message.get("bot_id")
        from_our_bot = own_bot_id is not None and message.get("bot_id") == own_bot_id
        if from_owner or from_our_bot:
            keep.append(message)
    return keep


def create_loop_agent(
    *,
    bot_name: str,
    icon_emoji: str,
    provider: Provider,
    existing_handles: set[str],
    sort_order: int,
    loop_id: str,
    metadata: dict,
) -> TeamAgent:
    handle = _unique_loop_handle(bot_name, existing_handles)
    digest = hashlib.sha256(loop_id.encode()).digest()
    words = re.findall(r"[A-Za-z0-9]+", bot_name)
    initials = "".join(word[0].upper() for word in words[:3]) or "LB"
    merged_metadata = {"loop_id": loop_id, "icon_url": None, "icon_badge": None}
    merged_metadata.update(metadata)
    return TeamAgent(
        agent_id=f"loopagent_{uuid.uuid4().hex[:8]}",
        handle=handle,
        full_name=bot_name.strip(),
        initials=initials,
        color_hex=COLORS[int.from_bytes(digest[:4], "big") % len(COLORS)],
        avatar_slug="0",
        icon_emoji=f":{icon_emoji.strip(':') or 'robot_face'}:",
        role="loop",
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("repeat",),
        sort_order=sort_order,
        provider_preference=provider,
        status=TeamAgentStatus.ACTIVE,
        kind=TeamAgentKind.LOOP,
        hired_at=utc_now(),
        metadata=merged_metadata,
    )


def provisional_loop_agent(
    *,
    provider: Provider,
    existing_handles: set[str],
    sort_order: int,
    loop_id: str,
) -> TeamAgent:
    agent = create_loop_agent(
        bot_name="New Loop Bot",
        icon_emoji="robot_face",
        provider=provider,
        existing_handles=existing_handles,
        sort_order=sort_order,
        loop_id=loop_id,
        metadata={"provisional": True},
    )
    provisional_handle = _unique_loop_handle(f"loop-{loop_id[-6:]}", existing_handles)
    return replace(agent, handle=provisional_handle)


def default_loop_provider(commands) -> Provider:
    return Provider.CLAUDE if shutil.which(commands.claude_binary) else Provider.CODEX


def format_loop_timestamp(value: datetime, timezone: str | None) -> str:
    try:
        zone = ZoneInfo(timezone or "UTC")
    except ZoneInfoNotFoundError:
        zone = ZoneInfo("UTC")
    local = value.astimezone(zone)
    clock = local.strftime("%I:%M %p %Z").lstrip("0")
    return f"{local.strftime('%a %b')} {local.day}, {clock}"


def describe_loop_schedule(recurrence: dict[str, object], timezone: str | None) -> str:
    frequency = recurrence.get("frequency")
    if frequency == "interval":
        seconds = interval_seconds_from_recurrence(recurrence)
        return format_interval_seconds(seconds) if seconds is not None else "invalid interval"
    time_text = recurrence.get("time")
    zone = timezone or recurrence.get("timezone") or "UTC"
    if frequency == "weekly":
        weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        weekday = recurrence.get("weekday")
        day = weekdays[weekday] if isinstance(weekday, int) and 0 <= weekday <= 6 else "weekly"
        return f"every {day} at {time_text} {zone}"
    if frequency == "daily":
        return f"daily at {time_text} {zone}"
    return "recurring schedule"


def normalize_loop_channel_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
    return (normalized or "loop-agent")[:80].rstrip("-")


def _parse_loop_icon(value: object) -> LoopIconSpec | str:
    if value is None:
        return LoopIconSpec(emoji="robot_face")
    if not isinstance(value, dict):
        return "loop icon must be an object"
    emoji = value.get("emoji", "robot_face")
    if not isinstance(emoji, str):
        return "icon.emoji must be a standard Slack emoji name"
    emoji = emoji.strip().strip(":")
    if not emoji or not _ICON_EMOJI_RE.fullmatch(emoji):
        return "icon.emoji must contain only lowercase letters, numbers, _, +, or -"
    badge_value = value.get("badge")
    if badge_value is None:
        return LoopIconSpec(emoji=emoji)
    badge = parse_loop_badge_spec(badge_value)
    if isinstance(badge, str):
        return badge
    return LoopIconSpec(emoji=emoji, badge=badge)


def _unique_loop_handle(bot_name: str, existing_handles: set[str]) -> str:
    base = re.sub(r"[^a-z0-9_-]+", "-", bot_name.lower()).strip("-_")
    if not base or not base[0].isalpha():
        base = f"loop-{base}".strip("-")
    base = base[:32].rstrip("-_")
    if len(base) < 2:
        base = "loop-bot"
    try:
        candidate = normalize_handle(base)
    except ValueError:
        candidate = "loop-bot"
    if candidate not in existing_handles:
        return candidate
    suffix = 2
    while True:
        suffix_text = str(suffix)
        candidate = f"{base[: 32 - len(suffix_text)]}{suffix_text}".rstrip("-_")
        if candidate not in existing_handles:
            return normalize_handle(candidate)
        suffix += 1


def _strip_leading_mention(text: str) -> str:
    return _LEADING_MENTION_RE.sub("", text).strip()


def _last_match(pattern: str, value: str):
    matches = list(re.finditer(pattern, value, re.IGNORECASE))
    return matches[-1] if matches else None


def _signal_body(signal: str, prefix: str) -> str | None:
    stripped = signal.strip()
    if not stripped.upper().startswith(prefix):
        return None
    return stripped[len(prefix) :].strip()


def _json_object(body: str, *, label: str) -> dict[str, Any] | str:
    if not body:
        return f"missing {label} JSON"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return f"invalid {label} JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return f"{label} JSON must be an object"
    return payload


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _head_tail(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n… content truncated …\n"
    if limit <= 0:
        return ""
    if limit <= len(marker):
        return marker[:limit]
    available = max(0, limit - len(marker))
    head = available // 2
    return f"{value[:head]}{marker}{value[-(available - head) :]}"
