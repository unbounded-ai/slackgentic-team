from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

from agent_harness.models import (
    DEFAULT_PERMISSION_MODE,
    AgentTaskKind,
    AssignmentMode,
    PermissionMode,
    WorkDependency,
    WorkDependencyKind,
    WorkRequest,
    parse_timestamp,
    utc_now,
)
from agent_harness.team.routing import parse_work_request

AGENT_DEFERRED_SIGNAL_PREFIX = "SLACKGENTIC: DEPEND "
DEFERRED_RESOLUTION_METADATA_KEY = "deferred_resolution"
DEFERRED_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY = "deferred_resolution_original_text"
DEFERRED_RESOLUTION_ATTEMPTS_METADATA_KEY = "deferred_resolution_attempts"
DEFERRED_RESOLUTION_OCCUPIED_HANDLES_METADATA_KEY = "deferred_resolution_occupied_handles"
MAX_DEFERRED_RESOLUTION_ATTEMPTS = 3

ANYONE_TARGETS = {"anyone", "any agent", "somebody", "someone", "whoever"}

# Detection: either the message starts with after/once/wait-for plus a thread or
# handle plus follow-on instruction, or it is a schedule request that also names
# such a gate.
_AFTER_PREFIX_RE = re.compile(
    r"^\s*(?:<@[A-Z0-9]+>\s*[:,]?\s*)?(after|once|wait\s+for|wait\s+until|when)\b",
    re.IGNORECASE,
)
_DEP_GATE_WORD_RE = (
    r"(?:"
    r"done|"
    r"complete|completed|completes?|"
    r"finish|finished|finishes?|"
    r"landed?|lands?|"
    r"free|frees?\s+up|free\s+up|"
    r"wrapped?\s+up|wraps?\s+up|"
    r"merged?|merges?|"
    r"goes?\s+in"
    r")"
)
_SCHEDULE_WITH_GATE_RE = re.compile(
    rf"\b(after|once|when|wait\s+for|wait\s+until)\b.{{0,200}}\b(?:is|are|be|been|has|have|"
    rf"gets?|becomes?|to)?\s*{_DEP_GATE_WORD_RE}\b",
    re.IGNORECASE | re.DOTALL,
)
# The legacy in-thread "I am waiting on the linked thread" idiom — handled by
# the existing SessionDependency recorder, not by DAG deferred work.
_LEGACY_THREAD_WAIT_PHRASES = (
    "wait for this",
    "wait for that",
    "wait for the other",
    "after this lands",
    "after that lands",
    "when this lands",
    "when that lands",
    "once this goes in",
    "once that goes in",
    "after this goes in",
    "after that goes in",
)
SLACK_PERMALINK_FRAGMENT_RE = re.compile(r"https?://[^/\s>]+\.slack\.com/archives/")
_AT_HANDLE_RE = re.compile(r"@[A-Za-z][A-Za-z0-9_-]{1,31}")


def looks_like_deferred_request(text: str) -> bool:
    stripped = re.sub(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", "", text).strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    starts_with_schedule = any(
        lowered.startswith(verb + " ") for verb in ("schedule", "remind", "reminder")
    )
    has_after_prefix = bool(_AFTER_PREFIX_RE.match(text))
    if not has_after_prefix and not starts_with_schedule:
        return False
    if any(phrase in lowered for phrase in _LEGACY_THREAD_WAIT_PHRASES):
        return False
    has_permalink = bool(SLACK_PERMALINK_FRAGMENT_RE.search(stripped))
    has_at_handle = bool(_AT_HANDLE_RE.search(stripped))
    if not has_permalink and not has_at_handle:
        return False
    if not _SCHEDULE_WITH_GATE_RE.search(stripped):
        # The gate phrasing ("finishes", "is free", "lands", "goes in", "done")
        # is what disambiguates this from an ordinary @handle prompt.
        return False
    # Require a follow-on instruction after the dep clause.
    return bool(_strip_dep_clause(stripped))


def _strip_dep_clause(text: str) -> str:
    cleaned = SLACK_PERMALINK_FRAGMENT_RE.sub(" ", text)
    cleaned = _AT_HANDLE_RE.sub(" ", cleaned)
    # Drop the lead-in connectors so we can see whether real instruction text remains.
    cleaned = re.sub(
        rf"\b(after|once|when|wait\s+for|wait\s+until|and|then|is|are|be|been|has|have|"
        rf"gets?|becomes?|to|,|{_DEP_GATE_WORD_RE})\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[,.;:]+", " ", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"[<>]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def is_agent_deferred_signal(signal: str) -> bool:
    return signal.strip().upper().startswith(AGENT_DEFERRED_SIGNAL_PREFIX)


@dataclass(frozen=True)
class ParsedAgentDeferred:
    request: WorkRequest
    depends_on: tuple[WorkDependency, ...]
    after_dep_delay_seconds: int | None
    run_at: datetime | None
    description: str


@dataclass(frozen=True)
class AgentDeferredParseResult:
    deferred: ParsedAgentDeferred | None = None
    error: str | None = None


def build_deferred_resolution_prompt(
    text: str,
    agent_handles: list[str] | tuple[str, ...],
    *,
    occupied: list[dict[str, str]] | tuple[dict[str, str], ...] = (),
    now: datetime | None = None,
    validation_error: str | None = None,
) -> str:
    reference = now or utc_now()
    handle_text = ", ".join(f"@{handle}" for handle in agent_handles) or "(no active agents)"
    if occupied:
        occupied_lines = "\n".join(
            f"- @{item['handle']} is currently working on task_id={item['task_id']}"
            for item in occupied
            if item.get("handle") and item.get("task_id")
        )
    else:
        occupied_lines = "(no agents are currently occupied)"
    lines = [
        "Interpret this Slack request and create exactly one Slackgentic deferred work entry.",
        "",
        f"Current UTC time: {reference.isoformat()}",
        f"Active Slackgentic agent handles: {handle_text}",
        "Currently occupied agents and their active task ids:",
        occupied_lines,
        "",
        f"User request: {text.strip()}",
        "",
        (
            "Resolve which threads and which busy agents this request depends on. Use the "
            "Slack permalinks the user pasted to identify thread dependencies. Use the "
            "occupied agents list to record agent-busy dependencies as the agent's current "
            "task_id or external_session dependency id (snapshot it now)."
        ),
        "",
        "Emit exactly one hidden control line on its own final line:",
        f"{AGENT_DEFERRED_SIGNAL_PREFIX}<json>",
        "",
        "The JSON object must have this shape:",
        json.dumps(
            {
                "task": "task for the future agent to run",
                "target": "somebody or an active handle without @",
                "task_kind": "work",
                "dangerous_mode": False,
                "depends_on": [
                    {
                        "kind": "thread",
                        "permalink": "https://.../archives/CXXX/p1700000000000000",
                    },
                    {
                        "kind": "agent_busy",
                        "handle": "riley",
                        "task_id": "task_abc",
                    },
                ],
                "delay": {"seconds": 1200},
                "run_at": "2026-05-16T01:23:00Z",
                "description": "after the linked deploy thread finishes and riley is free",
            },
            indent=2,
        ),
        "",
        (
            "Rules: depends_on must include at least one entry. Use kind=thread with the "
            "full permalink the user pasted (do not invent permalinks). Use kind=agent_busy "
            "with handle and the current task_id from the occupied list. If the user writes "
            "'after @ravi is done @eli do X', @ravi is the dependency and @eli is the "
            "target for the future task. Either delay OR "
            "run_at may be set, never both. delay.seconds is the number of seconds to wait "
            "after every dependency is satisfied. run_at is an absolute UTC timestamp; the "
            "task will fire at max(deps_satisfied_at, run_at), never earlier than deps. The "
            "target must be either somebody/anyone or one of the active handles listed "
            "above. If the request is ambiguous enough that no deferred work can be "
            "created, ask one concise Slack-visible clarification question and do not emit "
            "a control line."
        ),
    ]
    if validation_error:
        lines.extend(
            [
                "",
                f"Your previous deferred control line was invalid: {validation_error}",
                (
                    "Try again now. Emit only the corrected control line unless you need a "
                    "clarifying answer from the user."
                ),
            ]
        )
    return "\n".join(lines)


def parse_agent_deferred_signal(
    signal: str,
    *,
    known_handles: list[str] | tuple[str, ...],
    occupied_task_ids: dict[str, str] | None = None,
    now: datetime | None = None,
) -> AgentDeferredParseResult:
    stripped = signal.strip()
    if not stripped.upper().startswith(AGENT_DEFERRED_SIGNAL_PREFIX):
        return AgentDeferredParseResult()
    body = stripped[len(AGENT_DEFERRED_SIGNAL_PREFIX) :].strip()
    if not body:
        return AgentDeferredParseResult(error="missing deferred JSON")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return AgentDeferredParseResult(error=f"invalid deferred JSON: {exc.msg}")
    if not isinstance(payload, dict):
        return AgentDeferredParseResult(error="deferred JSON must be an object")
    return _parse_payload(
        payload,
        known_handles=known_handles,
        occupied_task_ids=occupied_task_ids or {},
        now=now or utc_now(),
    )


def _parse_payload(
    payload: dict[str, object],
    *,
    known_handles: list[str] | tuple[str, ...],
    occupied_task_ids: dict[str, str],
    now: datetime,
) -> AgentDeferredParseResult:
    task = payload.get("task")
    if not isinstance(task, str) or not task.strip():
        return AgentDeferredParseResult(error="deferred JSON must include a non-empty task")
    target = payload.get("target", "somebody")
    request = _work_request(
        task.strip(),
        target,
        known_handles=known_handles,
        task_kind=payload.get("task_kind"),
        dangerous_mode=payload.get("dangerous_mode"),
    )
    if isinstance(request, str):
        return AgentDeferredParseResult(error=request)
    depends_on_raw = payload.get("depends_on")
    if not isinstance(depends_on_raw, list) or not depends_on_raw:
        return AgentDeferredParseResult(
            error="depends_on must be a non-empty list of dependency objects"
        )
    deps: list[WorkDependency] = []
    normalized_handles = {handle.lower(): handle for handle in known_handles}
    seen: set[tuple[str, ...]] = set()
    for item in depends_on_raw:
        parsed = _parse_dependency(
            item,
            normalized_handles=normalized_handles,
            occupied_task_ids=occupied_task_ids,
        )
        if isinstance(parsed, str):
            return AgentDeferredParseResult(error=parsed)
        key = _dep_key(parsed)
        if key in seen:
            continue
        seen.add(key)
        deps.append(parsed)
    if not deps:
        return AgentDeferredParseResult(
            error="depends_on must include at least one resolvable dependency"
        )
    delay_seconds: int | None = None
    delay = payload.get("delay")
    if delay is not None:
        if not isinstance(delay, dict):
            return AgentDeferredParseResult(error="delay must be an object with seconds")
        seconds = delay.get("seconds")
        if not isinstance(seconds, int) or seconds < 0:
            return AgentDeferredParseResult(error="delay.seconds must be a non-negative integer")
        delay_seconds = seconds
    run_at: datetime | None = None
    run_at_value = payload.get("run_at")
    if run_at_value is not None and run_at_value != "":
        parsed_run_at = parse_timestamp(run_at_value)
        if parsed_run_at is None:
            return AgentDeferredParseResult(error="run_at must be a parseable ISO timestamp")
        run_at = parsed_run_at
    if delay_seconds is not None and run_at is not None:
        return AgentDeferredParseResult(error="set at most one of delay or run_at")
    description_value = payload.get("description")
    if isinstance(description_value, str) and description_value.strip():
        description = description_value.strip()
    else:
        description = _default_description(deps, delay_seconds, run_at)
    return AgentDeferredParseResult(
        deferred=ParsedAgentDeferred(
            request=request,
            depends_on=tuple(deps),
            after_dep_delay_seconds=delay_seconds,
            run_at=run_at,
            description=description,
        )
    )


def _parse_dependency(
    item: object,
    *,
    normalized_handles: dict[str, str],
    occupied_task_ids: dict[str, str],
) -> WorkDependency | str:
    if not isinstance(item, dict):
        return "each dependency must be an object"
    kind_value = item.get("kind")
    if kind_value == WorkDependencyKind.THREAD.value:
        permalink = item.get("permalink")
        channel_id = item.get("channel_id")
        thread_ts = item.get("thread_ts")
        if isinstance(permalink, str) and permalink.strip():
            return WorkDependency(
                kind=WorkDependencyKind.THREAD,
                permalink=permalink.strip(),
                channel_id=channel_id if isinstance(channel_id, str) else None,
                thread_ts=thread_ts if isinstance(thread_ts, str) else None,
            )
        if isinstance(channel_id, str) and isinstance(thread_ts, str):
            return WorkDependency(
                kind=WorkDependencyKind.THREAD,
                channel_id=channel_id,
                thread_ts=thread_ts,
            )
        return "thread dependency must include permalink or (channel_id, thread_ts)"
    if kind_value == WorkDependencyKind.AGENT_BUSY.value:
        handle_value = item.get("handle")
        if not isinstance(handle_value, str) or not handle_value.strip():
            return "agent_busy dependency must include handle"
        normalized = handle_value.strip().lstrip("@").lower()
        resolved_handle = normalized_handles.get(normalized)
        if resolved_handle is None:
            return f"agent_busy handle is not an active agent: {handle_value}"
        task_id_value = item.get("task_id")
        if isinstance(task_id_value, str) and task_id_value.strip():
            task_id = task_id_value.strip()
        else:
            task_id = occupied_task_ids.get(resolved_handle)
            if not task_id:
                return (
                    f"agent_busy handle @{resolved_handle} is not currently occupied; "
                    "include task_id explicitly or pick a busy agent"
                )
        return WorkDependency(
            kind=WorkDependencyKind.AGENT_BUSY,
            handle=resolved_handle,
            task_id=task_id,
        )
    return f"dependency kind must be 'thread' or 'agent_busy' (got {kind_value!r})"


def _dep_key(dep: WorkDependency) -> tuple[str, ...]:
    if dep.kind == WorkDependencyKind.THREAD:
        return ("thread", dep.permalink or "", dep.channel_id or "", dep.thread_ts or "")
    return ("agent_busy", dep.handle or "", dep.task_id or "")


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
        embedded_request = parse_work_request(task, known_handles)
        if (
            embedded_request is not None
            and embedded_request.assignment_mode == AssignmentMode.SPECIFIC
        ):
            assignment_mode = AssignmentMode.SPECIFIC
            requested_handle = embedded_request.requested_handle
            task = embedded_request.prompt
        else:
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
        permission_mode=(
            PermissionMode.DANGEROUS if bool(dangerous_mode) else DEFAULT_PERMISSION_MODE
        ),
    )


def _default_description(
    deps: list[WorkDependency],
    delay_seconds: int | None,
    run_at: datetime | None,
) -> str:
    pieces: list[str] = []
    for dep in deps:
        if dep.kind == WorkDependencyKind.THREAD:
            pieces.append(f"thread {dep.permalink or f'{dep.channel_id}/{dep.thread_ts}'}")
        else:
            pieces.append(f"@{dep.handle}")
    gate = "after " + " and ".join(pieces) + " finish"
    if delay_seconds:
        gate += f", then wait {delay_seconds}s"
    elif run_at is not None:
        gate += f", then at {run_at.isoformat(timespec='minutes')}"
    return gate
