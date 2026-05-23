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
    TeamAgent,
    TeamAgentKind,
    WorkRequest,
    utc_now,
)

AGENT_PM_PLAN_SIGNAL_PREFIX = "SLACKGENTIC: PM_PLAN "
PM_RESOLUTION_METADATA_KEY = "pm_resolution"
PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY = "pm_resolution_original_text"
PM_RESOLUTION_ATTEMPTS_METADATA_KEY = "pm_resolution_attempts"
PM_INITIATIVE_ID_METADATA_KEY = "pm_initiative_id"
PM_SUBTASK_LOCAL_ID_METADATA_KEY = "pm_subtask_local_id"
# Marks the resolver task as an extension run. Stored on agent_task.metadata
# so that the PM_PLAN parser knows to allow depends_on references to the
# existing subtasks listed in the metadata value.
PM_EXTENSION_KNOWN_IDS_METADATA_KEY = "pm_extension_known_ids"
PM_EXTENSION_CONTEXT_METADATA_KEY = "pm_extension_context"
PM_REPLAN_CONTEXT_METADATA_KEY = "pm_replan_context"
MAX_PM_RESOLUTION_ATTEMPTS = 3
MAX_PM_SUBTASKS = 20
MIN_PM_SUBTASKS = 1
PM_LOCAL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")

PM_VERBS = ("pm",)
ANYONE_TARGETS = {"anyone", "any agent", "somebody", "someone", "whoever", "any"}


@dataclass(frozen=True)
class ParsedPmSubtask:
    local_id: str
    title: str
    request: WorkRequest
    depends_on: tuple[str, ...]
    after_delay_seconds: int
    co_designers: tuple[str, ...] = ()

    @property
    def is_co_design(self) -> bool:
        return len(self.co_designers) >= 2


@dataclass(frozen=True)
class ParsedPmPlan:
    title: str
    summary: str
    subtasks: tuple[ParsedPmSubtask, ...]


@dataclass(frozen=True)
class AgentPmParseResult:
    plan: ParsedPmPlan | None = None
    error: str | None = None


_PM_LEAD_RE = re.compile(
    r"^\s*(?:<@[A-Z0-9]+>\s*[:,]?\s*)?pm\b(?P<separator>[\s:,-]*)(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_PM_VERB_RE = re.compile(
    r"^(?:plan|break\s+down|breakdown|break\s+this\s+down|kick\s*off|run)\b",
    re.IGNORECASE,
)
_PM_PUNCT_SEPARATORS = {":", ",", "-"}


def looks_like_pm_request(text: str) -> bool:
    """Return True when the message asks the PM to plan a project.

    The message must start with ``pm`` (optionally preceded by a Slack user
    mention) and then either punctuation (``:``, ``,``, ``-``) or one of the
    PM verbs (``plan``, ``break down``, ``kick off``, ``run``), followed by
    at least one word of project description.
    """
    if not text or not text.strip():
        return False
    match = _PM_LEAD_RE.match(text)
    if not match:
        return False
    rest = (match.group("rest") or "").lstrip()
    if not rest:
        return False
    separator = match.group("separator") or ""
    if any(symbol in separator for symbol in _PM_PUNCT_SEPARATORS):
        return True
    verb_match = _PM_VERB_RE.match(rest)
    if not verb_match:
        return False
    after_verb = rest[verb_match.end() :].strip()
    return bool(after_verb)


def extract_pm_request_body(text: str) -> str:
    """Pull just the project description out of a ``pm: ...`` message."""
    match = _PM_LEAD_RE.match(text)
    if not match:
        return text.strip()
    rest = (match.group("rest") or "").strip()
    if not rest:
        return ""
    stripped = _PM_VERB_RE.sub("", rest, count=1).strip()
    return stripped.lstrip(":,-").strip()


def is_agent_pm_plan_signal(signal: str) -> bool:
    return signal.strip().upper().startswith(AGENT_PM_PLAN_SIGNAL_PREFIX)


def build_pm_resolution_prompt(
    text: str,
    agent_handles: list[str] | tuple[str, ...],
    *,
    initiative_id: str,
    now: datetime | None = None,
    validation_error: str | None = None,
    replan_context: str | None = None,
    prior_plan_summary: str | None = None,
    extension_known_ids: tuple[str, ...] = (),
    extension_context: str | None = None,
) -> str:
    reference = now or utc_now()
    handles = ", ".join(f"@{handle}" for handle in agent_handles) or "(no active agents)"
    sample = {
        "title": "Short initiative title",
        "summary": "One to three sentences on what success looks like.",
        "subtasks": [
            {
                "id": "s1",
                "title": "Investigate the current state",
                "task": "Full prompt for the agent that runs this subtask.",
                "target": "somebody",
                "task_kind": "work",
                "dangerous_mode": False,
                "depends_on": [],
                "after_delay_seconds": 0,
            },
            {
                "id": "s2",
                "title": "Co-design the new schema",
                "task": "Brief shared by all co-designers — the synthesis "
                "stage merges their drafts.",
                "co_designers": ["alice", "bob"],
                "task_kind": "work",
                "dangerous_mode": False,
                "depends_on": ["s1"],
                "after_delay_seconds": 0,
            },
            {
                "id": "s3",
                "title": "Implement the change",
                "task": "Full prompt for the agent that runs this subtask.",
                "target": "somebody",
                "task_kind": "work",
                "dangerous_mode": False,
                "depends_on": ["s2"],
                "after_delay_seconds": 0,
            },
        ],
    }
    lines = [
        "You are acting as the Slackgentic PM for one initiative.",
        "Break the user's project into a small DAG of subtasks. Each subtask "
        "becomes a Slackgentic task that another agent will run.",
        "",
        f"Initiative id: {initiative_id}",
        f"Current UTC time: {reference.isoformat()}",
        f"Active Slackgentic agent handles: {handles}",
        "",
        f"User project: {text.strip()}",
        "",
        (
            "Plan rules:\n"
            f"- Emit between {MIN_PM_SUBTASKS} and {MAX_PM_SUBTASKS} subtasks. "
            "Fewer, larger subtasks are better than many micro-steps.\n"
            "- Each subtask `id` is a short slug (letters, digits, '_', '-', '.'), "
            "unique inside the plan.\n"
            "- `depends_on` may reference only ids in this same plan. Cycles are "
            "rejected.\n"
            "- `task` is the full prompt the worker agent will see. Write it as if "
            "you were assigning the subtask to a teammate who has not seen this "
            "conversation. Include the relevant project context and what 'done' "
            "looks like.\n"
            "- `target` is `somebody` (any agent) or one of the active handles "
            "without the leading @. Cross-model reviews are usually best left as "
            "`somebody` so the router can pick the best non-author.\n"
            "- `co_designers` is an OPTIONAL list of 2+ active handles. When "
            "present, the subtask fans out into one draft per co-designer "
            "running in parallel followed by an automatic synthesis stage. Use "
            "this for design work where you want independent perspectives "
            "reconciled — e.g. API design, naming, schema choices. Do not "
            "combine `co_designers` with a specific `target`.\n"
            "- `task_kind` is `work` or `review`.\n"
            "- `after_delay_seconds` is an optional delay applied once every "
            "dependency has finished. Default 0.\n"
            "- Do NOT reference threads, permalinks, or other initiatives in "
            "`depends_on`. If the project needs that, write it inline in the task "
            "prompt of a subtask and let that subtask handle it.\n"
            "- Requirements gathering: if the project is ambiguous, ask one "
            "concise Slack-visible clarification question and do not emit a "
            "control line. The user will reply in the thread; you can iterate "
            "for a few turns before producing the final plan."
        ),
        "",
        "Emit exactly one hidden control line on its own final line:",
        f"{AGENT_PM_PLAN_SIGNAL_PREFIX}<json>",
        "",
        "The JSON object must have this shape:",
        json.dumps(sample, indent=2),
    ]
    if validation_error:
        lines.extend(
            [
                "",
                f"Your previous PM_PLAN control line was invalid: {validation_error}",
                (
                    "Try again now. Emit only the corrected control line unless you "
                    "need a clarifying answer from the user."
                ),
            ]
        )
    if replan_context or prior_plan_summary:
        lines.append("")
        lines.append(
            "This is a REPLAN. An earlier plan for this initiative already ran. "
            "Subtasks that finished are listed below — do NOT include them in the "
            "new plan. Build the new DAG to cover only the remaining work, "
            "incorporating any new context the user provided."
        )
        if prior_plan_summary:
            lines.append("")
            lines.append("Prior plan state:")
            lines.append(prior_plan_summary)
        if replan_context:
            lines.append("")
            lines.append(f"User replan instructions: {replan_context}")
    if extension_known_ids:
        known_ids_text = ", ".join(repr(item) for item in extension_known_ids)
        lines.append("")
        lines.append(
            "This is an EXTENSION. The initiative already has running subtasks "
            "and you are ADDING new work. Reuse existing subtask ids only in "
            "`depends_on`; never redeclare them."
        )
        lines.append(f"Existing subtask ids you may reference: {known_ids_text}")
        if prior_plan_summary:
            lines.append("")
            lines.append("Current plan state:")
            lines.append(prior_plan_summary)
        if extension_context:
            lines.append("")
            lines.append(f"User extension instructions: {extension_context}")
    return "\n".join(lines)


def parse_agent_pm_plan_signal(
    signal: str,
    *,
    known_handles: list[str] | tuple[str, ...],
    allowed_external_dep_ids: tuple[str, ...] = (),
) -> AgentPmParseResult:
    stripped = signal.strip()
    if not stripped.upper().startswith(AGENT_PM_PLAN_SIGNAL_PREFIX):
        return AgentPmParseResult()
    body = stripped[len(AGENT_PM_PLAN_SIGNAL_PREFIX) :].strip()
    if not body:
        return AgentPmParseResult(error="missing PM_PLAN JSON")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return AgentPmParseResult(error=f"invalid PM_PLAN JSON: {exc.msg}")
    if not isinstance(payload, dict):
        return AgentPmParseResult(error="PM_PLAN JSON must be an object")
    return _parse_plan_payload(
        payload,
        known_handles=known_handles,
        allowed_external_dep_ids=allowed_external_dep_ids,
    )


def _parse_plan_payload(
    payload: dict[str, object],
    *,
    known_handles: list[str] | tuple[str, ...],
    allowed_external_dep_ids: tuple[str, ...] = (),
) -> AgentPmParseResult:
    title_raw = payload.get("title")
    if not isinstance(title_raw, str) or not title_raw.strip():
        return AgentPmParseResult(error="title must be a non-empty string")
    title = title_raw.strip()
    summary_raw = payload.get("summary")
    if not isinstance(summary_raw, str) or not summary_raw.strip():
        return AgentPmParseResult(error="summary must be a non-empty string")
    summary = summary_raw.strip()
    subtasks_raw = payload.get("subtasks")
    if not isinstance(subtasks_raw, list) or not subtasks_raw:
        return AgentPmParseResult(error="subtasks must be a non-empty list of objects")
    if len(subtasks_raw) < MIN_PM_SUBTASKS:
        return AgentPmParseResult(error=f"plan must include at least {MIN_PM_SUBTASKS} subtask")
    if len(subtasks_raw) > MAX_PM_SUBTASKS:
        return AgentPmParseResult(
            error=f"plan may include at most {MAX_PM_SUBTASKS} subtasks (got {len(subtasks_raw)})"
        )
    external_dep_ids = frozenset(allowed_external_dep_ids)
    parsed_by_id: dict[str, ParsedPmSubtask] = {}
    declaration_order: list[str] = []
    for index, item in enumerate(subtasks_raw):
        parsed = _parse_subtask(item, known_handles=known_handles, index=index)
        if isinstance(parsed, str):
            return AgentPmParseResult(error=parsed)
        if parsed.local_id in parsed_by_id:
            return AgentPmParseResult(
                error=f"subtask ids must be unique within a plan (duplicate {parsed.local_id!r})"
            )
        if parsed.local_id in external_dep_ids:
            return AgentPmParseResult(
                error=(
                    f"subtask id {parsed.local_id!r} collides with an existing "
                    "subtask in this initiative; pick a new id"
                )
            )
        parsed_by_id[parsed.local_id] = parsed
        declaration_order.append(parsed.local_id)
    for subtask in parsed_by_id.values():
        for dep in subtask.depends_on:
            if dep not in parsed_by_id and dep not in external_dep_ids:
                return AgentPmParseResult(
                    error=(
                        f"subtask {subtask.local_id!r} depends_on {dep!r}, "
                        "which is not in this plan"
                    )
                )
            if dep == subtask.local_id:
                return AgentPmParseResult(
                    error=f"subtask {subtask.local_id!r} cannot depend on itself"
                )
    dag_nodes = [
        (
            local_id,
            tuple(dep for dep in parsed_by_id[local_id].depends_on if dep in parsed_by_id),
        )
        for local_id in declaration_order
    ]
    sorted_ids = topological_sort(dag_nodes)
    if sorted_ids is None:
        cycle = find_dependency_cycle(dag_nodes)
        if cycle:
            cycle_text = " -> ".join(f"{node!r}" for node in cycle)
            return AgentPmParseResult(error=f"plan contains a dependency cycle: {cycle_text}")
        return AgentPmParseResult(error="plan contains a dependency cycle")
    sorted_subtasks = tuple(parsed_by_id[local_id] for local_id in sorted_ids)
    return AgentPmParseResult(
        plan=ParsedPmPlan(title=title, summary=summary, subtasks=sorted_subtasks)
    )


def _parse_subtask(
    item: object,
    *,
    known_handles: list[str] | tuple[str, ...],
    index: int,
) -> ParsedPmSubtask | str:
    if not isinstance(item, dict):
        return f"subtask #{index + 1} must be a JSON object"
    local_id_raw = item.get("id")
    if not isinstance(local_id_raw, str) or not local_id_raw.strip():
        return f"subtask #{index + 1} is missing a non-empty 'id'"
    local_id = local_id_raw.strip()
    if not PM_LOCAL_ID_RE.match(local_id):
        return (
            f"subtask id {local_id!r} must use only letters, digits, '_', '-', '.', "
            "and be at most 32 characters"
        )
    title_raw = item.get("title") or item.get("name") or local_id
    if not isinstance(title_raw, str) or not title_raw.strip():
        return f"subtask {local_id!r} is missing a 'title'"
    title = title_raw.strip()
    task_raw = item.get("task") or item.get("prompt")
    if not isinstance(task_raw, str) or not task_raw.strip():
        return f"subtask {local_id!r} must include a non-empty 'task' prompt"
    task = task_raw.strip()
    normalized_handles = {handle.lower(): handle for handle in known_handles}
    co_designers_raw = item.get("co_designers", [])
    if co_designers_raw is None:
        co_designers_raw = []
    if not isinstance(co_designers_raw, list):
        return f"subtask {local_id!r} co_designers must be a list of handles"
    co_designers: tuple[str, ...] = ()
    if co_designers_raw:
        seen_codesigners: set[str] = set()
        resolved_codesigners: list[str] = []
        for entry in co_designers_raw:
            if not isinstance(entry, str) or not entry.strip():
                return f"subtask {local_id!r} co_designers entries must be non-empty strings"
            normalized = entry.strip().lstrip("@").lower()
            if normalized in ANYONE_TARGETS:
                return (
                    f"subtask {local_id!r} co_designers must name specific handles, not 'somebody'"
                )
            resolved = normalized_handles.get(normalized)
            if resolved is None:
                handles_text = ", ".join(known_handles) or "(none)"
                return (
                    f"subtask {local_id!r} co_designer {entry!r} must be one of the "
                    f"active handles: {handles_text}"
                )
            if resolved in seen_codesigners:
                continue
            seen_codesigners.add(resolved)
            resolved_codesigners.append(resolved)
        if len(resolved_codesigners) == 1:
            return (
                f"subtask {local_id!r} co_designers must list at least 2 distinct "
                "handles; use 'target' for single-agent work"
            )
        co_designers = tuple(resolved_codesigners)
    target_raw = item.get("target", "somebody")
    if not isinstance(target_raw, str) or not target_raw.strip():
        return f"subtask {local_id!r} 'target' must be a non-empty string"
    normalized_target = target_raw.strip().lstrip("@").lower()
    if co_designers:
        # Co-design subtasks fan out into per-designer drafts plus a synthesis
        # subtask; the explicit `target`/`requested_handle` no longer routes a
        # single agent, so leave the WorkRequest in ANYONE mode for the
        # synthesis stage and let the expansion step pin the drafts.
        assignment_mode = AssignmentMode.ANYONE
        requested_handle: str | None = None
    elif normalized_target in ANYONE_TARGETS:
        assignment_mode = AssignmentMode.ANYONE
        requested_handle = None
    else:
        resolved = normalized_handles.get(normalized_target)
        if resolved is None:
            handles_text = ", ".join(known_handles) or "(none)"
            return (
                f"subtask {local_id!r} target {target_raw!r} must be 'somebody' or "
                f"one of the active handles: {handles_text}"
            )
        assignment_mode = AssignmentMode.SPECIFIC
        requested_handle = resolved
    try:
        task_kind = AgentTaskKind(str(item.get("task_kind") or AgentTaskKind.WORK.value))
    except ValueError:
        return f"subtask {local_id!r} task_kind must be 'work' or 'review'"
    dangerous_mode_raw = item.get("dangerous_mode", False)
    if not isinstance(dangerous_mode_raw, bool):
        return f"subtask {local_id!r} dangerous_mode must be a boolean"
    depends_raw = item.get("depends_on", [])
    if depends_raw is None:
        depends_raw = []
    if not isinstance(depends_raw, list):
        return f"subtask {local_id!r} depends_on must be a list of subtask ids"
    seen_deps: set[str] = set()
    depends_on: list[str] = []
    for dep in depends_raw:
        if not isinstance(dep, str) or not dep.strip():
            return f"subtask {local_id!r} depends_on entries must be non-empty strings"
        dep_clean = dep.strip()
        if dep_clean in seen_deps:
            continue
        seen_deps.add(dep_clean)
        depends_on.append(dep_clean)
    delay_raw = item.get("after_delay_seconds", 0)
    if isinstance(delay_raw, bool) or not isinstance(delay_raw, int):
        return f"subtask {local_id!r} after_delay_seconds must be an integer (got {delay_raw!r})"
    if delay_raw < 0:
        return f"subtask {local_id!r} after_delay_seconds must be >= 0"
    request = WorkRequest(
        prompt=task,
        assignment_mode=assignment_mode,
        requested_handle=requested_handle,
        task_kind=task_kind,
        permission_mode=PermissionMode.DANGEROUS if dangerous_mode_raw else DEFAULT_PERMISSION_MODE,
    )
    return ParsedPmSubtask(
        local_id=local_id,
        title=title,
        request=request,
        depends_on=tuple(depends_on),
        after_delay_seconds=int(delay_raw),
        co_designers=co_designers,
    )


CODESIGN_DRAFT_PROMPT_TEMPLATE = (
    "Co-design draft from @{handle}.\n"
    "\n"
    "You are one of {count} co-designers for the same subtask: {title}. The "
    "other co-designers ({others}) are producing their own drafts in parallel. "
    "Focus on YOUR draft from your own perspective — do not try to reconcile "
    "with theirs.\n"
    "\n"
    "Subtask brief:\n"
    "{prompt}\n"
    "\n"
    "When you are done, post your draft in this thread and mark the task "
    "complete. A separate synthesis subtask will reconcile all drafts."
)

CODESIGN_SYNTHESIS_PROMPT_TEMPLATE = (
    "Co-design synthesis for: {title}.\n"
    "\n"
    "{count} co-designers ({handles}) each produced an independent draft for "
    "this subtask. Their threads are linked above as dependencies.\n"
    "\n"
    "Original brief:\n"
    "{prompt}\n"
    "\n"
    "Read each draft, identify where they agree and where they diverge, and "
    "produce a single reconciled output. Call out the trade-offs you made and "
    "name the source draft for each major decision."
)


def serialize_parsed_pm_plan(plan: ParsedPmPlan) -> str:
    """Serialize an expanded ``ParsedPmPlan`` for later replay after approval.

    Stored verbatim on ``pm_initiatives.pending_plan_json``. Deserialization
    skips the JSON parser used for incoming PM signals and rebuilds the
    dataclasses directly, so handle-set drift between planning and approval
    does not invalidate the stored plan.
    """
    payload = {
        "title": plan.title,
        "summary": plan.summary,
        "subtasks": [
            {
                "id": subtask.local_id,
                "title": subtask.title,
                "prompt": subtask.request.prompt,
                "assignment_mode": subtask.request.assignment_mode.value,
                "requested_handle": subtask.request.requested_handle,
                "task_kind": subtask.request.task_kind.value,
                "permission_mode": subtask.request.permission_mode.value,
                "depends_on": list(subtask.depends_on),
                "after_delay_seconds": subtask.after_delay_seconds,
                "co_designers": list(subtask.co_designers),
            }
            for subtask in plan.subtasks
        ],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def deserialize_parsed_pm_plan(blob: str) -> ParsedPmPlan:
    data = json.loads(blob)
    subtasks: list[ParsedPmSubtask] = []
    for item in data["subtasks"]:
        request = WorkRequest(
            prompt=item["prompt"],
            assignment_mode=AssignmentMode(item["assignment_mode"]),
            requested_handle=item.get("requested_handle"),
            task_kind=AgentTaskKind(item.get("task_kind") or AgentTaskKind.WORK.value),
            permission_mode=PermissionMode(
                item.get("permission_mode") or DEFAULT_PERMISSION_MODE.value
            ),
        )
        subtasks.append(
            ParsedPmSubtask(
                local_id=item["id"],
                title=item["title"],
                request=request,
                depends_on=tuple(item.get("depends_on") or ()),
                after_delay_seconds=int(item.get("after_delay_seconds") or 0),
                co_designers=tuple(item.get("co_designers") or ()),
            )
        )
    return ParsedPmPlan(
        title=data["title"],
        summary=data["summary"],
        subtasks=tuple(subtasks),
    )


def expand_codesign_plan(plan: ParsedPmPlan) -> ParsedPmPlan:
    """Fan out co-design subtasks into per-designer drafts plus a synthesis stage.

    A subtask with ``co_designers=(h1, h2, ...)`` is expanded into N parallel
    draft subtasks (one pinned to each handle) and a synthesis subtask that
    depends on all of them. The synthesis subtask keeps the original
    ``local_id`` so that downstream ``depends_on`` references continue to
    resolve to "this subtask is done."
    """
    if not any(item.is_co_design for item in plan.subtasks):
        return plan
    expanded: list[ParsedPmSubtask] = []
    for subtask in plan.subtasks:
        if not subtask.is_co_design:
            expanded.append(subtask)
            continue
        draft_ids: list[str] = []
        for handle in subtask.co_designers:
            draft_id = f"{subtask.local_id}--{handle}"
            others = ", ".join(f"@{other}" for other in subtask.co_designers if other != handle)
            draft_prompt = CODESIGN_DRAFT_PROMPT_TEMPLATE.format(
                handle=handle,
                count=len(subtask.co_designers),
                others=others,
                title=subtask.title,
                prompt=subtask.request.prompt,
            )
            draft_request = WorkRequest(
                prompt=draft_prompt,
                assignment_mode=AssignmentMode.SPECIFIC,
                requested_handle=handle,
                task_kind=subtask.request.task_kind,
                permission_mode=subtask.request.permission_mode,
            )
            expanded.append(
                ParsedPmSubtask(
                    local_id=draft_id,
                    title=f"{subtask.title} — @{handle} draft",
                    request=draft_request,
                    depends_on=subtask.depends_on,
                    after_delay_seconds=subtask.after_delay_seconds,
                )
            )
            draft_ids.append(draft_id)
        synthesis_prompt = CODESIGN_SYNTHESIS_PROMPT_TEMPLATE.format(
            title=subtask.title,
            count=len(subtask.co_designers),
            handles=", ".join(f"@{handle}" for handle in subtask.co_designers),
            prompt=subtask.request.prompt,
        )
        synthesis_request = WorkRequest(
            prompt=synthesis_prompt,
            assignment_mode=AssignmentMode.ANYONE,
            requested_handle=None,
            task_kind=subtask.request.task_kind,
            permission_mode=subtask.request.permission_mode,
        )
        expanded.append(
            ParsedPmSubtask(
                local_id=subtask.local_id,
                title=f"{subtask.title} (synthesis)",
                request=synthesis_request,
                depends_on=tuple(draft_ids),
                after_delay_seconds=0,
            )
        )
    return ParsedPmPlan(title=plan.title, summary=plan.summary, subtasks=tuple(expanded))


PM_TAG_LEAD_RE = re.compile(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*(?P<rest>.*)$", re.DOTALL)

# Inline commands the user can drop into a PM initiative thread to get a
# direct rendering of the plan without going through the PM agent. Matched
# on the whole stripped message (after dropping a leading Slack mention),
# case-insensitive. Only triggers when there is no other text on the line.
_PM_STATUS_COMMAND_RE = re.compile(
    r"^/?(?:pm[\s:_-]+)?(?:status|plan|dag|tree|state)\??\s*$",
    re.IGNORECASE,
)


# Replan command. Matches against the stripped message (after dropping a
# leading Slack mention), and the body — if any — is treated as the user's
# new context to hand to the resolver.
_PM_REPLAN_RE = re.compile(
    r"^/?(?:pm[\s:_-]+)?(?:replan|reroute|redo)\b[\s:,-]*(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)


_PM_EXTEND_RE = re.compile(
    r"^/?(?:pm[\s:_-]+)?(?:extend|add|append)\b[\s:,-]*(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def parse_pm_extension_request(text: str) -> str | None:
    """Return the extension body if ``text`` is a ``pm extend`` command, else None.

    A body is required — extension without instructions is rejected upstream.
    """
    if not text or not text.strip():
        return None
    stripped = text.strip()
    bot_match = PM_TAG_LEAD_RE.match(stripped)
    if bot_match:
        stripped = (bot_match.group("rest") or "").strip()
    if not stripped:
        return None
    match = _PM_EXTEND_RE.match(stripped)
    if not match:
        return None
    body = (match.group("rest") or "").strip()
    if not body:
        return None
    return body


def parse_pm_replan_request(text: str) -> str | None:
    """Return the replan body if ``text`` is a ``pm replan`` command, else None.

    The body may be empty — the PM still gets the failure snapshot. Returns
    None when the message is not a replan command at all.
    """
    if not text or not text.strip():
        return None
    stripped = text.strip()
    bot_match = PM_TAG_LEAD_RE.match(stripped)
    if bot_match:
        stripped = (bot_match.group("rest") or "").strip()
    if not stripped:
        return None
    match = _PM_REPLAN_RE.match(stripped)
    if not match:
        return None
    return (match.group("rest") or "").strip()


def looks_like_pm_status_request(text: str) -> bool:
    if not text or not text.strip():
        return False
    stripped = text.strip()
    bot_match = PM_TAG_LEAD_RE.match(stripped)
    if bot_match:
        stripped = (bot_match.group("rest") or "").strip()
    if not stripped:
        return False
    return bool(_PM_STATUS_COMMAND_RE.match(stripped))


def message_targets_pm_agent(
    text: str,
    pm_handles: list[str] | tuple[str, ...],
) -> str | None:
    """Return the PM handle this message addresses, or None.

    The message must start with ``@pm-handle`` (Slack ``<@U…>`` mention,
    ``@handle`` mention, or plain ``handle:``) and have non-empty body.
    """
    if not text or not text.strip():
        return None
    if not pm_handles:
        return None
    stripped = text.lstrip()
    # Strip a leading Slack user-mention prefix (the dispatcher swaps these in)
    bot_match = PM_TAG_LEAD_RE.match(stripped)
    if bot_match:
        stripped = bot_match.group("rest").lstrip()
    if not stripped:
        return None
    pm_lookup = {handle.lower(): handle for handle in pm_handles}
    handle_match = re.match(
        r"^@?(?P<handle>[A-Za-z][A-Za-z0-9_-]{1,31})\b(?P<rest>.*)$",
        stripped,
        flags=re.DOTALL,
    )
    if not handle_match:
        return None
    handle = handle_match.group("handle").lower()
    resolved = pm_lookup.get(handle)
    if resolved is None:
        return None
    rest = handle_match.group("rest").lstrip(" \t:,-")
    if not rest.strip():
        return None
    return resolved


def filter_pm_agents(agents: list[TeamAgent] | tuple[TeamAgent, ...]) -> list[TeamAgent]:
    return [agent for agent in agents if agent.kind == TeamAgentKind.PM]


def filter_worker_agents(agents: list[TeamAgent] | tuple[TeamAgent, ...]) -> list[TeamAgent]:
    return [agent for agent in agents if agent.kind != TeamAgentKind.PM]


@dataclass(frozen=True)
class PmPlanEstimate:
    """Rough cost/time budget for an approved plan.

    The numbers are best-effort guides surfaced at the approval gate — they
    are NOT enforced and the watchdog does not cancel based on them. They
    only help the user decide whether to approve a plan or trim it first.
    """

    subtask_count: int
    critical_path_depth: int
    dangerous_count: int
    co_design_count: int
    min_wall_clock_seconds: int
    max_wall_clock_seconds: int


# Rough per-subtask wall-clock band used to estimate plan duration.
# Wide on purpose; the goal is "is this a 10-minute plan or an hour-long
# plan?" not a precise prediction.
_PM_SUBTASK_MIN_SECONDS = 120
_PM_SUBTASK_MAX_SECONDS = 900


def critical_path_depth(plan: ParsedPmPlan) -> int:
    """Return the length of the longest dependency chain in ``plan``.

    A plan with no dependencies has depth 1; a chain a -> b -> c has depth 3.
    """
    depth: dict[str, int] = {}
    by_id = {item.local_id: item for item in plan.subtasks}
    for subtask in plan.subtasks:
        if not subtask.depends_on:
            depth[subtask.local_id] = 1
            continue
        best = 0
        for dep in subtask.depends_on:
            if dep in by_id:
                best = max(best, depth.get(dep, 1))
        depth[subtask.local_id] = best + 1
    return max(depth.values(), default=0)


def estimate_pm_plan(plan: ParsedPmPlan) -> PmPlanEstimate:
    depth = critical_path_depth(plan)
    dangerous = sum(1 for item in plan.subtasks if item.request.dangerous_mode)
    co_design = sum(1 for item in plan.subtasks if item.is_co_design)
    return PmPlanEstimate(
        subtask_count=len(plan.subtasks),
        critical_path_depth=depth,
        dangerous_count=dangerous,
        co_design_count=co_design,
        min_wall_clock_seconds=depth * _PM_SUBTASK_MIN_SECONDS,
        max_wall_clock_seconds=len(plan.subtasks) * _PM_SUBTASK_MAX_SECONDS,
    )


def topological_sort(
    nodes: list[tuple[str, tuple[str, ...]]],
) -> list[str] | None:
    """Return a topological order of ``(node, deps)`` pairs, or None on a cycle.

    Roots (nodes with no remaining incoming edges) are emitted in the order
    they were declared in the input. This keeps the plan stable and easy
    for the user to read in Slack.
    """
    indegree: dict[str, int] = {name: 0 for name, _ in nodes}
    edges: dict[str, list[str]] = {name: [] for name, _ in nodes}
    declaration_order = [name for name, _ in nodes]
    for name, deps in nodes:
        for dep in deps:
            if dep not in indegree:
                return None
            edges[dep].append(name)
            indegree[name] += 1
    ready = [name for name in declaration_order if indegree[name] == 0]
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for downstream in edges[current]:
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                ready.append(downstream)
    if len(ordered) != len(nodes):
        return None
    return ordered


def find_dependency_cycle(
    nodes: list[tuple[str, tuple[str, ...]]],
) -> list[str]:
    """Return one cycle as ``[a, b, ..., a]``, or ``[]`` if the graph is acyclic.

    Used to give the user (and the PM resolver retry) a concrete cycle to
    inspect instead of a bare "contains a cycle" message.
    """
    graph: dict[str, list[str]] = {name: [] for name, _ in nodes}
    for name, deps in nodes:
        for dep in deps:
            if dep in graph:
                graph[name].append(dep)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(graph, WHITE)
    parent: dict[str, str | None] = dict.fromkeys(graph, None)
    cycle_edge: tuple[str, str] | None = None
    for start in graph:
        if color[start] != WHITE or cycle_edge is not None:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        color[start] = GRAY
        while stack:
            node, next_index = stack[-1]
            neighbours = graph[node]
            if next_index >= len(neighbours):
                color[node] = BLACK
                stack.pop()
                continue
            stack[-1] = (node, next_index + 1)
            nxt = neighbours[next_index]
            if color[nxt] == WHITE:
                color[nxt] = GRAY
                parent[nxt] = node
                stack.append((nxt, 0))
            elif color[nxt] == GRAY:
                cycle_edge = (node, nxt)
                stack.clear()
                break
    if cycle_edge is None:
        return []
    src, dst = cycle_edge
    path = [src]
    cursor = src
    while cursor != dst:
        nxt_parent = parent[cursor]
        if nxt_parent is None:
            return [dst, src, dst]
        cursor = nxt_parent
        path.append(cursor)
    path.append(src)
    return list(reversed(path))
