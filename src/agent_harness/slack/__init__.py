from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_harness.models import AgentTask, Provider, SlackThreadRef, TeamAgent
from agent_harness.team import (
    DEFAULT_CLAUDE_TEAM_SIZE,
    DEFAULT_CODEX_TEAM_SIZE,
    agent_identity_label,
)
from agent_harness.team.routing import parse_lightweight_handles

SLACK_PERMALINK_RE = re.compile(
    r"https://(?P<workspace>[^/]+)/archives/(?P<channel>[A-Z0-9]+)/p(?P<packed_ts>\d{16})"
    r"(?:\?(?P<query>[^>\s|]+))?"
)


@dataclass(frozen=True)
class AgentRosterStatus:
    label: str
    detail: str | None = None
    thread_url: str | None = None
    task_id: str | None = None
    session_provider: Provider | None = None
    session_id: str | None = None


def build_setup_modal(
    callback_id: str = "setup.initial",
    default_repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = str(default_repo_root or _default_repo_root())
    return {
        "type": "modal",
        "callback_id": callback_id,
        "title": {"type": "plain_text", "text": "Set up team"},
        "submit": {"type": "plain_text", "text": "Create"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "input",
                "block_id": "channel_name",
                "label": {"type": "plain_text", "text": "Agent channel name"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "initial_value": "agents",
                },
            },
            {
                "type": "input",
                "block_id": "visibility",
                "label": {"type": "plain_text", "text": "Channel visibility"},
                "element": {
                    "type": "radio_buttons",
                    "action_id": "value",
                    "initial_option": _option("Private", "private"),
                    "options": [
                        _option("Private", "private"),
                        _option("Public", "public"),
                    ],
                },
            },
            {
                "type": "input",
                "block_id": "codex_count",
                "label": {"type": "plain_text", "text": "Codex agents"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "initial_value": str(DEFAULT_CODEX_TEAM_SIZE),
                    "placeholder": {"type": "plain_text", "text": str(DEFAULT_CODEX_TEAM_SIZE)},
                },
            },
            {
                "type": "input",
                "block_id": "claude_count",
                "label": {"type": "plain_text", "text": "Claude agents"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "initial_value": str(DEFAULT_CLAUDE_TEAM_SIZE),
                    "placeholder": {"type": "plain_text", "text": str(DEFAULT_CLAUDE_TEAM_SIZE)},
                },
            },
            {
                "type": "input",
                "block_id": "repo_root",
                "label": {"type": "plain_text", "text": "Repos root"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "initial_value": repo_root,
                    "placeholder": {"type": "plain_text", "text": "~/code"},
                },
                "hint": {
                    "type": "plain_text",
                    "text": "Agents launch here by default. Named sibling repos can be selected from this root.",
                },
            },
        ],
    }


def build_start_session_modal(callback_id: str = "session.start") -> dict[str, Any]:
    return {
        "type": "modal",
        "callback_id": callback_id,
        "title": {"type": "plain_text", "text": "Start agent"},
        "submit": {"type": "plain_text", "text": "Start"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "input",
                "block_id": "agent",
                "label": {"type": "plain_text", "text": "Agent"},
                "element": {
                    "type": "static_select",
                    "action_id": "provider",
                    "options": [
                        _option("Codex", Provider.CODEX.value),
                        _option("Claude", Provider.CLAUDE.value),
                    ],
                },
            },
            {
                "type": "input",
                "block_id": "cwd",
                "label": {"type": "plain_text", "text": "Working directory"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "placeholder": {"type": "plain_text", "text": "/path/to/repo"},
                },
            },
            {
                "type": "input",
                "block_id": "prompt",
                "label": {"type": "plain_text", "text": "Prompt"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "multiline": True,
                },
            },
            {
                "type": "input",
                "block_id": "permissions",
                "optional": True,
                "label": {"type": "plain_text", "text": "Permissions"},
                "element": {
                    "type": "checkboxes",
                    "action_id": "dangerous",
                    "options": [
                        _option(
                            "Run with dangerous permission bypass",
                            "dangerous",
                            (
                                "Codex: --dangerously-bypass-approvals-and-sandbox; "
                                "Claude: --dangerously-skip-permissions"
                            ),
                        )
                    ],
                },
            },
            {
                "type": "input",
                "block_id": "worktree",
                "optional": True,
                "label": {"type": "plain_text", "text": "Worktree or branch"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "placeholder": {"type": "plain_text", "text": "optional"},
                },
            },
        ],
    }


def build_team_roster_blocks(
    agents: list[TeamAgent],
    statuses: dict[str, AgentRosterStatus] | None = None,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Agent team*  {len(agents)} active lightweight handles\n"
                    f"{_provider_breakdown_text(agents)}"
                ),
            },
        },
        {
            "type": "actions",
            "block_id": "team.roster.actions",
            "elements": [
                _button(
                    "Hire Auto",
                    "team.hire.auto",
                    encode_action_value("team.hire", count=1),
                    "primary",
                ),
                _button(
                    "Hire Codex",
                    "team.hire.codex",
                    encode_action_value("team.hire", count=1, provider=Provider.CODEX.value),
                ),
                _button(
                    "Hire Claude",
                    "team.hire.claude",
                    encode_action_value("team.hire", count=1, provider=Provider.CLAUDE.value),
                ),
            ],
        },
    ]
    for agent in agents:
        status = statuses.get(agent.agent_id) if statuses else None
        status_text = _agent_status_text(status)
        elements = []
        if status and status.task_id:
            elements.append(
                _button(
                    "Free up",
                    "task.done",
                    encode_action_value("task.done", task_id=status.task_id),
                    "primary",
                )
            )
        elif status and status.session_provider and status.session_id:
            elements.append(
                _button(
                    "Free up",
                    "external.session.finish",
                    encode_action_value(
                        "external.session.finish",
                        provider=status.session_provider.value,
                        session_id=status.session_id,
                    ),
                    "primary",
                )
            )
        if status and status.thread_url:
            elements.append(
                _button(
                    "Open thread",
                    "thread.open",
                    encode_action_value("thread.open"),
                    url=status.thread_url,
                )
            )
        elements.append(
            _button(
                "Fire",
                "team.fire",
                encode_action_value("team.fire", agent_id=agent.agent_id, handle=agent.handle),
                "danger",
            )
        )
        blocks.append(
            {
                "type": "section",
                "block_id": f"team.agent.{agent.agent_id}",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{agent_identity_label(agent)}*\n{status_text}",
                },
            }
        )
        blocks.append(
            {
                "type": "actions",
                "block_id": f"team.agent.actions.{agent.agent_id}",
                "elements": elements,
            }
        )
    return blocks


def _agent_status_text(status: AgentRosterStatus | None) -> str:
    if status is None:
        return "Available"
    parts = [status.label]
    if status.detail:
        parts.append(status.detail)
    return ": ".join(parts)


def _provider_breakdown_text(agents: list[TeamAgent]) -> str:
    counts = {Provider.CODEX: 0, Provider.CLAUDE: 0}
    unmapped = 0
    for agent in agents:
        if agent.provider_preference in counts:
            counts[agent.provider_preference] += 1
        else:
            unmapped += 1
    parts = [
        f"Codex {counts[Provider.CODEX]}",
        f"Claude {counts[Provider.CLAUDE]}",
    ]
    if unmapped:
        parts.append(f"Unmapped {unmapped}")
    return " / ".join(parts)


def build_channel_overview_blocks(
    slash_command: str,
    codex_command: str,
    claude_command: str,
) -> list[dict[str, Any]]:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Slackgentic is ready.*\n"
                    "Write anything in this channel to start a task, or write "
                    "`@agentname ...` to ask a specific agent. The agent replies "
                    "in your thread."
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Thread subtasks:*\n"
                    "Reply with `somebody ...` in a task thread to bring in another "
                    "agent for that subtask. The original agent picks the thread back "
                    "up with the added context."
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Commands:* type them directly in this channel, "
                    f"or run them as `{slash_command} <command>`\n"
                    f"`{slash_command} status`  usage and active sessions\n"
                    f"`{slash_command} show roster`  current team\n"
                    f"`{slash_command} hire 3 agents`  add capacity\n"
                    f"`{slash_command} fire everyone`  clear the team\n"
                    "`status`, `show roster`, `hire 3 agents` also work here"
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Sessions started outside Slack:*\n"
                    f"*Codex:* `{codex_command}`\n"
                    "*Claude:* run `slackgentic claude-channel --install` once, then "
                    f"`{claude_command}`\n"
                    "Each command creates a tracked Slack thread here. Restart already-open "
                    "Claude sessions after installing the channel. Slack replies and native "
                    "Claude tool approvals relay through it; no extra MCP flag is needed unless "
                    "you use `--strict-mcp-config`."
                ),
            },
        },
    ]


def build_external_session_capacity_blocks(
    provider: Provider,
    waiting_count: int = 1,
) -> list[dict[str, Any]]:
    label = provider.value.title()
    plural = "session" if waiting_count == 1 else "sessions"
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*No {label} team seat is available.*\n"
                    f"{waiting_count} {label} {plural} started outside Slack waiting. Hire one "
                    "matching agent and Slackgentic will backfill visible transcript "
                    "output into the tracked thread."
                ),
            },
        },
        {
            "type": "actions",
            "block_id": f"external.capacity.{provider.value}",
            "elements": [
                _button(
                    f"Hire 1 {label} agent",
                    f"external.capacity.hire.{provider.value}",
                    encode_action_value("team.hire", count=1, provider=provider.value),
                    "primary",
                )
            ],
        },
    ]


def build_task_thread_blocks(
    task: AgentTask,
    agent: TeamAgent,
    *,
    include_actions: bool = True,
) -> list[dict[str, Any]]:
    task_label = "PR review" if task.kind.value == "review" else "task"
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{agent.full_name}* `@{agent.handle}` picked up a {task_label}.\n"
                    f"*Task:* {task.prompt}"
                ),
            },
        },
    ]
    if include_actions:
        blocks.append(
            {
                "type": "actions",
                "block_id": f"task.actions.{task.task_id}",
                "elements": [
                    _button(
                        "Finish and free up this agent",
                        "task.done",
                        encode_action_value("task.done", task_id=task.task_id),
                        "primary",
                    ),
                ],
            }
        )
    return blocks


def normalize_slack_mrkdwn(text: str) -> str:
    segments = re.split(r"(```.*?```)", _wrap_markdown_tables(text), flags=re.DOTALL)
    return "".join(
        segment if segment.startswith("```") else _normalize_bold_markers(segment)
        for segment in segments
    )


def slack_blocks_for_markdown_table(text: str) -> list[dict[str, Any]] | None:
    parsed = _extract_single_markdown_table(text)
    if parsed is None:
        return None
    before, rows, after = parsed
    if len(rows) > 100 or not rows or len(rows[0]) > 20:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        return None
    blocks: list[dict[str, Any]] = []
    blocks.extend(_markdown_section_blocks(before))
    blocks.append(
        {
            "type": "table",
            "column_settings": [{"is_wrapped": True} for _ in range(width)],
            "rows": [
                [_table_cell(cell, bold=row_index == 0) for cell in row]
                for row_index, row in enumerate(rows)
            ],
        }
    )
    blocks.extend(_markdown_section_blocks(after))
    return blocks[:50]


def _extract_single_markdown_table(text: str) -> tuple[str, list[list[str]], str] | None:
    segments = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    match: tuple[str, list[list[str]], str] | None = None
    before_segments: list[str] = []
    for segment_index, segment in enumerate(segments):
        if segment.startswith("```"):
            before_segments.append(segment)
            continue
        parsed = _extract_markdown_table_from_segment(segment)
        if parsed is None:
            before_segments.append(segment)
            continue
        if match is not None:
            return None
        before, rows, after = parsed
        remaining = "".join(segments[segment_index + 1 :])
        match = ("".join(before_segments) + before, rows, after + remaining)
        before_segments.append(segment)
    return match


def _extract_markdown_table_from_segment(
    text: str,
) -> tuple[str, list[list[str]], str] | None:
    lines = text.splitlines(keepends=True)
    for index in range(len(lines)):
        if not _is_markdown_table_start(lines, index):
            continue
        table_lines = [lines[index], lines[index + 1]]
        cursor = index + 2
        while cursor < len(lines) and _is_markdown_table_row(lines[cursor]):
            table_lines.append(lines[cursor])
            cursor += 1
        rows = [_split_markdown_table_row(line) for line in table_lines[:1] + table_lines[2:]]
        return "".join(lines[:index]), rows, "".join(lines[cursor:])
    return None


def _markdown_section_blocks(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if not cleaned:
        return []
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": normalize_slack_mrkdwn(chunk)}}
        for chunk in _slack_text_chunks(cleaned)
    ]


def _slack_text_chunks(text: str, limit: int = 2800) -> list[str]:
    chunks: list[str] = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


def _table_cell(text: str, *, bold: bool = False) -> dict[str, Any]:
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_section",
                "elements": _rich_text_elements_from_table_cell(text, bold=bold),
            }
        ],
    }


def _rich_text_elements_from_table_cell(text: str, *, bold: bool = False) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for part in re.split(r"(`[^`]*`)", text.strip()):
        if not part:
            continue
        style: dict[str, bool] = {}
        value = part
        if part.startswith("`") and part.endswith("`"):
            value = part[1:-1]
            style["code"] = True
        else:
            value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
        if bold:
            style["bold"] = True
        element: dict[str, Any] = {"type": "text", "text": value}
        if style:
            element["style"] = style
        elements.append(element)
    return elements or [{"type": "text", "text": ""}]


def _wrap_markdown_tables(text: str) -> str:
    segments = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    return "".join(
        segment if segment.startswith("```") else _wrap_markdown_tables_in_segment(segment)
        for segment in segments
    )


def _wrap_markdown_tables_in_segment(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    index = 0
    while index < len(lines):
        if _is_markdown_table_start(lines, index):
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and _is_markdown_table_row(lines[index]):
                table_lines.append(lines[index])
                index += 1
            prefix = "" if not output or output[-1].endswith("\n") else "\n"
            suffix = "" if index >= len(lines) or lines[index].startswith("\n") else "\n"
            output.append(f"{prefix}```\n{''.join(table_lines).rstrip()}\n```{suffix}")
            continue
        output.append(lines[index])
        index += 1
    return "".join(output)


def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and _is_markdown_table_row(lines[index])
        and _is_markdown_table_separator(lines[index + 1])
    )


def _is_markdown_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.count("|") >= 2 and not stripped.startswith(">")


def _is_markdown_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not _is_markdown_table_row(stripped):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if len(cells) < 2:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _split_markdown_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def encode_action_value(action: str, **payload: Any) -> str:
    return json.dumps({"v": 1, "action": action, **payload}, separators=(",", ":"), sort_keys=True)


def decode_action_value(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict) or decoded.get("v") != 1:
        raise ValueError("unsupported Slack action value")
    action = decoded.get("action")
    if not isinstance(action, str) or not action:
        raise ValueError("Slack action value is missing action")
    return decoded


def dangerous_flag(provider: Provider | str) -> str:
    provider_value = provider if isinstance(provider, Provider) else Provider(provider)
    if provider_value == Provider.CODEX:
        return "--dangerously-bypass-approvals-and-sandbox"
    if provider_value == Provider.CLAUDE:
        return "--dangerously-skip-permissions"
    raise ValueError(f"unsupported provider: {provider_value}")


def parse_thread_ref(
    text: str,
    current_channel: str | None = None,
    current_thread_ts: str | None = None,
) -> SlackThreadRef | None:
    match = SLACK_PERMALINK_RE.search(text)
    if match:
        channel_id = match.group("channel")
        message_ts = unpack_slack_permalink_ts(match.group("packed_ts"))
        query = match.group("query") or ""
        thread_ts = _query_value(query, "thread_ts") or message_ts
        return SlackThreadRef(
            channel_id=channel_id,
            thread_ts=thread_ts,
            message_ts=message_ts,
            permalink=match.group(0),
        )
    if current_channel and current_thread_ts:
        return SlackThreadRef(channel_id=current_channel, thread_ts=current_thread_ts)
    return None


def parse_agent_handles(text: str) -> list[str]:
    return parse_lightweight_handles(text)


def is_dependency_intent(text: str) -> bool:
    normalized = text.lower()
    phrases = [
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
    ]
    return any(phrase in normalized for phrase in phrases)


def unpack_slack_permalink_ts(value: str) -> str:
    if len(value) != 16 or not value.isdigit():
        raise ValueError(f"invalid Slack permalink timestamp: {value}")
    return f"{value[:10]}.{value[10:]}"


def pack_slack_ts(value: str) -> str:
    if "." not in value:
        raise ValueError(f"invalid Slack timestamp: {value}")
    seconds, micros = value.split(".", 1)
    return f"{seconds}{micros[:6].ljust(6, '0')}"


def _query_value(query: str, name: str) -> str | None:
    for part in query.split("&"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key == name:
            return value
    return None


def _normalize_bold_markers(text: str) -> str:
    return re.sub(r"\*\*([^*\n][^*]*?)\*\*", r"*\1*", text)


def _button(
    text: str,
    action_id: str,
    value: str,
    style: str | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    button: dict[str, Any] = {
        "type": "button",
        "text": {"type": "plain_text", "text": text},
        "action_id": action_id,
        "value": value,
    }
    if style:
        button["style"] = style
    if url:
        button["url"] = url
    return button


def _default_repo_root() -> Path:
    cwd = Path.cwd()
    try:
        return cwd.parents[1]
    except IndexError:
        return cwd


def _option(text: str, value: str, description: str | None = None) -> dict[str, Any]:
    option: dict[str, Any] = {
        "text": {"type": "plain_text", "text": text},
        "value": value,
    }
    if description:
        option["description"] = {"type": "plain_text", "text": description[:75]}
    return option
