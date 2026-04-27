from __future__ import annotations

import json
import re
from typing import Any

from agent_harness.models import AgentTask, Provider, SlackThreadRef, TeamAgent
from agent_harness.routing import parse_lightweight_handles
from agent_harness.team import DEFAULT_CLAUDE_TEAM_SIZE, DEFAULT_CODEX_TEAM_SIZE

SLACK_PERMALINK_RE = re.compile(
    r"https://(?P<workspace>[^/]+)/archives/(?P<channel>[A-Z0-9]+)/p(?P<packed_ts>\d{16})"
    r"(?:\?(?P<query>[^>\s|]+))?"
)


def build_setup_modal(callback_id: str = "setup.initial") -> dict[str, Any]:
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


def build_team_roster_blocks(agents: list[TeamAgent]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Agent team*  {len(agents)} active lightweight handles",
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
        provider = agent.provider_preference.value if agent.provider_preference else "unmapped"
        blocks.append(
            {
                "type": "section",
                "block_id": f"team.agent.{agent.agent_id}",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{agent.full_name}* `@{agent.handle}`\n"
                        f"{agent.role} - {provider} - {agent.voice}"
                    ),
                },
                "accessory": _button(
                    "Fire",
                    "team.fire",
                    encode_action_value("team.fire", agent_id=agent.agent_id, handle=agent.handle),
                    "danger",
                ),
            }
        )
    return blocks


def build_task_thread_blocks(task: AgentTask, agent: TeamAgent) -> list[dict[str, Any]]:
    task_label = "PR review" if task.kind.value == "review" else "task"
    return [
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
        {
            "type": "actions",
            "block_id": f"task.actions.{task.task_id}",
            "elements": [
                _button(
                    "Mark Done",
                    "task.done",
                    encode_action_value("task.done", task_id=task.task_id),
                    "primary",
                ),
                _button(
                    "Pause",
                    "task.pause",
                    encode_action_value("task.pause", task_id=task.task_id),
                ),
                _button(
                    "Cancel",
                    "task.cancel",
                    encode_action_value("task.cancel", task_id=task.task_id),
                    "danger",
                ),
            ],
        },
    ]


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


def _button(
    text: str,
    action_id: str,
    value: str,
    style: str | None = None,
) -> dict[str, Any]:
    button: dict[str, Any] = {
        "type": "button",
        "text": {"type": "plain_text", "text": text},
        "action_id": action_id,
        "value": value,
    }
    if style:
        button["style"] = style
    return button


def _option(text: str, value: str, description: str | None = None) -> dict[str, Any]:
    option: dict[str, Any] = {
        "text": {"type": "plain_text", "text": text},
        "value": value,
    }
    if description:
        option["description"] = {"type": "plain_text", "text": description[:75]}
    return option
