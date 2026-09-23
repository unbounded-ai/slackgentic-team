from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_harness.models import (
    ASSIGNMENT_PROMPT_METADATA_KEY,
    DANGEROUS_MODE_METADATA_KEY,
    ORIGINAL_TASK_METADATA_KEY,
    ROSTER_SUMMARY_METADATA_KEY,
    AgentSession,
    AgentTask,
    Loop,
    LoopStatus,
    LoopVisibility,
    PermissionMode,
    Provider,
    SlackThreadRef,
    TeamAgent,
    TeamAgentKind,
)
from agent_harness.pr_links import pr_urls_from_metadata, slack_pr_links
from agent_harness.providers.usage import (
    AccountStatus,
    account_heading,
    account_tokens_line,
)
from agent_harness.team import (
    DEFAULT_CLAUDE_TEAM_SIZE,
    DEFAULT_CODEX_TEAM_SIZE,
    provider_logo_url,
)
from agent_harness.team.routing import parse_lightweight_handles
from agent_harness.updates import UpdateCandidate

if TYPE_CHECKING:
    from agent_harness.loops import LoopSpec

SLACK_PERMALINK_RE = re.compile(
    r"https://(?P<workspace>[^/]+)/archives/(?P<channel>[A-Z0-9]+)/p(?P<packed_ts>\d{16})"
    r"(?:\?(?P<query>[^>\s|]+))?"
)
SLACK_USER_REF_RE = re.compile(
    r"<@(?P<bracketed>[UW][A-Z0-9]{8,})>|"
    r"(?<![\w.-])@(?P<raw>[UW][A-Z0-9]{8,})(?![\w.-])"
)
SLACK_USER_AUTHOR_RE = re.compile(r"(?m)^(?P<user_id>[UW][A-Z0-9]{8,})(?=:)")

# Slack rejects a message whose blocks array holds more than 50 items with
# "invalid_blocks: no more than 50 items allowed". chat.update then falls back to
# a text-only render, which silently strips every button off the message.
SLACK_MAX_MESSAGE_BLOCKS = 50
ROSTER_AGENT_BLOCK_COUNT = 3

# chat.update rejects a longer text with "msg_too_long". The limit is far lower
# than chat.postMessage's, and Slack measures its own normalized form of the
# text: unicode emoji become :shortcode: and & < > become entities, so a glyph
# can cost more than 20 characters.
SLACK_MAX_UPDATE_TEXT_CHARS = 4000
# We cannot reproduce Slack's accounting exactly, and being a few characters over
# means a permanent rejection, so aim well short of the hard limit.
SLACK_UPDATE_TEXT_SAFETY_MARGIN = 400


@dataclass(frozen=True)
class AgentRosterStatus:
    label: str
    detail: str | None = None
    dangerous_mode: bool = False
    pr_urls: tuple[str, ...] = ()
    thread_url: str | None = None
    task_id: str | None = None
    session_provider: Provider | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class UnassignedExternalSessionListItem:
    session: AgentSession
    summary: str | None = None
    assignable_agents: tuple[TeamAgent, ...] = ()
    thread_url: str | None = None


def build_loop_create_guide_blocks(
    *,
    anchor_thread_ts: str | None = None,
) -> list[dict[str, Any]]:
    action_payload: dict[str, str] = {}
    if anchor_thread_ts:
        action_payload["anchor_thread_ts"] = anchor_thread_ts
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Create a loop"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "A loop runs a recurring task in its own Slack channel. "
                    "You do not need to create the channel yourself."
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*1.* Describe the task and schedule\n"
                    "*2.* Review the generated mission, channel, bot, and schedule\n"
                    "*3.* Click *Create loop* on the preview"
                ),
            },
        },
        {
            "type": "actions",
            "block_id": "loop.create.actions",
            "elements": [
                _button(
                    "Create a loop",
                    "loop.create.open",
                    encode_action_value("loop.create.open", **action_payload),
                    "primary",
                )
            ],
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        "Prefer text? Use `loop create <task and schedule>` "
                        "in the main agent channel."
                    ),
                }
            ],
        },
    ]


def build_loop_create_modal(
    *,
    channel_id: str,
    anchor_thread_ts: str | None = None,
    guide_message_ts: str | None = None,
) -> dict[str, Any]:
    metadata = {"channel_id": channel_id}
    if anchor_thread_ts:
        metadata["anchor_thread_ts"] = anchor_thread_ts
    if guide_message_ts:
        metadata["guide_message_ts"] = guide_message_ts
    automatic = _option("Automatic", "automatic", "Use the configured default provider")
    private = _option("Private", "private")
    every_run = _option("Post every run", "every-run", "Each run posts its report card")
    quiet = _option(
        "Only when attention is needed",
        "quiet",
        "All-clear runs stay silent and only update the pinned panel",
    )
    return {
        "type": "modal",
        "callback_id": "loop.create",
        "private_metadata": json.dumps(metadata, separators=(",", ":"), sort_keys=True),
        "title": {"type": "plain_text", "text": "Create a loop"},
        "submit": {"type": "plain_text", "text": "Preview loop"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "input",
                "block_id": "loop_mission",
                "label": {"type": "plain_text", "text": "What should the loop do?"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "multiline": True,
                    "max_length": 2800,
                    "placeholder": {
                        "type": "plain_text",
                        "text": "Check cloud costs and report material anomalies",
                    },
                },
            },
            {
                "type": "input",
                "block_id": "loop_schedule",
                "label": {"type": "plain_text", "text": "When should it run?"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "max_length": 300,
                    "placeholder": {
                        "type": "plain_text",
                        "text": "Every weekday at 9am PT",
                    },
                },
            },
            {
                "type": "input",
                "block_id": "loop_visibility",
                "label": {"type": "plain_text", "text": "Channel visibility"},
                "element": {
                    "type": "radio_buttons",
                    "action_id": "value",
                    "initial_option": private,
                    "options": [private, _option("Public", "public")],
                },
            },
            {
                "type": "input",
                "block_id": "loop_notify",
                "label": {"type": "plain_text", "text": "When should it post?"},
                "element": {
                    "type": "radio_buttons",
                    "action_id": "value",
                    "initial_option": every_run,
                    "options": [every_run, quiet],
                },
            },
            {
                "type": "input",
                "block_id": "loop_provider",
                "label": {"type": "plain_text", "text": "Provider"},
                "element": {
                    "type": "static_select",
                    "action_id": "value",
                    "initial_option": automatic,
                    "options": [
                        automatic,
                        _option("Codex", Provider.CODEX.value),
                        _option("Claude", Provider.CLAUDE.value),
                    ],
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            "Next, Slackgentic generates a preview. The channel is only "
                            "created after you approve it."
                        ),
                    }
                ],
            },
        ],
    }


def build_loop_preview_blocks(
    loop: Loop,
    spec: LoopSpec,
    *,
    next_run_text: str,
    include_actions: bool = True,
    footer: str | None = None,
) -> list[dict[str, Any]]:
    visibility = loop.visibility.value
    provider = loop.provider.value
    model = f" · Model: `{loop.model}`" if loop.model else ""
    cwd = f"\n• Working directory: `{loop.cwd}`" if loop.cwd else ""
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{spec.bot_name} — ready to create"[:150],
            },
        },
        {
            "type": "section",
            "block_id": f"loop.preview.details.{loop.loop_id}",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"• Channel: `#{spec.channel_name}` ({visibility}) · Provider: {provider}{model}\n"
                    f"• Schedule: {spec.schedule_description} (next run {next_run_text})\n"
                    f"• Permissions: {loop.permission_mode.value}{cwd}\n"
                    f"• Icon: :{spec.icon.emoji}:"
                    + ("\n• 🔕 Quiet: posts only when a run needs attention" if spec.quiet else "")
                ),
            },
        },
        {
            "type": "section",
            "block_id": f"loop.preview.mission.{loop.loop_id}",
            "text": {
                "type": "mrkdwn",
                "text": f"*Mission*\n{spec.mission[:2600]}",
            },
        },
        {
            "type": "context",
            "block_id": f"loop.preview.edits.{loop.loop_id}",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        "Reply to adjust: `name: …` · `icon: :emoji:|https://…` · "
                        "`channel: #…` · `visibility: public|private` · `schedule: …` · "
                        "`task: …` · `cwd: …` · `permissions: …` · `provider: …` · "
                        "`model: …`"
                    ),
                }
            ],
        },
    ]
    if footer:
        blocks.append(
            {
                "type": "context",
                "block_id": f"loop.preview.footer.{loop.loop_id}",
                "elements": [{"type": "mrkdwn", "text": footer[:2900]}],
            }
        )
    if include_actions:
        blocks.append(
            {
                "type": "actions",
                "block_id": f"loop.preview.actions.{loop.loop_id}",
                "elements": [
                    _button(
                        "Create loop",
                        "loop.approve",
                        encode_action_value("loop.approve", loop_id=loop.loop_id),
                        "primary",
                    ),
                    _button(
                        "Cancel",
                        "loop.cancel",
                        encode_action_value("loop.cancel", loop_id=loop.loop_id),
                        "danger",
                    ),
                ],
            }
        )
    return blocks


LOOP_RESULT_STYLES: dict[str, tuple[str, str]] = {
    "ok": ("✅", "All clear"),
    "found_issue": ("⚠️", "Needs attention"),
    "action_taken": ("🛠️", "Action taken"),
    "failed": ("❌", "Mission failed"),
}
LOOP_RUN_ERROR_EMOJI = "🚫"
LOOP_RUN_SKIPPED_EMOJI = "⏭️"
LOOP_RUN_RUNNING_EMOJI = "⏳"
LOOP_CAROUSEL_MAX_CARDS = 10
_LOOP_MARKDOWN_BLOCK_LIMIT = 11_000


@dataclass(frozen=True)
class LoopRunCardMetric:
    label: str
    value: str
    delta: str | None = None


@dataclass(frozen=True)
class LoopRunCardChart:
    type: str
    title: str
    categories: tuple[str, ...]
    series: tuple[tuple[str, tuple[float, ...]], ...]


@dataclass(frozen=True)
class LoopRunChip:
    """One run in a loop's recent-history strip."""

    run_number: int
    emoji: str
    url: str | None = None


def loop_result_style(status: str | None) -> tuple[str, str]:
    return LOOP_RESULT_STYLES.get(status or "ok", LOOP_RESULT_STYLES["ok"])


def build_loop_run_running_blocks(
    *,
    title: str,
    run_number: int,
    when_text: str,
    previous_headline: str | None = None,
    progress: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    text = f"{LOOP_RUN_RUNNING_EMOJI} {title} · run #{run_number} is working…"
    context = f"Run #{run_number} · {when_text} · progress notes in the thread"
    if previous_headline:
        context += f" · previously: {_mrkdwn_escape(previous_headline)}"
    blocks: list[dict[str, Any]] = [
        {
            "type": "task_card",
            "task_id": f"run-{run_number}",
            "title": f"{title} — working on it"[:150],
            "status": "in_progress",
            **({"details": _rich_text(progress)} if progress else {}),
        },
        {"type": "context", "elements": [{"type": "mrkdwn", "text": context[:2900]}]},
    ]
    return text, blocks


def build_loop_run_report_blocks(
    *,
    title: str,
    run_number: int,
    when_text: str,
    status: str,
    headline: str,
    report: str | None,
    metrics: tuple[LoopRunCardMetric, ...] | list[LoopRunCardMetric] = (),
    chart: LoopRunCardChart | None = None,
    duration_text: str | None = None,
    guard_note: str | None = None,
    feedback_value: dict[str, str] | None = None,
    notes_in_thread: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Render a finished run as a report-first card for the run's parent message."""
    emoji, label = loop_result_style(status)
    headline_text = _mrkdwn_escape(" ".join(headline.split()))[:300]
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"{emoji} *{headline_text}*"},
        }
    ]
    if metrics:
        blocks.append(
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": _loop_metric_field(metric)}
                    for metric in list(metrics)[:10]
                ],
            }
        )
    if chart is not None:
        blocks.append(_loop_chart_block(chart))
    if report and report.strip():
        blocks.append({"type": "markdown", "text": report.strip()[:_LOOP_MARKDOWN_BLOCK_LIMIT]})
    footer = [f"*{label}*", f"{_mrkdwn_escape(title)} · run #{run_number}", when_text]
    if duration_text:
        footer.append(f"took {duration_text}")
    if guard_note:
        footer.append(guard_note)
    footer.append("🧵 working notes in thread" if notes_in_thread else "🔕 quiet loop")
    blocks.append(
        {"type": "context", "elements": [{"type": "mrkdwn", "text": " · ".join(footer)[:2900]}]}
    )
    if feedback_value is not None:
        blocks.append(_loop_feedback_block(feedback_value))
    text = f"{emoji} {title}: {' '.join(headline.split())}"
    return text[:1000], blocks


def build_loop_run_error_blocks(
    *,
    title: str,
    run_number: int,
    when_text: str,
    error: str,
    paused: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    detail = " ".join(error.split())[:1500]
    if paused:
        detail += " The loop is paused after repeated failures; use Resume on the pinned card."
    blocks: list[dict[str, Any]] = [
        {
            "type": "task_card",
            "task_id": f"run-{run_number}",
            "title": f"Run #{run_number} did not finish",
            "status": "error",
            "output": _rich_text(detail),
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"{_mrkdwn_escape(title)} · run #{run_number} · {when_text}",
                }
            ],
        },
    ]
    return f"{LOOP_RUN_ERROR_EMOJI} {title}: run #{run_number} did not finish", blocks


def build_loop_panel_blocks(
    loop: Loop,
    *,
    bot_name: str,
    icon_emoji: str | None,
    schedule_text: str,
    next_run_text: str,
    recent_runs: list[LoopRunChip] | tuple[LoopRunChip, ...] = (),
    latest_headline: str | None = None,
    latest_url: str | None = None,
    running: bool = False,
    remembered_approvals: int = 0,
    context: str = "panel",
    quiet: bool = False,
    last_check_text: str | None = None,
) -> list[dict[str, Any]]:
    """The loop's control panel: pinned in its channel and reused by `loop status`."""
    icon = f"{icon_emoji} " if icon_emoji else ""
    state_text = _loop_state_text(loop, running=running)
    subtitle = f"{state_text} · {schedule_text}"
    if quiet:
        subtitle += " · 🔕 quiet"
    if loop.status == LoopStatus.ACTIVE and not running:
        subtitle += f" · next {next_run_text}"
    body = (
        f"Latest: {_mrkdwn_escape(latest_headline)}"
        if latest_headline
        else "No runs yet — the first report lands here."
    )
    card: dict[str, Any] = {
        "type": "card",
        "block_id": f"loop.{context}.card.{loop.loop_id}"[:255],
        "title": {"type": "mrkdwn", "text": f"{icon}*{_mrkdwn_escape(loop.title)}*"[:150]},
        "subtitle": {"type": "mrkdwn", "text": subtitle[:150]},
        "body": {"type": "mrkdwn", "text": _shorten_text(body, 200)},
    }
    buttons = _loop_panel_buttons(loop, running=running)
    if buttons:
        card["actions"] = buttons
    # The panel is a glanceable header; the full mission lives in Edit.
    mission = _shorten_text(loop.mission.strip(), 280)
    mission_section: dict[str, Any] = {
        "type": "section",
        "block_id": f"loop.{context}.mission.{loop.loop_id}"[:255],
        "text": {"type": "mrkdwn", "text": _quote_mrkdwn(mission)},
    }
    overflow = _loop_overflow(
        loop, include_primary=False, remembered_approvals=remembered_approvals
    )
    if overflow is not None:
        mission_section["accessory"] = overflow
    blocks: list[dict[str, Any]] = [card, mission_section]
    history: list[str] = []
    if recent_runs:
        chips = " ".join(
            f"<{chip.url}|{chip.emoji} #{chip.run_number}>"
            if chip.url
            else f"{chip.emoji} #{chip.run_number}"
            for chip in recent_runs
        )
        history.append(f"*Recent runs*  {chips}")
    if quiet:
        quiet_line = "🔕 *Quiet* — posts only when a run needs attention"
        if last_check_text:
            quiet_line += f" · {last_check_text}"
        history.append(quiet_line)
    if latest_url and latest_headline:
        history.append(f"<{latest_url}|Open the latest report →>")
    if history:
        blocks.append(
            {
                "type": "section",
                "block_id": f"loop.{context}.history.{loop.loop_id}"[:255],
                "text": {"type": "mrkdwn", "text": "\n".join(history)[:2900]},
            }
        )
    permissions = loop.permission_mode.value
    if loop.permission_mode == PermissionMode.READ_ONLY:
        permissions = "🛡️ read-only"
    if remembered_approvals:
        suffix = "s" if remembered_approvals != 1 else ""
        permissions += f" · {remembered_approvals} remembered approval{suffix}"
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"{_mrkdwn_escape(bot_name)} · {loop.provider.value} · {permissions} · "
                        f"owner <@{loop.owner_slack_user_id}>. Only the owner can instruct this "
                        "bot; other messages are never shown to it. Reply in a run's thread to "
                        "follow up, or post in the channel to leave a standing note."
                    )[:2900],
                }
            ],
        }
    )
    return blocks


def build_loop_list_blocks(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Every loop as a card in a carousel (ten cards per carousel).

    Each row carries ``loop``, ``icon_emoji``, ``channel_id``, ``channel_text``,
    ``next_run_text``, ``schedule_text``, ``recent_runs``, ``running`` and
    ``latest_headline``.
    """
    active = sum(1 for row in rows if row["loop"].status == LoopStatus.ACTIVE)
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🔁 Loops", "emoji": True},
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"{active} active · {len(rows)} total"
                        if rows
                        else "No loops yet. Create one to get started."
                    ),
                }
            ],
        },
    ]
    cards = [_loop_new_card(), *(_loop_list_card(row) for row in rows)]
    for start in range(0, len(cards), LOOP_CAROUSEL_MAX_CARDS):
        chunk = cards[start : start + LOOP_CAROUSEL_MAX_CARDS]
        blocks.append(
            {
                "type": "carousel",
                "block_id": f"loop.list.carousel.{start // LOOP_CAROUSEL_MAX_CARDS}",
                "elements": chunk,
            }
        )
    return blocks[:SLACK_MAX_MESSAGE_BLOCKS]


def _loop_new_card() -> dict[str, Any]:
    return {
        "type": "card",
        "block_id": "loop.list.new",
        "title": {"type": "mrkdwn", "text": ":heavy_plus_sign: *New loop*"},
        "body": {"type": "mrkdwn", "text": "A recurring task with its own channel."},
        "actions": [
            _button(
                "Create",
                "loop.create.open",
                encode_action_value("loop.create.open", source="list"),
                "primary",
            )
        ],
    }


def _loop_list_card(row: dict[str, Any]) -> dict[str, Any]:
    loop: Loop = row["loop"]
    icon = f"{row['icon_emoji']} " if row.get("icon_emoji") else ""
    chips = "".join(chip.emoji for chip in row.get("recent_runs") or ())
    latest = row.get("latest_headline")
    body_parts = [
        part
        for part in (row.get("channel_text"), chips, _mrkdwn_escape(latest) if latest else None)
        if part
    ]
    state = _loop_state_text(loop, running=bool(row.get("running")))
    if row.get("quiet"):
        state += " · 🔕"
    card: dict[str, Any] = {
        "type": "card",
        "block_id": f"loop.list.item.{loop.loop_id}"[:255],
        "title": {"type": "mrkdwn", "text": f"{icon}*{_mrkdwn_escape(loop.title)}*"[:150]},
        "subtitle": {
            "type": "mrkdwn",
            "text": (
                f"{state} · next {row['next_run_text']}"
                if loop.status == LoopStatus.ACTIVE
                else f"{state} · {row['schedule_text']}"
            )[:150],
        },
        "body": {
            "type": "mrkdwn",
            "text": _shorten_text(" ".join(body_parts) or row["schedule_text"], 200),
        },
    }
    # Cards hold at most three buttons: Pause/Resume, Edit, and Delete. The channel
    # name in the body opens the channel; Run now lives on the pinned panel.
    actions: list[dict[str, Any]] = []
    if loop.status == LoopStatus.ACTIVE:
        actions.append(
            _button(
                "⏸ Pause", "loop.pause", encode_action_value("loop.pause", loop_id=loop.loop_id)
            )
        )
    elif loop.status == LoopStatus.PAUSED:
        actions.append(
            _button(
                "▶ Resume",
                "loop.resume",
                encode_action_value("loop.resume", loop_id=loop.loop_id),
                "primary",
            )
        )
    actions.append(
        _button(
            "✏️ Edit",
            "loop.edit.open",
            encode_action_value("loop.edit.open", loop_id=loop.loop_id),
        )
    )
    actions.append(_loop_delete_button(loop))
    if actions:
        card["actions"] = actions[:3]
    return card


def _loop_delete_button(loop: Loop) -> dict[str, Any]:
    return _button(
        "🗑 Delete",
        "loop.delete",
        encode_action_value("loop.delete", loop_id=loop.loop_id),
        "danger",
        confirm=_loop_delete_confirm(loop),
    )


def _loop_delete_confirm(loop: Loop) -> dict[str, Any]:
    channel = f" and archives <#{loop.channel_id}>" if loop.channel_id else ""
    return {
        "title": {"type": "plain_text", "text": "Delete this loop?"},
        "text": {
            "type": "mrkdwn",
            "text": f"This stops *{_mrkdwn_escape(loop.title)}*{channel}. It cannot be undone."[
                :300
            ],
        },
        "confirm": {"type": "plain_text", "text": "Delete loop"},
        "deny": {"type": "plain_text", "text": "Keep it"},
        "style": "danger",
    }


def build_loop_delete_confirmation_blocks(loop: Loop) -> list[dict[str, Any]]:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"Delete *{_mrkdwn_escape(loop.title)}*? This stops the loop and archives "
                    "its channel. It cannot be undone."
                ),
            },
        },
        {
            "type": "actions",
            "block_id": f"loop.delete.confirm.{loop.loop_id}"[:255],
            "elements": [
                _button(
                    "Delete loop",
                    "loop.delete",
                    encode_action_value("loop.delete", loop_id=loop.loop_id),
                    "danger",
                ),
                _button(
                    "Keep it",
                    "loop.stop.dismiss",
                    encode_action_value("loop.stop.dismiss", loop_id=loop.loop_id),
                ),
            ],
        },
    ]


def build_loop_edit_modal(
    loop: Loop,
    *,
    schedule_text: str,
    channel_id: str,
    message_ts: str | None = None,
) -> dict[str, Any]:
    metadata = {"loop_id": loop.loop_id, "channel_id": channel_id}
    if message_ts:
        metadata["message_ts"] = message_ts
    modes = [
        _option("Read-only (recommended)", "read-only", "Never asks; blocks anything that writes"),
        _option("Safe auto", "safe-auto", "Edits and read-only commands run without asking"),
        _option("Locked", "locked", "Every tool call needs approval"),
        _option("Dangerous", "dangerous", "Bypass approvals and sandbox (asks to confirm)"),
    ]
    current_mode = next(
        (option for option in modes if option["value"] == loop.permission_mode.value), modes[0]
    )
    quiet = loop.metadata.get("quiet") is True
    quiet_option = _option(
        "Only post when a run needs attention",
        "quiet",
        "All-clear runs stay silent and only update the pinned panel",
    )
    cwd_element: dict[str, Any] = {
        "type": "plain_text_input",
        "action_id": "value",
        "placeholder": {"type": "plain_text", "text": "/workspace/repos/example-project"},
    }
    if loop.cwd:
        cwd_element["initial_value"] = loop.cwd
    private = _option("Private", LoopVisibility.PRIVATE.value)
    public = _option("Public", LoopVisibility.PUBLIC.value)
    return {
        "type": "modal",
        "callback_id": "loop.edit",
        "private_metadata": json.dumps(metadata, separators=(",", ":"), sort_keys=True),
        "title": {"type": "plain_text", "text": "Edit loop"},
        "submit": {"type": "plain_text", "text": "Save"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "alert",
                "level": "info",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "Loops run unattended. *Read-only* never waits for approval: reads run "
                        "automatically and anything that could change state is blocked."
                    ),
                },
            },
            {
                "type": "input",
                "block_id": "loop_mission",
                "label": {"type": "plain_text", "text": "Mission"},
                "hint": {
                    "type": "plain_text",
                    "text": "Edit freely; the loop agent rewrites it into a standing runbook.",
                },
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "multiline": True,
                    "max_length": 2900,
                    "initial_value": loop.mission[:2900],
                },
            },
            {
                "type": "input",
                "block_id": "loop_schedule",
                "label": {"type": "plain_text", "text": "Schedule"},
                "hint": {"type": "plain_text", "text": "Plain language, e.g. weekdays at 9am PT"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "max_length": 300,
                    "initial_value": schedule_text[:300],
                },
            },
            {
                "type": "input",
                "block_id": "loop_permissions",
                "label": {"type": "plain_text", "text": "Permissions"},
                "element": {
                    "type": "static_select",
                    "action_id": "value",
                    "initial_option": current_mode,
                    "options": modes,
                },
            },
            {
                "type": "input",
                "block_id": "loop_quiet",
                "optional": True,
                "label": {"type": "plain_text", "text": "Notifications"},
                "element": {
                    "type": "checkboxes",
                    "action_id": "value",
                    "options": [quiet_option],
                    **({"initial_options": [quiet_option]} if quiet else {}),
                },
            },
            {
                "type": "input",
                "block_id": "loop_cwd",
                "optional": True,
                "label": {"type": "plain_text", "text": "Reference directory"},
                "hint": {
                    "type": "plain_text",
                    "text": "The repo the loop reads. Read-only loops never change it.",
                },
                "element": cwd_element,
            },
            {
                "type": "input",
                "block_id": "loop_visibility",
                "label": {"type": "plain_text", "text": "Channel visibility"},
                "hint": {
                    "type": "plain_text",
                    "text": (
                        "⚠️ Changing this recreates the channel: a new channel with the same "
                        "name and members takes over the loop, and this one is archived with "
                        "its run history."
                    ),
                },
                "element": {
                    "type": "radio_buttons",
                    "action_id": "value",
                    "initial_option": (
                        public if loop.visibility == LoopVisibility.PUBLIC else private
                    ),
                    "options": [private, public],
                },
            },
        ],
    }


def downgrade_modern_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite newer Block Kit blocks into classic ones for clients or workspaces
    that reject them, keeping every button that still makes sense."""
    downgraded: list[dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "card":
            downgraded.extend(_downgrade_card(block))
        elif block_type == "carousel":
            for card in block.get("elements") or []:
                downgraded.extend(_downgrade_card(card))
        elif block_type == "markdown":
            text = normalize_slack_mrkdwn(str(block.get("text") or ""))
            for chunk in _split_mrkdwn(text, limit=2900)[:6]:
                downgraded.append({"type": "section", "text": {"type": "mrkdwn", "text": chunk}})
        elif block_type == "task_card":
            status = block.get("status")
            emoji = {"complete": "✅", "error": LOOP_RUN_ERROR_EMOJI}.get(
                str(status), LOOP_RUN_RUNNING_EMOJI
            )
            lines = [f"{emoji} *{_mrkdwn_escape(str(block.get('title') or ''))}*"]
            output = _rich_text_plain(block.get("output"))
            if output:
                lines.append(f"_{_mrkdwn_escape(output)}_")
            downgraded.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)[:2900]}}
            )
        elif block_type == "alert":
            text = (block.get("text") or {}).get("text") or ""
            downgraded.append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f":information_source: {text}"[:2900]}],
                }
            )
        elif block_type in {"data_visualization", "context_actions"}:
            continue
        else:
            downgraded.append(block)
    return downgraded[:SLACK_MAX_MESSAGE_BLOCKS]


def has_modern_blocks(blocks: list[dict[str, Any]] | None) -> bool:
    # Exactly the block types downgrade_modern_blocks rewrites.
    modern = {"card", "carousel", "markdown", "task_card", "alert", "data_visualization"}
    return any(block.get("type") in modern | {"context_actions"} for block in blocks or [])


def _downgrade_card(card: dict[str, Any]) -> list[dict[str, Any]]:
    lines = [
        str((card.get(key) or {}).get("text") or "")
        for key in ("title", "subtitle", "body")
        if (card.get(key) or {}).get("text")
    ]
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            **({"block_id": card["block_id"]} if card.get("block_id") else {}),
            "text": {"type": "mrkdwn", "text": "\n".join(lines)[:2900] or " "},
        }
    ]
    actions = card.get("actions") or []
    if actions:
        blocks.append({"type": "actions", "elements": actions})
    return blocks


def _loop_chart_block(chart: LoopRunCardChart) -> dict[str, Any]:
    return {
        "type": "data_visualization",
        "title": chart.title[:50],
        "chart": {
            "type": chart.type,
            "series": [
                {
                    "name": name[:20],
                    "data": [
                        {"label": label, "value": value}
                        for label, value in zip(chart.categories, values, strict=False)
                    ],
                }
                for name, values in chart.series[:12]
            ],
            "axis_config": {"categories": list(chart.categories)},
        },
    }


def _loop_feedback_block(value: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "context_actions",
        "block_id": f"loop.feedback.{value.get('run_id', '')}"[:255],
        "elements": [
            {
                "type": "feedback_buttons",
                "action_id": "loop.feedback",
                "positive_button": {
                    "text": {"type": "plain_text", "text": "Useful"},
                    "value": encode_action_value("loop.feedback", rating="up", **value),
                },
                "negative_button": {
                    "text": {"type": "plain_text", "text": "Not useful"},
                    "value": encode_action_value("loop.feedback", rating="down", **value),
                },
            }
        ],
    }


def _rich_text(text: str) -> dict[str, Any]:
    return {
        "type": "rich_text",
        "elements": [
            {"type": "rich_text_section", "elements": [{"type": "text", "text": text or " "}]}
        ],
    }


def _rich_text_plain(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    parts: list[str] = []
    for element in value.get("elements") or []:
        for inner in element.get("elements") or []:
            if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                parts.append(inner["text"])
    return "".join(parts)


def _shorten_text(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _loop_state_text(loop: Loop, *, running: bool = False) -> str:
    if loop.status == LoopStatus.ACTIVE:
        return "🔄 Running now" if running else "🟢 Active"
    if loop.status == LoopStatus.PAUSED:
        return "⏸️ Paused"
    if loop.status == LoopStatus.CANCELLED:
        return "⏹️ Stopped"
    return loop.status.value.replace("_", " ").capitalize()


def _loop_panel_buttons(loop: Loop, *, running: bool) -> list[dict[str, Any]]:
    if loop.status not in {LoopStatus.ACTIVE, LoopStatus.PAUSED}:
        return []
    elements: list[dict[str, Any]] = []
    if loop.status == LoopStatus.ACTIVE and not running:
        elements.append(
            _button(
                "▶ Run now",
                "loop.run_now",
                encode_action_value("loop.run_now", loop_id=loop.loop_id),
                "primary",
            )
        )
    if loop.status == LoopStatus.ACTIVE:
        elements.append(
            _button(
                "⏸ Pause", "loop.pause", encode_action_value("loop.pause", loop_id=loop.loop_id)
            )
        )
    else:
        elements.append(
            _button(
                "▶ Resume",
                "loop.resume",
                encode_action_value("loop.resume", loop_id=loop.loop_id),
                "primary",
            )
        )
    elements.append(
        _button(
            "✏️ Edit", "loop.edit.open", encode_action_value("loop.edit.open", loop_id=loop.loop_id)
        )
    )
    return elements


def _loop_overflow(
    loop: Loop,
    *,
    include_primary: bool,
    remembered_approvals: int = 0,
) -> dict[str, Any] | None:
    if loop.status not in {LoopStatus.ACTIVE, LoopStatus.PAUSED}:
        return None
    options: list[dict[str, Any]] = []
    if include_primary:
        if loop.status == LoopStatus.ACTIVE:
            options.append(
                _option("Run now", encode_action_value("loop.run_now", loop_id=loop.loop_id))
            )
            options.append(
                _option("Pause", encode_action_value("loop.pause", loop_id=loop.loop_id))
            )
        else:
            options.append(
                _option("Resume", encode_action_value("loop.resume", loop_id=loop.loop_id))
            )
    else:
        options.append(
            _option("Compact memory", encode_action_value("loop.compact", loop_id=loop.loop_id))
        )
        if remembered_approvals:
            options.append(
                _option(
                    "Forget remembered approvals",
                    encode_action_value("loop.approvals.reset", loop_id=loop.loop_id),
                )
            )
    options.append(
        _option("Stop loop…", encode_action_value("loop.stop.request", loop_id=loop.loop_id))
    )
    options.append(
        _option("Delete loop…", encode_action_value("loop.delete.request", loop_id=loop.loop_id))
    )
    return {"type": "overflow", "action_id": "loop.more", "options": options}


def _loop_metric_field(metric: LoopRunCardMetric) -> str:
    text = f"*{_mrkdwn_escape(metric.label)}*\n{_mrkdwn_escape(metric.value)}"
    if metric.delta:
        text += f"  _{_mrkdwn_escape(metric.delta)}_"
    return text[:1900]


def _quote_mrkdwn(text: str) -> str:
    return "\n".join(f">{line}" if line.strip() else ">" for line in text.splitlines())[:2900]


def _mrkdwn_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _split_mrkdwn(text: str, *, limit: int) -> list[str]:
    """Split on paragraph, then line boundaries so each chunk fits one section."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(paragraph) > limit:
            cut = paragraph.rfind("\n", 0, limit)
            if cut <= 0:
                cut = limit
            chunks.append(paragraph[:cut].rstrip())
            paragraph = paragraph[cut:].lstrip("\n")
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


def build_loop_stop_confirmation_blocks(loop: Loop, *, archive: bool) -> list[dict[str, Any]]:
    archive_text = " and archive its channel" if archive else ""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Stop *{loop.title}*{archive_text}? This cannot be resumed.",
            },
        },
        {
            "type": "actions",
            "block_id": f"loop.stop.confirm.{loop.loop_id}",
            "elements": [
                _button(
                    "Stop loop",
                    "loop.stop.confirm",
                    encode_action_value(
                        "loop.stop.confirm",
                        loop_id=loop.loop_id,
                        archive=archive,
                    ),
                    "danger",
                ),
                _button(
                    "Keep running",
                    "loop.stop.dismiss",
                    encode_action_value("loop.stop.dismiss", loop_id=loop.loop_id),
                ),
            ],
        },
    ]


def build_loop_dangerous_confirmation_blocks(loop: Loop) -> list[dict[str, Any]]:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"Set *{loop.title}* to `dangerous` permissions? Future runs may bypass "
                    "provider approval and sandbox protections."
                ),
            },
        },
        {
            "type": "actions",
            "block_id": f"loop.permissions.confirm.{loop.loop_id}",
            "elements": [
                _button(
                    "Use dangerous mode",
                    "loop.permissions.confirm",
                    encode_action_value(
                        "loop.permissions.confirm",
                        loop_id=loop.loop_id,
                    ),
                    "danger",
                ),
                _button(
                    "Cancel",
                    "loop.permissions.dismiss",
                    encode_action_value("loop.permissions.dismiss", loop_id=loop.loop_id),
                ),
            ],
        },
    ]


PROVIDER_SORT_ORDER = {
    Provider.CODEX: 0,
    Provider.CLAUDE: 1,
}


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


ROSTER_STATUS_STYLES: dict[str, str] = {
    "Available": "🟢",
    "Working": "🔨",
    "Queued": "⏳",
    "Occupied": "👀",
}
ROSTER_HIRE_OPTIONS: tuple[tuple[str, str | None, str | None], ...] = (
    ("Auto (best available)", None, None),
    ("Codex engineer", Provider.CODEX.value, None),
    ("Claude engineer", Provider.CLAUDE.value, None),
    ("PM (Codex)", Provider.CODEX.value, TeamAgentKind.PM.value),
    ("PM (Claude)", Provider.CLAUDE.value, TeamAgentKind.PM.value),
)


def build_team_roster_blocks(
    agents: list[TeamAgent],
    statuses: dict[str, AgentRosterStatus] | None = None,
    *,
    icon_urls: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """The team roster: a summary, team actions, and a carousel of agent cards per group."""
    visible_agents = [agent for agent in agents if agent.kind != TeamAgentKind.LOOP]
    engineers = [agent for agent in visible_agents if not agent.is_pm]
    pms = [agent for agent in visible_agents if agent.is_pm]
    busy = [agent for agent in engineers if not _agent_accepts_new_work(_status(agent, statuses))]
    free = [agent for agent in engineers if _agent_accepts_new_work(_status(agent, statuses))]
    summary = (
        f"*🤖 Agent team* · {len(visible_agents)} "
        f"{'agent' if len(visible_agents) == 1 else 'agents'} · {len(busy)} working · "
        f"{len(free)} free"
    )
    breakdown = _provider_breakdown_text(visible_agents)
    if pms:
        breakdown += f" · {len(pms)} {'PM' if len(pms) == 1 else 'PMs'}"
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "block_id": "team.roster.summary",
            "text": {"type": "mrkdwn", "text": f"{summary}\n{breakdown}"},
        },
        {
            "type": "actions",
            "block_id": "team.roster.actions",
            "elements": [
                _button(
                    "📝 Add work",
                    "roster.work.assign",
                    encode_action_value("roster.work.open", mode="now"),
                    "primary",
                ),
                {
                    "type": "static_select",
                    "action_id": "team.hire.select",
                    "placeholder": {"type": "plain_text", "text": "Hire…"},
                    "options": [
                        _option(
                            label,
                            encode_action_value(
                                "team.hire",
                                count=1,
                                **({"provider": provider} if provider else {}),
                                **({"kind": kind} if kind else {}),
                            ),
                        )
                        for label, provider, kind in ROSTER_HIRE_OPTIONS
                    ],
                },
            ],
        },
    ]
    groups = (
        ("team.section.working", "🔨 *Working now*", busy),
        ("team.section.pms", "📋 *Program managers*", pms),
        ("team.section.available", "🟢 *Available*", free),
    )
    for block_id, heading, members in groups:
        if not members:
            continue
        ordered = _sorted_roster_agents(members, statuses)
        blocks.append(
            {
                "type": "context",
                "block_id": block_id,
                "elements": [{"type": "mrkdwn", "text": f"{heading} · {len(members)}"}],
            }
        )
        for start in range(0, len(ordered), 10):
            blocks.append(
                {
                    "type": "carousel",
                    "block_id": f"{block_id}.cards.{start // 10}",
                    "elements": [
                        _agent_roster_card(
                            agent,
                            _status(agent, statuses),
                            (icon_urls or {}).get(agent.agent_id),
                        )
                        for agent in ordered[start : start + 10]
                    ],
                }
            )
    return blocks[:SLACK_MAX_MESSAGE_BLOCKS]


def _status(agent: TeamAgent, statuses: dict[str, AgentRosterStatus] | None):
    return statuses.get(agent.agent_id) if statuses else None


def _agent_roster_card(
    agent: TeamAgent,
    status: AgentRosterStatus | None,
    icon_url: str | None = None,
) -> dict[str, Any]:
    label = status.label if status else "Available"
    subtitle = f"{ROSTER_STATUS_STYLES.get(label, '•')} {label}"
    if agent.provider_preference is not None:
        subtitle += f" · {agent.provider_preference.value}"
    if agent.is_pm:
        subtitle += " · PM"
    if status and status.dangerous_mode:
        subtitle += " · ⚡ dangerous"
    body = _roster_card_body(status)
    card: dict[str, Any] = {
        "type": "card",
        "block_id": f"team.agent.{agent.agent_id}"[:255],
        "title": {
            "type": "mrkdwn",
            "text": f"*{_mrkdwn_escape(agent.full_name)}* `@{agent.handle}`"[:150],
        },
        "subtitle": {"type": "mrkdwn", "text": subtitle[:150]},
        "body": {"type": "mrkdwn", "text": body},
    }
    if icon_url:
        card["icon"] = {"type": "image", "image_url": icon_url, "alt_text": agent.full_name}
    actions: list[dict[str, Any]] = []
    if status and status.task_id:
        actions.append(
            _button(
                "Free up",
                "task.done",
                encode_action_value("task.done", task_id=status.task_id),
                "primary",
            )
        )
    elif status and status.session_provider and status.session_id:
        actions.append(
            _button(
                "Detach",
                "external.session.detach",
                encode_action_value(
                    "external.session.detach",
                    provider=status.session_provider.value,
                    session_id=status.session_id,
                ),
            )
        )
    if status and status.thread_url:
        actions.append(
            _button(
                "Open thread",
                "thread.open",
                encode_action_value("thread.open"),
                url=status.thread_url,
            )
        )
    if _agent_accepts_new_work(status):
        actions.append(
            _button(
                "Assign Project" if agent.is_pm else "Assign",
                "roster.work.assign",
                encode_action_value(
                    "roster.work.open",
                    mode="now",
                    agent_id=agent.agent_id,
                    handle=agent.handle,
                ),
                "primary",
            )
        )
    actions.append(
        _button(
            "Fire",
            "team.fire",
            encode_action_value("team.fire", agent_id=agent.agent_id, handle=agent.handle),
            "danger",
        )
    )
    card["actions"] = actions[:3]
    return card


def _roster_card_body(status: AgentRosterStatus | None) -> str:
    if status is None or status.label == "Available":
        return "Ready for work."
    detail = " ".join((status.detail or status.label).split())
    detail = _mrkdwn_escape(detail)
    pr_text = ""
    if status.pr_urls:
        label = "PRs" if len(status.pr_urls) > 1 else "PR"
        pr_text = f"\n*{label}:* {slack_pr_links(status.pr_urls, limit=2)}"
    budget = 200 - len(pr_text)
    if budget < 60:
        pr_text = ""
        budget = 200
    return _shorten_text(detail, budget) + pr_text


def _sorted_roster_agents(
    agents: list[TeamAgent],
    statuses: dict[str, AgentRosterStatus] | None,
) -> list[TeamAgent]:
    def sort_key(agent: TeamAgent) -> tuple[int, int, str, str]:
        status = statuses.get(agent.agent_id) if statuses else None
        availability_rank = 1 if _agent_accepts_new_work(status) else 0
        provider_rank = PROVIDER_SORT_ORDER.get(agent.provider_preference, len(PROVIDER_SORT_ORDER))
        return (
            availability_rank,
            provider_rank,
            agent.full_name.casefold(),
            agent.handle.casefold(),
        )

    return sorted(agents, key=sort_key)


def _agent_accepts_new_work(status: AgentRosterStatus | None) -> bool:
    return status is None or status.label == "Available"


def _agent_status_text(status: AgentRosterStatus | None) -> str:
    if status is None:
        return "Available"
    detail = _bold_roster_detail_prefixes(status.detail) if status.detail else None
    lines = [f"*{status.label}:* {detail}"] if detail else [status.label]
    if status.pr_urls:
        label = "PRs" if len(status.pr_urls) > 1 else "PR"
        lines.append(f"*{label}:* {slack_pr_links(status.pr_urls, limit=3)}")
    if status.dangerous_mode:
        lines.append("*Mode:* :zap: Dangerous")
    return "\n".join(lines)


def _bold_roster_detail_prefixes(value: str) -> str:
    return re.sub(r"(?<!\*)\b(PRs):", r"*\1:*", value)


def _plain_text_header(value: str) -> str:
    return value[:150]


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
    """The welcome card posted when the agent channel is set up."""
    guide = f"""**Start work** — write `somebody ...` for any free agent, or `@agentname ...` for someone specific. The agent replies in your thread and keeps the thread's context.

**Thread subtasks** — reply `somebody ...` in a task thread to pull in another agent; the original agent picks the thread back up with the added context.

**Loops** — `loop create` sets up a recurring, read-only report in its own channel. `loops` lists them.

**Dangerous mode** — add `#dangerous-mode` to launch without sandbox or approvals. Active dangerous tasks are flagged on the roster.

**Commands** — type them here, or run `{slash_command} <command>`:
- `status` usage and active sessions
- `show roster` the team with its controls
- `external sessions` unassigned outside-Slack sessions
- `scheduled tasks` active schedules
- `hire 3 agents` · `fire everyone`
- `settings` auto-update, release checks and repo root

**Sessions started outside Slack** — Codex: `{codex_command}`. Claude: run `slackgentic claude-channel --install` once, then `{claude_command}`. Each session gets a tracked thread here; Slack replies and tool approvals relay through it. Restart already-open Claude sessions after installing the channel."""
    return [
        {
            "type": "card",
            "block_id": "slackgentic.overview.card",
            "title": {"type": "mrkdwn", "text": "*👋 Slackgentic is ready*"},
            "subtitle": {"type": "mrkdwn", "text": "Your agent team works right here"},
            "body": {
                "type": "mrkdwn",
                "text": "Write `somebody ...` in this channel to start a task. Agents reply in threads.",
            },
        },
        {"type": "markdown", "text": guide},
    ]


SETTINGS_BLOCK_ID = "slackgentic.settings"


@dataclass(frozen=True)
class SettingsSnapshot:
    version: str
    auto_update: bool
    update_checks: bool
    repo_root: str
    last_checked_at: datetime | None = None
    update_available: str | None = None
    available: bool = True


def build_settings_blocks(settings: SettingsSnapshot) -> list[dict[str, Any]]:
    """The `settings` card: each row shows its current value and one control."""
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "block_id": SETTINGS_BLOCK_ID,
            "text": {"type": "mrkdwn", "text": "*⚙️ Slackgentic settings*"},
        }
    ]
    if settings.available:
        blocks.append(
            _settings_toggle_row(
                "auto_update",
                "Auto-update",
                settings.auto_update,
                "Installs new releases and restarts the service on its own, "
                "once no agent task is running.",
            )
        )
        blocks.append(
            _settings_toggle_row(
                "update_checks",
                "Release checks",
                settings.update_checks,
                "Looks for new releases and posts an update card here. Auto-update needs this on.",
            )
        )
    blocks.append(
        {
            "type": "section",
            "block_id": f"{SETTINGS_BLOCK_ID}.repo_root",
            "text": {
                "type": "mrkdwn",
                "text": f"*Repo root*  `{settings.repo_root}`\nWhere agents start work by default.",
            },
            "accessory": _button(
                "Change",
                "slackgentic.settings.repo_root",
                encode_action_value("settings.repo_root.open"),
            ),
        }
    )
    version_parts = [f"Running Slackgentic {settings.version}"]
    if settings.update_available:
        version_parts.append(f"{settings.update_available} is available")
    if settings.last_checked_at is not None:
        version_parts.append(
            f"checked {_slack_time(settings.last_checked_at, '{date_short_pretty} at {time}')}"
        )
    blocks.append(
        {
            "type": "context",
            "block_id": f"{SETTINGS_BLOCK_ID}.version",
            "elements": [{"type": "mrkdwn", "text": "  ·  ".join(version_parts)}],
        }
    )
    if settings.available and settings.update_checks:
        blocks.append(
            {
                "type": "actions",
                "block_id": f"{SETTINGS_BLOCK_ID}.actions",
                "elements": [
                    _button(
                        "Check for updates",
                        "slackgentic.settings.check_updates",
                        encode_action_value("settings.check_updates"),
                    )
                ],
            }
        )
    return blocks


def _settings_toggle_row(
    setting: str,
    label: str,
    enabled: bool,
    description: str,
) -> dict[str, Any]:
    state = "*On*" if enabled else "*Off*"
    return {
        "type": "section",
        "block_id": f"{SETTINGS_BLOCK_ID}.{setting}",
        "text": {"type": "mrkdwn", "text": f"*{label}*  {state}\n{description}"},
        "accessory": _button(
            "Turn off" if enabled else "Turn on",
            f"slackgentic.settings.{setting}",
            encode_action_value("settings.toggle", setting=setting, enabled=not enabled),
            None if enabled else "primary",
        ),
    }


def build_repo_root_modal(
    repo_root: str,
    *,
    channel_id: str,
    message_ts: str | None,
) -> dict[str, Any]:
    return {
        "type": "modal",
        "callback_id": "settings.repo_root",
        "private_metadata": json.dumps(
            {"channel_id": channel_id, "message_ts": message_ts},
            separators=(",", ":"),
        ),
        "title": {"type": "plain_text", "text": "Repo root"},
        "submit": {"type": "plain_text", "text": "Save"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
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
            }
        ],
    }


def build_update_prompt_blocks(
    candidate: UpdateCandidate,
    *,
    status_text: str | None = None,
    include_actions: bool = True,
    draining: bool = False,
) -> list[dict[str, Any]]:
    """An update card: what's new, one click to upgrade, and live install status.

    While the upgrade waits for running agents (`draining`), the card offers to
    install right away instead of waiting.
    """
    release = candidate.release
    if status_text is None:
        body = "Upgrade now to install the published release and restart the service."
    else:
        body = status_text
    card: dict[str, Any] = {
        "type": "card",
        "block_id": f"slackgentic.update.card.{release.version}"[:255],
        "title": {"type": "mrkdwn", "text": f"✨ *Slackgentic {release.version} is out*"[:150]},
        "subtitle": {
            "type": "mrkdwn",
            "text": f"You're on {candidate.current_version}"[:150],
        },
        "body": {"type": "mrkdwn", "text": _shorten_text(body, 200)},
    }
    if include_actions:
        if draining:
            install_button = _button(
                "Install now",
                "slackgentic.update.install_now",
                encode_action_value("update.install_now", version=release.version),
                "danger",
            )
        else:
            install_button = _button(
                "Upgrade now",
                "slackgentic.update.install",
                encode_action_value("update.install", version=release.version),
                "primary",
            )
        actions = [install_button]
        if release.html_url:
            actions.append(
                _button(
                    "What's new",
                    "slackgentic.update.notes",
                    encode_action_value("update.notes", version=release.version),
                    url=release.html_url,
                )
            )
        if not draining:
            actions.append(
                _button(
                    "Not now",
                    "slackgentic.update.dismiss",
                    encode_action_value("update.dismiss", version=release.version),
                )
            )
        card["actions"] = actions
    blocks: list[dict[str, Any]] = [card]
    release_notes = _release_notes_excerpt(release.body)
    if release_notes and status_text is None:
        blocks.append({"type": "markdown", "text": f"**What's new**\n{release_notes}"})
    return blocks


def _release_notes_excerpt(body: str | None, *, limit: int = 1200) -> str | None:
    if not body:
        return None
    normalized = "\n".join(line.rstrip() for line in body.replace("\r\n", "\n").splitlines())
    normalized = normalized.strip()
    if not normalized:
        return None
    headlines = _release_note_headlines(normalized)
    if not headlines:
        return None
    lines: list[str] = []
    for headline in headlines:
        line = f"- {headline}"
        candidate = "\n".join([*lines, line])
        if len(candidate) > limit:
            break
        lines.append(line)
    if lines:
        return "\n".join(lines)
    first = f"- {headlines[0]}"
    return first[: limit - 3].rstrip() + "..."


def _release_note_headlines(body: str) -> list[str]:
    headlines: list[str] = []
    seen: set[str] = set()
    saw_changes_header = False
    in_changes = False
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("##"):
            header = line.lstrip("#").strip().rstrip(":").lower()
            in_changes = header in {"what's changed", "whats changed", "changes"}
            saw_changes_header = saw_changes_header or in_changes
            continue
        if line.lower().startswith("**full changelog**"):
            break
        if saw_changes_header and not in_changes:
            continue
        match = re.match(r"^(?:[-*+]|\d+\.)\s+(?P<headline>.+)$", line)
        if not match:
            continue
        headline = _clean_release_note_headline(match.group("headline"))
        if not headline or headline in seen:
            continue
        seen.add(headline)
        headlines.append(headline)
    return headlines


def _clean_release_note_headline(value: str) -> str | None:
    headline = value.strip()
    if not headline or headline.lower().startswith("**full changelog**"):
        return None
    headline = re.sub(
        r"\s+by\s+@[A-Za-z0-9-]+\s+in\s+\[[^\]]+\]\([^)]+\)\s*$",
        "",
        headline,
    )
    headline = re.sub(
        r"\s+by\s+@[A-Za-z0-9-]+\s+in\s+https?://\S+\s*$",
        "",
        headline,
    )
    headline = re.sub(r"\s+by\s+@[A-Za-z0-9-]+\s*$", "", headline)
    headline = re.sub(r"\s+in\s+https?://\S+\s*$", "", headline)
    headline = re.sub(r"<https?://[^>|]+\|([^>]+)>", r"\1", headline)
    headline = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", headline)
    headline = re.sub(r"https?://\S+", "", headline)
    headline = re.sub(r"\s+\(#\d+\)\s*$", "", headline)
    headline = re.sub(r"\s+\[#\d+\]\s*$", "", headline)
    headline = re.sub(r"\s+", " ", headline).strip(" -")
    return headline or None


def build_external_session_capacity_blocks(
    provider: Provider,
    waiting_count: int = 1,
) -> list[dict[str, Any]]:
    label = provider.value.title()
    plural = "session" if waiting_count == 1 else "sessions"
    agent_plural = "agent" if waiting_count == 1 else "agents"
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*No {label} team seat is available.*\n"
                    f"{waiting_count} {label} {plural} started outside Slack waiting. Hire "
                    f"{waiting_count} matching {agent_plural} and Slackgentic will backfill "
                    "visible transcript "
                    "output into the tracked thread."
                ),
            },
        },
        {
            "type": "actions",
            "block_id": f"external.capacity.{provider.value}",
            "elements": [
                _button(
                    f"Hire {waiting_count} {label} {agent_plural}",
                    f"external.capacity.hire.{provider.value}",
                    encode_action_value(
                        "team.hire",
                        count=waiting_count,
                        provider=provider.value,
                    ),
                    "primary",
                )
            ],
        },
    ]


def format_external_session_capacity_text(
    provider: Provider,
    waiting_count: int = 1,
) -> str:
    label = provider.value.title()
    session_plural = "session is" if waiting_count == 1 else "sessions are"
    agent_plural = "agent" if waiting_count == 1 else "agents"
    return (
        f"No {label} team seat is available for sessions started outside Slack. "
        f"{waiting_count} {label} {session_plural} waiting. "
        f"Hire {waiting_count} matching {agent_plural} and Slackgentic will backfill "
        "the transcript."
    )


def build_unassigned_external_session_blocks(
    items: list[UnassignedExternalSessionListItem],
    *,
    total_count: int | None = None,
) -> list[dict[str, Any]]:
    count = len(items) if total_count is None else total_count
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Unassigned external sessions*  {count} waiting",
            },
        }
    ]
    if not items:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "No active outside-Slack sessions are waiting for an agent.",
                },
            }
        )
        return blocks
    for item in items:
        section: dict[str, Any] = {
            "type": "section",
            "block_id": _external_session_block_id(item.session),
            "text": {"type": "mrkdwn", "text": _unassigned_external_session_text(item)},
        }
        if item.assignable_agents:
            section["accessory"] = _button(
                "Assign",
                "external.session.assign.open",
                encode_action_value(
                    "external.session.assign.open",
                    provider=item.session.provider.value,
                    session_id=item.session.session_id,
                ),
                "primary",
            )
        blocks.append(section)
        if not item.assignable_agents:
            label = item.session.provider.value.title()
            blocks.append(
                {
                    "type": "context",
                    "block_id": f"{_external_session_block_id(item.session)}.capacity",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"No available {label} engineer can take this session.",
                        }
                    ],
                }
            )
    return blocks


def _unassigned_external_session_text(item: UnassignedExternalSessionListItem) -> str:
    session = item.session
    label = session.provider.value.title()
    parts = [f"*{label}* {session.status.value} session `{_short_session_id(session.session_id)}`"]
    if item.summary and item.summary.strip():
        parts.append(f"*Task:* {_short_plain_text(item.summary, 140)}")
    elif session.cwd:
        parts.append(f"*Workspace:* `{session.cwd.name or session.cwd}`")
    if session.git_branch:
        parts.append(f"*Branch:* `{session.git_branch}`")
    if session.model:
        parts.append(f"*Model:* `{session.model}`")
    if item.thread_url:
        parts.append(f"*Thread:* <{item.thread_url}|open>")
    return "\n".join(parts)


def _external_session_block_id(session: AgentSession) -> str:
    digest = hashlib.sha256(f"{session.provider.value}.{session.session_id}".encode()).hexdigest()
    return f"external.unassigned.{digest[:16]}"


def _short_session_id(session_id: str) -> str:
    return _short_plain_text(session_id, 12)


def _short_plain_text(value: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 1)].rstrip()}..."


IDLE_RELEASE_PROMPT_TEXT = (
    "_Idle. Reply in this thread with anything else you need, "
    "or use the button to free up this agent._"
)

IDLE_RELEASE_DISMISSED_TEXT = "_Continuing — picked up the next turn in this thread._"
IDLE_RELEASE_CLOSED_TEXT = "_Finished and freed up this agent._"


def build_idle_release_prompt_blocks(task: AgentTask) -> list[dict[str, Any]]:
    return [
        {
            "type": "section",
            "block_id": f"task.idle.{task.task_id}",
            "text": {
                "type": "mrkdwn",
                "text": IDLE_RELEASE_PROMPT_TEXT,
            },
        },
        {
            "type": "actions",
            "block_id": f"task.idle.actions.{task.task_id}",
            "elements": [
                _button(
                    "Free up this agent",
                    "task.done",
                    encode_action_value("task.done", task_id=task.task_id),
                    "primary",
                ),
            ],
        },
    ]


def build_idle_release_dismissed_blocks(task: AgentTask) -> list[dict[str, Any]]:
    return [
        {
            "type": "section",
            "block_id": f"task.idle.{task.task_id}",
            "text": {
                "type": "mrkdwn",
                "text": IDLE_RELEASE_DISMISSED_TEXT,
            },
        },
    ]


def build_idle_release_closed_blocks(task: AgentTask) -> list[dict[str, Any]]:
    return [
        {
            "type": "section",
            "block_id": f"task.idle.{task.task_id}",
            "text": {
                "type": "mrkdwn",
                "text": IDLE_RELEASE_CLOSED_TEXT,
            },
        },
    ]


def build_task_thread_blocks(
    task: AgentTask,
    agent: TeamAgent,
    *,
    include_actions: bool = True,
) -> list[dict[str, Any]]:
    """The task thread header: a live task card for the agent's work."""
    task_label = "PR review" if task.kind.value == "review" else "task"
    original = _task_original_prompt(task)
    summary = task.metadata.get(ROSTER_SUMMARY_METADATA_KEY)
    summary = summary.strip() if isinstance(summary, str) else ""
    finished = task.status.value in {"done", "cancelled"}
    card: dict[str, Any] = {
        "type": "task_card",
        "task_id": task.task_id[:255],
        "title": _task_card_title(original),
        "status": "complete" if finished else "in_progress",
        "details": _rich_text(original[:2500]),
    }
    if summary and summary != original:
        card["output"] = _rich_text(f"Latest: {summary[:1500]}")
    pr_urls = pr_urls_from_metadata(task.metadata)
    if pr_urls:
        card["sources"] = [
            {"type": "url", "url": url, "text": label}
            for url, label in _pr_source_labels(pr_urls)[:5]
        ]
    context = [f"*{_mrkdwn_escape(agent.full_name)}* `@{agent.handle}` picked up this {task_label}"]
    provider = task.session_provider or agent.provider_preference
    if task.metadata.get(DANGEROUS_MODE_METADATA_KEY):
        context.append("⚡ dangerous mode")
    if finished:
        context.append("✅ done")
    context_elements: list[dict[str, Any]] = []
    if provider is not None:
        # The provider's logo leads the byline, in place of its name.
        context_elements.append(
            {
                "type": "image",
                "image_url": provider_logo_url(provider),
                "alt_text": provider.value.capitalize(),
            }
        )
    context_elements.append({"type": "mrkdwn", "text": " · ".join(context)[:2900]})
    blocks: list[dict[str, Any]] = [card, {"type": "context", "elements": context_elements}]
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


def _task_card_title(prompt: str) -> str:
    first_line = next((line.strip() for line in prompt.splitlines() if line.strip()), "Task")
    return _shorten_text(first_line, 120)


def _pr_source_labels(urls: tuple[str, ...] | list[str]) -> list[tuple[str, str]]:
    labels = []
    for url in urls:
        match = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", url)
        labels.append((url, f"{match.group(1)}#{match.group(2)}" if match else url))
    return labels


def _task_display_lines(task: AgentTask) -> str:
    original_prompt = _task_original_prompt(task)
    lines = [f"*Original Task:* {original_prompt}"]
    summary = task.metadata.get(ROSTER_SUMMARY_METADATA_KEY)
    if isinstance(summary, str) and summary.strip():
        summary = summary.strip()
        if summary != original_prompt:
            lines.append(f"*Latest summary:* {summary}")
    return "\n".join(lines)


def _task_original_prompt(task: AgentTask) -> str:
    original_task = task.metadata.get(ORIGINAL_TASK_METADATA_KEY)
    if isinstance(original_task, str) and original_task.strip():
        return original_task.strip()
    value = task.metadata.get(ASSIGNMENT_PROMPT_METADATA_KEY)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return task.prompt


def normalize_slack_mrkdwn(text: str) -> str:
    segments = re.split(r"(```.*?```)", _wrap_markdown_tables(text), flags=re.DOTALL)
    return "".join(
        segment if segment.startswith("```") else _normalize_bold_markers(segment)
        for segment in segments
    )


def replace_slack_user_ids(
    text: str,
    display_name_for: Callable[[str], str | None] | None = None,
    *,
    fallback: str = "Slack user",
) -> str:
    def label(user_id: str) -> str:
        if display_name_for is not None:
            display_name = display_name_for(user_id)
            if display_name and display_name.strip():
                return display_name.strip()
        return fallback

    def replace_ref(match: re.Match[str]) -> str:
        user_id = match.group("bracketed") or match.group("raw")
        return label(user_id)

    text = SLACK_USER_REF_RE.sub(replace_ref, text)
    return SLACK_USER_AUTHOR_RE.sub(
        lambda match: label(match.group("user_id")),
        text,
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
    return blocks[:SLACK_MAX_MESSAGE_BLOCKS]


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
    # Slack's rich-text table validator rejects cells whose text strips to ""
    # ("must be more than 0 characters"), so a literal NBSP or other
    # whitespace-only placeholder is not safe \u2014 use a visible em-dash.
    blank_cell_text = "\u2014"
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
        if not value.strip():
            value = blank_cell_text
            style.pop("code", None)
        if bold:
            style["bold"] = True
        element: dict[str, Any] = {"type": "text", "text": value}
        if style:
            element["style"] = style
        elements.append(element)
    return elements or [{"type": "text", "text": blank_cell_text}]


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


STATUS_REFRESH_ACTION = "usage.refresh"


def build_status_blocks(
    accounts: list[AccountStatus],
    *,
    day_text: str,
    updated_at: datetime,
) -> list[dict[str, Any]]:
    """Quota and usage per signed-in account, most urgent numbers first."""
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "📊 Agent status", "emoji": True},
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"{day_text} · updated {_slack_time(updated_at, '{time}')}",
                }
            ],
        },
    ]
    for account in accounts:
        blocks.append({"type": "divider"})
        lines = [f"*{_mrkdwn_escape(account_heading(account))}*"]
        for window in account.windows:
            lines.append(_status_window_line(window))
        if account.quota_note:
            lines.append(f"_{_mrkdwn_escape(account.quota_note)}_")
        blocks.append(
            {
                "type": "section",
                "block_id": f"usage.account.{account.key}"[:255],
                "text": {"type": "mrkdwn", "text": "\n".join(lines)[:2900]},
            }
        )
        details = [account_tokens_line(account)]
        if account.top_sessions:
            top = " · ".join(
                f"{_mrkdwn_escape(_shorten_text(session.label, 48))} *{session.percent:.0f}%*"
                for session in account.top_sessions
            )
            details.append(f"Top today: {top}")
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": "\n".join(details)[:2900]}],
            }
        )
    blocks.append(
        {
            "type": "actions",
            "block_id": "usage.actions",
            "elements": [
                _button(
                    "↻ Refresh",
                    STATUS_REFRESH_ACTION,
                    encode_action_value(STATUS_REFRESH_ACTION),
                )
            ],
        }
    )
    return blocks


def _status_window_line(window) -> str:
    percent = window.used_percent
    reset = ""
    if window.resets_at is not None:
        reset = f"  ·  resets {_slack_time(window.resets_at, '{date_short_pretty} at {time}')}"
    return f"{_quota_bar(percent)}  *{percent:.0f}%* {window.label}{reset}"


def _quota_bar(percent: float, width: int = 10) -> str:
    clamped = max(0.0, min(100.0, percent))
    filled = int(clamped / 100.0 * width + 0.5)
    if clamped > 0 and filled == 0:
        filled = 1
    color = "🟥" if clamped >= 90 else "🟧" if clamped >= 75 else "🟩"
    return color * filled + "⬜" * (width - filled)


def _slack_time(value: datetime, token_format: str) -> str:
    fallback = value.astimezone(UTC).strftime("%b %d %H:%M UTC")
    return f"<!date^{int(value.timestamp())}^{token_format}|{fallback}>"


def _button(
    text: str,
    action_id: str,
    value: str,
    style: str | None = None,
    url: str | None = None,
    confirm: dict[str, Any] | None = None,
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
    if confirm:
        button["confirm"] = confirm
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
