from __future__ import annotations

import difflib
import json
import logging
import shlex
import threading
import time
from dataclasses import dataclass, field
from secrets import token_urlsafe
from typing import Any

from agent_harness.models import SlackThreadRef
from agent_harness.slack import encode_action_value
from agent_harness.slack.client import SlackGateway
from agent_harness.storage.store import Store

AGENT_REQUEST_ACTION = "agent.request"
LEGACY_CODEX_REQUEST_ACTION = "codex.request"
AGENT_REQUEST_ACTIONS = {AGENT_REQUEST_ACTION, LEGACY_CODEX_REQUEST_ACTION}
LOGGER = logging.getLogger(__name__)


@dataclass
class PendingAgentRequest:
    token: str
    request_id: object
    provider_label: str
    method: str
    params: dict[str, Any]
    thread: SlackThreadRef
    action_name: str = AGENT_REQUEST_ACTION
    event: threading.Event = field(default_factory=threading.Event)
    response: Any = None
    message_ts: str | None = None
    answers: dict[str, str] = field(default_factory=dict)


class SlackAgentRequestHandler:
    def __init__(
        self,
        gateway: SlackGateway,
        timeout_seconds: float = 1800.0,
        *,
        store: Store | None = None,
        provider_label: str = "Agent",
        action_name: str = AGENT_REQUEST_ACTION,
        poll_seconds: float = 0.05,
    ):
        self.gateway = gateway
        self.timeout_seconds = timeout_seconds
        self.store = store
        self.provider_label = provider_label
        self.action_name = action_name
        self.poll_seconds = poll_seconds
        self._pending: dict[str, PendingAgentRequest] = {}
        self._lock = threading.Lock()

    def handle_server_request(
        self,
        request: dict[str, Any],
        thread: SlackThreadRef,
        *,
        provider_label: str | None = None,
    ) -> Any:
        method = str(request.get("method") or "")
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        return self.handle_request(method, params, thread, provider_label=provider_label)

    def handle_request(
        self,
        method: str,
        params: dict[str, Any],
        thread: SlackThreadRef,
        *,
        provider_label: str | None = None,
    ) -> Any:
        pending = PendingAgentRequest(
            token=token_urlsafe(12),
            request_id=None,
            provider_label=provider_label or self.provider_label,
            method=method,
            params=params,
            thread=thread,
            action_name=self.action_name,
        )
        with self._lock:
            self._pending[pending.token] = pending
        text, blocks = _request_message(pending)
        posted = self.gateway.post_thread_reply(thread, text, blocks=blocks)
        pending.message_ts = posted.ts
        try:
            if not pending.event.wait(self.timeout_seconds):
                pending.response = _timeout_response(method)
                self._update_request_message(
                    pending, f"{pending.provider_label} request timed out."
                )
            return pending.response
        finally:
            with self._lock:
                self._pending.pop(pending.token, None)

    def handle_persistent_request(
        self,
        method: str,
        params: dict[str, Any],
        thread: SlackThreadRef,
        *,
        provider_label: str | None = None,
    ) -> Any:
        if self.store is None:
            return self.handle_request(method, params, thread, provider_label=provider_label)
        pending = PendingAgentRequest(
            token=token_urlsafe(12),
            request_id=None,
            provider_label=provider_label or self.provider_label,
            method=method,
            params=params,
            thread=thread,
            action_name=self.action_name,
        )
        self.store.create_slack_agent_request(
            pending.token,
            pending.provider_label,
            pending.method,
            pending.params,
            pending.thread,
        )
        text, blocks = _request_message(pending)
        posted = self.gateway.post_thread_reply(thread, text, blocks=blocks)
        pending.message_ts = posted.ts
        self.store.update_slack_agent_request_message_ts(pending.token, posted.ts)
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            resolved, response = self.store.get_slack_agent_request_response(pending.token)
            if resolved:
                return response
            time.sleep(self.poll_seconds)
        response = _timeout_response(method)
        self.store.resolve_slack_agent_request(pending.token, response)
        self._update_request_message(pending, f"{pending.provider_label} request timed out.")
        return response

    def handle_block_action(
        self,
        payload: dict[str, Any],
        channel_id: str,
        message_ts: str | None,
    ) -> bool:
        if payload.get("action") not in AGENT_REQUEST_ACTIONS:
            return False
        token = str(payload.get("token") or "")
        with self._lock:
            pending = self._pending.get(token)
        if pending is not None:
            return self._handle_memory_action(pending, payload, message_ts)
        if self.store is not None:
            return self._handle_persistent_action(token, payload, channel_id, message_ts)
        return True

    def _handle_memory_action(
        self,
        pending: PendingAgentRequest,
        payload: dict[str, Any],
        message_ts: str | None,
    ) -> bool:
        if pending.method == "item/tool/requestUserInput":
            self._handle_user_input_action(pending, payload)
            if not pending.event.is_set():
                text, blocks = _request_message(pending)
                self._update_request_message(pending, text, blocks=blocks, message_ts=message_ts)
            return True
        decision = str(payload.get("decision") or "")
        pending.response = _decision_response(pending.method, decision, pending.params)
        pending.event.set()
        self._update_request_message(
            pending,
            _resolved_text(pending.method, decision, pending.provider_label),
            message_ts=message_ts,
        )
        return True

    def _handle_persistent_action(
        self,
        token: str,
        payload: dict[str, Any],
        channel_id: str,
        message_ts: str | None,
    ) -> bool:
        row = self.store.get_slack_agent_request(token) if self.store else None
        if row is None:
            return True
        pending = _pending_from_row(row, fallback_channel_id=channel_id)
        if row["resolved_at"]:
            return True
        if pending.method == "item/tool/requestUserInput":
            response, answers = _user_input_response(pending, payload)
            if response is None:
                if self.store:
                    self.store.update_slack_agent_request_answers(token, answers)
                pending.answers = answers
                text, blocks = _request_message(pending)
                self._update_request_message(pending, text, blocks=blocks, message_ts=message_ts)
                return True
            if self.store:
                self.store.resolve_slack_agent_request(token, response)
            self._update_request_message(
                pending,
                _input_resolved_text(pending, payload),
                message_ts=message_ts,
            )
            return True
        decision = str(payload.get("decision") or "")
        response = _decision_response(pending.method, decision, pending.params)
        if self.store:
            self.store.resolve_slack_agent_request(token, response)
        self._update_request_message(
            pending,
            _resolved_text(pending.method, decision, pending.provider_label),
            message_ts=message_ts,
        )
        return True

    def _handle_user_input_action(
        self,
        pending: PendingAgentRequest,
        payload: dict[str, Any],
    ) -> None:
        response, answers = _user_input_response(pending, payload)
        pending.answers = answers
        if response is None:
            return
        pending.response = response
        pending.event.set()
        self._update_request_message(pending, _input_resolved_text(pending, payload))

    def _update_request_message(
        self,
        pending: PendingAgentRequest,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
        message_ts: str | None = None,
    ) -> None:
        ts = message_ts or pending.message_ts
        if not ts:
            return
        try:
            self.gateway.update_message(
                pending.thread.channel_id,
                ts,
                text[:2800],
                blocks=blocks,
            )
        except Exception:
            LOGGER.debug("failed to update Slack agent request message", exc_info=True)


def render_persistent_agent_request(
    row: Any, *, fallback_channel_id: str
) -> tuple[str, list[dict[str, Any]]]:
    return _request_message(_pending_from_row(row, fallback_channel_id=fallback_channel_id))


def _request_message(pending: PendingAgentRequest) -> tuple[str, list[dict[str, Any]]]:
    if pending.method == "item/tool/requestUserInput":
        return _input_request_message(pending)
    if pending.method == "claude/channel/permission":
        return _claude_permission_request_message(pending)
    text = _approval_text(pending.method, pending.params, pending.provider_label)
    return text, [
        {"type": "section", "text": {"type": "mrkdwn", "text": _mrkdwn(text)}},
        *_approval_action_blocks(pending),
    ]


def _input_request_message(pending: PendingAgentRequest) -> tuple[str, list[dict[str, Any]]]:
    questions = _questions(pending.params)
    text = f"{pending.provider_label} needs input."
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{text}*"}}
    ]
    for question_index, question in enumerate(questions):
        question_id = str(question.get("id") or "")
        selected = pending.answers.get(question_id)
        detail = (
            f"*{_plain(question.get('header'), 'Question')}*\n{_plain(question.get('question'))}"
        )
        if selected:
            detail = f"{detail}\nSelected: `{_truncate(selected, 80)}`"
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": _mrkdwn(detail)}})
        options = question.get("options") if isinstance(question.get("options"), list) else []
        elements = [
            _button(
                _plain(option.get("label"), "Option"),
                f"agent.request.input.{question_index}.{option_index}",
                pending,
                question_id=question_id,
                answer=_plain(option.get("label"), "Option"),
                style="primary" if selected == option.get("label") else None,
            )
            for option_index, option in enumerate(options[:5])
            if isinstance(option, dict)
        ]
        if elements:
            blocks.append(
                {
                    "type": "actions",
                    "block_id": f"agent.input.{pending.token}.{question_id}"[:255],
                    "elements": elements,
                }
            )
    blocks.append(
        {
            "type": "actions",
            "block_id": f"agent.input.cancel.{pending.token}"[:255],
            "elements": [
                _button(
                    "Cancel",
                    "agent.request.cancel",
                    pending,
                    decision="cancel",
                    style="danger",
                )
            ],
        }
    )
    return text, blocks[:50]


def _claude_permission_request_message(
    pending: PendingAgentRequest,
) -> tuple[str, list[dict[str, Any]]]:
    tool_name = _plain(pending.params.get("tool_name"), "tool")
    description = _plain(pending.params.get("description"), "")
    fallback = f"{pending.provider_label} requests tool approval: {tool_name}"
    summary_lines = [f"*{pending.provider_label} requests tool approval.*"]
    if tool_name:
        summary_lines.append(f"Tool: `{_truncate(tool_name, 200)}`")
    if description:
        summary_lines.append(_truncate(description, 500))
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": _mrkdwn("\n".join(summary_lines))},
        }
    ]
    diff_text = _permission_diff_text(pending.params)
    if not diff_text and _is_truncated_edit_permission(pending.params):
        blocks.extend(_truncated_edit_permission_blocks(pending.params))
        blocks.extend(_approval_action_blocks(pending))
        return fallback, blocks[:50]
    input_text = diff_text or _permission_input_text(pending.params)
    if input_text:
        chunks = _block_text_chunks(input_text, 2500)
        for index, chunk in enumerate(chunks[:6]):
            if diff_text:
                heading = "*Proposed diff*\n" if index == 0 else ""
                fence = "diff\n"
            else:
                heading = "*Input preview*\n" if index == 0 else ""
                fence = ""
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{heading}```{fence}{_code_block_text(chunk)}```",
                    },
                }
            )
        if len(chunks) > 6:
            label = "Diff" if diff_text else "Input preview"
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"{label} is too large for one Slack message.",
                        }
                    ],
                }
            )
        if _permission_input_was_truncated_before_slackgentic(pending.params):
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                "Claude only provided this truncated preview to Slackgentic. "
                                "Review the full request in the Claude terminal before allowing."
                            ),
                        }
                    ],
                }
            )
    blocks.extend(_approval_action_blocks(pending))
    return fallback, blocks[:50]


def _approval_action_blocks(pending: PendingAgentRequest) -> list[dict[str, Any]]:
    method = pending.method
    if method in {
        "item/commandExecution/requestApproval",
        "execCommandApproval",
        "item/fileChange/requestApproval",
        "applyPatchApproval",
        "agent/requestApproval",
    }:
        labels = [
            ("Approve", "approve", "primary"),
            ("Approve Session", "approve_session", None),
            ("Deny", "deny", None),
            ("Cancel Turn", "cancel", "danger"),
        ]
    elif method == "claude/channel/permission":
        labels = [
            ("Allow", "approve", "primary"),
            ("Deny", "deny", "danger"),
        ]
    elif method == "item/permissions/requestApproval":
        labels = [
            ("Approve Turn", "approve", "primary"),
            ("Approve Session", "approve_session", None),
            ("Deny", "deny", "danger"),
        ]
    else:
        labels = [("Cancel", "cancel", "danger")]
    return [
        {
            "type": "actions",
            "block_id": f"agent.request.{pending.token}"[:255],
            "elements": [
                _button(
                    label,
                    f"agent.request.{decision}",
                    pending,
                    decision=decision,
                    style=style,
                )
                for label, decision, style in labels
            ],
        }
    ]


def _approval_text(method: str, params: dict[str, Any], provider_label: str) -> str:
    if method in {"item/commandExecution/requestApproval", "execCommandApproval"}:
        command = _command_text(params.get("command"))
        reason = _plain(params.get("reason"), "")
        cwd = _plain(params.get("cwd"), "")
        lines = [f"*{provider_label} requests command approval.*"]
        if command:
            lines.append(f"Command: `{_truncate(command, 900)}`")
        if cwd:
            lines.append(f"cwd: `{_truncate(cwd, 200)}`")
        if reason:
            lines.append(_truncate(reason, 500))
        return "\n".join(lines)
    if method in {"item/fileChange/requestApproval", "applyPatchApproval"}:
        reason = _plain(params.get("reason"), "")
        grant_root = _plain(params.get("grantRoot"), "")
        lines = [f"*{provider_label} requests file-change approval.*"]
        if grant_root:
            lines.append(f"Grant root: `{_truncate(grant_root, 200)}`")
        if reason:
            lines.append(_truncate(reason, 500))
        return "\n".join(lines)
    if method == "item/permissions/requestApproval":
        reason = _plain(params.get("reason"), "")
        cwd = _plain(params.get("cwd"), "")
        lines = [f"*{provider_label} requests additional permissions.*"]
        if cwd:
            lines.append(f"cwd: `{_truncate(cwd, 200)}`")
        summary = _permissions_summary(params.get("permissions"))
        if summary:
            lines.append(summary)
        if reason:
            lines.append(_truncate(reason, 500))
        return "\n".join(lines)
    if method == "claude/channel/permission":
        tool_name = _plain(params.get("tool_name"), "tool")
        description = _plain(params.get("description"), "")
        input_preview = _plain(params.get("input_preview"), "")
        lines = [f"*{provider_label} requests tool approval.*"]
        if tool_name:
            lines.append(f"Tool: `{_truncate(tool_name, 200)}`")
        if description:
            lines.append(_truncate(description, 500))
        if input_preview:
            lines.append(f"Input: `{_truncate(input_preview, 900)}`")
        return "\n".join(lines)
    if method == "agent/requestApproval":
        title = _plain(params.get("title"), "approval")
        reason = _plain(params.get("reason"), "")
        lines = [f"*{provider_label} requests approval.*", _truncate(title, 500)]
        if reason:
            lines.append(_truncate(reason, 500))
        return "\n".join(lines)
    return f"*{provider_label} sent an unsupported request:* `{_truncate(method, 120)}`"


def _decision_response(method: str, decision: str, params: dict[str, Any]) -> Any:
    if method == "item/commandExecution/requestApproval":
        return {"decision": _new_approval_decision(decision)}
    if method == "item/fileChange/requestApproval":
        return {"decision": _file_change_decision(decision)}
    if method == "execCommandApproval":
        return {"decision": _legacy_approval_decision(decision)}
    if method == "applyPatchApproval":
        return {"decision": _legacy_approval_decision(decision)}
    if method == "agent/requestApproval":
        return {"decision": _legacy_approval_decision(decision)}
    if method == "item/permissions/requestApproval":
        if decision in {"approve", "approve_session"}:
            return {
                "permissions": params.get("permissions") or {},
                "scope": "session" if decision == "approve_session" else "turn",
            }
        return {"permissions": {}, "scope": "turn"}
    if method == "claude/channel/permission":
        return {"behavior": "allow" if decision == "approve" else "deny"}
    return None


def _new_approval_decision(decision: str) -> str:
    return {
        "approve": "accept",
        "approve_session": "acceptForSession",
        "deny": "decline",
        "cancel": "cancel",
    }.get(decision, "cancel")


def _file_change_decision(decision: str) -> str:
    return {
        "approve": "accept",
        "approve_session": "acceptForSession",
        "deny": "decline",
        "cancel": "cancel",
    }.get(decision, "cancel")


def _legacy_approval_decision(decision: str) -> str:
    return {
        "approve": "approved",
        "approve_session": "approved_for_session",
        "deny": "denied",
        "cancel": "abort",
    }.get(decision, "abort")


def _timeout_response(method: str) -> Any:
    if method == "item/tool/requestUserInput":
        return {"answers": {}}
    if method == "item/permissions/requestApproval":
        return {"permissions": {}, "scope": "turn"}
    if method == "claude/channel/permission":
        return {"behavior": "deny"}
    if method in {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}:
        return {"decision": "cancel"}
    if method in {"execCommandApproval", "applyPatchApproval", "agent/requestApproval"}:
        return {"decision": "abort"}
    return None


def _resolved_text(method: str, decision: str, provider_label: str) -> str:
    if method == "claude/channel/permission":
        return {
            "approve": f"Allowed {provider_label} tool request.",
            "deny": f"Denied {provider_label} tool request.",
            "cancel": f"Denied {provider_label} tool request.",
        }.get(decision, f"Handled {provider_label} tool request.")
    if method == "item/permissions/requestApproval":
        return {
            "approve": f"Approved {provider_label} permissions for this turn.",
            "approve_session": f"Approved {provider_label} permissions for this session.",
            "deny": f"Denied {provider_label} permissions.",
            "cancel": f"Denied {provider_label} permissions.",
        }.get(decision, f"Handled {provider_label} permissions request.")
    return {
        "approve": f"Approved {provider_label} request.",
        "approve_session": f"Approved {provider_label} request for this session.",
        "deny": f"Denied {provider_label} request.",
        "cancel": f"Cancelled {provider_label} turn.",
    }.get(decision, f"Handled {provider_label} request.")


def _input_resolved_text(pending: PendingAgentRequest, payload: dict[str, Any]) -> str:
    if payload.get("decision") == "cancel":
        return f"Cancelled {pending.provider_label} input request."
    return f"Answered {pending.provider_label} input request."


def _user_input_response(
    pending: PendingAgentRequest,
    payload: dict[str, Any],
) -> tuple[Any | None, dict[str, str]]:
    if payload.get("decision") == "cancel":
        return {"answers": {}}, dict(pending.answers)
    answers = dict(pending.answers)
    question_id = str(payload.get("question_id") or "")
    answer = str(payload.get("answer") or "")
    if not question_id or not answer:
        return None, answers
    answers[question_id] = answer
    required_ids = [
        str(question.get("id") or "")
        for question in _questions(pending.params)
        if question.get("id")
    ]
    if not required_ids or any(question_id not in answers for question_id in required_ids):
        return None, answers
    return {
        "answers": {
            question_id: {"answers": [answers[question_id]]} for question_id in required_ids
        }
    }, answers


def _button(
    text: str,
    action_id: str,
    pending: PendingAgentRequest,
    *,
    decision: str | None = None,
    question_id: str | None = None,
    answer: str | None = None,
    style: str | None = None,
) -> dict[str, Any]:
    value = encode_action_value(
        pending.action_name,
        token=pending.token,
        decision=decision,
        question_id=question_id,
        answer=answer,
    )
    button: dict[str, Any] = {
        "type": "button",
        "text": {"type": "plain_text", "text": _truncate(text, 75)},
        "action_id": action_id,
        "value": value,
    }
    if style:
        button["style"] = style
    return button


def _pending_from_row(row: Any, *, fallback_channel_id: str) -> PendingAgentRequest:
    try:
        params = json.loads(row["params_json"])
    except (TypeError, json.JSONDecodeError):
        params = {}
    if not isinstance(params, dict):
        params = {}
    try:
        answers = json.loads(row["answers_json"])
    except (TypeError, json.JSONDecodeError):
        answers = {}
    if not isinstance(answers, dict):
        answers = {}
    thread_channel_id = row["thread_channel_id"] or fallback_channel_id
    return PendingAgentRequest(
        token=str(row["token"]),
        request_id=None,
        provider_label=str(row["provider_label"] or "Agent"),
        method=str(row["method"] or ""),
        params=params,
        thread=SlackThreadRef(str(thread_channel_id), str(row["thread_ts"])),
        message_ts=row["message_ts"],
        answers={str(key): str(value) for key, value in answers.items()},
    )


def _questions(params: dict[str, Any]) -> list[dict[str, Any]]:
    questions = params.get("questions")
    if not isinstance(questions, list):
        return []
    return [question for question in questions if isinstance(question, dict)]


def _command_text(command: object) -> str:
    if isinstance(command, str):
        return command
    if isinstance(command, list):
        return " ".join(shlex.quote(str(part)) for part in command)
    return ""


def _permissions_summary(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    parts: list[str] = []
    network = value.get("network")
    if isinstance(network, dict) and network.get("enabled") is not None:
        parts.append(f"network: `{bool(network.get('enabled'))}`")
    file_system = value.get("fileSystem")
    if isinstance(file_system, dict):
        entries = file_system.get("entries")
        if isinstance(entries, list):
            parts.append(f"filesystem entries: `{len(entries)}`")
        read = file_system.get("read")
        write = file_system.get("write")
        if isinstance(read, list):
            parts.append(f"read paths: `{len(read)}`")
        if isinstance(write, list):
            parts.append(f"write paths: `{len(write)}`")
    return ", ".join(parts)


def _permission_diff_text(params: dict[str, Any]) -> str:
    explicit = _explicit_permission_diff(params)
    if explicit:
        return explicit
    tool_name = _plain(params.get("tool_name"), "")
    if tool_name not in {"Edit", "MultiEdit"}:
        return ""
    input_value = _permission_input_object(params)
    if not isinstance(input_value, dict):
        return ""
    file_path = _plain(input_value.get("file_path"), "file")
    edits = input_value.get("edits")
    if isinstance(edits, list):
        diffs = [_edit_diff_text(file_path, edit) for edit in edits if isinstance(edit, dict)]
        return "\n\n".join(diff for diff in diffs if diff)
    return _edit_diff_text(file_path, input_value)


def _is_truncated_edit_permission(params: dict[str, Any]) -> bool:
    return _plain(params.get("tool_name"), "") in {
        "Edit",
        "MultiEdit",
    } and _permission_input_was_truncated_before_slackgentic(params)


def _truncated_edit_permission_blocks(params: dict[str, Any]) -> list[dict[str, Any]]:
    file_path = _truncated_preview_field(params, "file_path")
    lines = ["*Diff unavailable in Slack.*"]
    if file_path:
        lines.append(f"File: `{_truncate(file_path, 240)}`")
    lines.append(
        "Claude only sent Slackgentic a shortened native preview for this edit. "
        "The Claude terminal has the proposed diff."
    )
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": _mrkdwn("\n".join(lines))},
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        "Restart this Claude session after the current turn to enable "
                        "Slack diffs when Claude includes full edit payloads."
                    ),
                }
            ],
        },
    ]


def _truncated_preview_field(params: dict[str, Any], key: str) -> str:
    preview = _plain(params.get("input_preview"), "")
    if not preview:
        return ""
    marker = f'"{key}":"'
    start = preview.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = preview.find('"', start)
    if end < 0:
        return ""
    return preview[start:end]


def _explicit_permission_diff(params: dict[str, Any]) -> str:
    for key in ("diff", "input_diff", "preview_diff"):
        value = _plain(params.get(key), "")
        if value:
            return value
    display = params.get("display")
    if isinstance(display, str) and display.strip():
        return display.strip()
    if isinstance(display, dict):
        for key in ("diff", "content", "text"):
            value = _plain(display.get(key), "")
            if value:
                return value
    return ""


def _edit_diff_text(file_path: str, edit: dict[str, Any]) -> str:
    old_string = edit.get("old_string")
    new_string = edit.get("new_string")
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        return ""
    lines = difflib.unified_diff(
        old_string.splitlines(),
        new_string.splitlines(),
        fromfile=f"{file_path} (current)",
        tofile=f"{file_path} (proposed)",
        lineterm="",
    )
    return "\n".join(lines) or "(no textual diff)"


def _permission_input_object(params: dict[str, Any]) -> object:
    for key in ("input", "arguments", "tool_input", "parameters"):
        if key in params:
            value = params[key]
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value
    preview = _plain(params.get("input_preview"), "")
    if preview:
        try:
            return json.loads(preview)
        except json.JSONDecodeError:
            return preview
    return None


def _permission_input_text(params: dict[str, Any]) -> str:
    for key in ("input", "arguments", "tool_input", "parameters"):
        if key in params:
            return _format_input_value(params[key])
    return _format_input_value(params.get("input_preview"))


def _permission_input_was_truncated_before_slackgentic(params: dict[str, Any]) -> bool:
    if any(key in params for key in ("input", "arguments", "tool_input", "parameters")):
        return False
    preview = _plain(params.get("input_preview"), "")
    if not preview:
        return False
    stripped = preview.rstrip()
    return stripped.endswith(("…", "...")) or "…," in stripped


def _format_input_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, sort_keys=True)
    text = _plain(value, "")
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _format_jsonish_preview(text)
    return json.dumps(parsed, indent=2, sort_keys=True)


def _format_jsonish_preview(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return stripped
    formatted = stripped
    formatted = formatted.replace('{"', '{\n  "')
    formatted = formatted.replace(',"', ',\n  "')
    formatted = formatted.replace('"}', '"\n}')
    return formatted


def _block_text_chunks(text: str, limit: int) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < max(1, limit // 2):
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks


def _code_block_text(text: str) -> str:
    return text.replace("```", "` ` `")


def _plain(value: object, default: str = "") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _mrkdwn(text: str) -> str:
    return _truncate(text, 2900)


def _truncate(text: str, limit: int) -> str:
    cleaned = text.replace("\x00", "")
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 12)]} ... [more]"
