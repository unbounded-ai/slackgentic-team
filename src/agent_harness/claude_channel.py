from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from agent_harness.config import load_config_from_env
from agent_harness.models import SlackThreadRef
from agent_harness.slack_agent_requests import SlackAgentRequestHandler
from agent_harness.store import Store

CHANNEL_NAME = "slackgentic"
CHANNEL_VERSION = "0.1.0"
CHANNEL_INSTRUCTIONS = (
    "Slackgentic forwards Slack thread replies into this Claude Code session as "
    '<channel source="slackgentic" ...> events. Treat each event as a user message '
    "from Slack for this existing session. Respond normally in the terminal; "
    "Slackgentic mirrors your visible assistant responses back to the Slack thread. "
    "Do not mention this transport unless it is directly relevant."
)


class ClaudeChannelServer:
    def __init__(
        self,
        store: Store,
        target_pid: int | None = None,
        poll_seconds: float = 0.2,
        request_handler: SlackAgentRequestHandler | None = None,
    ):
        self.store = store
        self.target_pid = target_pid or os.getppid()
        self.poll_seconds = poll_seconds
        self.request_handler = request_handler
        self._current_thread: SlackThreadRef | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._write_lock = threading.Lock()

    def run(self) -> int:
        reader = threading.Thread(target=self._read_loop, daemon=True)
        reader.start()
        while not self._stop.is_set():
            if not self._ready.wait(self.poll_seconds):
                continue
            self._deliver_pending()
            self._stop.wait(self.poll_seconds)
        return 0

    def _read_loop(self) -> None:
        for line in sys.stdin:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict):
                self._handle_message(message)
        self._stop.set()

    def _handle_message(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        message_id = message.get("id")
        if method == "initialize":
            params = message.get("params") if isinstance(message.get("params"), dict) else {}
            protocol = str(params.get("protocolVersion") or "2024-11-05")
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "protocolVersion": protocol,
                        "capabilities": {
                            "experimental": {
                                "claude/channel": {},
                            },
                        },
                        "serverInfo": {
                            "name": CHANNEL_NAME,
                            "version": CHANNEL_VERSION,
                        },
                        "instructions": CHANNEL_INSTRUCTIONS,
                    },
                }
            )
            self._ready.set()
            return
        if method == "notifications/initialized":
            self._ready.set()
            return
        if method == "ping" and message_id is not None:
            self._write({"jsonrpc": "2.0", "id": message_id, "result": {}})
            return
        if method == "tools/list" and message_id is not None:
            self._write({"jsonrpc": "2.0", "id": message_id, "result": {"tools": _tools()}})
            return
        if method == "tools/call" and message_id is not None:
            params = message.get("params") if isinstance(message.get("params"), dict) else {}
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": self._handle_tool_call(params),
                }
            )
            return
        if message_id is not None:
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
            )

    def _deliver_pending(self) -> None:
        for row in self.store.pending_claude_channel_messages(self.target_pid):
            try:
                meta = json.loads(row["meta_json"])
            except json.JSONDecodeError:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            safe_meta = {
                str(key): str(value) for key, value in meta.items() if _is_identifier(str(key))
            }
            self._remember_thread(safe_meta)
            self._write(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/claude/channel",
                    "params": {
                        "content": str(row["content"]),
                        "meta": safe_meta,
                    },
                }
            )
            self.store.mark_claude_channel_message_delivered(int(row["id"]))

    def _remember_thread(self, meta: dict[str, str]) -> None:
        channel_id = meta.get("slack_channel")
        thread_ts = meta.get("slack_thread_ts")
        if channel_id and thread_ts:
            self._current_thread = SlackThreadRef(channel_id, thread_ts)

    def _handle_tool_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if name == "request_user_input":
            return self._handle_request_user_input_tool(arguments)
        if name == "request_approval":
            return self._handle_request_approval_tool(arguments)
        return _tool_result(f"Unknown Slackgentic tool: {name}", is_error=True)

    def _handle_request_user_input_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.request_handler is None:
            return _tool_result("Slack request handling is not configured.", is_error=True)
        if self._current_thread is None:
            return _tool_result("No Slack thread is active for this Claude session.", is_error=True)
        question = _string_arg(arguments, "question")
        if not question:
            return _tool_result("request_user_input requires a question.", is_error=True)
        options = _tool_options(arguments.get("options"))
        if not options:
            return _tool_result("request_user_input requires at least one option.", is_error=True)
        params = {
            "questions": [
                {
                    "id": "answer",
                    "header": _string_arg(arguments, "header") or "Question",
                    "question": question,
                    "options": options,
                }
            ]
        }
        response = self.request_handler.handle_persistent_request(
            "item/tool/requestUserInput",
            params,
            self._current_thread,
            provider_label="Claude",
        )
        answer = _selected_answer(response, "answer")
        if not answer:
            return _tool_result("No answer was selected.", is_error=True)
        return _tool_result(answer)

    def _handle_request_approval_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.request_handler is None:
            return _tool_result("Slack request handling is not configured.", is_error=True)
        if self._current_thread is None:
            return _tool_result("No Slack thread is active for this Claude session.", is_error=True)
        method, params = _approval_request(arguments)
        response = self.request_handler.handle_persistent_request(
            method,
            params,
            self._current_thread,
            provider_label="Claude",
        )
        return _tool_result(json.dumps(response, sort_keys=True))

    def _write(self, message: dict[str, Any]) -> None:
        with self._write_lock:
            sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def run_channel_server(db_path: Path | None = None) -> int:
    config = load_config_from_env()
    store = Store(db_path or config.state_db)
    try:
        store.init_schema()
        request_handler = None
        if config.slack.bot_token:
            from agent_harness.slack_client import SlackGateway

            request_handler = SlackAgentRequestHandler(
                SlackGateway(config.slack.bot_token),
                store=store,
                provider_label="Claude",
            )
        return ClaudeChannelServer(store, request_handler=request_handler).run()
    finally:
        store.close()


def install_claude_mcp_server(command: str | None = None) -> None:
    resolved = command or _current_slackgentic_command()
    subprocess.run(
        [
            "claude",
            "mcp",
            "add",
            "--scope",
            "user",
            CHANNEL_NAME,
            "--",
            resolved,
            "claude-channel",
        ],
        check=True,
    )


def mcp_config(command: str = "slackgentic") -> dict[str, Any]:
    return {
        "mcpServers": {
            CHANNEL_NAME: {
                "command": command,
                "args": ["claude-channel"],
            }
        }
    }


def _current_slackgentic_command() -> str:
    candidate = Path(sys.argv[0])
    if candidate.name == "slackgentic":
        try:
            return str(candidate.expanduser().resolve())
        except OSError:
            return str(candidate)
    found = shutil.which("slackgentic")
    if found:
        return found
    return str(candidate)


def _is_identifier(value: str) -> bool:
    return (
        bool(value)
        and (value[0].isalpha() or value[0] == "_")
        and all(char.isalnum() or char == "_" for char in value)
    )


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "request_user_input",
            "description": "Ask the Slack thread for a choice and wait for the response.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "header": {"type": "string"},
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["label"],
                                },
                            ]
                        },
                    },
                },
                "required": ["question", "options"],
            },
        },
        {
            "name": "request_approval",
            "description": "Ask the Slack thread to approve or deny an action.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["generic", "command", "file_change", "permissions"],
                    },
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                    "command": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                    "cwd": {"type": "string"},
                    "grantRoot": {"type": "string"},
                    "permissions": {"type": "object"},
                },
                "required": ["title"],
            },
        },
    ]


def _tool_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        result["isError"] = True
    return result


def _tool_options(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    options: list[dict[str, str]] = []
    for item in value[:5]:
        if isinstance(item, str) and item.strip():
            options.append({"label": item.strip()})
        elif isinstance(item, dict):
            label = _string_arg(item, "label")
            if not label:
                continue
            option = {"label": label}
            description = _string_arg(item, "description")
            if description:
                option["description"] = description
            options.append(option)
    return options


def _approval_request(arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    kind = _string_arg(arguments, "kind") or "generic"
    reason = _string_arg(arguments, "reason")
    if kind == "command":
        params: dict[str, Any] = {"command": arguments.get("command") or ""}
        cwd = _string_arg(arguments, "cwd")
        if cwd:
            params["cwd"] = cwd
        if reason:
            params["reason"] = reason
        return "item/commandExecution/requestApproval", params
    if kind == "file_change":
        params = {}
        grant_root = _string_arg(arguments, "grantRoot") or _string_arg(arguments, "grant_root")
        if grant_root:
            params["grantRoot"] = grant_root
        if reason:
            params["reason"] = reason
        return "item/fileChange/requestApproval", params
    if kind == "permissions":
        params = {"permissions": arguments.get("permissions") or {}}
        cwd = _string_arg(arguments, "cwd")
        if cwd:
            params["cwd"] = cwd
        if reason:
            params["reason"] = reason
        return "item/permissions/requestApproval", params
    params = {"title": _string_arg(arguments, "title") or "Approval requested"}
    if reason:
        params["reason"] = reason
    return "agent/requestApproval", params


def _selected_answer(response: object, question_id: str) -> str | None:
    if not isinstance(response, dict):
        return None
    answers = response.get("answers")
    if not isinstance(answers, dict):
        return None
    selected = answers.get(question_id)
    if not isinstance(selected, dict):
        return None
    values = selected.get("answers")
    if not isinstance(values, list) or not values:
        return None
    return str(values[0])


def _string_arg(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""
