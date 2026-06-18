from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from agent_harness.config import load_config_from_env
from agent_harness.internal_notifications import is_internal_task_notification_text
from agent_harness.models import PermissionMode, Provider, SlackThreadRef
from agent_harness.permissions import (
    claude_channel_permission_mode_from_env,
    claude_safe_auto_permission_request_allowed,
)
from agent_harness.sessions.native_input import (
    claude_ask_user_question_answers,
    claude_ask_user_question_tool_result_text,
    claude_ask_user_question_updated_input,
    claude_native_input_setting_key,
    slack_request_params_for_claude_ask_user_question,
)
from agent_harness.slack import parse_thread_ref, replace_slack_user_ids
from agent_harness.slack.agent_requests import SlackAgentRequestHandler
from agent_harness.storage.store import Store

CHANNEL_NAME = "slackgentic"
CHANNEL_VERSION = "0.1.0"
SLACK_THREAD_CHANNEL_ENV = "SLACKGENTIC_CLAUDE_CHANNEL_ID"
SLACK_THREAD_TS_ENV = "SLACKGENTIC_CLAUDE_THREAD_TS"
DANGEROUS_MODE_ENV = "SLACKGENTIC_CLAUDE_DANGEROUS_MODE"
SLACK_CHANNEL_SETTING = "slack.channel_id"
CHANNEL_INSTRUCTIONS = (
    "Slackgentic forwards Slack thread replies into this Claude Code session as "
    '<channel source="slackgentic" ...> events. Treat only the body of each event '
    "as the user's Slack message for this existing session. Never quote, repeat, "
    "or discuss the channel tags or metadata. Respond normally in the terminal; "
    "Slackgentic mirrors your visible assistant responses back to the Slack thread. "
    "When a table is the clearest format, write one normal Markdown table in the "
    "message; Slackgentic renders one Markdown table per message as a native Slack "
    "table. If you need multiple tables, send separate messages. "
    "When you need Slack approval or a Slack choice, use the Slackgentic MCP tools "
    "`request_approval` or `request_user_input`; do not use Claude's built-in "
    "`AskUserQuestion` for Slack-mediated approvals or choices. Use "
    "`request_user_input` when Slack should choose among multiple options; use "
    "`request_approval` only for one concrete yes/no approval. Before a large "
    "file edit that may require Slack approval, summarize the intended change "
    "first because Claude may expose only a truncated native tool input preview. "
    "When opening a GitHub pull request, prefer the Slackgentic `create_pull_request` "
    "MCP tool; it runs the narrow `gh pr create` workflow through the channel so "
    "PR creation still works when Claude Bash is sandboxed. When you need the "
    "contents of another Slackgentic thread from a Slack link, use the "
    "`read_thread` MCP tool; it only reads links from the configured Slackgentic "
    "channel."
)
CODEX_MCP_INSTRUCTIONS = (
    "Slackgentic provides MCP tools for Slack-mediated workflows. When opening a "
    "GitHub pull request, prefer the `create_pull_request` tool; it runs the narrow "
    "`gh pr create` workflow through Slackgentic so PR creation still works when "
    "ordinary shell networking or sandbox policy gets in the way. When you need the "
    "contents of another Slackgentic thread from a Slack link, use `read_thread`; it "
    "only reads links from the configured Slackgentic channel."
)
SLACKGENTIC_MCP_TOOL_NAMES = {
    f"mcp__{CHANNEL_NAME}__create_pull_request",
    f"mcp__{CHANNEL_NAME}__read_thread",
    f"mcp__{CHANNEL_NAME}__request_approval",
    f"mcp__{CHANNEL_NAME}__request_user_input",
}
SLACKGENTIC_MCP_PERMISSION_ALLOW = tuple(sorted(SLACKGENTIC_MCP_TOOL_NAMES))
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
PR_URL_RE = re.compile(r"https://github\.com/[^\s>/]+/[^\s>/]+/pull/\d+[^\s>]*")
CREATE_PULL_REQUEST_TIMEOUT_SECONDS = 120
NATIVE_INPUT_HOOK_MARKER = "slackgentic.native_input.v1"


class PullRequestHeadPreparation(NamedTuple):
    head: str | None
    error: str | None


class ClaudeChannelServer:
    def __init__(
        self,
        store: Store,
        target_pid: int | None = None,
        poll_seconds: float = 0.2,
        request_handler: SlackAgentRequestHandler | None = None,
        gateway: Any | None = None,
        slack_channel_id: str | None = None,
        command_runner: CommandRunner | None = None,
        instructions: str = CHANNEL_INSTRUCTIONS,
    ):
        self.store = store
        self.target_pid = target_pid or os.getppid()
        self.poll_seconds = poll_seconds
        self.request_handler = request_handler
        self.gateway = gateway or getattr(request_handler, "gateway", None)
        self.slack_channel_id = _configured_slack_channel_id(store, slack_channel_id)
        self.command_runner = command_runner or subprocess.run
        self.instructions = instructions
        self._current_thread = _thread_from_env()
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
                            "tools": {},
                            "experimental": {
                                "claude/channel": {},
                                "claude/channel/permission": {},
                            },
                        },
                        "serverInfo": {
                            "name": CHANNEL_NAME,
                            "version": CHANNEL_VERSION,
                        },
                        "instructions": self.instructions,
                    },
                }
            )
            self._ready.set()
            return
        if method == "notifications/initialized":
            self._ready.set()
            return
        if method == "notifications/claude/channel/permission_request":
            params = message.get("params") if isinstance(message.get("params"), dict) else {}
            self._handle_permission_request_notification(params)
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
        if name == "create_pull_request":
            return self._handle_create_pull_request_tool(arguments)
        if name == "read_thread":
            return self._handle_read_thread_tool(arguments)
        return _tool_result(f"Unknown Slackgentic tool: {name}", is_error=True)

    def _handle_read_thread_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.gateway is None:
            return _tool_result("Slack thread access is not configured.", is_error=True)
        link = _string_arg(arguments, "url") or _string_arg(arguments, "link")
        if not link:
            return _tool_result("read_thread requires a Slack thread URL.", is_error=True)
        ref = parse_thread_ref(link)
        if ref is None:
            return _tool_result(
                "read_thread requires a valid Slack thread permalink.", is_error=True
            )
        allowed_channel = self._allowed_slack_channel_id()
        if not allowed_channel:
            return _tool_result(
                "read_thread requires a configured Slackgentic channel.",
                is_error=True,
            )
        if ref.channel_id != allowed_channel:
            return _tool_result(
                "read_thread can only read links from the configured Slackgentic channel.",
                is_error=True,
            )
        limit = _int_arg(arguments, "limit", default=40, minimum=1, maximum=200)
        try:
            messages = self.gateway.thread_messages(ref.channel_id, ref.thread_ts, limit=limit)
        except Exception as exc:
            return _tool_result(
                f"Failed to read Slack thread: {_exception_summary(exc)}",
                is_error=True,
            )
        if not messages:
            return _tool_result("No messages were found for that Slack thread.")
        transcript = _slack_thread_transcript(messages)
        if not transcript:
            return _tool_result(
                "The Slack thread was fetched, but it did not contain readable text messages."
            )
        count = len(messages)
        limit_note = (
            " The result may be truncated by the requested limit." if count >= limit else ""
        )
        return _tool_result(
            f"Slack thread transcript ({count} message{'s' if count != 1 else ''})."
            f"{limit_note}\n\n{transcript}"
        )

    def _allowed_slack_channel_id(self) -> str:
        if self.slack_channel_id:
            return self.slack_channel_id
        if self._current_thread is not None:
            return self._current_thread.channel_id
        return ""

    def _handle_create_pull_request_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        title = _string_arg(arguments, "title")
        if not title:
            return _tool_result("create_pull_request requires a title.", is_error=True)
        cwd = _cwd_arg(arguments)
        if cwd is None:
            return _tool_result(
                "create_pull_request cwd must be an existing directory.", is_error=True
            )
        try:
            head_preparation = _prepare_pull_request_head(
                arguments,
                cwd,
                self.command_runner,
            )
        except FileNotFoundError:
            return _tool_result("Git CLI `git` was not found on PATH.", is_error=True)
        except subprocess.TimeoutExpired as exc:
            return _tool_result(_timeout_message(exc), is_error=True)
        except Exception as exc:
            return _tool_result(f"Failed to prepare pull request branch: {exc}", is_error=True)
        if head_preparation.error:
            return _tool_result(head_preparation.error, is_error=True)

        args = _pull_request_create_args(arguments, title, default_head=head_preparation.head)
        try:
            completed = self.command_runner(
                args,
                cwd=cwd,
                text=True,
                capture_output=True,
                check=False,
                timeout=CREATE_PULL_REQUEST_TIMEOUT_SECONDS,
            )
        except FileNotFoundError:
            return _tool_result("GitHub CLI `gh` was not found on PATH.", is_error=True)
        except subprocess.TimeoutExpired as exc:
            return _tool_result(_timeout_message(exc), is_error=True)
        except Exception as exc:
            return _tool_result(f"Failed to create pull request: {exc}", is_error=True)

        output = _command_output(completed)
        url = _github_pr_url(output)
        if completed.returncode == 0:
            return _tool_result(url or output or "Pull request created.")
        if url and "already exists" in output.lower():
            return _tool_result(url)
        detail = output or f"`gh pr create` exited with {completed.returncode}."
        if url:
            detail = f"{detail}\nExisting pull request: {url}"
        return _tool_result(detail, is_error=True)

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

    def _handle_permission_request_notification(self, params: dict[str, Any]) -> None:
        request_id = params.get("request_id")
        if request_id is None:
            return
        worker = threading.Thread(
            target=self._handle_permission_request_worker,
            args=(str(request_id), dict(params)),
            daemon=True,
            name=f"slackgentic-claude-permission-{request_id}",
        )
        worker.start()

    def _handle_permission_request_worker(
        self,
        request_id: str,
        params: dict[str, Any],
    ) -> None:
        behavior = "deny"
        if _is_slackgentic_mcp_tool(_string_param(params, "tool_name")):
            self._write(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/claude/channel/permission",
                    "params": {"request_id": request_id, "behavior": "allow"},
                }
            )
            return
        if _dangerous_mode_from_env():
            # Dangerous-mode tasks already run Claude with
            # --dangerously-skip-permissions; routing the channel-side
            # permission request through Slack approval anyway would silently
            # deny on any imperfect round-trip (the bug that motivated this
            # branch). Mirror the CLI behavior and allow.
            self._write(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/claude/channel/permission",
                    "params": {"request_id": request_id, "behavior": "allow"},
                }
            )
            return
        if (
            _permission_mode_from_env() == PermissionMode.SAFE_AUTO
            and claude_safe_auto_permission_request_allowed(params)
        ):
            self._write(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/claude/channel/permission",
                    "params": {"request_id": request_id, "behavior": "allow"},
                }
            )
            return
        if self.request_handler is not None and self._current_thread is not None:
            try:
                request_params = dict(params)
                request_params["request_id"] = request_id
                request_params["tool_name"] = _string_param(params, "tool_name")
                request_params["description"] = _string_param(params, "description")
                request_params["input_preview"] = _string_param(params, "input_preview")
                response = self.request_handler.handle_persistent_request(
                    "claude/channel/permission",
                    request_params,
                    self._current_thread,
                    provider_label="Claude",
                )
            except Exception:
                response = None
            if isinstance(response, dict) and response.get("behavior") == "allow":
                behavior = "allow"
        self._write(
            {
                "jsonrpc": "2.0",
                "method": "notifications/claude/channel/permission",
                "params": {"request_id": request_id, "behavior": behavior},
            }
        )

    def _write(self, message: dict[str, Any]) -> None:
        with self._write_lock:
            sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def run_channel_server(db_path: Path | None = None, *, provider_label: str = "Claude") -> int:
    config = load_config_from_env()
    store = Store(db_path or config.state_db)
    try:
        store.init_schema()
        request_handler = None
        gateway = None
        if config.slack.bot_token:
            from agent_harness.slack.client import SlackGateway

            gateway = SlackGateway(config.slack.bot_token)
            request_handler = SlackAgentRequestHandler(
                gateway,
                store=store,
                provider_label=provider_label,
            )
        instructions = CODEX_MCP_INSTRUCTIONS if provider_label == "Codex" else CHANNEL_INSTRUCTIONS
        return ClaudeChannelServer(
            store,
            request_handler=request_handler,
            gateway=gateway,
            slack_channel_id=config.slack.channel_id,
            instructions=instructions,
        ).run()
    finally:
        store.close()


def run_native_input_hook(db_path: Path | None = None) -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0
    config = load_config_from_env()
    if not config.slack.bot_token:
        return 0
    store = Store(db_path or config.state_db)
    try:
        store.init_schema()
        from agent_harness.slack.client import SlackGateway

        result = handle_native_input_hook(
            payload,
            store,
            SlackGateway(config.slack.bot_token),
            poll_seconds=0.2,
        )
    finally:
        store.close()
    if result:
        sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


def handle_native_input_hook(
    payload: dict[str, Any],
    store: Store,
    gateway: Any,
    *,
    poll_seconds: float = 0.05,
) -> dict[str, Any] | None:
    if payload.get("hook_event_name") != "PreToolUse":
        return None
    if payload.get("tool_name") != "AskUserQuestion":
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    params = slack_request_params_for_claude_ask_user_question(tool_input)
    if params is None:
        return None
    session_id = str(payload.get("session_id") or "")
    tool_use_id = str(payload.get("tool_use_id") or "")
    thread = _native_input_thread(payload, store)
    if thread is None:
        return None
    if session_id and tool_use_id:
        store.set_setting(
            claude_native_input_setting_key(session_id, tool_use_id),
            json.dumps({"thread": thread.thread_ts}, sort_keys=True),
        )
    handler = SlackAgentRequestHandler(
        gateway,
        store=store,
        provider_label="Claude",
        poll_seconds=poll_seconds,
    )
    response = handler.handle_persistent_request(
        "item/tool/requestUserInput",
        params,
        thread,
        provider_label="Claude",
    )
    updated_input = claude_ask_user_question_updated_input(tool_input, response)
    if updated_input is None:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
            },
            "systemMessage": "Slackgentic did not receive an answer for this question.",
        }
    answers = claude_ask_user_question_answers(tool_input, response) or {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated_input,
        },
        "systemMessage": claude_ask_user_question_tool_result_text(tool_input, answers),
    }


def _native_input_thread(payload: dict[str, Any], store: Store) -> SlackThreadRef | None:
    env_thread = _thread_from_env()
    if env_thread is not None:
        return env_thread
    session_id = str(payload.get("session_id") or "")
    if not session_id:
        return None
    active_task = store.get_active_task_by_session(Provider.CLAUDE, session_id)
    if active_task is not None and active_task.channel_id and active_task.thread_ts:
        return SlackThreadRef(active_task.channel_id, active_task.thread_ts)
    return store.find_slack_thread_for_session(Provider.CLAUDE, session_id)


def install_claude_mcp_server(command: str | None = None, home: Path | None = None) -> None:
    if command is None:
        resolved, command_args = _current_slackgentic_invocation()
    else:
        resolved, command_args = command, []
    add_command = [
        "claude",
        "mcp",
        "add",
        "--scope",
        "user",
        CHANNEL_NAME,
        "--",
        resolved,
        *command_args,
        "claude-channel",
    ]
    completed = subprocess.run(add_command, check=False, capture_output=True, text=True)
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
    if completed.returncode != 0:
        if "already exists" not in output.lower():
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        subprocess.run(
            ["claude", "mcp", "remove", "--scope", "user", CHANNEL_NAME],
            check=False,
            capture_output=True,
            text=True,
        )
        completed = subprocess.run(add_command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
    ensure_claude_mcp_permissions(home)
    ensure_claude_native_input_hook(resolved, home, args=command_args)


def install_codex_mcp_server(
    command: str | None = None,
    home: Path | None = None,
    *,
    codex_binary: str = "codex",
) -> None:
    if command is None:
        resolved, command_args = _current_slackgentic_invocation()
    else:
        resolved, command_args = command, []
    add_command = [
        codex_binary,
        "mcp",
        "add",
        CHANNEL_NAME,
        "--",
        resolved,
        *command_args,
        "codex-mcp",
    ]
    env = _codex_mcp_env(home)
    completed = subprocess.run(
        add_command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
    if completed.returncode == 0:
        return
    if "already exists" not in output.lower():
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    subprocess.run(
        [codex_binary, "mcp", "remove", CHANNEL_NAME],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    completed = subprocess.run(
        add_command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )


def ensure_codex_mcp_server_registered(
    command: str | None = None,
    home: Path | None = None,
    *,
    codex_binary: str = "codex",
) -> bool:
    if is_codex_mcp_server_configured(home):
        return False
    if shutil.which(codex_binary) is None:
        return False
    install_codex_mcp_server(command, home, codex_binary=codex_binary)
    return True


def _codex_mcp_env(home: Path | None) -> dict[str, str] | None:
    if home is None:
        return None
    env = os.environ.copy()
    env["CODEX_HOME"] = str(home / ".codex")
    return env


def ensure_claude_mcp_permissions(home: Path | None = None) -> Path:
    home = home or Path.home()
    path = home / ".claude" / "settings.local.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    permissions = value.get("permissions")
    if not isinstance(permissions, dict):
        permissions = {}
        value["permissions"] = permissions
    allow = permissions.get("allow")
    if not isinstance(allow, list):
        allow = []
        permissions["allow"] = allow
    existing = {item for item in allow if isinstance(item, str)}
    changed = False
    for permission in SLACKGENTIC_MCP_PERMISSION_ALLOW:
        if permission not in existing:
            allow.append(permission)
            changed = True
    if changed or not path.exists():
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def ensure_claude_native_input_hook(
    command: str | None = None,
    home: Path | None = None,
    *,
    args: list[str] | None = None,
) -> Path:
    home = home or Path.home()
    if command is None:
        command, invocation_args = _current_slackgentic_invocation()
    else:
        invocation_args = list(args or [])
    local_path = home / ".claude" / "settings.local.json"
    user_path = home / ".claude" / "settings.json"
    _ensure_claude_native_input_hook_at_path(local_path, command, invocation_args)
    if user_path.exists():
        _ensure_claude_native_input_hook_at_path(user_path, command, invocation_args)
    return local_path


def _ensure_claude_native_input_hook_at_path(
    path: Path,
    command: str,
    invocation_args: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    hooks = value.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
        value["hooks"] = hooks
    pre_tool_use = hooks.get("PreToolUse")
    if not isinstance(pre_tool_use, list):
        pre_tool_use = []
        hooks["PreToolUse"] = pre_tool_use
    hook_command = shlex.join([command, *invocation_args, "claude-channel", "--native-input-hook"])
    entry = {
        "matcher": "AskUserQuestion",
        "hooks": [
            {
                "type": "command",
                "command": hook_command,
                "_slackgentic": NATIVE_INPUT_HOOK_MARKER,
            }
        ],
    }
    first_wildcard_index = _first_wildcard_hook_index(pre_tool_use)
    for index, candidate in enumerate(pre_tool_use):
        if not isinstance(candidate, dict):
            continue
        candidate_hooks = candidate.get("hooks")
        if not isinstance(candidate_hooks, list):
            continue
        if any(
            isinstance(hook, dict) and hook.get("_slackgentic") == NATIVE_INPUT_HOOK_MARKER
            for hook in candidate_hooks
        ):
            pre_tool_use.pop(index)
            insert_index = _first_wildcard_hook_index(pre_tool_use)
            pre_tool_use.insert(insert_index, entry)
            break
    else:
        pre_tool_use.insert(first_wildcard_index, entry)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _first_wildcard_hook_index(pre_tool_use: list[object]) -> int:
    for index, candidate in enumerate(pre_tool_use):
        if not isinstance(candidate, dict):
            continue
        matcher = candidate.get("matcher")
        if matcher is None or matcher == "*":
            return index
    return len(pre_tool_use)


def mcp_config(command: str = "slackgentic", args: list[str] | None = None) -> dict[str, Any]:
    return {
        "mcpServers": {
            CHANNEL_NAME: {
                "command": command,
                "args": [*(args or []), "claude-channel"],
            }
        }
    }


def is_slackgentic_mcp_server_configured(home: Path | None = None) -> bool:
    home = home or Path.home()
    for path in (
        home / ".claude.json",
        home / ".claude" / "settings.json",
        home / ".claude" / "settings.local.json",
    ):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _contains_slackgentic_mcp_config(value):
            return True
    return False


def is_codex_mcp_server_configured(home: Path | None = None) -> bool:
    home = home or Path.home()
    try:
        with (home / ".codex" / "config.toml").open("rb") as config:
            value = tomllib.load(config)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    servers = value.get("mcp_servers")
    if not isinstance(servers, dict):
        return False
    return _valid_mcp_server_config(servers.get(CHANNEL_NAME))


def session_transcript_has_slackgentic_mcp(
    transcript_path: Path,
    *,
    max_records: int = 80,
) -> bool:
    from agent_harness.storage.jsonl import iter_jsonl

    try:
        for index, (_, record) in enumerate(iter_jsonl(transcript_path), start=1):
            if _record_has_slackgentic_mcp(record):
                return True
            if index >= max_records:
                break
    except OSError:
        return False
    return False


def claude_session_has_slackgentic_mcp(session: object) -> bool:
    transcript_path = getattr(session, "transcript_path", None)
    if isinstance(transcript_path, Path):
        return session_transcript_has_slackgentic_mcp(transcript_path)
    return False


def _current_slackgentic_command() -> str:
    command, _ = _current_slackgentic_invocation()
    return command


def _current_slackgentic_invocation() -> tuple[str, list[str]]:
    candidate = Path(sys.argv[0])
    if candidate.name == "slackgentic":
        try:
            return str(candidate.expanduser().resolve()), []
        except OSError:
            return str(candidate), []
    found = shutil.which("slackgentic")
    if found:
        return found, []
    if candidate.name == "__main__.py" and candidate.parent.name == "agent_harness":
        return sys.executable, ["-m", "agent_harness"]
    try:
        resolved = candidate.expanduser().resolve()
    except OSError:
        resolved = candidate
    if os.access(resolved, os.X_OK):
        return str(resolved), []
    return sys.executable, [str(resolved)]


def _thread_from_env() -> SlackThreadRef | None:
    channel_id = os.environ.get(SLACK_THREAD_CHANNEL_ENV)
    thread_ts = os.environ.get(SLACK_THREAD_TS_ENV)
    if channel_id and thread_ts:
        return SlackThreadRef(channel_id, thread_ts)
    return None


def _configured_slack_channel_id(store: Store, configured: str | None) -> str:
    if configured and configured.strip():
        return configured.strip()
    try:
        stored = store.get_setting(SLACK_CHANNEL_SETTING)
    except Exception:
        return ""
    return stored.strip() if stored and stored.strip() else ""


def _dangerous_mode_from_env() -> bool:
    raw = os.environ.get(DANGEROUS_MODE_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _permission_mode_from_env() -> PermissionMode | None:
    return claude_channel_permission_mode_from_env(os.environ)


def _is_identifier(value: str) -> bool:
    return (
        bool(value)
        and (value[0].isalpha() or value[0] == "_")
        and all(char.isalnum() or char == "_" for char in value)
    )


def _contains_slackgentic_mcp_config(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    servers = value.get("mcpServers")
    if isinstance(servers, dict) and _valid_mcp_server_config(servers.get(CHANNEL_NAME)):
        return True
    return any(_contains_slackgentic_mcp_config(child) for child in value.values())


def _valid_mcp_server_config(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    command = value.get("command")
    args = value.get("args")
    return bool((isinstance(command, str) and command.strip()) or isinstance(args, list))


def _record_has_slackgentic_mcp(record: dict[str, Any]) -> bool:
    candidates: list[object] = [record.get("attachment")]
    attachments = record.get("attachments")
    if isinstance(attachments, list):
        candidates.extend(attachments)
    return any(_attachment_has_slackgentic_mcp(candidate) for candidate in candidates)


def _attachment_has_slackgentic_mcp(attachment: object) -> bool:
    if not isinstance(attachment, dict):
        return False
    added_names = {
        str(name) for name in attachment.get("addedNames") or [] if isinstance(name, str)
    }
    attachment_type = attachment.get("type")
    if attachment_type == "mcp_instructions_delta":
        return CHANNEL_NAME in added_names
    if attachment_type == "deferred_tools_delta":
        return bool(SLACKGENTIC_MCP_TOOL_NAMES.intersection(added_names))
    return False


def _is_slackgentic_mcp_tool(tool_name: str) -> bool:
    return tool_name in SLACKGENTIC_MCP_TOOL_NAMES


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "create_pull_request",
            "description": (
                "Create a GitHub pull request for a local or pushed branch using `gh pr create`. "
                "Use this when a PR is ready or when Bash cannot reach GitHub from a sandbox."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "base": {"type": "string"},
                    "head": {"type": "string"},
                    "repo": {
                        "type": "string",
                        "description": "Optional owner/repo target for gh.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional local repository directory.",
                    },
                    "draft": {"type": "boolean"},
                },
                "required": ["title"],
            },
        },
        {
            "name": "read_thread",
            "description": (
                "Read messages from a Slack thread permalink in the configured "
                "Slackgentic channel. Use this when an agent needs context from a "
                "linked Slackgentic thread."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Slack thread permalink to read.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Maximum messages to fetch. Defaults to 40.",
                    },
                },
                "required": ["url"],
            },
        },
        {
            "name": "request_user_input",
            "description": (
                "Ask the Slack thread to choose among multiple options and wait for the response."
            ),
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
            "description": (
                "Ask the Slack thread to approve or deny one concrete action. "
                "Do not use this for choosing among options."
            ),
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


def _prepare_pull_request_head(
    arguments: dict[str, Any],
    cwd: Path,
    command_runner: CommandRunner,
) -> PullRequestHeadPreparation:
    requested_head = _string_arg(arguments, "head")
    if ":" in requested_head:
        return PullRequestHeadPreparation(None, None)
    if not _is_git_work_tree(cwd, command_runner):
        return PullRequestHeadPreparation(None, None)

    branch = requested_head
    if branch:
        if not _local_branch_exists(cwd, branch, command_runner):
            return PullRequestHeadPreparation(None, None)
    else:
        branch = _current_git_branch(cwd, command_runner)
        if not branch:
            return PullRequestHeadPreparation(
                None,
                "create_pull_request could not determine the current branch. "
                "Pass head explicitly when running outside a branch.",
            )

    base = _string_arg(arguments, "base")
    if branch == base or (not base and branch in {"main", "master"}):
        return PullRequestHeadPreparation(None, None)

    remote, remote_branch, error = _push_target_for_branch(cwd, branch, command_runner)
    if error:
        return PullRequestHeadPreparation(None, error)

    completed = command_runner(
        [
            "git",
            "push",
            "--set-upstream",
            remote,
            f"refs/heads/{branch}:refs/heads/{remote_branch}",
        ],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=CREATE_PULL_REQUEST_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        detail = _command_output(completed) or f"`git push` exited with {completed.returncode}."
        return PullRequestHeadPreparation(
            None,
            f"Failed to push branch `{branch}` before creating pull request:\n{detail}",
        )
    return PullRequestHeadPreparation(remote_branch, None)


def _push_target_for_branch(
    cwd: Path,
    branch: str,
    command_runner: CommandRunner,
) -> tuple[str, str, str | None]:
    remotes = _git_remotes(cwd, command_runner)
    if not remotes:
        return "", "", "create_pull_request could not find a git remote to push the branch."

    upstream = command_runner(
        [
            "git",
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            f"{branch}@{{upstream}}",
        ],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=CREATE_PULL_REQUEST_TIMEOUT_SECONDS,
    )
    if upstream.returncode == 0:
        parsed = _parse_upstream_ref(upstream.stdout.strip(), remotes)
        if parsed is not None:
            return parsed[0], parsed[1], None

    remote = "origin" if "origin" in remotes else remotes[0]
    return remote, branch, None


def _is_git_work_tree(cwd: Path, command_runner: CommandRunner) -> bool:
    completed = command_runner(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=CREATE_PULL_REQUEST_TIMEOUT_SECONDS,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _local_branch_exists(cwd: Path, branch: str, command_runner: CommandRunner) -> bool:
    completed = command_runner(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=CREATE_PULL_REQUEST_TIMEOUT_SECONDS,
    )
    return completed.returncode == 0


def _current_git_branch(cwd: Path, command_runner: CommandRunner) -> str:
    completed = command_runner(
        ["git", "branch", "--show-current"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=CREATE_PULL_REQUEST_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _git_remotes(cwd: Path, command_runner: CommandRunner) -> list[str]:
    completed = command_runner(
        ["git", "remote"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=CREATE_PULL_REQUEST_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        return []
    return [remote.strip() for remote in completed.stdout.splitlines() if remote.strip()]


def _parse_upstream_ref(upstream: str, remotes: list[str]) -> tuple[str, str] | None:
    for remote in sorted(remotes, key=len, reverse=True):
        prefix = f"{remote}/"
        if upstream.startswith(prefix) and len(upstream) > len(prefix):
            return remote, upstream[len(prefix) :]
    return None


def _pull_request_create_args(
    arguments: dict[str, Any],
    title: str,
    *,
    default_head: str | None = None,
) -> list[str]:
    args = ["gh", "pr", "create", "--title", title, "--body", _string_arg(arguments, "body")]
    base = _string_arg(arguments, "base")
    if base:
        args.extend(["--base", base])
    head = _string_arg(arguments, "head") or default_head
    if head:
        args.extend(["--head", head])
    repo = _string_arg(arguments, "repo")
    if repo:
        args.extend(["--repo", repo])
    if _bool_arg(arguments, "draft"):
        args.append("--draft")
    return args


def _cwd_arg(arguments: dict[str, Any]) -> Path | None:
    raw = _string_arg(arguments, "cwd")
    candidate = Path(raw).expanduser() if raw else Path.cwd()
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.is_dir():
        return None
    return resolved


def _command_output(completed: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        value.strip() for value in (completed.stdout, completed.stderr) if value and value.strip()
    )


def _github_pr_url(text: str) -> str:
    match = PR_URL_RE.search(text)
    return match.group(0) if match else ""


def _timeout_message(exc: subprocess.TimeoutExpired) -> str:
    command = exc.cmd[0] if isinstance(exc.cmd, list) and exc.cmd else exc.cmd
    if isinstance(command, str) and command:
        label = command
        if isinstance(exc.cmd, list):
            if exc.cmd[:3] == ["gh", "pr", "create"]:
                label = "gh pr create"
            elif exc.cmd[:2] == ["git", "push"]:
                label = "git push"
        return f"`{label}` timed out after {CREATE_PULL_REQUEST_TIMEOUT_SECONDS} seconds."
    return f"Command timed out after {CREATE_PULL_REQUEST_TIMEOUT_SECONDS} seconds."


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


def _bool_arg(arguments: dict[str, Any], key: str) -> bool:
    return arguments.get(key) is True


def _int_arg(
    arguments: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = arguments.get(key)
    if isinstance(raw, bool):
        value = default
    elif isinstance(raw, int):
        value = raw
    elif isinstance(raw, str):
        try:
            value = int(raw.strip())
        except ValueError:
            value = default
    else:
        value = default
    return max(minimum, min(maximum, value))


def _string_param(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _slack_thread_transcript(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        text = replace_slack_user_ids(str(message.get("text") or "")).strip()
        if not text or is_internal_task_notification_text(text):
            continue
        author = _slack_message_author(message)
        timestamp = str(message.get("ts") or "").strip()
        prefix = f"[{timestamp}] {author}" if timestamp else author
        lines.append(f"{prefix}: {text.replace(chr(10), chr(10) + '    ')}")
    return "\n".join(lines)[-20000:]


def _slack_message_author(message: dict[str, Any]) -> str:
    username = _stringish(message.get("username"))
    if username:
        return username
    for key in ("user_profile", "bot_profile"):
        profile_name = _profile_display_name(message.get(key))
        if profile_name:
            return profile_name
    if _stringish(message.get("user")):
        return "Slack user"
    return "Slack"


def _profile_display_name(profile: object) -> str | None:
    if not isinstance(profile, dict):
        return None
    for key in ("display_name", "real_name", "name", "username"):
        value = _stringish(profile.get(key))
        if value:
            return value
    return None


def _stringish(value: object) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _exception_summary(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message[:500]
    return exc.__class__.__name__
