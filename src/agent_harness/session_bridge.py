from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Protocol

from agent_harness.codex_app_server import (
    CodexAppServerClient,
)
from agent_harness.config import AgentCommandConfig
from agent_harness.models import AgentSession, Provider, SessionStatus, SlackThreadRef, utc_now
from agent_harness.runner import _codex_trust_override
from agent_harness.session_terminal import SessionTerminalNotifier, TerminalTarget
from agent_harness.slack import dangerous_flag
from agent_harness.slack_agent_requests import SlackAgentRequestHandler
from agent_harness.slack_client import SlackGateway
from agent_harness.store import Store
from agent_harness.task_runtime import _process_output_chunks

LOGGER = logging.getLogger(__name__)
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ProcessKiller = Callable[[int, int], None]
SESSION_EXIT_COMMAND = "/exit"


class CodexLiveClient(Protocol):
    def send_to_thread(self, thread_id: str, text: str, cwd: Path | None = None) -> bool: ...


@dataclass(frozen=True)
class BridgeSend:
    session: AgentSession
    prompt: str
    slack_text: str
    thread: SlackThreadRef


class ExternalSessionBridge:
    def __init__(
        self,
        store: Store,
        gateway: SlackGateway,
        commands: AgentCommandConfig,
        command_runner: CommandRunner | None = None,
        terminal_notifier: SessionTerminalNotifier | None = None,
        timeout_seconds: int = 1800,
        channel_delivery_timeout: float = 2.0,
        codex_app_server_url: str | None = None,
        codex_live_client: CodexLiveClient | None = None,
        agent_request_handler: SlackAgentRequestHandler | None = None,
        process_killer: ProcessKiller | None = None,
    ):
        self.store = store
        self.gateway = gateway
        self.commands = commands
        self.command_runner = command_runner or subprocess.run
        self.terminal_notifier = terminal_notifier or SessionTerminalNotifier()
        self.timeout_seconds = timeout_seconds
        self.channel_delivery_timeout = channel_delivery_timeout
        self.codex_app_server_url = codex_app_server_url or commands.codex_app_server_url
        self.codex_live_client = codex_live_client
        self.agent_request_handler = agent_request_handler or SlackAgentRequestHandler(
            gateway,
            store=store,
            provider_label="Codex",
        )
        self.process_killer = process_killer or os.kill
        self._locks: dict[tuple[Provider, str], threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def send_to_session(
        self,
        session: AgentSession,
        text: str,
        thread: SlackThreadRef,
        slack_user: str | None = None,
    ) -> bool:
        prompt = build_external_session_prompt(text, slack_user)
        if not prompt.strip():
            return False
        if is_session_exit_request(prompt):
            if self._send_exit_to_live_session(session, prompt, thread, slack_user):
                return True
            return self._terminate_live_session(session, thread)
        self.store.add_session_bridge_prompt(session.provider, session.session_id, prompt)
        if self._send_to_codex_app_server(session, prompt, thread):
            return True
        if self._send_to_claude_channel(session, text, thread, slack_user):
            return True
        send = BridgeSend(
            session=session,
            prompt=prompt,
            slack_text=text,
            thread=thread,
        )
        worker = threading.Thread(
            target=self._run_send,
            args=(send,),
            daemon=True,
            name=f"slackgentic-bridge-{session.provider.value}-{session.session_id}",
        )
        worker.start()
        return True

    def handle_agent_request_block_action(
        self,
        payload: dict[str, object],
        channel_id: str,
        message_ts: str | None,
    ) -> bool:
        return self.agent_request_handler.handle_block_action(payload, channel_id, message_ts)

    def handle_codex_block_action(
        self,
        payload: dict[str, object],
        channel_id: str,
        message_ts: str | None,
    ) -> bool:
        return self.handle_agent_request_block_action(payload, channel_id, message_ts)

    def _run_send(self, send: BridgeSend) -> None:
        lock = self._lock_for(send.session.provider, send.session.session_id)
        with lock:
            self.terminal_notifier.notify_user_message(send.session, send.slack_text)
            try:
                completed = self.command_runner(
                    _resume_command(send.session, send.prompt, self.commands),
                    cwd=_command_cwd(send.session),
                    text=True,
                    capture_output=True,
                    input="",
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except Exception as exc:
                LOGGER.exception("failed to resume external %s session", send.session.provider)
                self.gateway.post_thread_reply(
                    send.thread,
                    (
                        f"Failed to send that to the external "
                        f"{send.session.provider.value} session: {exc}"
                    ),
                )
                self.terminal_notifier.notify_agent_response(send.session, str(exc))
                return
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                message = (
                    f"External {send.session.provider.value} session returned exit code "
                    f"{completed.returncode}."
                )
                if detail:
                    message = f"{message}\n{detail[-1200:]}"
                self.gateway.post_thread_reply(send.thread, message)
                self.terminal_notifier.notify_agent_response(send.session, message)
                return
            response = _render_resume_output(send.session.provider, completed.stdout)
            if response:
                self.terminal_notifier.notify_agent_response(send.session, response)

    def _lock_for(self, provider: Provider, session_id: str) -> threading.Lock:
        key = (provider, session_id)
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def _send_to_codex_app_server(
        self,
        session: AgentSession,
        prompt: str,
        thread: SlackThreadRef | None = None,
    ) -> bool:
        if session.provider != Provider.CODEX:
            return False
        url = self.codex_app_server_url
        if not url or url.lower() == "off":
            return False
        if self.codex_live_client is not None:
            client = self.codex_live_client
        else:
            server_request_handler = None
            if thread is not None:

                def server_request_handler(request):
                    return self.agent_request_handler.handle_server_request(
                        request,
                        thread,
                        provider_label="Codex",
                    )

            client = CodexAppServerClient(url, server_request_handler=server_request_handler)
        try:
            return client.send_to_thread(session.session_id, prompt, _command_cwd(session))
        except Exception:
            LOGGER.debug("failed to send Codex prompt through app-server", exc_info=True)
            return False

    def _send_to_claude_channel(
        self,
        session: AgentSession,
        text: str,
        thread: SlackThreadRef,
        slack_user: str | None,
    ) -> bool:
        if session.provider != Provider.CLAUDE:
            return False
        target = self._channel_target_for_session(session)
        if target is None:
            return False
        message_id = self.store.enqueue_claude_channel_message(
            target.pid,
            session.session_id,
            text,
            {
                "slack_channel": thread.channel_id,
                "slack_thread_ts": thread.thread_ts,
                "slack_user": slack_user or "",
                "session_id": session.session_id,
            },
        )
        deadline = time.monotonic() + self.channel_delivery_timeout
        while time.monotonic() < deadline:
            if self.store.is_claude_channel_message_delivered(message_id):
                return True
            time.sleep(0.05)
        self.store.cancel_claude_channel_message(message_id)
        return False

    def _channel_target_for_session(self, session: AgentSession) -> TerminalTarget | None:
        if session.started_at is None and not _is_latest_active_session_for_cwd(
            self.store,
            session,
        ):
            return None
        targets = self.terminal_notifier.targets_for_session(session)
        if len(targets) != 1:
            return None
        target = targets[0]
        if not _slackgentic_channel_enabled(target.command):
            return None
        if not _target_start_matches_session(session, target):
            return None
        return target

    def _send_exit_to_live_session(
        self,
        session: AgentSession,
        prompt: str,
        thread: SlackThreadRef,
        slack_user: str | None,
    ) -> bool:
        if session.provider == Provider.CODEX:
            if not self._send_to_codex_app_server(session, prompt, thread):
                return False
        elif session.provider == Provider.CLAUDE:
            if not self._send_to_claude_channel(session, prompt, thread, slack_user):
                return False
        else:
            return False
        self.store.add_session_bridge_prompt(session.provider, session.session_id, prompt)
        self.gateway.post_thread_reply(
            thread,
            f"Sent `{prompt}` to the live {session.provider.value} session.",
        )
        return True

    def _terminate_live_session(self, session: AgentSession, thread: SlackThreadRef) -> bool:
        targets = [
            target
            for target in self.terminal_notifier.targets_for_session(session)
            if _target_start_matches_session(session, target)
        ]
        if not targets:
            self.gateway.post_thread_reply(
                thread,
                (
                    f"I could not find a live {session.provider.value} process for this "
                    "session, so nothing was terminated."
                ),
            )
            return True
        if len(targets) > 1:
            self.gateway.post_thread_reply(
                thread,
                (
                    f"I found {len(targets)} matching {session.provider.value} processes "
                    "for this session and did not terminate any of them."
                ),
            )
            return True
        target = targets[0]
        try:
            self.process_killer(target.pid, signal.SIGTERM)
        except ProcessLookupError:
            self.gateway.post_thread_reply(
                thread,
                f"That {session.provider.value} session process has already exited.",
            )
            return True
        except Exception as exc:
            LOGGER.exception("failed to terminate external %s session", session.provider)
            self.gateway.post_thread_reply(
                thread,
                f"Failed to terminate the {session.provider.value} session: {exc}",
            )
            return True
        self.gateway.post_thread_reply(
            thread,
            f"Terminated the external {session.provider.value} session.",
        )
        return True


def build_external_session_prompt(text: str, slack_user: str | None = None) -> str:
    return text.strip()


def is_session_exit_request(text: str) -> bool:
    stripped = text.strip()
    return stripped.lower() == SESSION_EXIT_COMMAND


def _resume_command(
    session: AgentSession,
    prompt: str,
    commands: AgentCommandConfig,
) -> list[str]:
    if session.provider == Provider.CODEX:
        args = [
            commands.codex_binary,
            "exec",
            "resume",
            "--json",
            "--skip-git-repo-check",
        ]
        cwd = _command_cwd(session)
        if cwd:
            args.extend(["-c", _codex_trust_override(cwd)])
        if commands.dangerous_by_default:
            args.append(dangerous_flag(Provider.CODEX))
        args.extend([session.session_id, prompt])
        return args
    if session.provider == Provider.CLAUDE:
        args = [
            commands.claude_binary,
            "--print",
            "--output-format",
            "json",
            "--resume",
            session.session_id,
        ]
        if commands.dangerous_by_default:
            args.append(dangerous_flag(Provider.CLAUDE))
        if session.permission_mode:
            args.extend(["--permission-mode", session.permission_mode])
        args.append(prompt)
        return args
    raise ValueError(f"unsupported provider: {session.provider}")


def _command_cwd(session: AgentSession) -> Path | None:
    if session.cwd and session.cwd.exists():
        return session.cwd
    return None


def _render_resume_output(provider: Provider, stdout: str) -> str | None:
    chunks, _ = _process_output_chunks(provider, stdout, final=True)
    rendered = "\n\n".join(chunks).strip()
    return rendered or None


def _is_latest_active_session_for_cwd(store: Store, session: AgentSession) -> bool:
    if session.cwd is None:
        return False
    cutoff_seconds = 30 * 60
    now = utc_now()
    matches: list[AgentSession] = []
    for candidate in store.list_sessions(session.provider):
        if candidate.status != SessionStatus.ACTIVE:
            continue
        if candidate.cwd is None or not _same_path(candidate.cwd, session.cwd):
            continue
        if candidate.last_seen_at is None:
            continue
        if (now - candidate.last_seen_at).total_seconds() > cutoff_seconds:
            continue
        matches.append(candidate)
    if not matches:
        return False
    matches.sort(key=lambda item: item.last_seen_at or item.started_at or now, reverse=True)
    return matches[0].session_id == session.session_id


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except OSError:
        return left.expanduser() == right.expanduser()


def _target_start_matches_session(session: AgentSession, target: TerminalTarget) -> bool:
    if session.started_at is None:
        return True
    if target.started_at is None:
        return False
    delta = abs(target.started_at.astimezone() - session.started_at.astimezone())
    return delta <= timedelta(minutes=15)


def _slackgentic_channel_enabled(command: str) -> bool:
    try:
        args = shlex.split(command)
    except ValueError:
        args = command.split()
    channel_tokens = {"server:slackgentic", "slackgentic"}
    for index, arg in enumerate(args):
        if arg in {"--channels", "--dangerously-load-development-channels"}:
            values = _channel_flag_values(args[index + 1 :])
            return any(token in channel_tokens for token in values)
        if arg.startswith(("--channels=", "--dangerously-load-development-channels=")):
            values = _split_channel_value(arg.split("=", 1)[1])
            return any(token in channel_tokens for token in values)
    return False


def _codex_remote_enabled(command: str, expected_url: str | None) -> bool:
    if not expected_url or expected_url.lower() == "off":
        return False
    try:
        args = shlex.split(command)
    except ValueError:
        args = command.split()
    for index, arg in enumerate(args):
        if arg == "--remote" and index + 1 < len(args):
            return _remote_urls_match(args[index + 1], expected_url)
        if arg.startswith("--remote="):
            return _remote_urls_match(arg.split("=", 1)[1], expected_url)
    return False


def _remote_urls_match(actual: str, expected: str) -> bool:
    actual = actual.strip()
    expected = expected.strip()
    if actual == expected:
        return True
    try:
        actual_parsed = urllib.parse.urlparse(actual)
        expected_parsed = urllib.parse.urlparse(expected)
    except ValueError:
        return False
    if actual_parsed.scheme != expected_parsed.scheme:
        return False
    actual_host = _loopback_host_alias(actual_parsed.hostname)
    expected_host = _loopback_host_alias(expected_parsed.hostname)
    return (
        actual_host == expected_host
        and actual_parsed.port == expected_parsed.port
        and (actual_parsed.path or "/") == (expected_parsed.path or "/")
    )


def _loopback_host_alias(host: str | None) -> str | None:
    if host is None:
        return None
    normalized = host.lower()
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return "localhost"
    return normalized


def _channel_flag_values(args: list[str]) -> list[str]:
    values: list[str] = []
    for arg in args:
        if arg.startswith("-"):
            break
        values.extend(_split_channel_value(arg))
    return values


def _split_channel_value(value: str) -> list[str]:
    return [item for item in re.split(r"[\s,]+", value) if item]
