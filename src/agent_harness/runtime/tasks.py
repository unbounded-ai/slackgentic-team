from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from agent_harness.config import AgentCommandConfig
from agent_harness.models import (
    AgentTask,
    AgentTaskStatus,
    Provider,
    SlackThreadRef,
    TeamAgent,
    parse_timestamp,
)
from agent_harness.runtime.runner import LaunchRequest, ManagedAgentProcess
from agent_harness.slack.client import SlackGateway
from agent_harness.storage.store import Store
from agent_harness.team import runtime_personality_prompt

LOGGER = logging.getLogger(__name__)
SETTING_REPO_ROOT = "slack.repo_root"
AGENT_THREAD_DONE_SIGNAL = "SLACKGENTIC: THREAD_DONE"
ProcessFactory = Callable[[LaunchRequest], ManagedAgentProcess]
TaskDoneCallback = Callable[[AgentTask, TeamAgent, SlackThreadRef], None]
AgentMessageCallback = Callable[[AgentTask, TeamAgent, SlackThreadRef, str], bool]
AgentControlCallback = Callable[[AgentTask, TeamAgent, SlackThreadRef, str], bool]
AgentIconUrlCallback = Callable[[TeamAgent], str | None]


@dataclass
class RunningTask:
    task: AgentTask
    agent: TeamAgent
    process: ManagedAgentProcess
    thread: SlackThreadRef
    worker: threading.Thread
    output_buffer: str = ""
    session_buffer: str = ""
    observed_agent_messages: set[str] | None = None
    control_signals: list[str] = field(default_factory=list)
    visible_message_count: int = 0


class ManagedTaskRuntime:
    def __init__(
        self,
        store: Store,
        gateway: SlackGateway,
        commands: AgentCommandConfig,
        process_factory: ProcessFactory | None = None,
        poll_seconds: float = 1.0,
        on_task_done: TaskDoneCallback | None = None,
        on_agent_message: AgentMessageCallback | None = None,
        on_agent_control: AgentControlCallback | None = None,
        agent_icon_url: AgentIconUrlCallback | None = None,
        home: Path | None = None,
    ):
        self.store = store
        self.gateway = gateway
        self.commands = commands
        self.process_factory = process_factory or ManagedAgentProcess
        self.poll_seconds = poll_seconds
        self.on_task_done = on_task_done
        self.on_agent_message = on_agent_message
        self.on_agent_control = on_agent_control
        self.agent_icon_url = agent_icon_url
        self.home = home or Path.home()
        self._running: dict[str, RunningTask] = {}
        self._lock = threading.Lock()

    def start_task(self, task: AgentTask, agent: TeamAgent, thread: SlackThreadRef) -> bool:
        provider = agent.provider_preference or Provider.CODEX
        cwd = _task_cwd(task, self._default_cwd())
        request = LaunchRequest(
            provider=provider,
            prompt=build_task_prompt(agent, task),
            cwd=cwd,
            dangerous=self.commands.dangerous_by_default,
            resume_session_id=(
                task.session_id
                if task.session_id
                and (task.session_provider is None or task.session_provider == provider)
                else None
            ),
            codex_binary=self.commands.codex_binary,
            claude_binary=self.commands.claude_binary,
        )
        process = self.process_factory(request)
        try:
            process.start()
        except Exception as exc:
            self.gateway.post_thread_reply(
                thread,
                f"Failed to start {provider.value}: {exc}",
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
            self.store.delete_managed_thread_task(task.task_id)
            return False

        self.store.update_agent_task_status(task.task_id, AgentTaskStatus.ACTIVE)
        self.store.update_agent_task_session(task.task_id, provider, task.session_id)
        self.store.upsert_managed_thread_task(task, thread)
        worker = threading.Thread(
            target=self._stream_task,
            args=(task.task_id,),
            daemon=True,
            name=f"slackgentic-{task.task_id}",
        )
        running = RunningTask(task=task, agent=agent, process=process, thread=thread, worker=worker)
        with self._lock:
            self._running[task.task_id] = running
        worker.start()
        return True

    def send_to_task(self, task_id: str, message: str) -> bool:
        running = self._get_running(task_id)
        if running is None:
            return False
        if _provider_for_running(running) in {Provider.CODEX, Provider.CLAUDE}:
            return False
        try:
            running.process.send(message)
        except Exception:
            LOGGER.debug("failed to send message to running managed task", exc_info=True)
            return False
        return True

    def stop_task(self, task_id: str, status: AgentTaskStatus = AgentTaskStatus.CANCELLED) -> bool:
        running = self._get_running(task_id)
        if running is None:
            self.store.update_agent_task_status(task_id, status)
            return False
        running.process.terminate()
        self.store.update_agent_task_status(task_id, status)
        with self._lock:
            self._running.pop(task_id, None)
        return True

    def running_task_for_thread(self, channel_id: str, thread_ts: str) -> RunningTask | None:
        with self._lock:
            for running in self._running.values():
                if (
                    running.thread.channel_id == channel_id
                    and running.thread.thread_ts == thread_ts
                ):
                    return running
        return None

    def has_running_tasks(self) -> bool:
        with self._lock:
            return bool(self._running)

    def _default_cwd(self) -> Path:
        configured = self.store.get_setting(SETTING_REPO_ROOT)
        if configured:
            path = Path(configured).expanduser()
            if path.exists():
                return path
        return self.commands.default_cwd

    def _stream_task(self, task_id: str) -> None:
        while True:
            running = self._get_running(task_id)
            if running is None:
                return
            output = running.process.read_available()
            self._capture_session_id(running, output)
            chunks, running.output_buffer = _process_output_chunks(
                _provider_for_running(running),
                output,
                running.output_buffer,
            )
            for chunk in chunks:
                self._post_agent_chunk(running, chunk)
            if not running.process.is_alive():
                time.sleep(0.05)
                tail = running.process.read_available(max_reads=100, timeout=0.05)
                self._capture_session_id(running, tail, final=True)
                chunks, running.output_buffer = _process_output_chunks(
                    _provider_for_running(running),
                    tail,
                    running.output_buffer,
                    final=True,
                )
                for chunk in chunks:
                    self._post_agent_chunk(running, chunk)
                with self._lock:
                    self._running.pop(task_id, None)
                try:
                    completed_task = self.store.get_agent_task(task_id) or running.task
                except Exception:
                    LOGGER.debug("failed to load completed task", exc_info=True)
                    completed_task = running.task
                recovered_message = self._recover_unseen_visible_message(completed_task, running)
                if recovered_message:
                    self._post_agent_chunk(running, recovered_message)
                if running.visible_message_count == 0 and not running.control_signals:
                    recovered_message = self._recover_unseen_visible_message(
                        completed_task,
                        running,
                    )
                    if recovered_message:
                        self._post_agent_chunk(running, recovered_message)
                        if self._handle_agent_control_signals(running, completed_task):
                            return
                        if self.on_task_done:
                            self.on_task_done(completed_task, running.agent, running.thread)
                        return
                    self.gateway.post_thread_reply(
                        running.thread,
                        (
                            f"{running.agent.full_name} finished without a "
                            "Slack-visible response. I cancelled this run so the "
                            "silent exit is visible instead of looking stuck."
                        ),
                        persona=running.agent,
                        icon_url=self._agent_icon_url(running.agent),
                    )
                    self.store.update_agent_task_status(task_id, AgentTaskStatus.CANCELLED)
                    return
                if self._handle_agent_control_signals(running, completed_task):
                    return
                if self.on_task_done:
                    self.on_task_done(completed_task, running.agent, running.thread)
                return
            time.sleep(self.poll_seconds)

    def _post_agent_chunk(self, running: RunningTask, chunk: str) -> None:
        visible_text, control_signals = _extract_agent_control_signals(chunk)
        running.control_signals.extend(control_signals)
        if not visible_text:
            return
        self.gateway.post_thread_reply(
            running.thread,
            visible_text,
            persona=running.agent,
            icon_url=self._agent_icon_url(running.agent),
        )
        running.visible_message_count += 1
        normalized = visible_text.strip()
        already_observed = False
        if normalized:
            if running.observed_agent_messages is None:
                running.observed_agent_messages = set()
            already_observed = normalized in running.observed_agent_messages
            running.observed_agent_messages.add(normalized)
        if self.on_agent_message is None:
            return
        if not normalized:
            return
        if already_observed:
            return
        try:
            self.on_agent_message(running.task, running.agent, running.thread, normalized)
        except Exception:
            LOGGER.exception("failed to handle agent-authored Slack message")

    def _handle_agent_control_signals(self, running: RunningTask, task: AgentTask) -> bool:
        if self.on_agent_control is None:
            return False
        handled = False
        for signal in dict.fromkeys(running.control_signals):
            try:
                handled = (
                    self.on_agent_control(task, running.agent, running.thread, signal) or handled
                )
            except Exception:
                LOGGER.exception("failed to handle agent control signal")
        return handled

    def _recover_visible_message(self, task: AgentTask, running: RunningTask) -> str | None:
        if _provider_for_running(running) != Provider.CODEX or not task.session_id:
            return None
        return _latest_codex_transcript_message(
            task.session_id,
            self.home,
            since=task.created_at,
        )

    def _recover_unseen_visible_message(
        self,
        task: AgentTask,
        running: RunningTask,
    ) -> str | None:
        message = self._recover_visible_message(task, running)
        if not message:
            return None
        normalized = message.strip()
        if not normalized:
            return None
        if running.observed_agent_messages and normalized in running.observed_agent_messages:
            return None
        return message

    def _agent_icon_url(self, agent: TeamAgent) -> str | None:
        if self.agent_icon_url is None:
            return None
        try:
            return self.agent_icon_url(agent)
        except Exception:
            LOGGER.debug("failed to resolve agent icon URL", exc_info=True)
            return None

    def _capture_session_id(
        self,
        running: RunningTask,
        output: str,
        *,
        final: bool = False,
    ) -> None:
        session_id, running.session_buffer = _session_id_from_output(
            _provider_for_running(running),
            output,
            running.session_buffer,
            final=final,
        )
        if not session_id or running.task.session_id == session_id:
            return
        provider = _provider_for_running(running)
        self.store.update_agent_task_session(running.task.task_id, provider, session_id)
        running.task = replace(
            running.task,
            session_provider=provider,
            session_id=session_id,
        )

    def _get_running(self, task_id: str) -> RunningTask | None:
        with self._lock:
            return self._running.get(task_id)


def build_task_prompt(agent: TeamAgent, task: AgentTask) -> str:
    lines = [
        runtime_personality_prompt(agent),
        "",
        "You are working from Slack. Keep progress updates concise.",
        (
            "Only write Slack-visible messages for major progress updates, blockers, "
            "questions, and final results. Do not narrate tool calls or raw command output "
            "unless the user explicitly asks for it."
        ),
        (
            "When a table is the clearest format, write one normal Markdown table in the "
            "message. Slackgentic renders one Markdown table per message as a native Slack "
            "table. If you need multiple tables, send separate messages."
        ),
        (
            "You may ask another agent for a review or second opinion by sending one "
            "separate Slack-visible message beginning exactly `somebody review ...` "
            "with the concrete item to review. After that message, stop and wait; "
            "Slackgentic will route the review and resume you with the review context."
        ),
        (
            "If the user is clearly closing the whole Slack thread or says no more work "
            f"is needed, write your normal brief final reply and then add a final line "
            f"exactly `{AGENT_THREAD_DONE_SIGNAL}`. Slackgentic hides that line and "
            "marks the whole thread done. Do not use this signal just because your "
            "current task is complete; use it only when the entire thread should be closed."
        ),
        (
            "When you hand work to a specific agent, use that agent's exact Slackgentic "
            "`@handle` from the thread or roster."
        ),
        (
            "When you need a named external/session agent in the Slack thread to choose, "
            "continue, or respond, put your context and options first. Then end with a "
            "separate final paragraph whose first token is that agent's `@handle`, for "
            "example `@nell pick one before I proceed`. Do not put that callback handle "
            "inline near the beginning or bury it before the options."
        ),
        f"Task kind: {task.kind.value}",
        f"Task: {task.prompt}",
    ]
    pr_url = task.metadata.get("pr_url")
    if pr_url:
        lines.append(f"Pull request URL: {pr_url}")
        lines.append("Review the PR and report findings with concrete file or diff references.")
    thread_context = task.metadata.get("thread_context")
    if isinstance(thread_context, str) and thread_context.strip():
        lines.extend(
            [
                "",
                "Private Slack thread context. Use this to understand prior messages, "
                "handoffs, reviews, and user intent. Do not quote this heading or describe "
                "it as hidden context in Slack-visible replies:",
                thread_context.strip(),
            ]
        )
    return "\n".join(lines)


def _task_cwd(task: AgentTask, default_cwd: Path) -> Path:
    configured = task.metadata.get("cwd")
    if isinstance(configured, str) and configured:
        path = Path(configured).expanduser()
        if path.exists():
            return path
    return _requested_repo_cwd(task.prompt, default_cwd)


def _requested_repo_cwd(prompt: str, default_cwd: Path) -> Path:
    root = default_cwd
    if not root.exists() or not root.is_dir():
        return default_cwd
    try:
        repo_names = {
            path.name.lower(): path
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
    except OSError:
        return default_cwd
    patterns = [
        r"\b(?:in|inside|for)\s+(?:repo\s+|repository\s+)?(?P<repo>[A-Za-z0-9_.-]+)\b",
        r"\b(?:repo|repository)\s+(?P<repo>[A-Za-z0-9_.-]+)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            repo = match.group("repo").lower()
            if repo in repo_names:
                return repo_names[repo]
    return default_cwd


def _slack_chunks(text: str, limit: int = 2800) -> list[str]:
    cleaned = _clean_terminal_output(text)
    if not cleaned:
        return []
    chunks: list[str] = []
    while cleaned:
        chunks.append(cleaned[:limit])
        cleaned = cleaned[limit:]
    return chunks


def _extract_agent_control_signals(text: str) -> tuple[str, list[str]]:
    visible_lines: list[str] = []
    signals: list[str] = []
    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line.strip()).upper()
        if normalized == AGENT_THREAD_DONE_SIGNAL:
            signals.append(AGENT_THREAD_DONE_SIGNAL)
            continue
        visible_lines.append(line)
    return "\n".join(visible_lines).strip(), signals


def _process_output_chunks(
    provider: Provider,
    text: str,
    buffer: str = "",
    final: bool = False,
) -> tuple[list[str], str]:
    if provider == Provider.CODEX:
        return _codex_exec_chunks(text, buffer, final=final)
    return _claude_json_chunks(text, buffer, final=final)


def _session_id_from_output(
    provider: Provider,
    text: str,
    buffer: str = "",
    final: bool = False,
) -> tuple[str | None, str]:
    combined = buffer + text
    if not combined:
        return None, buffer
    lines = combined.splitlines(keepends=True)
    next_buffer = ""
    if lines and not _line_has_ending(lines[-1]) and not final:
        next_buffer = lines.pop()

    session_id = None
    for line in lines:
        candidate = _session_id_from_line(provider, line.strip())
        if candidate:
            session_id = candidate
    return session_id, next_buffer


def _session_id_from_line(provider: Provider, line: str) -> str | None:
    if not line:
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    if provider == Provider.CODEX and event.get("type") == "thread.started":
        value = event.get("thread_id")
        return str(value) if value else None
    if provider == Provider.CLAUDE:
        value = event.get("session_id") or event.get("sessionId")
        return str(value) if value else None
    return None


def _codex_exec_chunks(
    text: str,
    buffer: str = "",
    final: bool = False,
    limit: int = 2800,
) -> tuple[list[str], str]:
    combined = buffer + text
    if not combined:
        return [], buffer

    lines = combined.splitlines(keepends=True)
    next_buffer = ""
    if lines and not _line_has_ending(lines[-1]) and not final:
        next_buffer = lines.pop()

    rendered: list[str] = []
    for line in lines:
        message = _render_codex_exec_line(line.strip())
        if message:
            rendered.extend(_slack_chunks(message, limit=limit))
    return rendered, next_buffer


def _render_codex_exec_line(line: str) -> str | None:
    if not line or line == "Reading additional input from stdin...":
        return None
    if "failed to record rollout items" in line:
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return _clean_terminal_output(line) or None
    if not isinstance(event, dict):
        return None

    event_type = event.get("type")
    if event_type == "item.completed":
        item = event.get("item")
        return _render_codex_item(item if isinstance(item, dict) else {})
    if event_type == "event_msg":
        payload = event.get("payload")
        return _render_codex_event_payload(payload if isinstance(payload, dict) else {})
    if event_type == "agent_message":
        message = event.get("message") or event.get("text")
        return _clean_terminal_output(str(message)) if message else None
    if event_type == "response_item":
        payload = event.get("payload")
        return _render_codex_response_item(payload if isinstance(payload, dict) else {})
    if event_type == "error":
        message = event.get("message") or event.get("error")
        return f"Codex error: {message}" if message else "Codex error."
    return None


def _render_codex_item(item: dict) -> str | None:
    item_type = item.get("type")
    if item_type == "agent_message":
        text = item.get("text")
        return _clean_terminal_output(str(text)) if text else None
    return None


def _render_codex_event_payload(payload: dict) -> str | None:
    payload_type = payload.get("type")
    if payload_type == "agent_message":
        message = payload.get("message") or payload.get("text")
        return _clean_terminal_output(str(message)) if message else None
    return None


def _render_codex_response_item(payload: dict) -> str | None:
    if payload.get("type") != "message" or payload.get("role") != "assistant":
        return None
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"output_text", "text"} and isinstance(item.get("text"), str):
            cleaned = _clean_terminal_output(item["text"])
            if cleaned:
                parts.append(cleaned)
    return "\n\n".join(parts) if parts else None


def _latest_codex_transcript_message(
    session_id: str,
    home: Path,
    *,
    since: datetime | None = None,
) -> str | None:
    sessions_root = home / ".codex" / "sessions"
    if not sessions_root.exists():
        return None
    try:
        paths = sorted(
            sessions_root.rglob(f"*{session_id}.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    for path in paths:
        message = _latest_codex_transcript_message_from_path(path, since=since)
        if message:
            return message
    return None


def _latest_codex_transcript_message_from_path(
    path: Path,
    *,
    since: datetime | None = None,
) -> str | None:
    latest: str | None = None
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        timestamp = parse_timestamp(record.get("timestamp"))
        if since is not None and timestamp is not None and timestamp < since:
            continue
        message = _codex_transcript_record_message(record)
        if message:
            latest = message
    return latest


def _codex_transcript_record_message(record: dict) -> str | None:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    if record.get("type") == "event_msg" and payload.get("type") == "agent_message":
        message = payload.get("message") or payload.get("text")
        return _clean_terminal_output(str(message)) if message else None
    if record.get("type") == "response_item":
        return _render_codex_response_item(payload)
    return None


def _claude_json_chunks(
    text: str,
    buffer: str = "",
    final: bool = False,
    limit: int = 2800,
) -> tuple[list[str], str]:
    combined = buffer + text
    if not combined:
        return [], buffer

    lines = combined.splitlines(keepends=True)
    next_buffer = ""
    if lines and not _line_has_ending(lines[-1]) and not final:
        next_buffer = lines.pop()

    rendered: list[str] = []
    for line in lines:
        message = _render_claude_json_line(line.strip())
        if message:
            rendered.extend(_slack_chunks(message, limit=limit))
    return rendered, next_buffer


def _render_claude_json_line(line: str) -> str | None:
    if not line:
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        cleaned = _clean_terminal_output(line)
        if cleaned.lower().startswith("error:"):
            return cleaned
        return None
    if not isinstance(event, dict):
        return None

    event_type = event.get("type")
    if event_type == "result":
        if event.get("is_error"):
            message = event.get("result") or event.get("api_error_status") or event.get("subtype")
            return _format_claude_error(message)
        result = event.get("result")
        return _clean_terminal_output(str(result)) if result else None
    if event_type == "error":
        message = event.get("message") or event.get("error")
        return _format_claude_error(message)
    return None


def _format_claude_error(message: object) -> str:
    return f"Claude error: {_clean_terminal_output(str(message))}" if message else "Claude error."


def _line_has_ending(line: str) -> bool:
    return line.endswith(("\n", "\r"))


def _provider_for_running(running: RunningTask) -> Provider:
    return running.task.session_provider or running.agent.provider_preference or Provider.CODEX


def _clean_terminal_output(text: str) -> str:
    without_osc = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
    without_string_controls = re.sub(
        r"\x1b[PX^_].*?\x1b\\",
        "",
        without_osc,
        flags=re.DOTALL,
    )
    no_ansi = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", without_string_controls)
    no_ansi = re.sub(r"\x1b[@-_][0-?]*[ -/]*[@-~]", "", no_ansi)
    no_control = "".join(ch for ch in no_ansi if ch in "\n\t" or ord(ch) >= 32)
    return no_control.strip()
