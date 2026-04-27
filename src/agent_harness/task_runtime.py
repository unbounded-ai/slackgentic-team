from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent_harness.config import AgentCommandConfig
from agent_harness.models import AgentTask, AgentTaskStatus, Provider, SlackThreadRef, TeamAgent
from agent_harness.runner import LaunchRequest, ManagedAgentProcess
from agent_harness.slack_client import SlackGateway
from agent_harness.store import Store
from agent_harness.team import runtime_personality_prompt

LOGGER = logging.getLogger(__name__)
ProcessFactory = Callable[[LaunchRequest], ManagedAgentProcess]
TaskDoneCallback = Callable[[AgentTask, TeamAgent, SlackThreadRef], None]
AgentMessageCallback = Callable[[AgentTask, TeamAgent, SlackThreadRef, str], bool]


@dataclass
class RunningTask:
    task: AgentTask
    agent: TeamAgent
    process: ManagedAgentProcess
    thread: SlackThreadRef
    worker: threading.Thread
    output_buffer: str = ""
    observed_agent_messages: set[str] | None = None


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
    ):
        self.store = store
        self.gateway = gateway
        self.commands = commands
        self.process_factory = process_factory or ManagedAgentProcess
        self.poll_seconds = poll_seconds
        self.on_task_done = on_task_done
        self.on_agent_message = on_agent_message
        self._running: dict[str, RunningTask] = {}
        self._lock = threading.Lock()

    def start_task(self, task: AgentTask, agent: TeamAgent, thread: SlackThreadRef) -> bool:
        provider = agent.provider_preference or Provider.CODEX
        cwd = _task_cwd(task, self.commands.default_cwd)
        request = LaunchRequest(
            provider=provider,
            prompt=build_task_prompt(agent, task),
            cwd=cwd,
            dangerous=self.commands.dangerous_by_default,
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
            )
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
            return False

        self.store.update_agent_task_status(task.task_id, AgentTaskStatus.ACTIVE)
        self.store.update_agent_task_session(task.task_id, provider, task.session_id)
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
        if _provider_for_running(running) == Provider.CODEX:
            return False
        running.process.send(message)
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

    def _stream_task(self, task_id: str) -> None:
        while True:
            running = self._get_running(task_id)
            if running is None:
                return
            output = running.process.read_available()
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
                chunks, running.output_buffer = _process_output_chunks(
                    _provider_for_running(running),
                    tail,
                    running.output_buffer,
                    final=True,
                )
                for chunk in chunks:
                    self._post_agent_chunk(running, chunk)
                self.store.update_agent_task_status(task_id, AgentTaskStatus.DONE)
                with self._lock:
                    self._running.pop(task_id, None)
                completed_task = self.store.get_agent_task(task_id) or running.task
                if self.on_task_done:
                    self.on_task_done(completed_task, running.agent, running.thread)
                return
            time.sleep(self.poll_seconds)

    def _post_agent_chunk(self, running: RunningTask, chunk: str) -> None:
        self.gateway.post_thread_reply(running.thread, chunk, persona=running.agent)
        if self.on_agent_message is None:
            return
        normalized = chunk.strip()
        if not normalized:
            return
        if running.observed_agent_messages is None:
            running.observed_agent_messages = set()
        if normalized in running.observed_agent_messages:
            return
        running.observed_agent_messages.add(normalized)
        try:
            self.on_agent_message(running.task, running.agent, running.thread, normalized)
        except Exception:
            LOGGER.exception("failed to handle agent-authored Slack message")

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
            "If you are unsure and need another agent to review before you continue, send "
            "one separate Slack-visible message beginning exactly `somebody review ...` "
            "with the concrete item to review. After that message, stop and wait; "
            "Slackgentic will route the review and resume the right agent with the thread "
            "context."
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
                "Slack thread context:",
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
    root = default_cwd.parent
    repo_names = {
        path.name.lower(): path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
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


def _process_output_chunks(
    provider: Provider,
    text: str,
    buffer: str = "",
    final: bool = False,
) -> tuple[list[str], str]:
    if provider == Provider.CODEX:
        return _codex_exec_chunks(text, buffer, final=final)
    return _claude_json_chunks(text, buffer, final=final)


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
    if event_type == "assistant":
        return _render_claude_assistant_message(event.get("message"))
    if event_type == "error":
        message = event.get("message") or event.get("error")
        return _format_claude_error(message)
    return None


def _format_claude_error(message: object) -> str:
    return f"Claude error: {_clean_terminal_output(str(message))}" if message else "Claude error."


def _render_claude_assistant_message(message: object) -> str | None:
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return _clean_terminal_output(content) or None
    if not isinstance(content, list):
        return None

    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            cleaned = _clean_terminal_output(item["text"])
            if cleaned:
                text_parts.append(cleaned)
    return "\n\n".join(text_parts) if text_parts else None


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
