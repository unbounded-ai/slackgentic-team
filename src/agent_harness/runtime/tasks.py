from __future__ import annotations

import json
import logging
import platform
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from secrets import token_urlsafe

from agent_harness.bash_policy import (
    allowed_bash_session_tools_for_command,
    allowed_bash_tools_for_command,
    classify_bash_command,
)
from agent_harness.config import AgentCommandConfig
from agent_harness.deferred import AGENT_DEFERRED_SIGNAL_PREFIX
from agent_harness.models import (
    AgentTask,
    AgentTaskStatus,
    PermissionMode,
    Provider,
    SlackThreadRef,
    TeamAgent,
    parse_timestamp,
    utc_now,
)
from agent_harness.permissions import (
    claude_safe_auto_permission_request_allowed,
    task_permission_mode,
)
from agent_harness.pm import AGENT_PM_PLAN_SIGNAL_PREFIX
from agent_harness.pr_links import pr_urls_from_metadata
from agent_harness.providers.claude import is_synthetic_claude_assistant_record
from agent_harness.runtime.runner import LaunchRequest, ManagedAgentProcess
from agent_harness.schedules import AGENT_SCHEDULE_SIGNAL_PREFIX
from agent_harness.sessions.claude_channel import (
    SLACKGENTIC_MCP_PERMISSION_ALLOW,
    is_slackgentic_mcp_server_configured,
)
from agent_harness.sessions.managed_session import clear_managed_session, record_managed_session
from agent_harness.slack import replace_slack_user_ids
from agent_harness.slack.client import SlackGateway
from agent_harness.storage.store import Store
from agent_harness.team import runtime_personality_prompt
from agent_harness.timers import AGENT_TIMER_SIGNAL_PREFIX

LOGGER = logging.getLogger(__name__)
SETTING_REPO_ROOT = "slack.repo_root"
AGENT_THREAD_DONE_SIGNAL = "SLACKGENTIC: THREAD_DONE"
AGENT_ROSTER_STATUS_SIGNAL_PREFIX = "SLACKGENTIC: ROSTER "
AGENT_REACTION_SIGNAL_PREFIX = "SLACKGENTIC: REACT "
MANAGED_RUN_STARTED_METADATA_KEY = "managed_run_started_at"
MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY = "managed_run_resume_attempts"
MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY = "managed_run_stall_recoveries"
MANAGED_RUN_ORIGINAL_PROMPT_METADATA_KEY = "managed_run_original_prompt"
MANAGED_RUN_EMPTY_TEXT_BLOCK_API_RETRIES_METADATA_KEY = "managed_run_empty_text_block_api_retries"
MANAGED_RUN_MAX_RESUMES = 3
MANAGED_RUN_MAX_STALL_RECOVERIES = 2
MANAGED_RUN_MAX_EMPTY_TEXT_BLOCK_API_RETRIES = 2
MANAGED_RUN_MAX_RESUME_AGE = timedelta(minutes=15)
CODEX_THREAD_START_TIMEOUT = timedelta(minutes=2)
MANAGED_RUN_PROGRESS_WARNING_TIMEOUT = timedelta(minutes=5)
MANAGED_RUN_STALL_TIMEOUT = timedelta(minutes=15)
CLAUDE_EFFORT_SETTING = "effortLevel"
CLAUDE_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})
CLAUDE_DEFAULT_EFFORT = "xhigh"
FAST_STREAM_POLL_SECONDS = 0.1
MIN_STREAM_POLL_SECONDS = 0.01
TRANSCRIPT_ACTIVITY_STAT_INTERVAL_SECONDS = 5.0
TRANSCRIPT_ACTIVITY_DISCOVERY_INTERVAL_SECONDS = 30.0
MACOS_TCC_PROTECTED_HOME_DIRS = (
    "Desktop",
    "Documents",
    "Downloads",
    "Movies",
    "Music",
    "Pictures",
)
ProcessFactory = Callable[[LaunchRequest], ManagedAgentProcess]
TaskDoneCallback = Callable[[AgentTask, TeamAgent, SlackThreadRef], None]
AgentMessageCallback = Callable[[AgentTask, TeamAgent, SlackThreadRef, str, str | None], bool]
AgentControlCallback = Callable[[AgentTask, TeamAgent, SlackThreadRef, str], bool]
AgentIconUrlCallback = Callable[[TeamAgent], str | None]
TranscriptActivitySignature = tuple[str, int, int]
TranscriptActivitySnapshot = tuple[Path, TranscriptActivitySignature]


@dataclass
class RunningTask:
    task: AgentTask
    agent: TeamAgent
    process: ManagedAgentProcess
    thread: SlackThreadRef
    worker: threading.Thread
    resume_session_id: str | None = None
    allowed_tools: tuple[str, ...] = ()
    output_buffer: str = ""
    session_buffer: str = ""
    permission_buffer: str = ""
    permission_denials: list[dict] = field(default_factory=list)
    permission_denial_keys: set[str] = field(default_factory=set)
    permission_tool_uses: dict[str, dict] = field(default_factory=dict)
    permission_request_tokens: dict[str, str] = field(default_factory=dict)
    resume_error_buffer: str = ""
    missing_resume_session: bool = False
    api_error_buffer: str = ""
    empty_text_block_api_error: bool = False
    observed_agent_messages: set[str] | None = None
    control_signals: list[str] = field(default_factory=list)
    terminal_control_signal_handled: bool = False
    visible_message_count: int = 0
    started_monotonic: float = field(default_factory=time.monotonic)
    last_output_monotonic: float = field(default_factory=time.monotonic)
    last_activity_monotonic: float = field(default_factory=time.monotonic)
    last_visible_message_monotonic: float | None = None
    transcript_activity_session_id: str | None = None
    transcript_activity_path: Path | None = None
    transcript_activity_signature: TranscriptActivitySignature | None = None
    transcript_activity_checked: bool = False
    last_transcript_activity_check_monotonic: float = 0.0
    last_transcript_discovery_monotonic: float = 0.0
    progress_warning_monotonic: float | None = None
    stop_requested: bool = False
    wake_event: threading.Event = field(default_factory=threading.Event)


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
        codex_thread_start_timeout: timedelta = CODEX_THREAD_START_TIMEOUT,
        agent_start_timeout: timedelta | None = None,
        agent_progress_timeout: timedelta | None = None,
        agent_stall_timeout: timedelta | None = None,
        max_stall_recoveries: int | None = None,
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
        explicit_legacy_start_timeout = codex_thread_start_timeout != CODEX_THREAD_START_TIMEOUT
        self.agent_start_timeout = (
            agent_start_timeout
            if agent_start_timeout is not None
            else (
                codex_thread_start_timeout
                if explicit_legacy_start_timeout
                else _seconds_config_timedelta(
                    commands.agent_start_timeout_seconds,
                    default=CODEX_THREAD_START_TIMEOUT,
                )
            )
        )
        self.codex_thread_start_timeout = self.agent_start_timeout
        self.agent_progress_timeout = (
            agent_progress_timeout
            if agent_progress_timeout is not None
            else _seconds_config_timedelta(
                commands.agent_progress_timeout_seconds,
                default=MANAGED_RUN_PROGRESS_WARNING_TIMEOUT,
            )
        )
        self.agent_stall_timeout = (
            agent_stall_timeout
            if agent_stall_timeout is not None
            else _seconds_config_timedelta(
                commands.agent_stall_timeout_seconds,
                default=MANAGED_RUN_STALL_TIMEOUT,
            )
        )
        configured_recoveries = getattr(
            commands,
            "agent_stall_recovery_attempts",
            MANAGED_RUN_MAX_STALL_RECOVERIES,
        )
        self.max_stall_recoveries = (
            max_stall_recoveries
            if max_stall_recoveries is not None
            else max(0, configured_recoveries)
        )
        self._running: dict[str, RunningTask] = {}
        self._lock = threading.Lock()

    def start_task(
        self,
        task: AgentTask,
        agent: TeamAgent,
        thread: SlackThreadRef,
        *,
        allowed_tools: tuple[str, ...] = (),
    ) -> bool:
        with self._lock:
            current = self._running.get(task.task_id)
            if current is not None and (
                current.worker is not threading.current_thread() or current.process.is_alive()
            ):
                LOGGER.debug(
                    "refusing duplicate start for task %s; a worker is already running",
                    task.task_id,
                )
                return False
            if current is not None:
                # The current worker can restart its own just-finished task from
                # the completion callback, for example to flush queued follow-ups.
                self._running.pop(task.task_id, None)
        provider = agent.provider_preference or Provider.CODEX
        default_cwd = self._default_cwd()
        cwd = _task_cwd(task, default_cwd)
        tcc_issue = _macos_tcc_protected_cwd_issue(
            cwd,
            home=self.home,
            allow=self.commands.allow_macos_tcc_protected_paths,
        )
        if tcc_issue:
            self.gateway.post_thread_reply(
                thread,
                (
                    f"I did not start {provider.value} because {tcc_issue}. "
                    "Move the repo under an unprotected working directory such as ~/code, "
                    "or set SLACKGENTIC_ALLOW_MACOS_TCC_PROTECTED_PATHS=true after granting "
                    "the chosen runtime the required macOS privacy access."
                ),
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
            self.store.delete_managed_thread_task(task.task_id)
            return False
        claude_channel = provider == Provider.CLAUDE and is_slackgentic_mcp_server_configured(
            self.home
        )
        launch_allowed_tools = _initial_allowed_tools(
            provider,
            allowed_tools,
            claude_channel=claude_channel,
        )
        mode = task_permission_mode(task)
        if self.commands.dangerous_by_default and mode != PermissionMode.DANGEROUS:
            mode = PermissionMode.DANGEROUS
        request = LaunchRequest(
            provider=provider,
            prompt=build_task_prompt(agent, task),
            cwd=cwd,
            permission_mode=mode,
            resume_session_id=(
                task.session_id
                if task.session_id
                and (task.session_provider is None or task.session_provider == provider)
                else None
            ),
            claude_channel=claude_channel,
            slack_channel_id=thread.channel_id,
            slack_thread_ts=thread.thread_ts,
            allowed_tools=launch_allowed_tools,
            safe_auto_extra_roots=_safe_auto_extra_roots(provider, cwd, default_cwd),
            codex_binary=self.commands.codex_binary,
            claude_binary=self.commands.claude_binary,
            claude_effort=(
                _claude_effort_from_settings(cwd, self.home)
                if provider == Provider.CLAUDE
                else None
            ),
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

        task = self._mark_managed_run_started(task)
        self.store.update_agent_task_status(task.task_id, AgentTaskStatus.ACTIVE)
        self.store.update_agent_task_session(task.task_id, provider, task.session_id)
        if task.session_id:
            record_managed_session(
                self.store,
                provider,
                task.session_id,
                agent.agent_id,
                dangerous_mode=mode == PermissionMode.DANGEROUS,
            )
        self.store.upsert_managed_thread_task(task, thread)
        worker = threading.Thread(
            target=self._stream_task,
            args=(task.task_id,),
            daemon=True,
            name=f"slackgentic-{task.task_id}",
        )
        running = RunningTask(
            task=task,
            agent=agent,
            process=process,
            thread=thread,
            worker=worker,
            resume_session_id=request.resume_session_id,
            allowed_tools=launch_allowed_tools,
        )
        if request.resume_session_id:
            # The transcript already exists from a prior worker run (e.g. a
            # daemon restart). Anything visible in it was already posted to
            # Slack by that prior worker; without seeding the dedup set here,
            # the recovery branch at process-exit (_recover_unseen_visible_messages)
            # would re-post the entire history and the user sees every prior
            # reply duplicated in the thread.
            try:
                historical = self._recover_visible_messages(task, running)
            except Exception:
                LOGGER.debug("failed to seed observed messages from transcript", exc_info=True)
                historical = []
            if historical:
                running.observed_agent_messages = {
                    message.strip() for message in historical if message.strip()
                }
        with self._lock:
            self._running[task.task_id] = running
        worker.start()
        return True

    def send_to_task(self, task_id: str, message: str) -> bool:
        return self._send_to_running_task(task_id, message, allow_cli=False)

    def send_to_interrupted_task(self, task_id: str, message: str) -> bool:
        return self._send_to_running_task(task_id, message, allow_cli=True)

    def _send_to_running_task(self, task_id: str, message: str, *, allow_cli: bool) -> bool:
        running = self._get_running(task_id)
        if running is None:
            return False
        provider = _provider_for_running(running)
        # Codex --print closes stdin after the initial prompt, so a live send
        # is impossible until allow_cli=True signals the interactive follow-up
        # path. Claude --input-format=stream-json keeps stdin open and accepts
        # follow-up user turns at any time, so we let those through.
        if not allow_cli and provider == Provider.CODEX:
            return False
        try:
            running.process.send(message)
        except Exception:
            LOGGER.debug("failed to send message to running managed task", exc_info=True)
            return False
        running.wake_event.set()
        return True

    def stop_task(
        self,
        task_id: str,
        status: AgentTaskStatus | None = AgentTaskStatus.CANCELLED,
        *,
        join_timeout: float = 2.0,
        kill_join_timeout: float | None = None,
    ) -> bool:
        running = self._get_running(task_id)
        if running is None:
            if status is not None:
                self._clear_managed_run_started(task_id)
                self._clear_managed_session_for_task(task_id)
                self.store.update_agent_task_status(task_id, status)
            return False
        running.stop_requested = True
        running.wake_event.set()
        running.process.terminate()
        if status is not None:
            self._clear_managed_run_started(task_id)
            self._clear_managed_session_for_task(task_id)
            self.store.update_agent_task_status(task_id, status)
        if running.worker is threading.current_thread():
            self._remove_running_task(task_id, running)
            return True
        running.worker.join(timeout=max(0.0, join_timeout))
        if not running.worker.is_alive():
            self._remove_running_task(task_id, running)
            return True
        # SIGTERM did not free the worker. Escalate to a hard kill so a hung
        # provider process cannot indefinitely block restart of this task.
        LOGGER.warning(
            "managed task worker for %s did not exit on terminate; escalating to kill",
            task_id,
        )
        try:
            running.process.kill()
        except Exception:
            LOGGER.debug("failed to escalate kill for managed task %s", task_id, exc_info=True)
        escalation_timeout = kill_join_timeout if kill_join_timeout is not None else join_timeout
        running.worker.join(timeout=max(0.0, escalation_timeout))
        if not running.worker.is_alive():
            self._remove_running_task(task_id, running)
            return True
        LOGGER.warning("managed task worker did not stop for %s", task_id)
        return False

    def interrupt_task(self, task_id: str) -> bool:
        running = self._get_running(task_id)
        if running is None:
            return False
        interrupt = getattr(running.process, "interrupt", None)
        try:
            if callable(interrupt):
                interrupt()
            else:
                running.process.send("\x1b")
        except Exception:
            LOGGER.debug("failed to interrupt running managed task", exc_info=True)
            return False
        running.wake_event.set()
        return True

    def is_task_running(self, task_id: str) -> bool:
        return self._get_running(task_id) is not None

    def stop_all_running_tasks(
        self,
        status: AgentTaskStatus | None = AgentTaskStatus.CANCELLED,
        *,
        join_timeout: float = 2.0,
    ) -> int:
        with self._lock:
            running_tasks = list(self._running.items())
        stopped = 0
        for task_id, running in running_tasks:
            try:
                running.stop_requested = True
                running.wake_event.set()
                running.process.terminate()
                if status is not None:
                    self._clear_managed_run_started(task_id)
                    self._clear_managed_session_for_task(task_id)
                    self.store.update_agent_task_status(task_id, status)
            except Exception:
                LOGGER.debug(
                    "failed to request stop for running task %s during shutdown",
                    task_id,
                    exc_info=True,
                )
        deadline = time.monotonic() + max(0.0, join_timeout)
        for task_id, running in running_tasks:
            current = self._get_running(task_id)
            if current is not running:
                stopped += 1
                continue
            if running.worker is threading.current_thread():
                self._remove_running_task(task_id, running)
                stopped += 1
                continue
            remaining = max(0.0, deadline - time.monotonic())
            running.worker.join(timeout=remaining)
            if running.worker.is_alive():
                LOGGER.warning("managed task worker did not stop for %s", task_id)
                continue
            self._remove_running_task(task_id, running)
            stopped += 1
        return stopped

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

    def running_tasks(self) -> list[RunningTask]:
        with self._lock:
            return list(self._running.values())

    def _remove_running_task(self, task_id: str, running: RunningTask) -> None:
        with self._lock:
            if self._running.get(task_id) is running:
                self._running.pop(task_id, None)

    def _default_cwd(self) -> Path:
        configured = self.store.get_setting(SETTING_REPO_ROOT)
        if configured:
            path = Path(configured).expanduser()
            if path.exists():
                return path
        return self.commands.default_cwd

    def _stream_task(self, task_id: str) -> None:
        try:
            self._stream_task_loop(task_id)
        except Exception:
            LOGGER.exception("managed task worker failed for %s", task_id)
            self._handle_task_worker_failure(task_id)
        finally:
            running = self._get_running(task_id)
            if running is not None and running.worker is threading.current_thread():
                self._remove_running_task(task_id, running)

    def _stream_task_loop(self, task_id: str) -> None:
        sleep_seconds = _fast_stream_poll_seconds(self.poll_seconds)
        while True:
            running = self._get_running(task_id)
            if running is None:
                return
            output = running.process.read_available()
            had_output = bool(output)
            if had_output:
                now = time.monotonic()
                running.last_output_monotonic = now
                running.last_activity_monotonic = now
            if running.stop_requested:
                self._remove_running_task(task_id, running)
                return
            self._capture_session_id(running, output)
            self._capture_transcript_activity(running)
            permission_denied = self._capture_permission_denials(running, output)
            self._capture_resume_errors(running, output)
            self._capture_empty_text_block_api_error(running, output)
            if permission_denied and running.process.is_alive():
                try:
                    running.process.terminate()
                except Exception:
                    LOGGER.debug("failed to stop Claude after permission denial", exc_info=True)
            if running.missing_resume_session:
                chunks = []
            else:
                chunks, running.output_buffer = _process_output_chunks(
                    _provider_for_running(running),
                    output,
                    running.output_buffer,
                )
            for chunk in chunks:
                self._post_agent_chunk(running, chunk)
            if running.stop_requested:
                self._remove_running_task(task_id, running)
                return
            if self._managed_task_start_timed_out(running):
                self._handle_managed_task_start_timeout(running)
                return
            if self._managed_task_progress_warning_due(running):
                self._handle_managed_task_progress_warning(running)
            if self._managed_task_stall_timed_out(running):
                self._handle_managed_task_stall_timeout(running)
                return
            if not running.process.is_alive():
                time.sleep(0.05)
                tail = running.process.read_available(max_reads=100, timeout=0.05)
                if running.stop_requested:
                    # The process exited because we asked it to; do not post the
                    # tail buffer, recovered transcript lines, or "finished without
                    # a response" message — the user already released the task.
                    self._remove_running_task(task_id, running)
                    return
                self._capture_session_id(running, tail, final=True)
                self._capture_permission_denials(running, tail, final=True)
                self._capture_resume_errors(running, tail, final=True)
                self._capture_empty_text_block_api_error(running, tail, final=True)
                if running.missing_resume_session:
                    chunks = []
                else:
                    chunks, running.output_buffer = _process_output_chunks(
                        _provider_for_running(running),
                        tail,
                        running.output_buffer,
                        final=True,
                    )
                for chunk in chunks:
                    self._post_agent_chunk(running, chunk)
                try:
                    completed_task = self.store.get_agent_task(task_id) or running.task
                except Exception:
                    LOGGER.debug("failed to load completed task", exc_info=True)
                    completed_task = running.task
                if self._retry_missing_claude_resume(running, completed_task):
                    return
                if self._retry_empty_text_block_api_error(running, completed_task):
                    return
                if running.permission_denials:
                    self._handle_claude_permission_denial(running, completed_task)
                    return
                for recovered_message in self._recover_unseen_visible_messages(
                    completed_task,
                    running,
                ):
                    self._post_agent_chunk(running, recovered_message)
                if (
                    running.visible_message_count == 0
                    and not running.control_signals
                    and not running.terminal_control_signal_handled
                ):
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
                    self._clear_managed_run_started(task_id)
                    self._clear_managed_session_for_task(task_id)
                    self.store.update_agent_task_status(task_id, AgentTaskStatus.CANCELLED)
                    return
                completed_task = self._clear_managed_run_started(task_id) or completed_task
                if running.terminal_control_signal_handled:
                    return
                handled_signals = self._handle_agent_control_signals(running, completed_task)
                if AGENT_THREAD_DONE_SIGNAL in handled_signals:
                    return
                if self.on_task_done:
                    self.on_task_done(completed_task, running.agent, running.thread)
                return
            sleep_seconds = _next_stream_poll_seconds(
                sleep_seconds,
                configured_poll_seconds=self.poll_seconds,
                had_output=had_output,
            )
            if running.wake_event.wait(sleep_seconds):
                running.wake_event.clear()
                sleep_seconds = _fast_stream_poll_seconds(self.poll_seconds)

    def _codex_thread_start_timed_out(self, running: RunningTask) -> bool:
        return self._managed_task_start_timed_out(running)

    def _managed_task_start_timed_out(self, running: RunningTask) -> bool:
        if running.task.session_id:
            return False
        if running.visible_message_count or running.control_signals:
            return False
        if not running.process.is_alive():
            return False
        timeout_seconds = self.agent_start_timeout.total_seconds()
        if timeout_seconds <= 0:
            return True
        return time.monotonic() - running.started_monotonic >= timeout_seconds

    def _handle_codex_thread_start_timeout(self, running: RunningTask) -> None:
        self._handle_managed_task_start_timeout(running)

    def _handle_managed_task_start_timeout(self, running: RunningTask) -> None:
        task_id = running.task.task_id
        provider = _provider_for_running(running)
        try:
            running.process.terminate()
        except Exception:
            LOGGER.debug("failed to stop managed task after startup timeout", exc_info=True)
        with self._lock:
            self._running.pop(task_id, None)
        self._clear_managed_run_started(task_id)
        self._clear_managed_session_for_task(task_id)
        self.store.update_agent_task_status(task_id, AgentTaskStatus.CANCELLED)
        self.store.delete_managed_thread_task(task_id)
        self.gateway.post_thread_reply(
            running.thread,
            (
                f"{running.agent.full_name} did not finish starting {provider.value.capitalize()} "
                f"within {int(self.agent_start_timeout.total_seconds())} seconds, "
                "so I stopped this run instead of leaving it stuck."
            ),
            persona=running.agent,
            icon_url=self._agent_icon_url(running.agent),
        )

    def _managed_task_progress_warning_due(self, running: RunningTask) -> bool:
        if not running.process.is_alive():
            return False
        if not running.task.session_id and running.visible_message_count <= 0:
            return False
        timeout_seconds = self.agent_progress_timeout.total_seconds()
        if timeout_seconds <= 0:
            return False
        last_visible = running.last_visible_message_monotonic or running.started_monotonic
        last_warning = running.progress_warning_monotonic
        if last_warning is not None and last_warning >= last_visible:
            return False
        now = time.monotonic()
        if now - last_visible < timeout_seconds:
            return False
        # The warning's wording only fits when the worker has gone silent in
        # every channel. A worker still producing provider output or appending
        # to its transcript is making progress; posting the warning then is
        # noise the stall timeout already covers if real silence sets in.
        return now - running.last_activity_monotonic >= timeout_seconds

    def _handle_managed_task_progress_warning(self, running: RunningTask) -> None:
        running.progress_warning_monotonic = time.monotonic()
        self.gateway.post_thread_reply(
            running.thread,
            (
                f"{running.agent.full_name} has not posted Slack-visible progress for "
                f"{_format_timeout_duration(self.agent_progress_timeout)}. I am leaving "
                "the run active, but will restart it if there is no observed activity "
                "(provider output, transcript updates, or Slack-visible progress) for "
                f"{_format_timeout_duration(self.agent_stall_timeout)}."
            ),
            persona=running.agent,
            icon_url=self._agent_icon_url(running.agent),
        )

    def _managed_task_progress_timed_out(self, running: RunningTask) -> bool:
        return self._managed_task_stall_timed_out(running)

    def _managed_task_stall_timed_out(self, running: RunningTask) -> bool:
        if not running.process.is_alive():
            return False
        if not running.task.session_id and running.visible_message_count <= 0:
            return False
        timeout_seconds = self.agent_stall_timeout.total_seconds()
        if timeout_seconds < 0:
            return False
        if timeout_seconds == 0:
            return True
        return time.monotonic() - running.last_activity_monotonic >= timeout_seconds

    def _handle_managed_task_progress_timeout(self, running: RunningTask) -> None:
        self._handle_managed_task_stall_timeout(running)

    def _handle_managed_task_stall_timeout(self, running: RunningTask) -> None:
        task_id = running.task.task_id
        provider = _provider_for_running(running)
        try:
            running.process.terminate()
        except Exception:
            LOGGER.debug("failed to stop managed task after stall timeout", exc_info=True)
        with self._lock:
            self._running.pop(task_id, None)
        current = self.store.get_agent_task(task_id) or running.task
        attempts = managed_run_stall_recoveries(current)
        if not current.session_id or attempts >= self.max_stall_recoveries:
            self._clear_managed_run_started(task_id)
            self.store.update_agent_task_status(task_id, AgentTaskStatus.CANCELLED)
            self.store.delete_managed_thread_task(task_id)
            self.gateway.post_thread_reply(
                running.thread,
                _progress_timeout_cancel_message(
                    running.agent.full_name,
                    provider,
                    self.agent_stall_timeout,
                    has_session=bool(current.session_id),
                ),
                persona=running.agent,
                icon_url=self._agent_icon_url(running.agent),
            )
            return

        recovery_task = _task_for_progress_stall_recovery(
            current,
            self.agent_stall_timeout,
            attempts=attempts + 1,
        )
        self.store.upsert_agent_task(recovery_task)
        self.gateway.post_thread_reply(
            running.thread,
            (
                f"{running.agent.full_name} had no observed activity for "
                f"{_format_timeout_duration(self.agent_stall_timeout)}: no provider output, "
                "no transcript updates, and no Slack-visible progress. I stopped that "
                "run and restarted it with a status request."
            ),
            persona=running.agent,
            icon_url=self._agent_icon_url(running.agent),
        )
        self.start_task(
            recovery_task,
            running.agent,
            running.thread,
            allowed_tools=running.allowed_tools,
        )

    def _handle_task_worker_failure(self, task_id: str) -> None:
        running = self._get_running(task_id)
        with self._lock:
            self._running.pop(task_id, None)
        if running is not None:
            try:
                if running.process.is_alive():
                    running.process.terminate()
            except Exception:
                LOGGER.debug(
                    "failed to terminate child process for crashed task %s",
                    task_id,
                    exc_info=True,
                )
        try:
            current = self.store.get_agent_task(task_id)
        except Exception:
            LOGGER.debug("failed to load crashed task", exc_info=True)
            current = None
        if current is not None and current.status in {
            AgentTaskStatus.QUEUED,
            AgentTaskStatus.ACTIVE,
        }:
            self._clear_managed_run_started(task_id)
            self._clear_managed_session_for_task(task_id)
            self.store.update_agent_task_status(task_id, AgentTaskStatus.CANCELLED)
            self.store.delete_managed_thread_task(task_id)
        if running is None:
            return
        try:
            self.gateway.post_thread_reply(
                running.thread,
                (
                    f"{running.agent.full_name} hit an internal Slackgentic error and "
                    "this run was stopped instead of being left active."
                ),
                persona=running.agent,
                icon_url=self._agent_icon_url(running.agent),
            )
        except Exception:
            LOGGER.debug("failed to post managed task worker failure", exc_info=True)

    def _post_agent_chunk(self, running: RunningTask, chunk: str) -> None:
        visible_text, control_signals = _extract_agent_control_signals(chunk)
        hide_visible_text = False
        deferred_signals: list[str] = []
        terminal_signals: list[str] = []
        for signal in control_signals:
            if is_agent_roster_status_signal(signal) or is_agent_reaction_signal(signal):
                self._handle_immediate_agent_control_signal(running, signal)
            elif signal == AGENT_THREAD_DONE_SIGNAL:
                terminal_signals.append(signal)
            else:
                deferred_signals.append(signal)
                if _is_schedule_control_signal(signal):
                    hide_visible_text = True
        running.control_signals.extend(deferred_signals)

        def handle_terminal_signals() -> None:
            for signal in terminal_signals:
                if self._handle_immediate_agent_control_signal(running, signal):
                    running.terminal_control_signal_handled = True
                else:
                    running.control_signals.append(signal)

        if hide_visible_text:
            visible_text = ""
        if not visible_text:
            handle_terminal_signals()
            return
        normalized = visible_text.strip()
        if not normalized:
            handle_terminal_signals()
            return
        if running.stop_requested:
            # The user explicitly released or stopped the task; pending output
            # from the child process must not reach Slack after that signal.
            return
        if running.observed_agent_messages is None:
            running.observed_agent_messages = set()
        if normalized in running.observed_agent_messages:
            handle_terminal_signals()
            return
        try:
            posted = self.gateway.post_thread_reply(
                running.thread,
                visible_text,
                persona=running.agent,
                icon_url=self._agent_icon_url(running.agent),
            )
        except Exception:
            handle_terminal_signals()
            raise
        now = time.monotonic()
        running.visible_message_count += 1
        running.last_visible_message_monotonic = now
        running.last_activity_monotonic = now
        running.progress_warning_monotonic = None
        running.observed_agent_messages.add(normalized)
        if self.on_agent_message is not None:
            try:
                self.on_agent_message(
                    running.task,
                    running.agent,
                    running.thread,
                    normalized,
                    posted.ts,
                )
            except Exception:
                LOGGER.exception("failed to handle agent-authored Slack message")
        handle_terminal_signals()

    def _handle_immediate_agent_control_signal(self, running: RunningTask, signal: str) -> bool:
        if self.on_agent_control is None:
            return False
        try:
            return bool(self.on_agent_control(running.task, running.agent, running.thread, signal))
        except Exception:
            LOGGER.exception("failed to handle immediate agent control signal")
            return False

    def _handle_agent_control_signals(self, running: RunningTask, task: AgentTask) -> set[str]:
        if self.on_agent_control is None:
            return set()
        handled: set[str] = set()
        for signal in dict.fromkeys(running.control_signals):
            try:
                if self.on_agent_control(task, running.agent, running.thread, signal):
                    handled.add(signal)
            except Exception:
                LOGGER.exception("failed to handle agent control signal")
        return handled

    def _recover_visible_message(self, task: AgentTask, running: RunningTask) -> str | None:
        messages = self._recover_visible_messages(task, running)
        return messages[-1] if messages else None

    def _recover_visible_messages(self, task: AgentTask, running: RunningTask) -> list[str]:
        provider = _provider_for_running(running)
        if not task.session_id:
            return []
        if provider == Provider.CODEX:
            return _codex_transcript_messages(
                task.session_id,
                self.home,
                since=task.created_at,
            )
        if provider == Provider.CLAUDE:
            return _claude_transcript_messages(
                task.session_id,
                self.home,
                since=task.created_at,
            )
        return []

    def _recover_unseen_visible_message(
        self,
        task: AgentTask,
        running: RunningTask,
    ) -> str | None:
        messages = self._recover_unseen_visible_messages(task, running)
        return messages[-1] if messages else None

    def _recover_unseen_visible_messages(
        self,
        task: AgentTask,
        running: RunningTask,
    ) -> list[str]:
        recovered: list[str] = []
        for message in self._recover_visible_messages(task, running):
            normalized = message.strip()
            if not normalized:
                continue
            if running.observed_agent_messages and normalized in running.observed_agent_messages:
                continue
            recovered.append(message)
        return recovered

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
        record_managed_session(
            self.store,
            provider,
            session_id,
            running.agent.agent_id,
            dangerous_mode=running.process.request.dangerous,
        )
        running.task = replace(
            running.task,
            session_provider=provider,
            session_id=session_id,
        )

    def _capture_transcript_activity(self, running: RunningTask) -> None:
        session_id = running.task.session_id
        if not session_id:
            return
        if running.transcript_activity_session_id != session_id:
            running.transcript_activity_session_id = session_id
            running.transcript_activity_path = None
            running.transcript_activity_signature = None
            running.transcript_activity_checked = False
            running.last_transcript_discovery_monotonic = 0.0

        now = time.monotonic()
        if (
            now - running.last_transcript_activity_check_monotonic
            < TRANSCRIPT_ACTIVITY_STAT_INTERVAL_SECONDS
        ):
            return
        running.last_transcript_activity_check_monotonic = now

        snapshot: TranscriptActivitySnapshot | None = None
        if running.transcript_activity_path is not None:
            signature = _transcript_activity_signature_for_path(running.transcript_activity_path)
            if signature is None:
                running.transcript_activity_path = None
            else:
                snapshot = (running.transcript_activity_path, signature)

        if snapshot is None:
            if (
                now - running.last_transcript_discovery_monotonic
                < TRANSCRIPT_ACTIVITY_DISCOVERY_INTERVAL_SECONDS
            ):
                running.transcript_activity_checked = True
                return
            running.last_transcript_discovery_monotonic = now
            snapshot = _session_transcript_activity_snapshot(
                _provider_for_running(running),
                session_id,
                self.home,
            )

        if snapshot is None:
            running.transcript_activity_checked = True
            return
        path, signature = snapshot
        running.transcript_activity_path = path
        if running.transcript_activity_signature is None:
            running.transcript_activity_signature = signature
            if running.transcript_activity_checked:
                running.last_activity_monotonic = now
            running.transcript_activity_checked = True
            return
        if signature == running.transcript_activity_signature:
            running.transcript_activity_checked = True
            return
        running.transcript_activity_signature = signature
        running.transcript_activity_checked = True
        running.last_activity_monotonic = now

    def _capture_permission_denials(
        self,
        running: RunningTask,
        output: str,
        *,
        final: bool = False,
    ) -> bool:
        if _provider_for_running(running) != Provider.CLAUDE:
            return False
        if running.process.request.dangerous:
            return False
        denials, running.permission_buffer = _claude_permission_denials(
            output,
            running.permission_buffer,
            final=final,
            tool_uses=running.permission_tool_uses,
        )
        new_denial = False
        for denial in denials:
            key = _claude_denial_key(denial)
            if key in running.permission_denial_keys:
                continue
            running.permission_denial_keys.add(key)
            running.permission_denials.append(denial)
            if not _claude_safe_auto_denial_allowed(running, denial):
                self._ensure_claude_permission_request(running, denial)
            new_denial = True
            break
        return new_denial

    def _capture_resume_errors(
        self,
        running: RunningTask,
        output: str,
        *,
        final: bool = False,
    ) -> None:
        if _provider_for_running(running) != Provider.CLAUDE:
            return
        missing, running.resume_error_buffer = _claude_missing_resume_session(
            output,
            running.resume_error_buffer,
            final=final,
        )
        running.missing_resume_session = running.missing_resume_session or missing

    def _retry_missing_claude_resume(
        self,
        running: RunningTask,
        completed_task: AgentTask,
    ) -> bool:
        if (
            _provider_for_running(running) != Provider.CLAUDE
            or not running.missing_resume_session
            or not running.resume_session_id
        ):
            return False
        LOGGER.info(
            "Claude session %s was missing; restarting task %s without resume",
            running.resume_session_id,
            completed_task.task_id,
        )
        retry_task = replace(
            completed_task,
            session_provider=Provider.CLAUDE,
            session_id=None,
            status=AgentTaskStatus.ACTIVE,
            updated_at=utc_now(),
        )
        self.store.upsert_agent_task(retry_task)
        self._remove_running_task(completed_task.task_id, running)
        self.start_task(
            retry_task,
            running.agent,
            running.thread,
            allowed_tools=running.allowed_tools,
        )
        return True

    def _capture_empty_text_block_api_error(
        self,
        running: RunningTask,
        output: str,
        *,
        final: bool = False,
    ) -> None:
        if _provider_for_running(running) != Provider.CLAUDE:
            return
        detected, running.api_error_buffer = _claude_empty_text_block_api_error(
            output,
            running.api_error_buffer,
            final=final,
        )
        if detected:
            running.empty_text_block_api_error = True

    def _retry_empty_text_block_api_error(
        self,
        running: RunningTask,
        completed_task: AgentTask,
    ) -> bool:
        # The Claude API rejects requests whose conversation history contains
        # an empty text content block in any assistant turn. The model
        # occasionally emits an empty text block alongside a tool_use; once it
        # is in the session, every subsequent request fails the same way. The
        # only reliable recovery is a fresh session — resuming the same
        # session_id would just replay the bad turn back to the API.
        if (
            _provider_for_running(running) != Provider.CLAUDE
            or not running.empty_text_block_api_error
        ):
            return False
        current = self.store.get_agent_task(completed_task.task_id) or completed_task
        attempts = managed_run_empty_text_block_api_retries(current)
        if attempts >= MANAGED_RUN_MAX_EMPTY_TEXT_BLOCK_API_RETRIES:
            return False
        LOGGER.info(
            "Claude rejected empty text content block for task %s; restarting with "
            "fresh session (attempt %s/%s)",
            completed_task.task_id,
            attempts + 1,
            MANAGED_RUN_MAX_EMPTY_TEXT_BLOCK_API_RETRIES,
        )
        metadata = dict(current.metadata)
        metadata[MANAGED_RUN_EMPTY_TEXT_BLOCK_API_RETRIES_METADATA_KEY] = attempts + 1
        retry_task = replace(
            current,
            session_provider=Provider.CLAUDE,
            session_id=None,
            status=AgentTaskStatus.ACTIVE,
            metadata=metadata,
            updated_at=utc_now(),
        )
        self.store.upsert_agent_task(retry_task)
        self.gateway.post_thread_reply(
            running.thread,
            (
                f"{running.agent.full_name} hit a transient Claude SDK error "
                "(empty text content block). Restarting with a fresh session "
                "and retrying."
            ),
            persona=running.agent,
            icon_url=self._agent_icon_url(running.agent),
        )
        self._remove_running_task(completed_task.task_id, running)
        self.start_task(
            retry_task,
            running.agent,
            running.thread,
            allowed_tools=running.allowed_tools,
        )
        return True

    def _handle_claude_permission_denial(
        self,
        running: RunningTask,
        completed_task: AgentTask,
    ) -> None:
        denial = running.permission_denials[0]
        internal_tool = _slackgentic_mcp_tool_for_claude_denial(denial)
        if internal_tool:
            self._retry_claude_permission_denial(
                running,
                completed_task,
                denial,
                (internal_tool,),
            )
            return
        safe_auto_allowed_tools = _safe_auto_allowed_tools_for_claude_denial(running, denial)
        if safe_auto_allowed_tools:
            self._retry_claude_permission_denial(
                running,
                completed_task,
                denial,
                safe_auto_allowed_tools,
            )
            return
        allowed_tools = _allowed_tools_for_claude_denial(denial)
        if not allowed_tools:
            self.gateway.post_thread_reply(
                running.thread,
                "Claude requested a tool approval I cannot safely resume automatically.",
                persona=running.agent,
                icon_url=self._agent_icon_url(running.agent),
            )
            self.store.update_agent_task_status(completed_task.task_id, AgentTaskStatus.CANCELLED)
            self.store.delete_managed_thread_task(completed_task.task_id)
            return

        token = self._ensure_claude_permission_request(running, denial)
        response = None
        if token:
            from agent_harness.slack.agent_requests import SlackAgentRequestHandler

            handler = SlackAgentRequestHandler(
                self.gateway,
                store=self.store,
                provider_label="Claude",
            )
            response = handler.wait_for_persistent_request(token)
        if not isinstance(response, dict) or response.get("behavior") != "allow":
            self.gateway.post_thread_reply(
                running.thread,
                "Claude tool approval was denied; I stopped this run.",
                persona=running.agent,
                icon_url=self._agent_icon_url(running.agent),
            )
            self.store.update_agent_task_status(completed_task.task_id, AgentTaskStatus.CANCELLED)
            self.store.delete_managed_thread_task(completed_task.task_id)
            return
        if response.get("scope") == "session":
            allowed_tools = _append_allowed_tools(
                allowed_tools,
                _allowed_session_tools_for_claude_denial(denial),
            )
        self._retry_claude_permission_denial(running, completed_task, denial, allowed_tools)

    def _retry_claude_permission_denial(
        self,
        running: RunningTask,
        completed_task: AgentTask,
        denial: dict,
        allowed_tools: tuple[str, ...],
    ) -> None:
        current = self.store.get_agent_task(completed_task.task_id) or completed_task
        retry_task = replace(
            current,
            prompt=_claude_permission_retry_prompt(denial),
            status=AgentTaskStatus.ACTIVE,
            updated_at=utc_now(),
        )
        self.store.upsert_agent_task(retry_task)
        self._remove_running_task(completed_task.task_id, running)
        self.start_task(
            retry_task,
            running.agent,
            running.thread,
            allowed_tools=_append_allowed_tools(running.allowed_tools, allowed_tools),
        )

    def _ensure_claude_permission_request(
        self,
        running: RunningTask,
        denial: dict,
    ) -> str | None:
        if _slackgentic_mcp_tool_for_claude_denial(denial):
            return None
        if not _allowed_tools_for_claude_denial(denial):
            return None
        key = _claude_denial_key(denial)
        existing = running.permission_request_tokens.get(key)
        if existing:
            return existing
        from agent_harness.slack.agent_requests import SlackAgentRequestHandler

        handler = SlackAgentRequestHandler(
            self.gateway,
            store=self.store,
            provider_label="Claude",
        )
        status_text = _claude_permission_status_text(denial)
        if status_text:
            self.gateway.post_thread_reply(
                running.thread,
                status_text,
                persona=running.agent,
                icon_url=self._agent_icon_url(running.agent),
            )
        pending = handler.create_persistent_request(
            "claude/channel/permission",
            _claude_denial_request_params(denial),
            running.thread,
            provider_label="Claude",
        )
        running.permission_request_tokens[key] = pending.token
        return pending.token

    def _get_running(self, task_id: str) -> RunningTask | None:
        with self._lock:
            return self._running.get(task_id)

    def _mark_managed_run_started(self, task: AgentTask) -> AgentTask:
        metadata = dict(task.metadata)
        metadata[MANAGED_RUN_STARTED_METADATA_KEY] = utc_now().isoformat()
        updated = replace(task, metadata=metadata, updated_at=utc_now())
        self.store.upsert_agent_task(updated)
        return updated

    def _clear_managed_session_for_task(self, task_id: str) -> None:
        try:
            current = self.store.get_agent_task(task_id)
        except Exception:
            LOGGER.debug("failed to load task before clearing managed session", exc_info=True)
            return
        if current is None or current.session_provider is None or not current.session_id:
            return
        try:
            clear_managed_session(self.store, current.session_provider, current.session_id)
        except Exception:
            LOGGER.debug("failed to clear managed session for task %s", task_id, exc_info=True)

    def _clear_managed_run_started(self, task_id: str) -> AgentTask | None:
        try:
            current = self.store.get_agent_task(task_id)
        except Exception:
            LOGGER.debug("failed to load task before clearing managed run marker", exc_info=True)
            return None
        if current is None:
            return current
        has_marker = MANAGED_RUN_STARTED_METADATA_KEY in current.metadata
        has_attempts = MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY in current.metadata
        has_stall_recoveries = MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY in current.metadata
        has_original_prompt = MANAGED_RUN_ORIGINAL_PROMPT_METADATA_KEY in current.metadata
        has_text_block_retries = (
            MANAGED_RUN_EMPTY_TEXT_BLOCK_API_RETRIES_METADATA_KEY in current.metadata
        )
        if (
            not has_marker
            and not has_attempts
            and not has_stall_recoveries
            and not has_original_prompt
            and not has_text_block_retries
        ):
            return current
        metadata = dict(current.metadata)
        metadata.pop(MANAGED_RUN_STARTED_METADATA_KEY, None)
        metadata.pop(MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY, None)
        metadata.pop(MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY, None)
        metadata.pop(MANAGED_RUN_ORIGINAL_PROMPT_METADATA_KEY, None)
        metadata.pop(MANAGED_RUN_EMPTY_TEXT_BLOCK_API_RETRIES_METADATA_KEY, None)
        updated = replace(current, metadata=metadata, updated_at=utc_now())
        try:
            self.store.upsert_agent_task(updated)
        except Exception:
            LOGGER.debug("failed to clear managed run marker", exc_info=True)
            return current
        return updated

    def resume_orphaned_task(
        self,
        task: AgentTask,
        agent: TeamAgent,
        thread: SlackThreadRef,
    ) -> bool:
        try:
            current = self.store.get_agent_task(task.task_id) or task
        except Exception:
            LOGGER.debug("failed to load task before resume bump", exc_info=True)
            current = task
        attempts = managed_run_resume_attempts(current) + 1
        metadata = dict(current.metadata)
        metadata[MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY] = attempts
        bumped = replace(current, metadata=metadata, updated_at=utc_now())
        try:
            self.store.upsert_agent_task(bumped)
        except Exception:
            LOGGER.debug("failed to record managed run resume attempt", exc_info=True)
            bumped = current
        return self.start_task(bumped, agent, thread)


def managed_run_resume_attempts(task: AgentTask) -> int:
    value = task.metadata.get(MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return max(0, int(value.strip()))
    return 0


def managed_run_stall_recoveries(task: AgentTask) -> int:
    value = task.metadata.get(MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return max(0, int(value.strip()))
    return 0


def _task_for_progress_stall_recovery(
    task: AgentTask,
    timeout: timedelta,
    *,
    attempts: int,
) -> AgentTask:
    metadata = dict(task.metadata)
    original_prompt = metadata.get(MANAGED_RUN_ORIGINAL_PROMPT_METADATA_KEY)
    if not isinstance(original_prompt, str) or not original_prompt.strip():
        original_prompt = task.prompt
    metadata[MANAGED_RUN_ORIGINAL_PROMPT_METADATA_KEY] = original_prompt
    metadata[MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY] = attempts
    return replace(
        task,
        prompt=_progress_stall_recovery_prompt(original_prompt, timeout),
        status=AgentTaskStatus.ACTIVE,
        metadata=metadata,
        updated_at=utc_now(),
    )


def _progress_stall_recovery_prompt(original_prompt: str, timeout: timedelta) -> str:
    return (
        "The previous managed run was stopped because Slackgentic observed no activity "
        f"for {_format_timeout_duration(timeout)}: no provider output, no transcript "
        "updates, and no Slack-visible progress. First post a concise Slack-visible "
        "status update in the thread explaining what you know so far, then continue "
        "the task. Do not redo completed work; inspect the local repo/session state "
        "and continue from there. If you are blocked, say exactly what is blocking "
        "progress.\n\n"
        f"Original task: {original_prompt}"
    )


def _progress_timeout_cancel_message(
    agent_name: str,
    provider: Provider,
    timeout: timedelta,
    *,
    has_session: bool,
) -> str:
    duration = _format_timeout_duration(timeout)
    if not has_session:
        return (
            f"{agent_name} had no observed activity for {duration}, and Slackgentic does "
            f"not have a {provider.value} session id to resume safely. I stopped this "
            "run instead of leaving it stuck."
        )
    return (
        f"{agent_name} repeatedly had no observed activity for {duration}. I stopped this "
        "run instead of restarting it again."
    )


def _format_timeout_duration(timeout: timedelta) -> str:
    seconds = max(0, int(timeout.total_seconds()))
    if seconds % 60 == 0 and seconds >= 60:
        minutes = seconds // 60
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit}"
    unit = "second" if seconds == 1 else "seconds"
    return f"{seconds} {unit}"


def _seconds_config_timedelta(value: float, *, default: timedelta) -> timedelta:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return default
    if seconds < 0:
        return default
    return timedelta(seconds=seconds)


def _fast_stream_poll_seconds(configured_poll_seconds: float) -> float:
    return min(
        max(MIN_STREAM_POLL_SECONDS, configured_poll_seconds),
        FAST_STREAM_POLL_SECONDS,
    )


def _next_stream_poll_seconds(
    current_sleep_seconds: float,
    *,
    configured_poll_seconds: float,
    had_output: bool,
) -> float:
    fast = _fast_stream_poll_seconds(configured_poll_seconds)
    if had_output:
        return fast
    max_sleep = max(fast, configured_poll_seconds)
    return min(max_sleep, max(fast, current_sleep_seconds * 1.5))


def managed_run_started_age(
    task: AgentTask,
    *,
    now: datetime | None = None,
) -> timedelta | None:
    marker = task.metadata.get(MANAGED_RUN_STARTED_METADATA_KEY)
    if not isinstance(marker, str):
        return None
    started = parse_timestamp(marker)
    if started is None:
        return None
    reference = now or utc_now()
    return reference - started


def should_resume_managed_run(
    task: AgentTask,
    *,
    now: datetime | None = None,
    max_age: timedelta = MANAGED_RUN_MAX_RESUME_AGE,
    max_resumes: int = MANAGED_RUN_MAX_RESUMES,
) -> bool:
    if MANAGED_RUN_STARTED_METADATA_KEY not in task.metadata:
        return False
    if managed_run_resume_attempts(task) >= max_resumes:
        return False
    age = managed_run_started_age(task, now=now)
    if age is None:
        return False
    return age <= max_age


def _initial_allowed_tools(
    provider: Provider,
    allowed_tools: tuple[str, ...],
    *,
    claude_channel: bool,
) -> tuple[str, ...]:
    if provider == Provider.CLAUDE and claude_channel:
        return _append_allowed_tools(allowed_tools, SLACKGENTIC_MCP_PERMISSION_ALLOW)
    return allowed_tools


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
            "For ongoing work, send a concise Slack-visible progress update at least every "
            "5 minutes, and sooner for meaningful progress, blockers, or decisions."
        ),
        (
            "When a progress update changes what the roster should say you are working on, "
            "add a hidden roster line on its own line in the exact form "
            f"`{AGENT_ROSTER_STATUS_SIGNAL_PREFIX}<one-line summary>`. The summary should "
            "name the broader goal and current phase, not just the latest tiny step. "
            "Slackgentic hides that line and updates the roster."
        ),
        (
            "Direct Slack questions are not optional. If a Slack reply asks a question, "
            "answer it explicitly in Slack before continuing implementation work, opening "
            "a PR, or scheduling a delayed follow-up."
        ),
        (
            "For delayed follow-ups, do not rely on terminal sleeps or background timers. "
            "Send a concise Slack-visible status update, then put a hidden timer control "
            "line on its own final line in the exact form "
            f"`{AGENT_TIMER_SIGNAL_PREFIX}<delay-or-UTC-time> | <instruction>`, for example "
            f"`{AGENT_TIMER_SIGNAL_PREFIX}10m | Re-check the PR comments and CI, then keep "
            "iterating until they are clear.` Slackgentic hides that line and resumes the "
            "same agent in this thread when the timer is due."
        ),
        (
            "When a table is the clearest format, write one normal Markdown table in the "
            "message. Slackgentic renders one Markdown table per message as a native Slack "
            "table. If you need multiple tables, send separate messages."
        ),
        (
            "Slack reactions are a natural way to acknowledge the most recent message in "
            "this thread, from the user or from another agent, without sending a new "
            "reply. Add a hidden line on its own line in the exact form "
            f"`{AGENT_REACTION_SIGNAL_PREFIX}<emoji-name>` and Slackgentic will hide the "
            "line and react to that latest message. Reach for one when it fits the "
            f"moment: `{AGENT_REACTION_SIGNAL_PREFIX}eyes` when you're picking something "
            f"up, `{AGENT_REACTION_SIGNAL_PREFIX}thumbsup` to agree with a teammate, "
            f"`{AGENT_REACTION_SIGNAL_PREFIX}white_check_mark` to confirm something is "
            f"done, `{AGENT_REACTION_SIGNAL_PREFIX}tada` when a teammate ships something, "
            f"`{AGENT_REACTION_SIGNAL_PREFIX}thinking_face` when a call is genuinely "
            "close. Aim for roughly one reaction every few of your replies, not on "
            "every message, and skip it when a real reply already covers what you'd "
            "say."
        ),
        (
            "When the user explicitly asks for another agent, someone else, a review, "
            "or a second opinion, that request is mandatory. Send one separate "
            "Slack-visible message beginning exactly `somebody review ...` with the "
            "concrete item to review. After that message, stop and wait; Slackgentic "
            "will route the review and resume you with the review context. Do not "
            "continue implementation work, open a PR, schedule a delayed follow-up, "
            "or send a final answer as a substitute before the review context returns. "
            "If routing fails or no review context arrives, say that concrete blocker "
            "in Slack and use an available delegation fallback when possible."
        ),
        (
            "If the user is clearly closing the whole Slack thread or says no more work "
            f"is needed, write your normal brief final reply and then add a final line "
            f"exactly `{AGENT_THREAD_DONE_SIGNAL}`. Slackgentic hides that line and "
            "marks the whole thread done. Do not use this signal just because your "
            "current task is complete; use it only when the entire thread should be closed."
        ),
        (
            "When you believe the work the Slack user asked for is done and you have "
            "nothing else queued, send one short Slack-visible closing message that names "
            "what landed in a single line and asks whether there is anything else or "
            "whether you should be released. Then stop. Do not emit "
            f"`{AGENT_THREAD_DONE_SIGNAL}`, do not open a new turn on your own, and do "
            "not silently exit — Slackgentic will surface a release button under your "
            "closing message so the Slack user can free you up with one click or reply "
            "with the next step."
        ),
        (
            "When you hand work to a specific agent, use that agent's exact Slackgentic "
            "@handle from the thread or roster. Write the handle as plain text with no "
            "backticks or inline code formatting. Put that plain @handle at the start "
            "of its own final paragraph when it is meant to route work or get that "
            "agent to look at something."
        ),
        (
            "When you need a named external/session agent in the Slack thread to choose, "
            "continue, or respond, put your context and options first. Then end with a "
            "separate final paragraph whose first token is that agent's plain @handle, "
            "for example @nell pick one before I proceed. Do not put that callback "
            "handle inline near the beginning, wrap it in backticks, or bury it before "
            "the options."
        ),
        (
            "Do not write raw Slack user IDs such as U12345 or @U12345 in visible replies. "
            "Use the person's display name when known, or say the Slack user."
        ),
        (
            "When opening a GitHub pull request, prefer Slackgentic's "
            "`create_pull_request` MCP tool if it is available; it runs the narrow PR "
            "creation workflow through Slackgentic so PR creation still works when "
            "ordinary shell networking or sandbox policy gets in the way."
        ),
    ]
    if _task_dangerous_mode(task):
        lines.append(
            "Dangerous mode is enabled for this task. Do not request Slack approval or "
            "host-tool escalation for command, file, network, or sandbox access; this "
            "run is already launched with no-approval permissions. Only ask Slack when "
            "you need an actual user decision."
        )
    lines.extend(
        [
            f"Task kind: {task.kind.value}",
            f"Task: {task.prompt}",
        ]
    )
    pr_urls = pr_urls_from_metadata(task.metadata)
    if pr_urls:
        if len(pr_urls) == 1:
            lines.append(f"Pull request URL: {pr_urls[0]}")
        else:
            lines.append("Pull request URLs:")
            lines.extend(f"- {url}" for url in pr_urls)
        lines.append("Review the PR(s) and report findings with concrete file or diff references.")
    thread_context = task.metadata.get("thread_context")
    if isinstance(thread_context, str) and thread_context.strip():
        sanitized_thread_context = replace_slack_user_ids(thread_context.strip())
        lines.extend(
            [
                "",
                "Private Slack thread context. Use this to understand prior messages, "
                "handoffs, reviews, and user intent. Do not quote this heading or describe "
                "it as hidden context in Slack-visible replies:",
                sanitized_thread_context,
            ]
        )
    return "\n".join(lines)


def _task_dangerous_mode(task: AgentTask) -> bool:
    return task_permission_mode(task) == PermissionMode.DANGEROUS


def _task_cwd(task: AgentTask, default_cwd: Path) -> Path:
    configured = task.metadata.get("cwd")
    if isinstance(configured, str) and configured:
        path = Path(configured).expanduser()
        if path.exists():
            return path
    return _requested_repo_cwd(task.prompt, default_cwd)


def _safe_auto_extra_roots(provider: Provider, cwd: Path, default_cwd: Path) -> tuple[Path, ...]:
    if provider not in {Provider.CODEX, Provider.CLAUDE}:
        return ()
    root = _resolved_path(default_cwd)
    if not root.exists() or not root.is_dir():
        return ()
    resolved_cwd = _resolved_path(cwd)
    if not (_path_is_relative_to(resolved_cwd, root) or _path_is_relative_to(root, resolved_cwd)):
        return ()
    return (root,)


def _claude_effort_from_settings(cwd: Path, home: Path) -> str:
    effort: str | None = None
    for path in _claude_settings_paths(cwd, home):
        value = _claude_effort_from_settings_file(path)
        if value is not None:
            effort = value
    return effort or CLAUDE_DEFAULT_EFFORT


def _claude_settings_paths(cwd: Path, home: Path) -> tuple[Path, ...]:
    paths: list[Path] = [
        home / ".claude" / "settings.json",
        home / ".claude" / "settings.local.json",
    ]
    for root in reversed((cwd, *cwd.parents)):
        paths.append(root / ".claude" / "settings.json")
        paths.append(root / ".claude" / "settings.local.json")
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(_resolved_path(path))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return tuple(deduped)


def _claude_effort_from_settings_file(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get(CLAUDE_EFFORT_SETTING)
    if not isinstance(value, str):
        return None
    effort = value.strip().lower()
    return effort if effort in CLAUDE_EFFORT_LEVELS else None


def _macos_tcc_protected_cwd_issue(
    cwd: Path,
    *,
    home: Path | None = None,
    allow: bool = False,
) -> str | None:
    if allow or platform.system().lower() != "darwin":
        return None
    path = _resolved_path(cwd)
    home_paths = [_resolved_path(home)] if home is not None else []
    real_home = _resolved_path(Path.home())
    if real_home not in home_paths:
        home_paths.append(real_home)
    for home_path in home_paths:
        for dirname in MACOS_TCC_PROTECTED_HOME_DIRS:
            protected_root = home_path / dirname
            if _path_is_relative_to(path, protected_root):
                return f"the working directory {path} is inside macOS-protected {dirname}"
    if _path_is_relative_to(path, Path("/Volumes")):
        return f"the working directory {path} is on a macOS-protected mounted volume"
    return None


def _resolved_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except OSError:
        return path.expanduser().absolute()


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


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
        if len(cleaned) <= limit:
            chunks.append(cleaned)
            break
        cut = _slack_chunk_break(cleaned, limit)
        chunks.append(cleaned[:cut].rstrip())
        cleaned = cleaned[cut:].lstrip()
    return [chunk for chunk in chunks if chunk]


def _slack_chunk_break(text: str, limit: int) -> int:
    window = text[:limit]
    for separator in ("\n\n", "\n", ". ", " "):
        idx = window.rfind(separator)
        if idx > 0:
            return idx + len(separator)
    return limit


def _extract_agent_control_signals(text: str) -> tuple[str, list[str]]:
    visible_lines: list[str] = []
    signals: list[str] = []
    for line in text.splitlines():
        if _is_agent_transport_leak(line):
            continue
        normalized = re.sub(r"\s+", " ", line.strip()).upper()
        if normalized == AGENT_THREAD_DONE_SIGNAL:
            signals.append(AGENT_THREAD_DONE_SIGNAL)
            continue
        if normalized.startswith(AGENT_TIMER_SIGNAL_PREFIX):
            signals.append(line.strip())
            continue
        if normalized.startswith(AGENT_ROSTER_STATUS_SIGNAL_PREFIX):
            signals.append(line.strip())
            continue
        if normalized.startswith(AGENT_REACTION_SIGNAL_PREFIX):
            signals.append(line.strip())
            continue
        if normalized.startswith(AGENT_SCHEDULE_SIGNAL_PREFIX):
            signals.append(line.strip())
            continue
        if normalized.startswith(AGENT_DEFERRED_SIGNAL_PREFIX):
            signals.append(line.strip())
            continue
        if normalized.startswith(AGENT_PM_PLAN_SIGNAL_PREFIX):
            signals.append(line.strip())
            continue
        visible_lines.append(line)
    return "\n".join(visible_lines).strip(), signals


def _is_schedule_control_signal(signal: str) -> bool:
    normalized = re.sub(r"\s+", " ", signal.strip()).upper()
    return normalized.startswith(AGENT_SCHEDULE_SIGNAL_PREFIX)


def is_agent_roster_status_signal(signal: str) -> bool:
    return re.sub(r"\s+", " ", signal.strip()).upper().startswith(AGENT_ROSTER_STATUS_SIGNAL_PREFIX)


def is_agent_reaction_signal(signal: str) -> bool:
    return re.sub(r"\s+", " ", signal.strip()).upper().startswith(AGENT_REACTION_SIGNAL_PREFIX)


def parse_agent_reaction_signal(signal: str) -> str | None:
    if not is_agent_reaction_signal(signal):
        return None
    reaction = signal.strip()[len(AGENT_REACTION_SIGNAL_PREFIX) :].strip().strip(":")
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", reaction):
        return None
    return reaction.lower()


def parse_agent_roster_status_signal(signal: str) -> str | None:
    if not is_agent_roster_status_signal(signal):
        return None
    summary = signal.strip()[len(AGENT_ROSTER_STATUS_SIGNAL_PREFIX) :].strip()
    return summary or None


def _process_output_chunks(
    provider: Provider,
    text: str,
    buffer: str = "",
    final: bool = False,
) -> tuple[list[str], str]:
    if provider == Provider.CODEX:
        return _codex_exec_chunks(text, buffer, final=final)
    return _claude_json_chunks(text, buffer, final=final)


def _claude_permission_denials(
    text: str,
    buffer: str = "",
    final: bool = False,
    tool_uses: dict[str, dict] | None = None,
) -> tuple[list[dict], str]:
    combined = buffer + text
    if not combined:
        return [], buffer
    lines = combined.splitlines(keepends=True)
    next_buffer = ""
    if lines and not _line_has_ending(lines[-1]) and not final:
        next_buffer = lines.pop()

    denials: list[dict] = []
    for line in lines:
        try:
            event = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if tool_uses is not None:
            _remember_claude_tool_uses(event, tool_uses)
        denials.extend(_claude_result_permission_denials(event))
        if tool_uses is not None:
            denials.extend(_claude_stream_permission_denials(event, tool_uses))
    return denials, next_buffer


def _claude_missing_resume_session(
    text: str,
    buffer: str = "",
    final: bool = False,
) -> tuple[bool, str]:
    combined = buffer + text
    if not combined:
        return False, buffer
    lines = combined.splitlines(keepends=True)
    next_buffer = ""
    if lines and not _line_has_ending(lines[-1]) and not final:
        next_buffer = lines.pop()

    missing = any(_claude_line_missing_resume_session(line.strip()) for line in lines)
    return missing, next_buffer


def _claude_line_missing_resume_session(line: str) -> bool:
    if "No conversation found with session ID:" in line:
        return True
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return False
    if not isinstance(event, dict):
        return False
    errors = event.get("errors")
    if not isinstance(errors, list):
        return False
    return any(
        isinstance(error, str) and "No conversation found with session ID:" in error
        for error in errors
    )


_EMPTY_TEXT_BLOCK_API_ERROR_NEEDLE = "text content blocks must be non-empty"


def _claude_empty_text_block_api_error(
    text: str,
    buffer: str = "",
    final: bool = False,
) -> tuple[bool, str]:
    combined = buffer + text
    if not combined:
        return False, buffer
    lines = combined.splitlines(keepends=True)
    next_buffer = ""
    if lines and not _line_has_ending(lines[-1]) and not final:
        next_buffer = lines.pop()
    detected = any(_claude_line_has_empty_text_block_api_error(line.strip()) for line in lines)
    return detected, next_buffer


def _claude_line_has_empty_text_block_api_error(line: str) -> bool:
    # Only match the error when it surfaces in the SDK's own output channels
    # (a top-level assistant text block or a result-event error string).
    # Plain substring matching would also fire on sub-agent tool_result lines
    # carrying the same error text, which is a recoverable nested-agent case
    # that does not require restarting the main session.
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return False
    if not isinstance(event, dict):
        return False
    if event.get("type") == "assistant":
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    continue
                value = part.get("text")
                if isinstance(value, str) and _EMPTY_TEXT_BLOCK_API_ERROR_NEEDLE in value:
                    return True
    errors = event.get("errors")
    return isinstance(errors, list) and any(
        isinstance(err, str) and _EMPTY_TEXT_BLOCK_API_ERROR_NEEDLE in err for err in errors
    )


def managed_run_empty_text_block_api_retries(task: AgentTask) -> int:
    value = task.metadata.get(MANAGED_RUN_EMPTY_TEXT_BLOCK_API_RETRIES_METADATA_KEY)
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _claude_result_permission_denials(event: object) -> list[dict]:
    if not isinstance(event, dict) or event.get("type") != "result":
        return []
    denials = event.get("permission_denials")
    if not isinstance(denials, list):
        return []
    return [denial for denial in denials if isinstance(denial, dict)]


def _remember_claude_tool_uses(event: dict, tool_uses: dict[str, dict]) -> None:
    if event.get("type") != "assistant":
        return
    message = event.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool_use":
            continue
        tool_use_id = item.get("id")
        tool_name = item.get("name")
        if not isinstance(tool_use_id, str) or not isinstance(tool_name, str):
            continue
        tool_input = item.get("input")
        tool_uses[tool_use_id] = {
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "tool_input": tool_input if isinstance(tool_input, dict) else {},
        }


def _claude_stream_permission_denials(event: dict, tool_uses: dict[str, dict]) -> list[dict]:
    if event.get("type") != "user":
        return []
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    denials: list[dict] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool_result":
            continue
        if not item.get("is_error"):
            continue
        tool_use_id = item.get("tool_use_id")
        if not isinstance(tool_use_id, str):
            continue
        if not _claude_tool_result_is_permission_denial(item):
            continue
        tool_use = tool_uses.get(tool_use_id)
        if tool_use:
            denials.append(dict(tool_use))
    return denials


def _claude_tool_result_is_permission_denial(item: dict) -> bool:
    values = [item.get("content"), item.get("toolUseResult")]
    text = "\n".join(str(value) for value in values if value)
    text = text.lower()
    return (
        "requires approval" in text
        or "approve before running" in text
        or "requested permissions" in text
        or "haven't granted it yet" in text
    )


def _claude_denial_key(denial: dict) -> str:
    tool_name = str(denial.get("tool_name") or "")
    tool_input = denial.get("tool_input")
    normalized_input = tool_input if isinstance(tool_input, dict) else {}
    return json.dumps(
        {"tool_name": tool_name, "tool_input": normalized_input},
        sort_keys=True,
        default=str,
    )


def _allowed_tool_for_claude_denial(denial: dict) -> str | None:
    allowed_tools = _allowed_tools_for_claude_denial(denial)
    return allowed_tools[0] if allowed_tools else None


def _allowed_tools_for_claude_denial(denial: dict) -> tuple[str, ...]:
    tool_name = str(denial.get("tool_name") or "")
    tool_input = denial.get("tool_input")
    if tool_name == "Bash" and isinstance(tool_input, dict):
        command = tool_input.get("command")
        if isinstance(command, str):
            return _allowed_bash_tools_for_command(command)
    if tool_name in {"Edit", "MultiEdit", "Write"}:
        return (tool_name,)
    return ()


def _claude_safe_auto_denial_allowed(running: RunningTask, denial: dict) -> bool:
    return bool(_safe_auto_allowed_tools_for_claude_denial(running, denial))


def _safe_auto_allowed_tools_for_claude_denial(
    running: RunningTask,
    denial: dict,
) -> tuple[str, ...]:
    if running.process.request.permission_mode != PermissionMode.SAFE_AUTO:
        return ()
    tool_name = str(denial.get("tool_name") or "")
    tool_input = denial.get("tool_input")
    if tool_name == "Bash" and isinstance(tool_input, dict):
        command = tool_input.get("command")
        if not isinstance(command, str):
            return ()
        decision = classify_bash_command(command)
        if not decision.safe:
            LOGGER.info("safe-auto denied Claude Bash runtime retry: %s", decision.reason)
            return ()
        return decision.safe_allowed_tools
    params: dict[str, object] = {"tool_name": tool_name}
    if isinstance(tool_input, dict):
        params["tool_input"] = tool_input
    if not claude_safe_auto_permission_request_allowed(params):
        return ()
    allowed_tools = _allowed_tools_for_claude_denial(denial)
    if allowed_tools:
        return allowed_tools
    return (tool_name,) if tool_name else ()


def _slackgentic_mcp_tool_for_claude_denial(denial: dict) -> str | None:
    tool_name = str(denial.get("tool_name") or "")
    if tool_name in SLACKGENTIC_MCP_PERMISSION_ALLOW:
        return tool_name
    return None


def _allowed_session_tools_for_claude_denial(denial: dict) -> tuple[str, ...]:
    tool_name = str(denial.get("tool_name") or "")
    tool_input = denial.get("tool_input")
    if tool_name == "Bash" and isinstance(tool_input, dict):
        command = tool_input.get("command")
        if isinstance(command, str):
            return _allowed_bash_session_tools_for_command(command)
    return ()


def _append_allowed_tool(existing: tuple[str, ...], allowed_tool: str) -> tuple[str, ...]:
    if allowed_tool in existing:
        return existing
    return (*existing, allowed_tool)


def _append_allowed_tools(
    existing: tuple[str, ...],
    allowed_tools: tuple[str, ...],
) -> tuple[str, ...]:
    next_tools = existing
    for allowed_tool in allowed_tools:
        next_tools = _append_allowed_tool(next_tools, allowed_tool)
    return next_tools


def _allowed_bash_tools_for_command(command: str) -> tuple[str, ...]:
    return allowed_bash_tools_for_command(command)


def _allowed_bash_session_tools_for_command(command: str) -> tuple[str, ...]:
    return allowed_bash_session_tools_for_command(command)


def _allowed_bash_tool_for_command(command: str) -> str | None:
    allowed_tools = _allowed_bash_tools_for_command(command)
    return allowed_tools[0] if allowed_tools else None


def _claude_denial_request_params(denial: dict) -> dict[str, object]:
    tool_name = str(denial.get("tool_name") or "tool")
    tool_input = denial.get("tool_input")
    params: dict[str, object] = {
        "request_id": token_urlsafe(6),
        "tool_name": tool_name,
        "can_allow_session": True,
    }
    if isinstance(tool_input, dict):
        description = tool_input.get("description")
        if isinstance(description, str) and description.strip():
            params["description"] = description
        params["input_preview"] = json.dumps(tool_input, indent=2, sort_keys=True)
    return params


def _claude_permission_status_text(denial: dict) -> str | None:
    tool_name = str(denial.get("tool_name") or "tool")
    tool_input = denial.get("tool_input")
    if not isinstance(tool_input, dict):
        return f"I'm blocked on approval before I can continue: use Claude `{tool_name}`."

    description = tool_input.get("description")
    description_text = description.strip() if isinstance(description, str) else ""
    if tool_name == "Bash":
        command = tool_input.get("command")
        if isinstance(command, str) and command.strip():
            if description_text:
                return (
                    "I'm blocked on approval before I can continue: "
                    f"{_status_sentence(description_text)} "
                    f"(`{_status_inline_code(command.strip(), 180)}`)."
                )
            return (
                "I'm blocked on approval before I can continue: "
                f"run `{_status_inline_code(command.strip(), 220)}`."
            )
    if tool_name in {"Edit", "MultiEdit", "Write"}:
        file_path = tool_input.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            verb = {
                "Edit": "edit",
                "MultiEdit": "edit",
                "Write": "write",
            }.get(tool_name, "change")
            return (
                "I'm blocked on approval before I can continue: "
                f"{verb} `{_status_inline_code(file_path.strip(), 220)}`."
            )
    if description_text:
        return (
            f"I'm blocked on approval before I can continue: {_status_sentence(description_text)}."
        )
    return f"I'm blocked on approval before I can continue: use Claude `{tool_name}`."


def _status_sentence(value: str) -> str:
    value = value.strip().rstrip(".")
    if not value:
        return value
    return value[:1].lower() + value[1:]


def _status_inline_code(value: str, limit: int) -> str:
    compact = " ".join(value.replace("`", "'").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "..."


def _claude_permission_retry_prompt(denial: dict) -> str:
    if _slackgentic_mcp_tool_for_claude_denial(denial):
        return (
            "The internal Slackgentic Slack request tool is now allowed. "
            "Retry the Slack request if it is still needed, then continue the task."
        )
    tool_input = denial.get("tool_input")
    if isinstance(tool_input, dict):
        command = tool_input.get("command")
        if isinstance(command, str) and command.strip():
            return (
                "The Slack user approved the previously denied Bash command. "
                f"Run this exact approved command if it is still needed, then continue the task: {command}"
            )
    return "The Slack user approved the previously denied tool request. Continue the task."


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
        if event.get("type") == "result" and (
            event.get("is_error") or str(event.get("subtype") or "").startswith("error")
        ):
            return None
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
    if _is_agent_transport_leak(line):
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        if line.lstrip().startswith("{"):
            return None
        cleaned = _clean_terminal_output(line)
        if cleaned.lower().startswith(("error:", "codex error:")):
            return cleaned
        return None
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


def _transcript_activity_signature_for_path(path: Path) -> TranscriptActivitySignature | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_mtime_ns, stat.st_size)


def _session_transcript_activity_snapshot(
    provider: Provider,
    session_id: str | None,
    home: Path,
) -> TranscriptActivitySnapshot | None:
    if not session_id:
        return None
    if provider == Provider.CODEX:
        root = home / ".codex" / "sessions"
        pattern = f"*{session_id}.jsonl"
    elif provider == Provider.CLAUDE:
        root = home / ".claude" / "projects"
        pattern = f"{session_id}.jsonl"
    else:
        return None
    if not root.exists():
        return None
    latest: tuple[int, int, str, TranscriptActivitySnapshot] | None = None
    try:
        for path in root.rglob(pattern):
            signature = _transcript_activity_signature_for_path(path)
            if signature is None:
                continue
            _, mtime_ns, size = signature
            candidate = (mtime_ns, size, str(path), (path, signature))
            if latest is None or candidate[:3] > latest[:3]:
                latest = candidate
    except OSError:
        return None
    return latest[3] if latest is not None else None


def _latest_codex_transcript_message(
    session_id: str,
    home: Path,
    *,
    since: datetime | None = None,
) -> str | None:
    messages = _codex_transcript_messages(session_id, home, since=since)
    return messages[-1] if messages else None


def _codex_transcript_messages(
    session_id: str,
    home: Path,
    *,
    since: datetime | None = None,
) -> list[str]:
    sessions_root = home / ".codex" / "sessions"
    if not sessions_root.exists():
        return []
    try:
        paths = sorted(
            sessions_root.rglob(f"*{session_id}.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    for path in paths:
        messages = _codex_transcript_messages_from_path(path, since=since)
        if messages:
            return messages
    return []


def _latest_codex_transcript_message_from_path(
    path: Path,
    *,
    since: datetime | None = None,
) -> str | None:
    messages = _codex_transcript_messages_from_path(path, since=since)
    return messages[-1] if messages else None


def _codex_transcript_messages_from_path(
    path: Path,
    *,
    since: datetime | None = None,
) -> list[str]:
    messages: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
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
            messages.append(message)
    return messages


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


def _latest_claude_transcript_message(
    session_id: str,
    home: Path,
    *,
    since: datetime | None = None,
) -> str | None:
    messages = _claude_transcript_messages(session_id, home, since=since)
    return messages[-1] if messages else None


def _claude_transcript_messages(
    session_id: str,
    home: Path,
    *,
    since: datetime | None = None,
) -> list[str]:
    projects_root = home / ".claude" / "projects"
    if not projects_root.exists():
        return []
    try:
        paths = sorted(
            projects_root.rglob(f"{session_id}.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    for path in paths:
        messages = _claude_transcript_messages_from_path(path, since=since)
        if messages:
            return messages
    return []


def _latest_claude_transcript_message_from_path(
    path: Path,
    *,
    since: datetime | None = None,
) -> str | None:
    messages = _claude_transcript_messages_from_path(path, since=since)
    return messages[-1] if messages else None


def _claude_transcript_messages_from_path(
    path: Path,
    *,
    since: datetime | None = None,
) -> list[str]:
    messages: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("type") != "assistant":
            continue
        if is_synthetic_claude_assistant_record(record):
            continue
        timestamp = parse_timestamp(record.get("timestamp"))
        if since is not None and timestamp is not None and timestamp < since:
            continue
        message = _claude_assistant_message_text(record)
        if message:
            messages.append(message)
    return messages


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
        if _claude_result_permission_denials(event):
            return None
        if event.get("is_error"):
            message = event.get("result") or event.get("api_error_status") or event.get("subtype")
            return _format_claude_error(message)
        result = event.get("result")
        return _clean_terminal_output(str(result)) if result else None
    if event_type == "assistant":
        if is_synthetic_claude_assistant_record(event):
            return None
        return _claude_assistant_message_text(event)
    if event_type == "error":
        message = event.get("message") or event.get("error")
        return _format_claude_error(message)
    return None


def _claude_assistant_message_text(record: dict) -> str | None:
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        cleaned = _clean_terminal_output(content)
        return cleaned or None
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        cleaned = _clean_terminal_output(text)
        if cleaned:
            parts.append(cleaned)
    if not parts:
        return None
    return "\n\n".join(parts)


def _format_claude_error(message: object) -> str:
    return f"Claude error: {_clean_terminal_output(str(message))}" if message else "Claude error."


def _line_has_ending(line: str) -> bool:
    return line.endswith(("\n", "\r"))


def _is_agent_transport_leak(text: str) -> bool:
    cleaned = _clean_terminal_output(text)
    normalized = " ".join(cleaned.split())
    if not normalized:
        return False
    if "write_stdin failed: Unknown process id" in normalized:
        return True
    if "codex_core::tools::router" in normalized and (
        "ERROR" in normalized or "error=" in normalized or "failed" in normalized
    ):
        return True
    if re.search(r'"exit_code"\s*:\s*-?\d+', normalized) and re.search(
        r'"status"\s*:\s*"(?:completed|failed|running)"',
        normalized,
    ):
        return True
    return bool(re.search(r'"recipient_name"\s*:\s*"functions\.', normalized))


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
