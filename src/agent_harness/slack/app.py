from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

from agent_harness.config import AppConfig, load_config_from_env
from agent_harness.deferred import (
    DEFERRED_RESOLUTION_ATTEMPTS_METADATA_KEY,
    DEFERRED_RESOLUTION_METADATA_KEY,
    DEFERRED_RESOLUTION_OCCUPIED_HANDLES_METADATA_KEY,
    DEFERRED_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY,
    MAX_DEFERRED_RESOLUTION_ATTEMPTS,
    build_deferred_resolution_prompt,
    is_agent_deferred_signal,
    looks_like_deferred_request,
    parse_agent_deferred_signal,
)
from agent_harness.internal_notifications import is_internal_task_notification_text
from agent_harness.models import (
    ASSIGNMENT_PROMPT_METADATA_KEY,
    DANGEROUS_MODE_METADATA_KEY,
    DEFAULT_PERMISSION_MODE,
    ORIGINAL_TASK_METADATA_KEY,
    PERMISSION_MODE_METADATA_KEY,
    PR_URL_METADATA_KEY,
    PR_URLS_METADATA_KEY,
    ROSTER_SUMMARY_METADATA_KEY,
    AgentSession,
    AgentTask,
    AgentTaskKind,
    AgentTaskStatus,
    AssignmentMode,
    DeferredWork,
    DeferredWorkStatus,
    PendingWorkRequest,
    PendingWorkRequestStatus,
    PermissionMode,
    PmInitiative,
    PmInitiativeStatus,
    PmSubtask,
    Provider,
    ScheduledTimer,
    ScheduledTimerStatus,
    ScheduledWork,
    ScheduledWorkKind,
    ScheduledWorkStatus,
    SessionDependency,
    SessionStatus,
    SlackThreadRef,
    TeamAgent,
    TeamAgentKind,
    TeamAgentStatus,
    WorkDependency,
    WorkDependencyKind,
    WorkRequest,
    deferred_work_dependency_id,
    external_session_dependency_id,
    parse_deferred_work_dependency_id,
    parse_external_session_dependency_id,
    parse_scheduled_work_dependency_id,
    parse_timestamp,
    scheduled_work_dependency_id,
    utc_now,
)
from agent_harness.pm import (
    MAX_PM_RESOLUTION_ATTEMPTS,
    MAX_PM_SUBTASKS,
    PM_EXTENSION_CONTEXT_METADATA_KEY,
    PM_EXTENSION_KNOWN_IDS_METADATA_KEY,
    PM_INITIATIVE_ID_METADATA_KEY,
    PM_REPLAN_CONTEXT_METADATA_KEY,
    PM_RESOLUTION_ATTEMPTS_METADATA_KEY,
    PM_RESOLUTION_METADATA_KEY,
    PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY,
    PM_SUBTASK_LOCAL_ID_METADATA_KEY,
    ParsedPmPlan,
    ParsedPmSubtask,
    PmPlanEstimate,
    build_pm_resolution_prompt,
    deserialize_parsed_pm_plan,
    estimate_pm_plan,
    expand_codesign_plan,
    extract_pm_request_body,
    filter_pm_agents,
    filter_worker_agents,
    is_agent_pm_plan_signal,
    looks_like_pm_request,
    looks_like_pm_status_request,
    message_targets_pm_agent,
    parse_agent_pm_plan_signal,
    parse_pm_extension_request,
    parse_pm_replan_request,
    render_pm_plan_dag,
    serialize_parsed_pm_plan,
)
from agent_harness.pr_links import metadata_with_pr_urls, pr_urls_from_metadata
from agent_harness.providers import ClaudeProvider, CodexProvider
from agent_harness.providers.usage import (
    collect_daily_usage,
    collect_weekly_usage,
    day_string,
    format_daily_usage,
)
from agent_harness.runtime.codex_app_server import (
    DEFAULT_CODEX_APP_SERVER_URL,
    CodexAppServerManager,
)
from agent_harness.runtime.health import LoopBackoff, ProcessCpuWatchdog, log_loop_failure
from agent_harness.runtime.power import ActiveSessionAwakeKeeper
from agent_harness.runtime.tasks import (
    AGENT_THREAD_DONE_SIGNAL,
    MANAGED_RUN_MAX_STALL_RECOVERIES,
    MANAGED_RUN_ORIGINAL_PROMPT_METADATA_KEY,
    MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY,
    MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY,
    MANAGED_RUN_STALL_TIMEOUT,
    MANAGED_RUN_STARTED_METADATA_KEY,
    ManagedTaskRuntime,
    managed_run_resume_attempts,
    managed_run_stall_recoveries,
    managed_run_started_age,
    parse_agent_reaction_signal,
    parse_agent_roster_status_signal,
    should_resume_managed_run,
)
from agent_harness.runtime.tasks import (
    SETTING_REPO_ROOT as TASK_RUNTIME_REPO_ROOT_SETTING,
)
from agent_harness.schedules import (
    MAX_SCHEDULE_RESOLUTION_ATTEMPTS,
    SCHEDULE_RESOLUTION_ATTEMPTS_METADATA_KEY,
    SCHEDULE_RESOLUTION_METADATA_KEY,
    SCHEDULE_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY,
    build_schedule_resolution_prompt,
    format_interval_seconds,
    interval_seconds_from_recurrence,
    is_agent_schedule_signal,
    looks_like_schedule_request,
    next_run_after,
    parse_agent_schedule_signal,
)
from agent_harness.sessions.bridge import ExternalSessionBridge
from agent_harness.sessions.claude_channel import (
    ensure_claude_native_input_hook,
    ensure_codex_mcp_server_registered,
    is_slackgentic_mcp_server_configured,
)
from agent_harness.sessions.managed_session import (
    clear_managed_session,
    managed_session_agents,
)
from agent_harness.sessions.mirror import (
    CAPACITY_NOTICE_TS_PREFIX,
    EXTERNAL_SESSION_AGENT_PREFIX,
    EXTERNAL_SESSION_IGNORED_PREFIX,
    EXTERNAL_SESSION_LIVE_TARGET_PREFIX,
    EXTERNAL_SESSION_MISSING_TARGET_PREFIX,
    EXTERNAL_SESSION_SUMMARY_PREFIX,
    HUMAN_DISPLAY_NAME_SETTING,
    HUMAN_IMAGE_URL_SETTING,
    PENDING_EXTERNAL_SESSION_PREFIX,
    SessionMirror,
    format_session_parent,
)
from agent_harness.slack import (
    IDLE_RELEASE_PROMPT_TEXT,
    AgentRosterStatus,
    build_channel_overview_blocks,
    build_external_session_capacity_blocks,
    build_idle_release_closed_blocks,
    build_idle_release_dismissed_blocks,
    build_idle_release_prompt_blocks,
    build_setup_modal,
    build_task_thread_blocks,
    build_team_roster_blocks,
    build_update_prompt_blocks,
    decode_action_value,
    encode_action_value,
    is_dependency_intent,
    parse_thread_ref,
    replace_slack_user_ids,
)
from agent_harness.slack.agent_requests import (
    AGENT_REQUEST_ACTIONS,
    render_persistent_agent_request,
)
from agent_harness.slack.client import SlackGateway
from agent_harness.storage.store import Store
from agent_harness.team import (
    AGENT_CONTEXT_PLACEHOLDER,
    AGENT_LIMIT_MESSAGE,
    DEFAULT_CLAUDE_TEAM_SIZE,
    DEFAULT_CODEX_TEAM_SIZE,
    MAX_TEAM_AGENTS,
    agent_personal_context,
    build_initialization_messages,
    create_agent_task,
    format_agent_assignment,
    format_agent_handoff_assignment,
    format_agent_handoff_request,
    format_agent_introduction,
    hire_team_agents,
    normalize_handle,
)
from agent_harness.team.assignment import assign_work_request
from agent_harness.team.commands import (
    FireCommand,
    FireEveryoneCommand,
    HireCommand,
    RepoRootCommand,
    RosterCommand,
    ScheduledTasksCommand,
    parse_team_command,
)
from agent_harness.team.routing import (
    canonicalize_agent_mentions,
    parse_lightweight_handles,
    parse_work_request,
    strip_dangerous_mode_tag,
)
from agent_harness.timers import is_agent_timer_signal, parse_agent_timer_signal
from agent_harness.updates import (
    GitHubReleaseSource,
    SelfUpdater,
    SlackgenticUpdateRunner,
    UpdateCandidate,
    UpdateChecker,
    detect_source_root,
)

LOGGER = logging.getLogger(__name__)
SETTING_CHANNEL_ID = "slack.channel_id"
SETTING_ROSTER_TS = "slack.roster_ts"
SETTING_ROSTER_MESSAGE_PREFIX = "slack.roster_message."
SETTING_ROSTER_DISCOVERY_PREFIX = "slack.roster_discovery."
SETTING_ROSTER_RENDER_HASH_PREFIX = "slack.roster_render_hash."
SETTING_ROSTER_PINNED_PREFIX = "slack.roster_pinned."
SETTING_USAGE_TS_PREFIX = "slack.usage_ts."
SETTING_HUMAN_USER_ID = "slack.human_user_id"
SETTING_HUMAN_USER_DISPLAY_NAME_PREFIX = "slack.user_display_name."
SETTING_REPO_ROOT = TASK_RUNTIME_REPO_ROOT_SETTING
SETTING_EXTERNAL_SESSION_DELIVERY_PREFIX = "external_session_delivery."
TASK_REACTION_ACKNOWLEDGED = "eyes"
TASK_REACTION_QUEUED = "inbox_tray"
TASK_REACTION_IN_PROGRESS = "hourglass_flowing_sand"
TASK_REACTION_DONE = "white_check_mark"
TASK_STATUS_REACTIONS = (
    TASK_REACTION_ACKNOWLEDGED,
    TASK_REACTION_QUEUED,
    TASK_REACTION_IN_PROGRESS,
    TASK_REACTION_DONE,
    "thumbsup",
    "warning",
    "test_tube",
    "rocket",
    "link",
    "memo",
    "thinking_face",
)
SETTING_AGENT_AVATAR_BASE_URL = "slack.agent_avatar_base_url"
SETTING_SLACK_BACKFILL_LAST_AWAKE = "slack.backfill.last_awake_unix"
SETTING_SLACK_BACKFILL_LAST_THREAD_SCAN = "slack.backfill.last_thread_scan_unix"
SETTING_SLACK_BACKFILL_THREAD_SCAN_PREFIX = "slack.backfill.thread_scan_unix."
SETTING_SLACK_MESSAGE_PROCESSED_PREFIX = "slack.message.processed."
SETTING_MESSAGE_STATUS_REACTION_PREFIX = "slack.message.status_reaction."
SETTING_AGENT_AUTHORED_MESSAGE_PREFIX = "slack.agent_message."
SETTING_SLACK_REACTION_PROCESSED_PREFIX = "slack.reaction.processed."
_PM_BLOCKER_PREFIX = "pm.blocker."
_PM_STATUS_MESSAGE_PREFIX = "pm.status_message."
_PM_APPROVAL_BLOCKER_SECONDS = 5 * 60
_PM_STALLED_TASK_SECONDS = 30 * 60
# Slack errors that indicate the initiative thread is permanently gone.
# A retry loop against a dead thread is pure noise, so the watchdog cancels
# the initiative once it sees one of these.
_PM_DEAD_THREAD_SLACK_ERRORS = frozenset(
    {
        "channel_not_found",
        "thread_not_found",
        "message_not_found",
        "is_archived",
        "not_in_channel",
    }
)


def _pm_blocker_setting_key(initiative_id: str, local_id: str, kind: str) -> str:
    safe_local = re.sub(r"[^A-Za-z0-9_.-]", "_", local_id) or "_"
    safe_kind = re.sub(r"[^A-Za-z0-9_.:-]", "_", kind) or "_"
    return f"{_PM_BLOCKER_PREFIX}{initiative_id}.{safe_local}.{safe_kind}"


def _pm_status_message_setting_key(initiative_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", initiative_id) or "_"
    return f"{_PM_STATUS_MESSAGE_PREFIX}{safe_id}"


def _pm_worker_handles(agents: list[TeamAgent] | tuple[TeamAgent, ...]) -> list[str]:
    return [agent.handle for agent in filter_worker_agents(agents)]


def _pm_worker_model_map(agents: list[TeamAgent] | tuple[TeamAgent, ...]) -> dict[str, str]:
    return {
        agent.handle: agent.provider_preference.value
        for agent in filter_worker_agents(agents)
        if agent.provider_preference is not None
    }


def _slack_error_code(exc: Exception) -> str | None:
    """Return the Slack ``error`` field for a SlackApiError, or None."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        code = response.get("error")
    except Exception:
        return None
    return code if isinstance(code, str) else None


SLACK_BACKFILL_FETCH_LIMIT = 500
SLACK_BACKFILL_KNOWN_THREAD_LIMIT = 200
SLACK_BACKFILL_GRACE_SECONDS = 5.0
SLACK_BACKFILL_SLEEP_GAP_SECONDS = 30.0
SLACK_BACKFILL_THREAD_POLL_SECONDS = 30.0
SLACK_BACKFILL_THREAD_INITIAL_LOOKBACK_SECONDS = 5 * 60.0
SLACK_SOCKET_WORKER_THREADS = 4
SLACK_SOCKET_MAX_PENDING_REQUESTS = 16
SLACK_SOCKET_PONG_GRACE_SECONDS = 30.0
SLACK_SOCKET_STALE_SECONDS = 30.0
DEFAULT_AGENT_AVATAR_BASE_URL = (
    "https://raw.githubusercontent.com/unbounded-ai/slackgentic-team/main/docs/assets/avatars"
)
DISABLED_AVATAR_BASE_VALUES = {"", "0", "false", "no", "none", "off"}
CAPACITY_MESSAGE = (
    "No agents are available right now. Hire more agents and I will resume this thread "
    "automatically."
)
CLAUDE_EXTERNAL_COMMAND = "claude --dangerously-load-development-channels server:slackgentic"
CLAUDE_CHANNEL_PERMISSION_METHOD = "claude/channel/permission"
SLACKGENTIC_MCP_PERMISSION_TOOLS = frozenset(
    {
        "mcp__slackgentic__request_approval",
        "mcp__slackgentic__request_user_input",
    }
)
_ACTIVE_PM_INITIATIVE_STATUSES = frozenset(
    {
        PmInitiativeStatus.PLANNING,
        PmInitiativeStatus.AWAITING_APPROVAL,
        PmInitiativeStatus.ACTIVE,
    }
)
AUTO_ALLOWED_CLAUDE_PERMISSION_TEXT = (
    "Allowed internal Claude Slackgentic request; rendering the Slack prompt now."
)
REVIEW_DELEGATE_PROMPT = (
    "Continue the original task using @{sender_handle}'s review above. "
    "Address any required changes, then give the user the final result "
    "or a concise status update. If no changes are needed, say so."
)
REVIEW_DELEGATE_VISIBLE_PROMPT = "continue using my review above."
THREAD_CONTEXT_DELEGATE_PROMPT = (
    "Continue the original task using the Slack thread context above. "
    "Address any required changes, then give the user the final result "
    "or a concise status update. If no changes are needed, say so."
)
THREAD_CONTEXT_DELEGATE_VISIBLE_PROMPT = "continue using the thread context above."
ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_METADATA_KEY = "active_thread_followup_message_ts"
ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_VALUES_METADATA_KEY = "active_thread_followup_message_ts_values"
IDLE_RELEASE_PROMPT_MESSAGE_TS_METADATA_KEY = "idle_release_prompt_message_ts"
LINKED_THREAD_ROUTE_INTENT_RE = re.compile(
    r"\b(?:this|the)\s+unrelated\s+thread\b"
    r"|\binstead\s+of\s+(?:this|the\s+current|current)\s+(?:thread|one)\b"
    r"|\boriginal\b.{0,120}\b(?:thread|one)\b"
    r"|\b(?:correct|right|target|linked)\s+(?:thread|one)\b"
    r"|\b(?:reply|respond|continue|post|route|move|send|put)\b.{0,80}"
    r"\b(?:there|(?:that|the|this|original|linked|correct|right)\s+(?:thread|one))\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class SlackReplyTarget:
    channel_id: str
    thread_ts: str | None = None


@dataclass(frozen=True)
class ThreadDelegationIntent:
    target_agent_id: str
    prompt_template: str
    visible_prompt_template: str | None = None


@dataclass(frozen=True)
class _PmReservedTask:
    subtask: ParsedPmSubtask
    agent: TeamAgent
    task: AgentTask
    thread: SlackThreadRef
    extra_depends_on: tuple[WorkDependency, ...]


@dataclass(frozen=True)
class _PmCapacityShortfall:
    required_workers: int
    available_workers: int
    hire_count: int
    unavailable_handles: tuple[str, ...] = ()


class SlackTeamController:
    def __init__(
        self,
        store: Store,
        gateway: SlackGateway,
        default_channel_id: str | None = None,
        runtime: ManagedTaskRuntime | None = None,
        home: Path | None = None,
        ignored_bot_id: str | None = None,
        session_bridge: ExternalSessionBridge | None = None,
        team_id: str = "local",
        codex_app_server_url: str | None = DEFAULT_CODEX_APP_SERVER_URL,
        slash_command: str = "/slackgentic",
        default_cwd: Path | None = None,
    ):
        self.store = store
        self.gateway = gateway
        self.default_channel_id = default_channel_id
        self.runtime = runtime
        self.home = home
        self.ignored_bot_id = ignored_bot_id
        self.session_bridge = session_bridge
        self.team_id = team_id
        self.codex_app_server_url = codex_app_server_url or DEFAULT_CODEX_APP_SERVER_URL
        self.slash_command = slash_command
        self.default_cwd = default_cwd or Path.cwd()
        self.update_runner: SlackgenticUpdateRunner | None = None
        self._slack_message_lock = threading.Lock()
        self._normalize_existing_agents()

    def set_update_runner(self, update_runner: SlackgenticUpdateRunner | None) -> None:
        self.update_runner = update_runner

    def _run_after_view_ack(self, label: str, callback: Callable[[], None]) -> None:
        def run() -> None:
            try:
                callback()
            except Exception:
                LOGGER.exception("failed to complete Slack view submission: %s", label)

        threading.Thread(
            target=run,
            name=f"slack-view-{label}",
            daemon=True,
        ).start()

    def handle_block_action(self, payload: dict) -> None:
        action = _first_action(payload)
        if action is None:
            return
        decoded = decode_action_value(action.get("value") or "{}")
        action_name = decoded["action"]
        channel_id = _payload_channel_id(payload) or self.default_channel_id
        message_ts = _payload_message_ts(payload)
        if not channel_id:
            LOGGER.warning("Slack action had no channel_id")
            return
        if action_name == "team.hire":
            self._hire_from_action(decoded, channel_id, message_ts)
        elif action_name == "team.fire":
            self._fire_from_action(decoded, channel_id, message_ts)
        elif action_name == "roster.work.open":
            self._open_roster_work_modal(decoded, channel_id, message_ts, payload.get("trigger_id"))
        elif action_name.startswith("task."):
            self._task_from_action(decoded, channel_id, message_ts)
        elif action_name.startswith("schedule."):
            self._schedule_from_action(decoded, channel_id, message_ts, payload.get("trigger_id"))
        elif action_name in {"external.session.detach", "external.session.finish"}:
            self._external_session_detach_from_action(decoded, channel_id)
        elif action_name.startswith("update."):
            self._update_from_action(decoded, channel_id, message_ts)
        elif action_name.startswith("pm_initiative."):
            self._pm_initiative_from_action(decoded, payload, channel_id, message_ts)
        elif action_name in AGENT_REQUEST_ACTIONS and self.session_bridge is not None:
            self.session_bridge.handle_agent_request_block_action(decoded, channel_id, message_ts)

    def handle_view_submission(
        self,
        payload: dict,
        *,
        async_success: bool = False,
    ) -> dict | None:
        view = payload.get("view") or {}
        callback_id = view.get("callback_id")
        if callback_id == "schedule.change":
            return self._handle_schedule_change_submission(payload, async_success=async_success)
        if callback_id == "roster.work":
            return self._handle_roster_work_submission(payload, async_success=async_success)
        if callback_id != "setup.initial":
            return None
        values = view.get("state", {}).get("values", {})
        channel_name = _view_plain_value(values, "channel_name", "value") or "agents"
        visibility = _view_selected_value(values, "visibility", "value") or "private"
        codex_count = _view_int_value(
            values,
            "codex_count",
            "value",
            default=DEFAULT_CODEX_TEAM_SIZE,
        )
        claude_count = _view_int_value(
            values,
            "claude_count",
            "value",
            default=DEFAULT_CLAUDE_TEAM_SIZE,
        )
        if codex_count + claude_count > MAX_TEAM_AGENTS:
            return {
                "response_action": "errors",
                "errors": {
                    "codex_count": f"{AGENT_LIMIT_MESSAGE} Max team size is {MAX_TEAM_AGENTS}.",
                    "claude_count": f"{AGENT_LIMIT_MESSAGE} Max team size is {MAX_TEAM_AGENTS}.",
                },
            }
        repo_root_value = _view_plain_value(values, "repo_root", "value") or str(
            _suggested_repo_root(self.default_cwd)
        )
        repo_root = _validated_repo_root(repo_root_value)
        if repo_root is None:
            return {
                "response_action": "errors",
                "errors": {"repo_root": "Use an existing local folder path."},
            }

        channel_id = self.gateway.create_channel(
            _normalize_channel_name(channel_name),
            is_private=visibility != "public",
        )
        user_id = (payload.get("user") or {}).get("id")
        if user_id:
            self._remember_human_user(user_id, payload.get("user"))
            self.gateway.invite_users(channel_id, [user_id])
        self.store.set_setting(SETTING_CHANNEL_ID, channel_id)
        self.store.set_setting(SETTING_REPO_ROOT, str(repo_root))
        self._ensure_initial_team(codex_count, claude_count)
        self.post_channel_overview(channel_id)
        self.post_roster(channel_id)
        return None

    def handle_slash_command(self, payload: dict) -> None:
        text = (payload.get("text") or "").strip()
        channel_id = self._slash_command_target_channel_id(payload)
        if not channel_id:
            LOGGER.warning("Slack slash command had no channel_id")
            return
        self._remember_human_user(
            payload.get("user_id"),
            {"display_name": payload.get("user_name"), "name": payload.get("user_name")},
        )
        if text.lower() in {"setup", "init", "configure"}:
            trigger_id = payload.get("trigger_id")
            if trigger_id:
                self.gateway.open_view(
                    trigger_id,
                    build_setup_modal(default_repo_root=self._default_repo_root_for_setup()),
                )
            return
        usage_kind = _usage_request_kind(text)
        if usage_kind:
            self.publish_usage(
                channel_id,
                show_loading=True,
                fresh=usage_kind == "status",
            )
            return
        command = parse_team_command(text)
        if command:
            self.handle_team_command(
                command,
                SlackReplyTarget(channel_id=channel_id),
                requested_by_slack_user=payload.get("user_id"),
            )
            return
        self.refresh_or_post_roster(channel_id)

    def handle_event(self, payload: dict) -> None:
        event = payload.get("event") or {}
        if event.get("type") in {"reaction_added", "reaction_removed"}:
            self._handle_reaction_event(event)
            return
        channel_id, message_ts = self._recoverable_user_message_ref(event)
        if channel_id is None or message_ts is None:
            return
        with self._slack_message_lock:
            if self._slack_message_processed(channel_id, message_ts):
                return
            self._handle_unprocessed_user_message_event(event, channel_id)
            self._mark_slack_message_processed(channel_id, message_ts)

    def _handle_reaction_event(self, event: dict) -> None:
        event_type = event.get("type")
        item = event.get("item")
        if event_type not in {"reaction_added", "reaction_removed"}:
            return
        if not isinstance(item, dict) or item.get("type") != "message":
            return
        channel_id = item.get("channel")
        message_ts = item.get("ts")
        reaction_name = event.get("reaction")
        if not (
            isinstance(channel_id, str)
            and channel_id
            and isinstance(message_ts, str)
            and message_ts
            and isinstance(reaction_name, str)
            and reaction_name
        ):
            return
        if not self._is_agent_channel(channel_id):
            return
        user_id = event.get("user")
        if isinstance(user_id, str) and self._is_bot_user(user_id):
            return
        processed_key = _slack_reaction_processed_key(_reaction_event_dedupe_id(event))
        with self._slack_message_lock:
            if self.store.get_setting(processed_key):
                return
            handled = self._relay_user_reaction_to_agent(
                event_type=event_type,
                channel_id=channel_id,
                message_ts=message_ts,
                reaction_name=reaction_name,
                user_id=user_id if isinstance(user_id, str) else None,
            )
            if handled:
                self.store.set_setting(processed_key, utc_now().isoformat())

    def _relay_user_reaction_to_agent(
        self,
        *,
        event_type: str,
        channel_id: str,
        message_ts: str,
        reaction_name: str,
        user_id: str | None,
    ) -> bool:
        record = self._agent_authored_message_record(channel_id, message_ts)
        if record is None:
            return False
        task_id = _record_string(record, "task_id")
        if not task_id:
            return False
        task = self.store.get_agent_task(task_id)
        if task is None:
            return False
        thread_ts = _record_string(record, "thread_ts") or task.thread_ts
        if not thread_ts:
            return False
        agent_id = _record_string(record, "agent_id") or task.agent_id
        agent = self.store.get_team_agent(agent_id)
        if agent is None:
            return False
        active_task = self._active_thread_task_for_agent(task, channel_id, thread_ts)
        if active_task is None:
            return False
        if active_task.status != AgentTaskStatus.ACTIVE:
            return False
        is_running = getattr(self.runtime, "is_task_running", None)
        if not callable(is_running) or not is_running(active_task.task_id):
            return False
        self._remember_human_user(user_id)
        prompt = _reaction_relay_prompt(
            reaction_name,
            self._display_name_for_slack_user(user_id) if user_id else None,
            removed=event_type == "reaction_removed",
        )
        request = WorkRequest(
            prompt=prompt,
            assignment_mode=AssignmentMode.SPECIFIC,
            requested_handle=agent.handle,
        )
        return self._continue_same_thread_agent_task(
            request,
            active_task,
            agent,
            SlackThreadRef(channel_id, thread_ts, task.parent_message_ts),
            requested_by_slack_user=user_id,
        )

    def _is_bot_user(self, user_id: str) -> bool:
        try:
            bot_user_id = self.gateway.bot_user_id()
        except Exception:
            LOGGER.debug("failed to look up bot user id for reaction event", exc_info=True)
            bot_user_id = None
        return bool(bot_user_id and user_id == bot_user_id)

    def _remember_agent_authored_message(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        message_ts: str | None,
        text: str | None = None,
    ) -> None:
        if not message_ts:
            return
        payload = {
            "task_id": task.task_id,
            "agent_id": agent.agent_id,
            "thread_ts": thread.thread_ts,
        }
        if text:
            payload["text"] = _shorten(text, 500)
        try:
            self.store.set_setting(
                _agent_authored_message_setting_key(thread.channel_id, message_ts),
                json.dumps(payload, sort_keys=True),
            )
        except Exception:
            LOGGER.debug("failed to remember agent-authored Slack message", exc_info=True)

    def _agent_authored_message_record(
        self,
        channel_id: str,
        message_ts: str,
    ) -> dict[str, object] | None:
        raw = self.store.get_setting(_agent_authored_message_setting_key(channel_id, message_ts))
        if not raw:
            return None
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return record if isinstance(record, dict) else None

    def _recoverable_user_message_ref(self, event: dict) -> tuple[str | None, str | None]:
        event_type = event.get("type")
        if event_type not in {"message", "app_mention"}:
            return None, None
        if event.get("subtype"):
            return None, None
        bot_id = event.get("bot_id")
        if bot_id and (self.ignored_bot_id is None or bot_id == self.ignored_bot_id):
            return None, None
        if not event.get("user"):
            return None, None
        channel_id = event.get("channel")
        if not channel_id or not self._is_agent_channel(channel_id):
            return None, None
        message_ts = event.get("ts")
        if not message_ts:
            return None, None
        if self.store.is_slack_agent_request_message(channel_id, message_ts):
            return None, None
        if self.store.is_mirrored_slack_message(channel_id, message_ts):
            return None, None
        if self._has_work_for_request_message(channel_id, message_ts):
            return None, None
        return channel_id, message_ts

    def _handle_unprocessed_user_message_event(self, event: dict, channel_id: str) -> None:
        self._remember_human_user(event.get("user"), event.get("user_profile"))
        text = event.get("text") or ""
        if self._handle_external_session_thread_reply(event, channel_id, text):
            return
        usage_kind = _usage_request_kind(text)
        if usage_kind:
            self.publish_usage(
                channel_id,
                show_loading=True,
                fresh=usage_kind == "status",
            )
            return
        target = SlackReplyTarget(
            channel_id=channel_id,
            thread_ts=event.get("thread_ts"),
        )
        command = parse_team_command(text)
        if command:
            self.handle_team_command(command, target, requested_by_slack_user=event.get("user"))
            return
        if self._handle_pm_request(event, channel_id, text):
            return
        if self._handle_deferred_work_request(event, channel_id, text):
            return
        if self._handle_scheduled_work_request(event, channel_id, text):
            return
        if self._handle_task_thread_reply(event, channel_id, text):
            return
        active_agents = self.store.list_team_agents()
        thread = self._request_thread_anchor(event, channel_id, text)
        multi_requests = _multi_specific_work_requests(text, active_agents, split_newlines=True)
        if multi_requests:
            pm_targets_in_multi = [
                req.requested_handle
                for req in multi_requests
                if self._pm_agent_for_request_target(req) is not None
            ]
            if pm_targets_in_multi:
                self._mark_message_acknowledged(channel_id, event.get("ts"))
                handles_text = ", ".join(f"@{h}" for h in pm_targets_in_multi)
                noun = "PMs" if len(pm_targets_in_multi) > 1 else "a PM"
                self.gateway.post_thread_reply(
                    thread,
                    (
                        f"{handles_text} {'are' if len(pm_targets_in_multi) > 1 else 'is'} "
                        f"{noun} and must plan in their own thread. Address PMs "
                        "alone (`pm: <brief>` or `@<pm-handle> <brief>` on its "
                        "own line) so each PM initiative gets its own approval "
                        "card."
                    ),
                )
                return
            self._mark_message_acknowledged(channel_id, event.get("ts"))
            self._start_specific_requests_in_thread(
                multi_requests,
                thread,
                requested_by_slack_user=event.get("user"),
                extra_metadata=self._with_linked_thread_context(
                    {"request_message_ts": event.get("ts")},
                    text,
                    current_thread=thread,
                ),
            )
            return
        request = _channel_work_request(text, active_agents)
        if request is None:
            return
        pm_target = self._pm_agent_for_request_target(request)
        if pm_target is not None:
            # Defensive: `_handle_pm_request` already routes recognised
            # `@<pm-handle>` text through PM creation. If a brief somehow
            # reaches this fallback while still targeting a PM, redirect
            # rather than silently bypass the PM contract.
            self._mark_message_acknowledged(channel_id, event.get("ts"))
            dispatched = self._dispatch_pm_initiative(
                channel_id=channel_id,
                project_body=request.prompt.strip() or request.prompt,
                requested_by_slack_user=event.get("user"),
                thread=thread,
                targeted_pm_handle=pm_target.handle,
                request_message_ts=event.get("ts"),
                text_for_linked_context=text,
            )
            if dispatched:
                self._mark_message_in_progress(channel_id, event.get("ts"))
            return
        self._mark_message_acknowledged(channel_id, event.get("ts"))
        extra_metadata = self._with_linked_thread_context(
            {"request_message_ts": event.get("ts")},
            text,
            current_thread=thread,
        )
        author_agent = (
            self.store.get_team_agent(request.author_handle) if request.author_handle else None
        )
        sticky_excluded_agent_ids = self._anyone_context_agent_ids(request, active_agents)
        excluded_agent_ids = set(self._busy_agent_ids_for_assignment())
        excluded_agent_ids.update(sticky_excluded_agent_ids)
        result = assign_work_request(
            self.store,
            request,
            channel_id,
            requested_by_slack_user=event.get("user"),
            author_agent=author_agent,
            extra_metadata=extra_metadata,
            exclude_agent_ids=excluded_agent_ids,
        )
        if result is None:
            self._post_assignment_unavailable(
                SlackReplyTarget(
                    channel_id=channel_id,
                    thread_ts=thread.thread_ts,
                ),
                request,
                requested_by_slack_user=event.get("user"),
                author_agent=author_agent,
                extra_metadata=extra_metadata,
                exclude_agent_ids=sticky_excluded_agent_ids,
            )
            return
        blocks = _task_thread_blocks(result.task, result.agent)
        posted = self.gateway.post_thread_reply(
            thread,
            format_agent_assignment(
                result.agent,
                result.request.prompt,
                event.get("user"),
                dangerous_mode=_task_dangerous_mode(result.task),
            ),
            persona=result.agent,
            blocks=blocks,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(
            result.task.task_id,
            thread.thread_ts,
            posted.ts,
        )
        task = self.store.get_agent_task(result.task.task_id) or result.task
        self._start_runtime_task(task, result.agent, thread)
        self._mark_message_in_progress(channel_id, event.get("ts"))

    def handle_team_command(
        self,
        command: (
            HireCommand
            | FireCommand
            | FireEveryoneCommand
            | RepoRootCommand
            | RosterCommand
            | ScheduledTasksCommand
        ),
        target: SlackReplyTarget,
        *,
        requested_by_slack_user: str | None = None,
    ) -> None:
        if isinstance(command, HireCommand):
            if not self._can_hire(command.count):
                self._post_text(
                    target, f"{AGENT_LIMIT_MESSAGE} Max team size is {MAX_TEAM_AGENTS}."
                )
                return
            if self._handle_pm_capacity_hire_command(
                command,
                target,
                requested_by_slack_user=requested_by_slack_user,
            ):
                return
            hired = self.hire_agents(command.count, command.provider)
            summary = ", ".join(
                f"@{agent.handle} ({agent.provider_preference.value})" for agent in hired
            )
            self.gateway.post_message(
                target.channel_id,
                f"Hired {len(hired)} agent(s): {summary}",
                thread_ts=target.thread_ts,
            )
            for agent in hired:
                self._post_agent_reply(target, format_agent_introduction(agent), agent)
            self._resume_pending_work_requests(target.channel_id)
            self.refresh_or_post_roster(target.channel_id)
            return
        if isinstance(command, FireCommand):
            fired = self.store.fire_team_agent(command.handle)
            if fired is None:
                self.gateway.post_message(
                    target.channel_id,
                    f"I could not find an active agent named @{command.handle}.",
                    thread_ts=target.thread_ts,
                )
            else:
                self.gateway.post_message(
                    target.channel_id,
                    f"Fired @{fired.handle} ({fired.full_name}).",
                    thread_ts=target.thread_ts,
                )
            self.refresh_or_post_roster(target.channel_id)
            return
        if isinstance(command, FireEveryoneCommand):
            agents = self.store.list_team_agents()
            for agent in agents:
                self.store.fire_team_agent(agent.agent_id)
            self.gateway.post_message(
                target.channel_id,
                f"Fired {len(agents)} agent(s).",
                thread_ts=target.thread_ts,
            )
            self.refresh_or_post_roster(target.channel_id)
            return
        if isinstance(command, RepoRootCommand):
            if command.path is None:
                root = self._configured_repo_root() or _suggested_repo_root(self.default_cwd)
                self._post_text(target, f"Repo root: `{root}`")
                return
            root = _validated_repo_root(str(command.path))
            if root is None:
                self._post_text(target, "Use an existing local folder path.")
                return
            self.store.set_setting(SETTING_REPO_ROOT, str(root))
            self._post_text(target, f"Repo root set to `{root}`.")
            return
        if isinstance(command, ScheduledTasksCommand):
            self._post_scheduled_tasks(target)
            return
        self.post_roster(
            target.channel_id,
            thread_ts=target.thread_ts,
            remember=target.thread_ts is None,
        )

    def _handle_pm_capacity_hire_command(
        self,
        command: HireCommand,
        target: SlackReplyTarget,
        *,
        requested_by_slack_user: str | None = None,
    ) -> bool:
        initiative = self._awaiting_pm_initiative_for_thread(target.channel_id, target.thread_ts)
        if initiative is None:
            return False
        if not initiative.pending_plan_json:
            return False
        try:
            plan = deserialize_parsed_pm_plan(initiative.pending_plan_json)
        except Exception:
            return False
        shortfall = self._pm_plan_capacity_shortfall(plan)
        if shortfall is None or shortfall.hire_count <= 0:
            return False

        hired = self.hire_agents(command.count, command.provider)
        summary = ", ".join(
            f"@{agent.handle} ({agent.provider_preference.value})" for agent in hired
        )
        self.gateway.post_message(
            target.channel_id,
            (
                f"Hired {len(hired)} agent(s) for PM initiative "
                f"`{initiative.initiative_id}`: {summary}"
            ),
            thread_ts=target.thread_ts,
        )
        for agent in hired:
            try:
                self._post_agent_reply(target, format_agent_introduction(agent), agent)
            except Exception:
                LOGGER.debug("failed to post hired agent introduction", exc_info=True)

        latest = self.store.get_pm_initiative(initiative.initiative_id) or initiative
        promoted = self._execute_pm_plan(latest, approver_slack_user=requested_by_slack_user)
        if promoted is not None:
            self._strip_pm_plan_buttons(
                promoted,
                promoted.pending_plan_message_ts or initiative.pending_plan_message_ts,
                status_label="approved",
            )
        self.refresh_or_post_roster(target.channel_id)
        return True

    def _awaiting_pm_initiative_for_thread(
        self,
        channel_id: str,
        thread_ts: str | None,
    ) -> PmInitiative | None:
        if not thread_ts:
            return None
        matches = [
            initiative
            for initiative in self.store.list_pm_initiatives(
                statuses=(PmInitiativeStatus.AWAITING_APPROVAL,),
                limit=200,
            )
            if initiative.channel_id == channel_id and initiative.thread_ts == thread_ts
        ]
        return matches[-1] if matches else None

    def _post_agent_reply(self, target: SlackReplyTarget, text: str, agent) -> None:
        if target.thread_ts:
            self.gateway.post_thread_reply(
                SlackThreadRef(target.channel_id, target.thread_ts),
                text,
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
        else:
            self.gateway.post_session_parent(
                target.channel_id,
                text,
                agent,
                icon_url=self._agent_icon_url(agent),
            )

    def _post_capacity_message(self, target: SlackReplyTarget) -> None:
        self._post_text(target, CAPACITY_MESSAGE)

    def _post_assignment_unavailable(
        self,
        target: SlackReplyTarget,
        request: WorkRequest,
        *,
        requested_by_slack_user: str | None = None,
        author_agent=None,
        extra_metadata: dict[str, object] | None = None,
        exclude_agent_ids: set[str] | frozenset[str] | tuple[str, ...] | None = None,
    ) -> None:
        metadata = dict(extra_metadata or {})
        if self._should_queue_unavailable_request(request, target):
            request_message_ts = metadata.get("request_message_ts")
            self.store.create_pending_work_request(
                SlackThreadRef(
                    target.channel_id,
                    target.thread_ts or "",
                    request_message_ts if isinstance(request_message_ts, str) else None,
                ),
                request,
                requested_by_slack_user=requested_by_slack_user,
                author_agent=author_agent,
                extra_metadata=metadata,
                exclude_agent_ids=exclude_agent_ids,
            )
        self._post_text(target, self._assignment_unavailable_text(request))

    def _should_queue_unavailable_request(
        self,
        request: WorkRequest,
        target: SlackReplyTarget,
    ) -> bool:
        return bool(target.thread_ts)

    def _resume_pending_work_requests(self, channel_id: str) -> int:
        resumed = 0
        for pending in self.store.list_pending_work_requests(channel_id=channel_id):
            if not self.store.idle_team_agents():
                break
            if self._try_resume_pending_work_request(pending):
                resumed += 1
        return resumed

    def _try_resume_pending_work_request(self, pending: PendingWorkRequest) -> bool:
        request_message_ts = _pending_request_message_ts(pending)
        if request_message_ts and self._has_task_for_request_message(
            pending.channel_id,
            request_message_ts,
        ):
            self.store.update_pending_work_request_status(
                pending.pending_id,
                PendingWorkRequestStatus.CANCELLED,
            )
            return False
        if self._is_pending_for_dead_thread(pending):
            # The destination thread is no longer alive — every task in it is
            # done or cancelled, and any parent/delegating task is also
            # finished. Reviving here would post "Capacity is available now."
            # into a thread the user has already closed out, dragging an
            # agent back to a discussion that wrapped hours or days ago.
            LOGGER.info(
                "cancelling stale pending work request %s; thread %s/%s has no live work",
                pending.pending_id,
                pending.channel_id,
                pending.thread_ts,
            )
            self.store.update_pending_work_request_status(
                pending.pending_id,
                PendingWorkRequestStatus.CANCELLED,
            )
            return False
        author_agent = None
        if pending.author_agent_id:
            author_agent = self.store.get_team_agent(pending.author_agent_id)
        if author_agent is None and pending.request.author_handle:
            author_agent = self.store.get_team_agent(pending.request.author_handle)
        exclude_agent_ids = set(pending.exclude_agent_ids)
        scheduled_work_id = pending.extra_metadata.get("scheduled_work_id")
        deferred_work_id = pending.extra_metadata.get("deferred_work_id")
        exclude_agent_ids.update(
            self._busy_agent_ids_for_assignment(
                ignore_schedule_id=scheduled_work_id
                if isinstance(scheduled_work_id, str)
                else None,
                ignore_deferred_id=deferred_work_id if isinstance(deferred_work_id, str) else None,
            )
        )
        result = assign_work_request(
            self.store,
            pending.request,
            pending.channel_id,
            requested_by_slack_user=pending.requested_by_slack_user,
            author_agent=author_agent,
            extra_metadata=pending.extra_metadata,
            exclude_agent_ids=exclude_agent_ids,
        )
        if result is None:
            return False
        thread = SlackThreadRef(
            pending.channel_id,
            pending.thread_ts,
            pending.message_ts,
        )
        assignment_text = format_agent_assignment(
            result.agent,
            result.request.prompt,
            pending.requested_by_slack_user,
            dangerous_mode=_task_dangerous_mode(result.task),
        )
        text = f"Capacity is available now.\n\n{assignment_text}"
        blocks = _task_thread_blocks(result.task, result.agent)
        posted = self.gateway.post_thread_reply(
            thread,
            text,
            persona=result.agent,
            blocks=blocks,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(result.task.task_id, thread.thread_ts, posted.ts)
        self.store.update_pending_work_request_status(
            pending.pending_id,
            PendingWorkRequestStatus.ASSIGNED,
        )
        self._record_pending_work_started_task(pending, result.task.task_id)
        task = self.store.get_agent_task(result.task.task_id) or result.task
        self._start_runtime_task(task, result.agent, thread)
        if request_message_ts:
            self._mark_message_in_progress(pending.channel_id, request_message_ts)
        return True

    def _record_pending_work_started_task(
        self,
        pending: PendingWorkRequest,
        task_id: str,
    ) -> None:
        scheduled_work_id = pending.extra_metadata.get("scheduled_work_id")
        if isinstance(scheduled_work_id, str) and scheduled_work_id:
            self.store.update_scheduled_work_last_task(scheduled_work_id, last_task_id=task_id)
        deferred_work_id = pending.extra_metadata.get("deferred_work_id")
        if isinstance(deferred_work_id, str) and deferred_work_id:
            self.store.update_deferred_work_last_task(deferred_work_id, last_task_id=task_id)

    def _is_pending_for_dead_thread(self, pending: PendingWorkRequest) -> bool:
        # Scheduled/deferred pendings drive future-tense work that is supposed
        # to land in its own fresh thread; never treat those as stale.
        if pending.extra_metadata.get("scheduled_work_id"):
            return False
        if pending.extra_metadata.get("deferred_work_id"):
            return False
        parent_task_id = pending.extra_metadata.get("parent_task_id")
        if not isinstance(parent_task_id, str) or not parent_task_id:
            parent_task_id = pending.extra_metadata.get("delegated_from_task_id")
        if isinstance(parent_task_id, str) and parent_task_id:
            parent_task = self.store.get_agent_task(parent_task_id)
            if parent_task is None:
                # The parent row is gone; play it safe and revive normally.
                return False
            if parent_task.status not in {
                AgentTaskStatus.DONE,
                AgentTaskStatus.CANCELLED,
            }:
                return False
        elif not pending.thread_ts:
            # No parent and no thread anchor — nothing to declare dead.
            return False
        if not pending.thread_ts:
            return False
        thread_tasks = [
            task
            for task in self.store.list_agent_tasks(include_done=True)
            if task.channel_id == pending.channel_id and task.thread_ts == pending.thread_ts
        ]
        if not thread_tasks:
            # Fresh thread (top-level request that found no idle agents and
            # never produced a thread reply). Still eligible for revival.
            return False
        return all(
            task.status in {AgentTaskStatus.DONE, AgentTaskStatus.CANCELLED}
            for task in thread_tasks
        )

    def _post_text(self, target: SlackReplyTarget, text: str) -> None:
        if target.thread_ts:
            self.gateway.post_thread_reply(
                SlackThreadRef(target.channel_id, target.thread_ts),
                text,
            )
        else:
            self.gateway.post_message(target.channel_id, text)

    def _post_scheduled_tasks(self, target: SlackReplyTarget) -> None:
        scheduled = self.store.list_scheduled_work(
            statuses=(ScheduledWorkStatus.PENDING, ScheduledWorkStatus.CLAIMED),
            channel_id=target.channel_id,
            limit=20,
        )
        text = _scheduled_tasks_text(scheduled)
        blocks = self._scheduled_tasks_blocks(scheduled)
        if target.thread_ts:
            self.gateway.post_thread_reply(
                SlackThreadRef(target.channel_id, target.thread_ts),
                text,
                blocks=blocks,
            )
        else:
            self.gateway.post_message(target.channel_id, text, blocks=blocks)

    def _scheduled_tasks_blocks(self, scheduled: list[ScheduledWork]) -> list[dict] | None:
        if not scheduled:
            return None
        blocks: list[dict] = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Scheduled tasks*  {len(scheduled)} active",
                },
            }
        ]
        for item in scheduled:
            thread_url = self._thread_permalink(item.channel_id, item.thread_ts)
            blocks.append(
                {
                    "type": "section",
                    "block_id": f"schedule.item.{item.schedule_id}",
                    "text": {
                        "type": "mrkdwn",
                        "text": _scheduled_work_block_text(item),
                    },
                }
            )
            elements = [
                _slack_button(
                    "Deschedule",
                    "schedule.cancel",
                    encode_action_value("schedule.cancel", schedule_id=item.schedule_id),
                    "danger",
                ),
                _slack_button(
                    "Change schedule",
                    "schedule.change",
                    encode_action_value("schedule.change", schedule_id=item.schedule_id),
                ),
            ]
            if thread_url:
                elements.append(
                    _slack_button(
                        "Open thread",
                        "thread.open",
                        encode_action_value("thread.open"),
                        url=thread_url,
                    )
                )
            blocks.append(
                {
                    "type": "actions",
                    "block_id": f"schedule.actions.{item.schedule_id}",
                    "elements": elements,
                }
            )
        return blocks

    def _assignment_unavailable_text(self, request: WorkRequest) -> str:
        if request.assignment_mode == AssignmentMode.SPECIFIC and request.requested_handle:
            agent = self.store.get_team_agent(request.requested_handle)
            if agent is not None:
                idle_ids = {item.agent_id for item in self.store.idle_team_agents()}
                if (
                    agent.agent_id not in idle_ids
                    or agent.agent_id in self._busy_agent_ids_for_assignment()
                ):
                    return (
                        f"That specific agent is busy. Wait for @{agent.handle}, "
                        "or ask someone else."
                    )
        return CAPACITY_MESSAGE

    def _handle_pm_request(self, event: dict, channel_id: str, text: str) -> bool:
        # If this message is already a reply inside an active agent thread, let
        # the task-thread-reply path handle it (so status questions for an
        # existing PM thread go to that PM, not a brand-new initiative).
        thread_ts = event.get("thread_ts")
        if (
            thread_ts
            and thread_ts != event.get("ts")
            and self.store.get_original_agent_task_by_thread(channel_id, thread_ts) is not None
        ):
            return False
        raw_text = re.sub(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", "", text).strip()
        active_agents = self.store.list_team_agents()
        pm_agents = filter_pm_agents(active_agents)
        targeted_pm_handle: str | None = None
        if pm_agents:
            targeted_pm_handle = message_targets_pm_agent(
                raw_text, [agent.handle for agent in pm_agents]
            )
        if targeted_pm_handle:
            project_body = _strip_handle_prefix(raw_text, targeted_pm_handle)
            if not project_body:
                return False
        elif looks_like_pm_request(raw_text):
            project_body = extract_pm_request_body(raw_text)
            if not project_body:
                return False
        elif pm_agents:
            # Also intercept generic `@<pm-handle> ...` work requests that
            # message_targets_pm_agent does not recognize (e.g. ones carrying
            # `#dangerous-mode` tags or backtick-quoted handles). Without
            # this branch the brief falls through to the worker-task path,
            # the PM runs without an initiative row, and the resulting
            # PM_PLAN signal has nothing to attach to.
            parsed_request = _channel_work_request(text, active_agents)
            if (
                parsed_request is None
                or parsed_request.assignment_mode != AssignmentMode.SPECIFIC
                or not parsed_request.requested_handle
            ):
                return False
            pm_agent_match = next(
                (agent for agent in pm_agents if agent.handle == parsed_request.requested_handle),
                None,
            )
            if pm_agent_match is None:
                return False
            targeted_pm_handle = pm_agent_match.handle
            project_body = parsed_request.prompt.strip()
            if not project_body:
                return False
        else:
            return False
        thread = self._request_thread_anchor(event, channel_id, text)
        self._mark_message_acknowledged(channel_id, event.get("ts"))
        dispatched = self._dispatch_pm_initiative(
            channel_id=channel_id,
            project_body=project_body,
            requested_by_slack_user=event.get("user"),
            thread=thread,
            targeted_pm_handle=targeted_pm_handle,
            request_message_ts=event.get("ts"),
        )
        if dispatched:
            self._mark_message_in_progress(channel_id, event.get("ts"))
        return dispatched

    def _dispatch_pm_initiative(
        self,
        *,
        channel_id: str,
        project_body: str,
        requested_by_slack_user: str | None,
        thread: SlackThreadRef,
        targeted_pm_handle: str | None,
        request_message_ts: str | None,
        text_for_linked_context: str | None = None,
    ) -> bool:
        """Create a PM initiative and dispatch the resolver task.

        Single chokepoint for every entry point that wants to hand a brief
        to a PM-kind agent (text routing, roster modal, multi-specific
        guards). Callers are responsible for posting the thread parent
        before invoking this helper.
        """
        active_agents = self.store.list_team_agents()
        pm_agents = filter_pm_agents(active_agents)
        if not active_agents:
            self._post_capacity_message(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread.thread_ts)
            )
            return True
        title = _shorten(project_body, 80) or "PM initiative"
        initiative = self.store.create_pm_initiative(
            SlackThreadRef(channel_id, thread.thread_ts, thread.message_ts),
            title=title,
            summary=project_body,
            requested_by_slack_user=requested_by_slack_user,
        )
        available_workers = self._pm_available_worker_agents()
        resolver_prompt = build_pm_resolution_prompt(
            project_body,
            _pm_worker_handles(available_workers),
            initiative_id=initiative.initiative_id,
            now=utc_now(),
            agent_models=_pm_worker_model_map(available_workers),
        )
        busy_agent_ids = self._busy_agent_ids_for_assignment()
        if targeted_pm_handle:
            request = WorkRequest(
                prompt=resolver_prompt,
                assignment_mode=AssignmentMode.SPECIFIC,
                requested_handle=targeted_pm_handle,
            )
        elif pm_agents:
            # No specific PM tagged but PM-kind agents exist. Pin the resolver
            # to a PM (preferring an idle one) so the PM owns the initiative
            # end-to-end; worker agents never run PM resolvers when PMs exist.
            idle_agent_ids = {agent.agent_id for agent in self.store.idle_team_agents()}
            available_pms = [agent for agent in pm_agents if agent.agent_id not in busy_agent_ids]
            idle_pms = [agent for agent in available_pms if agent.agent_id in idle_agent_ids]
            chosen_pms = idle_pms or available_pms or pm_agents
            pm_handles = sorted(agent.handle for agent in chosen_pms)
            digest_handle = pm_handles[hash(initiative.initiative_id) % len(pm_handles)]
            request = WorkRequest(
                prompt=resolver_prompt,
                assignment_mode=AssignmentMode.SPECIFIC,
                requested_handle=digest_handle,
            )
        else:
            request = WorkRequest(
                prompt=resolver_prompt,
                assignment_mode=AssignmentMode.ANYONE,
            )
        extra_metadata = self._with_linked_thread_context(
            {
                PM_RESOLUTION_METADATA_KEY: True,
                PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: project_body,
                PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                ASSIGNMENT_PROMPT_METADATA_KEY: project_body,
                "request_message_ts": request_message_ts,
            },
            text_for_linked_context or project_body,
            current_thread=thread,
        )
        result = assign_work_request(
            self.store,
            request,
            channel_id,
            requested_by_slack_user=requested_by_slack_user,
            extra_metadata=extra_metadata,
            exclude_agent_ids=busy_agent_ids,
        )
        if result is None:
            self.store.update_pm_initiative_status(
                initiative.initiative_id, PmInitiativeStatus.CANCELLED
            )
            self._post_assignment_unavailable(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread.thread_ts),
                request,
                requested_by_slack_user=requested_by_slack_user,
                extra_metadata=extra_metadata,
            )
            return True
        posted = self.gateway.post_thread_reply(
            thread,
            (
                f"@{result.agent.handle} is the PM for this initiative and will "
                f"break it into subtasks: `{_shorten(project_body, 180)}`"
            ),
            persona=result.agent,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(result.task.task_id, thread.thread_ts, posted.ts)
        self.store.update_pm_initiative_owner(
            initiative.initiative_id,
            pm_agent_id=result.agent.agent_id,
            pm_task_id=result.task.task_id,
        )
        task = self.store.get_agent_task(result.task.task_id) or result.task
        if self.runtime:
            self.runtime.start_task(task, result.agent, thread)
        return True

    def _pm_available_worker_agents(
        self,
        *,
        ignore_task_id: str | None = None,
        ignore_pm_initiative_id: str | None = None,
    ) -> list[TeamAgent]:
        busy_agent_ids = self._pm_busy_worker_agent_ids(
            ignore_task_id=ignore_task_id,
            ignore_pm_initiative_id=ignore_pm_initiative_id,
        )
        return [
            agent
            for agent in filter_worker_agents(self.store.list_team_agents())
            if agent.agent_id not in busy_agent_ids
        ]

    def _pm_agent_for_request_target(self, request: WorkRequest):
        """Return the PM-kind TeamAgent the request targets, or None.

        Used by every assignment entry point to detect when a brief that
        looks like worker work is actually being handed to a PM agent —
        in which case the entry point must redirect through
        :meth:`_dispatch_pm_initiative` instead of falling through to
        worker dispatch (the PM model requires an ``pm_initiatives`` row
        so the ``PM_PLAN`` signal can land an approval card).
        """
        if request.assignment_mode != AssignmentMode.SPECIFIC or not request.requested_handle:
            return None
        agent = self.store.get_team_agent(request.requested_handle)
        if agent is None or agent.kind != TeamAgentKind.PM:
            return None
        return agent

    def _handle_deferred_work_request(self, event: dict, channel_id: str, text: str) -> bool:
        deferred_text = re.sub(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", "", text).strip()
        if not looks_like_deferred_request(deferred_text):
            return False
        thread = self._request_thread_anchor(event, channel_id, text)
        active_agents = self.store.list_team_agents()
        if not active_agents:
            self._post_capacity_message(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread.thread_ts)
            )
            return True
        occupied = self._occupied_handle_task_ids()
        resolver_prompt = build_deferred_resolution_prompt(
            deferred_text,
            [agent.handle for agent in active_agents],
            occupied=[{"handle": handle, "task_id": task_id} for handle, task_id in occupied],
            now=utc_now(),
        )
        request = WorkRequest(
            prompt=resolver_prompt,
            assignment_mode=AssignmentMode.ANYONE,
        )
        extra_metadata = self._with_linked_thread_context(
            {
                DEFERRED_RESOLUTION_METADATA_KEY: True,
                DEFERRED_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: deferred_text,
                DEFERRED_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                DEFERRED_RESOLUTION_OCCUPIED_HANDLES_METADATA_KEY: [
                    {"handle": handle, "task_id": task_id} for handle, task_id in occupied
                ],
                ASSIGNMENT_PROMPT_METADATA_KEY: deferred_text,
                "request_message_ts": event.get("ts"),
            },
            deferred_text,
            current_thread=thread,
        )
        self._mark_message_acknowledged(channel_id, event.get("ts"))
        result = assign_work_request(
            self.store,
            request,
            channel_id,
            requested_by_slack_user=event.get("user"),
            extra_metadata=extra_metadata,
            exclude_agent_ids=self._busy_agent_ids_for_assignment(),
        )
        if result is None:
            self._post_assignment_unavailable(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread.thread_ts),
                request,
                requested_by_slack_user=event.get("user"),
                extra_metadata=extra_metadata,
            )
            return True
        posted = self.gateway.post_thread_reply(
            thread,
            (
                f"@{result.agent.handle} is resolving this deferred request: "
                f"`{_shorten(deferred_text, 180)}`"
            ),
            persona=result.agent,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(result.task.task_id, thread.thread_ts, posted.ts)
        task = self.store.get_agent_task(result.task.task_id) or result.task
        if self.runtime:
            self.runtime.start_task(task, result.agent, thread)
        self._mark_message_in_progress(channel_id, event.get("ts"))
        return True

    def _occupied_handle_task_ids(self) -> list[tuple[str, str]]:
        occupied: list[tuple[str, str]] = []
        seen_handles: set[str] = set()
        agents = self.store.list_team_agents()
        agents_by_id = {agent.agent_id: agent for agent in agents}
        for agent in agents:
            task = self.store.active_task_for_agent(agent.agent_id)
            if task is not None and task.status in {
                AgentTaskStatus.QUEUED,
                AgentTaskStatus.ACTIVE,
            }:
                occupied.append((agent.handle, task.task_id))
                seen_handles.add(agent.handle)
        for agent_id, session in self._active_external_sessions_by_agent().items():
            agent = agents_by_id.get(agent_id)
            if agent is None or agent.handle in seen_handles:
                continue
            occupied.append(
                (
                    agent.handle,
                    external_session_dependency_id(session.provider, session.session_id),
                )
            )
            seen_handles.add(agent.handle)
        for agent_id, deferred_items in self._pending_deferred_work_by_agent(agents).items():
            agent = agents_by_id.get(agent_id)
            if agent is None or agent.handle in seen_handles:
                continue
            occupied.append(
                (
                    agent.handle,
                    deferred_work_dependency_id(deferred_items[0].deferred_id),
                )
            )
            seen_handles.add(agent.handle)
        for agent_id, scheduled_items in self._pending_scheduled_work_by_agent(agents).items():
            agent = agents_by_id.get(agent_id)
            if agent is None or agent.handle in seen_handles:
                continue
            occupied.append(
                (
                    agent.handle,
                    scheduled_work_dependency_id(scheduled_items[0].schedule_id),
                )
            )
            seen_handles.add(agent.handle)
        return occupied

    def _occupied_handle_dependency_labels(self) -> list[tuple[str, str, str]]:
        return [
            (handle, task_id, self._agent_busy_dependency_label(task_id))
            for handle, task_id in self._occupied_handle_task_ids()
        ]

    def _agent_busy_dependency_label(self, task_id: str | None) -> str:
        external_session = parse_external_session_dependency_id(task_id)
        if external_session is not None:
            provider, session_id = external_session
            prefix = f"{provider.value} external session"
            summary = self.store.get_setting(
                f"{EXTERNAL_SESSION_SUMMARY_PREFIX}{provider.value}.{session_id}"
            )
            if summary and summary.strip():
                return f"{prefix}: {_shorten(summary, 140)}"
            session = self.store.get_session(provider, session_id)
            if session and session.cwd:
                return f"{prefix}: {session.cwd.name or session.cwd}"
            return prefix
        deferred_id = parse_deferred_work_dependency_id(task_id)
        if deferred_id is not None:
            deferred = self.store.get_deferred_work(deferred_id)
            if deferred is not None:
                return _shorten(_deferred_work_roster_detail(deferred), 140)
            return "deferred task"
        schedule_id = parse_scheduled_work_dependency_id(task_id)
        if schedule_id is not None:
            scheduled = self.store.get_scheduled_work(schedule_id)
            if scheduled is not None:
                return _shorten(
                    (
                        f"Scheduled task: {scheduled.prompt}; "
                        f"{_format_scheduled_work_schedule(scheduled)}; "
                        f"next run `{_format_schedule_timestamp(scheduled.next_run_at)}`"
                    ),
                    140,
                )
            return "scheduled task"
        if task_id:
            task = self.store.get_agent_task(task_id)
            if task is not None:
                return _shorten(_task_roster_summary(task), 140)
        return "current task"

    def _prune_stale_managed_sessions(self, agents) -> None:
        """Drop managed_session settings that no longer match an active task.

        A managed session record only stays valid while the runtime is actively
        driving a task on that session. Once the task moves to a terminal state
        the record should be cleared; otherwise the roster has no honest way to
        explain the agent's occupancy. Defensive cleanup keeps the invariant
        "Occupied implies a current deferred or active task" intact even when
        an upstream code path forgets to call clear_managed_session.
        """
        valid_keys: set[tuple[Provider, str]] = set()
        for agent in agents:
            task = self.store.active_task_for_agent(agent.agent_id)
            if task is None:
                continue
            if task.session_provider is None or not task.session_id:
                continue
            valid_keys.add((task.session_provider, task.session_id))
        for (provider, session_id), _agent_id in managed_session_agents(self.store).items():
            if (provider, session_id) in valid_keys:
                continue
            try:
                clear_managed_session(self.store, provider, session_id)
            except Exception:
                LOGGER.debug(
                    "failed to clear stale managed session %s/%s",
                    provider.value,
                    session_id,
                    exc_info=True,
                )

    def _handle_scheduled_work_request(self, event: dict, channel_id: str, text: str) -> bool:
        schedule_text = re.sub(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", "", text).strip()
        if not looks_like_schedule_request(schedule_text):
            return False
        thread = self._request_thread_anchor(event, channel_id, text)
        active_agents = self.store.list_team_agents()
        if not active_agents:
            self._post_capacity_message(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread.thread_ts)
            )
            return True
        resolver_prompt = build_schedule_resolution_prompt(
            schedule_text,
            [agent.handle for agent in active_agents],
            now=utc_now(),
        )
        request = WorkRequest(
            prompt=resolver_prompt,
            assignment_mode=AssignmentMode.ANYONE,
        )
        extra_metadata = self._with_linked_thread_context(
            {
                SCHEDULE_RESOLUTION_METADATA_KEY: True,
                SCHEDULE_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: schedule_text,
                SCHEDULE_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                ASSIGNMENT_PROMPT_METADATA_KEY: schedule_text,
                "request_message_ts": event.get("ts"),
            },
            schedule_text,
            current_thread=thread,
        )
        self._mark_message_acknowledged(channel_id, event.get("ts"))
        result = assign_work_request(
            self.store,
            request,
            channel_id,
            requested_by_slack_user=event.get("user"),
            extra_metadata=extra_metadata,
            exclude_agent_ids=self._busy_agent_ids_for_assignment(),
        )
        if result is None:
            self._post_assignment_unavailable(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread.thread_ts),
                request,
                requested_by_slack_user=event.get("user"),
                extra_metadata=extra_metadata,
            )
            return True
        posted = self.gateway.post_thread_reply(
            thread,
            f"@{result.agent.handle} scheduling.",
            persona=result.agent,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(result.task.task_id, thread.thread_ts, posted.ts)
        task = self.store.get_agent_task(result.task.task_id) or result.task
        if self.runtime:
            self.runtime.start_task(task, result.agent, thread)
        self._mark_message_in_progress(channel_id, event.get("ts"))
        return True

    def fire_due_scheduled_work(self, scheduled: ScheduledWork) -> bool:
        claimed = self.store.claim_scheduled_work(scheduled.schedule_id)
        if claimed is None:
            return True
        request = _work_request_from_scheduled_work(claimed)
        thread = SlackThreadRef(claimed.channel_id, claimed.thread_ts, claimed.message_ts)
        author_agent = (
            self.store.get_team_agent(request.author_handle) if request.author_handle else None
        )
        if (
            request.assignment_mode == AssignmentMode.SPECIFIC
            and request.requested_handle
            and self.store.get_team_agent(request.requested_handle) is None
        ):
            self.store.update_scheduled_work_status(
                claimed.schedule_id,
                ScheduledWorkStatus.CANCELLED,
            )
            self.gateway.post_thread_reply(
                thread,
                (
                    f"Cancelled schedule `{claimed.schedule_id}` because "
                    f"@{request.requested_handle} is not an active agent."
                ),
            )
            return True
        metadata = {
            "scheduled_work_id": claimed.schedule_id,
            "scheduled_work_due_at": claimed.next_run_at.isoformat(),
        }
        result = assign_work_request(
            self.store,
            request,
            claimed.channel_id,
            requested_by_slack_user=claimed.requested_by_slack_user,
            author_agent=author_agent,
            extra_metadata=metadata,
            exclude_agent_ids=self._busy_agent_ids_for_assignment(
                ignore_schedule_id=claimed.schedule_id
            ),
        )
        next_run_at = _next_scheduled_work_run(claimed)
        if result is None:
            pending = self.store.create_pending_work_request(
                thread,
                request,
                requested_by_slack_user=claimed.requested_by_slack_user,
                author_agent=author_agent,
                extra_metadata=metadata,
                exclude_agent_ids=self._busy_agent_ids_for_assignment(
                    ignore_schedule_id=claimed.schedule_id
                ),
            )
            self.gateway.post_thread_reply(
                thread,
                (
                    f"Scheduled task `{claimed.schedule_id}` is due, but no matching "
                    "agent is available. I queued it and will start it when capacity opens."
                ),
            )
            self.store.complete_scheduled_work(
                claimed.schedule_id,
                last_task_id=pending.pending_id,
                next_run_at=next_run_at,
            )
            self.refresh_or_post_roster(claimed.channel_id)
            return True
        text = f"Scheduled task `{claimed.schedule_id}` is due now.\n\n" + format_agent_assignment(
            result.agent,
            result.request.prompt,
            claimed.requested_by_slack_user,
            dangerous_mode=_task_dangerous_mode(result.task),
        )
        blocks = _task_thread_blocks(result.task, result.agent)
        posted = self.gateway.post_thread_reply(
            thread,
            text,
            persona=result.agent,
            blocks=blocks,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(result.task.task_id, thread.thread_ts, posted.ts)
        task = self.store.get_agent_task(result.task.task_id) or result.task
        if self.runtime:
            self.runtime.start_task(task, result.agent, thread)
        self.store.complete_scheduled_work(
            claimed.schedule_id,
            last_task_id=result.task.task_id,
            next_run_at=next_run_at,
        )
        self.refresh_or_post_roster(claimed.channel_id)
        return True

    def _roster_statuses(self, agents) -> dict[str, AgentRosterStatus]:
        self._prune_stale_managed_sessions(agents)
        statuses = {agent.agent_id: AgentRosterStatus("Available") for agent in agents}
        for agent in agents:
            task = self.store.active_task_for_agent(agent.agent_id)
            if task is None:
                continue
            if task.status == AgentTaskStatus.QUEUED:
                label = "Queued"
                detail = _task_roster_detail(task)
            elif self._task_is_runtime_running(task.task_id):
                label = "Working"
                detail = _task_roster_detail(task)
            else:
                label = "Occupied"
                detail = (
                    f"Open thread: {_shorten(_task_roster_summary(task), 140)}\n"
                    f"*Original Task:* {_shorten(_task_original_prompt(task), 180)}"
                )
            statuses[agent.agent_id] = AgentRosterStatus(
                label,
                detail,
                dangerous_mode=_task_dangerous_mode(task),
                pr_urls=pr_urls_from_metadata(task.metadata),
                thread_url=self._thread_permalink(task.channel_id, task.thread_ts),
                task_id=task.task_id,
            )
        for running in self._running_managed_tasks():
            agent = getattr(running, "agent", None)
            task = getattr(running, "task", None)
            thread = getattr(running, "thread", None)
            if agent is None or task is None or thread is None:
                continue
            if agent.agent_id not in statuses:
                continue
            task = self.store.get_agent_task(task.task_id) or task
            if task.status not in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}:
                continue
            statuses[agent.agent_id] = AgentRosterStatus(
                "Working",
                _task_roster_detail(task),
                dangerous_mode=_task_dangerous_mode(task),
                pr_urls=pr_urls_from_metadata(task.metadata),
                thread_url=self._thread_permalink(thread.channel_id, thread.thread_ts),
                task_id=task.task_id,
            )
        pm_initiatives_by_agent = self._pm_owner_initiatives_by_agent(agents)
        for agent in agents:
            if statuses[agent.agent_id].label != "Available":
                continue
            initiatives = pm_initiatives_by_agent.get(agent.agent_id)
            if not initiatives:
                continue
            initiative = initiatives[0]
            detail = _pm_initiative_roster_detail(initiative)
            if len(initiatives) > 1:
                detail = f"{detail}; +{len(initiatives) - 1} more"
            statuses[agent.agent_id] = AgentRosterStatus(
                "Occupied",
                detail,
                thread_url=self._thread_permalink(initiative.channel_id, initiative.thread_ts),
            )
        deferred_by_agent = self._pending_deferred_work_by_agent(agents)
        for agent in agents:
            if statuses[agent.agent_id].label != "Available":
                continue
            deferred_items = deferred_by_agent.get(agent.agent_id)
            if not deferred_items:
                continue
            deferred = deferred_items[0]
            detail = _deferred_work_roster_detail(deferred)
            if len(deferred_items) > 1:
                detail = f"{detail}; +{len(deferred_items) - 1} more"
            statuses[agent.agent_id] = AgentRosterStatus(
                "Occupied",
                detail,
                dangerous_mode=deferred.dangerous_mode,
                thread_url=self._thread_permalink(deferred.channel_id, deferred.thread_ts),
            )
        scheduled_by_agent = self._pending_scheduled_work_by_agent(agents)
        for agent in agents:
            if statuses[agent.agent_id].label != "Available":
                continue
            scheduled_items = scheduled_by_agent.get(agent.agent_id)
            if not scheduled_items:
                continue
            scheduled = scheduled_items[0]
            detail = (
                f"Scheduled task: {_shorten(scheduled.prompt, 110)}; "
                f"{_format_scheduled_work_schedule(scheduled)}; "
                f"next run `{_format_schedule_timestamp(scheduled.next_run_at)}`"
            )
            if len(scheduled_items) > 1:
                detail = f"{detail}; +{len(scheduled_items) - 1} more"
            statuses[agent.agent_id] = AgentRosterStatus(
                "Occupied",
                detail,
                dangerous_mode=scheduled.dangerous_mode,
                pr_urls=(scheduled.pr_url,) if scheduled.pr_url else (),
                thread_url=self._thread_permalink(scheduled.channel_id, scheduled.thread_ts),
            )
        agents_by_id = {agent.agent_id: agent for agent in agents}
        for agent_id, session in self._active_external_sessions_by_agent().items():
            if agent_id not in statuses:
                continue
            if statuses[agent_id].label != "Available":
                continue
            thread_url = self._external_session_roster_thread_url(
                session,
                agents_by_id.get(agent_id),
            )
            detail_parts = [f"{session.provider.value} session outside Slack"]
            summary = self.store.get_setting(
                f"{EXTERNAL_SESSION_SUMMARY_PREFIX}{session.provider.value}.{session.session_id}"
            )
            if summary and summary.strip():
                detail_parts.append(_shorten(summary, 100))
            elif session.cwd:
                detail_parts.append(str(session.cwd.name or session.cwd))
            else:
                detail_parts.append(_shorten(session.session_id, 12))
            statuses[agent_id] = AgentRosterStatus(
                "Occupied",
                ": ".join(detail_parts),
                dangerous_mode=_external_session_dangerous_mode(session),
                thread_url=thread_url,
                session_provider=session.provider,
                session_id=session.session_id,
            )
        return statuses

    def _pending_deferred_work_by_agent(self, agents) -> dict[str, list[DeferredWork]]:
        agents_by_handle = {agent.handle: agent for agent in agents}
        deferred_by_agent: dict[str, list[DeferredWork]] = {}
        for deferred in self.store.list_deferred_work(
            statuses=(
                DeferredWorkStatus.WAITING_DEPS,
                DeferredWorkStatus.READY,
                DeferredWorkStatus.CLAIMED,
            )
        ):
            if deferred.assignment_mode != AssignmentMode.SPECIFIC or not deferred.requested_handle:
                continue
            agent = agents_by_handle.get(deferred.requested_handle)
            if agent is None:
                continue
            deferred_by_agent.setdefault(agent.agent_id, []).append(deferred)
        return deferred_by_agent

    def _pending_scheduled_work_by_agent(self, agents) -> dict[str, list[ScheduledWork]]:
        agents_by_handle = {agent.handle: agent for agent in agents}
        scheduled_by_agent: dict[str, list[ScheduledWork]] = {}
        for scheduled in self.store.list_scheduled_work(
            statuses=(ScheduledWorkStatus.PENDING, ScheduledWorkStatus.CLAIMED)
        ):
            if (
                scheduled.assignment_mode != AssignmentMode.SPECIFIC
                or not scheduled.requested_handle
            ):
                continue
            agent = agents_by_handle.get(scheduled.requested_handle)
            if agent is None:
                continue
            scheduled_by_agent.setdefault(agent.agent_id, []).append(scheduled)
        return scheduled_by_agent

    def _pm_owner_initiatives_by_agent(self, agents) -> dict[str, list[PmInitiative]]:
        agent_ids = {agent.agent_id for agent in agents}
        initiatives_by_agent: dict[str, list[PmInitiative]] = {}
        for initiative in self.store.list_pm_initiatives():
            if initiative.status not in _ACTIVE_PM_INITIATIVE_STATUSES:
                continue
            if not initiative.pm_agent_id or initiative.pm_agent_id not in agent_ids:
                continue
            initiatives_by_agent.setdefault(initiative.pm_agent_id, []).append(initiative)
        return initiatives_by_agent

    def _running_managed_tasks(self):
        if self.runtime is None:
            return []
        running_tasks = getattr(self.runtime, "running_tasks", None)
        if not callable(running_tasks):
            return []
        try:
            return running_tasks()
        except Exception:
            LOGGER.debug("failed to read running managed tasks for roster", exc_info=True)
            return []

    def _task_is_runtime_running(self, task_id: str) -> bool:
        if self.runtime is None:
            return False
        is_running = getattr(self.runtime, "is_task_running", None)
        if not callable(is_running):
            return False
        try:
            return bool(is_running(task_id))
        except Exception:
            LOGGER.debug("failed to read managed task runtime state", exc_info=True)
            return False

    def _external_session_permalink(self, session) -> str | None:
        channel_id = self._configured_agent_channel_id()
        if channel_id is None:
            return None
        thread = self._external_session_thread_for_channel(
            session.provider, session.session_id, channel_id
        )
        if thread is None:
            return None
        return self._thread_permalink(thread.channel_id, thread.thread_ts)

    def _external_session_thread_for_channel(
        self,
        provider: Provider,
        session_id: str,
        channel_id: str,
    ) -> SlackThreadRef | None:
        thread = self.store.get_slack_thread_for_session(
            provider,
            session_id,
            self.team_id,
            channel_id,
        )
        if thread is not None:
            return thread
        return self.store.find_slack_thread_for_session_channel(provider, session_id, channel_id)

    def _thread_permalink(self, channel_id: str, thread_ts: str | None) -> str | None:
        if not thread_ts:
            return None
        try:
            return self.gateway.permalink(channel_id, thread_ts)
        except Exception:
            LOGGER.debug("failed to get Slack permalink for %s:%s", channel_id, thread_ts)
            return None

    def _external_session_roster_thread_url(
        self,
        session: AgentSession,
        agent: TeamAgent | None,
    ) -> str | None:
        thread_url = self._external_session_permalink(session)
        if thread_url is not None or agent is None:
            return thread_url
        channel_id = self._configured_agent_channel_id()
        if channel_id is None:
            return None
        try:
            self._ensure_external_session_thread(channel_id, session, agent)
        except Exception:
            LOGGER.debug("failed to create missing external session roster thread", exc_info=True)
            return None
        return self._external_session_permalink(session)

    def _busy_agent_ids_for_assignment(
        self,
        ignore_schedule_id: str | None = None,
        ignore_deferred_id: str | None = None,
        ignore_pm_initiative_id: str | None = None,
    ) -> set[str]:
        busy = set(self._external_busy_agent_ids())
        busy.update(self._pm_owner_busy_agent_ids(ignore_initiative_id=ignore_pm_initiative_id))
        busy.update(self._scheduled_busy_agent_ids(ignore_schedule_id=ignore_schedule_id))
        busy.update(self._deferred_busy_agent_ids(ignore_deferred_id=ignore_deferred_id))
        return busy

    def _busy_agent_ids_for_roster_work_target(self) -> set[str]:
        busy = {task.agent_id for task in self.store.list_agent_tasks()}
        busy.update(self._busy_agent_ids_for_assignment())
        return busy

    def _scheduled_busy_agent_ids(self, ignore_schedule_id: str | None = None) -> set[str]:
        agents_by_handle = {agent.handle: agent.agent_id for agent in self.store.list_team_agents()}
        busy: set[str] = set()
        for scheduled in self.store.list_scheduled_work(
            statuses=(ScheduledWorkStatus.PENDING, ScheduledWorkStatus.CLAIMED)
        ):
            if ignore_schedule_id and scheduled.schedule_id == ignore_schedule_id:
                continue
            if scheduled.assignment_mode == AssignmentMode.SPECIFIC and scheduled.requested_handle:
                agent_id = agents_by_handle.get(scheduled.requested_handle)
                if agent_id:
                    busy.add(agent_id)
        return busy

    def _deferred_busy_agent_ids(self, ignore_deferred_id: str | None = None) -> set[str]:
        agents_by_handle = {agent.handle: agent.agent_id for agent in self.store.list_team_agents()}
        busy: set[str] = set()
        for deferred in self.store.list_deferred_work(
            statuses=(
                DeferredWorkStatus.WAITING_DEPS,
                DeferredWorkStatus.READY,
                DeferredWorkStatus.CLAIMED,
            )
        ):
            if ignore_deferred_id and deferred.deferred_id == ignore_deferred_id:
                continue
            if deferred.assignment_mode == AssignmentMode.SPECIFIC and deferred.requested_handle:
                agent_id = agents_by_handle.get(deferred.requested_handle)
                if agent_id:
                    busy.add(agent_id)
        return busy

    def _external_busy_agent_ids(self) -> set[str]:
        return set(self._active_external_sessions_by_agent())

    def _pm_owner_busy_agent_ids(self, *, ignore_initiative_id: str | None = None) -> set[str]:
        return {
            initiative.pm_agent_id
            for initiative in self.store.list_pm_initiatives()
            if initiative.pm_agent_id
            and initiative.initiative_id != ignore_initiative_id
            and initiative.status in _ACTIVE_PM_INITIATIVE_STATUSES
        }

    def _active_external_sessions_by_agent(self):
        sessions_by_agent = {}
        for key, agent_id in self.store.list_settings(EXTERNAL_SESSION_AGENT_PREFIX).items():
            session = self._session_for_external_agent_setting(key)
            agent = self.store.get_team_agent(agent_id)
            if agent is None:
                continue
            if agent.kind == TeamAgentKind.PM:
                self.store.delete_setting(key)
                if session is not None:
                    self.store.set_setting(
                        _external_session_ignored_setting_key(session),
                        utc_now().isoformat(),
                    )
                continue
            if session is None or session.status not in {
                SessionStatus.ACTIVE,
                SessionStatus.IDLE,
            }:
                continue
            existing = sessions_by_agent.get(agent_id)
            if existing is None or (
                existing.status != SessionStatus.ACTIVE and session.status == SessionStatus.ACTIVE
            ):
                sessions_by_agent[agent_id] = session
        return sessions_by_agent

    def _session_for_external_agent_setting(self, key: str):
        parsed = _parse_external_session_setting_key(key, EXTERNAL_SESSION_AGENT_PREFIX)
        if parsed is None:
            return None
        provider, session_id = parsed
        return self.store.get_session(provider, session_id)

    def _slash_command_target_channel_id(self, payload: dict) -> str | None:
        return self._configured_agent_channel_id() or payload.get("channel_id")

    def _configured_agent_channel_id(self) -> str | None:
        return (
            self.store.get_setting(SETTING_CHANNEL_ID)
            or self.store.get_setting("slack_channel_id")
            or self.default_channel_id
        )

    def _default_repo_root_for_setup(self) -> Path:
        configured = self._configured_repo_root()
        if configured is not None:
            return configured
        return _suggested_repo_root(self.default_cwd)

    def _configured_repo_root(self) -> Path | None:
        configured = self.store.get_setting(SETTING_REPO_ROOT)
        if not configured:
            return None
        path = Path(configured).expanduser()
        return path if path.exists() and path.is_dir() else None

    def hire_agents(
        self,
        count: int,
        provider: Provider | None = None,
        *,
        kind: TeamAgentKind = TeamAgentKind.ENGINEER,
    ):
        if not self._can_hire(count):
            raise ValueError(AGENT_LIMIT_MESSAGE)
        active_agents = self.store.list_team_agents()
        hired = hire_team_agents(
            active_agents,
            count,
            provider,
            start_sort_order=self.store.next_team_sort_order(),
            balance_agents=active_agents,
            avatar_agents=active_agents,
            randomize_identities=True,
            kind=kind,
        )
        for agent in hired:
            self._release_inactive_handle(agent.handle)
            self.store.upsert_team_agent(agent)
        return hired

    def _release_inactive_handle(self, handle: str) -> None:
        existing = self.store.get_team_agent(handle, include_fired=True)
        if existing is None or existing.status == TeamAgentStatus.ACTIVE:
            return
        used_handles = {
            agent.handle
            for agent in self.store.list_team_agents(include_fired=True)
            if agent.agent_id != existing.agent_id
        }
        retired = _retired_inactive_handle(existing, used_handles)
        self.store.upsert_team_agent(replace(existing, handle=retired))

    def _can_hire(self, count: int) -> bool:
        active_count = len(self.store.list_team_agents())
        return count >= 1 and active_count + count <= MAX_TEAM_AGENTS

    def _capacity_notice_ts_key(self, provider: Provider) -> str:
        return f"{CAPACITY_NOTICE_TS_PREFIX}{provider.value}"

    def _external_capacity_hire_clicked(
        self,
        provider: Provider | None,
        message_ts: str | None,
    ) -> bool:
        if provider is None or not message_ts:
            return False
        return self.store.get_setting(self._capacity_notice_ts_key(provider)) == message_ts

    def _assign_hired_agents_to_pending_external_sessions(
        self,
        channel_id: str,
        hired: list,
        provider: Provider | None = None,
    ) -> int:
        available_by_provider: dict[Provider, list] = {}
        external_busy_ids = self._external_busy_agent_ids()
        for agent in hired:
            if agent.is_pm or agent.provider_preference is None:
                continue
            if provider is not None and agent.provider_preference != provider:
                continue
            if agent.agent_id in external_busy_ids:
                continue
            if self.store.active_task_for_agent(agent.agent_id) is not None:
                continue
            available_by_provider.setdefault(agent.provider_preference, []).append(agent)
        if not available_by_provider:
            return 0

        touched_providers: set[Provider] = set()
        assigned = 0
        for key in sorted(self.store.list_settings(PENDING_EXTERNAL_SESSION_PREFIX)):
            parsed = _parse_external_session_setting_key(key, PENDING_EXTERNAL_SESSION_PREFIX)
            if parsed is None:
                continue
            session_provider, session_id = parsed
            candidates = available_by_provider.get(session_provider)
            if not candidates:
                continue
            touched_providers.add(session_provider)
            session = self.store.get_session(session_provider, session_id)
            if session is None or session.status not in {SessionStatus.ACTIVE, SessionStatus.IDLE}:
                self.store.delete_setting(key)
                continue
            assigned_key = f"{EXTERNAL_SESSION_AGENT_PREFIX}{session_provider.value}.{session_id}"
            if self.store.get_setting(assigned_key):
                self.store.delete_setting(key)
                continue
            agent = candidates.pop(0)
            self.store.set_setting(assigned_key, agent.agent_id)
            self.store.delete_setting(key)
            self._ensure_external_session_thread(channel_id, session, agent)
            assigned += 1

        for touched_provider in touched_providers:
            self._update_external_capacity_notice(channel_id, touched_provider)
        return assigned

    def _ensure_external_session_thread(
        self,
        channel_id: str,
        session: AgentSession,
        agent: TeamAgent,
    ) -> SlackThreadRef:
        existing = self._external_session_thread_for_channel(
            session.provider,
            session.session_id,
            channel_id,
        )
        if existing is not None:
            return existing
        summary = self.store.get_setting(
            f"{EXTERNAL_SESSION_SUMMARY_PREFIX}{session.provider.value}.{session.session_id}"
        )
        posted = self.gateway.post_session_parent(
            channel_id,
            format_session_parent(session, summary),
            agent,
            icon_url=self._agent_icon_url(agent),
        )
        thread = SlackThreadRef(channel_id, posted.ts, posted.ts)
        self.store.upsert_slack_thread_for_session(
            session.provider,
            session.session_id,
            self.team_id,
            thread,
        )
        return thread

    def _update_external_capacity_notice(self, channel_id: str, provider: Provider) -> None:
        setting_key = self._capacity_notice_ts_key(provider)
        existing_ts = self.store.get_setting(setting_key)
        if not existing_ts:
            return
        pending_count = len(
            self.store.list_settings(f"{PENDING_EXTERNAL_SESSION_PREFIX}{provider.value}.")
        )
        if pending_count > 0:
            label = provider.value.title()
            plural = "session is" if pending_count == 1 else "sessions are"
            text = (
                f"No {label} team seat is available for sessions started outside Slack. "
                f"{pending_count} {label} {plural} waiting. "
                "Hire one matching agent and Slackgentic will backfill the transcript."
            )
            self.gateway.update_message(
                channel_id,
                existing_ts,
                text,
                blocks=build_external_session_capacity_blocks(provider, pending_count),
            )
            return
        label = provider.value.title()
        self.gateway.update_message(
            channel_id,
            existing_ts,
            f"{label} capacity for sessions started outside Slack is available now.",
        )
        self.store.delete_setting(setting_key)

    def post_channel_overview(self, channel_id: str) -> str:
        codex_command = f"codex --remote {self.codex_app_server_url}"
        command = self.slash_command
        text = "\n".join(
            [
                (
                    "Slackgentic is ready. Write anything here to start a task, "
                    "or use `@agentname ...` to ask a specific agent."
                ),
                (
                    "In a task thread, reply with `somebody ...` to bring in "
                    "another agent for a subtask; the original agent picks it "
                    "back up with the added context."
                ),
                (
                    "Add `#dangerous-mode` to a task to launch that agent with "
                    "Codex no-sandbox/no-approval mode or Claude skip-permissions. "
                    "The roster marks active dangerous-mode tasks."
                ),
                (
                    "Run commands by typing them directly in this channel, "
                    f"or as `{command} <command>`: "
                    f"`{command} status`, `{command} show roster`, "
                    f"`{command} scheduled tasks`, `{command} hire 3 agents`, "
                    "or just `status`, `show roster`, `scheduled tasks`, "
                    "`hire 3 agents`."
                ),
                f"Codex outside Slack: `{codex_command}` creates a tracking thread here.",
                (
                    "Claude outside Slack: run `slackgentic claude-channel --install` once, "
                    f"then `{CLAUDE_EXTERNAL_COMMAND}` creates a tracking thread here. Restart "
                    "already-open Claude sessions after installing. Slack replies and native "
                    "Claude tool approvals relay through it; no extra MCP flag is needed unless "
                    "you use `--strict-mcp-config`."
                ),
            ]
        )
        posted = self.gateway.post_message(
            channel_id,
            text,
            blocks=build_channel_overview_blocks(
                command,
                codex_command,
                CLAUDE_EXTERNAL_COMMAND,
            ),
        )
        self._pin_message(channel_id, posted.ts, "channel overview message")
        return posted.ts

    def post_roster(
        self,
        channel_id: str,
        thread_ts: str | None = None,
        remember: bool = True,
    ) -> str:
        agents = self.store.list_team_agents()
        statuses = self._roster_statuses(agents)
        text = _roster_text(agents, statuses)
        blocks = build_team_roster_blocks(agents, statuses)
        posted = self.gateway.post_message(
            channel_id,
            text,
            blocks=blocks,
            thread_ts=thread_ts,
        )
        if remember:
            self._remember_roster_message(channel_id, posted.ts)
            self._remember_roster_render(channel_id, posted.ts, text, blocks)
            if thread_ts is None:
                self._pin_roster_once(channel_id, posted.ts)
        return posted.ts

    def refresh_or_post_roster(
        self,
        channel_id: str,
        *,
        discover: bool = True,
        pin: bool = True,
    ) -> str:
        if discover:
            self._discover_recent_roster_messages(channel_id)
        roster_ts_values = self._remembered_roster_ts_values(channel_id)
        if roster_ts_values:
            text, blocks = self._current_roster_payload()
            latest_roster_ts = roster_ts_values[-1]
            self.store.set_setting(SETTING_CHANNEL_ID, channel_id)
            self.store.set_setting(SETTING_ROSTER_TS, latest_roster_ts)
            if pin:
                try:
                    self._pin_roster_once(channel_id, latest_roster_ts)
                except Exception:
                    LOGGER.debug("failed to pin latest Slack roster message", exc_info=True)
            for roster_ts in roster_ts_values:
                if self._roster_render_is_current(channel_id, roster_ts, text, blocks):
                    continue
                try:
                    self.gateway.update_message(
                        channel_id,
                        roster_ts,
                        text,
                        blocks=blocks,
                    )
                    self._remember_roster_render(channel_id, roster_ts, text, blocks)
                except Exception:
                    LOGGER.debug(
                        "failed to update Slack roster message %s",
                        roster_ts,
                        exc_info=True,
                    )
            return latest_roster_ts
        return self.post_roster(channel_id)

    def refresh_roster_message_or_post(
        self,
        channel_id: str,
        message_ts: str | None,
        *,
        pin: bool = False,
    ) -> str:
        if not message_ts:
            return self.refresh_or_post_roster(channel_id, discover=False, pin=pin)
        self._remember_roster_message(channel_id, message_ts)
        text, blocks = self._current_roster_payload()
        if pin:
            try:
                self._pin_roster_once(channel_id, message_ts)
            except Exception:
                LOGGER.debug("failed to pin Slack roster message", exc_info=True)
        if self._roster_render_is_current(channel_id, message_ts, text, blocks):
            return message_ts
        try:
            self.gateway.update_message(
                channel_id,
                message_ts,
                text,
                blocks=blocks,
            )
            self._remember_roster_render(channel_id, message_ts, text, blocks)
            return message_ts
        except Exception:
            LOGGER.debug(
                "failed to update Slack roster message %s; posting replacement",
                message_ts,
                exc_info=True,
            )
            return self.post_roster(channel_id)

    def _current_roster_payload(self) -> tuple[str, list[dict]]:
        agents = self.store.list_team_agents()
        statuses = self._roster_statuses(agents)
        return _roster_text(agents, statuses), build_team_roster_blocks(agents, statuses)

    def _refresh_existing_roster(self, channel_id: str) -> None:
        if self._remembered_roster_ts_values(channel_id):
            self.refresh_or_post_roster(channel_id, discover=False, pin=False)
            return
        self._discover_recent_roster_messages(channel_id)
        if self._remembered_roster_ts_values(channel_id):
            self.refresh_or_post_roster(channel_id, discover=False, pin=False)

    def _start_runtime_task(self, task: AgentTask, agent, thread: SlackThreadRef) -> bool:
        started = self.runtime.start_task(task, agent, thread) if self.runtime else True
        if started:
            self._remember_agent_authored_message(task, agent, thread, task.parent_message_ts)
            self._refresh_existing_roster(thread.channel_id)
        return started

    def _remember_roster_message(self, channel_id: str, message_ts: str) -> None:
        self.store.set_setting(SETTING_CHANNEL_ID, channel_id)
        self.store.set_setting(SETTING_ROSTER_TS, message_ts)
        self.store.set_setting(_roster_message_setting_key(channel_id, message_ts), message_ts)

    def _remember_roster_render(
        self,
        channel_id: str,
        message_ts: str,
        text: str,
        blocks: list[dict],
    ) -> None:
        self.store.set_setting(
            _roster_render_hash_setting_key(channel_id, message_ts),
            _roster_render_hash(text, blocks),
        )

    def _roster_render_is_current(
        self,
        channel_id: str,
        message_ts: str,
        text: str,
        blocks: list[dict],
    ) -> bool:
        return self.store.get_setting(
            _roster_render_hash_setting_key(channel_id, message_ts)
        ) == _roster_render_hash(text, blocks)

    def _remembered_roster_ts_values(self, channel_id: str) -> list[str]:
        values: list[str] = []
        legacy_ts = self.store.get_setting(SETTING_ROSTER_TS)
        if legacy_ts:
            values.append(legacy_ts)
        prefix = _roster_message_channel_prefix(channel_id)
        for value in self.store.list_settings(prefix).values():
            if value:
                values.append(value)
        return sorted(dict.fromkeys(values), key=_slack_ts_sort_key)

    def _discover_recent_roster_messages(self, channel_id: str) -> None:
        discovery_key = _roster_discovery_setting_key(channel_id)
        if self.store.get_setting(discovery_key):
            return
        try:
            messages = self.gateway.channel_messages(channel_id, limit=200)
        except Exception:
            LOGGER.debug("failed to discover recent Slack roster messages", exc_info=True)
            return
        for message in messages:
            message_ts = message.get("ts")
            if not isinstance(message_ts, str) or not message_ts:
                continue
            if not _is_roster_message(message):
                continue
            self.store.set_setting(_roster_message_setting_key(channel_id, message_ts), message_ts)
        self.store.set_setting(discovery_key, utc_now().isoformat())

    def handle_external_session_occupancy_change(self, channel_id: str) -> None:
        try:
            self.evaluate_pending_deferred_work()
        except Exception:
            LOGGER.debug(
                "failed to evaluate deferred work after external session occupancy changed",
                exc_info=True,
            )
        self._resume_pending_work_requests(channel_id)
        self.refresh_or_post_roster(channel_id)

    def resume_pending_work_requests(self, channel_id: str) -> int:
        resumed = self._resume_pending_work_requests(channel_id)
        if resumed:
            self.refresh_or_post_roster(channel_id)
        return resumed

    def resume_pending_work_requests_for_configured_channel(self) -> int:
        channel_id = self._configured_agent_channel_id()
        if not channel_id:
            return 0
        return self.resume_pending_work_requests(channel_id)

    def _pin_roster(self, channel_id: str, message_ts: str) -> bool:
        return self._pin_message(channel_id, message_ts, "roster message")

    def _pin_roster_once(self, channel_id: str, message_ts: str) -> None:
        key = _roster_pinned_setting_key(channel_id, message_ts)
        if self.store.get_setting(key):
            return
        if self._pin_roster(channel_id, message_ts):
            self.store.set_setting(key, utc_now().isoformat())

    def _pin_message(self, channel_id: str, message_ts: str, label: str) -> bool:
        try:
            self.gateway.pin_message(channel_id, message_ts)
        except Exception:
            LOGGER.debug("failed to pin Slack %s", label, exc_info=True)
            return False
        return True

    def post_initial_introductions(self, channel_id: str) -> None:
        agents = self.store.list_team_agents()
        messages = build_initialization_messages(agents)
        if messages:
            self.gateway.post_team_initialization(
                channel_id,
                agents,
                messages,
                icon_url_for=self._agent_icon_url,
            )

    def publish_usage(
        self,
        channel_id: str,
        *,
        show_loading: bool = False,
        fresh: bool = False,
    ) -> str:
        day = day_string("today")
        setting_key = f"{SETTING_USAGE_TS_PREFIX}{day}"
        ts = None if fresh else self.store.get_setting(setting_key)
        if show_loading and (
            not ts
            or not self._try_update_message(
                channel_id,
                ts,
                ":hourglass_flowing_sand: Getting status...",
            )
        ):
            posted = self.gateway.post_message(
                channel_id,
                ":hourglass_flowing_sand: Getting status...",
            )
            ts = posted.ts
            self.store.set_setting(setting_key, ts)

        text = format_daily_usage(
            day,
            collect_daily_usage(day, home=self.home),
            collect_weekly_usage(day, home=self.home),
        )
        if ts and self._try_update_message(channel_id, ts, text):
            return ts
        posted = self.gateway.post_message(channel_id, text)
        self.store.set_setting(setting_key, posted.ts)
        return posted.ts

    def _try_update_message(
        self,
        channel_id: str,
        ts: str,
        text: str,
        *,
        blocks: list[dict] | None = None,
    ) -> bool:
        try:
            self.gateway.update_message(channel_id, ts, text, blocks=blocks)
        except Exception:
            LOGGER.debug("failed to update Slack message %s in %s", ts, channel_id, exc_info=True)
            return False
        return True

    def cancel_orphaned_active_tasks(self) -> int:
        if self.runtime is None:
            return 0
        resumed = 0
        now = utc_now()
        for task in self.store.list_agent_tasks():
            if task.status not in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}:
                continue
            if not task.thread_ts:
                continue
            is_running = getattr(self.runtime, "is_task_running", None)
            if callable(is_running) and is_running(task.task_id):
                continue
            if MANAGED_RUN_STARTED_METADATA_KEY not in task.metadata:
                continue
            if not should_resume_managed_run(task, now=now):
                LOGGER.info(
                    "skipping resume of orphaned task %s (age/attempt bounds exceeded: attempts=%d)",
                    task.task_id,
                    managed_run_resume_attempts(task),
                )
                self._abandon_orphaned_task(task)
                continue
            agent = self.store.get_team_agent(task.agent_id)
            if agent is None:
                continue
            thread = SlackThreadRef(task.channel_id, task.thread_ts, task.parent_message_ts)
            resume = getattr(self.runtime, "resume_orphaned_task", None)
            started = (
                resume(task, agent, thread)
                if callable(resume)
                else self.runtime.start_task(task, agent, thread)
            )
            if started:
                resumed += 1
        return resumed

    def _abandon_orphaned_task(self, task: AgentTask) -> None:
        metadata = dict(task.metadata)
        metadata.pop(MANAGED_RUN_STARTED_METADATA_KEY, None)
        metadata.pop("managed_run_resume_attempts", None)
        cancelled = replace(
            task,
            status=AgentTaskStatus.CANCELLED,
            metadata=metadata,
            updated_at=utc_now(),
        )
        try:
            self.store.upsert_agent_task(cancelled)
            self.store.delete_managed_thread_task(task.task_id)
            if task.session_provider is not None and task.session_id:
                clear_managed_session(self.store, task.session_provider, task.session_id)
            self._refresh_after_task_terminal_state(cancelled)
        except Exception:
            LOGGER.debug("failed to abandon stale orphaned task %s", task.task_id, exc_info=True)

    def _refresh_after_task_terminal_state(self, task: AgentTask) -> None:
        try:
            self.evaluate_pending_deferred_work()
            if self.runtime is not None:
                self._fire_due_deferred_work_now(limit=MAX_PM_SUBTASKS)
        except Exception:
            LOGGER.debug(
                "failed to advance deferred work after task terminal transition",
                exc_info=True,
            )
        self._refresh_pm_status_for_task(task)

    def _ensure_initial_team(self, codex_count: int, claude_count: int) -> None:
        if self.store.list_team_agents(include_fired=True):
            return
        if codex_count + claude_count < 1:
            return
        agents = []
        rng = random.SystemRandom()
        if codex_count:
            agents.extend(
                hire_team_agents(
                    agents,
                    codex_count,
                    Provider.CODEX,
                    randomize_identities=True,
                    rng=rng,
                )
            )
        if claude_count:
            agents.extend(
                hire_team_agents(
                    agents,
                    claude_count,
                    Provider.CLAUDE,
                    randomize_identities=True,
                    rng=rng,
                )
            )
        for agent in agents:
            self.store.upsert_team_agent(agent)

    def _normalize_existing_agents(self) -> None:
        for agent in self.store.list_team_agents(include_fired=True):
            metadata = dict(agent.metadata)
            metadata.setdefault("personal_context", agent_personal_context(agent))
            updated = replace(
                agent,
                role=AGENT_CONTEXT_PLACEHOLDER,
                personality=AGENT_CONTEXT_PLACEHOLDER,
                voice=AGENT_CONTEXT_PLACEHOLDER,
                unique_strength=AGENT_CONTEXT_PLACEHOLDER,
                metadata=metadata,
            )
            if updated != agent:
                self.store.upsert_team_agent(updated)

    def update_channel_id(self) -> str | None:
        return self.default_channel_id or self.store.get_setting(SETTING_CHANNEL_ID)

    def post_update_prompt(self, channel_id: str, candidate: UpdateCandidate) -> str | None:
        posted = self.gateway.post_message(
            channel_id,
            _update_prompt_text(candidate),
            blocks=build_update_prompt_blocks(candidate),
        )
        return posted.ts

    def _update_from_action(
        self,
        payload: dict,
        channel_id: str,
        message_ts: str | None,
    ) -> None:
        if not message_ts:
            return
        update_runner = self.update_runner
        version = payload.get("version")
        if update_runner is None or not isinstance(version, str) or not version:
            self.gateway.post_message(
                channel_id,
                "Slackgentic update handling is not available in this process.",
                thread_ts=message_ts,
            )
            return
        action = payload.get("action")
        if action == "update.install":
            update_runner.start_upgrade(version, channel_id, message_ts)
        elif action == "update.dismiss":
            update_runner.dismiss(version, channel_id, message_ts)

    def _hire_from_action(
        self,
        payload: dict,
        channel_id: str,
        roster_ts: str | None,
    ) -> None:
        count = int(payload.get("count") or 1)
        provider_text = payload.get("provider")
        provider = Provider(provider_text) if provider_text else None
        kind_text = payload.get("kind")
        kind = TeamAgentKind(kind_text) if kind_text else TeamAgentKind.ENGINEER
        if not self._can_hire(count):
            thread_ts = roster_ts or self.store.get_setting(SETTING_ROSTER_TS)
            self._post_text(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread_ts),
                f"{AGENT_LIMIT_MESSAGE} Max team size is {MAX_TEAM_AGENTS}.",
            )
            return
        hired = self.hire_agents(count, provider, kind=kind)
        external_capacity_hire = self._external_capacity_hire_clicked(provider, roster_ts)
        if external_capacity_hire:
            self._assign_hired_agents_to_pending_external_sessions(
                channel_id,
                hired,
                provider,
            )
        ts = roster_ts or self.store.get_setting(SETTING_ROSTER_TS)
        if not ts:
            ts = self.post_roster(channel_id)
        thread = SlackThreadRef(channel_id=channel_id, thread_ts=roster_ts or ts)
        for agent in hired:
            try:
                self.gateway.post_thread_reply(
                    thread,
                    format_agent_introduction(agent),
                    persona=agent,
                    icon_url=self._agent_icon_url(agent),
                )
            except Exception:
                LOGGER.debug("failed to post hired agent introduction", exc_info=True)
        self._resume_pending_work_requests(channel_id)
        if not external_capacity_hire:
            self._assign_hired_agents_to_pending_external_sessions(channel_id, hired, provider)
        roster_update_ts = (
            self.store.get_setting(SETTING_ROSTER_TS)
            if external_capacity_hire
            else roster_ts or self.store.get_setting(SETTING_ROSTER_TS)
        )
        self.refresh_roster_message_or_post(channel_id, roster_update_ts, pin=True)

    def _fire_from_action(
        self,
        payload: dict,
        channel_id: str,
        roster_ts: str | None,
    ) -> None:
        handle = payload.get("handle") or payload.get("agent_id")
        detached_count = 0
        if handle:
            agent = self.store.get_team_agent(str(handle), include_fired=True)
            if agent is not None:
                detached_count = self._detach_external_sessions_for_agent(
                    agent.agent_id,
                    channel_id,
                    post_thread_reply=False,
                    refresh=False,
                )
        if handle:
            self.store.fire_team_agent(str(handle))
        ts = self.refresh_roster_message_or_post(channel_id, roster_ts, pin=True)
        thread = SlackThreadRef(channel_id=channel_id, thread_ts=roster_ts or ts)
        if handle:
            if detached_count:
                self.gateway.post_thread_reply(
                    thread,
                    f"Detached external session and removed @{str(handle).lstrip('@')}.",
                )
            else:
                self.gateway.post_thread_reply(thread, f"Removed @{str(handle).lstrip('@')}.")

    def _open_roster_work_modal(
        self,
        payload: dict,
        channel_id: str,
        roster_ts: str | None,
        trigger_id: str | None,
    ) -> None:
        if not trigger_id:
            thread_ts = roster_ts or self.store.get_setting(SETTING_ROSTER_TS)
            self._post_text(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread_ts),
                "Use `show roster` again to open the work form from Slack.",
            )
            return
        agent = None
        agent_id = str(payload.get("agent_id") or "")
        handle = str(payload.get("handle") or "")
        if agent_id or handle:
            agent = self.store.get_team_agent(agent_id or handle)
            if agent is None:
                self._post_text(
                    SlackReplyTarget(channel_id=channel_id, thread_ts=roster_ts),
                    "That agent is no longer active.",
                )
                self.refresh_or_post_roster(channel_id)
                return
            if agent.agent_id in self._busy_agent_ids_for_roster_work_target():
                self._post_text(
                    SlackReplyTarget(channel_id=channel_id, thread_ts=roster_ts),
                    f"@{agent.handle} is already occupied. Choose an available agent.",
                )
                self.refresh_or_post_roster(channel_id)
                return
        self.gateway.open_view(
            trigger_id,
            _roster_work_modal(
                agent,
                channel_id=channel_id,
                message_ts=roster_ts,
                initial_timing=str(payload.get("mode") or "now"),
                occupied_handles=self._occupied_handle_dependency_labels(),
            ),
        )

    def _handle_roster_work_submission(
        self,
        payload: dict,
        *,
        async_success: bool = False,
    ) -> dict | None:
        view = payload.get("view") or {}
        metadata = _decode_roster_work_metadata(view.get("private_metadata"))
        channel_id = metadata.get("channel_id") or self._configured_agent_channel_id()
        if not channel_id:
            return None
        values = view.get("state", {}).get("values", {})
        request, request_error = self._roster_work_request_from_values(values, metadata)
        if request_error is not None:
            return _view_errors(*request_error)
        assert request is not None
        target_error = self._specific_target_busy_error(request)
        if target_error is not None:
            return _view_errors("roster_work_prompt", target_error)

        timing = _view_selected_value(values, "roster_work_timing", "value") or "now"
        dependency, dependency_error = self._roster_work_dependency_from_values(values)
        if dependency_error is not None:
            return _view_errors("roster_work_dependency", dependency_error)

        pm_target = self._pm_agent_for_request_target(request)
        if pm_target is not None and (
            timing in {"once", "daily", "weekly"} or dependency is not None
        ):
            return _view_errors(
                "roster_work_prompt",
                (
                    f"@{pm_target.handle} is a PM and plans a fresh initiative "
                    "each time it is engaged. Schedule or defer a worker agent "
                    "instead, or set timing to 'now' to start the PM "
                    "immediately."
                ),
            )

        run_at_text = (_view_plain_value(values, "roster_work_run_at", "value") or "").strip()
        delay_seconds, delay_error = _optional_delay_seconds(
            _view_plain_value(values, "roster_work_delay", "value")
        )
        if delay_error:
            return _view_errors("roster_work_delay", delay_error)
        if delay_seconds is not None and dependency is None:
            return _view_errors("roster_work_dependency", "Delay only applies after a dependency.")

        requested_by = (payload.get("user") or {}).get("id")
        if dependency is not None:
            if timing in {"daily", "weekly"}:
                return _view_errors(
                    "roster_work_timing",
                    "Repeating schedules cannot wait on an agent yet. Use now or once.",
                )
            run_at, run_at_error = _optional_future_timestamp(run_at_text)
            if run_at_error:
                return _view_errors("roster_work_run_at", run_at_error)

            def callback() -> None:
                self._create_roster_deferred_work(
                    channel_id,
                    request,
                    dependency,
                    run_at=run_at,
                    delay_seconds=delay_seconds,
                    requested_by_slack_user=requested_by,
                )

            if async_success:
                self._run_after_view_ack("roster-work-deferred", callback)
            else:
                callback()
            return None

        if timing == "now":
            if run_at_text:
                return _view_errors("roster_work_timing", "Choose once to use a run-at time.")

            def callback() -> None:
                self._start_roster_work_now(
                    request,
                    channel_id,
                    requested_by_slack_user=requested_by,
                )

            if async_success:
                self._run_after_view_ack("roster-work-now", callback)
            else:
                callback()
            return None

        if timing == "once":
            run_at, run_at_error = _required_future_timestamp(run_at_text)
            if run_at_error:
                return _view_errors("roster_work_run_at", run_at_error)

            def callback() -> None:
                self._create_roster_scheduled_work(
                    channel_id,
                    request,
                    schedule_kind=ScheduledWorkKind.ONE_OFF,
                    next_run_at=run_at,
                    recurrence={},
                    timezone=None,
                    description=f"once at {_format_schedule_timestamp(run_at)}",
                    requested_by_slack_user=requested_by,
                )

            if async_success:
                self._run_after_view_ack("roster-work-scheduled", callback)
            else:
                callback()
            return None

        if timing in {"daily", "weekly"}:
            recurrence, recurrence_error = _recurrence_from_roster_work_values(values, timing)
            if recurrence_error is not None:
                return _view_errors(*recurrence_error)
            next_run_at = next_run_after(recurrence, after=utc_now())
            if next_run_at is None:
                return _view_errors("roster_work_time", "Could not compute the next run.")
            timezone = str(recurrence["timezone"])
            if timing == "weekly":
                weekday = _weekday_label(recurrence.get("weekday")) or "selected day"
                description = f"weekly on {weekday} at {recurrence['time']} {timezone}"
            else:
                description = f"daily at {recurrence['time']} {timezone}"

            def callback() -> None:
                self._create_roster_scheduled_work(
                    channel_id,
                    request,
                    schedule_kind=ScheduledWorkKind.RECURRING,
                    next_run_at=next_run_at,
                    recurrence=recurrence,
                    timezone=timezone,
                    description=description,
                    requested_by_slack_user=requested_by,
                )

            if async_success:
                self._run_after_view_ack("roster-work-scheduled", callback)
            else:
                callback()
            return None

        return _view_errors("roster_work_timing", "Choose when this work should run.")

    def _roster_work_request_from_values(
        self,
        values: dict,
        metadata: dict[str, str],
    ) -> tuple[WorkRequest | None, tuple[str, str] | None]:
        prompt = (_view_plain_value(values, "roster_work_prompt", "value") or "").strip()
        if not prompt:
            return None, ("roster_work_prompt", "Enter the work to assign.")
        kind_value = _view_selected_value(values, "roster_work_kind", "value") or "work"
        try:
            task_kind = AgentTaskKind(kind_value)
        except ValueError:
            return None, ("roster_work_kind", "Choose work or review.")
        dangerous = "dangerous" in _view_checked_values(
            values, "roster_work_permissions", "dangerous"
        )
        agent = None
        handle = metadata.get("handle")
        agent_id = metadata.get("agent_id")
        if handle or agent_id:
            agent = self.store.get_team_agent(agent_id or handle or "")
            if agent is None:
                return None, ("roster_work_prompt", "That target agent is no longer active.")
        return (
            WorkRequest(
                prompt=prompt,
                assignment_mode=AssignmentMode.SPECIFIC if agent else AssignmentMode.ANYONE,
                requested_handle=agent.handle if agent else None,
                task_kind=task_kind,
                permission_mode=(
                    PermissionMode.DANGEROUS if dangerous else DEFAULT_PERMISSION_MODE
                ),
            ),
            None,
        )

    def _specific_target_busy_error(self, request: WorkRequest) -> str | None:
        if request.assignment_mode != AssignmentMode.SPECIFIC or not request.requested_handle:
            return None
        agent = self.store.get_team_agent(request.requested_handle)
        if agent is None:
            return "That target agent is no longer active."
        if agent.agent_id in self._busy_agent_ids_for_roster_work_target():
            return f"@{agent.handle} is already occupied. Choose an available agent."
        return None

    def _roster_work_dependency_from_values(
        self,
        values: dict,
    ) -> tuple[WorkDependency | None, str | None]:
        selected = _view_selected_value(values, "roster_work_dependency", "value") or "none"
        if selected == "none":
            return None, None
        try:
            handle = normalize_handle(selected)
        except ValueError:
            return None, "Choose an occupied agent."
        occupied = dict(self._occupied_handle_task_ids())
        task_id = occupied.get(handle)
        if not task_id:
            return None, f"@{handle} is not currently occupied."
        return (
            WorkDependency(
                kind=WorkDependencyKind.AGENT_BUSY,
                handle=handle,
                task_id=task_id,
            ),
            None,
        )

    def _start_roster_work_now(
        self,
        request: WorkRequest,
        channel_id: str,
        *,
        requested_by_slack_user: str | None,
    ) -> bool:
        pm_target = self._pm_agent_for_request_target(request)
        if pm_target is not None:
            # PM agents must always own a `pm_initiatives` row so the
            # PM_PLAN signal they emit can find an approval card to attach
            # to. Worker-style assignment bypasses that contract — redirect
            # through the PM dispatcher via a posted roster work parent
            # that becomes the initiative thread.
            thread = self._create_roster_work_parent(channel_id, request, "PM brief")
            dispatched = self._dispatch_pm_initiative(
                channel_id=channel_id,
                project_body=request.prompt.strip() or request.prompt,
                requested_by_slack_user=requested_by_slack_user,
                thread=thread,
                targeted_pm_handle=pm_target.handle,
                request_message_ts=thread.message_ts,
            )
            self.refresh_or_post_roster(channel_id)
            return dispatched
        result = assign_work_request(
            self.store,
            request,
            channel_id,
            requested_by_slack_user=requested_by_slack_user,
            exclude_agent_ids=self._busy_agent_ids_for_assignment(),
        )
        if result is None:
            self._post_assignment_unavailable(
                SlackReplyTarget(channel_id=channel_id),
                request,
                requested_by_slack_user=requested_by_slack_user,
            )
            self.refresh_or_post_roster(channel_id)
            return False
        blocks = _task_thread_blocks(result.task, result.agent)
        text = format_agent_assignment(
            result.agent,
            result.request.prompt,
            requested_by_slack_user,
            dangerous_mode=_task_dangerous_mode(result.task),
        )
        thread = self.gateway.post_task_parent(
            channel_id,
            text,
            result.agent,
            blocks=blocks,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(
            result.task.task_id,
            thread.thread_ts,
            thread.message_ts,
        )
        task = self.store.get_agent_task(result.task.task_id) or result.task
        self._start_runtime_task(task, result.agent, thread)
        self.refresh_or_post_roster(channel_id)
        return True

    def _create_roster_scheduled_work(
        self,
        channel_id: str,
        request: WorkRequest,
        *,
        schedule_kind: ScheduledWorkKind,
        next_run_at,
        recurrence: dict[str, object],
        timezone: str | None,
        description: str,
        requested_by_slack_user: str | None,
    ) -> ScheduledWork:
        thread = self._create_roster_work_parent(channel_id, request, "Scheduling work")
        scheduled = self.store.create_scheduled_work_request(
            thread,
            request,
            schedule_kind=schedule_kind,
            next_run_at=next_run_at,
            recurrence=recurrence,
            timezone=timezone,
            requested_by_slack_user=requested_by_slack_user,
        )
        text = _format_scheduled_work_ack(scheduled, description, request)
        if not self._try_update_message(thread.channel_id, thread.thread_ts, text):
            self.gateway.post_thread_reply(thread, text)
        self.refresh_or_post_roster(channel_id)
        return scheduled

    def _create_roster_deferred_work(
        self,
        channel_id: str,
        request: WorkRequest,
        dependency: WorkDependency,
        *,
        run_at,
        delay_seconds: int | None,
        requested_by_slack_user: str | None,
    ) -> DeferredWork:
        thread = self._create_roster_work_parent(channel_id, request, "Deferring work")
        deferred = self.store.create_deferred_work_request(
            thread,
            request,
            depends_on=(dependency,),
            after_dep_delay_seconds=delay_seconds,
            run_at=run_at,
            description=f"after @{dependency.handle} finishes",
            requested_by_slack_user=requested_by_slack_user,
        )
        text = _format_deferred_work_ack(
            deferred,
            deferred.description or f"after @{dependency.handle} finishes",
            request,
            dependency_labeler=self._agent_busy_dependency_label,
        )
        if not self._try_update_message(thread.channel_id, thread.thread_ts, text):
            self.gateway.post_thread_reply(thread, text)
        self.evaluate_pending_deferred_work(deferred.deferred_id)
        self.refresh_or_post_roster(channel_id)
        return deferred

    def _create_roster_work_parent(
        self,
        channel_id: str,
        request: WorkRequest,
        label: str,
    ) -> SlackThreadRef:
        target = "somebody"
        if request.assignment_mode == AssignmentMode.SPECIFIC and request.requested_handle:
            target = f"@{request.requested_handle}"
        posted = self.gateway.post_message(
            channel_id,
            f"{label} for {target}: `{_shorten(request.prompt, 180)}`",
        )
        return SlackThreadRef(channel_id, posted.ts, posted.ts)

    def _task_from_action(
        self,
        payload: dict,
        channel_id: str,
        message_ts: str | None,
    ) -> None:
        task_id = payload.get("task_id")
        if not task_id:
            return
        action = payload["action"]
        task = self.store.get_agent_task(str(task_id))
        thread_ts = task.thread_ts if task and task.thread_ts else message_ts
        thread = SlackThreadRef(channel_id=channel_id, thread_ts=thread_ts or "")
        if action in {"task.done", "task.finish", "task.cancel", "task.pause"}:
            closed_tasks: list[AgentTask] = []
            if task is not None and task.thread_ts:
                for thread_task in self.store.list_agent_tasks(include_done=True):
                    if (
                        thread_task.agent_id == task.agent_id
                        and thread_task.channel_id == task.channel_id
                        and thread_task.thread_ts == task.thread_ts
                        and thread_task.status in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}
                    ):
                        if self.runtime:
                            self.runtime.stop_task(thread_task.task_id, AgentTaskStatus.DONE)
                        else:
                            self.store.update_agent_task_status(
                                thread_task.task_id,
                                AgentTaskStatus.DONE,
                            )
                        closed_tasks.append(thread_task)
            elif self.runtime:
                if task is not None and task.status in {
                    AgentTaskStatus.QUEUED,
                    AgentTaskStatus.ACTIVE,
                }:
                    self.runtime.stop_task(str(task_id), AgentTaskStatus.DONE)
                    closed_tasks.append(task)
            else:
                if task is not None and task.status in {
                    AgentTaskStatus.QUEUED,
                    AgentTaskStatus.ACTIVE,
                }:
                    self.store.update_agent_task_status(str(task_id), AgentTaskStatus.DONE)
                    closed_tasks.append(task)
            if not closed_tasks and task is not None:
                self._mark_task_complete(task, thread, include_thread=True)
            for completed_task in closed_tasks:
                self._mark_task_complete(completed_task, thread, include_thread=True)
            if closed_tasks and thread.thread_ts:
                self.gateway.post_thread_reply(thread, "Finished and freed up this agent.")
            try:
                self.evaluate_pending_deferred_work()
            except Exception:
                LOGGER.debug(
                    "failed to evaluate deferred work after task button completion",
                    exc_info=True,
                )
            self._resume_pending_work_requests(channel_id)
            self.refresh_or_post_roster(channel_id)

    def _pm_initiative_from_action(
        self,
        payload: dict,
        slack_payload: dict,
        channel_id: str,
        message_ts: str | None,
    ) -> None:
        action = payload.get("action") or ""
        initiative_id = str(payload.get("initiative_id") or "")
        if not initiative_id:
            return
        initiative = self.store.get_pm_initiative(initiative_id)
        if initiative is None:
            return
        actor = (slack_payload.get("user") or {}).get("id")
        thread = SlackThreadRef(
            channel_id=initiative.channel_id,
            thread_ts=initiative.thread_ts,
        )
        if action == "pm_initiative.start":
            if initiative.status != PmInitiativeStatus.AWAITING_APPROVAL:
                self._strip_pm_plan_buttons(
                    initiative, message_ts, status_label=initiative.status.value
                )
                return
            plan_message_ts = initiative.pending_plan_message_ts
            promoted = self._execute_pm_plan(initiative, approver_slack_user=actor)
            if promoted is not None:
                self._strip_pm_plan_buttons(promoted, message_ts, status_label="approved")
                if plan_message_ts and plan_message_ts != message_ts:
                    self._strip_pm_plan_buttons(promoted, plan_message_ts, status_label="approved")
            return
        if action == "pm_initiative.hire_and_start":
            if initiative.status != PmInitiativeStatus.AWAITING_APPROVAL:
                self._strip_pm_plan_buttons(
                    initiative, message_ts, status_label=initiative.status.value
                )
                return
            self._hire_for_pm_capacity_and_execute(
                initiative,
                channel_id,
                message_ts,
                approver_slack_user=actor,
            )
            return
        if action == "pm_initiative.cancel":
            if initiative.status not in {
                PmInitiativeStatus.AWAITING_APPROVAL,
                PmInitiativeStatus.PLANNING,
            }:
                self._strip_pm_plan_buttons(
                    initiative, message_ts, status_label=initiative.status.value
                )
                return
            self.store.set_pm_initiative_pending_plan(
                initiative.initiative_id,
                plan_json=None,
                status=PmInitiativeStatus.CANCELLED,
            )
            mention = f"<@{actor}> " if actor else ""
            self.gateway.post_thread_reply(
                thread,
                f"{mention}cancelled this PM initiative. No subtasks were started.",
            )
            self._strip_pm_plan_buttons(initiative, message_ts, status_label="cancelled")
            self._clear_pm_initiative_request_status(initiative)
            return

    def _strip_pm_plan_buttons(
        self,
        initiative: PmInitiative,
        message_ts: str | None,
        *,
        status_label: str,
    ) -> None:
        target_ts = message_ts or initiative.pending_plan_message_ts
        if not target_ts:
            return
        suffix = f"\n\n_Plan {status_label}._"
        try:
            existing = self.gateway.thread_messages(
                initiative.channel_id, initiative.thread_ts, limit=200
            )
        except Exception:
            existing = []
        original_text = ""
        for message in existing:
            if message.get("ts") == target_ts:
                original_text = message.get("text") or ""
                break
        new_text = (original_text or f"PM plan for `{initiative.initiative_id}`") + suffix
        resolved_blocks = [
            {
                "type": "section",
                "block_id": f"pm.plan.resolved.{initiative.initiative_id}.{index}",
                "text": {"type": "mrkdwn", "text": chunk},
            }
            for index, chunk in enumerate(_slack_mrkdwn_chunks(new_text), start=1)
        ]
        self._try_update_message(
            initiative.channel_id,
            target_ts,
            _shorten(new_text, 2600),
            blocks=resolved_blocks,
        )

    def _schedule_from_action(
        self,
        payload: dict,
        channel_id: str,
        message_ts: str | None,
        trigger_id: str | None,
    ) -> None:
        schedule_id = str(payload.get("schedule_id") or "")
        if not schedule_id:
            return
        action = payload.get("action")
        if action == "schedule.cancel":
            cancelled = self.store.cancel_scheduled_work(schedule_id)
            if cancelled is None:
                self.gateway.post_message(
                    channel_id,
                    f"I could not find an active schedule named `{schedule_id}`.",
                )
                return
            self.gateway.post_thread_reply(
                SlackThreadRef(cancelled.channel_id, cancelled.thread_ts),
                f"Descheduled `{schedule_id}`.",
            )
            self._update_scheduled_tasks_message(channel_id, message_ts)
            self.refresh_or_post_roster(channel_id)
            return
        if action == "schedule.change":
            scheduled = self.store.get_scheduled_work(schedule_id)
            if scheduled is None or scheduled.status not in {
                ScheduledWorkStatus.PENDING,
                ScheduledWorkStatus.CLAIMED,
            }:
                self.gateway.post_message(
                    channel_id,
                    f"I could not find an active schedule named `{schedule_id}`.",
                )
                return
            if not trigger_id:
                self.gateway.post_message(
                    channel_id,
                    f"Use `scheduled tasks` again to change `{schedule_id}` from Slack.",
                )
                return
            self.gateway.open_view(
                trigger_id,
                _schedule_change_modal(scheduled, channel_id=channel_id, message_ts=message_ts),
            )

    def _handle_schedule_change_submission(
        self,
        payload: dict,
        *,
        async_success: bool = False,
    ) -> dict | None:
        view = payload.get("view") or {}
        metadata = _decode_schedule_change_metadata(view.get("private_metadata"))
        schedule_id = metadata.get("schedule_id")
        if not schedule_id:
            return None
        scheduled = self.store.get_scheduled_work(schedule_id)
        if scheduled is None or scheduled.status not in {
            ScheduledWorkStatus.PENDING,
            ScheduledWorkStatus.CLAIMED,
        }:
            return _view_errors("schedule_next_run", "That schedule is no longer active.")
        values = view.get("state", {}).get("values", {})
        next_run_text = (_view_plain_value(values, "schedule_next_run", "value") or "").strip()
        next_run_at = parse_timestamp(next_run_text)
        if next_run_at is None:
            return _view_errors(
                "schedule_next_run",
                "Enter an ISO timestamp, for example 2026-05-16T17:00:00-05:00.",
            )
        if next_run_at <= utc_now():
            return _view_errors("schedule_next_run", "Choose a future time.")
        recurrence = None
        timezone = scheduled.timezone
        if scheduled.schedule_kind == ScheduledWorkKind.RECURRING:
            recurrence_text = (
                _view_plain_value(values, "schedule_recurrence", "value") or ""
            ).strip()
            if recurrence_text:
                try:
                    parsed_recurrence = json.loads(recurrence_text)
                except json.JSONDecodeError:
                    return _view_errors("schedule_recurrence", "Enter valid JSON.")
                if not isinstance(parsed_recurrence, dict):
                    return _view_errors("schedule_recurrence", "Recurrence must be a JSON object.")
                if next_run_after(parsed_recurrence, after=next_run_at) is None:
                    return _view_errors(
                        "schedule_recurrence",
                        "Use daily/weekly recurrence with time/timezone or interval recurrence.",
                    )
                recurrence = parsed_recurrence
                recurrence_timezone = parsed_recurrence.get("timezone")
                if isinstance(recurrence_timezone, str) and recurrence_timezone.strip():
                    timezone = recurrence_timezone.strip()
        updated = self.store.reschedule_scheduled_work(
            schedule_id,
            next_run_at=next_run_at,
            recurrence=recurrence,
            timezone=timezone,
        )
        if updated is None:
            return _view_errors("schedule_next_run", "That schedule could not be changed.")

        def notify() -> None:
            self.gateway.post_thread_reply(
                SlackThreadRef(updated.channel_id, updated.thread_ts),
                (
                    f"Changed schedule `{schedule_id}`. "
                    f"Next run: `{_format_schedule_timestamp(updated.next_run_at)}`."
                ),
            )
            channel_id = metadata.get("channel_id")
            message_ts = metadata.get("message_ts")
            if channel_id:
                self._update_scheduled_tasks_message(channel_id, message_ts)
                self.refresh_or_post_roster(channel_id)

        if async_success:
            self._run_after_view_ack("schedule-change", notify)
        else:
            notify()
        return None

    def _update_scheduled_tasks_message(self, channel_id: str, message_ts: str | None) -> None:
        if not message_ts:
            return
        scheduled = self.store.list_scheduled_work(
            statuses=(ScheduledWorkStatus.PENDING, ScheduledWorkStatus.CLAIMED),
            channel_id=channel_id,
            limit=20,
        )
        try:
            self.gateway.update_message(
                channel_id,
                message_ts,
                _scheduled_tasks_text(scheduled),
                blocks=self._scheduled_tasks_blocks(scheduled),
            )
        except Exception:
            LOGGER.debug("failed to update scheduled tasks message", exc_info=True)

    def _external_session_detach_from_action(self, payload: dict, channel_id: str) -> None:
        provider_text = str(payload.get("provider") or "")
        session_id = str(payload.get("session_id") or "")
        try:
            provider = Provider(provider_text)
        except ValueError:
            return
        if not session_id:
            return
        self._detach_external_session(
            provider,
            session_id,
            channel_id,
            post_thread_reply=True,
            refresh=True,
        )

    def _detach_external_sessions_for_agent(
        self,
        agent_id: str,
        channel_id: str,
        *,
        post_thread_reply: bool,
        refresh: bool,
    ) -> int:
        sessions = []
        for key, assigned_agent_id in self.store.list_settings(
            EXTERNAL_SESSION_AGENT_PREFIX
        ).items():
            if assigned_agent_id != agent_id:
                continue
            parsed = _parse_external_session_setting_key(key, EXTERNAL_SESSION_AGENT_PREFIX)
            if parsed is not None:
                sessions.append(parsed)
        detached_count = 0
        for provider, session_id in sessions:
            if self._detach_external_session(
                provider,
                session_id,
                channel_id,
                post_thread_reply=post_thread_reply,
                refresh=False,
            ):
                detached_count += 1
        if refresh and sessions:
            self.handle_external_session_occupancy_change(channel_id)
        return detached_count

    def _detach_external_session(
        self,
        provider: Provider,
        session_id: str,
        channel_id: str,
        *,
        post_thread_reply: bool,
        refresh: bool,
    ) -> bool:
        thread = self._external_session_thread_for_channel(provider, session_id, channel_id)
        key_suffix = f"{provider.value}.{session_id}"
        ignored_key = f"{EXTERNAL_SESSION_IGNORED_PREFIX}{key_suffix}"
        already_ignored = self.store.get_setting(ignored_key) is not None
        self.store.set_setting(
            ignored_key,
            utc_now().isoformat(),
        )
        changed = self.store.clear_external_session_tracking(provider, session_id)
        if thread is not None and post_thread_reply:
            self.gateway.post_thread_reply(
                thread,
                "Detached this agent from the external session. "
                "This session will be ignored from now on.",
            )
        if refresh:
            self.handle_external_session_occupancy_change(channel_id)
        return changed or not already_ignored

    def _handle_task_thread_reply(self, event: dict, channel_id: str, text: str) -> bool:
        thread_ts = event.get("thread_ts")
        message_ts = event.get("ts")
        if not thread_ts or thread_ts == message_ts:
            return False
        task = self.store.get_original_agent_task_by_thread(channel_id, thread_ts)
        if task is None:
            return False
        agent = self.store.get_team_agent(task.agent_id)
        if agent is None and task.status in {AgentTaskStatus.DONE, AgentTaskStatus.CANCELLED}:
            return True
        self._mark_message_acknowledged(channel_id, message_ts)
        if self._maybe_post_pm_status_command(task, channel_id, thread_ts, message_ts, text):
            return True
        if self._maybe_start_pm_replan(task, channel_id, thread_ts, message_ts, text):
            return True
        if self._maybe_start_pm_extension(task, channel_id, thread_ts, message_ts, text):
            return True
        active_task = self._active_thread_task_for_agent(task, channel_id, thread_ts)
        target_task = active_task or task
        if _is_stop_command(text):
            thread = SlackThreadRef(channel_id, thread_ts, task.parent_message_ts)
            if self._stop_multiple_active_thread_tasks(
                channel_id,
                thread_ts,
                thread,
                message_ts,
                event.get("user"),
            ):
                return True
            self._interrupt_thread_task(
                target_task,
                thread,
                message_ts,
            )
            return True
        if self._handle_thread_work_request(task, event, text, agent):
            return True
        if self._record_dependency_if_requested(task.task_id, event, text, agent):
            return True
        followup_text = self._wrap_pm_followup_text(target_task, agent, text)
        if self.runtime and self.runtime.send_to_task(
            target_task.task_id,
            _live_thread_followup_prompt(followup_text),
        ):
            self._remember_request_message_for_task(target_task, message_ts)
            self._mark_message_in_progress(channel_id, message_ts)
            return True
        if agent:
            started = self._start_thread_followup(target_task, event, followup_text, agent)
            if started and message_ts:
                self._mark_message_in_progress(channel_id, message_ts)
            return started
        if task.status in {AgentTaskStatus.DONE, AgentTaskStatus.CANCELLED}:
            return True
        self.gateway.post_thread_reply(
            SlackThreadRef(channel_id, thread_ts),
            "This task's agent is no longer available. You need to hire more agents before I can continue it.",
            persona=agent,
        )
        return True

    def _interrupt_thread_task(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
        message_ts: str | None,
    ) -> None:
        stopped = False
        if self.runtime is not None:
            interrupt = getattr(self.runtime, "interrupt_task", None)
            if callable(interrupt):
                stopped = bool(interrupt(task.task_id))
            else:
                stopped = bool(self.runtime.stop_task(task.task_id, status=None))
        self._mark_message_complete(thread.channel_id, message_ts)
        if stopped:
            self.gateway.post_thread_reply(
                thread,
                "Interrupted the current run. Send the next instruction here to continue.",
            )
            self.refresh_or_post_roster(thread.channel_id)
            return
        # No live worker to interrupt. If the task is still marked ACTIVE/QUEUED
        # in the DB, that's an orphaned-state mismatch (e.g. the worker exited
        # without transitioning the row). Stop should resolve it instead of
        # leaving stale state that confuses follow-up routing.
        latest = self.store.get_agent_task(task.task_id) or task
        if latest.status in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}:
            self._abandon_orphaned_task(latest)
            self.gateway.post_thread_reply(
                thread,
                "Cleared the stale active state on this task. Send the next instruction here to continue.",
            )
        else:
            self.gateway.post_thread_reply(
                thread,
                "No active run is currently running for this task.",
            )
        self.refresh_or_post_roster(thread.channel_id)

    def _stop_multiple_active_thread_tasks(
        self,
        channel_id: str,
        thread_ts: str,
        thread: SlackThreadRef,
        message_ts: str | None,
        slack_user: str | None,
    ) -> bool:
        thread_tasks = [
            item
            for item in self.store.list_agent_tasks(include_done=True)
            if item.channel_id == channel_id
            and item.thread_ts == thread_ts
            and item.status in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}
        ]
        if len(thread_tasks) <= 1:
            return False
        stopped = 0
        for thread_task in thread_tasks:
            if self._stop_thread_task(thread_task, thread, slack_user):
                stopped += 1
            self._mark_task_complete(thread_task, thread, include_thread=False)
        self._mark_message_complete(channel_id, message_ts)
        if stopped:
            self.gateway.post_thread_reply(
                thread,
                f"Stopped {stopped} active runs in this thread.",
            )
        else:
            self.gateway.post_thread_reply(
                thread,
                "No active run is currently running in this thread.",
            )
        self.refresh_or_post_roster(channel_id)
        return True

    def _stop_thread_task(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
        slack_user: str | None,
    ) -> bool:
        stopped = False
        if self.runtime is not None:
            try:
                stopped = bool(self.runtime.stop_task(task.task_id, AgentTaskStatus.CANCELLED))
            except Exception:
                LOGGER.debug("failed to stop thread task %s", task.task_id, exc_info=True)
        if (
            not stopped
            and self.session_bridge is not None
            and task.session_provider is not None
            and task.session_id
        ):
            session = self.store.get_session(task.session_provider, task.session_id)
            if session is not None:
                try:
                    stopped = bool(
                        self.session_bridge.send_to_session(
                            session,
                            "/exit",
                            thread,
                            slack_user=slack_user,
                        )
                    )
                except Exception:
                    LOGGER.debug(
                        "failed to stop external session for thread task %s",
                        task.task_id,
                        exc_info=True,
                    )
        self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
        if task.session_provider is not None and task.session_id:
            clear_managed_session(self.store, task.session_provider, task.session_id)
        return stopped or task.status in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}

    def _handle_external_session_thread_reply(
        self,
        event: dict,
        channel_id: str,
        text: str,
    ) -> bool:
        thread_ts = event.get("thread_ts")
        message_ts = event.get("ts")
        if not thread_ts or thread_ts == message_ts:
            return False
        session = self.store.get_session_for_slack_thread(self.team_id, channel_id, thread_ts)
        if session is None:
            return False
        self._mark_message_acknowledged(channel_id, message_ts)
        if self._handle_external_thread_work_request(session, event, text):
            return True
        if self.session_bridge is None:
            self.gateway.post_thread_reply(
                SlackThreadRef(channel_id, thread_ts),
                "I found the external session thread, but no session bridge is configured.",
            )
            return True
        if not self._reserve_external_session_agent_for_reply(session, channel_id, thread_ts):
            return True
        sent = self.session_bridge.send_to_session(
            session,
            text,
            SlackThreadRef(channel_id, thread_ts),
            slack_user=event.get("user"),
        )
        if sent:
            self._mark_external_session_message_delivered(channel_id, message_ts, session)
            self._mark_message_in_progress(channel_id, message_ts)
        return sent

    def _reserve_external_session_agent_for_reply(
        self,
        session,
        channel_id: str,
        thread_ts: str,
    ) -> bool:
        setting_key = _external_session_agent_setting_key(session)
        assigned_agent_id = self.store.get_setting(setting_key)
        if assigned_agent_id:
            assigned_agent = self.store.get_team_agent(assigned_agent_id)
            if assigned_agent is not None and assigned_agent.kind != TeamAgentKind.PM:
                return True
            self.store.delete_setting(setting_key)
            if assigned_agent is not None and assigned_agent.kind == TeamAgentKind.PM:
                self.store.set_setting(
                    _external_session_ignored_setting_key(session),
                    utc_now().isoformat(),
                )
        if session.status == SessionStatus.ACTIVE:
            return True
        external_busy_agent_ids = self._busy_agent_ids_for_assignment()
        available = [
            agent
            for agent in self.store.idle_team_agents()
            if agent.provider_preference == session.provider
            and agent.kind != TeamAgentKind.PM
            and agent.agent_id not in external_busy_agent_ids
        ]
        if not available:
            self.gateway.post_thread_reply(
                SlackThreadRef(channel_id, thread_ts),
                (
                    f"No available {session.provider.value} agent can revive this "
                    "session right now. Hire or free that provider, then reply "
                    "here again. Or reply with `somebody ...` to let another "
                    "agent start a new session using this thread as context."
                ),
            )
            return False
        self.store.set_setting(setting_key, available[0].agent_id)
        self.refresh_or_post_roster(channel_id)
        return True

    def _handle_external_thread_work_request(self, session, event: dict, text: str) -> bool:
        active_agents = self.store.list_team_agents()
        channel_id = event["channel"]
        thread_ts = event["thread_ts"]
        multi_requests = _multi_specific_work_requests(text, active_agents, split_newlines=True)
        if multi_requests:
            extra_metadata = self._external_thread_task_metadata(session, channel_id, thread_ts)
            extra_metadata["request_message_ts"] = event.get("ts")
            extra_metadata = self._with_linked_thread_context(
                extra_metadata,
                text,
                current_thread=SlackThreadRef(channel_id, thread_ts),
            )
            return self._start_specific_requests_in_thread(
                multi_requests,
                SlackThreadRef(channel_id, thread_ts),
                requested_by_slack_user=event.get("user"),
                extra_metadata=extra_metadata,
            )
        request = _parse_work_request_for_agents(text, active_agents, split_newlines=True)
        if request is None:
            return False
        author_agent = (
            self.store.get_team_agent(request.author_handle) if request.author_handle else None
        )
        target_agent = self.store.get_team_agent(request.requested_handle or "")
        if target_agent is not None and self._start_external_session_agent_followup(
            request,
            target_agent,
            SlackThreadRef(channel_id, thread_ts),
            requested_by_slack_user=event.get("user"),
            request_message_ts=event.get("ts"),
        ):
            self._mark_message_in_progress(channel_id, event.get("ts"))
            return True
        same_thread_agent = self._same_thread_requested_agent(request, channel_id, thread_ts)
        if same_thread_agent is not None:
            started = self._start_same_thread_agent_followup(
                request,
                same_thread_agent,
                SlackThreadRef(channel_id, thread_ts),
                requested_by_slack_user=event.get("user"),
                author_agent=author_agent,
                request_message_ts=event.get("ts"),
            )
            if started:
                self._mark_message_in_progress(channel_id, event.get("ts"))
            return started
        extra_metadata = self._external_thread_task_metadata(session, channel_id, thread_ts)
        extra_metadata["request_message_ts"] = event.get("ts")
        extra_metadata = self._with_linked_thread_context(
            extra_metadata,
            request.prompt,
            current_thread=SlackThreadRef(channel_id, thread_ts),
        )
        sticky_excluded_agent_ids = self._anyone_context_agent_ids(request, active_agents)
        excluded_agent_ids = set(self._busy_agent_ids_for_assignment())
        excluded_agent_ids.update(sticky_excluded_agent_ids)
        result = assign_work_request(
            self.store,
            request,
            channel_id,
            requested_by_slack_user=event.get("user"),
            author_agent=author_agent,
            extra_metadata=extra_metadata,
            exclude_agent_ids=excluded_agent_ids,
        )
        if result is None:
            self._post_assignment_unavailable(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread_ts),
                request,
                requested_by_slack_user=event.get("user"),
                author_agent=author_agent,
                extra_metadata=extra_metadata,
                exclude_agent_ids=sticky_excluded_agent_ids,
            )
            return True
        posted = self.gateway.post_thread_reply(
            SlackThreadRef(channel_id, thread_ts),
            format_agent_assignment(
                result.agent,
                result.request.prompt,
                event.get("user"),
                dangerous_mode=_task_dangerous_mode(result.task),
            ),
            persona=result.agent,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(result.task.task_id, thread_ts, posted.ts)
        task = self.store.get_agent_task(result.task.task_id) or result.task
        self._start_runtime_task(task, result.agent, SlackThreadRef(channel_id, thread_ts))
        self._mark_message_in_progress(channel_id, event.get("ts"))
        return True

    def _handle_thread_work_request(
        self,
        parent_task: AgentTask,
        event: dict,
        text: str,
        parent_agent,
    ) -> bool:
        active_agents = self.store.list_team_agents()
        channel_id = event["channel"]
        thread_ts = event["thread_ts"]
        current_thread = SlackThreadRef(channel_id, thread_ts)
        target_thread = (
            self._linked_thread_route_target(
                text,
                SlackThreadRef(channel_id, thread_ts, event.get("ts")),
            )
            or current_thread
        )
        multi_requests = _multi_specific_work_requests(text, active_agents)
        if multi_requests:
            extra_metadata = self._thread_task_metadata(parent_task, channel_id, thread_ts)
            extra_metadata["request_message_ts"] = event.get("ts")
            extra_metadata = self._with_linked_thread_context(
                extra_metadata,
                text,
                current_thread=current_thread,
            )
            return self._start_specific_requests_in_thread(
                multi_requests,
                target_thread,
                requested_by_slack_user=event.get("user"),
                author_agent=parent_agent,
                extra_metadata=extra_metadata,
                context_task=parent_task,
            )
        request = _parse_work_request_for_agents(text, active_agents, split_newlines=True)
        if request is None:
            return False
        extra_metadata = self._thread_task_metadata(parent_task, channel_id, thread_ts)
        extra_metadata["request_message_ts"] = event.get("ts")
        extra_metadata = self._with_linked_thread_context(
            extra_metadata,
            request.prompt,
            current_thread=current_thread,
        )
        same_thread_agent = self._same_thread_requested_agent(
            request,
            target_thread.channel_id,
            target_thread.thread_ts,
        )
        if same_thread_agent is not None:
            started = self._start_same_thread_agent_followup(
                request,
                same_thread_agent,
                target_thread,
                requested_by_slack_user=event.get("user"),
                author_agent=parent_agent,
                context_task=parent_task,
                request_message_ts=event.get("ts"),
            )
            if started:
                self._mark_message_in_progress(channel_id, event.get("ts"))
            return started
        delegation = _thread_delegation_intent(request.prompt, parent_agent, active_agents)
        excluded_agent_ids: set[str] = set(self._busy_agent_ids_for_assignment())
        sticky_excluded_agent_ids: set[str] = set(
            self._anyone_context_agent_ids(request, active_agents, parent_agent)
        )
        excluded_agent_ids.update(sticky_excluded_agent_ids)
        if delegation:
            sticky_excluded_agent_ids.add(delegation.target_agent_id)
            excluded_agent_ids.add(delegation.target_agent_id)
            extra_metadata["delegate_to_agent_id"] = delegation.target_agent_id
            extra_metadata["delegate_prompt"] = delegation.prompt_template
            if delegation.visible_prompt_template:
                extra_metadata["delegate_visible_prompt"] = delegation.visible_prompt_template
        result = assign_work_request(
            self.store,
            request,
            target_thread.channel_id,
            requested_by_slack_user=event.get("user"),
            author_agent=parent_agent,
            extra_metadata=extra_metadata,
            exclude_agent_ids=excluded_agent_ids,
        )
        if result is None:
            self._post_assignment_unavailable(
                SlackReplyTarget(
                    channel_id=target_thread.channel_id,
                    thread_ts=target_thread.thread_ts,
                ),
                request,
                requested_by_slack_user=event.get("user"),
                author_agent=parent_agent,
                extra_metadata=extra_metadata,
                exclude_agent_ids=sticky_excluded_agent_ids,
            )
            return True
        posted = self.gateway.post_thread_reply(
            target_thread,
            format_agent_assignment(
                result.agent,
                result.request.prompt,
                event.get("user"),
                dangerous_mode=_task_dangerous_mode(result.task),
            ),
            persona=result.agent,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(
            result.task.task_id,
            target_thread.thread_ts,
            posted.ts,
        )
        task = self.store.get_agent_task(result.task.task_id) or result.task
        self._start_runtime_task(task, result.agent, target_thread)
        self._mark_message_in_progress(channel_id, event.get("ts"))
        return True

    def handle_runtime_agent_message(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        text: str,
        message_ts: str | None = None,
    ) -> bool:
        if not thread.thread_ts:
            return False
        self._remember_agent_authored_message(task, agent, thread, message_ts, text)
        task = self._record_task_pr_urls(task, agent, thread, text)
        if self._handle_agent_authored_specific_request(task, agent, thread, text, message_ts):
            return True
        active_agents = self.store.list_team_agents()
        final_handle_request = _agent_authored_final_handle_callback_request(
            text,
            active_agents,
            sender_agent=agent,
        )
        if final_handle_request is not None:
            handled = self._start_agent_authored_specific_request(
                final_handle_request,
                task,
                agent,
                thread,
                text,
                message_ts=message_ts,
            )
            if handled:
                return True
        request = _agent_authored_review_request(text, active_agents)
        if request is None:
            self._clear_answered_request_status_reactions(task, thread)
            return False
        metadata = self._thread_task_metadata(task, thread.channel_id, thread.thread_ts)
        if message_ts:
            metadata["request_message_ts"] = message_ts
        delegate_to_agent_id = task.metadata.get("delegate_to_agent_id")
        if not isinstance(delegate_to_agent_id, str):
            delegate_to_agent_id = agent.agent_id
        metadata["delegate_to_agent_id"] = delegate_to_agent_id
        metadata = self._with_linked_thread_context(
            metadata,
            request.prompt,
            current_thread=thread,
        )
        delegate_prompt = task.metadata.get("delegate_prompt")
        if not isinstance(delegate_prompt, str) or not delegate_prompt.strip():
            delegate_prompt = REVIEW_DELEGATE_PROMPT
        metadata["delegate_prompt"] = delegate_prompt
        delegate_visible_prompt = task.metadata.get("delegate_visible_prompt")
        if isinstance(delegate_visible_prompt, str) and delegate_visible_prompt.strip():
            metadata["delegate_visible_prompt"] = delegate_visible_prompt
        else:
            metadata["delegate_visible_prompt"] = REVIEW_DELEGATE_VISIBLE_PROMPT
        result = assign_work_request(
            self.store,
            request,
            thread.channel_id,
            requested_by_slack_user=task.requested_by_slack_user,
            author_agent=agent,
            extra_metadata=metadata,
            exclude_agent_ids={
                *self._anyone_context_agent_ids(request, active_agents, agent),
                *self._busy_agent_ids_for_assignment(),
            },
        )
        if result is None:
            self._post_assignment_unavailable(
                SlackReplyTarget(channel_id=thread.channel_id, thread_ts=thread.thread_ts),
                request,
                requested_by_slack_user=task.requested_by_slack_user,
                author_agent=agent,
                extra_metadata=metadata,
                exclude_agent_ids=self._anyone_context_agent_ids(request, active_agents, agent),
            )
            return True
        if message_ts:
            self._mark_message_acknowledged(thread.channel_id, message_ts)
        posted = self.gateway.post_thread_reply(
            thread,
            format_agent_handoff_assignment(result.agent, agent, result.request.prompt),
            persona=result.agent,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(result.task.task_id, thread.thread_ts, posted.ts)
        reviewer_task = self.store.get_agent_task(result.task.task_id) or result.task
        started = self._start_runtime_task(reviewer_task, result.agent, thread)
        if started and message_ts:
            self._mark_message_in_progress(thread.channel_id, message_ts)
        elif message_ts:
            self._clear_message_status_reactions(thread.channel_id, message_ts)
        return True

    def handle_mirrored_session_agent_message(
        self,
        session,
        agent,
        thread: SlackThreadRef,
        text: str,
        message_ts: str | None = None,
    ) -> bool:
        task = self.store.get_active_task_by_session(session.provider, session.session_id)
        if task is None:
            return False
        return self.handle_runtime_agent_message(task, agent, thread, text, message_ts)

    def _record_task_pr_urls(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        *texts_or_urls: str | None,
    ) -> AgentTask:
        current = self.store.get_agent_task(task.task_id) or task
        metadata = metadata_with_pr_urls(current.metadata, *texts_or_urls)
        if metadata == current.metadata:
            return current
        updated = replace(current, metadata=metadata, updated_at=utc_now())
        self.store.upsert_agent_task(updated)
        updated = self.store.get_agent_task(updated.task_id) or updated
        self._refresh_task_thread_header(updated, agent)
        if thread.channel_id:
            self._refresh_existing_roster(thread.channel_id)
        return updated

    def _handle_agent_authored_specific_request(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        text: str,
        message_ts: str | None = None,
    ) -> bool:
        active_agents = self.store.list_team_agents()
        multi_requests = _multi_specific_work_requests(text, active_agents, split_newlines=True)
        if multi_requests:
            handled = False
            for request in multi_requests:
                if self._start_agent_authored_specific_request(
                    request,
                    task,
                    agent,
                    thread,
                    text,
                    message_ts=message_ts,
                ):
                    handled = True
            return handled
        request = _parse_work_request_for_agents(text, active_agents, split_newlines=True)
        if request is None or request.assignment_mode != AssignmentMode.SPECIFIC:
            return False
        return self._start_agent_authored_specific_request(
            request,
            task,
            agent,
            thread,
            text,
            message_ts=message_ts,
        )

    def _start_agent_authored_specific_request(
        self,
        request: WorkRequest,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        text: str,
        *,
        message_ts: str | None = None,
    ) -> bool:
        target_agent = self.store.get_team_agent(request.requested_handle or "")
        if target_agent is None or target_agent.agent_id == agent.agent_id:
            return False
        if self._start_external_session_agent_followup(
            request,
            target_agent,
            thread,
            requested_by_slack_user=task.requested_by_slack_user,
            delivery_text=_agent_authored_external_callback_text(text, request),
            request_message_ts=message_ts,
        ):
            return True
        if task.metadata.get("delegate_to_agent_id") == target_agent.agent_id:
            return False
        if message_ts:
            self._mark_message_acknowledged(thread.channel_id, message_ts)
        same_thread_task = self._latest_task_for_agent_thread(
            target_agent.agent_id,
            thread.channel_id,
            thread.thread_ts,
        )
        if same_thread_task is not None:
            handled = self._continue_same_thread_agent_task(
                request,
                same_thread_task,
                target_agent,
                thread,
                requested_by_slack_user=task.requested_by_slack_user,
                request_message_ts=message_ts,
            )
            if handled and message_ts:
                self._mark_message_in_progress(thread.channel_id, message_ts)
            elif message_ts:
                self._clear_message_status_reactions(thread.channel_id, message_ts)
            return handled
        metadata = self._thread_task_metadata(task, thread.channel_id, thread.thread_ts)
        if message_ts:
            metadata["request_message_ts"] = message_ts
        metadata = self._with_linked_thread_context(
            metadata,
            request.prompt,
            current_thread=thread,
        )
        metadata["delegated_from_task_id"] = task.task_id
        metadata["delegated_from_agent_id"] = agent.agent_id
        result = assign_work_request(
            self.store,
            request,
            thread.channel_id,
            requested_by_slack_user=task.requested_by_slack_user,
            author_agent=agent,
            extra_metadata=metadata,
            exclude_agent_ids=self._busy_agent_ids_for_assignment(),
        )
        if result is None:
            self._post_assignment_unavailable(
                SlackReplyTarget(channel_id=thread.channel_id, thread_ts=thread.thread_ts),
                request,
                requested_by_slack_user=task.requested_by_slack_user,
                author_agent=agent,
                extra_metadata=metadata,
            )
            return True
        posted = self.gateway.post_thread_reply(
            thread,
            format_agent_handoff_assignment(result.agent, agent, result.request.prompt),
            persona=result.agent,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(result.task.task_id, thread.thread_ts, posted.ts)
        delegated_task = self.store.get_agent_task(result.task.task_id) or result.task
        started = self._start_runtime_task(delegated_task, result.agent, thread)
        if started and message_ts:
            self._mark_message_in_progress(thread.channel_id, message_ts)
        elif message_ts:
            self._clear_message_status_reactions(thread.channel_id, message_ts)
        return True

    def _start_specific_requests_in_thread(
        self,
        requests: list[WorkRequest],
        thread: SlackThreadRef,
        *,
        requested_by_slack_user: str | None,
        author_agent=None,
        extra_metadata: dict[str, object] | None = None,
        context_task: AgentTask | None = None,
    ) -> bool:
        handled = False
        started = False
        request_message_ts = thread.message_ts
        if not request_message_ts and extra_metadata:
            value = extra_metadata.get("request_message_ts")
            if isinstance(value, str) and value:
                request_message_ts = value
        for request in requests:
            target_agent = self.store.get_team_agent(request.requested_handle or "")
            if target_agent is None:
                continue
            if self._start_external_session_agent_followup(
                request,
                target_agent,
                thread,
                requested_by_slack_user=requested_by_slack_user,
                request_message_ts=request_message_ts,
            ):
                handled = True
                started = True
                continue
            same_thread_agent = self._same_thread_requested_agent(
                request,
                thread.channel_id,
                thread.thread_ts,
            )
            if same_thread_agent is not None:
                same_thread_started = self._start_same_thread_agent_followup(
                    request,
                    same_thread_agent,
                    thread,
                    requested_by_slack_user=requested_by_slack_user,
                    author_agent=author_agent,
                    context_task=context_task,
                    request_message_ts=request_message_ts,
                )
                started = started or same_thread_started
                handled = same_thread_started or handled
                continue
            result = assign_work_request(
                self.store,
                request,
                thread.channel_id,
                requested_by_slack_user=requested_by_slack_user,
                author_agent=author_agent,
                extra_metadata=dict(extra_metadata or {}),
                exclude_agent_ids=self._busy_agent_ids_for_assignment(),
            )
            if result is None:
                self._post_assignment_unavailable(
                    SlackReplyTarget(channel_id=thread.channel_id, thread_ts=thread.thread_ts),
                    request,
                    requested_by_slack_user=requested_by_slack_user,
                    author_agent=author_agent,
                    extra_metadata=extra_metadata,
                )
                handled = True
                continue
            blocks = _task_thread_blocks(result.task, result.agent)
            posted = self.gateway.post_thread_reply(
                thread,
                format_agent_assignment(
                    result.agent,
                    result.request.prompt,
                    requested_by_slack_user,
                    dangerous_mode=_task_dangerous_mode(result.task),
                ),
                persona=result.agent,
                blocks=blocks,
                icon_url=self._agent_icon_url(result.agent),
            )
            self.store.update_agent_task_thread(result.task.task_id, thread.thread_ts, posted.ts)
            task = self.store.get_agent_task(result.task.task_id) or result.task
            self._start_runtime_task(task, result.agent, thread)
            started = True
            handled = True
        if started and request_message_ts:
            self._mark_message_in_progress(thread.channel_id, request_message_ts)
        return handled

    def _same_thread_requested_agent(
        self,
        request: WorkRequest,
        channel_id: str,
        thread_ts: str,
    ):
        if request.assignment_mode != AssignmentMode.SPECIFIC or not request.requested_handle:
            return None
        agent = self.store.get_team_agent(request.requested_handle)
        if agent is None:
            return None
        if self._latest_task_for_agent_thread(agent.agent_id, channel_id, thread_ts) is None:
            return None
        return agent

    def _anyone_context_agent_ids(
        self,
        request: WorkRequest,
        active_agents,
        *context_agents,
    ) -> set[str]:
        if request.assignment_mode != AssignmentMode.ANYONE:
            return set()
        excluded = {agent.agent_id for agent in context_agents if agent is not None}
        prompt = canonicalize_agent_mentions(request.prompt, active_agents)
        mentioned_handles = set(parse_lightweight_handles(prompt))
        if mentioned_handles:
            excluded.update(
                agent.agent_id for agent in active_agents if agent.handle in mentioned_handles
            )
        return excluded

    def _start_same_thread_agent_followup(
        self,
        request: WorkRequest,
        agent,
        thread: SlackThreadRef,
        *,
        requested_by_slack_user: str | None,
        author_agent=None,
        context_task: AgentTask | None = None,
        sender_agent=None,
        request_message_ts: str | None = None,
    ) -> bool:
        previous_task = None
        if context_task is not None and context_task.agent_id == agent.agent_id:
            previous_task = context_task
        if previous_task is None:
            previous_task = self._latest_task_for_agent_thread(
                agent.agent_id,
                thread.channel_id,
                thread.thread_ts,
            )
        live_prompt = _live_thread_followup_prompt(request.prompt)
        if (
            previous_task
            and self.runtime
            and self.runtime.send_to_task(
                previous_task.task_id,
                live_prompt,
            )
        ):
            self._remember_request_message_for_task(previous_task, request_message_ts)
            return True
        if previous_task:
            return self._continue_same_thread_agent_task(
                request,
                previous_task,
                agent,
                thread,
                requested_by_slack_user=requested_by_slack_user,
                request_message_ts=request_message_ts,
                try_live_send=False,
            )
        extra_metadata: dict[str, object] = {}
        if request_message_ts:
            extra_metadata["request_message_ts"] = request_message_ts
        extra_metadata = self._with_linked_thread_context(
            extra_metadata,
            request.prompt,
            current_thread=thread,
        )
        result = assign_work_request(
            self.store,
            request,
            thread.channel_id,
            requested_by_slack_user=requested_by_slack_user,
            author_agent=author_agent,
            extra_metadata=extra_metadata,
            force_agent=agent,
        )
        if result is None:
            return False
        task = self._task_with_prior_session(result.task, previous_task)
        self.store.upsert_agent_task(task)
        if sender_agent is not None:
            text = format_agent_handoff_assignment(
                agent,
                sender_agent,
                result.request.prompt,
                dangerous_mode=_task_dangerous_mode(task),
            )
        else:
            text = format_agent_assignment(
                agent,
                result.request.prompt,
                requested_by_slack_user,
                dangerous_mode=_task_dangerous_mode(task),
            )
        posted = self.gateway.post_thread_reply(
            thread,
            text,
            persona=agent,
            icon_url=self._agent_icon_url(agent),
        )
        self.store.update_agent_task_thread(task.task_id, thread.thread_ts, posted.ts)
        task = self.store.get_agent_task(task.task_id) or task
        self._start_runtime_task(task, agent, thread)
        return True

    def _start_external_session_agent_followup(
        self,
        request: WorkRequest,
        agent,
        thread: SlackThreadRef,
        *,
        requested_by_slack_user: str | None,
        delivery_text: str | None = None,
        request_message_ts: str | None = None,
    ) -> bool:
        session = self._external_session_for_thread_agent(
            agent,
            thread.channel_id,
            thread.thread_ts,
        )
        if session is None:
            return False
        if self.session_bridge is None:
            self.gateway.post_thread_reply(
                thread,
                "I found the external session thread, but no session bridge is configured.",
            )
            return True
        sent = self.session_bridge.send_to_session(
            session,
            delivery_text or request.prompt,
            thread,
            slack_user=requested_by_slack_user,
        )
        if sent:
            self._mark_external_session_message_delivered(
                thread.channel_id,
                request_message_ts,
                session,
            )
        return sent

    def _external_session_for_thread_agent(
        self,
        agent,
        channel_id: str,
        thread_ts: str,
    ):
        session = self.store.get_session_for_slack_thread(self.team_id, channel_id, thread_ts)
        if session is None:
            return None
        assigned_agent_id = self.store.get_setting(
            f"{EXTERNAL_SESSION_AGENT_PREFIX}{session.provider.value}.{session.session_id}"
        )
        if assigned_agent_id != agent.agent_id:
            return None
        return session

    def _continue_same_thread_agent_task(
        self,
        request: WorkRequest,
        previous_task: AgentTask,
        agent,
        thread: SlackThreadRef,
        *,
        requested_by_slack_user: str | None,
        request_message_ts: str | None = None,
        try_live_send: bool = True,
    ) -> bool:
        previous_task = self._dismiss_idle_release_prompt(previous_task, thread)
        previous_task = self._record_task_pr_urls(previous_task, agent, thread, request.prompt)
        if (
            try_live_send
            and self.runtime
            and self.runtime.send_to_task(
                previous_task.task_id,
                _live_thread_followup_prompt(request.prompt),
            )
        ):
            self._remember_request_message_for_task(previous_task, request_message_ts)
            return True
        is_running = getattr(self.runtime, "is_task_running", None)
        if callable(is_running) and is_running(previous_task.task_id):
            if self._send_live_managed_task_followup(
                previous_task,
                request.prompt,
                thread,
                requested_by_slack_user,
            ):
                self._remember_request_message_for_task(previous_task, request_message_ts)
                return True
            LOGGER.debug(
                "interrupting same-thread continuation for task %s; worker is still running",
                previous_task.task_id,
            )
            if self._interrupt_running_task_for_followup(
                previous_task,
                request,
                thread,
                requested_by_slack_user=requested_by_slack_user,
                request_message_ts=request_message_ts,
            ):
                return True
            # Live delivery failed and we could not interrupt the running
            # worker to inject the new prompt. Rather than stash the message
            # in a metadata queue (which silently delays delivery and turned
            # the inbox into a graveyard), terminate the worker and resume
            # the session fresh so the user's instruction is the next thing
            # the agent sees.
            LOGGER.warning(
                "could not deliver same-thread follow-up live; terminating worker for fresh resume on task %s",
                previous_task.task_id,
            )
            stop_succeeded = True
            if self.runtime is not None:
                try:
                    stop_succeeded = bool(
                        self.runtime.stop_task(previous_task.task_id, status=None)
                    )
                except Exception:
                    stop_succeeded = False
                    LOGGER.debug(
                        "failed to stop worker before fresh same-thread resume",
                        exc_info=True,
                    )
            if not stop_succeeded:
                # The worker did not exit even after the escalation in
                # stop_task; the runtime's slot for this task_id is still
                # occupied, so start_task below would silently refuse the
                # fresh resume and the user's message would disappear into
                # the task row with no worker to pick it up. Tell the user
                # so they can retry instead of waiting forever.
                LOGGER.warning(
                    "fresh same-thread resume blocked for task %s; surfacing delivery failure to Slack",
                    previous_task.task_id,
                )
                self._post_followup_delivery_failed_notice(thread, agent)
                return False
        metadata = self._thread_task_metadata(previous_task, thread.channel_id, thread.thread_ts)
        metadata[ASSIGNMENT_PROMPT_METADATA_KEY] = _task_assignment_prompt(previous_task)
        metadata = metadata_with_pr_urls(metadata, request.prompt)
        for key in ("request_message_ts", "request_message_ts_history"):
            if key in previous_task.metadata:
                metadata[key] = previous_task.metadata[key]
        if request_message_ts:
            metadata = _metadata_with_request_message_ts(metadata, request_message_ts)
        metadata = self._with_linked_thread_context(
            metadata,
            request.prompt,
            current_thread=thread,
        )
        if request.dangerous_mode or _task_dangerous_mode(previous_task):
            metadata[DANGEROUS_MODE_METADATA_KEY] = True
        task = replace(
            previous_task,
            prompt=request.prompt,
            status=AgentTaskStatus.ACTIVE,
            requested_by_slack_user=requested_by_slack_user
            or previous_task.requested_by_slack_user,
            updated_at=utc_now(),
            metadata=metadata,
        )
        self.store.upsert_agent_task(task)
        self._restore_task_action_buttons_if_active(task)
        return self._start_runtime_task(task, agent, thread)

    def _post_followup_delivery_failed_notice(
        self,
        thread: SlackThreadRef,
        agent,
    ) -> None:
        handle = getattr(agent, "handle", None) or "the agent"
        text = (
            f"I couldn't deliver your message to @{handle} — the previous run "
            "is stuck and a fresh resume could not be started. Please send the "
            "message again in a moment."
        )
        try:
            self.gateway.post_thread_reply(thread, text)
        except Exception:
            LOGGER.debug(
                "failed to post same-thread follow-up delivery failure notice",
                exc_info=True,
            )

    def _send_live_managed_task_followup(
        self,
        task: AgentTask,
        text: str,
        thread: SlackThreadRef,
        slack_user: str | None,
    ) -> bool:
        if self.session_bridge is None or not task.session_provider or not task.session_id:
            return False
        session = self.store.get_session(task.session_provider, task.session_id)
        if session is None:
            return False
        send_live = getattr(self.session_bridge, "send_live_to_session", None)
        if not callable(send_live):
            return False
        try:
            return bool(send_live(session, text, thread, slack_user=slack_user))
        except Exception:
            LOGGER.debug("failed to send same-thread follow-up to live session", exc_info=True)
            return False

    def _active_thread_task_for_agent(
        self,
        task: AgentTask,
        channel_id: str,
        thread_ts: str,
    ) -> AgentTask | None:
        managed_task = self.store.get_managed_thread_task(channel_id, thread_ts, task.agent_id)
        if managed_task is not None:
            return managed_task
        latest_task = self._latest_task_for_agent_thread(task.agent_id, channel_id, thread_ts)
        if latest_task and latest_task.status in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}:
            return latest_task
        if task.status in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}:
            return task
        return None

    def _latest_task_for_agent_thread(
        self,
        agent_id: str,
        channel_id: str,
        thread_ts: str,
    ) -> AgentTask | None:
        matches = [
            task
            for task in self.store.list_agent_tasks(include_done=True)
            if task.agent_id == agent_id
            and task.channel_id == channel_id
            and task.thread_ts == thread_ts
        ]
        if not matches:
            return None
        return max(matches, key=lambda task: task.created_at)

    def _task_with_prior_session(
        self,
        task: AgentTask,
        previous_task: AgentTask | None,
    ) -> AgentTask:
        if previous_task is None or not previous_task.session_id:
            return task
        return replace(
            task,
            session_provider=previous_task.session_provider,
            session_id=previous_task.session_id,
        )

    def _remember_request_message_for_task(
        self,
        task: AgentTask,
        message_ts: str | None,
    ) -> AgentTask:
        if not message_ts:
            return task
        metadata = _metadata_with_request_message_ts(task.metadata, message_ts)
        if metadata == task.metadata:
            return task
        updated = replace(task, metadata=metadata, updated_at=utc_now())
        try:
            self.store.upsert_agent_task(updated)
        except Exception:
            LOGGER.debug("failed to remember task request message", exc_info=True)
            return task
        return self.store.get_agent_task(task.task_id) or updated

    def _interrupt_running_task_for_followup(
        self,
        task: AgentTask,
        request: WorkRequest,
        thread: SlackThreadRef,
        *,
        requested_by_slack_user: str | None,
        request_message_ts: str | None,
    ) -> bool:
        if self.runtime is None:
            return False
        interrupt = getattr(self.runtime, "interrupt_task", None)
        send = getattr(self.runtime, "send_to_interrupted_task", None)
        if not callable(interrupt) or not callable(send):
            return False
        if not request.prompt.strip():
            return False
        prompt = _live_thread_followup_prompt(request.prompt)
        latest = self.store.get_agent_task(task.task_id) or task
        try:
            interrupted = bool(interrupt(task.task_id))
        except Exception:
            LOGGER.debug(
                "failed to interrupt running task for same-thread follow-up", exc_info=True
            )
            return False
        if not interrupted:
            return False
        try:
            sent = bool(send(latest.task_id, prompt))
        except Exception:
            LOGGER.debug(
                "failed to send follow-up to interrupted running task",
                exc_info=True,
            )
            return False
        if not sent:
            return False
        metadata = dict(latest.metadata)
        active_message_ts_values = _active_thread_followup_message_ts_values(metadata)
        if request_message_ts and request_message_ts not in active_message_ts_values:
            active_message_ts_values.append(request_message_ts)
        if active_message_ts_values:
            metadata[ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_VALUES_METADATA_KEY] = (
                active_message_ts_values
            )
            metadata.pop(ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_METADATA_KEY, None)
        metadata[ASSIGNMENT_PROMPT_METADATA_KEY] = _task_assignment_prompt(latest)
        metadata = metadata_with_pr_urls(metadata, prompt)
        metadata = self._with_linked_thread_context(
            metadata,
            prompt,
            current_thread=thread,
        )
        updated = replace(
            latest,
            prompt=prompt,
            requested_by_slack_user=requested_by_slack_user or latest.requested_by_slack_user,
            updated_at=utc_now(),
            metadata=metadata,
        )
        self.store.upsert_agent_task(updated)
        if request_message_ts:
            self._mark_message_in_progress(thread.channel_id, request_message_ts)
        self.refresh_or_post_roster(thread.channel_id)
        return True

    def _complete_active_thread_followup(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
    ) -> AgentTask:
        latest = self.store.get_agent_task(task.task_id) or task
        message_ts_values = _active_thread_followup_message_ts_values(latest.metadata)
        if not message_ts_values:
            return latest
        metadata = dict(latest.metadata)
        metadata.pop(ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_METADATA_KEY, None)
        metadata.pop(ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_VALUES_METADATA_KEY, None)
        updated = replace(latest, metadata=metadata, updated_at=utc_now())
        self.store.upsert_agent_task(updated)
        # Just clear the pending status reactions on these messages instead of
        # adding :white_check_mark:. A finished run does not mean the user's
        # specific follow-up was actually addressed, and the prior behavior of
        # stamping a checkmark on every interrupted/queued follow-up — even
        # when the agent never responded to it — made the indicator dishonest.
        # :white_check_mark: stays reserved for explicit completion: "stop",
        # task release, or `_reconcile_pending_thread_reactions` warning.
        for message_ts in message_ts_values:
            self._clear_pending_message_status_reactions(thread.channel_id, message_ts)
        return self.store.get_agent_task(task.task_id) or updated

    def handle_runtime_task_done(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
    ) -> None:
        task_is_child = _is_subtask(task) or _is_external_thread_helper_task(task)
        if task_is_child:
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.DONE)
            task = self.store.get_agent_task(task.task_id) or task
            self._clear_task_request_status_reactions(task, thread)
            self._remove_task_action_buttons_if_resolved(task)
        if task.session_provider is not None and task.session_id:
            clear_managed_session(self.store, task.session_provider, task.session_id)
        try:
            self.evaluate_pending_deferred_work()
            self._fire_due_deferred_work_now(limit=MAX_PM_SUBTASKS)
        except Exception:
            LOGGER.debug(
                "failed to evaluate deferred work after runtime task done",
                exc_info=True,
            )
        task = self._complete_active_thread_followup(task, thread)
        self._clear_completed_run_pending_reactions(task, thread)
        if task_is_child:
            task = self.store.get_agent_task(task.task_id) or task
        else:
            task = self._hold_completed_runtime_task_open(task, thread)
        self._refresh_pm_status_for_task(task)
        self.refresh_or_post_roster(thread.channel_id)
        delegate_to_agent_id = task.metadata.get("delegate_to_agent_id")
        delegate_prompt = task.metadata.get("delegate_prompt")
        if not isinstance(delegate_to_agent_id, str) or not isinstance(delegate_prompt, str):
            if not task_is_child:
                self._post_idle_release_prompt(task, agent, thread)
            return
        if not delegate_prompt.strip() or not thread.thread_ts:
            if not task_is_child:
                self._post_idle_release_prompt(task, agent, thread)
            return
        target_agent = self.store.get_team_agent(delegate_to_agent_id)
        if target_agent is None:
            if not task_is_child:
                self._post_idle_release_prompt(task, agent, thread)
            return
        if target_agent.agent_id == agent.agent_id:
            if not task_is_child:
                self._post_idle_release_prompt(task, agent, thread)
            return
        if target_agent.agent_id in self._busy_agent_ids_for_assignment():
            if not task_is_child:
                self._post_idle_release_prompt(task, agent, thread)
            return
        visible_prompt = task.metadata.get("delegate_visible_prompt")
        if not isinstance(visible_prompt, str) or not visible_prompt.strip():
            visible_prompt = delegate_prompt
        delegate_prompt = _render_delegate_template(delegate_prompt, agent, target_agent)
        visible_prompt = _render_delegate_template(visible_prompt, agent, target_agent)
        parent_task = task
        parent_task_id = task.metadata.get("parent_task_id")
        if isinstance(parent_task_id, str):
            parent_task = self.store.get_agent_task(parent_task_id) or task
        metadata = self._thread_task_metadata(parent_task, thread.channel_id, thread.thread_ts)
        metadata["delegated_from_task_id"] = task.task_id
        metadata["delegated_from_agent_id"] = agent.agent_id
        request = WorkRequest(
            prompt=delegate_prompt,
            assignment_mode=AssignmentMode.SPECIFIC,
            requested_handle=target_agent.handle,
        )
        target_previous_task = self._latest_task_for_agent_thread(
            target_agent.agent_id,
            thread.channel_id,
            thread.thread_ts,
        )
        handoff_request_posted = False
        if target_previous_task is not None:
            self.gateway.post_thread_reply(
                thread,
                format_agent_handoff_request(agent, target_agent, visible_prompt),
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            handoff_request_posted = True
            if self._continue_same_thread_agent_task(
                request,
                target_previous_task,
                target_agent,
                thread,
                requested_by_slack_user=task.requested_by_slack_user,
            ):
                return
        result = assign_work_request(
            self.store,
            request,
            thread.channel_id,
            requested_by_slack_user=task.requested_by_slack_user,
            extra_metadata=metadata,
            force_agent=target_agent,
        )
        if result is None:
            return
        delegated_task = self._task_with_prior_session(result.task, target_previous_task)
        self.store.upsert_agent_task(delegated_task)
        if not handoff_request_posted:
            self.gateway.post_thread_reply(
                thread,
                format_agent_handoff_request(agent, target_agent, visible_prompt),
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
        posted = self.gateway.post_thread_reply(
            thread,
            format_agent_handoff_assignment(target_agent, agent, result.request.prompt),
            persona=target_agent,
            icon_url=self._agent_icon_url(target_agent),
        )
        self.store.update_agent_task_thread(delegated_task.task_id, thread.thread_ts, posted.ts)
        delegated_task = self.store.get_agent_task(delegated_task.task_id) or delegated_task
        self._start_runtime_task(delegated_task, target_agent, thread)

    def _hold_completed_runtime_task_open(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
    ) -> AgentTask:
        current = self.store.get_agent_task(task.task_id) or task
        if current.status in {AgentTaskStatus.DONE, AgentTaskStatus.CANCELLED}:
            return current
        metadata = dict(current.metadata)
        for key in (
            MANAGED_RUN_STARTED_METADATA_KEY,
            MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY,
            MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY,
            MANAGED_RUN_ORIGINAL_PROMPT_METADATA_KEY,
        ):
            metadata.pop(key, None)
        held = current
        if current.status != AgentTaskStatus.ACTIVE or metadata != current.metadata:
            held = replace(
                current,
                status=AgentTaskStatus.ACTIVE,
                metadata=metadata,
                updated_at=utc_now(),
            )
            self.store.upsert_agent_task(held)
        if thread.thread_ts:
            self.store.upsert_managed_thread_task(held, thread)
        self._restore_task_action_buttons_if_active(held)
        return self.store.get_agent_task(current.task_id) or held

    def _post_idle_release_prompt(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
    ) -> None:
        if not thread.thread_ts:
            return
        current = self.store.get_agent_task(task.task_id) or task
        if current.status in {AgentTaskStatus.DONE, AgentTaskStatus.CANCELLED}:
            return
        existing_ts = current.metadata.get(IDLE_RELEASE_PROMPT_MESSAGE_TS_METADATA_KEY)
        if isinstance(existing_ts, str) and existing_ts:
            return
        try:
            posted = self.gateway.post_thread_reply(
                thread,
                IDLE_RELEASE_PROMPT_TEXT,
                blocks=build_idle_release_prompt_blocks(current),
            )
        except Exception:
            LOGGER.debug("failed to post idle release prompt", exc_info=True)
            return
        message_ts = getattr(posted, "ts", None)
        if not message_ts:
            return
        latest = self.store.get_agent_task(current.task_id) or current
        if latest.status in {AgentTaskStatus.DONE, AgentTaskStatus.CANCELLED}:
            return
        metadata = dict(latest.metadata)
        metadata[IDLE_RELEASE_PROMPT_MESSAGE_TS_METADATA_KEY] = message_ts
        updated = replace(latest, metadata=metadata, updated_at=utc_now())
        self.store.upsert_agent_task(updated)

    def _dismiss_idle_release_prompt(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
    ) -> AgentTask:
        current = self.store.get_agent_task(task.task_id) or task
        message_ts = current.metadata.get(IDLE_RELEASE_PROMPT_MESSAGE_TS_METADATA_KEY)
        if not isinstance(message_ts, str) or not message_ts:
            return current
        channel_id = thread.channel_id or current.channel_id
        if channel_id:
            try:
                self.gateway.update_message(
                    channel_id,
                    message_ts,
                    " ",
                    blocks=build_idle_release_dismissed_blocks(current),
                )
            except Exception:
                LOGGER.debug("failed to dismiss idle release prompt", exc_info=True)
        metadata = dict(current.metadata)
        metadata.pop(IDLE_RELEASE_PROMPT_MESSAGE_TS_METADATA_KEY, None)
        if metadata == current.metadata:
            return current
        updated = replace(current, metadata=metadata, updated_at=utc_now())
        self.store.upsert_agent_task(updated)
        return self.store.get_agent_task(current.task_id) or updated

    def handle_runtime_agent_control(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        signal: str,
    ) -> bool:
        roster_summary = parse_agent_roster_status_signal(signal)
        if roster_summary is not None:
            return self._update_task_roster_summary(task, agent, thread, roster_summary)
        reaction_name = parse_agent_reaction_signal(signal)
        if reaction_name is not None:
            return self._react_to_latest_thread_message(task, agent, thread, reaction_name)
        if signal == AGENT_THREAD_DONE_SIGNAL:
            return self._complete_task_thread(thread.channel_id, thread.thread_ts)
        if is_agent_timer_signal(signal):
            return self._schedule_agent_timer(task, agent, thread, signal)
        if is_agent_schedule_signal(signal):
            return self._schedule_user_work_from_agent(task, agent, thread, signal)
        if is_agent_deferred_signal(signal):
            return self._create_deferred_work_from_agent(task, agent, thread, signal)
        if is_agent_pm_plan_signal(signal):
            return self._create_pm_initiative_from_agent(task, agent, thread, signal)
        return False

    def _update_task_roster_summary(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        summary: str,
    ) -> bool:
        summary = _roster_summary_line(summary)
        if not summary:
            return False
        current = self.store.get_agent_task(task.task_id) or task
        metadata = metadata_with_pr_urls(current.metadata, summary)
        if metadata.get(ROSTER_SUMMARY_METADATA_KEY) != summary:
            metadata[ROSTER_SUMMARY_METADATA_KEY] = summary
        if metadata != current.metadata:
            current = replace(current, metadata=metadata, updated_at=utc_now())
            self.store.upsert_agent_task(current)
        self._refresh_task_thread_header(current, agent)
        if thread.channel_id:
            self.refresh_or_post_roster(thread.channel_id)
        return True

    def _refresh_task_thread_header(self, task: AgentTask, agent) -> None:
        if not task.channel_id or not task.parent_message_ts:
            return
        try:
            self.gateway.update_message(
                task.channel_id,
                task.parent_message_ts,
                format_agent_assignment(
                    agent,
                    _task_original_prompt(task),
                    task.requested_by_slack_user,
                    dangerous_mode=_task_dangerous_mode(task),
                    latest_summary=_task_roster_summary(task),
                ),
                blocks=_task_thread_blocks(task, agent),
            )
        except Exception:
            LOGGER.debug("failed to refresh task thread header", exc_info=True)

    def _react_to_latest_thread_message(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        reaction_name: str,
    ) -> bool:
        current = self.store.get_agent_task(task.task_id) or task
        user_message_ts = _latest_user_message_ts_for_reaction(current.metadata)
        message_ts = self._latest_reactable_message_ts(
            task,
            agent,
            thread,
            fallback_ts=user_message_ts,
        )
        if not message_ts:
            return False
        try:
            self.gateway.add_reaction(thread.channel_id, message_ts, reaction_name)
        except Exception:
            LOGGER.debug("failed to add agent-requested Slack reaction", exc_info=True)
            return False
        if reaction_name in TASK_STATUS_REACTIONS and message_ts == user_message_ts:
            self.store.set_setting(
                _message_status_reaction_setting_key(thread.channel_id, message_ts),
                reaction_name,
            )
        return True

    def _latest_reactable_message_ts(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        *,
        fallback_ts: str | None,
    ) -> str | None:
        if not thread.channel_id or not thread.thread_ts:
            return fallback_ts
        try:
            messages = self.gateway.thread_messages(
                thread.channel_id,
                thread.thread_ts,
                limit=20,
            )
        except Exception:
            LOGGER.debug("failed to fetch thread messages for reaction", exc_info=True)
            messages = []
        skip_ts = {thread.thread_ts}
        if task.parent_message_ts:
            skip_ts.add(task.parent_message_ts)
        for message in reversed(messages):
            ts = message.get("ts")
            if not isinstance(ts, str) or not ts or ts in skip_ts:
                continue
            record = self._agent_authored_message_record(thread.channel_id, ts)
            if record and _record_string(record, "agent_id") == agent.agent_id:
                continue
            return ts
        return fallback_ts

    def _schedule_agent_timer(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        signal: str,
    ) -> bool:
        parsed = parse_agent_timer_signal(signal)
        if parsed.request is None:
            message = parsed.error or "invalid timer control signal"
            self.gateway.post_thread_reply(
                thread,
                f"I could not schedule that timer: {message}.",
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            return True
        self.store.create_scheduled_timer(
            task,
            thread,
            prompt=parsed.request.prompt,
            due_at=parsed.request.due_at,
        )
        return True

    def _schedule_user_work_from_agent(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        signal: str,
    ) -> bool:
        active_agents = self.store.list_team_agents()
        parsed = parse_agent_schedule_signal(
            signal,
            known_handles=[item.handle for item in active_agents],
            now=utc_now(),
        )
        if parsed.schedule is None:
            return self._retry_schedule_resolution(
                task, agent, thread, parsed.error or "invalid schedule"
            )
        scheduled = self.store.create_scheduled_work_request(
            thread,
            parsed.schedule.request,
            schedule_kind=parsed.schedule.schedule_kind,
            next_run_at=parsed.schedule.next_run_at,
            recurrence=parsed.schedule.recurrence,
            timezone=parsed.schedule.timezone,
            requested_by_slack_user=task.requested_by_slack_user,
        )
        self.gateway.post_thread_reply(
            thread,
            _format_scheduled_work_ack(
                scheduled,
                parsed.schedule.description,
                parsed.schedule.request,
            ),
            persona=agent,
            icon_url=self._agent_icon_url(agent),
        )
        self.store.update_agent_task_status(task.task_id, AgentTaskStatus.DONE)
        resolved_task = self.store.get_agent_task(task.task_id) or task
        self._remove_task_action_buttons_if_resolved(resolved_task)
        for message_ts in _task_request_message_ts_values(resolved_task):
            self._mark_message_complete(thread.channel_id, message_ts)
        self.refresh_or_post_roster(thread.channel_id)
        return True

    def _retry_schedule_resolution(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        error: str,
    ) -> bool:
        if not task.metadata.get(SCHEDULE_RESOLUTION_METADATA_KEY):
            self.gateway.post_thread_reply(
                thread,
                f"I could not create that schedule: {error}.",
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            return False
        attempts = int(task.metadata.get(SCHEDULE_RESOLUTION_ATTEMPTS_METADATA_KEY) or 0)
        if attempts >= MAX_SCHEDULE_RESOLUTION_ATTEMPTS or not self.runtime:
            self.gateway.post_thread_reply(
                thread,
                f"I could not create that schedule after validation retries: {error}.",
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
            self.refresh_or_post_roster(thread.channel_id)
            return True
        original_text = task.metadata.get(SCHEDULE_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY)
        if not isinstance(original_text, str) or not original_text.strip():
            original_text = task.prompt
        active_agents = self.store.list_team_agents()
        retry_prompt = build_schedule_resolution_prompt(
            original_text,
            [item.handle for item in active_agents],
            now=utc_now(),
            validation_error=error,
        )
        metadata = dict(task.metadata)
        metadata[SCHEDULE_RESOLUTION_ATTEMPTS_METADATA_KEY] = attempts + 1
        retry_task = replace(
            task,
            prompt=retry_prompt,
            status=AgentTaskStatus.ACTIVE,
            updated_at=utc_now(),
            metadata=metadata,
        )
        self.store.upsert_agent_task(retry_task)
        started = self.runtime.start_task(retry_task, agent, thread)
        if not started:
            self.gateway.post_thread_reply(
                thread,
                "I could not restart schedule validation, so I cancelled this schedule request.",
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
            self.refresh_or_post_roster(thread.channel_id)
        return True

    def _create_deferred_work_from_agent(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        signal: str,
    ) -> bool:
        active_agents = self.store.list_team_agents()
        occupied_task_ids = dict(self._occupied_handle_task_ids())
        parsed = parse_agent_deferred_signal(
            signal,
            known_handles=[item.handle for item in active_agents],
            occupied_task_ids=occupied_task_ids,
            now=utc_now(),
        )
        if parsed.deferred is None:
            return self._retry_deferred_resolution(
                task, agent, thread, parsed.error or "invalid deferred request"
            )
        resolved_deps, error = self._resolve_deferred_dependencies(parsed.deferred.depends_on)
        if error is not None:
            return self._retry_deferred_resolution(task, agent, thread, error)
        deferred = self.store.create_deferred_work_request(
            thread,
            parsed.deferred.request,
            depends_on=tuple(resolved_deps),
            after_dep_delay_seconds=parsed.deferred.after_dep_delay_seconds,
            run_at=parsed.deferred.run_at,
            description=parsed.deferred.description,
            requested_by_slack_user=task.requested_by_slack_user,
        )
        self.gateway.post_thread_reply(
            thread,
            _format_deferred_work_ack(
                deferred,
                parsed.deferred.description,
                parsed.deferred.request,
                dependency_labeler=self._agent_busy_dependency_label,
            ),
            persona=agent,
            icon_url=self._agent_icon_url(agent),
        )
        # Best-effort: evaluate dependencies immediately so already-finished deps fire promptly.
        self.evaluate_pending_deferred_work(deferred.deferred_id)
        self.store.update_agent_task_status(task.task_id, AgentTaskStatus.DONE)
        resolved_task = self.store.get_agent_task(task.task_id) or task
        self._remove_task_action_buttons_if_resolved(resolved_task)
        for message_ts in _task_request_message_ts_values(resolved_task):
            self._mark_message_complete(thread.channel_id, message_ts)
        self.refresh_or_post_roster(thread.channel_id)
        return True

    def _retry_deferred_resolution(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        error: str,
    ) -> bool:
        if not task.metadata.get(DEFERRED_RESOLUTION_METADATA_KEY):
            self.gateway.post_thread_reply(
                thread,
                f"I could not create that deferred request: {error}.",
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            return False
        attempts = int(task.metadata.get(DEFERRED_RESOLUTION_ATTEMPTS_METADATA_KEY) or 0)
        if attempts >= MAX_DEFERRED_RESOLUTION_ATTEMPTS or not self.runtime:
            self.gateway.post_thread_reply(
                thread,
                (
                    "I could not resolve that deferred request after "
                    f"{MAX_DEFERRED_RESOLUTION_ATTEMPTS} validation retries: {error}. "
                    "Cancelling the request — please double check and try again."
                ),
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
            self.refresh_or_post_roster(thread.channel_id)
            return True
        original_text = task.metadata.get(DEFERRED_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY)
        if not isinstance(original_text, str) or not original_text.strip():
            original_text = task.prompt
        active_agents = self.store.list_team_agents()
        occupied_meta = task.metadata.get(DEFERRED_RESOLUTION_OCCUPIED_HANDLES_METADATA_KEY) or []
        occupied: list[dict[str, str]] = []
        if isinstance(occupied_meta, list):
            for item in occupied_meta:
                if not isinstance(item, dict):
                    continue
                handle = item.get("handle")
                task_id_value = item.get("task_id")
                if isinstance(handle, str) and isinstance(task_id_value, str):
                    occupied.append({"handle": handle, "task_id": task_id_value})
        retry_prompt = build_deferred_resolution_prompt(
            original_text,
            [item.handle for item in active_agents],
            occupied=occupied,
            now=utc_now(),
            validation_error=error,
        )
        metadata = dict(task.metadata)
        metadata[DEFERRED_RESOLUTION_ATTEMPTS_METADATA_KEY] = attempts + 1
        retry_task = replace(
            task,
            prompt=retry_prompt,
            status=AgentTaskStatus.ACTIVE,
            updated_at=utc_now(),
            metadata=metadata,
        )
        self.store.upsert_agent_task(retry_task)
        started = self.runtime.start_task(retry_task, agent, thread)
        if not started:
            self.gateway.post_thread_reply(
                thread,
                ("I could not restart deferred validation, so I cancelled this deferred request."),
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
            self.refresh_or_post_roster(thread.channel_id)
        return True

    def _resolve_deferred_dependencies(
        self,
        deps: tuple[WorkDependency, ...],
    ) -> tuple[list[WorkDependency], str | None]:
        active_agents = {agent.handle for agent in self.store.list_team_agents()}
        resolved: list[WorkDependency] = []
        for dep in deps:
            if dep.kind == WorkDependencyKind.THREAD:
                channel_id = dep.channel_id
                thread_ts = dep.thread_ts
                if (not channel_id or not thread_ts) and dep.permalink:
                    parsed_ref = parse_thread_ref(dep.permalink)
                    if parsed_ref is not None:
                        channel_id = channel_id or parsed_ref.channel_id
                        thread_ts = thread_ts or parsed_ref.thread_ts
                if not channel_id or not thread_ts:
                    return (
                        [],
                        f"could not parse thread dependency permalink: {dep.permalink}",
                    )
                thread_tasks = self.store.list_thread_agent_tasks(channel_id, thread_ts)
                if not thread_tasks:
                    return (
                        [],
                        (
                            f"thread dependency does not match a tracked Slackgentic task: "
                            f"{dep.permalink or f'{channel_id}/{thread_ts}'}"
                        ),
                    )
                resolved.append(
                    replace(
                        dep,
                        channel_id=channel_id,
                        thread_ts=thread_ts,
                        task_id=dep.task_id or thread_tasks[0].task_id,
                    )
                )
                continue
            if dep.kind == WorkDependencyKind.AGENT_BUSY:
                if not dep.handle or dep.handle not in active_agents:
                    return (
                        [],
                        f"agent_busy handle is not active: {dep.handle}",
                    )
                if not dep.task_id:
                    return (
                        [],
                        f"agent_busy dependency for @{dep.handle} is missing task_id",
                    )
                if parse_external_session_dependency_id(dep.task_id) is not None:
                    resolved.append(dep)
                    continue
                deferred_id = parse_deferred_work_dependency_id(dep.task_id)
                if deferred_id is not None:
                    deferred = self.store.get_deferred_work(deferred_id)
                    if deferred is None:
                        return (
                            [],
                            f"deferred_work dependency {deferred_id} for @{dep.handle} not found",
                        )
                    if (
                        deferred.assignment_mode != AssignmentMode.SPECIFIC
                        or deferred.requested_handle != dep.handle
                    ):
                        return (
                            [],
                            (
                                f"deferred_work dependency {deferred_id} is not assigned "
                                f"to @{dep.handle}"
                            ),
                        )
                    resolved.append(dep)
                    continue
                schedule_id = parse_scheduled_work_dependency_id(dep.task_id)
                if schedule_id is not None:
                    scheduled = self.store.get_scheduled_work(schedule_id)
                    if scheduled is None:
                        return (
                            [],
                            f"scheduled_work dependency {schedule_id} for @{dep.handle} not found",
                        )
                    if (
                        scheduled.assignment_mode != AssignmentMode.SPECIFIC
                        or scheduled.requested_handle != dep.handle
                    ):
                        return (
                            [],
                            (
                                f"scheduled_work dependency {schedule_id} is not assigned "
                                f"to @{dep.handle}"
                            ),
                        )
                    resolved.append(dep)
                    continue
                task = self.store.get_agent_task(dep.task_id)
                if task is None:
                    return (
                        [],
                        (
                            f"agent_busy task_id {dep.task_id} for @{dep.handle} not "
                            "found; ensure the agent is currently occupied"
                        ),
                    )
                resolved.append(dep)
                continue
            return ([], f"unsupported dependency kind: {dep.kind}")
        return (resolved, None)

    def _create_pm_initiative_from_agent(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        signal: str,
    ) -> bool:
        initiative_id = task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY)
        if not isinstance(initiative_id, str) or not initiative_id:
            self.gateway.post_thread_reply(
                thread,
                "I received a PM plan but could not match it to an initiative. Discarding.",
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            return True
        initiative = self.store.get_pm_initiative(initiative_id)
        if initiative is None:
            self.gateway.post_thread_reply(
                thread,
                f"I received a PM plan for unknown initiative `{initiative_id}`. Discarding.",
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            return True
        if initiative.status != PmInitiativeStatus.PLANNING:
            self.gateway.post_thread_reply(
                thread,
                (
                    f"PM initiative `{initiative_id}` is already {initiative.status.value}; "
                    "ignoring late PM plan."
                ),
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            return True
        available_workers = self._pm_available_worker_agents(
            ignore_task_id=task.task_id,
            ignore_pm_initiative_id=initiative.initiative_id,
        )
        extension_known_ids = self._task_extension_known_ids(task)
        parsed = parse_agent_pm_plan_signal(
            signal,
            known_handles=_pm_worker_handles(available_workers),
            allowed_external_dep_ids=extension_known_ids,
            handle_models=_pm_worker_model_map(available_workers),
        )
        if parsed.plan is None:
            return self._retry_pm_resolution(task, agent, thread, parsed.error or "invalid PM plan")
        plan = expand_codesign_plan(parsed.plan)
        shortfall = self._pm_plan_capacity_shortfall(
            plan,
            ignore_task_id=task.task_id,
            ignore_pm_initiative_id=initiative.initiative_id,
        )
        # Park the plan until the requester approves it. Subtasks are NOT
        # inserted yet; nothing dispatches until the "Start executing" button
        # fires _execute_pm_plan.
        parked = self.store.set_pm_initiative_pending_plan(
            initiative.initiative_id,
            plan_json=serialize_parsed_pm_plan(plan),
            status=PmInitiativeStatus.AWAITING_APPROVAL,
        )
        if parked is None:
            return True
        plan_text = _format_pm_plan_ack(
            parked,
            plan,
            requested_by=initiative.requested_by_slack_user,
            capacity_blocked=shortfall is not None,
        )
        if shortfall is None:
            plan_blocks = _pm_plan_approval_blocks(parked, plan_text)
        else:
            plan_blocks = _pm_plan_capacity_blocks(parked, plan_text, shortfall)
            plan_text = f"{plan_text}\n\n{_format_pm_capacity_shortfall_message(shortfall)}"
        posted = self.gateway.post_thread_reply(
            thread,
            plan_text,
            persona=agent,
            icon_url=self._agent_icon_url(agent),
            blocks=plan_blocks,
        )
        # Remember the message ts so we can strip the buttons once the user
        # acts on the plan.
        with suppress(Exception):
            self.store.set_pm_initiative_pending_plan(
                parked.initiative_id,
                plan_json=parked.pending_plan_json,
                status=PmInitiativeStatus.AWAITING_APPROVAL,
                pending_plan_message_ts=posted.ts,
            )
        # The PM stays attached through the initiative metadata, but the live
        # resolver worker is done once the approval card is parked. Keeping it
        # active lets the managed-run stall watchdog post confusing recovery
        # prompts while the system is correctly waiting for Start executing or
        # already executing subtasks.
        self._finish_pm_resolver_task_after_plan(task)
        resolved_task = self.store.get_agent_task(task.task_id) or task
        self._remove_task_action_buttons_if_resolved(resolved_task)
        self._mark_pm_initiative_request_in_progress(parked, resolved_task)
        self.refresh_or_post_roster(thread.channel_id)
        return True

    def _finish_pm_resolver_task_after_plan(self, task: AgentTask) -> None:
        if self.runtime is not None:
            stop_task = getattr(self.runtime, "stop_task", None)
            if callable(stop_task):
                with suppress(Exception):
                    stop_task(task.task_id, AgentTaskStatus.DONE)
        self.store.update_agent_task_status(task.task_id, AgentTaskStatus.DONE)

    def _execute_pm_plan(
        self,
        initiative: PmInitiative,
        *,
        approver_slack_user: str | None = None,
    ) -> PmInitiative | None:
        """Approve a parked plan: insert subtasks, promote to ACTIVE, fire roots.

        Returns the active initiative on success, or ``None`` if the plan
        could not be applied (invalid JSON, persistence failure, etc.). On
        failure the initiative is moved to CANCELLED and a Slack-visible
        message is posted in its thread.
        """
        if not initiative.pending_plan_json:
            return None
        thread = SlackThreadRef(
            channel_id=initiative.channel_id,
            thread_ts=initiative.thread_ts,
        )
        try:
            plan = deserialize_parsed_pm_plan(initiative.pending_plan_json)
        except Exception as exc:
            LOGGER.exception(
                "failed to deserialize PM plan for initiative %s", initiative.initiative_id
            )
            self.store.set_pm_initiative_pending_plan(
                initiative.initiative_id,
                plan_json=None,
                status=PmInitiativeStatus.CANCELLED,
            )
            self.gateway.post_thread_reply(
                thread,
                f"I could not load the parked PM plan for `{initiative.initiative_id}`: {exc}.",
            )
            return None
        shortfall = self._pm_plan_capacity_shortfall(plan)
        if shortfall is not None:
            self._mark_pm_initiative_request_in_progress(initiative)
            self.gateway.post_thread_reply(
                thread,
                _format_pm_capacity_shortfall_message(shortfall),
                blocks=_pm_capacity_shortfall_blocks(initiative, shortfall),
            )
            return None
        promoted = self.store.set_pm_initiative_pending_plan(
            initiative.initiative_id,
            plan_json=None,
            status=PmInitiativeStatus.ACTIVE,
        )
        if promoted is None:
            return None
        try:
            self._reserve_pm_plan_threads(promoted, plan)
        except Exception as exc:
            LOGGER.exception(
                "failed to persist PM subtasks for initiative %s", initiative.initiative_id
            )
            self.store.update_pm_initiative_status(
                initiative.initiative_id, PmInitiativeStatus.CANCELLED
            )
            with suppress(Exception):
                self.store.cancel_pending_pm_subtask_work(initiative.initiative_id)
            self._cancel_pm_reserved_tasks(initiative.initiative_id)
            self.gateway.post_thread_reply(
                thread,
                f"I could not save the PM plan for `{initiative.initiative_id}`: {exc}.",
            )
            return None
        mention = f"<@{approver_slack_user}>" if approver_slack_user else "Approved"
        self.gateway.post_thread_reply(
            thread,
            f"{mention} — reserved agents and created subtask threads. I will keep "
            "this PM thread updated as subtasks fire and finish.",
        )
        self._post_or_update_pm_status_message(initiative.initiative_id)
        # Best-effort: fire any root subtasks immediately rather than waiting
        # for the next DeferredWorkRunner tick.
        try:
            self._fire_due_pm_initiative_deferred_work_now(promoted.initiative_id)
        except Exception:
            LOGGER.debug("failed to eagerly fire PM root subtasks", exc_info=True)
        return promoted

    def _hire_for_pm_capacity_and_execute(
        self,
        initiative: PmInitiative,
        channel_id: str,
        message_ts: str | None,
        *,
        approver_slack_user: str | None = None,
    ) -> PmInitiative | None:
        thread = SlackThreadRef(channel_id=initiative.channel_id, thread_ts=initiative.thread_ts)
        if not initiative.pending_plan_json:
            return None
        try:
            plan = deserialize_parsed_pm_plan(initiative.pending_plan_json)
        except Exception:
            return self._execute_pm_plan(initiative, approver_slack_user=approver_slack_user)
        shortfall = self._pm_plan_capacity_shortfall(plan)
        if shortfall is not None and shortfall.unavailable_handles:
            self.gateway.post_thread_reply(
                thread,
                _format_pm_capacity_shortfall_message(shortfall),
            )
            return None
        hire_count = shortfall.hire_count if shortfall is not None else 0
        if hire_count > 0:
            if not self._can_hire(hire_count):
                self.gateway.post_thread_reply(
                    thread,
                    f"{AGENT_LIMIT_MESSAGE} Max team size is {MAX_TEAM_AGENTS}.",
                )
                return None
            hired = self.hire_agents(hire_count)
            summary = ", ".join(
                f"@{agent.handle} ({agent.provider_preference.value})" for agent in hired
            )
            self.gateway.post_thread_reply(
                thread,
                (
                    f"Hired {len(hired)} worker agent(s) for PM initiative "
                    f"`{initiative.initiative_id}`: {summary}"
                ),
            )
            for agent in hired:
                try:
                    self.gateway.post_thread_reply(
                        thread,
                        format_agent_introduction(agent),
                        persona=agent,
                        icon_url=self._agent_icon_url(agent),
                    )
                except Exception:
                    LOGGER.debug("failed to post hired PM-capacity introduction", exc_info=True)
        latest = self.store.get_pm_initiative(initiative.initiative_id) or initiative
        plan_message_ts = latest.pending_plan_message_ts or initiative.pending_plan_message_ts
        promoted = self._execute_pm_plan(latest, approver_slack_user=approver_slack_user)
        if promoted is not None:
            self._strip_pm_plan_buttons(promoted, message_ts, status_label="approved")
            if plan_message_ts and plan_message_ts != message_ts:
                self._strip_pm_plan_buttons(promoted, plan_message_ts, status_label="approved")
        self.refresh_or_post_roster(channel_id)
        return promoted

    def _fire_due_pm_initiative_deferred_work_now(self, initiative_id: str) -> int:
        fired = 0
        now = utc_now()
        for subtask in self.store.list_pm_subtasks(initiative_id):
            deferred = self.store.get_deferred_work(subtask.deferred_id)
            if deferred is None or deferred.status != DeferredWorkStatus.READY:
                continue
            if deferred.fire_at is None or deferred.fire_at > now:
                continue
            if deferred.last_task_id:
                task = self.store.get_agent_task(deferred.last_task_id)
                if task is not None and task.status == AgentTaskStatus.ACTIVE:
                    continue
            if self.fire_due_deferred_work(deferred):
                fired += 1
        return fired

    def _reserve_pm_plan_threads(
        self,
        initiative: PmInitiative,
        plan: ParsedPmPlan,
    ) -> None:
        """Create every PM subtask thread and reserve its agent before execution."""
        active_agents = self.store.list_team_agents()
        busy_agent_ids = self._pm_busy_worker_agent_ids()
        worker_agents = [
            agent
            for agent in filter_worker_agents(active_agents)
            if agent.agent_id not in busy_agent_ids
        ]
        initial_busy = {
            agent.agent_id: self.store.active_task_for_agent(agent.agent_id)
            for agent in worker_agents
        }
        assigned_counts: dict[str, int] = {
            agent.agent_id: 1 if initial_busy.get(agent.agent_id) is not None else 0
            for agent in worker_agents
        }
        prior_subtasks = self.store.list_pm_subtasks(initiative.initiative_id)
        inserted: list[object] = list(prior_subtasks)
        last_reserved_by_agent = self._pm_last_reserved_subtask_by_agent(prior_subtasks)
        next_sort_order = max((item.sort_order for item in prior_subtasks), default=-1) + 1
        for offset, subtask in enumerate(plan.subtasks):
            agent = self._select_pm_subtask_agent(
                subtask,
                worker_agents=worker_agents,
                assigned_counts=assigned_counts,
            )
            if agent is None:
                raise ValueError(
                    f"no active worker agent is available for subtask {subtask.local_id}"
                )
            extra_deps = self._pm_capacity_dependencies(
                initiative,
                subtask,
                agent,
                last_reserved_by_agent=last_reserved_by_agent,
                initial_busy=initial_busy,
            )
            reserved = self._create_pm_reserved_subtask_thread(
                initiative,
                subtask,
                agent,
                extra_deps=extra_deps,
            )
            saved = self.store.add_pm_subtask_dispatch(
                initiative=initiative,
                local_id=subtask.local_id,
                title=subtask.title,
                request=subtask.request,
                plan_depends_on=subtask.depends_on,
                existing_subtasks=list(inserted),
                after_delay_seconds=subtask.after_delay_seconds,
                sort_order=next_sort_order + offset,
                thread=reserved.thread,
                extra_deferred_depends_on=reserved.extra_depends_on,
            )
            self._attach_reserved_task_to_deferred(
                reserved.task, reserved.thread, saved.deferred_id
            )
            self.store.update_deferred_work_last_task(
                saved.deferred_id,
                last_task_id=reserved.task.task_id,
            )
            inserted.append(saved)
            last_reserved_by_agent[agent.agent_id] = saved
            assigned_counts[agent.agent_id] = assigned_counts.get(agent.agent_id, 0) + 1

    def _pm_plan_capacity_shortfall(
        self,
        plan: ParsedPmPlan,
        *,
        ignore_task_id: str | None = None,
        ignore_pm_initiative_id: str | None = None,
    ) -> _PmCapacityShortfall | None:
        required_workers = _pm_plan_parallel_worker_demand(plan)
        if required_workers <= 0:
            return None
        active_workers = filter_worker_agents(self.store.list_team_agents())
        busy_agent_ids = self._pm_busy_worker_agent_ids(
            ignore_task_id=ignore_task_id,
            ignore_pm_initiative_id=ignore_pm_initiative_id,
        )
        worker_by_handle = {agent.handle: agent for agent in active_workers}
        specific_handles = tuple(
            dict.fromkeys(
                subtask.request.requested_handle
                for subtask in plan.subtasks
                if subtask.request.assignment_mode == AssignmentMode.SPECIFIC
                and subtask.request.requested_handle
            )
        )
        unavailable_handles = tuple(
            handle
            for handle in specific_handles
            if (agent := worker_by_handle.get(handle)) is None or agent.agent_id in busy_agent_ids
        )
        if unavailable_handles:
            return _PmCapacityShortfall(
                required_workers=required_workers,
                available_workers=sum(
                    1 for agent in active_workers if agent.agent_id not in busy_agent_ids
                ),
                hire_count=0,
                unavailable_handles=unavailable_handles,
            )
        available_workers = sum(
            1 for agent in active_workers if agent.agent_id not in busy_agent_ids
        )
        hire_count = max(0, required_workers - available_workers)
        if hire_count <= 0:
            return None
        return _PmCapacityShortfall(
            required_workers=required_workers,
            available_workers=available_workers,
            hire_count=hire_count,
        )

    def _pm_busy_worker_agent_ids(
        self,
        *,
        ignore_task_id: str | None = None,
        ignore_pm_initiative_id: str | None = None,
    ) -> set[str]:
        busy = {
            task.agent_id
            for task in self.store.list_agent_tasks()
            if not ignore_task_id or task.task_id != ignore_task_id
        }
        busy.update(self._external_busy_agent_ids())
        busy.update(self._pm_owner_busy_agent_ids(ignore_initiative_id=ignore_pm_initiative_id))
        busy.update(self._scheduled_busy_agent_ids())
        busy.update(self._deferred_busy_agent_ids())
        return busy

    def _select_pm_subtask_agent(
        self,
        subtask: ParsedPmSubtask,
        *,
        worker_agents: list[TeamAgent],
        assigned_counts: dict[str, int],
    ) -> TeamAgent | None:
        if (
            subtask.request.assignment_mode == AssignmentMode.SPECIFIC
            and subtask.request.requested_handle
        ):
            requested = subtask.request.requested_handle
            return next((agent for agent in worker_agents if agent.handle == requested), None)
        candidates = worker_agents
        if subtask.request.task_kind == AgentTaskKind.REVIEW:
            reviewers = [
                agent for agent in candidates if agent.provider_preference == Provider.CLAUDE
            ]
            if reviewers:
                candidates = reviewers
        if not candidates:
            return None
        min_count = min(assigned_counts.get(agent.agent_id, 0) for agent in candidates)
        least_loaded = [
            agent for agent in candidates if assigned_counts.get(agent.agent_id, 0) == min_count
        ]
        digest = hashlib.sha256(f"{subtask.local_id}:{subtask.request.prompt}".encode()).digest()
        return sorted(least_loaded, key=lambda item: item.handle)[digest[0] % len(least_loaded)]

    def _pm_capacity_dependencies(
        self,
        initiative: PmInitiative,
        subtask: ParsedPmSubtask,
        agent: TeamAgent,
        *,
        last_reserved_by_agent: dict[str, object],
        initial_busy: dict[str, AgentTask | None],
    ) -> tuple[WorkDependency, ...]:
        previous = last_reserved_by_agent.get(agent.agent_id)
        if previous is not None and getattr(previous, "local_id", None) not in subtask.depends_on:
            return (
                WorkDependency(
                    kind=WorkDependencyKind.SUBTASK,
                    task_id=getattr(previous, "deferred_id", None),
                    initiative_id=initiative.initiative_id,
                    local_id=getattr(previous, "local_id", None),
                    description=f"agent @{agent.handle} reservation",
                ),
            )
        if previous is None:
            busy = initial_busy.get(agent.agent_id)
            if busy is not None and busy.status in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}:
                return (
                    WorkDependency(
                        kind=WorkDependencyKind.AGENT_BUSY,
                        task_id=busy.task_id,
                        handle=agent.handle,
                        description=f"agent @{agent.handle} availability",
                    ),
                )
        return ()

    def _pm_last_reserved_subtask_by_agent(self, subtasks: list[object]) -> dict[str, object]:
        by_agent: dict[str, object] = {}
        for subtask in subtasks:
            deferred_id = getattr(subtask, "deferred_id", None)
            if not isinstance(deferred_id, str):
                continue
            deferred = self.store.get_deferred_work(deferred_id)
            if deferred is None or not deferred.last_task_id:
                continue
            task = self.store.get_agent_task(deferred.last_task_id)
            if task is None:
                continue
            by_agent[task.agent_id] = subtask
        return by_agent

    def _create_pm_reserved_subtask_thread(
        self,
        initiative: PmInitiative,
        subtask: ParsedPmSubtask,
        agent: TeamAgent,
        *,
        extra_deps: tuple[WorkDependency, ...],
    ) -> _PmReservedTask:
        task = create_agent_task(
            agent,
            subtask.request.prompt,
            initiative.channel_id,
            requested_by_slack_user=initiative.requested_by_slack_user,
            kind=subtask.request.task_kind,
        )
        metadata = dict(task.metadata)
        metadata.update(
            {
                PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                PM_SUBTASK_LOCAL_ID_METADATA_KEY: subtask.local_id,
                "parent_task_id": initiative.pm_task_id or f"pm:{initiative.initiative_id}",
                ASSIGNMENT_PROMPT_METADATA_KEY: subtask.request.prompt,
                ORIGINAL_TASK_METADATA_KEY: subtask.request.prompt,
                ROSTER_SUMMARY_METADATA_KEY: f"Reserved PM subtask `{subtask.local_id}`.",
            }
        )
        if subtask.request.author_handle:
            metadata["author_handle"] = subtask.request.author_handle
        if subtask.request.pr_url:
            metadata[PR_URL_METADATA_KEY] = subtask.request.pr_url
        if subtask.request.permission_mode != DEFAULT_PERMISSION_MODE:
            metadata[PERMISSION_MODE_METADATA_KEY] = subtask.request.permission_mode.value
        if subtask.request.dangerous_mode:
            metadata[DANGEROUS_MODE_METADATA_KEY] = True
        metadata = metadata_with_pr_urls(metadata, subtask.request.prompt, subtask.request.pr_url)
        task = replace(task, metadata=metadata)
        text = _format_pm_subtask_parent_text(
            initiative,
            subtask,
            agent,
            pm_thread_url=self._thread_permalink(initiative.channel_id, initiative.thread_ts),
            extra_deps=extra_deps,
        )
        blocks = _pm_subtask_parent_blocks(text)
        posted = self.gateway.post_session_parent(
            initiative.channel_id,
            text,
            persona=agent,
            icon_url=self._agent_icon_url(agent),
            blocks=blocks,
        )
        thread = SlackThreadRef(initiative.channel_id, posted.ts, posted.ts)
        task = replace(task, thread_ts=thread.thread_ts, parent_message_ts=thread.message_ts)
        self.store.upsert_agent_task(task)
        return _PmReservedTask(
            subtask=subtask,
            agent=agent,
            task=task,
            thread=thread,
            extra_depends_on=extra_deps,
        )

    def _attach_reserved_task_to_deferred(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
        deferred_id: str,
    ) -> AgentTask:
        latest = self.store.get_agent_task(task.task_id) or task
        metadata = dict(latest.metadata)
        metadata["deferred_work_id"] = deferred_id
        updated = replace(
            latest,
            thread_ts=thread.thread_ts,
            parent_message_ts=thread.message_ts,
            metadata=metadata,
            updated_at=utc_now(),
        )
        self.store.upsert_agent_task(updated)
        return updated

    def _cancel_pm_reserved_tasks(self, initiative_id: str) -> None:
        for task in self.store.list_agent_tasks(include_done=True):
            if task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY) != initiative_id:
                continue
            if task.status in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}:
                self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)

    def _retry_pm_resolution(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        error: str,
    ) -> bool:
        if not task.metadata.get(PM_RESOLUTION_METADATA_KEY):
            self.gateway.post_thread_reply(
                thread,
                f"I could not save that PM plan: {error}.",
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            return False
        attempts = int(task.metadata.get(PM_RESOLUTION_ATTEMPTS_METADATA_KEY) or 0)
        initiative_id = task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY)
        if attempts >= MAX_PM_RESOLUTION_ATTEMPTS or not self.runtime:
            self.gateway.post_thread_reply(
                thread,
                (
                    "I could not produce a valid PM plan after "
                    f"{MAX_PM_RESOLUTION_ATTEMPTS} attempts: {error}. "
                    "Cancelling this initiative."
                ),
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            if isinstance(initiative_id, str) and initiative_id:
                self.store.update_pm_initiative_status(initiative_id, PmInitiativeStatus.CANCELLED)
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
            self.refresh_or_post_roster(thread.channel_id)
            return True
        original_text = task.metadata.get(PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY)
        if not isinstance(original_text, str) or not original_text.strip():
            original_text = task.prompt
        available_workers = self._pm_available_worker_agents(
            ignore_task_id=task.task_id,
            ignore_pm_initiative_id=str(initiative_id) if initiative_id else None,
        )
        extension_known_ids = self._task_extension_known_ids(task)
        extension_context_raw = task.metadata.get(PM_EXTENSION_CONTEXT_METADATA_KEY)
        extension_context = (
            extension_context_raw if isinstance(extension_context_raw, str) else None
        )
        replan_context_raw = task.metadata.get(PM_REPLAN_CONTEXT_METADATA_KEY)
        replan_context = replan_context_raw if isinstance(replan_context_raw, str) else None
        prior_summary: str | None = None
        if isinstance(initiative_id, str) and initiative_id:
            prior_summary = self._build_pm_initiative_plan_view(initiative_id)
        retry_prompt = build_pm_resolution_prompt(
            original_text,
            _pm_worker_handles(available_workers),
            initiative_id=str(initiative_id) if initiative_id else "",
            now=utc_now(),
            validation_error=error,
            replan_context=replan_context,
            extension_known_ids=extension_known_ids,
            extension_context=extension_context,
            prior_plan_summary=prior_summary if (replan_context or extension_known_ids) else None,
            agent_models=_pm_worker_model_map(available_workers),
        )
        metadata = dict(task.metadata)
        metadata[PM_RESOLUTION_ATTEMPTS_METADATA_KEY] = attempts + 1
        retry_task = replace(
            task,
            prompt=retry_prompt,
            status=AgentTaskStatus.ACTIVE,
            updated_at=utc_now(),
            metadata=metadata,
        )
        self.store.upsert_agent_task(retry_task)
        started = self.runtime.start_task(retry_task, agent, thread)
        if not started:
            self.gateway.post_thread_reply(
                thread,
                (
                    "I could not restart PM plan validation "
                    f"(last validation error: {error}), so I cancelled this initiative."
                ),
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
            if isinstance(initiative_id, str) and initiative_id:
                self.store.update_pm_initiative_status(initiative_id, PmInitiativeStatus.CANCELLED)
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
            self.refresh_or_post_roster(thread.channel_id)
        return True

    def _maybe_start_pm_replan(
        self,
        task: AgentTask,
        channel_id: str,
        thread_ts: str,
        message_ts: str | None,
        text: str,
    ) -> bool:
        """Re-run the PM resolver after a failure or course-correction.

        Cancels every non-DONE subtask, resets the initiative to PLANNING,
        and starts a fresh resolver task carrying the user's new context
        plus a snapshot of what already finished. The PM's new plan goes
        through the same approval gate as the original.
        """
        replan_body = parse_pm_replan_request(text)
        if replan_body is None:
            return False
        initiative_id = task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY)
        if not isinstance(initiative_id, str) or not initiative_id:
            return False
        initiative = self.store.get_pm_initiative(initiative_id)
        if initiative is None or initiative.status in {
            PmInitiativeStatus.DONE,
            PmInitiativeStatus.CANCELLED,
        }:
            return False
        if self.runtime is None:
            self.gateway.post_thread_reply(
                SlackThreadRef(channel_id, thread_ts, task.parent_message_ts),
                "Replan is unavailable because the runtime is not connected.",
            )
            self._mark_message_complete(channel_id, message_ts)
            return True
        agent = self.store.get_team_agent(task.agent_id)
        if (
            agent is None
            or getattr(agent, "kind", None) != TeamAgentKind.PM
            or task.status not in {AgentTaskStatus.ACTIVE, AgentTaskStatus.DONE}
        ):
            # Only PM-kind agents stay attached past plan approval. For
            # worker-fallback resolvers the agent is back to general duty and
            # restarting the task would race with whatever it is doing now.
            self.gateway.post_thread_reply(
                SlackThreadRef(channel_id, thread_ts, task.parent_message_ts),
                (
                    "Replan needs a PM-kind agent still attached to this initiative. "
                    "Hire a PM (`team hire --kind pm`) and start a new `pm: ...` "
                    "initiative instead."
                ),
            )
            self._mark_message_complete(channel_id, message_ts)
            return True
        if self._pm_owner_is_busy(agent, task):
            self.gateway.post_thread_reply(
                SlackThreadRef(channel_id, thread_ts, task.parent_message_ts),
                (
                    f"@{agent.handle} is busy right now, so I did not start replanning. "
                    "Try again when that PM is free, or hire another PM and start a new "
                    "`pm: ...` initiative."
                ),
            )
            self._mark_message_complete(channel_id, message_ts)
            return True
        # Snapshot the prior plan BEFORE we cancel anything so the resolver
        # sees what already finished and can build on it.
        prior_summary = self._build_pm_initiative_plan_view(initiative_id)
        cancelled = 0
        try:
            cancelled = self.store.cancel_pending_pm_subtask_work(initiative_id)
        except Exception:
            LOGGER.exception(
                "failed to cancel pending PM subtasks during replan for %s",
                initiative_id,
            )
        self.store.set_pm_initiative_pending_plan(
            initiative_id,
            plan_json=None,
            status=PmInitiativeStatus.PLANNING,
        )
        original_text = task.metadata.get(PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY)
        if not isinstance(original_text, str) or not original_text.strip():
            original_text = initiative.summary or initiative.title
        available_workers = self._pm_available_worker_agents(
            ignore_task_id=task.task_id,
            ignore_pm_initiative_id=initiative_id,
        )
        retry_prompt = build_pm_resolution_prompt(
            original_text,
            _pm_worker_handles(available_workers),
            initiative_id=initiative_id,
            now=utc_now(),
            replan_context=replan_body or None,
            prior_plan_summary=prior_summary,
            agent_models=_pm_worker_model_map(available_workers),
        )
        metadata = dict(task.metadata)
        metadata[PM_RESOLUTION_METADATA_KEY] = True
        metadata[PM_RESOLUTION_ATTEMPTS_METADATA_KEY] = 0
        metadata[PM_INITIATIVE_ID_METADATA_KEY] = initiative_id
        metadata[PM_REPLAN_CONTEXT_METADATA_KEY] = replan_body
        # A replan run resets extension state so the new prompt is unambiguous.
        metadata.pop(PM_EXTENSION_KNOWN_IDS_METADATA_KEY, None)
        metadata.pop(PM_EXTENSION_CONTEXT_METADATA_KEY, None)
        retry_task = replace(
            task,
            prompt=retry_prompt,
            status=AgentTaskStatus.ACTIVE,
            updated_at=utc_now(),
            metadata=metadata,
        )
        self.store.upsert_agent_task(retry_task)
        thread = SlackThreadRef(channel_id, thread_ts, task.parent_message_ts)
        # PM-kind agents keep their resolver task ACTIVE past approval so
        # the user can keep asking questions or replanning. That means
        # the runtime still holds a live worker for `retry_task.task_id`
        # and `start_task` would refuse to overlay a fresh resolver run
        # on top of it (the slot is occupied). Stop the prior worker
        # first so the fresh prompt actually launches.
        with suppress(Exception):
            self.runtime.stop_task(retry_task.task_id, status=None)
        started = self.runtime.start_task(retry_task, agent, thread)
        if not started:
            self.store.update_pm_initiative_status(initiative_id, PmInitiativeStatus.CANCELLED)
            self.gateway.post_thread_reply(
                thread,
                "I could not restart PM planning, so I cancelled this initiative.",
            )
            self._mark_message_complete(channel_id, message_ts)
            return True
        cancelled_text = f" Cancelled {cancelled} in-flight subtask(s)." if cancelled else ""
        self.gateway.post_thread_reply(
            thread,
            f"Replanning this initiative.{cancelled_text} I will park the new plan "
            "for approval before any new subtasks run.",
        )
        self._mark_message_in_progress(channel_id, message_ts)
        return True

    def _task_extension_known_ids(self, task: AgentTask) -> tuple[str, ...]:
        raw = task.metadata.get(PM_EXTENSION_KNOWN_IDS_METADATA_KEY)
        if not isinstance(raw, (list, tuple)):
            return ()
        return tuple(item for item in raw if isinstance(item, str) and item)

    def _maybe_start_pm_extension(
        self,
        task: AgentTask,
        channel_id: str,
        thread_ts: str,
        message_ts: str | None,
        text: str,
    ) -> bool:
        """Add new subtasks to an ACTIVE initiative without cancelling existing work.

        The PM agent receives the current DAG plus the user's extension
        instructions and emits a fresh PM_PLAN whose ``depends_on`` entries
        may reference existing subtask ids. The new plan goes through the
        normal approval gate; existing subtasks keep running.
        """
        extension_body = parse_pm_extension_request(text)
        if extension_body is None:
            return False
        initiative_id = task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY)
        if not isinstance(initiative_id, str) or not initiative_id:
            return False
        initiative = self.store.get_pm_initiative(initiative_id)
        if initiative is None:
            return False
        if initiative.status != PmInitiativeStatus.ACTIVE:
            self.gateway.post_thread_reply(
                SlackThreadRef(channel_id, thread_ts, task.parent_message_ts),
                (
                    f"Extension only applies to ACTIVE initiatives — this one is "
                    f"{initiative.status.value}."
                ),
            )
            self._mark_message_complete(channel_id, message_ts)
            return True
        if self.runtime is None:
            self.gateway.post_thread_reply(
                SlackThreadRef(channel_id, thread_ts, task.parent_message_ts),
                "Extension is unavailable because the runtime is not connected.",
            )
            self._mark_message_complete(channel_id, message_ts)
            return True
        agent = self.store.get_team_agent(task.agent_id)
        if (
            agent is None
            or getattr(agent, "kind", None) != TeamAgentKind.PM
            or task.status not in {AgentTaskStatus.ACTIVE, AgentTaskStatus.DONE}
        ):
            self.gateway.post_thread_reply(
                SlackThreadRef(channel_id, thread_ts, task.parent_message_ts),
                (
                    "Extension needs a PM-kind agent still attached to this initiative. "
                    "Hire a PM and start a new `pm: ...` initiative instead."
                ),
            )
            self._mark_message_complete(channel_id, message_ts)
            return True
        if self._pm_owner_is_busy(agent, task):
            self.gateway.post_thread_reply(
                SlackThreadRef(channel_id, thread_ts, task.parent_message_ts),
                (
                    f"@{agent.handle} is busy right now, so I did not start planning "
                    "the extension. Try again when that PM is free, or hire another PM "
                    "and start a new `pm: ...` initiative."
                ),
            )
            self._mark_message_complete(channel_id, message_ts)
            return True
        existing = self.store.list_pm_subtasks(initiative_id)
        if not existing:
            self.gateway.post_thread_reply(
                SlackThreadRef(channel_id, thread_ts, task.parent_message_ts),
                "Extension needs at least one existing subtask. Approve the first plan first.",
            )
            self._mark_message_complete(channel_id, message_ts)
            return True
        existing_ids = tuple(item.local_id for item in existing)
        prior_summary = self._build_pm_initiative_plan_view(initiative_id)
        # Park the extension request: drop back to PLANNING so the PM_PLAN
        # handler treats the next signal as a fresh plan to park for approval.
        self.store.set_pm_initiative_pending_plan(
            initiative_id,
            plan_json=None,
            status=PmInitiativeStatus.PLANNING,
        )
        original_text = task.metadata.get(PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY)
        if not isinstance(original_text, str) or not original_text.strip():
            original_text = initiative.summary or initiative.title
        available_workers = self._pm_available_worker_agents(
            ignore_task_id=task.task_id,
            ignore_pm_initiative_id=initiative_id,
        )
        retry_prompt = build_pm_resolution_prompt(
            original_text,
            _pm_worker_handles(available_workers),
            initiative_id=initiative_id,
            now=utc_now(),
            extension_known_ids=existing_ids,
            extension_context=extension_body,
            prior_plan_summary=prior_summary,
            agent_models=_pm_worker_model_map(available_workers),
        )
        metadata = dict(task.metadata)
        metadata[PM_RESOLUTION_METADATA_KEY] = True
        metadata[PM_RESOLUTION_ATTEMPTS_METADATA_KEY] = 0
        metadata[PM_INITIATIVE_ID_METADATA_KEY] = initiative_id
        metadata[PM_EXTENSION_KNOWN_IDS_METADATA_KEY] = list(existing_ids)
        metadata[PM_EXTENSION_CONTEXT_METADATA_KEY] = extension_body
        # An extension run resets replan state so the new prompt is unambiguous.
        metadata.pop(PM_REPLAN_CONTEXT_METADATA_KEY, None)
        retry_task = replace(
            task,
            prompt=retry_prompt,
            status=AgentTaskStatus.ACTIVE,
            updated_at=utc_now(),
            metadata=metadata,
        )
        self.store.upsert_agent_task(retry_task)
        thread = SlackThreadRef(channel_id, thread_ts, task.parent_message_ts)
        started = self.runtime.start_task(retry_task, agent, thread)
        if not started:
            # Restore the initiative to ACTIVE — we cleared pending_plan_json
            # but the existing subtasks are still running.
            self.store.update_pm_initiative_status(initiative_id, PmInitiativeStatus.ACTIVE)
            self.gateway.post_thread_reply(
                thread,
                "I could not restart the PM agent to plan the extension. "
                "Existing subtasks keep running unchanged.",
            )
            self._mark_message_complete(channel_id, message_ts)
            return True
        self.gateway.post_thread_reply(
            thread,
            "Planning the extension. I will park the new subtasks for approval "
            "before they run; existing subtasks keep running.",
        )
        self._mark_message_in_progress(channel_id, message_ts)
        return True

    def _pm_owner_is_busy(self, agent: TeamAgent, task: AgentTask) -> bool:
        active = self.store.active_task_for_agent(agent.agent_id)
        if active is not None and active.task_id != task.task_id:
            return True
        initiative_id = task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY)
        ignore_pm_initiative_id = initiative_id if isinstance(initiative_id, str) else None
        return agent.agent_id in self._busy_agent_ids_for_assignment(
            ignore_pm_initiative_id=ignore_pm_initiative_id,
        )

    def _maybe_post_pm_status_command(
        self,
        task: AgentTask,
        channel_id: str,
        thread_ts: str,
        message_ts: str | None,
        text: str,
    ) -> bool:
        """Render the DAG directly when a user types a ``pm status``-style command.

        Skips the agent round-trip so the answer is instant. Only fires inside a
        thread whose oldest task is a PM resolver (which carries the initiative
        id on its metadata); other threads fall through.
        """
        if not looks_like_pm_status_request(text):
            return False
        initiative_id = task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY)
        if not isinstance(initiative_id, str) or not initiative_id:
            return False
        view = self._build_pm_initiative_plan_view(initiative_id)
        if view is None:
            return False
        self.gateway.post_thread_reply(
            SlackThreadRef(channel_id, thread_ts, task.parent_message_ts),
            view,
        )
        self._mark_message_complete(channel_id, message_ts)
        return True

    def _wrap_pm_followup_text(
        self,
        task: AgentTask,
        agent,
        text: str,
    ) -> str:
        """Prepend a ``[PM HARNESS: initiative state]`` snapshot for PM owners.

        The PM persona prompt promises the agent will see this snapshot on
        every follow-up so it can answer status questions without re-reading
        the whole thread. Engineer-kind resolvers (used only when no PM agent
        is hired) do not get the snapshot — the original behavior is preserved.
        """
        if agent is None or getattr(agent, "kind", None) != TeamAgentKind.PM:
            return text
        initiative_id = task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY)
        if not isinstance(initiative_id, str) or not initiative_id:
            return text
        snapshot = self._build_pm_initiative_snapshot(initiative_id)
        if not snapshot:
            return text
        return f"{snapshot}\n\n{text}"

    def _build_pm_initiative_snapshot(self, initiative_id: str) -> str | None:
        view = self._build_pm_initiative_plan_view(
            initiative_id, header="[PM HARNESS: initiative state]"
        )
        if view is None:
            return None
        return view

    def _build_pm_initiative_plan_view(
        self,
        initiative_id: str,
        *,
        header: str | None = None,
    ) -> str | None:
        """Render the current DAG with per-subtask status, owner, and deps.

        Used by:
        - the PM agent follow-up snapshot (with the ``[PM HARNESS: ...]`` header)
        - the on-demand ``pm status`` command in initiative threads
        - the terminal recap posted by the watchdog
        """
        initiative = self.store.get_pm_initiative(initiative_id)
        if initiative is None:
            return None
        subtasks = self.store.list_pm_subtasks(initiative_id)
        lines: list[str] = []
        if header:
            lines.append(header)
        lines.extend(
            [
                f"id: {initiative.initiative_id}",
                f"title: {initiative.title}",
                f"status: {_pm_initiative_status_label(initiative.status)}",
            ]
        )
        if not subtasks:
            lines.append("subtasks: (none yet — still in planning)")
            return "\n".join(lines)
        counts: dict[str, int] = {}
        rows: list[str] = []
        for subtask in subtasks:
            deferred = self.store.get_deferred_work(subtask.deferred_id)
            if deferred is None:
                rows.append(
                    f"  - {subtask.local_id}: {subtask.title} "
                    f"[{_pm_subtask_status_label('missing')}]"
                )
                counts["missing"] = counts.get("missing", 0) + 1
                continue
            worker_task = None
            if deferred.last_task_id:
                worker_task = self.store.get_agent_task(deferred.last_task_id)
            owner = self._pm_subtask_owner_label(subtask, deferred, worker_task)
            status_value = _pm_deferred_display_status(deferred, worker_task)
            counts[status_value] = counts.get(status_value, 0) + 1
            deps_text = (
                f"deps: {', '.join(subtask.depends_on)}" if subtask.depends_on else "deps: none"
            )
            thread_url = self._pm_subtask_thread_url(deferred, worker_task)
            link_text = f", thread: <{thread_url}|open>" if thread_url else ""
            rows.append(
                f"  - {subtask.local_id}: {subtask.title} "
                f"[{_pm_subtask_status_label(status_value)}, owner={owner}, "
                f"{deps_text}{link_text}]"
            )
        if counts:
            summary = ", ".join(
                f"{count} {_pm_subtask_status_label(state)}"
                for state, count in sorted(
                    counts.items(), key=lambda kv: _pm_subtask_status_sort_key(kv[0])
                )
            )
            lines.append(f"subtasks ({summary}):")
        else:
            lines.append("subtasks:")
        lines.extend(rows)
        return "\n".join(lines)

    def _pm_subtask_thread_url(
        self,
        deferred: DeferredWork,
        task: AgentTask | None,
    ) -> str | None:
        if task is not None:
            message_ts = task.parent_message_ts or task.thread_ts
            url = self._thread_permalink(task.channel_id, message_ts)
            if url:
                return url
        return self._thread_permalink(
            deferred.channel_id, deferred.message_ts or deferred.thread_ts
        )

    def _post_or_update_pm_status_message(self, initiative_id: str) -> str | None:
        initiative = self.store.get_pm_initiative(initiative_id)
        if initiative is None:
            return None
        view = self._build_pm_initiative_plan_view(
            initiative_id,
            header="PM initiative status",
        )
        if view is None:
            return None
        key = _pm_status_message_setting_key(initiative_id)
        existing_ts = self.store.get_setting(key)
        if existing_ts and self._try_update_pm_status_message(
            initiative.channel_id,
            existing_ts,
            view,
        ):
            return existing_ts
        thread = SlackThreadRef(initiative.channel_id, initiative.thread_ts, initiative.message_ts)
        try:
            posted = self._post_pm_status_reply(thread, view)
        except Exception:
            LOGGER.debug("failed to post PM status message", exc_info=True)
            return None
        self.store.set_setting(key, posted.ts)
        return posted.ts

    def _post_pm_status_reply(self, thread: SlackThreadRef, view: str):
        try:
            return self.gateway.post_thread_reply(
                thread,
                view,
                unfurl_links=False,
                unfurl_media=False,
            )
        except TypeError as exc:
            if "unfurl" not in str(exc):
                raise
            return self.gateway.post_thread_reply(thread, view)

    def _try_update_pm_status_message(self, channel_id: str, ts: str, view: str) -> bool:
        try:
            self.gateway.update_message(
                channel_id,
                ts,
                view,
                unfurl_links=False,
                unfurl_media=False,
                attachments=[],
            )
        except TypeError as exc:
            if "unfurl" not in str(exc) and "attachments" not in str(exc):
                LOGGER.debug(
                    "failed to update PM status Slack message %s in %s",
                    ts,
                    channel_id,
                    exc_info=True,
                )
                return False
            try:
                self.gateway.update_message(channel_id, ts, view)
            except Exception:
                LOGGER.debug(
                    "failed to update PM status Slack message %s in %s",
                    ts,
                    channel_id,
                    exc_info=True,
                )
                return False
            return True
        except Exception:
            LOGGER.debug(
                "failed to update PM status Slack message %s in %s",
                ts,
                channel_id,
                exc_info=True,
            )
            return False
        return True

    def _pm_subtask_owner_label(
        self,
        subtask: PmSubtask,
        deferred: DeferredWork,
        task: AgentTask | None,
    ) -> str:
        if task is not None:
            agent = self.store.get_team_agent(task.agent_id)
            if agent is not None:
                return f"@{agent.handle}"
        for thread_task in reversed(
            self.store.list_thread_agent_tasks(deferred.channel_id, deferred.thread_ts)
        ):
            if (
                thread_task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY) == subtask.initiative_id
                and thread_task.metadata.get(PM_SUBTASK_LOCAL_ID_METADATA_KEY) == subtask.local_id
            ):
                agent = self.store.get_team_agent(thread_task.agent_id)
                if agent is not None:
                    return f"@{agent.handle}"
        if deferred.requested_handle:
            return f"@{deferred.requested_handle}"
        return "unassigned"

    def _refresh_pm_status_for_deferred(self, deferred_id: str) -> None:
        subtask = self.store.get_pm_subtask_by_deferred_id(deferred_id)
        if subtask is None:
            return
        self._post_or_update_pm_status_message(subtask.initiative_id)

    def _refresh_pm_status_for_task(self, task: AgentTask) -> None:
        initiative_id = task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY)
        if not isinstance(initiative_id, str) or not initiative_id:
            return
        self._post_or_update_pm_status_message(initiative_id)

    def watch_pm_initiatives(self) -> int:
        active = self.store.list_pm_initiatives(
            statuses=(PmInitiativeStatus.ACTIVE,),
            limit=50,
        )
        surfaced = 0
        for initiative in active:
            try:
                surfaced += self._watch_one_pm_initiative(initiative)
            except Exception:
                # Do NOT mark watchdog_last_run_at on failure: a monitor that
                # checks for a stale heartbeat needs to notice that this tick
                # never produced a clean evaluation.
                LOGGER.debug(
                    "failed to evaluate PM initiative %s",
                    initiative.initiative_id,
                    exc_info=True,
                )
                continue
            try:
                self.store.mark_pm_initiative_watchdog_run(initiative.initiative_id)
            except Exception:
                LOGGER.debug(
                    "failed to mark PM initiative watchdog run %s",
                    initiative.initiative_id,
                    exc_info=True,
                )
        return surfaced

    def _watch_one_pm_initiative(self, initiative: PmInitiative) -> int:
        if initiative.status != PmInitiativeStatus.ACTIVE:
            # PLANNING: resolver is still running. AWAITING_APPROVAL: the user
            # has not pressed Start executing yet. DONE / CANCELLED: terminal.
            # The query in watch_pm_initiatives already filters to ACTIVE, so
            # this guard only catches a status change between query and call.
            return 0
        subtasks = self.store.list_pm_subtasks(initiative.initiative_id)
        if not subtasks:
            # Active without subtasks means the plan was applied but the rows
            # disappeared — surface once and cancel.
            self._surface_pm_blocker(
                initiative,
                local_id="*",
                kind="missing_subtasks",
                message=(
                    f"PM initiative `{initiative.initiative_id}` has no subtasks. Cancelling."
                ),
            )
            self.store.update_pm_initiative_status(
                initiative.initiative_id, PmInitiativeStatus.CANCELLED
            )
            self._post_or_update_pm_status_message(initiative.initiative_id)
            return 1
        if self.runtime is not None:
            try:
                self.evaluate_pending_deferred_work()
                self._fire_due_deferred_work_now(limit=MAX_PM_SUBTASKS)
            except Exception:
                LOGGER.debug(
                    "failed to advance PM deferred work during watchdog tick",
                    exc_info=True,
                )
        surfaced = 0
        per_subtask: list[tuple[object, DeferredWork | None, AgentTask | None]] = []
        for subtask in subtasks:
            deferred = self.store.get_deferred_work(subtask.deferred_id)
            task = None
            if deferred is not None and deferred.last_task_id:
                task = self.store.get_agent_task(deferred.last_task_id)
            per_subtask.append((subtask, deferred, task))

        for index, (subtask, deferred, task) in enumerate(per_subtask):
            if deferred is None or task is None or task.status != AgentTaskStatus.CANCELLED:
                continue
            if deferred.status != DeferredWorkStatus.CANCELLED:
                self.store.update_deferred_work_status(
                    deferred.deferred_id,
                    DeferredWorkStatus.CANCELLED,
                )
                deferred = self.store.get_deferred_work(deferred.deferred_id) or replace(
                    deferred,
                    status=DeferredWorkStatus.CANCELLED,
                    updated_at=utc_now(),
                )
                per_subtask[index] = (subtask, deferred, task)
            surfaced += int(
                self._surface_pm_blocker(
                    initiative,
                    local_id=subtask.local_id,
                    kind="cancelled_task",
                    message=(
                        f":warning: PM initiative `{initiative.initiative_id}` subtask "
                        f"`{subtask.local_id}` was cancelled before it completed. "
                        "I will not treat it as successful or start downstream work "
                        "until it is rerun or the PM plan is updated."
                    ),
                )
            )

        all_terminal = True
        all_done = True
        for _subtask, deferred, task in per_subtask:
            if deferred is None:
                all_done = False
                continue
            if deferred.status not in {DeferredWorkStatus.DONE, DeferredWorkStatus.CANCELLED}:
                all_terminal = False
                all_done = False
                continue
            if deferred.status == DeferredWorkStatus.CANCELLED:
                all_done = False
                continue
            if task is not None and task.status != AgentTaskStatus.DONE:
                all_terminal = False
                all_done = False
        if all_terminal:
            self._post_pm_initiative_recap(initiative, per_subtask, all_done=all_done)
            new_status = PmInitiativeStatus.DONE if all_done else PmInitiativeStatus.CANCELLED
            self.store.update_pm_initiative_status(initiative.initiative_id, new_status)
            if all_done:
                self._mark_pm_initiative_request_complete(initiative)
            else:
                self._clear_pm_initiative_request_status(initiative)
            self._post_or_update_pm_status_message(initiative.initiative_id)
            self._clear_pm_surfaced_keys(initiative.initiative_id)
            return 1
        now = utc_now()
        for subtask, deferred, task in per_subtask:
            if task is None or deferred is None:
                continue
            if task.status != AgentTaskStatus.ACTIVE:
                continue
            if not task.thread_ts:
                continue
            for pending in self.store.list_pending_slack_agent_requests():
                if pending["thread_channel_id"] != task.channel_id:
                    continue
                if pending["thread_ts"] != task.thread_ts:
                    continue
                created_raw = pending["created_at"]
                created_at = parse_timestamp(created_raw) if isinstance(created_raw, str) else None
                if created_at is None:
                    continue
                age = (now - created_at).total_seconds()
                if age < _PM_APPROVAL_BLOCKER_SECONDS:
                    continue
                method = pending["method"] or "approval"
                surfaced += int(
                    self._surface_pm_blocker(
                        initiative,
                        local_id=subtask.local_id,
                        kind=f"approval:{method}",
                        message=(
                            f":warning: PM initiative `{initiative.initiative_id}` subtask "
                            f"`{subtask.local_id}` is waiting on `{method}` approval/input for "
                            f"{int(age // 60)} min. @{self._handle_for_agent_id(task.agent_id)} "
                            "is blocked."
                        ),
                    )
                )
            updated_at = task.updated_at
            if updated_at is None:
                continue
            inactive_for = (now - updated_at).total_seconds()
            if inactive_for < _PM_STALLED_TASK_SECONDS:
                continue
            is_running = getattr(self.runtime, "is_task_running", None)
            if callable(is_running) and is_running(task.task_id):
                continue
            surfaced += int(
                self._surface_pm_blocker(
                    initiative,
                    local_id=subtask.local_id,
                    kind="stalled_task",
                    message=(
                        f":warning: PM initiative `{initiative.initiative_id}` subtask "
                        f"`{subtask.local_id}` has been ACTIVE for "
                        f"{int(inactive_for // 60)} min with no recent activity. "
                        f"@{self._handle_for_agent_id(task.agent_id)} may need a nudge."
                    ),
                )
            )
        return surfaced

    def _post_pm_initiative_recap(
        self,
        initiative: PmInitiative,
        per_subtask,
        *,
        all_done: bool,
    ) -> None:
        emoji = ":white_check_mark:" if all_done else ":no_entry:"
        status_word = "complete" if all_done else "cancelled"
        lines = [
            f"{emoji} PM initiative *{initiative.title}* (`{initiative.initiative_id}`) {status_word}.",
        ]
        for subtask, deferred, task in per_subtask:
            if deferred is None:
                state = "missing"
            elif deferred.status == DeferredWorkStatus.DONE and (
                task is None or task.status == AgentTaskStatus.DONE
            ):
                state = "done"
            elif deferred.status == DeferredWorkStatus.CANCELLED or (
                task is not None and task.status == AgentTaskStatus.CANCELLED
            ):
                state = "cancelled"
            else:
                state = task.status.value if task is not None else deferred.status.value
            lines.append(f"- `{subtask.local_id}` — {subtask.title}: {state}")
        thread = SlackThreadRef(initiative.channel_id, initiative.thread_ts, initiative.message_ts)
        try:
            self.gateway.post_thread_reply(thread, "\n".join(lines))
        except Exception as exc:
            if _slack_error_code(exc) in _PM_DEAD_THREAD_SLACK_ERRORS:
                LOGGER.warning(
                    "PM initiative %s recap thread unreachable (%s); skipping recap",
                    initiative.initiative_id,
                    _slack_error_code(exc),
                )
                return
            raise

    def _surface_pm_blocker(
        self,
        initiative: PmInitiative,
        *,
        local_id: str,
        kind: str,
        message: str,
    ) -> bool:
        key = _pm_blocker_setting_key(initiative.initiative_id, local_id, kind)
        if self.store.get_setting(key):
            return False
        thread = SlackThreadRef(initiative.channel_id, initiative.thread_ts, initiative.message_ts)
        try:
            mention_user = initiative.requested_by_slack_user
            text = message
            if mention_user and f"<@{mention_user}>" not in text:
                text = f"<@{mention_user}> {text}"
            self.gateway.post_thread_reply(thread, text)
        except Exception as exc:
            if _slack_error_code(exc) in _PM_DEAD_THREAD_SLACK_ERRORS:
                LOGGER.warning(
                    "PM initiative %s thread is unreachable (%s); cancelling",
                    initiative.initiative_id,
                    _slack_error_code(exc),
                )
                # Pin the dedup key so a watchdog tick can't loop on the same
                # blocker, and move the initiative to CANCELLED so subsequent
                # ticks skip it entirely.
                with suppress(Exception):
                    self.store.set_setting(key, utc_now().isoformat())
                with suppress(Exception):
                    self.store.update_pm_initiative_status(
                        initiative.initiative_id, PmInitiativeStatus.CANCELLED
                    )
                return False
            LOGGER.debug("failed to surface PM blocker", exc_info=True)
            return False
        self.store.set_setting(key, utc_now().isoformat())
        return True

    def _clear_pm_surfaced_keys(self, initiative_id: str) -> None:
        prefix = f"{_PM_BLOCKER_PREFIX}{initiative_id}."
        try:
            keys = list(self.store.list_settings(prefix=prefix).keys())
        except Exception:
            return
        for key in keys:
            try:
                self.store.delete_setting(key)
            except Exception:
                LOGGER.debug("failed to clear PM blocker setting %s", key, exc_info=True)

    def _handle_for_agent_id(self, agent_id: str | None) -> str:
        if not agent_id:
            return "agent"
        agent = self.store.get_team_agent(agent_id)
        if agent is None:
            return agent_id
        return agent.handle

    def evaluate_pending_deferred_work(self, deferred_id: str | None = None) -> int:
        promoted = 0
        promoted_channels: set[str] = set()
        if deferred_id is not None:
            row = self.store.get_deferred_work(deferred_id)
            rows = [row] if row is not None else []
        else:
            rows = self.store.list_waiting_deferred_work(limit=200)
        for row in rows:
            if row.status != DeferredWorkStatus.WAITING_DEPS:
                continue
            cancelled_dependency = self._pm_cancelled_dependency(row)
            if cancelled_dependency is not None:
                dep, upstream = cancelled_dependency
                subtask = self.store.get_pm_subtask_by_deferred_id(row.deferred_id)
                initiative = (
                    self.store.get_pm_initiative(subtask.initiative_id) if subtask else None
                )
                blocked_local_id = dep.local_id or upstream.deferred_id
                surfaced = False
                if subtask is not None and initiative is not None:
                    surfaced = self._surface_pm_blocker(
                        initiative,
                        local_id=subtask.local_id,
                        kind=f"cancelled_dependency:{blocked_local_id}",
                        message=(
                            f":warning: PM initiative `{initiative.initiative_id}` subtask "
                            f"`{subtask.local_id}` is waiting because dependency "
                            f"`{blocked_local_id}` was cancelled. I will not start downstream "
                            "work until that prerequisite is rerun or the PM plan is updated."
                        ),
                    )
                if surfaced:
                    self._refresh_pm_status_for_deferred(row.deferred_id)
                continue
            satisfied, _missing = self.store.evaluate_deferred_dependencies(row)
            if not satisfied:
                continue
            now = utc_now()
            if row.after_dep_delay_seconds:
                fire_at = now + timedelta(seconds=row.after_dep_delay_seconds)
            elif row.run_at is not None and row.run_at > now:
                fire_at = row.run_at
            else:
                fire_at = now
            updated = self.store.mark_deferred_work_ready(
                row.deferred_id,
                fire_at=fire_at,
                deps_satisfied_at=now,
            )
            if updated is None:
                continue
            promoted += 1
            self.gateway.post_thread_reply(
                SlackThreadRef(row.channel_id, row.thread_ts),
                _format_deferred_ready_message(updated),
            )
            self._refresh_pm_status_for_deferred(row.deferred_id)
            promoted_channels.add(row.channel_id)
        for channel_id in promoted_channels:
            self._refresh_existing_roster(channel_id)
        return promoted

    def _pm_cancelled_dependency(
        self,
        deferred: DeferredWork,
    ) -> tuple[WorkDependency, DeferredWork] | None:
        if self.store.get_pm_subtask_by_deferred_id(deferred.deferred_id) is None:
            return None
        for dep in deferred.depends_on:
            if dep.kind != WorkDependencyKind.SUBTASK or not dep.task_id:
                continue
            upstream = self.store.get_deferred_work(dep.task_id)
            if upstream is None:
                continue
            if upstream.status == DeferredWorkStatus.CANCELLED:
                return dep, upstream
            if upstream.last_task_id:
                task = self.store.get_agent_task(upstream.last_task_id)
                if task is not None and task.status == AgentTaskStatus.CANCELLED:
                    return dep, upstream
        return None

    def _fire_due_deferred_work_now(self, *, limit: int = 50) -> int:
        fired = 0
        for deferred in self.store.list_due_deferred_work(limit=limit):
            if deferred.last_task_id:
                task = self.store.get_agent_task(deferred.last_task_id)
                if task is not None and task.status == AgentTaskStatus.ACTIVE:
                    continue
            if self.fire_due_deferred_work(deferred):
                fired += 1
        return fired

    def fire_due_deferred_work(self, deferred: DeferredWork) -> bool:
        claimed = self.store.claim_deferred_work(deferred.deferred_id)
        if claimed is None:
            return True
        reserved_task = self._reserved_task_for_deferred(claimed)
        if reserved_task is not None:
            return self._fire_reserved_deferred_work(claimed, reserved_task)
        thread = SlackThreadRef(claimed.channel_id, claimed.thread_ts, claimed.message_ts)
        request = _work_request_from_deferred_work(claimed)
        author_agent = (
            self.store.get_team_agent(request.author_handle) if request.author_handle else None
        )
        if (
            request.assignment_mode == AssignmentMode.SPECIFIC
            and request.requested_handle
            and self.store.get_team_agent(request.requested_handle) is None
        ):
            self.store.update_deferred_work_status(
                claimed.deferred_id,
                DeferredWorkStatus.CANCELLED,
            )
            self.gateway.post_thread_reply(
                thread,
                (
                    f"Cancelled deferred task `{_shorten(claimed.prompt, 140)}` because "
                    f"@{request.requested_handle} is not an active agent."
                ),
            )
            self._refresh_pm_status_for_deferred(claimed.deferred_id)
            return True
        metadata = {
            "deferred_work_id": claimed.deferred_id,
            "deferred_work_fire_at": claimed.fire_at.isoformat() if claimed.fire_at else "",
        }
        result = assign_work_request(
            self.store,
            request,
            claimed.channel_id,
            requested_by_slack_user=claimed.requested_by_slack_user,
            author_agent=author_agent,
            extra_metadata=metadata,
            exclude_agent_ids=self._busy_agent_ids_for_assignment(
                ignore_deferred_id=claimed.deferred_id
            ),
        )
        if result is None:
            pending = self.store.create_pending_work_request(
                thread,
                request,
                requested_by_slack_user=claimed.requested_by_slack_user,
                author_agent=author_agent,
                extra_metadata=metadata,
                exclude_agent_ids=self._busy_agent_ids_for_assignment(
                    ignore_deferred_id=claimed.deferred_id
                ),
            )
            self.gateway.post_thread_reply(
                thread,
                (
                    f"Deferred task `{_shorten(claimed.prompt, 140)}` is ready, "
                    "but no matching agent "
                    "is available. I queued it and will start it when capacity opens."
                ),
            )
            self.store.complete_deferred_work(
                claimed.deferred_id,
                last_task_id=pending.pending_id,
            )
            self.refresh_or_post_roster(claimed.channel_id)
            self._refresh_pm_status_for_deferred(claimed.deferred_id)
            return True
        text = "Deferred task is ready now.\n\n" + format_agent_assignment(
            result.agent,
            result.request.prompt,
            claimed.requested_by_slack_user,
            dangerous_mode=_task_dangerous_mode(result.task),
        )
        blocks = _task_thread_blocks(result.task, result.agent)
        posted = self.gateway.post_thread_reply(
            thread,
            text,
            persona=result.agent,
            blocks=blocks,
            icon_url=self._agent_icon_url(result.agent),
        )
        self.store.update_agent_task_thread(result.task.task_id, thread.thread_ts, posted.ts)
        task = self.store.get_agent_task(result.task.task_id) or result.task
        if self.runtime:
            self.runtime.start_task(task, result.agent, thread)
        self.store.complete_deferred_work(
            claimed.deferred_id,
            last_task_id=result.task.task_id,
        )
        self.refresh_or_post_roster(claimed.channel_id)
        self._refresh_pm_status_for_deferred(claimed.deferred_id)
        return True

    def _reserved_task_for_deferred(self, deferred: DeferredWork) -> AgentTask | None:
        if not deferred.last_task_id or deferred.last_task_id.startswith("pending_"):
            return None
        task = self.store.get_agent_task(deferred.last_task_id)
        if task is None:
            return None
        if task.metadata.get("deferred_work_id") != deferred.deferred_id:
            return None
        if not isinstance(task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY), str):
            return None
        return task

    def _fire_reserved_deferred_work(self, deferred: DeferredWork, task: AgentTask) -> bool:
        thread = SlackThreadRef(
            task.channel_id or deferred.channel_id,
            task.thread_ts or deferred.thread_ts,
            task.parent_message_ts or deferred.message_ts,
        )
        agent = self.store.get_team_agent(task.agent_id)
        if agent is None:
            self.store.update_deferred_work_status(
                deferred.deferred_id,
                DeferredWorkStatus.CANCELLED,
            )
            self.gateway.post_thread_reply(
                thread,
                "Cancelled this reserved PM subtask because its assigned agent is no longer active.",
            )
            self._refresh_pm_status_for_deferred(deferred.deferred_id)
            return True
        if task.status in {AgentTaskStatus.DONE, AgentTaskStatus.CANCELLED}:
            self.store.update_deferred_work_status(
                deferred.deferred_id,
                DeferredWorkStatus.CANCELLED,
            )
            self._refresh_pm_status_for_deferred(deferred.deferred_id)
            return True
        task = self._task_with_pm_dependency_thread_context(deferred, task)
        self.gateway.post_thread_reply(
            thread,
            "Dependencies are satisfied. Starting this PM subtask now.",
            persona=agent,
            icon_url=self._agent_icon_url(agent),
        )
        started = self._start_runtime_task(task, agent, thread)
        if not started:
            latest = self.store.get_agent_task(task.task_id) or task
            if latest.status == AgentTaskStatus.CANCELLED:
                self.store.update_deferred_work_status(
                    deferred.deferred_id,
                    DeferredWorkStatus.CANCELLED,
                )
            else:
                self.store.release_deferred_work(deferred.deferred_id)
            self._refresh_pm_status_for_deferred(deferred.deferred_id)
            return True
        latest = self.store.get_agent_task(task.task_id) or task
        self._refresh_task_thread_header(latest, agent)
        self.store.complete_deferred_work(
            deferred.deferred_id,
            last_task_id=task.task_id,
        )
        self._refresh_pm_status_for_deferred(deferred.deferred_id)
        return True

    def _task_with_pm_dependency_thread_context(
        self,
        deferred: DeferredWork,
        task: AgentTask,
    ) -> AgentTask:
        dependency_context = self._pm_dependency_thread_context(deferred)
        if not dependency_context:
            return task
        latest = self.store.get_agent_task(task.task_id) or task
        existing = latest.metadata.get("thread_context")
        if isinstance(existing, str) and dependency_context in existing:
            return latest
        metadata = dict(latest.metadata)
        if isinstance(existing, str) and existing.strip():
            thread_context = f"{existing.strip()}\n\n{dependency_context}"
        else:
            thread_context = dependency_context
        metadata["thread_context"] = thread_context[-20000:]
        updated = replace(latest, metadata=metadata, updated_at=utc_now())
        self.store.upsert_agent_task(updated)
        return updated

    def _pm_dependency_thread_context(self, deferred: DeferredWork) -> str | None:
        sections: list[str] = []
        for dep in deferred.depends_on:
            if dep.kind != WorkDependencyKind.SUBTASK or not dep.task_id:
                continue
            upstream = self.store.get_deferred_work(dep.task_id)
            if upstream is None:
                continue
            context = self._thread_context(upstream.channel_id, upstream.thread_ts)
            if not context and upstream.message_ts and upstream.message_ts != upstream.thread_ts:
                context = self._thread_context(upstream.channel_id, upstream.message_ts)
            if not context:
                continue
            subtask = self.store.get_pm_subtask_by_deferred_id(upstream.deferred_id)
            local_id = dep.local_id or (subtask.local_id if subtask else upstream.deferred_id)
            title = subtask.title if subtask else upstream.description or upstream.prompt
            sections.append(f"PM dependency subtask `{local_id}` — {title}\n{context}")
        if not sections:
            return None
        return "\n\n".join(sections)[-20000:]

    def fire_due_scheduled_timer(self, timer: ScheduledTimer) -> bool:
        agent = self.store.get_team_agent(timer.agent_id)
        if agent is None:
            self.store.update_scheduled_timer_status(
                timer.timer_id,
                ScheduledTimerStatus.CANCELLED,
            )
            return True
        previous_task = self.store.get_agent_task(timer.task_id)
        if previous_task is None or previous_task.status == AgentTaskStatus.CANCELLED:
            self.store.update_scheduled_timer_status(
                timer.timer_id,
                ScheduledTimerStatus.CANCELLED,
            )
            return True
        active_task = self.store.active_task_for_agent(agent.agent_id)
        if active_task is not None and active_task.task_id != previous_task.task_id:
            return False
        is_running = getattr(self.runtime, "is_task_running", None)
        if callable(is_running) and is_running(previous_task.task_id):
            return False
        claimed = self.store.claim_scheduled_timer(timer.timer_id)
        if claimed is None:
            return True
        request = WorkRequest(
            prompt=claimed.prompt,
            assignment_mode=AssignmentMode.SPECIFIC,
            requested_handle=agent.handle,
        )
        thread = SlackThreadRef(
            claimed.channel_id,
            claimed.thread_ts,
            claimed.parent_message_ts,
        )
        started = self._continue_same_thread_agent_task(
            request,
            previous_task,
            agent,
            thread,
            requested_by_slack_user=previous_task.requested_by_slack_user,
            try_live_send=False,
        )
        if started:
            self.store.update_scheduled_timer_status(
                claimed.timer_id,
                ScheduledTimerStatus.FIRED,
            )
            self.refresh_or_post_roster(claimed.channel_id)
            return True
        self.store.update_scheduled_timer_status(
            claimed.timer_id,
            ScheduledTimerStatus.PENDING,
        )
        return False

    def _complete_task_thread(self, channel_id: str, thread_ts: str | None) -> bool:
        if not thread_ts:
            return False
        thread = SlackThreadRef(channel_id, thread_ts)
        completed = 0
        completed_tasks: list[AgentTask] = []
        for thread_task in self.store.list_agent_tasks(include_done=True):
            if (
                thread_task.channel_id == channel_id
                and thread_task.thread_ts == thread_ts
                and thread_task.status in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}
            ):
                if self.runtime:
                    self.runtime.stop_task(thread_task.task_id, AgentTaskStatus.DONE)
                    latest = self.store.get_agent_task(thread_task.task_id)
                    if latest is not None and latest.status not in {
                        AgentTaskStatus.DONE,
                        AgentTaskStatus.CANCELLED,
                    }:
                        self.store.update_agent_task_status(
                            thread_task.task_id,
                            AgentTaskStatus.DONE,
                        )
                else:
                    self.store.update_agent_task_status(thread_task.task_id, AgentTaskStatus.DONE)
                self._mark_task_complete(thread_task, thread, include_thread=True)
                completed += 1
                completed_tasks.append(
                    self.store.get_agent_task(thread_task.task_id) or thread_task
                )

        cancelled = 0
        for pending in self.store.list_pending_work_requests(channel_id=channel_id, limit=500):
            if pending.thread_ts != thread_ts:
                continue
            self.store.update_pending_work_request_status(
                pending.pending_id,
                PendingWorkRequestStatus.CANCELLED,
            )
            cancelled += 1

        cancelled_timers = self.store.cancel_scheduled_timers_for_thread(channel_id, thread_ts)
        cancelled_scheduled_work = self.store.cancel_scheduled_work_for_thread(
            channel_id,
            thread_ts,
        )
        cancelled_deferred_work = self.store.cancel_deferred_work_for_thread(
            channel_id,
            thread_ts,
        )
        cancelled_pm_initiatives = self.store.cancel_pm_initiative_for_thread(
            channel_id,
            thread_ts,
        )
        if (
            not completed
            and not cancelled
            and not cancelled_timers
            and not cancelled_scheduled_work
            and not cancelled_deferred_work
            and not cancelled_pm_initiatives
        ):
            return False
        # Threads outside this one may have just satisfied a dependency.
        try:
            self.evaluate_pending_deferred_work()
            if self.runtime is not None:
                self._fire_due_deferred_work_now(limit=MAX_PM_SUBTASKS)
        except Exception:
            LOGGER.debug("failed to evaluate deferred work after thread completion", exc_info=True)
        self._resume_pending_work_requests(channel_id)
        for completed_task in completed_tasks:
            self._refresh_pm_status_for_task(completed_task)
        self.refresh_or_post_roster(channel_id)
        return True

    def _start_thread_followup(self, parent_task: AgentTask, event: dict, text: str, agent) -> bool:
        channel_id = event["channel"]
        thread_ts = event["thread_ts"]
        prompt = text.strip()
        if not prompt:
            return False
        request = WorkRequest(
            prompt=prompt,
            assignment_mode=AssignmentMode.SPECIFIC,
            requested_handle=agent.handle,
        )
        active_task = self.store.active_task_for_agent(agent.agent_id)
        if active_task is not None and active_task.task_id != parent_task.task_id:
            metadata = self._thread_task_metadata(parent_task, channel_id, thread_ts)
            metadata[ASSIGNMENT_PROMPT_METADATA_KEY] = _task_assignment_prompt(parent_task)
            metadata["request_message_ts"] = event.get("ts")
            self._post_assignment_unavailable(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread_ts),
                request,
                requested_by_slack_user=event.get("user"),
                extra_metadata=metadata,
            )
            return True
        return self._continue_same_thread_agent_task(
            request,
            parent_task,
            agent,
            SlackThreadRef(channel_id, thread_ts),
            requested_by_slack_user=event.get("user"),
            request_message_ts=event.get("ts"),
            try_live_send=False,
        )

    def _thread_task_metadata(
        self,
        parent_task: AgentTask,
        channel_id: str,
        thread_ts: str,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "parent_task_id": parent_task.task_id,
            "parent_agent_id": parent_task.agent_id,
            ORIGINAL_TASK_METADATA_KEY: _task_original_prompt(parent_task),
        }
        if parent_task.metadata.get("cwd"):
            metadata["cwd"] = parent_task.metadata["cwd"]
        parent_pr_urls = pr_urls_from_metadata(parent_task.metadata)
        if parent_pr_urls:
            metadata[PR_URL_METADATA_KEY] = parent_pr_urls[0]
            metadata[PR_URLS_METADATA_KEY] = list(parent_pr_urls)
        context = self._thread_context(channel_id, thread_ts)
        prompt_context = f"Original task: {_task_original_prompt(parent_task)}"
        if context:
            metadata["thread_context"] = f"{prompt_context}\n{context}"
        else:
            metadata["thread_context"] = prompt_context
        return metadata

    def _external_thread_task_metadata(
        self,
        session,
        channel_id: str,
        thread_ts: str,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "external_session_provider": session.provider.value,
            "external_session_id": session.session_id,
        }
        if session.cwd:
            metadata["cwd"] = str(session.cwd)
        context = self._thread_context(channel_id, thread_ts)
        if context:
            metadata["thread_context"] = context
        return metadata

    def _with_linked_thread_context(
        self,
        metadata: dict[str, object],
        text: str,
        *,
        current_thread: SlackThreadRef | None,
    ) -> dict[str, object]:
        linked_context = self._linked_thread_context(text, current_thread=current_thread)
        if not linked_context:
            return metadata
        updated = dict(metadata)
        existing = updated.get("thread_context")
        if isinstance(existing, str) and existing.strip():
            updated["thread_context"] = f"{existing.strip()}\n\n{linked_context}"
        else:
            updated["thread_context"] = linked_context
        return updated

    def _linked_thread_context(
        self,
        text: str,
        *,
        current_thread: SlackThreadRef | None,
    ) -> str | None:
        ref = self._resolve_linked_thread_ref(text)
        if ref is None or not ref.thread_ts:
            return None
        if (
            current_thread is not None
            and ref.channel_id == current_thread.channel_id
            and ref.thread_ts == current_thread.thread_ts
        ):
            return None
        context = self._thread_context(ref.channel_id, ref.thread_ts)
        if not context and ref.message_ts and ref.message_ts != ref.thread_ts:
            context = self._thread_context(ref.channel_id, ref.message_ts)
        if not context:
            return None
        label = "Linked Slack thread from task prompt"
        if ref.permalink:
            label = f"{label}: {ref.permalink}"
        return f"{label}\n{context}"

    def _request_thread_anchor(
        self,
        event: dict,
        channel_id: str,
        text: str,
    ) -> SlackThreadRef:
        message_ts = event.get("ts")
        user_thread_ts = event.get("thread_ts")
        if user_thread_ts:
            current_thread = SlackThreadRef(
                channel_id=channel_id,
                thread_ts=user_thread_ts,
                message_ts=message_ts,
            )
            routed = self._linked_thread_route_target(text, current_thread)
            if routed is not None:
                return routed
            return SlackThreadRef(
                channel_id=channel_id,
                thread_ts=user_thread_ts,
                message_ts=message_ts,
            )
        current_thread = SlackThreadRef(
            channel_id=channel_id,
            thread_ts=message_ts,
            message_ts=message_ts,
        )
        routed = self._linked_thread_route_target(text, current_thread)
        if routed is not None:
            return routed
        return current_thread

    def _linked_thread_route_target(
        self,
        text: str,
        current_thread: SlackThreadRef,
    ) -> SlackThreadRef | None:
        if not LINKED_THREAD_ROUTE_INTENT_RE.search(text):
            return None
        ref = self._resolve_linked_thread_ref(text)
        if ref is None or not ref.thread_ts:
            return None
        if ref.channel_id != current_thread.channel_id:
            return None
        if ref.thread_ts == current_thread.thread_ts:
            return None
        return SlackThreadRef(
            channel_id=ref.channel_id,
            thread_ts=ref.thread_ts,
            message_ts=current_thread.message_ts,
            permalink=ref.permalink,
        )

    def _resolve_linked_thread_ref(self, text: str) -> SlackThreadRef | None:
        ref = parse_thread_ref(text)
        if ref is None or not ref.thread_ts:
            return ref
        if ref.thread_ts != ref.message_ts:
            return ref
        try:
            messages = self.gateway.thread_messages(ref.channel_id, ref.thread_ts, limit=1)
        except Exception:
            LOGGER.debug("failed to resolve linked Slack thread parent", exc_info=True)
            return ref
        for message in messages:
            candidate = message.get("thread_ts") or message.get("ts")
            if isinstance(candidate, str) and candidate and candidate != ref.thread_ts:
                return replace(ref, thread_ts=candidate)
        return ref

    def _thread_context(self, channel_id: str, thread_ts: str) -> str | None:
        try:
            messages = self.gateway.thread_messages(channel_id, thread_ts, limit=20)
        except Exception:
            LOGGER.debug("failed to fetch Slack thread context", exc_info=True)
            return None
        lines: list[str] = []
        for message in messages[-12:]:
            text = self._thread_context_text(message.get("text") or "").strip()
            if not text:
                continue
            if is_internal_task_notification_text(text):
                continue
            author = self._thread_context_author(message)
            lines.append(f"{author}: {text}")
        context = "\n".join(lines)
        return context[-6000:] if context else None

    def _thread_context_author(self, message: dict) -> str:
        username = message.get("username")
        if isinstance(username, str) and username.strip():
            return username.strip()
        profile_name = _profile_display_name(message.get("user_profile"))
        if profile_name:
            return profile_name
        user_id = message.get("user")
        if isinstance(user_id, str) and user_id.strip():
            return self._display_name_for_slack_user(user_id) or "Slack user"
        return "Slack"

    def _thread_context_text(self, text: str) -> str:
        return replace_slack_user_ids(text, self._display_name_for_slack_user)

    def _display_name_for_slack_user(self, user_id: str) -> str | None:
        configured = self.store.get_setting(_human_user_display_name_key(user_id))
        if configured and configured.strip():
            return configured.strip()
        remembered_id = self.store.get_setting(SETTING_HUMAN_USER_ID)
        if remembered_id == user_id:
            remembered_name = self.store.get_setting(HUMAN_DISPLAY_NAME_SETTING)
            if remembered_name and remembered_name.strip():
                return remembered_name.strip()
        return None

    def _record_dependency_if_requested(
        self,
        task_id: str,
        event: dict,
        text: str,
        agent,
    ) -> bool:
        if not is_dependency_intent(text):
            return False
        ref = parse_thread_ref(text)
        if ref is None:
            return False
        self.store.add_dependency(
            SessionDependency(
                blocked_session_id=task_id,
                blocking_thread=ref,
                created_by_slack_user=event.get("user"),
                reason=text,
            )
        )
        self.gateway.post_thread_reply(
            SlackThreadRef(event["channel"], event["thread_ts"]),
            "Recorded that dependency. I will keep this task marked as waiting on that thread.",
            persona=agent,
        )
        return True

    def _mark_message_acknowledged(self, channel_id: str, message_ts: str | None) -> None:
        self._set_task_status_reaction(channel_id, message_ts, TASK_REACTION_ACKNOWLEDGED)

    def _mark_message_in_progress(self, channel_id: str, message_ts: str | None) -> None:
        self._set_task_status_reaction(channel_id, message_ts, TASK_REACTION_IN_PROGRESS)

    def _mark_task_complete(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
        *,
        include_thread: bool = False,
    ) -> None:
        task = self.store.get_agent_task(task.task_id) or task
        self._remove_task_action_buttons_if_resolved(task)
        task = self._clear_idle_release_prompt_on_close(task, thread)
        if include_thread and task.thread_ts:
            self._mark_message_complete(thread.channel_id, task.thread_ts)
        for message_ts in _task_request_message_ts_values(task):
            if include_thread and message_ts == task.thread_ts:
                continue
            self._clear_message_status_reactions(thread.channel_id, message_ts)
        self._reconcile_pending_thread_reactions(task, thread)

    def _mark_pm_initiative_request_in_progress(
        self,
        initiative: PmInitiative,
        task: AgentTask | None = None,
    ) -> None:
        for message_ts in self._pm_initiative_request_message_ts_values(initiative, task):
            self._mark_message_in_progress(initiative.channel_id, message_ts)

    def _mark_pm_initiative_request_complete(self, initiative: PmInitiative) -> None:
        for message_ts in self._pm_initiative_request_message_ts_values(initiative):
            self._mark_message_complete(initiative.channel_id, message_ts)

    def _clear_pm_initiative_request_status(self, initiative: PmInitiative) -> None:
        for message_ts in self._pm_initiative_request_message_ts_values(initiative):
            self._clear_message_status_reactions(initiative.channel_id, message_ts)

    def _pm_initiative_request_message_ts_values(
        self,
        initiative: PmInitiative,
        task: AgentTask | None = None,
    ) -> tuple[str, ...]:
        if task is None and initiative.pm_task_id:
            task = self.store.get_agent_task(initiative.pm_task_id)
        values: list[str] = []
        if task is not None:
            values.extend(_task_request_message_ts_values(task))
        if initiative.message_ts:
            values.append(initiative.message_ts)
        return tuple(dict.fromkeys(value for value in values if value))

    def _clear_idle_release_prompt_on_close(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
    ) -> AgentTask:
        current = self.store.get_agent_task(task.task_id) or task
        message_ts = current.metadata.get(IDLE_RELEASE_PROMPT_MESSAGE_TS_METADATA_KEY)
        if not isinstance(message_ts, str) or not message_ts:
            return current
        channel_id = thread.channel_id or current.channel_id
        if channel_id:
            try:
                self.gateway.update_message(
                    channel_id,
                    message_ts,
                    " ",
                    blocks=build_idle_release_closed_blocks(current),
                )
            except Exception:
                LOGGER.debug(
                    "failed to clear idle release prompt on task close",
                    exc_info=True,
                )
        metadata = dict(current.metadata)
        metadata.pop(IDLE_RELEASE_PROMPT_MESSAGE_TS_METADATA_KEY, None)
        if metadata == current.metadata:
            return current
        updated = replace(current, metadata=metadata, updated_at=utc_now())
        self.store.upsert_agent_task(updated)
        return self.store.get_agent_task(current.task_id) or updated

    def _reconcile_pending_thread_reactions(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
    ) -> None:
        # Backstop: surface and clear thread messages that the bot acknowledged
        # or marked queued/in-progress but never finalized. Otherwise the user
        # sees status reactions lingering forever after the task closes.
        if not thread.thread_ts:
            return
        lookup = getattr(self.gateway, "bot_user_id", None)
        bot_user_id = lookup() if callable(lookup) else None
        if not bot_user_id:
            return
        handled = set(_task_request_message_ts_values(task))
        if task.thread_ts:
            handled.add(task.thread_ts)
        try:
            messages = self.gateway.thread_messages(
                thread.channel_id,
                thread.thread_ts,
                limit=200,
            )
        except Exception:
            LOGGER.debug("failed to fetch thread for pending-reaction sweep", exc_info=True)
            return
        pending_reactions = {
            TASK_REACTION_ACKNOWLEDGED,
            TASK_REACTION_QUEUED,
            TASK_REACTION_IN_PROGRESS,
        }
        pending: list[str] = []
        for message in messages:
            message_ts = message.get("ts")
            if not message_ts or message_ts in handled:
                continue
            for reaction in message.get("reactions") or []:
                if reaction.get("name") in pending_reactions and bot_user_id in (
                    reaction.get("users") or []
                ):
                    pending.append(message_ts)
                    break
        if not pending:
            return
        links: list[str] = []
        for message_ts in pending:
            try:
                url = self.gateway.permalink(thread.channel_id, message_ts)
            except Exception:
                url = None
            links.append(f"- {url}" if url else f"- `{message_ts}`")
        try:
            self.gateway.post_thread_reply(
                thread,
                (
                    "Closing this task, but these replies were acknowledged "
                    "and never marked complete — clearing their status "
                    "reactions:\n" + "\n".join(links)
                ),
            )
        except Exception:
            LOGGER.debug("failed to post pending-reaction warning", exc_info=True)
        for message_ts in pending:
            self._clear_message_status_reactions(thread.channel_id, message_ts)

    def _clear_task_request_status_reactions(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
    ) -> None:
        task = self.store.get_agent_task(task.task_id) or task
        for message_ts in _task_request_message_ts_values(task):
            self._clear_message_status_reactions(thread.channel_id, message_ts)

    def _clear_answered_request_status_reactions(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
    ) -> None:
        task = self.store.get_agent_task(task.task_id) or task
        for message_ts in _task_request_message_ts_values(task):
            if message_ts == task.thread_ts and not (
                _is_subtask(task) or _is_external_thread_helper_task(task)
            ):
                continue
            self._clear_message_status_reactions(thread.channel_id, message_ts)

    def _clear_completed_run_pending_reactions(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
    ) -> None:
        task = self.store.get_agent_task(task.task_id) or task
        if self._keeps_request_status_in_progress(task, thread):
            return
        request_message_ts_values: list[str] = []
        request_message_ts = task.metadata.get("request_message_ts")
        if isinstance(request_message_ts, str) and request_message_ts:
            request_message_ts_values.append(request_message_ts)
        history = task.metadata.get("request_message_ts_history")
        if isinstance(history, list):
            request_message_ts_values.extend(item for item in history if isinstance(item, str))
        for message_ts in dict.fromkeys(request_message_ts_values):
            self._clear_pending_message_status_reactions(thread.channel_id, message_ts)

    def _keeps_request_status_in_progress(self, task: AgentTask, thread: SlackThreadRef) -> bool:
        if isinstance(task.metadata.get("delegate_to_agent_id"), str) or isinstance(
            task.metadata.get("delegate_prompt"), str
        ):
            return True
        for thread_task in self.store.list_thread_agent_tasks(thread.channel_id, thread.thread_ts):
            if thread_task.task_id == task.task_id:
                continue
            if thread_task.status in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}:
                return True
        return False

    def _mark_message_complete(self, channel_id: str, message_ts: str | None) -> None:
        self._set_task_status_reaction(channel_id, message_ts, TASK_REACTION_DONE)

    def _set_task_status_reaction(
        self,
        channel_id: str,
        message_ts: str | None,
        reaction_name: str,
    ) -> None:
        if not message_ts:
            return
        current_reaction = self.store.get_setting(
            _message_status_reaction_setting_key(channel_id, message_ts)
        )
        if current_reaction == reaction_name:
            return
        previous_reaction = current_reaction if current_reaction in TASK_STATUS_REACTIONS else None
        try:
            self.gateway.add_reaction(channel_id, message_ts, reaction_name)
            self.store.set_setting(
                _message_status_reaction_setting_key(channel_id, message_ts),
                reaction_name,
            )
        except Exception:
            LOGGER.debug("failed to add Slack task status reaction", exc_info=True)
            return
        if previous_reaction:
            self._remove_status_reaction(channel_id, message_ts, previous_reaction)

    def _clear_message_status_reactions(
        self,
        channel_id: str,
        message_ts: str | None,
        *,
        except_reaction: str | None = None,
        known_reaction: str | None = None,
    ) -> None:
        if not message_ts:
            return
        remove_reaction = getattr(self.gateway, "remove_reaction", None)
        if not callable(remove_reaction):
            return
        status_key = _message_status_reaction_setting_key(channel_id, message_ts)
        stored_reaction = known_reaction or self.store.get_setting(status_key)
        if stored_reaction in TASK_STATUS_REACTIONS:
            reactions = (stored_reaction,)
        else:
            reactions = TASK_STATUS_REACTIONS
        for reaction in reactions:
            if reaction == except_reaction:
                continue
            try:
                remove_reaction(channel_id, message_ts, reaction)
            except Exception:
                LOGGER.debug("failed to remove stale Slack task reaction", exc_info=True)
        if except_reaction and stored_reaction == except_reaction:
            return
        self.store.delete_setting(status_key)

    def _remove_status_reaction(
        self,
        channel_id: str,
        message_ts: str,
        reaction: str,
    ) -> None:
        remove_reaction = getattr(self.gateway, "remove_reaction", None)
        if not callable(remove_reaction):
            return
        try:
            remove_reaction(channel_id, message_ts, reaction)
        except Exception:
            LOGGER.debug("failed to remove stale Slack task reaction", exc_info=True)

    def _clear_pending_message_status_reactions(
        self,
        channel_id: str,
        message_ts: str | None,
    ) -> None:
        if not message_ts:
            return
        remove_reaction = getattr(self.gateway, "remove_reaction", None)
        if callable(remove_reaction):
            for reaction in (
                TASK_REACTION_ACKNOWLEDGED,
                TASK_REACTION_QUEUED,
                TASK_REACTION_IN_PROGRESS,
            ):
                try:
                    remove_reaction(channel_id, message_ts, reaction)
                except Exception:
                    LOGGER.debug("failed to remove pending Slack task reaction", exc_info=True)

    def _remove_task_action_buttons_if_resolved(self, task: AgentTask) -> None:
        if task.status not in {AgentTaskStatus.DONE, AgentTaskStatus.CANCELLED}:
            return
        if not task.parent_message_ts:
            return
        agent = self.store.get_team_agent(task.agent_id, include_fired=True)
        if agent is None:
            return
        try:
            self.gateway.update_message(
                task.channel_id,
                task.parent_message_ts,
                format_agent_assignment(
                    agent,
                    _task_roster_summary(task),
                    task.requested_by_slack_user,
                    dangerous_mode=_task_dangerous_mode(task),
                ),
                blocks=build_task_thread_blocks(task, agent, include_actions=False),
            )
        except Exception:
            LOGGER.debug("failed to remove resolved task action buttons", exc_info=True)

    def _restore_task_action_buttons_if_active(self, task: AgentTask) -> None:
        if not _task_should_show_action_buttons(task):
            return
        if not task.parent_message_ts:
            return
        agent = self.store.get_team_agent(task.agent_id, include_fired=True)
        if agent is None:
            return
        try:
            self.gateway.update_message(
                task.channel_id,
                task.parent_message_ts,
                format_agent_assignment(
                    agent,
                    _task_roster_summary(task),
                    task.requested_by_slack_user,
                    dangerous_mode=_task_dangerous_mode(task),
                ),
                blocks=build_task_thread_blocks(task, agent, include_actions=True),
            )
        except Exception:
            LOGGER.debug("failed to restore active task action buttons", exc_info=True)

    def _agent_icon_url(self, agent) -> str | None:
        base_url = (
            self.store.get_setting(SETTING_AGENT_AVATAR_BASE_URL)
            or os.environ.get("SLACKGENTIC_AGENT_AVATAR_BASE_URL")
            or DEFAULT_AGENT_AVATAR_BASE_URL
        ).strip()
        if base_url.lower() in DISABLED_AVATAR_BASE_VALUES:
            return None
        return f"{base_url.rstrip('/')}/{agent.avatar_slug}.png"

    def _is_agent_channel(self, channel_id: str) -> bool:
        configured = self._configured_agent_channel_id()
        return configured is None or configured == channel_id

    def _remember_human_user(self, user_id: str | None, profile: object = None) -> None:
        if not user_id:
            return
        self.store.set_setting(SETTING_HUMAN_USER_ID, user_id)
        display_name = _profile_display_name(profile)
        image_url = _profile_image_url(profile)
        if display_name is None or image_url is None:
            try:
                slack_profile = self.gateway.user_profile(user_id)
            except Exception:
                LOGGER.debug("failed to fetch Slack user display name", exc_info=True)
                slack_profile = None
            if slack_profile:
                display_name = display_name or slack_profile.display_name
                image_url = image_url or slack_profile.image_url
        if display_name:
            self.store.set_setting(HUMAN_DISPLAY_NAME_SETTING, display_name)
            self.store.set_setting(_human_user_display_name_key(user_id), display_name)
        if image_url:
            self.store.set_setting(HUMAN_IMAGE_URL_SETTING, image_url)

    def _slack_message_processed(self, channel_id: str, message_ts: str | None) -> bool:
        if not message_ts:
            return False
        return bool(self.store.get_setting(_slack_message_processed_key(channel_id, message_ts)))

    def _processed_slack_message_needs_backfill(self, event: dict) -> bool:
        channel_id = event.get("channel")
        if not isinstance(channel_id, str) or not channel_id:
            return False
        message_ts = event.get("ts")
        if not isinstance(message_ts, str) or not message_ts:
            return False
        if not _event_has_slackgentic_progress_reaction(event):
            return False
        if self._has_external_session_delivery_for_message(channel_id, message_ts):
            return False
        if self._external_session_for_backfilled_event(event) is not None:
            return True
        return not self._has_work_for_request_message(channel_id, message_ts)

    def _external_session_for_backfilled_event(self, event: dict):
        channel_id = event.get("channel")
        message_ts = event.get("ts")
        thread_ts = event.get("thread_ts")
        if (
            not isinstance(channel_id, str)
            or not channel_id
            or not isinstance(message_ts, str)
            or not message_ts
            or not isinstance(thread_ts, str)
            or not thread_ts
            or thread_ts == message_ts
        ):
            return None
        return self.store.get_session_for_slack_thread(self.team_id, channel_id, thread_ts)

    def _has_task_for_request_message(self, channel_id: str, message_ts: str) -> bool:
        for task in self.store.list_agent_tasks(include_done=True):
            if task.channel_id != channel_id:
                continue
            if task.thread_ts == message_ts:
                return True
            if task.parent_message_ts == message_ts:
                return True
            if _metadata_has_message_ts(task.metadata, message_ts):
                return True
            if message_ts in _active_thread_followup_message_ts_values(task.metadata):
                return True
        return False

    def _has_work_for_request_message(self, channel_id: str, message_ts: str) -> bool:
        if self._has_task_for_request_message(channel_id, message_ts):
            return True
        if self._has_external_session_delivery_for_message(channel_id, message_ts):
            return True
        for pending in self.store.list_pending_work_requests(limit=500):
            if pending.channel_id != channel_id:
                continue
            if pending.thread_ts == message_ts:
                return True
            if pending.message_ts == message_ts:
                return True
            if _metadata_has_message_ts(pending.extra_metadata, message_ts):
                return True
        return False

    def _clear_slack_message_processed(self, channel_id: str, message_ts: str | None) -> None:
        if not message_ts:
            return
        self.store.delete_setting(_slack_message_processed_key(channel_id, message_ts))

    def _mark_slack_message_processed(self, channel_id: str, message_ts: str | None) -> None:
        if not message_ts:
            return
        self.store.set_setting(
            _slack_message_processed_key(channel_id, message_ts),
            utc_now().isoformat(),
        )

    def _mark_external_session_message_delivered(
        self,
        channel_id: str,
        message_ts: str | None,
        session,
    ) -> None:
        if not message_ts:
            return
        self.store.set_setting(
            _external_session_delivery_setting_key(channel_id, message_ts),
            f"{session.provider.value}:{session.session_id}",
        )

    def _has_external_session_delivery_for_message(
        self,
        channel_id: str,
        message_ts: str,
    ) -> bool:
        return bool(
            self.store.get_setting(_external_session_delivery_setting_key(channel_id, message_ts))
        )


class ClaudePermissionAutoResolver:
    def __init__(
        self,
        store: Store,
        gateway: SlackGateway,
        poll_seconds: float = 1.0,
    ):
        self.store = store
        self.gateway = gateway
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reformatted_tokens: set[str] = set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="slackgentic-claude-permission-auto-resolver",
        )
        self._thread.start()

    def stop(self) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            return not self._thread.is_alive()
        return True

    def resolve_once(self) -> int:
        resolved = 0
        for row in self.store.list_pending_slack_agent_requests(CLAUDE_CHANNEL_PERMISSION_METHOD):
            if not _is_slackgentic_mcp_permission_request(row["params_json"]):
                self._reformat_once(row)
                continue
            self.store.resolve_slack_agent_request(row["token"], {"behavior": "allow"})
            resolved += 1
            if row["message_ts"]:
                try:
                    self.gateway.update_message(
                        row["thread_channel_id"],
                        row["message_ts"],
                        AUTO_ALLOWED_CLAUDE_PERMISSION_TEXT,
                        blocks=[],
                    )
                except Exception:
                    LOGGER.debug("failed to update auto-allowed Claude permission", exc_info=True)
        return resolved

    def _reformat_once(self, row) -> None:
        token = str(row["token"])
        if token in self._reformatted_tokens or not row["message_ts"]:
            return
        self._reformatted_tokens.add(token)
        try:
            text, blocks = render_persistent_agent_request(
                row,
                fallback_channel_id=row["thread_channel_id"],
            )
            self.gateway.update_message(
                row["thread_channel_id"],
                row["message_ts"],
                text,
                blocks=blocks,
            )
        except Exception:
            LOGGER.debug("failed to reformat Claude permission prompt", exc_info=True)

    def _run(self) -> None:
        backoff = LoopBackoff(base_seconds=self.poll_seconds, max_seconds=30.0)
        while not self._stop.is_set():
            try:
                self.resolve_once()
                backoff.reset()
            except Exception:
                log_loop_failure(
                    LOGGER,
                    "failed to auto-resolve Claude Slackgentic permission",
                    backoff,
                )
                if backoff.wait(self._stop):
                    break
            self._stop.wait(self.poll_seconds)


@dataclass(frozen=True)
class _SlackBackfillEventScan:
    events: list[dict]
    channel_history_ok: bool
    thread_history_ok: bool
    scanned_thread_ts: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SlackBackfillResult:
    recovered: int
    channel_history_ok: bool
    thread_history_ok: bool


class SlackMessageBackfill:
    def __init__(
        self,
        store: Store,
        gateway: SlackGateway,
        controller: SlackTeamController,
        *,
        team_id: str,
        poll_seconds: float = 5.0,
        sleep_gap_seconds: float = SLACK_BACKFILL_SLEEP_GAP_SECONDS,
        grace_seconds: float = SLACK_BACKFILL_GRACE_SECONDS,
        thread_poll_seconds: float = SLACK_BACKFILL_THREAD_POLL_SECONDS,
        thread_initial_lookback_seconds: float = SLACK_BACKFILL_THREAD_INITIAL_LOOKBACK_SECONDS,
        now: Callable[[], float] | None = None,
    ):
        self.store = store
        self.gateway = gateway
        self.controller = controller
        self.team_id = team_id
        self.poll_seconds = poll_seconds
        self.sleep_gap_seconds = sleep_gap_seconds
        self.grace_seconds = grace_seconds
        self.thread_poll_seconds = thread_poll_seconds
        self.thread_initial_lookback_seconds = thread_initial_lookback_seconds
        self.now = now or time.time
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_awake_unix: float | None = None
        self._last_thread_scan_unix: float | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.sync_once()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="slackgentic-slack-message-backfill",
        )
        self._thread.start()

    def stop(self) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            return not self._thread.is_alive()
        return True

    def sync_once(self) -> int:
        now = float(self.now())
        previous = self._last_awake_unix
        if previous is None:
            previous = _float_setting(self.store.get_setting(SETTING_SLACK_BACKFILL_LAST_AWAKE))
        recovered = 0
        if previous is not None:
            include_threads = now - previous >= self.sleep_gap_seconds
            oldest_unix = max(0.0, previous - self.grace_seconds)
            thread_oldest_unix: float | None = None
            if not include_threads:
                last_thread_scan = self._last_thread_scan_unix
                if last_thread_scan is None:
                    last_thread_scan = _float_setting(
                        self.store.get_setting(SETTING_SLACK_BACKFILL_LAST_THREAD_SCAN)
                    )
                if self._thread_scan_due(now, last_thread_scan):
                    include_threads = True
                    if last_thread_scan is None:
                        thread_oldest_unix = max(
                            0.0,
                            now - self.thread_initial_lookback_seconds,
                        )
                    else:
                        thread_oldest_unix = max(0.0, last_thread_scan - self.grace_seconds)
            elif include_threads:
                last_thread_scan = self._last_thread_scan_unix
                if last_thread_scan is None:
                    last_thread_scan = _float_setting(
                        self.store.get_setting(SETTING_SLACK_BACKFILL_LAST_THREAD_SCAN)
                    )
                if last_thread_scan is not None:
                    thread_oldest_unix = max(0.0, last_thread_scan - self.grace_seconds)
            result = self._recover_since_result(
                _slack_ts_from_unix(oldest_unix),
                include_threads=include_threads,
                thread_oldest=(
                    _slack_ts_from_unix(thread_oldest_unix)
                    if thread_oldest_unix is not None
                    else None
                ),
            )
            recovered = result.recovered
            if result.channel_history_ok:
                self._record_last_awake(now)
            if include_threads and result.thread_history_ok:
                self._record_last_thread_scan(now)
            return recovered
        self._record_last_awake(now)
        return recovered

    def recover_since(self, oldest: str, *, include_threads: bool = True) -> int:
        return self._recover_since_result(oldest, include_threads=include_threads).recovered

    def _recover_since_result(
        self,
        oldest: str,
        *,
        include_threads: bool = True,
        thread_oldest: str | None = None,
    ) -> _SlackBackfillResult:
        channel_id = self.controller._configured_agent_channel_id()
        if not channel_id:
            return _SlackBackfillResult(
                recovered=0,
                channel_history_ok=True,
                thread_history_ok=True,
            )
        scan = self._events_since(
            channel_id,
            oldest,
            include_threads=include_threads,
            thread_oldest=thread_oldest,
        )
        if not scan.channel_history_ok:
            return _SlackBackfillResult(
                recovered=0,
                channel_history_ok=False,
                thread_history_ok=False,
            )
        recovered = 0
        for event in sorted(scan.events, key=lambda item: _slack_ts_sort_key(item.get("ts"))):
            event_channel_id, message_ts = self.controller._recoverable_user_message_ref(event)
            if event_channel_id is None or message_ts is None:
                continue
            if self.controller._slack_message_processed(event_channel_id, message_ts):
                if not self.controller._processed_slack_message_needs_backfill(event):
                    continue
                self.controller._clear_slack_message_processed(event_channel_id, message_ts)
            try:
                self.controller.handle_event({"event": event})
                recovered += 1
            except Exception:
                LOGGER.exception(
                    "failed to process backfilled Slack message %s:%s",
                    event_channel_id,
                    message_ts,
                )
        scan_finished_at = float(self.now())
        for thread_ts in scan.scanned_thread_ts:
            self._record_thread_scan(channel_id, thread_ts, scan_finished_at)
        return _SlackBackfillResult(
            recovered=recovered,
            channel_history_ok=scan.channel_history_ok,
            thread_history_ok=scan.thread_history_ok,
        )

    def _thread_scan_due(self, now: float, previous: float | None) -> bool:
        return self.thread_poll_seconds > 0 and (
            previous is None or now - previous >= self.thread_poll_seconds
        )

    def _record_last_awake(self, now: float) -> None:
        self._last_awake_unix = now
        self.store.set_setting(SETTING_SLACK_BACKFILL_LAST_AWAKE, f"{now:.6f}")

    def _record_last_thread_scan(self, now: float) -> None:
        self._last_thread_scan_unix = now
        self.store.set_setting(SETTING_SLACK_BACKFILL_LAST_THREAD_SCAN, f"{now:.6f}")

    def _record_thread_scan(self, channel_id: str, thread_ts: str, now: float) -> None:
        self.store.set_setting(
            _slack_backfill_thread_scan_key(channel_id, thread_ts),
            f"{now:.6f}",
        )

    def _events_since(
        self,
        channel_id: str,
        oldest: str,
        *,
        include_threads: bool = True,
        thread_oldest: str | None = None,
    ) -> _SlackBackfillEventScan:
        events_by_ts: dict[str, dict] = {}
        thread_ts_values = self._known_thread_ts(channel_id) if include_threads else set()
        try:
            channel_messages = self.gateway.channel_messages(
                channel_id,
                oldest=oldest,
                limit=SLACK_BACKFILL_FETCH_LIMIT,
            )
        except Exception:
            LOGGER.exception("failed to fetch Slack channel history for backfill")
            return _SlackBackfillEventScan(
                events=[],
                channel_history_ok=False,
                thread_history_ok=False,
            )
        for message in channel_messages:
            event = _history_message_event(channel_id, message)
            message_ts = event.get("ts")
            if isinstance(message_ts, str):
                events_by_ts[message_ts] = event
            thread_ts = event.get("thread_ts")
            if isinstance(thread_ts, str) and thread_ts:
                thread_ts_values.add(thread_ts)
            elif isinstance(message_ts, str) and message.get("reply_count"):
                thread_ts_values.add(message_ts)

        if not include_threads:
            return _SlackBackfillEventScan(
                events=list(events_by_ts.values()),
                channel_history_ok=True,
                thread_history_ok=True,
            )

        thread_history_ok = True
        scanned_thread_ts: list[str] = []
        base_thread_oldest = thread_oldest or oldest
        for thread_ts in self._ordered_thread_ts(channel_id, thread_ts_values):
            scan_oldest = self._oldest_for_thread_scan(
                channel_id,
                thread_ts,
                base_thread_oldest,
            )
            try:
                thread_messages = self.gateway.thread_messages(
                    channel_id,
                    thread_ts,
                    oldest=scan_oldest,
                    limit=SLACK_BACKFILL_FETCH_LIMIT,
                )
            except Exception as exc:
                if _slack_error_code(exc) not in _PM_DEAD_THREAD_SLACK_ERRORS:
                    thread_history_ok = False
                LOGGER.debug(
                    "failed to fetch Slack thread history for backfill: %s:%s",
                    channel_id,
                    thread_ts,
                    exc_info=True,
                )
                if _slack_error_code(exc) == "ratelimited":
                    break
                continue
            scanned_thread_ts.append(thread_ts)
            for message in thread_messages:
                event = _history_message_event(channel_id, message, default_thread_ts=thread_ts)
                message_ts = event.get("ts")
                if isinstance(message_ts, str):
                    events_by_ts[message_ts] = event
        return _SlackBackfillEventScan(
            events=list(events_by_ts.values()),
            channel_history_ok=True,
            thread_history_ok=thread_history_ok,
            scanned_thread_ts=tuple(scanned_thread_ts),
        )

    def _oldest_for_thread_scan(
        self,
        channel_id: str,
        thread_ts: str,
        fallback_oldest: str,
    ) -> str:
        last_thread_scan = _float_setting(
            self.store.get_setting(_slack_backfill_thread_scan_key(channel_id, thread_ts))
        )
        if last_thread_scan is None:
            return fallback_oldest
        oldest_unix = max(0.0, last_thread_scan - self.grace_seconds)
        return _slack_ts_from_unix(oldest_unix)

    def _ordered_thread_ts(self, channel_id: str, thread_ts_values: set[str]) -> list[str]:
        return sorted(
            thread_ts_values,
            key=lambda thread_ts: self._thread_backfill_sort_key(channel_id, thread_ts),
        )

    def _thread_backfill_sort_key(self, channel_id: str, thread_ts: str) -> tuple[int, float]:
        try:
            session = self.store.get_session_for_slack_thread(self.team_id, channel_id, thread_ts)
        except sqlite3.Error:
            session = None
        if session is not None and self._tracked_external_session(session):
            last_seen = session.last_seen_at or session.started_at or utc_now()
            return (0, -last_seen.timestamp())
        return (1, -_slack_ts_sort_key(thread_ts))

    def _tracked_external_session(self, session: AgentSession) -> bool:
        suffix = f"{session.provider.value}.{session.session_id}"
        for prefix in (
            EXTERNAL_SESSION_AGENT_PREFIX,
            PENDING_EXTERNAL_SESSION_PREFIX,
            EXTERNAL_SESSION_LIVE_TARGET_PREFIX,
            EXTERNAL_SESSION_MISSING_TARGET_PREFIX,
        ):
            if self.store.get_setting(f"{prefix}{suffix}"):
                return True
        return False

    def _known_thread_ts(self, channel_id: str) -> set[str]:
        thread_ts_values: set[str] = set()
        try:
            tasks = [
                task
                for task in self.store.list_agent_tasks(include_done=True)
                if task.channel_id == channel_id and task.thread_ts
            ]
        except sqlite3.Error:
            return thread_ts_values
        tasks.sort(
            key=lambda task: (
                task.status not in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE},
                -(task.updated_at.timestamp()),
            )
        )
        for task in tasks[:SLACK_BACKFILL_KNOWN_THREAD_LIMIT]:
            if task.thread_ts:
                thread_ts_values.add(task.thread_ts)
        remaining = max(0, SLACK_BACKFILL_KNOWN_THREAD_LIMIT - len(thread_ts_values))
        try:
            sessions = sorted(
                self.store.list_sessions(),
                key=lambda session: (
                    session.status != SessionStatus.ACTIVE,
                    -((session.last_seen_at or utc_now()).timestamp()),
                ),
            )
        except sqlite3.Error:
            return thread_ts_values
        for session in sessions[:remaining]:
            try:
                thread = self.store.get_slack_thread_for_session(
                    session.provider,
                    session.session_id,
                    self.team_id,
                    channel_id,
                )
            except sqlite3.Error:
                # The store can transiently fail under daemon shutdown races;
                # skip this session rather than aborting the whole backfill.
                continue
            if thread and thread.thread_ts:
                thread_ts_values.add(thread.thread_ts)
        return thread_ts_values

    def _run(self) -> None:
        backoff = LoopBackoff(base_seconds=self.poll_seconds, max_seconds=60.0)
        while not self._stop.wait(self.poll_seconds):
            try:
                self.sync_once()
                backoff.reset()
            except Exception:
                log_loop_failure(LOGGER, "failed to backfill missed Slack messages", backoff)
                if backoff.wait(self._stop):
                    break


class ScheduledTimerRunner:
    def __init__(
        self,
        store: Store,
        controller: SlackTeamController,
        *,
        poll_seconds: float = 5.0,
    ):
        self.store = store
        self.controller = controller
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="slackgentic-scheduled-timers",
        )
        self._thread.start()

    def stop(self) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            return not self._thread.is_alive()
        return True

    def sync_once(self) -> int:
        fired = 0
        for timer in self.store.list_due_scheduled_timers(limit=50):
            if self.controller.fire_due_scheduled_timer(timer):
                fired += 1
        return fired

    def _run(self) -> None:
        backoff = LoopBackoff(base_seconds=self.poll_seconds, max_seconds=60.0)
        while not self._stop.wait(self.poll_seconds):
            try:
                self.sync_once()
                backoff.reset()
            except Exception:
                log_loop_failure(LOGGER, "failed to run scheduled Slackgentic timers", backoff)
                if backoff.wait(self._stop):
                    break


class ScheduledWorkRunner:
    def __init__(
        self,
        store: Store,
        controller: SlackTeamController,
        *,
        poll_seconds: float = 5.0,
    ):
        self.store = store
        self.controller = controller
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="slackgentic-scheduled-work",
        )
        self._thread.start()

    def stop(self) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            return not self._thread.is_alive()
        return True

    def sync_once(self) -> int:
        fired = 0
        for scheduled in self.store.list_due_scheduled_work(limit=50):
            if self.controller.fire_due_scheduled_work(scheduled):
                fired += 1
        return fired

    def _run(self) -> None:
        backoff = LoopBackoff(base_seconds=self.poll_seconds, max_seconds=60.0)
        while not self._stop.wait(self.poll_seconds):
            try:
                self.sync_once()
                backoff.reset()
            except Exception:
                log_loop_failure(LOGGER, "failed to run scheduled Slackgentic work", backoff)
                if backoff.wait(self._stop):
                    break


class DeferredWorkRunner:
    def __init__(
        self,
        store: Store,
        controller: SlackTeamController,
        *,
        poll_seconds: float = 5.0,
    ):
        self.store = store
        self.controller = controller
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="slackgentic-deferred-work",
        )
        self._thread.start()

    def stop(self) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            return not self._thread.is_alive()
        return True

    def sync_once(self) -> int:
        promoted = self.controller.evaluate_pending_deferred_work()
        fired = 0
        for deferred in self.store.list_due_deferred_work(limit=50):
            if self.controller.fire_due_deferred_work(deferred):
                fired += 1
        return promoted + fired

    def _run(self) -> None:
        backoff = LoopBackoff(base_seconds=self.poll_seconds, max_seconds=60.0)
        while not self._stop.wait(self.poll_seconds):
            try:
                self.sync_once()
                backoff.reset()
            except Exception:
                log_loop_failure(LOGGER, "failed to run deferred Slackgentic work", backoff)
                if backoff.wait(self._stop):
                    break


class PMInitiativeRunner:
    """Watchdog for PM initiatives.

    Surfaces blockers without ever starting or stopping tasks. Dispatch is
    owned by ``DeferredWorkRunner``. The watchdog only reads state and posts.
    """

    def __init__(
        self,
        store: Store,
        controller: SlackTeamController,
        *,
        poll_seconds: float = 30.0,
    ):
        self.store = store
        self.controller = controller
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="slackgentic-pm-watchdog",
        )
        self._thread.start()

    def stop(self) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            return not self._thread.is_alive()
        return True

    def sync_once(self) -> int:
        return self.controller.watch_pm_initiatives()

    def _run(self) -> None:
        backoff = LoopBackoff(base_seconds=self.poll_seconds, max_seconds=120.0)
        while not self._stop.wait(self.poll_seconds):
            try:
                self.sync_once()
                backoff.reset()
            except Exception:
                log_loop_failure(LOGGER, "failed to run Slackgentic PM watchdog", backoff)
                if backoff.wait(self._stop):
                    break


class SocketModeSlackApp:
    def __init__(self, config: AppConfig):
        if not config.slack.bot_token or not config.slack.app_token:
            raise ValueError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required")
        self.config = config
        self._request_executor = ThreadPoolExecutor(
            max_workers=SLACK_SOCKET_WORKER_THREADS,
            thread_name_prefix="slackgentic-socket",
        )
        self._request_slots = threading.BoundedSemaphore(SLACK_SOCKET_MAX_PENDING_REQUESTS)
        self._shutdown = threading.Event()
        self.store = Store(config.state_db)
        self.store.init_schema()
        self.gateway = SlackGateway(config.slack.bot_token)
        auth = self.gateway.auth_test()
        _ensure_codex_mcp_for_slack_app(config)
        _ensure_claude_native_input_hook_for_slack_app(config)
        self.codex_app_server = None
        codex_app_server_url = config.commands.codex_app_server_url
        if config.commands.codex_app_server_autostart and codex_app_server_url:
            self.codex_app_server = CodexAppServerManager(
                config.commands,
                url=codex_app_server_url,
            )
            started_url = self.codex_app_server.start()
            if started_url:
                codex_app_server_url = started_url
                self.store.set_setting("codex.app_server_url", started_url)
        self.runtime = ManagedTaskRuntime(
            self.store,
            self.gateway,
            config.commands,
            poll_seconds=config.poll_seconds,
            home=config.home,
        )
        self.session_bridge = ExternalSessionBridge(
            self.store,
            self.gateway,
            config.commands,
            codex_app_server_url=codex_app_server_url,
        )
        self.controller = SlackTeamController(
            self.store,
            self.gateway,
            default_channel_id=config.slack.channel_id,
            runtime=self.runtime,
            home=config.home,
            ignored_bot_id=auth.get("bot_id"),
            session_bridge=self.session_bridge,
            team_id=config.slack.team_id or "local",
            codex_app_server_url=codex_app_server_url,
            slash_command=config.slack.slash_command,
            default_cwd=config.commands.default_cwd,
        )
        self.update_runner = SlackgenticUpdateRunner(
            store=self.store,
            checker=UpdateChecker(GitHubReleaseSource(config.updates.repository)),
            updater=SelfUpdater(repository=config.updates.repository),
            channel_id=self.controller.update_channel_id,
            prompt=self.controller.post_update_prompt,
            update_message=self.gateway.update_message,
            status_blocks=lambda candidate, status, include_actions: build_update_prompt_blocks(
                candidate,
                status_text=status,
                include_actions=include_actions,
            ),
            restart=self._restart_after_update,
            enabled=config.updates.enabled,
            poll_seconds=config.updates.check_interval_seconds,
        )
        self.controller.set_update_runner(self.update_runner)
        self.update_runner.start()
        self.runtime.on_task_done = self.controller.handle_runtime_task_done
        self.runtime.on_agent_message = self.controller.handle_runtime_agent_message
        self.runtime.on_agent_control = self.controller.handle_runtime_agent_control
        self.runtime.agent_icon_url = self.controller._agent_icon_url
        self.controller.cancel_orphaned_active_tasks()
        self.session_mirror = SessionMirror(
            self.store,
            self.gateway,
            [
                CodexProvider(home=config.home),
                ClaudeProvider(home=config.home),
            ],
            team_id=config.slack.team_id or "local",
            channel_id=config.slack.channel_id,
            poll_seconds=max(
                config.poll_seconds,
                config.sessions.external_session_mirror_poll_seconds,
            ),
            codex_app_server_url=codex_app_server_url,
            on_external_session_occupancy_change=(
                self.controller.handle_external_session_occupancy_change
            ),
            on_agent_message=self.controller.handle_mirrored_session_agent_message,
            home=config.home,
            ignored_cwd_patterns=config.sessions.ignored_external_session_cwds,
            allowed_cwd_prefixes=config.sessions.allowed_external_session_cwd_prefixes,
        )
        self.session_mirror.start()
        self.controller.resume_pending_work_requests_for_configured_channel()
        self.scheduled_timers = ScheduledTimerRunner(
            self.store,
            self.controller,
            poll_seconds=max(config.poll_seconds, 1.0),
        )
        self.scheduled_timers.start()
        self.scheduled_work = ScheduledWorkRunner(
            self.store,
            self.controller,
            poll_seconds=max(config.poll_seconds, 1.0),
        )
        self.scheduled_work.start()
        self.deferred_work = DeferredWorkRunner(
            self.store,
            self.controller,
            poll_seconds=max(config.poll_seconds, 1.0),
        )
        self.deferred_work.start()
        self.pm_watchdog = PMInitiativeRunner(
            self.store,
            self.controller,
            poll_seconds=max(config.poll_seconds, 30.0),
        )
        self.pm_watchdog.start()
        self.slack_message_backfill = SlackMessageBackfill(
            self.store,
            self.gateway,
            self.controller,
            team_id=config.slack.team_id or "local",
            poll_seconds=max(config.poll_seconds, 5.0),
        )
        self.slack_message_backfill.start()
        self.claude_permission_auto_resolver = ClaudePermissionAutoResolver(
            self.store,
            self.gateway,
            poll_seconds=min(max(config.poll_seconds, 0.2), 2.0),
        )
        self.claude_permission_auto_resolver.start()
        self.awake_keeper = ActiveSessionAwakeKeeper(
            lambda: self.runtime.has_running_tasks() or self.session_mirror.has_active_sessions()
        )
        self.awake_keeper.start()
        self.cpu_watchdog = ProcessCpuWatchdog()
        self.cpu_watchdog.start()

    def close(self) -> None:
        all_stopped = True
        request_executor = getattr(self, "_request_executor", None)
        if request_executor is not None:
            request_executor.shutdown(wait=True)
        update_runner = getattr(self, "update_runner", None)
        if update_runner is not None:
            all_stopped = update_runner.stop() and all_stopped
        all_stopped = self.cpu_watchdog.stop() and all_stopped
        self.awake_keeper.stop()
        all_stopped = self.claude_permission_auto_resolver.stop() and all_stopped
        all_stopped = self.slack_message_backfill.stop() and all_stopped
        all_stopped = self.pm_watchdog.stop() and all_stopped
        all_stopped = self.deferred_work.stop() and all_stopped
        all_stopped = self.scheduled_work.stop() and all_stopped
        all_stopped = self.scheduled_timers.stop() and all_stopped
        all_stopped = self.session_mirror.stop() and all_stopped
        if self.runtime is not None:
            stop_all = getattr(self.runtime, "stop_all_running_tasks", None)
            if callable(stop_all):
                try:
                    # Preserve task state and managed-run markers so launchd restarts
                    # can resume interrupted work instead of freeing the agent.
                    stop_all(status=None)
                except Exception:
                    LOGGER.exception("failed to stop running managed tasks during shutdown")
                    all_stopped = False
                if self.runtime.has_running_tasks():
                    LOGGER.warning(
                        "managed task workers did not stop before daemon shutdown; "
                        "leaving Store open to avoid closing a database still in use"
                    )
                    all_stopped = False
        if self.codex_app_server:
            self.codex_app_server.close()
        if all_stopped:
            self.store.close()
        else:
            LOGGER.warning("background workers are still active; Store close skipped")

    def request_shutdown(self) -> None:
        self._shutdown.set()

    def _restart_after_update(self) -> None:
        from agent_harness.service import installed_services_match, restart_service

        try:
            specs = self._service_specs_after_update()
            if specs is not None and installed_services_match(specs):
                result = restart_service(force=True)
            elif self._schedule_service_reinstall_after_update():
                return
            else:
                result = restart_service(force=True)
        except Exception as exc:
            self.request_shutdown()
            raise RuntimeError(
                "installed the update, but automatic service restart failed"
            ) from exc
        if result != 0:
            self.request_shutdown()
            raise RuntimeError(
                f"installed the update, but service restart exited with status {result}"
            )

    def _schedule_service_reinstall_after_update(self) -> bool:
        from agent_harness.service import start_update_helper

        executable = Path(sys.argv[0])
        if not executable.exists():
            found = shutil.which("slackgentic")
            if not found:
                return False
            executable = Path(found)
        workdir = _service_reinstall_workdir(Path(__file__).resolve())
        command = _service_reinstall_command(
            self.config,
            executable=executable.resolve(),
            working_directory=workdir,
        )
        version = self.store.get_setting("slackgentic.update.installing_version") or "unknown"
        start_update_helper(
            executable=executable.resolve(),
            state_db=self.config.state_db,
            version=version,
            command=command,
            log_dir=self.config.state_db.parent / "logs",
            working_directory=workdir,
            config_file=self.config.config_file,
        )
        return True

    def _service_specs_after_update(self):
        from agent_harness.service import (
            build_codex_app_server_service_spec,
            build_service_spec,
        )

        executable = Path(sys.argv[0])
        if not executable.exists():
            found = shutil.which("slackgentic")
            if not found:
                return None
            executable = Path(found)
        workdir = _service_reinstall_workdir(Path(__file__).resolve())
        daemon_spec = build_service_spec(
            executable=executable.resolve(),
            working_directory=workdir,
            config_file=self.config.config_file,
            ignored_external_session_cwds=list(self.config.sessions.ignored_external_session_cwds),
            allowed_external_session_cwd_prefixes=list(
                self.config.sessions.allowed_external_session_cwd_prefixes
            ),
        )
        specs = [daemon_spec]
        codex_app_server_url = self.config.commands.codex_app_server_url
        if codex_app_server_url:
            codex_binary = _resolved_command_path(self.config.commands.codex_binary)
            specs.append(
                build_codex_app_server_service_spec(
                    executable=Path(codex_binary) if codex_binary else None,
                    working_directory=workdir,
                    url=codex_app_server_url,
                )
            )
        return specs

    def run_forever(self) -> None:
        import signal

        from slack_sdk.socket_mode import SocketModeClient
        from slack_sdk.socket_mode.response import SocketModeResponse

        client = SocketModeClient(
            app_token=self.config.slack.app_token or "",
            web_client=self.gateway.client,
        )

        def listener(socket_client, request) -> None:
            if _is_view_submission_request(request):
                try:
                    payload = self.handle_request(request)
                except Exception:
                    LOGGER.exception("failed to handle Slack Socket Mode request")
                    payload = None
                socket_client.send_socket_mode_response(
                    SocketModeResponse(envelope_id=request.envelope_id, payload=payload)
                )
                return
            socket_client.send_socket_mode_response(
                SocketModeResponse(envelope_id=request.envelope_id)
            )
            self._submit_acknowledged_request(request)

        client.socket_mode_request_listeners.append(listener)
        shutdown = getattr(self, "_shutdown", threading.Event())
        self._shutdown = shutdown
        shutdown.clear()
        socket_connected_at: float | None = None

        def _request_shutdown(_signum=None, _frame=None) -> None:
            shutdown.set()

        previous_handlers: dict[int, object] = {}
        for sig_name in ("SIGTERM", "SIGINT"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                previous_handlers[sig] = signal.signal(sig, _request_shutdown)
            except (ValueError, OSError):
                continue
        try:
            connect_backoff = LoopBackoff(base_seconds=1.0, max_seconds=60.0)
            while not shutdown.is_set():
                try:
                    client.connect()
                    socket_connected_at = time.monotonic()
                    connect_backoff.reset()
                    break
                except Exception:
                    log_loop_failure(
                        LOGGER,
                        "failed to connect Slack Socket Mode client",
                        connect_backoff,
                    )
                    if connect_backoff.wait(shutdown):
                        break
            while not shutdown.is_set():
                is_connected = getattr(client, "is_connected", None)
                disconnected = callable(is_connected) and not is_connected()
                stale = _socket_mode_connection_stale(
                    client,
                    connected_at=socket_connected_at,
                )
                if stale and not disconnected:
                    LOGGER.warning(
                        "Slack Socket Mode connection is stale; reconnecting (session id: %s)",
                        client.session_id(),
                    )
                if disconnected or stale:
                    try:
                        if stale:
                            client.connect_to_new_endpoint(force=True)
                        else:
                            client.connect()
                        socket_connected_at = time.monotonic()
                        connect_backoff.reset()
                    except Exception:
                        log_loop_failure(
                            LOGGER,
                            "failed to reconnect Slack Socket Mode client",
                            connect_backoff,
                        )
                        if connect_backoff.wait(shutdown):
                            break
                        continue
                if shutdown.wait(timeout=1.0):
                    break
        except KeyboardInterrupt:
            shutdown.set()
        finally:
            for sig, previous in previous_handlers.items():
                try:
                    signal.signal(sig, previous)
                except (ValueError, OSError, TypeError):
                    continue
            try:
                client.close()
            except Exception:
                LOGGER.debug("failed to close Slack socket client cleanly", exc_info=True)
            self.close()

    def handle_request(self, request) -> dict | None:
        if request.type == "interactive":
            payload = request.payload
            payload_type = payload.get("type")
            if payload_type == "block_actions":
                self.controller.handle_block_action(payload)
            elif payload_type == "view_submission":
                return self.controller.handle_view_submission(payload, async_success=True)
            return None
        if request.type == "events_api":
            self.controller.handle_event(request.payload)
        elif request.type == "slash_commands":
            self.controller.handle_slash_command(request.payload)
        return None

    def _submit_acknowledged_request(self, request) -> None:
        request_executor = getattr(self, "_request_executor", None)
        if request_executor is None:
            self._handle_acknowledged_request(request)
            return
        request_slots = getattr(self, "_request_slots", None)
        if request_slots is not None and not request_slots.acquire(blocking=False):
            LOGGER.debug("Slack request worker backlog is full; handling request inline")
            self._handle_acknowledged_request(request)
            return
        try:
            request_executor.submit(self._handle_acknowledged_request_with_slot, request)
        except RuntimeError:
            if request_slots is not None:
                request_slots.release()
            LOGGER.debug("Slack request executor is shut down; handling request inline")
            self._handle_acknowledged_request(request)

    def _handle_acknowledged_request_with_slot(self, request) -> None:
        try:
            self._handle_acknowledged_request(request)
        finally:
            request_slots = getattr(self, "_request_slots", None)
            if request_slots is not None:
                request_slots.release()

    def _handle_acknowledged_request(self, request) -> None:
        try:
            self.handle_request(request)
        except Exception:
            LOGGER.exception("failed to handle Slack Socket Mode request")


def _ensure_codex_mcp_for_slack_app(config: AppConfig) -> None:
    try:
        registered = ensure_codex_mcp_server_registered(
            home=config.home,
            codex_binary=config.commands.codex_binary,
        )
    except Exception:
        LOGGER.warning("failed to register Codex MCP server", exc_info=True)
        return
    if registered:
        LOGGER.info("registered Codex MCP server: slackgentic")


def _ensure_claude_native_input_hook_for_slack_app(config: AppConfig) -> None:
    try:
        if not is_slackgentic_mcp_server_configured(config.home):
            return
        ensure_claude_native_input_hook(home=config.home)
    except Exception:
        LOGGER.warning("failed to register Claude native input hook", exc_info=True)


def _service_reinstall_workdir(package_file: Path | None = None) -> Path:
    source_root = detect_source_root(package_file or Path(__file__).resolve())
    return (source_root or Path.cwd()).resolve()


def _service_reinstall_command(
    config: AppConfig,
    *,
    executable: Path,
    working_directory: Path,
) -> list[str]:
    command = [
        str(executable),
        "service",
        "install",
        "--workdir",
        str(working_directory),
        "--config-file",
        str(config.config_file),
    ]
    codex_app_server_url = config.commands.codex_app_server_url
    if codex_app_server_url:
        command.extend(["--codex-app-server-url", codex_app_server_url])
        codex_binary = _resolved_command_path(config.commands.codex_binary)
        if codex_binary:
            command.extend(["--codex-binary", codex_binary])
    else:
        command.append("--no-codex-app-server")
    for cwd_pattern in config.sessions.ignored_external_session_cwds:
        command.extend(["--ignore-external-session-cwd", cwd_pattern])
    for cwd_prefix in config.sessions.allowed_external_session_cwd_prefixes:
        command.extend(["--allow-external-session-cwd-prefix", cwd_prefix])
    return command


def _service_reinstall_environment(
    working_directory: Path,
    environ: os._Environ[str] | dict[str, str],
) -> dict[str, str]:
    env = dict(environ)
    env["PYTHONPATH"] = str(working_directory / "src")
    return env


def _resolved_command_path(command: str) -> str | None:
    if not command:
        return None
    path = Path(command).expanduser()
    if path.is_absolute() or len(path.parts) > 1:
        return str(path)
    return shutil.which(command) or command


def run_slack_app(config: AppConfig | None = None) -> int:
    app = SocketModeSlackApp(config or load_config_from_env())
    try:
        app.run_forever()
    finally:
        app.close()
    return 0


def _update_prompt_text(candidate: UpdateCandidate) -> str:
    return (
        f"Slackgentic {candidate.release.tag_name} is available "
        f"(current {candidate.current_version}). Upgrade now?"
    )


def _first_action(payload: dict) -> dict | None:
    actions = payload.get("actions") or []
    return actions[0] if actions else None


def _is_slackgentic_mcp_permission_request(params_json: str) -> bool:
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(params, dict):
        return False
    tool_name = params.get("tool_name")
    return isinstance(tool_name, str) and tool_name in SLACKGENTIC_MCP_PERMISSION_TOOLS


def _is_view_submission_request(request) -> bool:
    if getattr(request, "type", None) != "interactive":
        return False
    payload = getattr(request, "payload", None)
    return isinstance(payload, dict) and payload.get("type") == "view_submission"


def _socket_mode_connection_stale(
    client,
    *,
    connected_at: float | None,
    monotonic_now: float | None = None,
    wall_now: float | None = None,
    pong_grace_seconds: float = SLACK_SOCKET_PONG_GRACE_SECONDS,
    stale_seconds: float = SLACK_SOCKET_STALE_SECONDS,
) -> bool:
    is_connected = getattr(client, "is_connected", None)
    if callable(is_connected) and not is_connected():
        return True
    current_session = getattr(client, "current_session", None)
    last_pong = getattr(current_session, "last_ping_pong_time", None)
    if isinstance(last_pong, (int, float)):
        now = time.time() if wall_now is None else wall_now
        return now - float(last_pong) > stale_seconds
    if connected_at is None:
        return False
    now_monotonic = time.monotonic() if monotonic_now is None else monotonic_now
    return now_monotonic - connected_at > pong_grace_seconds


def _is_stop_command(text: str) -> bool:
    cleaned = re.sub(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", "", text).strip()
    return re.fullmatch(r"(?:please\s+)?stop[.!]?", cleaned, flags=re.IGNORECASE) is not None


def _payload_channel_id(payload: dict) -> str | None:
    channel = payload.get("channel") or {}
    if isinstance(channel, dict):
        return channel.get("id")
    return None


def _payload_message_ts(payload: dict) -> str | None:
    message = payload.get("message") or {}
    if isinstance(message, dict):
        return message.get("ts")
    return None


def _view_plain_value(values: dict, block_id: str, action_id: str) -> str | None:
    item = values.get(block_id, {}).get(action_id, {})
    return item.get("value")


def _view_selected_value(values: dict, block_id: str, action_id: str) -> str | None:
    item = values.get(block_id, {}).get(action_id, {})
    selected = item.get("selected_option") or {}
    return selected.get("value")


def _view_checked_values(values: dict, block_id: str, action_id: str) -> set[str]:
    item = values.get(block_id, {}).get(action_id, {})
    selected = item.get("selected_options") or []
    return {
        option["value"]
        for option in selected
        if isinstance(option, dict) and isinstance(option.get("value"), str)
    }


def _view_int_value(values: dict, block_id: str, action_id: str, default: int) -> int:
    value = _view_plain_value(values, block_id, action_id)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(parsed, 0)


def _normalize_channel_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-") or "agents"


def _suggested_repo_root(default_cwd: Path) -> Path:
    try:
        resolved = default_cwd.expanduser().resolve()
    except OSError:
        resolved = default_cwd.expanduser()
    try:
        return resolved.parents[1]
    except IndexError:
        return resolved


def _validated_repo_root(value: str) -> Path | None:
    raw = value.strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except OSError:
        path = path.expanduser()
    if not path.exists() or not path.is_dir():
        return None
    return path


def _roster_text(agents, statuses: dict[str, AgentRosterStatus] | None = None) -> str:
    if not statuses:
        return f"Agent roster: {len(agents)} active lightweight handles"
    occupied = sum(1 for status in statuses.values() if status.label != "Available")
    available = max(0, len(agents) - occupied)
    return (
        f"Agent roster: {len(agents)} active lightweight handles, "
        f"{available} available, {occupied} occupied"
    )


def _roster_message_channel_prefix(channel_id: str) -> str:
    return f"{SETTING_ROSTER_MESSAGE_PREFIX}{channel_id}."


def _roster_message_setting_key(channel_id: str, message_ts: str) -> str:
    return f"{_roster_message_channel_prefix(channel_id)}{message_ts}"


def _roster_render_hash_setting_key(channel_id: str, message_ts: str) -> str:
    return f"{SETTING_ROSTER_RENDER_HASH_PREFIX}{channel_id}.{message_ts}"


def _roster_pinned_setting_key(channel_id: str, message_ts: str) -> str:
    return f"{SETTING_ROSTER_PINNED_PREFIX}{channel_id}.{message_ts}"


def _roster_discovery_setting_key(channel_id: str) -> str:
    return f"{SETTING_ROSTER_DISCOVERY_PREFIX}{channel_id}"


def _roster_render_hash(text: str, blocks: list[dict]) -> str:
    payload = json.dumps(
        {"text": text, "blocks": blocks},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _message_status_reaction_setting_key(channel_id: str, message_ts: str) -> str:
    return f"{SETTING_MESSAGE_STATUS_REACTION_PREFIX}{channel_id}.{message_ts}"


def _agent_authored_message_setting_key(channel_id: str, message_ts: str) -> str:
    return f"{SETTING_AGENT_AUTHORED_MESSAGE_PREFIX}{channel_id}.{message_ts}"


def _slack_reaction_processed_key(dedupe_id: str) -> str:
    return f"{SETTING_SLACK_REACTION_PROCESSED_PREFIX}{dedupe_id}"


def _is_roster_message(message: dict) -> bool:
    text = message.get("text")
    if isinstance(text, str) and text.startswith("Agent roster:"):
        return True
    blocks = message.get("blocks")
    if not isinstance(blocks, list):
        return False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text_obj = block.get("text")
        if not isinstance(text_obj, dict):
            continue
        value = text_obj.get("text")
        if isinstance(value, str) and value.startswith("*Agent team*"):
            return True
    return False


def _slack_ts_sort_key(value: str) -> tuple[float, str]:
    try:
        return (float(value), value)
    except ValueError:
        return (0.0, value)


def _shorten(value: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 1)].rstrip()}..."


def _strip_handle_prefix(text: str, handle: str) -> str:
    """Drop a leading ``@handle`` or ``handle:`` from ``text`` and return the rest."""
    pattern = re.compile(
        rf"^\s*@?{re.escape(handle)}\b[\s:,-]*",
        re.IGNORECASE,
    )
    return pattern.sub("", text, count=1).strip()


def _reaction_event_dedupe_id(event: dict) -> str:
    event_ts = event.get("event_ts")
    if isinstance(event_ts, str) and event_ts:
        return event_ts
    payload = json.dumps(event, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reaction_relay_prompt(
    reaction_name: str,
    display_name: str | None,
    *,
    removed: bool = False,
) -> str:
    actor = display_name.strip() if display_name and display_name.strip() else "A Slack user"
    if removed:
        return (
            f"{actor} removed their :{reaction_name}: reaction from one of your "
            "Slack-visible messages in this task thread. Treat this as lightweight "
            "feedback; do not reply unless it changes what you should do."
        )
    return (
        f"{actor} reacted with :{reaction_name}: to one of your Slack-visible messages "
        "in this task thread. Treat this as lightweight feedback; do not reply unless "
        "it changes what you should do."
    )


def _record_string(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _event_has_slackgentic_progress_reaction(event: dict) -> bool:
    reactions = event.get("reactions")
    if not isinstance(reactions, list):
        return False
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        if reaction.get("name") in {
            TASK_REACTION_ACKNOWLEDGED,
            TASK_REACTION_QUEUED,
            TASK_REACTION_IN_PROGRESS,
            TASK_REACTION_DONE,
        }:
            return True
    return False


def _task_request_message_ts_values(task: AgentTask) -> list[str]:
    values: list[str] = []
    request_message_ts = task.metadata.get("request_message_ts")
    if isinstance(request_message_ts, str) and request_message_ts:
        values.append(request_message_ts)
    history = task.metadata.get("request_message_ts_history")
    if isinstance(history, list):
        values.extend(item for item in history if isinstance(item, str) and item)
    if not values and task.parent_message_ts:
        values.append(task.parent_message_ts)
    return list(dict.fromkeys(values))


def _metadata_with_request_message_ts(
    metadata: dict[str, object],
    message_ts: str,
) -> dict[str, object]:
    updated = dict(metadata)
    current = updated.get("request_message_ts")
    history = updated.get("request_message_ts_history")
    values: list[str] = []
    if isinstance(current, str) and current:
        values.append(current)
    if isinstance(history, list):
        values.extend(item for item in history if isinstance(item, str) and item)
    values.append(message_ts)
    deduped = list(dict.fromkeys(values))
    updated["request_message_ts"] = message_ts
    prior = [item for item in deduped if item != message_ts]
    if prior:
        updated["request_message_ts_history"] = prior[-20:]
    else:
        updated.pop("request_message_ts_history", None)
    return updated


def _metadata_has_message_ts(metadata: dict[str, object], message_ts: str) -> bool:
    request_message_ts = metadata.get("request_message_ts")
    if isinstance(request_message_ts, str) and request_message_ts == message_ts:
        return True
    request_message_ts_history = metadata.get("request_message_ts_history")
    if isinstance(request_message_ts_history, list):
        return any(
            item == message_ts for item in request_message_ts_history if isinstance(item, str)
        )
    return False


def _active_thread_followup_message_ts_values(metadata: dict[str, object]) -> list[str]:
    values: list[str] = []
    raw_values = metadata.get(ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_VALUES_METADATA_KEY)
    if isinstance(raw_values, list):
        values.extend(item for item in raw_values if isinstance(item, str) and item)
    raw_value = metadata.get(ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_METADATA_KEY)
    if isinstance(raw_value, str) and raw_value:
        values.append(raw_value)
    return list(dict.fromkeys(values))


def _latest_user_message_ts_for_reaction(metadata: dict[str, object]) -> str | None:
    active_values = _active_thread_followup_message_ts_values(metadata)
    if active_values:
        return active_values[-1]
    request_message_ts = metadata.get("request_message_ts")
    if isinstance(request_message_ts, str) and request_message_ts:
        return request_message_ts
    history = metadata.get("request_message_ts_history")
    if isinstance(history, list):
        for item in reversed(history):
            if isinstance(item, str) and item:
                return item
    return None


def _live_thread_followup_prompt(prompt: str) -> str:
    if not _looks_like_direct_question(prompt):
        return prompt
    return (
        "The user sent this follow-up while you were already working. "
        "It asks a direct question; answer it explicitly in Slack before continuing "
        "implementation work, opening a PR, or scheduling a delayed follow-up.\n\n"
        f"{prompt}"
    )


def _looks_like_direct_question(text: str) -> bool:
    compact = " ".join(text.strip().lower().split())
    if not compact:
        return False
    if "?" in compact:
        return True
    if compact.startswith(("do not ", "don't ")):
        return False
    starters = (
        "are ",
        "can ",
        "can you ",
        "could ",
        "could you ",
        "did ",
        "do ",
        "does ",
        "how ",
        "is ",
        "should ",
        "what ",
        "when ",
        "where ",
        "why ",
        "will ",
        "would ",
    )
    return compact.startswith(starters)


def _pending_request_message_ts(pending: PendingWorkRequest) -> str | None:
    if pending.message_ts:
        return pending.message_ts
    request_message_ts = pending.extra_metadata.get("request_message_ts")
    if isinstance(request_message_ts, str) and request_message_ts:
        return request_message_ts
    return pending.thread_ts or None


def _retired_inactive_handle(agent, used_handles: set[str]) -> str:
    suffix = re.sub(r"[^a-z0-9_-]", "", str(agent.agent_id).lower())[-12:] or "agent"
    base = f"retired-{suffix}"[:32].rstrip("-_")
    if len(base) < 2:
        base = "retired-agent"
    candidate = base
    counter = 2
    while candidate in used_handles:
        tail = f"-{counter}"
        candidate = f"{base[: 32 - len(tail)]}{tail}"
        counter += 1
    return candidate


def _slack_message_processed_key(channel_id: str, message_ts: str) -> str:
    return f"{SETTING_SLACK_MESSAGE_PROCESSED_PREFIX}{channel_id}.{message_ts}"


def _slack_backfill_thread_scan_key(channel_id: str, thread_ts: str) -> str:
    return f"{SETTING_SLACK_BACKFILL_THREAD_SCAN_PREFIX}{channel_id}.{thread_ts}"


def _external_session_delivery_setting_key(channel_id: str, message_ts: str) -> str:
    return f"{SETTING_EXTERNAL_SESSION_DELIVERY_PREFIX}{channel_id}.{message_ts}"


def _human_user_display_name_key(user_id: str) -> str:
    return f"{SETTING_HUMAN_USER_DISPLAY_NAME_PREFIX}{user_id}"


def _float_setting(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _slack_ts_from_unix(value: float) -> str:
    return f"{value:.6f}"


def _slack_ts_sort_key(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _history_message_event(
    channel_id: str,
    message: dict,
    *,
    default_thread_ts: str | None = None,
) -> dict:
    event = dict(message)
    event.setdefault("type", "message")
    event["channel"] = channel_id
    if default_thread_ts and event.get("ts") != default_thread_ts:
        event.setdefault("thread_ts", default_thread_ts)
    return event


def _task_dangerous_mode(task: AgentTask) -> bool:
    return bool(task.metadata.get(DANGEROUS_MODE_METADATA_KEY))


def _task_roster_summary(task: AgentTask) -> str:
    summary = task.metadata.get(ROSTER_SUMMARY_METADATA_KEY)
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return _task_assignment_prompt(task)


def _task_roster_detail(task: AgentTask) -> str:
    return (
        f"{_shorten(_task_roster_summary(task), 140)}\n"
        f"*Original Task:* {_shorten(_task_original_prompt(task), 180)}"
    )


def _roster_summary_line(value: str) -> str:
    return _shorten(value, 160)


def _external_session_dangerous_mode(session) -> bool:
    if str(session.permission_mode or "") == PermissionMode.DANGEROUS.value:
        return True
    return bool(session.metadata.get(DANGEROUS_MODE_METADATA_KEY))


def _task_assignment_prompt(task: AgentTask) -> str:
    prompt = task.metadata.get(ASSIGNMENT_PROMPT_METADATA_KEY)
    return prompt if isinstance(prompt, str) and prompt.strip() else task.prompt


def _task_original_prompt(task: AgentTask) -> str:
    original_task = task.metadata.get(ORIGINAL_TASK_METADATA_KEY)
    if isinstance(original_task, str) and original_task.strip():
        return original_task.strip()
    return _task_assignment_prompt(task)


def _work_request_from_scheduled_work(scheduled: ScheduledWork) -> WorkRequest:
    return WorkRequest(
        prompt=scheduled.prompt,
        assignment_mode=scheduled.assignment_mode,
        requested_handle=scheduled.requested_handle,
        task_kind=scheduled.task_kind,
        author_handle=scheduled.author_handle,
        pr_url=scheduled.pr_url,
        permission_mode=(
            PermissionMode.DANGEROUS if scheduled.dangerous_mode else DEFAULT_PERMISSION_MODE
        ),
    )


def _next_scheduled_work_run(scheduled: ScheduledWork):
    if scheduled.schedule_kind != ScheduledWorkKind.RECURRING:
        return None
    return next_run_after(scheduled.recurrence, after=utc_now())


def _format_scheduled_work_ack(
    scheduled: ScheduledWork,
    description: str,
    request: WorkRequest,
) -> str:
    target = "somebody"
    if request.assignment_mode == AssignmentMode.SPECIFIC and request.requested_handle:
        target = f"@{request.requested_handle}"
    return (
        f"Scheduled: {target} `{_shorten(request.prompt, 120)}`; {description}; "
        f"next `{_format_schedule_timestamp(scheduled.next_run_at)}`."
    )


def _scheduled_tasks_text(scheduled: list[ScheduledWork]) -> str:
    if not scheduled:
        return "Scheduled tasks: none."
    lines = [f"Scheduled tasks: {len(scheduled)} active"]
    for item in scheduled:
        lines.append(
            f"- `{item.schedule_id}` {_scheduled_work_target_text(item)} "
            f"`{_shorten(item.prompt, 80)}`; {_format_scheduled_work_schedule(item)}; "
            f"next run `{_format_schedule_timestamp(item.next_run_at)}`"
        )
    return "\n".join(lines)


def _scheduled_work_block_text(scheduled: ScheduledWork) -> str:
    return "\n".join(
        [
            f"*`{scheduled.schedule_id}`*  {_scheduled_work_target_text(scheduled)}",
            f"*Task:* {_shorten(scheduled.prompt, 220)}",
            f"*Schedule:* {_format_scheduled_work_schedule(scheduled)}",
            f"*Next run:* `{_format_schedule_timestamp(scheduled.next_run_at)}`",
            f"*Status:* {scheduled.status.value}",
        ]
    )


def _scheduled_work_target_text(scheduled: ScheduledWork) -> str:
    if scheduled.assignment_mode == AssignmentMode.SPECIFIC and scheduled.requested_handle:
        return f"@{scheduled.requested_handle}"
    return "somebody"


def _format_scheduled_work_schedule(scheduled: ScheduledWork) -> str:
    if scheduled.schedule_kind == ScheduledWorkKind.ONE_OFF:
        return "one-off"
    recurrence = scheduled.recurrence
    frequency = recurrence.get("frequency")
    time_text = recurrence.get("time")
    timezone = recurrence.get("timezone") or scheduled.timezone
    if frequency == "weekly":
        weekday = _weekday_label(recurrence.get("weekday"))
        if weekday and time_text and timezone:
            return f"weekly on {weekday} at {time_text} {timezone}"
    if frequency == "daily" and time_text and timezone:
        return f"daily at {time_text} {timezone}"
    if frequency == "interval":
        interval_seconds = interval_seconds_from_recurrence(recurrence)
        if interval_seconds is not None:
            return format_interval_seconds(interval_seconds)
    return "recurring"


def _format_schedule_timestamp(value) -> str:
    return value.isoformat(timespec="minutes")


def _work_request_from_deferred_work(deferred: DeferredWork) -> WorkRequest:
    return WorkRequest(
        prompt=deferred.prompt,
        assignment_mode=deferred.assignment_mode,
        requested_handle=deferred.requested_handle,
        task_kind=deferred.task_kind,
        author_handle=deferred.author_handle,
        pr_url=deferred.pr_url,
        permission_mode=(
            PermissionMode.DANGEROUS if deferred.dangerous_mode else DEFAULT_PERMISSION_MODE
        ),
    )


def _deferred_work_roster_detail(deferred: DeferredWork) -> str:
    if deferred.status == DeferredWorkStatus.READY:
        status_text = (
            f"ready; starts `{_format_schedule_timestamp(deferred.fire_at)}`"
            if deferred.fire_at is not None
            else "ready"
        )
    elif deferred.status == DeferredWorkStatus.CLAIMED:
        status_text = "starting"
    else:
        status_text = "waiting"
    detail = f"Deferred task: {_shorten(deferred.prompt, 110)}; {status_text}"
    if deferred.description:
        detail = f"{detail}; {_shorten(deferred.description, 90)}"
    return detail


def _format_deferred_work_ack(
    deferred: DeferredWork,
    description: str,
    request: WorkRequest,
    *,
    dependency_labeler: Callable[[str | None], str] | None = None,
) -> str:
    target = "somebody"
    if request.assignment_mode == AssignmentMode.SPECIFIC and request.requested_handle:
        target = f"@{request.requested_handle}"
    dep_lines = []
    for dep in deferred.depends_on:
        if dep.kind == WorkDependencyKind.THREAD:
            label = dep.permalink or f"{dep.channel_id}/{dep.thread_ts}"
            dep_lines.append(f"- thread {label}")
        else:
            label = _format_agent_busy_dependency(dep.task_id, dependency_labeler)
            dep_lines.append(f"- @{dep.handle}: {label}")
    deps_text = "\n".join(dep_lines)
    timing = ""
    if deferred.after_dep_delay_seconds:
        timing = f" then wait {deferred.after_dep_delay_seconds}s"
    elif deferred.run_at is not None:
        timing = f" then at {_format_schedule_timestamp(deferred.run_at)}"
    return (
        f"Deferred: {target} will run `{request.prompt}` "
        f"{description}.{timing}\nWaiting on:\n{deps_text}"
    )


def _format_agent_busy_dependency(
    task_id: str | None,
    dependency_labeler: Callable[[str | None], str] | None = None,
) -> str:
    if dependency_labeler is not None:
        label = dependency_labeler(task_id)
        if label:
            return label
    external_session = parse_external_session_dependency_id(task_id)
    if external_session is not None:
        provider, _session_id = external_session
        return f"{provider.value} external session"
    deferred_id = parse_deferred_work_dependency_id(task_id)
    if deferred_id is not None:
        return "deferred task"
    schedule_id = parse_scheduled_work_dependency_id(task_id)
    if schedule_id is not None:
        return "scheduled task"
    return "current task"


def _format_pm_plan_ack(
    initiative: PmInitiative,
    plan: ParsedPmPlan,
    *,
    requested_by: str | None = None,
    capacity_blocked: bool = False,
) -> str:
    mention = f"<@{requested_by}>" if requested_by else "you"
    estimate = estimate_pm_plan(plan)
    if capacity_blocked:
        start_note = (
            f"{mention} — review the plan below. I will *not* start any subtasks "
            "until the capacity issue below is resolved."
        )
    else:
        start_note = (
            f"{mention} — review the plan below. I will *not* start any subtasks "
            "until you click *Start executing*."
        )
    lines = [
        f":clipboard: PM plan ready for *{plan.title}* (`{initiative.initiative_id}`).",
        plan.summary,
        "",
        start_note,
        "",
        f"Estimate: {_format_pm_plan_estimate(estimate)}.",
        "",
        "DAG chart:",
        f"```text\n{render_pm_plan_dag(plan)}\n```",
        "",
        "Subtasks:",
    ]
    for index, subtask in enumerate(plan.subtasks, start=1):
        target = "anyone"
        if (
            subtask.request.assignment_mode == AssignmentMode.SPECIFIC
            and subtask.request.requested_handle
        ):
            target = f"@{subtask.request.requested_handle}"
        deps = ", ".join(subtask.depends_on) if subtask.depends_on else "none"
        kind = subtask.request.task_kind.value
        danger = " (dangerous)" if subtask.request.dangerous_mode else ""
        lines.append(
            f"{index}. `{subtask.local_id}` — {subtask.title} → {target} ({kind}{danger}); "
            f"deps: {deps}"
        )
    return "\n".join(lines)


def _format_pm_plan_estimate(estimate: PmPlanEstimate) -> str:
    parts = [
        f"{estimate.subtask_count} subtasks",
        f"critical path {estimate.critical_path_depth} deep",
    ]
    if estimate.co_design_count:
        parts.append(f"{estimate.co_design_count} co-design fan-out(s)")
    if estimate.dangerous_count:
        parts.append(f"{estimate.dangerous_count} dangerous-mode")
    parts.append(
        "rough wall-clock "
        f"{_format_pm_duration(estimate.min_wall_clock_seconds)}-"
        f"{_format_pm_duration(estimate.max_wall_clock_seconds)}"
    )
    return ", ".join(parts)


def _format_pm_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remainder = minutes % 60
    if remainder == 0:
        return f"{hours}h"
    return f"{hours}h{remainder:02d}m"


def _pm_plan_approval_blocks(
    initiative: PmInitiative,
    plan_text: str,
) -> list[dict]:
    blocks = [
        {
            "type": "section",
            "block_id": f"pm.plan.{initiative.initiative_id}.{index}",
            "text": {"type": "mrkdwn", "text": chunk},
        }
        for index, chunk in enumerate(_slack_mrkdwn_chunks(plan_text), start=1)
    ]
    blocks.append(
        {
            "type": "actions",
            "block_id": f"pm.plan.actions.{initiative.initiative_id}",
            "elements": [
                _slack_button(
                    "Start executing",
                    "pm_initiative.start",
                    encode_action_value(
                        "pm_initiative.start", initiative_id=initiative.initiative_id
                    ),
                    style="primary",
                ),
                _slack_button(
                    "Cancel",
                    "pm_initiative.cancel",
                    encode_action_value(
                        "pm_initiative.cancel", initiative_id=initiative.initiative_id
                    ),
                    style="danger",
                ),
            ],
        }
    )
    return blocks


def _pm_plan_capacity_blocks(
    initiative: PmInitiative,
    plan_text: str,
    shortfall: _PmCapacityShortfall,
) -> list[dict]:
    blocks = [
        {
            "type": "section",
            "block_id": f"pm.plan.{initiative.initiative_id}.{index}",
            "text": {"type": "mrkdwn", "text": chunk},
        }
        for index, chunk in enumerate(_slack_mrkdwn_chunks(plan_text), start=1)
    ]
    capacity_blocks = _pm_capacity_shortfall_blocks(initiative, shortfall)
    if capacity_blocks:
        blocks.extend(capacity_blocks)
    return blocks


def _slack_mrkdwn_chunks(text: str, *, max_chars: int = 2900, max_chunks: int = 48) -> list[str]:
    remaining_lines = text.splitlines() or [text]
    chunks: list[str] = []
    current = ""
    truncated = False
    for line in remaining_lines:
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
            if len(chunks) >= max_chunks:
                truncated = True
                break
        while len(line) > max_chars and len(chunks) < max_chunks:
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if len(chunks) >= max_chunks:
            truncated = bool(line)
            break
        current = line
    if current and len(chunks) < max_chunks:
        chunks.append(current)
    elif current:
        truncated = True
    if not chunks:
        chunks.append(" ")
    if truncated and chunks:
        chunks[-1] = _shorten(chunks[-1], max_chars - 40) + "\n\n_Plan truncated in preview._"
    return chunks


def _format_pm_subtask_parent_text(
    initiative: PmInitiative,
    subtask: ParsedPmSubtask,
    agent: TeamAgent,
    *,
    pm_thread_url: str | None,
    extra_deps: tuple[WorkDependency, ...],
) -> str:
    plan_deps = ", ".join(subtask.depends_on) if subtask.depends_on else "none"
    target = f"@{agent.handle}"
    status = "ready to start" if not subtask.depends_on and not extra_deps else "reserved"
    lines = [
        f"PM subtask `{subtask.local_id}` for *{initiative.title}* is {status} for {target}.",
        f"*Title:* {subtask.title}",
        f"*Plan dependencies:* {plan_deps}",
    ]
    if extra_deps:
        extra = ", ".join(dep.description or dep.task_id or dep.kind.value for dep in extra_deps)
        lines.append(f"*Capacity guard:* {extra}")
    if pm_thread_url:
        lines.append(f"*PM thread:* <{pm_thread_url}|open status thread>")
    lines.append(f"*Original Task:* {subtask.request.prompt}")
    return "\n".join(lines)


def _pm_initiative_roster_detail(initiative: PmInitiative) -> str:
    status = {
        PmInitiativeStatus.PLANNING: "planning",
        PmInitiativeStatus.AWAITING_APPROVAL: "awaiting approval",
        PmInitiativeStatus.ACTIVE: "active",
    }.get(initiative.status, initiative.status.value)
    return f"PM initiative {status}: {_shorten(initiative.title, 140)}"


def _format_pm_capacity_shortfall_message(shortfall: _PmCapacityShortfall) -> str:
    if shortfall.unavailable_handles:
        handles = ", ".join(f"@{handle}" for handle in shortfall.unavailable_handles)
        pronoun = "is" if len(shortfall.unavailable_handles) == 1 else "are"
        return (
            "I can't start this PM plan yet because it reserves specific worker "
            f"agents that are not free right now: {handles} {pronoun} unavailable. "
            "The plan is still parked. Wait for those agents to finish or ask the PM "
            "to replan with available agents; then use *Try start again* here or "
            "*Start executing* on the original approval card."
        )
    required = shortfall.required_workers
    available = shortfall.available_workers
    hire_count = shortfall.hire_count
    required_word = "agent" if required == 1 else "agents"
    available_word = "agent" if available == 1 else "agents"
    available_verb = "is" if available == 1 else "are"
    hire_word = "agent" if hire_count == 1 else "agents"
    return (
        "I can't start this PM plan yet because there are not enough free worker "
        f"agents. The plan needs {required} available worker {required_word} at "
        f"its widest parallel step, but only {available} worker {available_word} "
        f"{available_verb} free right now. Use *Hire and reserve {hire_count} "
        f"{hire_word}* below to add capacity directly to this PM initiative."
    )


def _pm_capacity_shortfall_blocks(
    initiative: PmInitiative,
    shortfall: _PmCapacityShortfall,
) -> list[dict] | None:
    if shortfall.unavailable_handles:
        text = _format_pm_capacity_shortfall_message(shortfall)
        return [
            {
                "type": "section",
                "block_id": f"pm.capacity.{initiative.initiative_id}",
                "text": {"type": "mrkdwn", "text": text},
            },
            {
                "type": "actions",
                "block_id": f"pm.capacity.actions.{initiative.initiative_id}",
                "elements": [
                    _slack_button(
                        "Try start again",
                        "pm_initiative.start",
                        encode_action_value(
                            "pm_initiative.start",
                            initiative_id=initiative.initiative_id,
                        ),
                        style="primary",
                    ),
                    _slack_button(
                        "Cancel",
                        "pm_initiative.cancel",
                        encode_action_value(
                            "pm_initiative.cancel", initiative_id=initiative.initiative_id
                        ),
                        style="danger",
                    ),
                ],
            },
        ]
    if shortfall.hire_count <= 0:
        return None
    hire_word = "agent" if shortfall.hire_count == 1 else "agents"
    text = _format_pm_capacity_shortfall_message(shortfall)
    return [
        {
            "type": "section",
            "block_id": f"pm.capacity.{initiative.initiative_id}",
            "text": {"type": "mrkdwn", "text": text},
        },
        {
            "type": "actions",
            "block_id": f"pm.capacity.actions.{initiative.initiative_id}",
            "elements": [
                _slack_button(
                    f"Hire and reserve {shortfall.hire_count} {hire_word}",
                    "pm_initiative.hire_and_start",
                    encode_action_value(
                        "pm_initiative.hire_and_start",
                        initiative_id=initiative.initiative_id,
                    ),
                    style="primary",
                ),
                _slack_button(
                    "Try start again",
                    "pm_initiative.start",
                    encode_action_value(
                        "pm_initiative.start",
                        initiative_id=initiative.initiative_id,
                    ),
                ),
                _slack_button(
                    "Cancel",
                    "pm_initiative.cancel",
                    encode_action_value(
                        "pm_initiative.cancel", initiative_id=initiative.initiative_id
                    ),
                    style="danger",
                ),
            ],
        },
    ]


def _pm_plan_parallel_worker_demand(plan: ParsedPmPlan) -> int:
    """Return the widest same-depth worker demand in a topologically sorted plan."""
    depth_by_id: dict[str, int] = {}
    width_by_depth: dict[int, int] = {}
    for subtask in plan.subtasks:
        dep_depths = [depth_by_id[dep] for dep in subtask.depends_on if dep in depth_by_id]
        depth = (max(dep_depths) + 1) if dep_depths else 1
        depth_by_id[subtask.local_id] = depth
        width_by_depth[depth] = width_by_depth.get(depth, 0) + 1
    return max(width_by_depth.values(), default=0)


def _pm_subtask_parent_blocks(text: str) -> list[dict]:
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": chunk},
        }
        for chunk in _slack_mrkdwn_chunks(text, max_chunks=6)
    ]


def _format_deferred_ready_message(deferred: DeferredWork) -> str:
    fire_at = _format_schedule_timestamp(deferred.fire_at) if deferred.fire_at else "shortly"
    return (
        f"Dependencies for deferred task `{_shorten(deferred.prompt, 180)}` are satisfied. "
        f"Starting at `{fire_at}` UTC."
    )


def _pm_deferred_display_status(deferred: DeferredWork, task: AgentTask | None) -> str:
    if task is not None:
        if task.status == AgentTaskStatus.ACTIVE:
            recoveries = managed_run_stall_recoveries(task)
            if recoveries >= MANAGED_RUN_MAX_STALL_RECOVERIES:
                return "stalled"
            if recoveries > 0:
                return "recovering"
            age = managed_run_started_age(task)
            if age is not None and age > MANAGED_RUN_STALL_TIMEOUT:
                return "stalled"
            return "active"
        if task.status == AgentTaskStatus.DONE:
            return "done"
        if task.status == AgentTaskStatus.CANCELLED:
            return "cancelled"
        if task.status == AgentTaskStatus.QUEUED:
            if deferred.status == DeferredWorkStatus.WAITING_DEPS:
                return "reserved"
            if deferred.status == DeferredWorkStatus.READY:
                return "ready"
            if deferred.status == DeferredWorkStatus.CLAIMED:
                return "starting"
            return "queued"
    if deferred.status == DeferredWorkStatus.WAITING_DEPS:
        return "waiting_deps"
    return deferred.status.value


def _pm_initiative_status_label(status: PmInitiativeStatus) -> str:
    labels = {
        PmInitiativeStatus.PLANNING: ":memo: planning",
        PmInitiativeStatus.AWAITING_APPROVAL: ":hourglass_flowing_sand: awaiting approval",
        PmInitiativeStatus.ACTIVE: ":large_blue_circle: active",
        PmInitiativeStatus.DONE: ":white_check_mark: done",
        PmInitiativeStatus.CANCELLED: ":no_entry: cancelled",
    }
    return labels.get(status, status.value)


def _pm_subtask_status_label(status: str) -> str:
    labels = {
        "missing": ":warning: missing",
        "reserved": ":bookmark_tabs: reserved",
        "waiting_deps": ":hourglass_flowing_sand: waiting_deps",
        "ready": ":large_green_circle: ready",
        "starting": ":rocket: starting",
        "active": ":large_blue_circle: active",
        "recovering": ":arrows_counterclockwise: recovering",
        "stalled": ":warning: stalled",
        "queued": ":inbox_tray: queued",
        "done": ":white_check_mark: done",
        "cancelled": ":no_entry: cancelled",
        "claimed": ":rocket: claimed",
    }
    return labels.get(status, status)


def _pm_subtask_status_sort_key(status: str) -> tuple[int, str]:
    order = {
        "missing": 0,
        "cancelled": 1,
        "stalled": 2,
        "recovering": 3,
        "active": 4,
        "starting": 5,
        "ready": 6,
        "reserved": 7,
        "waiting_deps": 8,
        "queued": 9,
        "done": 10,
    }
    return (order.get(status, 99), status)


def _weekday_label(value) -> str | None:
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    if isinstance(value, int) and 0 <= value < len(weekdays):
        return weekdays[value]
    return None


def _slack_button(
    text: str,
    action_id: str,
    value: str,
    style: str | None = None,
    url: str | None = None,
) -> dict:
    button: dict[str, object] = {
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


def _roster_work_modal(
    agent,
    *,
    channel_id: str,
    message_ts: str | None,
    initial_timing: str,
    occupied_handles: list[tuple[str, str, str]],
) -> dict:
    timing_options = [
        _modal_option("Now", "now", "Start as soon as capacity is available."),
        _modal_option("Once", "once", "Run once at the Run at timestamp."),
        _modal_option("Daily", "daily", "Repeat every day at a local time."),
        _modal_option("Weekly", "weekly", "Repeat weekly at a local time."),
    ]
    timing_by_value = {option["value"]: option for option in timing_options}
    initial_timing_option = timing_by_value.get(initial_timing) or timing_by_value["now"]
    target_label = f"@{agent.handle}" if agent is not None else "somebody"
    metadata = {
        "channel_id": channel_id,
        "message_ts": message_ts,
    }
    if agent is not None:
        metadata["agent_id"] = agent.agent_id
        metadata["handle"] = agent.handle
    blocks: list[dict] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Target:* {target_label}",
            },
        },
        {
            "type": "input",
            "block_id": "roster_work_prompt",
            "label": {"type": "plain_text", "text": "Work"},
            "element": {
                "type": "plain_text_input",
                "action_id": "value",
                "multiline": True,
                "placeholder": {"type": "plain_text", "text": "Describe the task"},
            },
        },
        {
            "type": "input",
            "block_id": "roster_work_kind",
            "label": {"type": "plain_text", "text": "Kind"},
            "element": {
                "type": "static_select",
                "action_id": "value",
                "initial_option": _modal_option("Work", AgentTaskKind.WORK.value),
                "options": [
                    _modal_option("Work", AgentTaskKind.WORK.value),
                    _modal_option("Review", AgentTaskKind.REVIEW.value),
                ],
            },
        },
        {
            "type": "input",
            "block_id": "roster_work_timing",
            "label": {"type": "plain_text", "text": "Timing"},
            "element": {
                "type": "static_select",
                "action_id": "value",
                "initial_option": initial_timing_option,
                "options": timing_options,
            },
        },
        {
            "type": "input",
            "block_id": "roster_work_run_at",
            "optional": True,
            "label": {"type": "plain_text", "text": "Run at"},
            "element": {
                "type": "plain_text_input",
                "action_id": "value",
                "placeholder": {"type": "plain_text", "text": "2026-05-16T17:00:00-05:00"},
            },
            "hint": {
                "type": "plain_text",
                "text": "Required for once. Optional no-earlier-than time when waiting on an agent.",
            },
        },
        {
            "type": "input",
            "block_id": "roster_work_time",
            "optional": True,
            "label": {"type": "plain_text", "text": "Repeat time"},
            "element": {
                "type": "plain_text_input",
                "action_id": "value",
                "placeholder": {"type": "plain_text", "text": "17:00"},
            },
            "hint": {"type": "plain_text", "text": "HH:MM, required for daily or weekly."},
        },
        {
            "type": "input",
            "block_id": "roster_work_timezone",
            "optional": True,
            "label": {"type": "plain_text", "text": "Repeat timezone"},
            "element": {
                "type": "plain_text_input",
                "action_id": "value",
                "initial_value": _default_roster_work_timezone(),
                "placeholder": {"type": "plain_text", "text": "America/Chicago"},
            },
            "hint": {
                "type": "plain_text",
                "text": "Required for daily or weekly repeats. Unused for now or once.",
            },
        },
        {
            "type": "input",
            "block_id": "roster_work_weekday",
            "optional": True,
            "label": {"type": "plain_text", "text": "Repeat weekday"},
            "element": {
                "type": "static_select",
                "action_id": "value",
                "placeholder": {"type": "plain_text", "text": "Choose for weekly"},
                "options": [
                    _modal_option("Monday", "0"),
                    _modal_option("Tuesday", "1"),
                    _modal_option("Wednesday", "2"),
                    _modal_option("Thursday", "3"),
                    _modal_option("Friday", "4"),
                    _modal_option("Saturday", "5"),
                    _modal_option("Sunday", "6"),
                ],
            },
        },
        {
            "type": "input",
            "block_id": "roster_work_dependency",
            "optional": True,
            "label": {"type": "plain_text", "text": "Wait for agent"},
            "element": {
                "type": "static_select",
                "action_id": "value",
                "initial_option": _modal_option("No dependency", "none"),
                "options": _roster_dependency_options(occupied_handles),
            },
            "hint": {
                "type": "plain_text",
                "text": (
                    "Optional: start after the selected agent's current, deferred, or "
                    "scheduled work finishes."
                ),
            },
        },
        {
            "type": "input",
            "block_id": "roster_work_delay",
            "optional": True,
            "label": {"type": "plain_text", "text": "Delay after dependency"},
            "element": {
                "type": "plain_text_input",
                "action_id": "value",
                "placeholder": {"type": "plain_text", "text": "seconds"},
            },
        },
        {
            "type": "input",
            "block_id": "roster_work_permissions",
            "optional": True,
            "label": {"type": "plain_text", "text": "Permissions"},
            "element": {
                "type": "checkboxes",
                "action_id": "dangerous",
                "options": [
                    _modal_option(
                        "Dangerous mode",
                        "dangerous",
                        "Bypass Codex sandbox/approvals or Claude permissions.",
                    )
                ],
            },
        },
    ]
    return {
        "type": "modal",
        "callback_id": "roster.work",
        "private_metadata": json.dumps(metadata, separators=(",", ":"), sort_keys=True),
        "title": {"type": "plain_text", "text": "Roster work"},
        "submit": {"type": "plain_text", "text": "Create"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": blocks,
    }


def _roster_dependency_options(occupied_handles: list[tuple[str, str, str]]) -> list[dict]:
    options = [_modal_option("No dependency", "none")]
    seen: set[str] = set()
    for handle, _task_id, label in occupied_handles:
        if handle in seen:
            continue
        seen.add(handle)
        options.append(
            _modal_option(
                f"@{handle}",
                handle,
                label,
            )
        )
        if len(options) >= 100:
            break
    return options


def _default_roster_work_timezone() -> str:
    candidates: list[str] = []
    env_tz = os.environ.get("TZ")
    if env_tz:
        candidates.append(env_tz.lstrip(":"))
    try:
        localtime = str(Path("/etc/localtime").resolve(strict=False))
    except OSError:
        localtime = ""
    marker = "/zoneinfo/"
    if marker in localtime:
        candidates.append(localtime.split(marker, 1)[1])
    for candidate in candidates:
        if candidate and _valid_roster_work_timezone(candidate):
            return candidate
    return "America/Chicago"


def _valid_roster_work_timezone(value: str) -> bool:
    recurrence = {"frequency": "daily", "time": "00:00", "timezone": value}
    return next_run_after(recurrence, after=utc_now()) is not None


def _modal_option(text: str, value: str, description: str | None = None) -> dict:
    option: dict[str, object] = {
        "text": {"type": "plain_text", "text": text[:75]},
        "value": value[:75],
    }
    if description:
        option["description"] = {"type": "plain_text", "text": description[:75]}
    return option


def _decode_roster_work_metadata(value) -> dict[str, str]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {str(key): str(item) for key, item in decoded.items() if item is not None}


def _required_future_timestamp(value: str) -> tuple[object | None, str | None]:
    if not value.strip():
        return None, "Enter a future ISO timestamp."
    return _optional_future_timestamp(value)


def _optional_future_timestamp(value: str) -> tuple[object | None, str | None]:
    if not value.strip():
        return None, None
    parsed = parse_timestamp(value.strip())
    if parsed is None:
        return None, "Enter an ISO timestamp, for example 2026-05-16T17:00:00-05:00."
    if parsed <= utc_now():
        return None, "Choose a future time."
    return parsed, None


def _optional_delay_seconds(value: str | None) -> tuple[int | None, str | None]:
    text = (value or "").strip()
    if not text:
        return None, None
    try:
        parsed = int(text)
    except ValueError:
        return None, "Enter a whole number of seconds."
    if parsed < 0:
        return None, "Delay must be zero or greater."
    return parsed, None


def _recurrence_from_roster_work_values(
    values: dict,
    timing: str,
) -> tuple[dict[str, object] | None, tuple[str, str] | None]:
    time_text = (_view_plain_value(values, "roster_work_time", "value") or "").strip()
    if re.fullmatch(r"\d{2}:\d{2}", time_text) is None:
        return None, ("roster_work_time", "Use HH:MM, for example 17:00.")
    timezone = (_view_plain_value(values, "roster_work_timezone", "value") or "").strip()
    if not timezone:
        return None, ("roster_work_timezone", "Enter an IANA timezone.")
    recurrence: dict[str, object] = {
        "frequency": timing,
        "time": time_text,
        "timezone": timezone,
    }
    if timing == "weekly":
        weekday_value = _view_selected_value(values, "roster_work_weekday", "value")
        if weekday_value is None:
            return None, ("roster_work_weekday", "Choose a weekday.")
        try:
            weekday = int(weekday_value)
        except ValueError:
            return None, ("roster_work_weekday", "Choose a weekday.")
        if weekday < 0 or weekday > 6:
            return None, ("roster_work_weekday", "Choose a weekday.")
        recurrence["weekday"] = weekday
    if next_run_after(recurrence, after=utc_now()) is None:
        return None, ("roster_work_timezone", "Use a valid IANA timezone.")
    return recurrence, None


def _schedule_change_modal(
    scheduled: ScheduledWork,
    *,
    channel_id: str,
    message_ts: str | None,
) -> dict:
    metadata = json.dumps(
        {
            "schedule_id": scheduled.schedule_id,
            "channel_id": channel_id,
            "message_ts": message_ts,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    blocks: list[dict] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{scheduled.schedule_id}*  {_scheduled_work_target_text(scheduled)}\n"
                    f"*Task:* {_shorten(scheduled.prompt, 180)}"
                ),
            },
        },
        {
            "type": "input",
            "block_id": "schedule_next_run",
            "label": {"type": "plain_text", "text": "Next run"},
            "element": {
                "type": "plain_text_input",
                "action_id": "value",
                "initial_value": _format_schedule_timestamp(scheduled.next_run_at),
                "placeholder": {"type": "plain_text", "text": "2026-05-16T17:00:00-05:00"},
            },
        },
    ]
    if scheduled.schedule_kind == ScheduledWorkKind.RECURRING:
        blocks.append(
            {
                "type": "input",
                "block_id": "schedule_recurrence",
                "label": {"type": "plain_text", "text": "Recurrence JSON"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "multiline": True,
                    "initial_value": json.dumps(scheduled.recurrence, indent=2, sort_keys=True),
                },
            }
        )
    return {
        "type": "modal",
        "callback_id": "schedule.change",
        "private_metadata": metadata,
        "title": {"type": "plain_text", "text": "Change schedule"},
        "submit": {"type": "plain_text", "text": "Save"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": blocks,
    }


def _decode_schedule_change_metadata(value) -> dict[str, str]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {str(key): str(item) for key, item in decoded.items() if item is not None}


def _view_errors(block_id: str, message: str) -> dict:
    return {"response_action": "errors", "errors": {block_id: message}}


def _parse_work_request_for_agents(
    text: str,
    agents,
    *,
    split_newlines: bool = False,
) -> WorkRequest | None:
    for candidate in _routing_text_candidates(text, split_newlines=split_newlines):
        canonical_text = canonicalize_agent_mentions(candidate, agents)
        request = parse_work_request(canonical_text, [agent.handle for agent in agents])
        if request is not None:
            return request
    return None


def _multi_specific_work_requests(
    text: str,
    agents,
    *,
    split_newlines: bool = False,
) -> list[WorkRequest]:
    known_handles = [agent.handle for agent in agents]
    for candidate in _routing_text_candidates(text, split_newlines=split_newlines):
        stripped_candidate, dangerous_mode = strip_dangerous_mode_tag(candidate)
        canonical_text = canonicalize_agent_mentions(stripped_candidate, agents)
        parsed = _multi_specific_prompt(canonical_text, set(known_handles))
        if parsed is None:
            continue
        prompt, ordered_handles = parsed
        if len(ordered_handles) < 2:
            continue
        requests: list[WorkRequest] = []
        for handle in ordered_handles:
            request = parse_work_request(f"@{handle} {prompt}", known_handles)
            if request is not None and request.assignment_mode == AssignmentMode.SPECIFIC:
                if dangerous_mode and not request.dangerous_mode:
                    request = replace(request, permission_mode=PermissionMode.DANGEROUS)
                requests.append(request)
        if len(requests) >= 2:
            return requests
    return []


def _agent_authored_review_request(text: str, agents) -> WorkRequest | None:
    for candidate in _routing_text_candidates(text, split_newlines=True):
        if not _is_agent_authored_review_request(candidate):
            continue
        request = _parse_work_request_for_agents(candidate, agents)
        if request is not None:
            return request
    return None


_FINAL_HANDLE_CALLBACK_RE = re.compile(r"^\s*@(?P<handle>[A-Za-z][A-Za-z0-9_-]{1,31})\s*[.!?]?\s*$")
_REVIEW_CALLBACK_HINT_RE = re.compile(
    r"\b(?:review|green|green[-\s]?light|sign[-\s]?off|approve|approval|check|decision)\b",
    re.IGNORECASE,
)


def _agent_authored_final_handle_callback_request(
    text: str,
    agents,
    *,
    sender_agent,
) -> WorkRequest | None:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text.strip()) if item.strip()]
    if len(paragraphs) < 2:
        return None
    final_paragraph = canonicalize_agent_mentions(paragraphs[-1], agents)
    match = _FINAL_HANDLE_CALLBACK_RE.match(final_paragraph)
    if not match:
        return None
    handle = normalize_handle(match.group("handle"))
    target = next((agent for agent in agents if agent.handle == handle), None)
    if target is None or target.agent_id == sender_agent.agent_id:
        return None
    context = "\n\n".join(paragraphs[:-1]).strip()
    sender_handle = getattr(sender_agent, "handle", "the sending agent")
    prompt = (
        f"Review @{sender_handle}'s latest Slack-visible message and respond in the same "
        "thread. If it asks for a decision, answer directly before taking further action."
    )
    task_kind = (
        AgentTaskKind.REVIEW if _REVIEW_CALLBACK_HINT_RE.search(context) else AgentTaskKind.WORK
    )
    return WorkRequest(
        prompt=prompt,
        assignment_mode=AssignmentMode.SPECIFIC,
        requested_handle=handle,
        task_kind=task_kind,
    )


def _agent_authored_external_callback_text(text: str, request: WorkRequest) -> str:
    stripped = text.strip()
    if "\n" not in stripped:
        return request.prompt
    return stripped


def _routing_text_candidates(text: str, *, split_newlines: bool) -> list[str]:
    candidates = [text]
    if split_newlines and "\n" in text:
        candidates.extend(line.strip() for line in text.splitlines() if line.strip())
    return list(dict.fromkeys(candidates))


def _multi_specific_prompt(text: str, known_handles: set[str]) -> tuple[str, list[str]] | None:
    mention = r"@[a-z][a-z0-9_-]{1,31}"
    mention_list = rf"{mention}(?:\s*(?:,|and)\s*{mention}|\s+{mention})+"
    patterns = [
        rf"^\s*(?P<mentions>{mention_list})\s*[:,]?\s*(?:please\s+)?(?P<prompt>.+)$",
        (
            rf"^\s*(?:ask|tell|have)\s+(?P<mentions>{mention_list})\s+"
            rf"to\s+(?P<prompt>.+)$"
        ),
        (
            rf"^\s*(?:can|could)\s+(?P<mentions>{mention_list})\s+"
            rf"(?:please\s+)?(?P<prompt>.+)$"
        ),
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        handles = [
            handle
            for handle in parse_lightweight_handles(match.group("mentions"))
            if handle in known_handles
        ]
        if len(set(handles)) < 2:
            continue
        prompt = match.group("prompt").strip(" \t\n\r.")
        if prompt:
            return prompt, list(dict.fromkeys(handles))
    return None


def _channel_work_request(text: str, agents) -> WorkRequest | None:
    # New tasks require an explicit trigger: a "somebody"/anyone prefix or a
    # leading @handle. We intentionally do NOT prepend "somebody" to bare
    # verb-led messages like "do gate it" — that fallback caused conversational
    # phrases to be misread as new work.
    canonical_text = canonicalize_agent_mentions(text, agents)
    known_handles = [agent.handle for agent in agents]
    return parse_work_request(canonical_text, known_handles)


def _usage_request_kind(text: str) -> str | None:
    normalized = re.sub(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", "", text).strip().lower()
    normalized = re.sub(r"^/(status|usage|tokens)\b", r"\1", normalized)
    if normalized == "status":
        return "status"
    if normalized in {
        "usage",
        "tokens",
        "token usage",
        "show usage",
        "show tokens",
    }:
        return "usage"
    return None


def _is_agent_authored_review_request(text: str) -> bool:
    cleaned = re.sub(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", "", text).strip()
    anyone_pattern = r"somebody|someone|anyone|any agent|whoever"
    return bool(
        re.match(
            rf"^(?:please\s+)?(?:{anyone_pattern})\s+(?:please\s+)?review\b",
            cleaned,
            flags=re.IGNORECASE,
        )
    )


def _profile_display_name(profile: object) -> str | None:
    if not isinstance(profile, dict):
        return None
    for key in ("display_name", "real_name", "name", "username"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _profile_image_url(profile: object) -> str | None:
    if not isinstance(profile, dict):
        return None
    for key in ("image_512", "image_192", "image_72", "image_48"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _original_agent_delegation_prompt(text: str, parent_agent) -> str | None:
    if parent_agent is None:
        return None
    handle = re.escape(parent_agent.handle)
    patterns = [
        rf"\b(?:ask|tell|have)\s+@?{handle}\s+to\s+(?P<prompt>.+)",
        r"\b(?:ask|tell|have)\s+(?:the\s+)?original\s+agent\s+to\s+(?P<prompt>.+)",
        r"\bgive\s+(?:the\s+)?original\s+agent\s+(?:a\s+)?task\s+to\s+(?P<prompt>.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            prompt = match.group("prompt").strip()
            return prompt or None
    return None


def _thread_delegation_intent(
    text: str,
    parent_agent,
    agents,
) -> ThreadDelegationIntent | None:
    text = canonicalize_agent_mentions(text, agents)
    original_prompt = _original_agent_delegation_prompt(text, parent_agent)
    if original_prompt and parent_agent:
        return ThreadDelegationIntent(parent_agent.agent_id, original_prompt)

    for agent in sorted(agents, key=lambda item: len(item.handle), reverse=True):
        intent = _explicit_agent_delegation_intent(text, agent)
        if intent:
            return intent
    return None


def _explicit_agent_delegation_intent(text: str, agent) -> ThreadDelegationIntent | None:
    handle = re.escape(agent.handle)
    back_patterns = [
        rf"\b(?:give|hand|pass|send)\s+(?:it|this|the\s+(?:task|work|result|thread))\s+back\s+to\s+@?{handle}(?:\s+to\s+(?P<prompt>.+))?",
        rf"\breturn\s+(?:it|this|the\s+(?:task|work|result|thread))\s+to\s+@?{handle}(?:\s+to\s+(?P<prompt>.+))?",
    ]
    for pattern in back_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        prompt = (match.groupdict().get("prompt") or "").strip()
        if prompt:
            return ThreadDelegationIntent(agent.agent_id, prompt)
        return ThreadDelegationIntent(
            target_agent_id=agent.agent_id,
            prompt_template=THREAD_CONTEXT_DELEGATE_PROMPT,
            visible_prompt_template=THREAD_CONTEXT_DELEGATE_VISIBLE_PROMPT,
        )

    exact_patterns = [
        rf"\b(?:ask|tell|have)\s+@?{handle}\s+to\s+(?P<prompt>.+)",
        rf"\bgive\s+@?{handle}\s+(?:a\s+)?task\s+to\s+(?P<prompt>.+)",
        rf"\bgive\s+@?{handle}\s+work\s+to\s+(?P<prompt>.+)",
    ]
    for pattern in exact_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            prompt = match.group("prompt").strip()
            if prompt:
                return ThreadDelegationIntent(agent.agent_id, prompt)

    vague_patterns = [
        rf"\bgive\s+@?{handle}\s+(?:some\s+)?work(?:\s+to\s+do)?\b",
        rf"\bgive\s+@?{handle}\s+(?:a\s+)?task\b",
    ]
    for pattern in vague_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            prompt = (
                "Review the Slack thread context and perform the concrete task "
                "@{sender_handle} assigned to you. If @{sender_handle} did not "
                "assign a concrete task, ask @{sender_handle} for one."
            )
            return ThreadDelegationIntent(
                target_agent_id=agent.agent_id,
                prompt_template=prompt,
                visible_prompt_template="take the task I assigned above.",
            )
    return None


def _render_delegate_template(text: str, sender, target) -> str:
    return (
        text.replace("{sender_handle}", sender.handle)
        .replace("{target_handle}", target.handle)
        .strip()
    )


def _task_thread_blocks(task: AgentTask, agent) -> list[dict]:
    return build_task_thread_blocks(
        task,
        agent,
        include_actions=_task_should_show_action_buttons(task),
    )


def _task_should_show_action_buttons(task: AgentTask) -> bool:
    if task.status not in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}:
        return False
    return not (_is_subtask(task) or _is_external_thread_helper_task(task))


def _is_subtask(task: AgentTask) -> bool:
    if isinstance(task.metadata.get(PM_SUBTASK_LOCAL_ID_METADATA_KEY), str):
        return True
    parent_task_id = task.metadata.get("parent_task_id")
    if isinstance(parent_task_id, str):
        if parent_task_id != task.task_id:
            return True
        # Review tasks are one-off helpers even when a same-thread continuation
        # reuses the prior review task row and makes it self-parented.
        if task.kind == AgentTaskKind.REVIEW:
            return True
    return any(
        isinstance(task.metadata.get(key), str)
        for key in ("delegated_from_task_id", "delegate_to_agent_id")
    )


def _is_external_thread_helper_task(task: AgentTask) -> bool:
    return isinstance(task.metadata.get("external_session_id"), str) and isinstance(
        task.metadata.get("external_session_provider"),
        str,
    )


def _external_session_agent_setting_key(session) -> str:
    return f"{EXTERNAL_SESSION_AGENT_PREFIX}{session.provider.value}.{session.session_id}"


def _external_session_ignored_setting_key(session) -> str:
    return f"{EXTERNAL_SESSION_IGNORED_PREFIX}{session.provider.value}.{session.session_id}"


def _parse_external_session_setting_key(
    key: str,
    prefix: str,
) -> tuple[Provider, str] | None:
    if not key.startswith(prefix):
        return None
    body = key.removeprefix(prefix)
    provider_text, separator, session_id = body.partition(".")
    if not separator or not session_id:
        return None
    try:
        provider = Provider(provider_text)
    except ValueError:
        return None
    return provider, session_id
