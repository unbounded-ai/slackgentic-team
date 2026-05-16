from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from agent_harness.config import AppConfig, load_config_from_env
from agent_harness.models import (
    ASSIGNMENT_PROMPT_METADATA_KEY,
    DANGEROUS_MODE_METADATA_KEY,
    DEFAULT_PERMISSION_MODE,
    AgentTask,
    AgentTaskKind,
    AgentTaskStatus,
    AssignmentMode,
    PendingWorkRequest,
    PendingWorkRequestStatus,
    PermissionMode,
    Provider,
    ScheduledTimer,
    ScheduledTimerStatus,
    ScheduledWork,
    ScheduledWorkKind,
    ScheduledWorkStatus,
    SessionDependency,
    SessionStatus,
    SlackThreadRef,
    TeamAgentStatus,
    WorkRequest,
    utc_now,
)
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
    MANAGED_RUN_STARTED_METADATA_KEY,
    ManagedTaskRuntime,
    managed_run_resume_attempts,
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
    is_agent_schedule_signal,
    looks_like_schedule_request,
    next_run_after,
    parse_agent_schedule_signal,
)
from agent_harness.sessions.bridge import ExternalSessionBridge
from agent_harness.sessions.mirror import (
    EXTERNAL_SESSION_AGENT_PREFIX,
    EXTERNAL_SESSION_IGNORED_PREFIX,
    EXTERNAL_SESSION_SUMMARY_PREFIX,
    HUMAN_DISPLAY_NAME_SETTING,
    HUMAN_IMAGE_URL_SETTING,
    PENDING_EXTERNAL_SESSION_PREFIX,
    SessionMirror,
)
from agent_harness.slack import (
    AgentRosterStatus,
    build_channel_overview_blocks,
    build_setup_modal,
    build_task_thread_blocks,
    build_team_roster_blocks,
    decode_action_value,
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
    format_agent_assignment,
    format_agent_handoff_assignment,
    format_agent_handoff_request,
    format_agent_introduction,
    hire_team_agents,
)
from agent_harness.team.assignment import assign_work_request
from agent_harness.team.commands import (
    FireCommand,
    FireEveryoneCommand,
    HireCommand,
    RepoRootCommand,
    RosterCommand,
    parse_team_command,
)
from agent_harness.team.routing import (
    canonicalize_agent_mentions,
    parse_lightweight_handles,
    parse_work_request,
    strip_dangerous_mode_tag,
)
from agent_harness.timers import is_agent_timer_signal, parse_agent_timer_signal

LOGGER = logging.getLogger(__name__)
SETTING_CHANNEL_ID = "slack.channel_id"
SETTING_ROSTER_TS = "slack.roster_ts"
SETTING_ROSTER_MESSAGE_PREFIX = "slack.roster_message."
SETTING_ROSTER_DISCOVERY_PREFIX = "slack.roster_discovery."
SETTING_USAGE_TS_PREFIX = "slack.usage_ts."
SETTING_HUMAN_USER_ID = "slack.human_user_id"
SETTING_HUMAN_USER_DISPLAY_NAME_PREFIX = "slack.user_display_name."
SETTING_REPO_ROOT = TASK_RUNTIME_REPO_ROOT_SETTING
TASK_REACTION_ACKNOWLEDGED = "eyes"
TASK_REACTION_IN_PROGRESS = "hourglass_flowing_sand"
TASK_REACTION_DONE = "white_check_mark"
TASK_STATUS_REACTIONS = (
    TASK_REACTION_ACKNOWLEDGED,
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
SETTING_SLACK_MESSAGE_PROCESSED_PREFIX = "slack.message.processed."
SLACK_BACKFILL_FETCH_LIMIT = 500
SLACK_BACKFILL_GRACE_SECONDS = 5.0
SLACK_BACKFILL_SLEEP_GAP_SECONDS = 30.0
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
QUEUED_THREAD_FOLLOWUPS_METADATA_KEY = "queued_thread_followups"
ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_METADATA_KEY = "active_thread_followup_message_ts"
ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_VALUES_METADATA_KEY = "active_thread_followup_message_ts_values"


@dataclass(frozen=True)
class SlackReplyTarget:
    channel_id: str
    thread_ts: str | None = None


@dataclass(frozen=True)
class ThreadDelegationIntent:
    target_agent_id: str
    prompt_template: str
    visible_prompt_template: str | None = None


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
        self._slack_message_lock = threading.Lock()
        self._normalize_existing_agents()

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
        elif action_name.startswith("task."):
            self._task_from_action(decoded, channel_id, message_ts)
        elif action_name == "external.session.finish":
            self._external_session_finish_from_action(decoded, channel_id)
        elif action_name in AGENT_REQUEST_ACTIONS and self.session_bridge is not None:
            self.session_bridge.handle_agent_request_block_action(decoded, channel_id, message_ts)

    def handle_view_submission(self, payload: dict) -> dict | None:
        view = payload.get("view") or {}
        if view.get("callback_id") != "setup.initial":
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
            self.handle_team_command(command, SlackReplyTarget(channel_id=channel_id))
            return
        self.refresh_or_post_roster(channel_id)

    def handle_event(self, payload: dict) -> None:
        event = payload.get("event") or {}
        channel_id, message_ts = self._recoverable_user_message_ref(event)
        if channel_id is None or message_ts is None:
            return
        with self._slack_message_lock:
            if self._slack_message_processed(channel_id, message_ts):
                return
            self._handle_unprocessed_user_message_event(event, channel_id)
            self._mark_slack_message_processed(channel_id, message_ts)

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
            self.handle_team_command(command, target)
            return
        if self._handle_scheduled_work_request(event, channel_id, text):
            return
        if self._handle_task_thread_reply(event, channel_id, text):
            return
        active_agents = self.store.list_team_agents()
        thread = self._request_thread_anchor(event, channel_id, text)
        multi_requests = _multi_specific_work_requests(text, active_agents, split_newlines=True)
        if multi_requests:
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
        excluded_agent_ids = set(self._external_busy_agent_ids())
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
        blocks = build_task_thread_blocks(result.task, result.agent)
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
        command: HireCommand | FireCommand | FireEveryoneCommand | RepoRootCommand | RosterCommand,
        target: SlackReplyTarget,
    ) -> None:
        if isinstance(command, HireCommand):
            if not self._can_hire(command.count):
                self._post_text(
                    target, f"{AGENT_LIMIT_MESSAGE} Max team size is {MAX_TEAM_AGENTS}."
                )
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
        self.post_roster(
            target.channel_id,
            thread_ts=target.thread_ts,
            remember=target.thread_ts is None,
        )

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
        author_agent = None
        if pending.author_agent_id:
            author_agent = self.store.get_team_agent(pending.author_agent_id)
        if author_agent is None and pending.request.author_handle:
            author_agent = self.store.get_team_agent(pending.request.author_handle)
        exclude_agent_ids = set(pending.exclude_agent_ids)
        exclude_agent_ids.update(self._external_busy_agent_ids())
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
        blocks = build_task_thread_blocks(result.task, result.agent)
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
        task = self.store.get_agent_task(result.task.task_id) or result.task
        self._start_runtime_task(task, result.agent, thread)
        if request_message_ts:
            self._mark_message_in_progress(pending.channel_id, request_message_ts)
        return True

    def _post_text(self, target: SlackReplyTarget, text: str) -> None:
        if target.thread_ts:
            self.gateway.post_thread_reply(
                SlackThreadRef(target.channel_id, target.thread_ts),
                text,
            )
        else:
            self.gateway.post_message(target.channel_id, text)

    def _assignment_unavailable_text(self, request: WorkRequest) -> str:
        if request.assignment_mode == AssignmentMode.SPECIFIC and request.requested_handle:
            agent = self.store.get_team_agent(request.requested_handle)
            if agent is not None:
                idle_ids = {item.agent_id for item in self.store.idle_team_agents()}
                if (
                    agent.agent_id not in idle_ids
                    or agent.agent_id in self._external_busy_agent_ids()
                ):
                    return (
                        f"That specific agent is busy. Wait for @{agent.handle}, "
                        "or ask someone else."
                    )
        return CAPACITY_MESSAGE

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
            exclude_agent_ids=self._external_busy_agent_ids(),
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
            (f"@{result.agent.handle} is creating this schedule: `{_shorten(schedule_text, 180)}`"),
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
            exclude_agent_ids=self._external_busy_agent_ids(),
        )
        next_run_at = _next_scheduled_work_run(claimed)
        if result is None:
            pending = self.store.create_pending_work_request(
                thread,
                request,
                requested_by_slack_user=claimed.requested_by_slack_user,
                author_agent=author_agent,
                extra_metadata=metadata,
                exclude_agent_ids=self._external_busy_agent_ids(),
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
            return True
        text = f"Scheduled task `{claimed.schedule_id}` is due now.\n\n" + format_agent_assignment(
            result.agent,
            result.request.prompt,
            claimed.requested_by_slack_user,
            dangerous_mode=_task_dangerous_mode(result.task),
        )
        blocks = build_task_thread_blocks(result.task, result.agent)
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
        statuses = {agent.agent_id: AgentRosterStatus("Available") for agent in agents}
        for agent in agents:
            task = self.store.active_task_for_agent(agent.agent_id)
            if task is None:
                continue
            label = "Queued" if task.status == AgentTaskStatus.QUEUED else "Occupied"
            detail_prefix = "Dangerous mode: " if _task_dangerous_mode(task) else ""
            statuses[agent.agent_id] = AgentRosterStatus(
                label,
                f"{detail_prefix}Slack task: {_shorten(task.prompt, 140)}",
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
            detail_prefix = "Dangerous mode: " if _task_dangerous_mode(task) else ""
            statuses[agent.agent_id] = AgentRosterStatus(
                "Occupied",
                f"{detail_prefix}Slack task: {_shorten(task.prompt, 140)}",
                thread_url=self._thread_permalink(thread.channel_id, thread.thread_ts),
                task_id=task.task_id,
            )
        for agent_id, session in self._active_external_sessions_by_agent().items():
            if agent_id not in statuses:
                continue
            if statuses[agent_id].label != "Available":
                continue
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
                thread_url=self._external_session_permalink(session),
                session_provider=session.provider,
                session_id=session.session_id,
            )
        return statuses

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

    def _external_session_permalink(self, session) -> str | None:
        channel_id = self._configured_agent_channel_id()
        if channel_id is None:
            return None
        thread = self.store.get_slack_thread_for_session(
            session.provider,
            session.session_id,
            self.team_id,
            channel_id,
        )
        if thread is None:
            return None
        return self._thread_permalink(thread.channel_id, thread.thread_ts)

    def _thread_permalink(self, channel_id: str, thread_ts: str | None) -> str | None:
        if not thread_ts:
            return None
        try:
            return self.gateway.permalink(channel_id, thread_ts)
        except Exception:
            LOGGER.debug("failed to get Slack permalink for %s:%s", channel_id, thread_ts)
            return None

    def _external_busy_agent_ids(self) -> set[str]:
        return set(self._active_external_sessions_by_agent())

    def _active_external_sessions_by_agent(self):
        sessions_by_agent = {}
        for key, agent_id in self.store.list_settings(EXTERNAL_SESSION_AGENT_PREFIX).items():
            session = self._session_for_external_agent_setting(key)
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
        session_key = key.removeprefix(EXTERNAL_SESSION_AGENT_PREFIX)
        provider_text, separator, session_id = session_key.partition(".")
        if not separator or not session_id:
            return None
        try:
            provider = Provider(provider_text)
        except ValueError:
            return None
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

    def hire_agents(self, count: int, provider: Provider | None = None):
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
                    f"`{command} hire 3 agents`, or just `status`, "
                    "`show roster`, `hire 3 agents`."
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
        posted = self.gateway.post_message(
            channel_id,
            _roster_text(agents, statuses),
            blocks=build_team_roster_blocks(agents, statuses),
            thread_ts=thread_ts,
        )
        if remember:
            self._remember_roster_message(channel_id, posted.ts)
            if thread_ts is None:
                self._pin_roster(channel_id, posted.ts)
        return posted.ts

    def refresh_or_post_roster(self, channel_id: str) -> str:
        self._discover_recent_roster_messages(channel_id)
        roster_ts_values = self._remembered_roster_ts_values(channel_id)
        if roster_ts_values:
            agents = self.store.list_team_agents()
            statuses = self._roster_statuses(agents)
            text = _roster_text(agents, statuses)
            blocks = build_team_roster_blocks(agents, statuses)
            latest_roster_ts = roster_ts_values[-1]
            self.store.set_setting(SETTING_CHANNEL_ID, channel_id)
            self.store.set_setting(SETTING_ROSTER_TS, latest_roster_ts)
            try:
                self._pin_roster(channel_id, latest_roster_ts)
            except Exception:
                LOGGER.debug("failed to pin latest Slack roster message", exc_info=True)
            for roster_ts in roster_ts_values:
                try:
                    self.gateway.update_message(
                        channel_id,
                        roster_ts,
                        text,
                        blocks=blocks,
                    )
                except Exception:
                    LOGGER.debug(
                        "failed to update Slack roster message %s",
                        roster_ts,
                        exc_info=True,
                    )
            return latest_roster_ts
        return self.post_roster(channel_id)

    def _refresh_existing_roster(self, channel_id: str) -> None:
        self._discover_recent_roster_messages(channel_id)
        if self._remembered_roster_ts_values(channel_id):
            self.refresh_or_post_roster(channel_id)

    def _start_runtime_task(self, task: AgentTask, agent, thread: SlackThreadRef) -> bool:
        started = self.runtime.start_task(task, agent, thread) if self.runtime else True
        if started:
            self._refresh_existing_roster(thread.channel_id)
        return started

    def _remember_roster_message(self, channel_id: str, message_ts: str) -> None:
        self.store.set_setting(SETTING_CHANNEL_ID, channel_id)
        self.store.set_setting(SETTING_ROSTER_TS, message_ts)
        self.store.set_setting(_roster_message_setting_key(channel_id, message_ts), message_ts)

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

    def _pin_roster(self, channel_id: str, message_ts: str) -> None:
        self._pin_message(channel_id, message_ts, "roster message")

    def _pin_message(self, channel_id: str, message_ts: str, label: str) -> None:
        try:
            self.gateway.pin_message(channel_id, message_ts)
        except Exception:
            LOGGER.debug("failed to pin Slack %s", label, exc_info=True)

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

    def _try_update_message(self, channel_id: str, ts: str, text: str) -> bool:
        try:
            self.gateway.update_message(channel_id, ts, text)
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
            if MANAGED_RUN_STARTED_METADATA_KEY not in task.metadata:
                continue
            is_running = getattr(self.runtime, "is_task_running", None)
            if callable(is_running) and is_running(task.task_id):
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
        except Exception:
            LOGGER.debug("failed to abandon stale orphaned task %s", task.task_id, exc_info=True)

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

    def _hire_from_action(
        self,
        payload: dict,
        channel_id: str,
        roster_ts: str | None,
    ) -> None:
        count = int(payload.get("count") or 1)
        provider_text = payload.get("provider")
        provider = Provider(provider_text) if provider_text else None
        if not self._can_hire(count):
            thread_ts = roster_ts or self.store.get_setting(SETTING_ROSTER_TS)
            self._post_text(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread_ts),
                f"{AGENT_LIMIT_MESSAGE} Max team size is {MAX_TEAM_AGENTS}.",
            )
            return
        hired = self.hire_agents(count, provider)
        ts = roster_ts or self.store.get_setting(SETTING_ROSTER_TS)
        if not ts:
            ts = self.post_roster(channel_id)
        thread = SlackThreadRef(channel_id=channel_id, thread_ts=roster_ts or ts)
        for agent in hired:
            self.gateway.post_thread_reply(
                thread,
                format_agent_introduction(agent),
                persona=agent,
                icon_url=self._agent_icon_url(agent),
            )
        self._resume_pending_work_requests(channel_id)
        self.refresh_or_post_roster(channel_id)

    def _fire_from_action(
        self,
        payload: dict,
        channel_id: str,
        roster_ts: str | None,
    ) -> None:
        handle = payload.get("handle") or payload.get("agent_id")
        if handle:
            self.store.fire_team_agent(str(handle))
        ts = self.refresh_or_post_roster(channel_id)
        thread = SlackThreadRef(channel_id=channel_id, thread_ts=roster_ts or ts)
        if handle:
            self.gateway.post_thread_reply(thread, f"Removed @{str(handle).lstrip('@')}.")

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
            completed_tasks: list[AgentTask] = []
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
                        completed_tasks.append(thread_task)
            elif self.runtime:
                self.runtime.stop_task(str(task_id), AgentTaskStatus.DONE)
            else:
                self.store.update_agent_task_status(str(task_id), AgentTaskStatus.DONE)
            if not completed_tasks and task is not None:
                completed_tasks.append(task)
            for completed_task in completed_tasks:
                self._mark_task_complete(completed_task, thread, include_thread=True)
            if thread.thread_ts:
                self.gateway.post_thread_reply(thread, "Finished and freed up this agent.")
            self._resume_pending_work_requests(channel_id)
            self.refresh_or_post_roster(channel_id)

    def _external_session_finish_from_action(self, payload: dict, channel_id: str) -> None:
        provider_text = str(payload.get("provider") or "")
        session_id = str(payload.get("session_id") or "")
        try:
            provider = Provider(provider_text)
        except ValueError:
            return
        if not session_id:
            return
        key_suffix = f"{provider.value}.{session_id}"
        thread = self.store.get_slack_thread_for_session(
            provider,
            session_id,
            self.team_id,
            channel_id,
        )
        self.store.delete_setting(f"{EXTERNAL_SESSION_AGENT_PREFIX}{key_suffix}")
        self.store.delete_setting(f"{PENDING_EXTERNAL_SESSION_PREFIX}{key_suffix}")
        self.store.set_setting(
            f"{EXTERNAL_SESSION_IGNORED_PREFIX}{key_suffix}",
            utc_now().isoformat(),
        )
        if thread is not None:
            self.gateway.post_thread_reply(thread, "Finished and freed up this agent.")
        self.handle_external_session_occupancy_change(channel_id)

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
        active_task = self._active_thread_task_for_agent(task, channel_id, thread_ts)
        target_task = active_task or task
        if _is_stop_command(text):
            self._interrupt_thread_task(
                target_task,
                agent,
                SlackThreadRef(channel_id, thread_ts, task.parent_message_ts),
                message_ts,
            )
            return True
        if self._handle_thread_work_request(task, event, text, agent):
            return True
        if self._record_dependency_if_requested(task.task_id, event, text, agent):
            return True
        if self.runtime and self.runtime.send_to_task(
            target_task.task_id,
            _live_thread_followup_prompt(text),
        ):
            self._remember_request_message_for_task(target_task, message_ts)
            self._mark_message_in_progress(channel_id, message_ts)
            return True
        if agent:
            started = self._start_thread_followup(target_task, event, text, agent)
            if started:
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
        agent,
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
            latest = self.store.get_agent_task(task.task_id) or task
            resumed_queued = False
            if agent is not None:
                resumed_queued = self._send_queued_thread_followups_to_interrupted_task(
                    latest,
                    thread,
                )
            self.gateway.post_thread_reply(
                thread,
                (
                    "Interrupted the current run and sent the queued follow-ups."
                    if resumed_queued
                    else "Interrupted the current run. Send the next instruction here to continue."
                ),
            )
        else:
            self.gateway.post_thread_reply(
                thread,
                "No active run is currently running for this task.",
            )
        self.refresh_or_post_roster(thread.channel_id)

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
        if assigned_agent_id and self.store.get_team_agent(assigned_agent_id):
            return True
        if session.status == SessionStatus.ACTIVE:
            return True
        external_busy_agent_ids = self._external_busy_agent_ids()
        available = [
            agent
            for agent in self.store.idle_team_agents()
            if agent.provider_preference == session.provider
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
        excluded_agent_ids = set(self._external_busy_agent_ids())
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
        multi_requests = _multi_specific_work_requests(text, active_agents)
        if multi_requests:
            extra_metadata = self._thread_task_metadata(parent_task, channel_id, thread_ts)
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
            current_thread=SlackThreadRef(channel_id, thread_ts),
        )
        same_thread_agent = self._same_thread_requested_agent(request, channel_id, thread_ts)
        if same_thread_agent is not None:
            started = self._start_same_thread_agent_followup(
                request,
                same_thread_agent,
                SlackThreadRef(channel_id, thread_ts),
                requested_by_slack_user=event.get("user"),
                author_agent=parent_agent,
                context_task=parent_task,
                request_message_ts=event.get("ts"),
            )
            if started:
                self._mark_message_in_progress(channel_id, event.get("ts"))
            return started
        delegation = _thread_delegation_intent(request.prompt, parent_agent, active_agents)
        excluded_agent_ids: set[str] = set(self._external_busy_agent_ids())
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
            channel_id,
            requested_by_slack_user=event.get("user"),
            author_agent=parent_agent,
            extra_metadata=extra_metadata,
            exclude_agent_ids=excluded_agent_ids,
        )
        if result is None:
            self._post_assignment_unavailable(
                SlackReplyTarget(channel_id=channel_id, thread_ts=thread_ts),
                request,
                requested_by_slack_user=event.get("user"),
                author_agent=parent_agent,
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
        if self._handle_agent_authored_specific_request(task, agent, thread, text, message_ts):
            return True
        active_agents = self.store.list_team_agents()
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
                *self._external_busy_agent_ids(),
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
            exclude_agent_ids=self._external_busy_agent_ids(),
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
                exclude_agent_ids=self._external_busy_agent_ids(),
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
            blocks = build_task_thread_blocks(result.task, result.agent)
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
        return self.session_bridge.send_to_session(
            session,
            delivery_text or request.prompt,
            thread,
            slack_user=requested_by_slack_user,
        )

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
            LOGGER.debug(
                "queueing same-thread continuation for task %s; worker is still running",
                previous_task.task_id,
            )
            self._queue_running_task_followup(
                previous_task,
                request,
                requested_by_slack_user=requested_by_slack_user,
                request_message_ts=request_message_ts,
            )
            return True
        metadata = self._thread_task_metadata(previous_task, thread.channel_id, thread.thread_ts)
        metadata[ASSIGNMENT_PROMPT_METADATA_KEY] = _task_assignment_prompt(previous_task)
        if request_message_ts:
            metadata["request_message_ts"] = request_message_ts
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

    def _queue_running_task_followup(
        self,
        task: AgentTask,
        request: WorkRequest,
        *,
        requested_by_slack_user: str | None,
        request_message_ts: str | None,
    ) -> AgentTask:
        metadata = dict(task.metadata)
        queued = _queued_thread_followups(metadata)
        queued.append(
            {
                "prompt": request.prompt,
                "message_ts": request_message_ts,
                "requested_by_slack_user": requested_by_slack_user,
                "created_at": utc_now().isoformat(),
            }
        )
        metadata[QUEUED_THREAD_FOLLOWUPS_METADATA_KEY] = queued[-20:]
        updated = replace(task, metadata=metadata, updated_at=utc_now())
        self.store.upsert_agent_task(updated)
        return self.store.get_agent_task(task.task_id) or updated

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
        for message_ts in message_ts_values:
            self._mark_message_complete(thread.channel_id, message_ts)
        return self.store.get_agent_task(task.task_id) or updated

    def _resume_queued_thread_followups(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
    ) -> bool:
        if self.runtime is None:
            return False
        latest = self.store.get_agent_task(task.task_id) or task
        queued = _queued_thread_followups(latest.metadata)
        if not queued:
            return False
        prompts = [item["prompt"] for item in queued if isinstance(item.get("prompt"), str)]
        if not prompts:
            return False
        metadata = dict(latest.metadata)
        metadata.pop(QUEUED_THREAD_FOLLOWUPS_METADATA_KEY, None)
        message_ts_values = _queued_thread_followup_message_ts_values(queued)
        if message_ts_values:
            metadata[ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_VALUES_METADATA_KEY] = message_ts_values
            metadata.pop(ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_METADATA_KEY, None)
        else:
            metadata.pop(ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_VALUES_METADATA_KEY, None)
            metadata.pop(ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_METADATA_KEY, None)
        for message_ts in message_ts_values:
            self._mark_message_in_progress(thread.channel_id, message_ts)
        metadata[ASSIGNMENT_PROMPT_METADATA_KEY] = _task_assignment_prompt(latest)
        prompt = _queued_followup_prompt(prompts)
        metadata = self._with_linked_thread_context(
            metadata,
            prompt,
            current_thread=thread,
        )
        requested_by_slack_user = (
            _last_queued_requested_by(queued) or latest.requested_by_slack_user
        )
        resumed = replace(
            latest,
            prompt=prompt,
            status=AgentTaskStatus.ACTIVE,
            requested_by_slack_user=requested_by_slack_user,
            updated_at=utc_now(),
            metadata=metadata,
        )
        self.store.upsert_agent_task(resumed)
        self._restore_task_action_buttons_if_active(resumed)
        started = self.runtime.start_task(resumed, agent, thread)
        if not started:
            self.store.upsert_agent_task(latest)
        return started

    def _send_queued_thread_followups_to_interrupted_task(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
    ) -> bool:
        if self.runtime is None:
            return False
        send = getattr(self.runtime, "send_to_interrupted_task", None)
        if not callable(send):
            return False
        latest = self.store.get_agent_task(task.task_id) or task
        queued = _queued_thread_followups(latest.metadata)
        if not queued:
            return False
        prompts = [item["prompt"] for item in queued if isinstance(item.get("prompt"), str)]
        if not prompts:
            return False
        prompt = _queued_followup_prompt(prompts)
        if not send(latest.task_id, prompt):
            return False
        metadata = dict(latest.metadata)
        metadata.pop(QUEUED_THREAD_FOLLOWUPS_METADATA_KEY, None)
        message_ts_values = _queued_thread_followup_message_ts_values(queued)
        if message_ts_values:
            metadata[ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_VALUES_METADATA_KEY] = message_ts_values
            metadata.pop(ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_METADATA_KEY, None)
        metadata[ASSIGNMENT_PROMPT_METADATA_KEY] = _task_assignment_prompt(latest)
        metadata = self._with_linked_thread_context(
            metadata,
            prompt,
            current_thread=thread,
        )
        updated = replace(
            latest,
            prompt=prompt,
            requested_by_slack_user=_last_queued_requested_by(queued)
            or latest.requested_by_slack_user,
            updated_at=utc_now(),
            metadata=metadata,
        )
        self.store.upsert_agent_task(updated)
        for message_ts in message_ts_values:
            self._mark_message_in_progress(thread.channel_id, message_ts)
        return True

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
        task = self._complete_active_thread_followup(task, thread)
        if self._resume_queued_thread_followups(task, agent, thread):
            return
        self.refresh_or_post_roster(thread.channel_id)
        delegate_to_agent_id = task.metadata.get("delegate_to_agent_id")
        delegate_prompt = task.metadata.get("delegate_prompt")
        if not isinstance(delegate_to_agent_id, str) or not isinstance(delegate_prompt, str):
            return
        if not delegate_prompt.strip() or not thread.thread_ts:
            return
        target_agent = self.store.get_team_agent(delegate_to_agent_id)
        if target_agent is None:
            return
        if target_agent.agent_id == agent.agent_id:
            return
        if target_agent.agent_id in self._external_busy_agent_ids():
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

    def handle_runtime_agent_control(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        signal: str,
    ) -> bool:
        if signal == AGENT_THREAD_DONE_SIGNAL:
            return self._complete_task_thread(thread.channel_id, thread.thread_ts)
        if is_agent_timer_signal(signal):
            return self._schedule_agent_timer(task, agent, thread, signal)
        if is_agent_schedule_signal(signal):
            return self._schedule_user_work_from_agent(task, agent, thread, signal)
        return False

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

    def fire_due_scheduled_timer(self, timer: ScheduledTimer) -> bool:
        agent = self.store.get_team_agent(timer.agent_id)
        if agent is None:
            self.store.update_scheduled_timer_status(
                timer.timer_id,
                ScheduledTimerStatus.CANCELLED,
            )
            return True
        previous_task = self.store.get_agent_task(timer.task_id)
        if previous_task is None or previous_task.status in {
            AgentTaskStatus.DONE,
            AgentTaskStatus.CANCELLED,
        }:
            self.store.update_scheduled_timer_status(
                timer.timer_id,
                ScheduledTimerStatus.CANCELLED,
            )
            return True
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
        for thread_task in self.store.list_agent_tasks(include_done=True):
            if (
                thread_task.channel_id == channel_id
                and thread_task.thread_ts == thread_ts
                and thread_task.status in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}
            ):
                if self.runtime:
                    self.runtime.stop_task(thread_task.task_id, AgentTaskStatus.DONE)
                else:
                    self.store.update_agent_task_status(thread_task.task_id, AgentTaskStatus.DONE)
                self._mark_task_complete(thread_task, thread, include_thread=True)
                completed += 1

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
        if (
            not completed
            and not cancelled
            and not cancelled_timers
            and not cancelled_scheduled_work
        ):
            return False
        self._resume_pending_work_requests(channel_id)
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
        }
        if parent_task.metadata.get("cwd"):
            metadata["cwd"] = parent_task.metadata["cwd"]
        context = self._thread_context(channel_id, thread_ts)
        prompt_context = f"Original task: {_task_assignment_prompt(parent_task)}"
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
            return SlackThreadRef(
                channel_id=channel_id,
                thread_ts=user_thread_ts,
                message_ts=message_ts,
            )
        linked = self._resolve_linked_thread_ref(text)
        if linked is not None and linked.channel_id == channel_id and linked.thread_ts:
            return SlackThreadRef(
                channel_id=channel_id,
                thread_ts=linked.thread_ts,
                message_ts=message_ts,
            )
        return SlackThreadRef(
            channel_id=channel_id,
            thread_ts=message_ts,
            message_ts=message_ts,
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
        if include_thread and task.thread_ts:
            self._mark_message_complete(thread.channel_id, task.thread_ts)
        for message_ts in _task_request_message_ts_values(task):
            if include_thread and message_ts == task.thread_ts:
                continue
            self._clear_message_status_reactions(thread.channel_id, message_ts)
        self._reconcile_pending_thread_reactions(task, thread)

    def _reconcile_pending_thread_reactions(
        self,
        task: AgentTask,
        thread: SlackThreadRef,
    ) -> None:
        # Backstop: surface and clear thread messages that the bot acknowledged
        # or marked in-progress but never finalized. Otherwise the user sees
        # :eyes: / :hourglass: lingering forever after the task closes.
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
        pending_reactions = {TASK_REACTION_ACKNOWLEDGED, TASK_REACTION_IN_PROGRESS}
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
        self._clear_message_status_reactions(
            channel_id,
            message_ts,
            except_reaction=reaction_name,
        )
        try:
            self.gateway.add_reaction(channel_id, message_ts, reaction_name)
        except Exception:
            LOGGER.debug("failed to add Slack task status reaction", exc_info=True)

    def _clear_message_status_reactions(
        self,
        channel_id: str,
        message_ts: str | None,
        *,
        except_reaction: str | None = None,
    ) -> None:
        if not message_ts:
            return
        remove_reaction = getattr(self.gateway, "remove_reaction", None)
        if callable(remove_reaction):
            for reaction in TASK_STATUS_REACTIONS:
                if reaction == except_reaction:
                    continue
                try:
                    remove_reaction(channel_id, message_ts, reaction)
                except Exception:
                    LOGGER.debug("failed to remove stale Slack task reaction", exc_info=True)

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
                    _task_assignment_prompt(task),
                    task.requested_by_slack_user,
                    dangerous_mode=_task_dangerous_mode(task),
                ),
                blocks=build_task_thread_blocks(task, agent, include_actions=False),
            )
        except Exception:
            LOGGER.debug("failed to remove resolved task action buttons", exc_info=True)

    def _restore_task_action_buttons_if_active(self, task: AgentTask) -> None:
        if task.status not in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}:
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
                    _task_assignment_prompt(task),
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
        return not self._has_work_for_request_message(channel_id, message_ts)

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
        return False

    def _has_work_for_request_message(self, channel_id: str, message_ts: str) -> bool:
        if self._has_task_for_request_message(channel_id, message_ts):
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
        now: Callable[[], float] | None = None,
    ):
        self.store = store
        self.gateway = gateway
        self.controller = controller
        self.team_id = team_id
        self.poll_seconds = poll_seconds
        self.sleep_gap_seconds = sleep_gap_seconds
        self.grace_seconds = grace_seconds
        self.now = now or time.time
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_awake_unix: float | None = None

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
        if previous is not None and now - previous >= self.sleep_gap_seconds:
            oldest = _slack_ts_from_unix(max(0.0, previous - self.grace_seconds))
            recovered = self.recover_since(oldest)
        self._last_awake_unix = now
        self.store.set_setting(SETTING_SLACK_BACKFILL_LAST_AWAKE, f"{now:.6f}")
        return recovered

    def recover_since(self, oldest: str) -> int:
        channel_id = self.controller._configured_agent_channel_id()
        if not channel_id:
            return 0
        events = self._events_since(channel_id, oldest)
        recovered = 0
        for event in sorted(events, key=lambda item: _slack_ts_sort_key(item.get("ts"))):
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
        return recovered

    def _events_since(self, channel_id: str, oldest: str) -> list[dict]:
        events_by_ts: dict[str, dict] = {}
        thread_ts_values = self._known_thread_ts(channel_id)
        try:
            channel_messages = self.gateway.channel_messages(
                channel_id,
                oldest=oldest,
                limit=SLACK_BACKFILL_FETCH_LIMIT,
            )
        except Exception:
            LOGGER.exception("failed to fetch Slack channel history for backfill")
            return []
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

        for thread_ts in sorted(thread_ts_values, key=_slack_ts_sort_key):
            try:
                thread_messages = self.gateway.thread_messages(
                    channel_id,
                    thread_ts,
                    oldest=oldest,
                    limit=SLACK_BACKFILL_FETCH_LIMIT,
                )
            except Exception:
                LOGGER.debug(
                    "failed to fetch Slack thread history for backfill: %s:%s",
                    channel_id,
                    thread_ts,
                    exc_info=True,
                )
                continue
            for message in thread_messages:
                event = _history_message_event(channel_id, message, default_thread_ts=thread_ts)
                message_ts = event.get("ts")
                if isinstance(message_ts, str):
                    events_by_ts[message_ts] = event
        return list(events_by_ts.values())

    def _known_thread_ts(self, channel_id: str) -> set[str]:
        thread_ts_values = {
            task.thread_ts
            for task in self.store.list_agent_tasks(include_done=True)
            if task.channel_id == channel_id and task.thread_ts
        }
        for session in self.store.list_sessions():
            thread = self.store.get_slack_thread_for_session(
                session.provider,
                session.session_id,
                self.team_id,
                channel_id,
            )
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


class SocketModeSlackApp:
    def __init__(self, config: AppConfig):
        if not config.slack.bot_token or not config.slack.app_token:
            raise ValueError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required")
        self.config = config
        self.store = Store(config.state_db)
        self.store.init_schema()
        self.gateway = SlackGateway(config.slack.bot_token)
        auth = self.gateway.auth_test()
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
            poll_seconds=max(config.poll_seconds, 2.0),
            codex_app_server_url=codex_app_server_url,
            on_external_session_occupancy_change=(
                self.controller.handle_external_session_occupancy_change
            ),
            home=config.home,
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
        all_stopped = self.cpu_watchdog.stop() and all_stopped
        self.awake_keeper.stop()
        all_stopped = self.claude_permission_auto_resolver.stop() and all_stopped
        all_stopped = self.slack_message_backfill.stop() and all_stopped
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
            try:
                self.handle_request(request)
            except Exception:
                LOGGER.exception("failed to handle Slack Socket Mode request")

        client.socket_mode_request_listeners.append(listener)
        client.connect()
        shutdown = threading.Event()

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
            while not shutdown.is_set():
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
                return self.controller.handle_view_submission(payload)
            return None
        if request.type == "events_api":
            self.controller.handle_event(request.payload)
        elif request.type == "slash_commands":
            self.controller.handle_slash_command(request.payload)
        return None


def run_slack_app(config: AppConfig | None = None) -> int:
    app = SocketModeSlackApp(config or load_config_from_env())
    try:
        app.run_forever()
    finally:
        app.close()
    return 0


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


def _roster_discovery_setting_key(channel_id: str) -> str:
    return f"{SETTING_ROSTER_DISCOVERY_PREFIX}{channel_id}"


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


def _event_has_slackgentic_progress_reaction(event: dict) -> bool:
    reactions = event.get("reactions")
    if not isinstance(reactions, list):
        return False
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        if reaction.get("name") in {
            "hourglass_flowing_sand",
            "eyes",
            "white_check_mark",
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


def _queued_thread_followups(metadata: dict[str, object]) -> list[dict[str, object]]:
    raw = metadata.get(QUEUED_THREAD_FOLLOWUPS_METADATA_KEY)
    if not isinstance(raw, list):
        return []
    queued: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        queued.append(dict(item))
    return queued


def _queued_thread_followup_message_ts_values(queued: list[dict[str, object]]) -> list[str]:
    values: list[str] = []
    for item in queued:
        message_ts = item.get("message_ts")
        if isinstance(message_ts, str) and message_ts:
            values.append(message_ts)
    return list(dict.fromkeys(values))


def _active_thread_followup_message_ts_values(metadata: dict[str, object]) -> list[str]:
    values: list[str] = []
    raw_values = metadata.get(ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_VALUES_METADATA_KEY)
    if isinstance(raw_values, list):
        values.extend(item for item in raw_values if isinstance(item, str) and item)
    raw_value = metadata.get(ACTIVE_THREAD_FOLLOWUP_MESSAGE_TS_METADATA_KEY)
    if isinstance(raw_value, str) and raw_value:
        values.append(raw_value)
    return list(dict.fromkeys(values))


def _queued_followup_prompt(prompts: list[str]) -> str:
    if len(prompts) == 1:
        question_instruction = (
            " If it asks a direct question, answer it explicitly before continuing "
            "or scheduling a delayed follow-up."
            if _looks_like_direct_question(prompts[0])
            else ""
        )
        return (
            "The user sent this follow-up while you were already working. "
            f"Treat it as the latest instruction and continue.{question_instruction}\n\n"
            f"{prompts[0]}"
        )
    formatted = "\n\n".join(f"{index}. {prompt}" for index, prompt in enumerate(prompts, start=1))
    question_instruction = (
        " If any follow-up asks a direct question, answer it explicitly before continuing "
        "or scheduling a delayed follow-up."
        if any(_looks_like_direct_question(prompt) for prompt in prompts)
        else ""
    )
    return (
        "The user sent these follow-ups while you were already working. "
        "Treat later messages as newer instructions and continue from the latest direction."
        f"{question_instruction}\n\n"
        f"{formatted}"
    )


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


def _last_queued_requested_by(queued: list[dict[str, object]]) -> str | None:
    for item in reversed(queued):
        requested_by = item.get("requested_by_slack_user")
        if isinstance(requested_by, str) and requested_by:
            return requested_by
    return None


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


def _task_assignment_prompt(task: AgentTask) -> str:
    prompt = task.metadata.get(ASSIGNMENT_PROMPT_METADATA_KEY)
    return prompt if isinstance(prompt, str) and prompt.strip() else task.prompt


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
        f"Scheduled `{scheduled.schedule_id}`: {target} will run "
        f"`{request.prompt}` {description}. "
        f"Next run: `{_format_schedule_timestamp(scheduled.next_run_at)}`."
    )


def _format_schedule_timestamp(value) -> str:
    return value.isoformat(timespec="minutes")


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
    canonical_text = canonicalize_agent_mentions(text, agents)
    known_handles = [agent.handle for agent in agents]
    request = parse_work_request(canonical_text, known_handles)
    if request is not None:
        return request
    cleaned = re.sub(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", "", canonical_text).strip()
    if not cleaned:
        return None
    return parse_work_request(f"somebody {cleaned}", known_handles)


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


def _is_subtask(task: AgentTask) -> bool:
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
