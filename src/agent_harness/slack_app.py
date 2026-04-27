from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from agent_harness.assignment import assign_channel_work_request, assign_work_request
from agent_harness.codex_app_server import CodexAppServerManager
from agent_harness.config import AppConfig, load_config_from_env
from agent_harness.models import (
    AgentTask,
    AgentTaskStatus,
    AssignmentMode,
    Provider,
    SessionDependency,
    SlackThreadRef,
    WorkRequest,
)
from agent_harness.providers import ClaudeProvider, CodexProvider
from agent_harness.routing import parse_work_request
from agent_harness.session_bridge import ExternalSessionBridge
from agent_harness.session_mirror import (
    HUMAN_DISPLAY_NAME_SETTING,
    HUMAN_IMAGE_URL_SETTING,
    SessionMirror,
)
from agent_harness.slack import (
    build_setup_modal,
    build_task_thread_blocks,
    build_team_roster_blocks,
    decode_action_value,
    is_dependency_intent,
    parse_thread_ref,
)
from agent_harness.slack_agent_requests import AGENT_REQUEST_ACTIONS
from agent_harness.slack_client import SlackGateway
from agent_harness.store import Store
from agent_harness.task_runtime import ManagedTaskRuntime
from agent_harness.team import (
    build_initial_model_team,
    build_initialization_messages,
    choose_reaction,
    format_agent_assignment,
    format_agent_handoff_assignment,
    format_agent_handoff_request,
    format_agent_introduction,
    hire_team_agents,
)
from agent_harness.team_commands import FireCommand, HireCommand, RosterCommand, parse_team_command
from agent_harness.usage import collect_daily_usage, day_string, format_daily_usage

LOGGER = logging.getLogger(__name__)
SETTING_CHANNEL_ID = "slack.channel_id"
SETTING_ROSTER_TS = "slack.roster_ts"
SETTING_USAGE_TS_PREFIX = "slack.usage_ts."
SETTING_HUMAN_USER_ID = "slack.human_user_id"


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
    ):
        self.store = store
        self.gateway = gateway
        self.default_channel_id = default_channel_id
        self.runtime = runtime
        self.home = home
        self.ignored_bot_id = ignored_bot_id
        self.session_bridge = session_bridge
        self.team_id = team_id

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
        elif action_name in AGENT_REQUEST_ACTIONS and self.session_bridge is not None:
            self.session_bridge.handle_agent_request_block_action(decoded, channel_id, message_ts)

    def handle_view_submission(self, payload: dict) -> dict | None:
        view = payload.get("view") or {}
        if view.get("callback_id") != "setup.initial":
            return None
        values = view.get("state", {}).get("values", {})
        channel_name = _view_plain_value(values, "channel_name", "value") or "agents"
        visibility = _view_selected_value(values, "visibility", "value") or "private"
        codex_count = _view_int_value(values, "codex_count", "value", default=5)
        claude_count = _view_int_value(values, "claude_count", "value", default=5)

        channel_id = self.gateway.create_channel(
            _normalize_channel_name(channel_name),
            is_private=visibility != "public",
        )
        user_id = (payload.get("user") or {}).get("id")
        if user_id:
            self._remember_human_user(user_id, payload.get("user"))
            self.gateway.invite_users(channel_id, [user_id])
        self.store.set_setting(SETTING_CHANNEL_ID, channel_id)
        self._ensure_initial_team(codex_count, claude_count)
        self.post_roster(channel_id)
        self.post_initial_introductions(channel_id)
        return None

    def handle_slash_command(self, payload: dict) -> None:
        text = (payload.get("text") or "").strip()
        channel_id = payload.get("channel_id") or self.default_channel_id
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
                self.gateway.open_view(trigger_id, build_setup_modal())
            return
        if _is_usage_request(text):
            self.publish_usage(channel_id)
            return
        command = parse_team_command(text)
        if command:
            self.handle_team_command(command, SlackReplyTarget(channel_id=channel_id))
            return
        self.refresh_or_post_roster(channel_id)

    def handle_event(self, payload: dict) -> None:
        event = payload.get("event") or {}
        event_type = event.get("type")
        if event_type not in {"message", "app_mention"}:
            return
        if event.get("subtype"):
            return
        bot_id = event.get("bot_id")
        if bot_id and (self.ignored_bot_id is None or bot_id == self.ignored_bot_id):
            return
        channel_id = event.get("channel")
        if not channel_id or not self._is_agent_channel(channel_id):
            return
        if self.store.is_mirrored_slack_message(channel_id, event.get("ts")):
            return
        self._remember_human_user(event.get("user"), event.get("user_profile"))
        text = event.get("text") or ""
        if self._handle_external_session_thread_reply(event, channel_id, text):
            return
        if _is_usage_request(text):
            self.publish_usage(channel_id)
            return
        target = SlackReplyTarget(
            channel_id=channel_id,
            thread_ts=event.get("thread_ts"),
        )
        command = parse_team_command(text)
        if command:
            self.handle_team_command(command, target)
            return
        if self._handle_task_thread_reply(event, channel_id, text):
            return
        result = assign_channel_work_request(
            self.store,
            text,
            channel_id,
            requested_by_slack_user=event.get("user"),
        )
        if result is None:
            return
        blocks = build_task_thread_blocks(result.task, result.agent)
        thread = self.gateway.post_task_parent(
            channel_id,
            format_agent_assignment(result.agent, result.request.prompt, event.get("user")),
            result.agent,
            blocks=blocks,
        )
        self.store.update_agent_task_thread(
            result.task.task_id,
            thread.thread_ts,
            thread.message_ts,
        )
        task = self.store.get_agent_task(result.task.task_id) or result.task
        if self.runtime:
            self.runtime.start_task(task, result.agent, thread)
        self._react(channel_id, event.get("ts"), result.agent, result.request.prompt)

    def handle_team_command(
        self,
        command: HireCommand | FireCommand | RosterCommand,
        target: SlackReplyTarget,
    ) -> None:
        if isinstance(command, HireCommand):
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
            )
        else:
            self.gateway.post_session_parent(target.channel_id, text, agent)

    def hire_agents(self, count: int, provider: Provider | None = None):
        all_agents = self.store.list_team_agents(include_fired=True)
        active_agents = self.store.list_team_agents()
        hired = hire_team_agents(
            all_agents,
            count,
            provider,
            start_sort_order=self.store.next_team_sort_order(),
            balance_agents=active_agents,
        )
        for agent in hired:
            self.store.upsert_team_agent(agent)
        return hired

    def post_roster(
        self,
        channel_id: str,
        thread_ts: str | None = None,
        remember: bool = True,
    ) -> str:
        agents = self.store.list_team_agents()
        posted = self.gateway.post_message(
            channel_id,
            _roster_text(agents),
            blocks=build_team_roster_blocks(agents),
            thread_ts=thread_ts,
        )
        if remember:
            self.store.set_setting(SETTING_CHANNEL_ID, channel_id)
            self.store.set_setting(SETTING_ROSTER_TS, posted.ts)
        return posted.ts

    def refresh_or_post_roster(self, channel_id: str) -> str:
        roster_ts = self.store.get_setting(SETTING_ROSTER_TS)
        if roster_ts:
            agents = self.store.list_team_agents()
            self.gateway.update_message(
                channel_id,
                roster_ts,
                _roster_text(agents),
                blocks=build_team_roster_blocks(agents),
            )
            return roster_ts
        return self.post_roster(channel_id)

    def post_initial_introductions(self, channel_id: str) -> None:
        agents = self.store.list_team_agents()
        messages = build_initialization_messages(agents)
        if messages:
            self.gateway.post_team_initialization(channel_id, agents, messages)

    def publish_usage(self, channel_id: str) -> str:
        day = day_string("today")
        text = format_daily_usage(day, collect_daily_usage(day, home=self.home))
        setting_key = f"{SETTING_USAGE_TS_PREFIX}{day}"
        ts = self.store.get_setting(setting_key)
        if ts:
            self.gateway.update_message(channel_id, ts, text)
            return ts
        posted = self.gateway.post_message(channel_id, text)
        self.store.set_setting(setting_key, posted.ts)
        return posted.ts

    def cancel_orphaned_active_tasks(self) -> int:
        cancelled = 0
        for task in self.store.list_agent_tasks():
            if task.status != AgentTaskStatus.ACTIVE:
                continue
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
            cancelled += 1
            if not task.thread_ts:
                continue
            try:
                self.gateway.post_thread_reply(
                    SlackThreadRef(task.channel_id, task.thread_ts),
                    (
                        "The local Slackgentic daemon restarted, so I detached this stale "
                        "task process. Start a fresh task if you want it rerun."
                    ),
                )
            except Exception:
                LOGGER.debug("failed to notify stale task thread", exc_info=True)
        return cancelled

    def _ensure_initial_team(self, codex_count: int, claude_count: int) -> None:
        if self.store.list_team_agents(include_fired=True):
            return
        for agent in build_initial_model_team(codex_count, claude_count):
            self.store.upsert_team_agent(agent)

    def _hire_from_action(
        self,
        payload: dict,
        channel_id: str,
        roster_ts: str | None,
    ) -> None:
        count = int(payload.get("count") or 1)
        provider_text = payload.get("provider")
        provider = Provider(provider_text) if provider_text else None
        hired = self.hire_agents(count, provider)
        ts = self.refresh_or_post_roster(channel_id)
        thread = SlackThreadRef(channel_id=channel_id, thread_ts=roster_ts or ts)
        for agent in hired:
            self.gateway.post_thread_reply(thread, format_agent_introduction(agent), persona=agent)

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
        if action == "task.done":
            if self.runtime:
                self.runtime.stop_task(str(task_id), AgentTaskStatus.DONE)
            else:
                self.store.update_agent_task_status(str(task_id), AgentTaskStatus.DONE)
            if thread.thread_ts:
                self.gateway.post_thread_reply(thread, "Marked done.")
        elif action == "task.cancel":
            if self.runtime:
                self.runtime.stop_task(str(task_id), AgentTaskStatus.CANCELLED)
            else:
                self.store.update_agent_task_status(str(task_id), AgentTaskStatus.CANCELLED)
            if thread.thread_ts:
                self.gateway.post_thread_reply(thread, "Cancelled.")
        elif action == "task.pause":
            if self.runtime:
                self.runtime.stop_task(str(task_id), AgentTaskStatus.QUEUED)
            else:
                self.store.update_agent_task_status(str(task_id), AgentTaskStatus.QUEUED)
            if thread.thread_ts:
                self.gateway.post_thread_reply(thread, "Paused. The task is queued again.")

    def _handle_task_thread_reply(self, event: dict, channel_id: str, text: str) -> bool:
        thread_ts = event.get("thread_ts")
        message_ts = event.get("ts")
        if not thread_ts or thread_ts == message_ts:
            return False
        task = self.store.get_original_agent_task_by_thread(channel_id, thread_ts)
        if task is None:
            return False
        agent = self.store.get_team_agent(task.agent_id)
        if agent:
            self._react(channel_id, message_ts, agent, text)
        if self._handle_thread_work_request(task, event, text, agent):
            return True
        if self._record_dependency_if_requested(task.task_id, event, text, agent):
            return True
        if self.runtime and self.runtime.send_to_task(task.task_id, text):
            return True
        if agent and task.status in {
            AgentTaskStatus.ACTIVE,
            AgentTaskStatus.DONE,
            AgentTaskStatus.CANCELLED,
        }:
            return self._start_thread_followup(task, event, text, agent)
        if task.status in {AgentTaskStatus.DONE, AgentTaskStatus.CANCELLED}:
            return True
        self.gateway.post_thread_reply(
            SlackThreadRef(channel_id, thread_ts),
            "I found the task thread, but no managed process is attached right now.",
            persona=agent,
        )
        return True

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
        if self.session_bridge is None:
            self.gateway.post_thread_reply(
                SlackThreadRef(channel_id, thread_ts),
                "I found the external session thread, but no session bridge is configured.",
            )
            return True
        return self.session_bridge.send_to_session(
            session,
            text,
            SlackThreadRef(channel_id, thread_ts),
            slack_user=event.get("user"),
        )

    def _handle_thread_work_request(
        self,
        parent_task: AgentTask,
        event: dict,
        text: str,
        parent_agent,
    ) -> bool:
        active_agents = self.store.list_team_agents()
        request = parse_work_request(text, [agent.handle for agent in active_agents])
        if request is None:
            return False
        channel_id = event["channel"]
        thread_ts = event["thread_ts"]
        extra_metadata = self._thread_task_metadata(parent_task, channel_id, thread_ts)
        delegation = _thread_delegation_intent(request.prompt, parent_agent, active_agents)
        excluded_agent_ids: set[str] = set()
        if delegation:
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
            return True
        posted = self.gateway.post_thread_reply(
            SlackThreadRef(channel_id, thread_ts),
            format_agent_assignment(result.agent, result.request.prompt, event.get("user")),
            persona=result.agent,
        )
        self.store.update_agent_task_thread(result.task.task_id, thread_ts, posted.ts)
        task = self.store.get_agent_task(result.task.task_id) or result.task
        if self.runtime:
            self.runtime.start_task(task, result.agent, SlackThreadRef(channel_id, thread_ts))
        return True

    def handle_runtime_agent_message(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
        text: str,
    ) -> bool:
        if not thread.thread_ts:
            return False
        if not _is_agent_authored_review_request(text):
            return False
        active_agents = self.store.list_team_agents()
        request = parse_work_request(text, [item.handle for item in active_agents])
        if request is None:
            return False
        metadata = self._thread_task_metadata(task, thread.channel_id, thread.thread_ts)
        metadata["delegate_to_agent_id"] = agent.agent_id
        metadata["delegate_prompt"] = (
            "Continue the original task using @{sender_handle}'s review above. "
            "Address any required changes, then give the user the final result "
            "or a concise status update. If no changes are needed, say so."
        )
        metadata["delegate_visible_prompt"] = "continue using my review above."
        result = assign_work_request(
            self.store,
            request,
            thread.channel_id,
            requested_by_slack_user=task.requested_by_slack_user,
            author_agent=agent,
            extra_metadata=metadata,
            exclude_agent_ids={agent.agent_id},
        )
        if result is None:
            return True
        posted = self.gateway.post_thread_reply(
            thread,
            format_agent_handoff_assignment(result.agent, agent, result.request.prompt),
            persona=result.agent,
        )
        self.store.update_agent_task_thread(result.task.task_id, thread.thread_ts, posted.ts)
        reviewer_task = self.store.get_agent_task(result.task.task_id) or result.task
        if self.runtime:
            self.runtime.start_task(reviewer_task, result.agent, thread)
        return True

    def handle_runtime_task_done(
        self,
        task: AgentTask,
        agent,
        thread: SlackThreadRef,
    ) -> None:
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
        self.gateway.post_thread_reply(
            thread,
            format_agent_handoff_request(agent, target_agent, visible_prompt),
            persona=agent,
        )
        posted = self.gateway.post_thread_reply(
            thread,
            format_agent_handoff_assignment(target_agent, agent, result.request.prompt),
            persona=target_agent,
        )
        self.store.update_agent_task_thread(result.task.task_id, thread.thread_ts, posted.ts)
        delegated_task = self.store.get_agent_task(result.task.task_id) or result.task
        if self.runtime:
            self.runtime.start_task(delegated_task, target_agent, thread)

    def _start_thread_followup(self, parent_task: AgentTask, event: dict, text: str, agent) -> bool:
        channel_id = event["channel"]
        thread_ts = event["thread_ts"]
        request = parse_work_request(f"@{agent.handle} {text}", [agent.handle])
        if request is None:
            return False
        result = assign_work_request(
            self.store,
            request,
            channel_id,
            requested_by_slack_user=event.get("user"),
            extra_metadata=self._thread_task_metadata(parent_task, channel_id, thread_ts),
            force_agent=agent,
        )
        if result is None:
            return False
        posted = self.gateway.post_thread_reply(
            SlackThreadRef(channel_id, thread_ts),
            format_agent_assignment(agent, result.request.prompt, event.get("user")),
            persona=agent,
        )
        self.store.update_agent_task_thread(result.task.task_id, thread_ts, posted.ts)
        task = self.store.get_agent_task(result.task.task_id) or result.task
        if self.runtime:
            self.runtime.start_task(task, agent, SlackThreadRef(channel_id, thread_ts))
        return True

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
        if context:
            metadata["thread_context"] = context
        return metadata

    def _thread_context(self, channel_id: str, thread_ts: str) -> str | None:
        try:
            messages = self.gateway.thread_messages(channel_id, thread_ts, limit=20)
        except Exception:
            LOGGER.debug("failed to fetch Slack thread context", exc_info=True)
            return None
        lines: list[str] = []
        for message in messages[-12:]:
            text = (message.get("text") or "").strip()
            if not text:
                continue
            author = message.get("username") or message.get("user") or "Slack"
            lines.append(f"{author}: {text}")
        context = "\n".join(lines)
        return context[-6000:] if context else None

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

    def _react(
        self,
        channel_id: str,
        message_ts: str | None,
        agent,
        text: str,
    ) -> None:
        if not message_ts or not agent:
            return
        try:
            self.gateway.add_reaction(channel_id, message_ts, choose_reaction(agent, text))
        except Exception:
            LOGGER.debug("failed to add Slack reaction", exc_info=True)

    def _is_agent_channel(self, channel_id: str) -> bool:
        configured = (
            self.default_channel_id
            or self.store.get_setting(SETTING_CHANNEL_ID)
            or self.store.get_setting("slack_channel_id")
        )
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
        if image_url:
            self.store.set_setting(HUMAN_IMAGE_URL_SETTING, image_url)


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
        )
        self.runtime.on_task_done = self.controller.handle_runtime_task_done
        self.runtime.on_agent_message = self.controller.handle_runtime_agent_message
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
        )
        self.session_mirror.start()

    def close(self) -> None:
        self.session_mirror.stop()
        if self.codex_app_server:
            self.codex_app_server.close()
        self.store.close()

    def run_forever(self) -> None:
        from slack_sdk.socket_mode import SocketModeClient
        from slack_sdk.socket_mode.response import SocketModeResponse

        client = SocketModeClient(
            app_token=self.config.slack.app_token or "",
            web_client=self.gateway.client,
        )

        def listener(socket_client, request) -> None:
            socket_client.send_socket_mode_response(
                SocketModeResponse(envelope_id=request.envelope_id)
            )
            try:
                self.handle_request(request)
            except Exception:
                LOGGER.exception("failed to handle Slack Socket Mode request")

        client.socket_mode_request_listeners.append(listener)
        client.connect()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            client.close()

    def handle_request(self, request) -> None:
        if request.type == "interactive":
            payload = request.payload
            payload_type = payload.get("type")
            if payload_type == "block_actions":
                self.controller.handle_block_action(payload)
            elif payload_type == "view_submission":
                self.controller.handle_view_submission(payload)
            return
        if request.type == "events_api":
            self.controller.handle_event(request.payload)
        elif request.type == "slash_commands":
            self.controller.handle_slash_command(request.payload)


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


def _roster_text(agents) -> str:
    return f"Agent roster: {len(agents)} active lightweight handles"


def _is_usage_request(text: str) -> bool:
    normalized = re.sub(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", "", text).strip().lower()
    normalized = re.sub(r"^/(status|usage|tokens)\b", r"\1", normalized)
    return normalized in {
        "status",
        "usage",
        "tokens",
        "token usage",
        "show usage",
        "show tokens",
    }


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
