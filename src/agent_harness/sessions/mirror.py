from __future__ import annotations

import getpass
import html
import logging
import os
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from agent_harness.models import (
    AgentEvent,
    AgentSession,
    AgentTask,
    AgentTaskStatus,
    Provider,
    SessionStatus,
    SlackThreadRef,
    TeamAgent,
    utc_now,
)
from agent_harness.providers.base import AgentProvider
from agent_harness.runtime.codex_app_server import DEFAULT_CODEX_APP_SERVER_URL
from agent_harness.runtime.tasks import build_task_prompt
from agent_harness.sessions.bridge import _codex_remote_enabled, _slackgentic_channel_enabled
from agent_harness.sessions.claude_channel import (
    claude_session_has_slackgentic_mcp,
    is_slackgentic_mcp_server_configured,
)
from agent_harness.sessions.terminal import SessionTerminalNotifier
from agent_harness.slack import build_external_session_capacity_blocks
from agent_harness.slack.client import SlackGateway
from agent_harness.storage.store import Store

LOGGER = logging.getLogger(__name__)
SLACKGENTIC_CHANNEL_BLOCK_RE = re.compile(
    r"<channel\b(?=[^>]*\bsource=[\"']slackgentic[\"'])[^>]*>.*?</channel>",
    flags=re.IGNORECASE | re.DOTALL,
)
HUMAN_DISPLAY_NAME_SETTING = "slack.human_display_name"
HUMAN_IMAGE_URL_SETTING = "slack.human_image_url"
EXTERNAL_SESSION_AGENT_PREFIX = "external_session_agent."
EXTERNAL_SESSION_IGNORED_PREFIX = "external_session_ignored."
EXTERNAL_SESSION_LIVE_TARGET_PREFIX = "external_session_live_target."
EXTERNAL_SESSION_MISSING_TARGET_PREFIX = "external_session_missing_target."
EXTERNAL_SESSION_SUMMARY_PREFIX = "external_session_summary."
PENDING_EXTERNAL_SESSION_PREFIX = "external_session_pending."
CAPACITY_NOTICE_TS_PREFIX = "external_session_capacity_notice_ts."
DEFAULT_AGENT_AVATAR_BASE_URL = (
    "https://raw.githubusercontent.com/unbounded-ai/slackgentic-team/main/docs/assets/avatars"
)
DISABLED_AVATAR_BASE_VALUES = {"", "0", "false", "no", "none", "off"}
SETTING_AGENT_AVATAR_BASE_URL = "slack.agent_avatar_base_url"


@dataclass(frozen=True)
class RenderedSessionEvent:
    text: str
    author: str


class SessionMirror:
    def __init__(
        self,
        store: Store,
        gateway: SlackGateway,
        providers: Iterable[AgentProvider],
        team_id: str,
        channel_id: str | None = None,
        poll_seconds: float = 2.0,
        terminal_notifier: SessionTerminalNotifier | None = None,
        codex_app_server_url: str | None = DEFAULT_CODEX_APP_SERVER_URL,
        on_external_session_occupancy_change: Callable[[str], None] | None = None,
        home: Path | None = None,
    ):
        self.store = store
        self.gateway = gateway
        self.providers = list(providers)
        self.team_id = team_id
        self.channel_id = channel_id
        self.poll_seconds = poll_seconds
        self.terminal_notifier = terminal_notifier or SessionTerminalNotifier()
        self.codex_app_server_url = codex_app_server_url
        self.on_external_session_occupancy_change = on_external_session_occupancy_change
        self.home = home or Path.home()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sync_once(backfill_new_sessions=False)
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="slackgentic-session-mirror",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def has_active_sessions(self) -> bool:
        for provider in self.providers:
            for session in provider.discover():
                if session.status == SessionStatus.ACTIVE:
                    return True
        return False

    def sync_once(self, backfill_new_sessions: bool = True) -> None:
        channel_id = self._channel_id()
        if not channel_id:
            return
        sync_sessions: list[tuple[AgentProvider, AgentSession]] = []
        active_session_keys: set[str] = set()
        for provider in self.providers:
            for session in provider.discover():
                if self._external_terminal_session_closed(session):
                    session = replace(session, status=SessionStatus.DONE)
                self.store.upsert_session(session)
                self._cleanup_hidden_external_session_summary(session)
                if self._skip_managed_task_session(session, channel_id):
                    continue
                if not self._should_sync_session(session):
                    continue
                active_session_keys.add(_external_session_key(session.provider, session.session_id))
                sync_sessions.append((provider, session))
        self._cleanup_inactive_external_sessions(active_session_keys, channel_id)
        sync_sessions.sort(
            key=lambda item: self._thread_for_session(item[1], channel_id) is not None
        )
        for provider, session in sync_sessions:
            agent = self._team_agent_for_session(session, active_session_keys, channel_id)
            if agent is None:
                continue
            thread = self._thread_for_session(session, channel_id)
            if thread is None:
                if not backfill_new_sessions:
                    self._mark_session_seen(provider, session)
                    continue
                chunks, max_line = self._new_chunks(provider, session)
                if not chunks:
                    self._update_cursor_if_needed(session, max_line)
                    continue
                thread = self._post_session_parent(session, channel_id, agent)
                self._post_chunks(session, thread, chunks, agent)
                self._update_parent_summary(session, thread, chunks)
                self._update_cursor_if_needed(session, max_line)
                continue
            self._post_session_channel_notice_once(session, thread)
            self._mirror_new_events(provider, session, thread, agent)
        self._sync_capacity_notices(channel_id)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.sync_once(backfill_new_sessions=True)
            except Exception:
                LOGGER.exception("failed to mirror external agent sessions")

    def _thread_for_session(
        self,
        session: AgentSession,
        channel_id: str,
    ) -> SlackThreadRef | None:
        return self.store.get_slack_thread_for_session(
            session.provider,
            session.session_id,
            self.team_id,
            channel_id,
        )

    def _post_session_parent(
        self,
        session: AgentSession,
        channel_id: str,
        agent: TeamAgent,
    ) -> SlackThreadRef:
        channel_notice = self._session_channel_notice(session)
        text = self._session_parent_text(session, channel_notice)
        posted = self.gateway.post_session_parent(
            channel_id,
            text,
            agent,
            icon_url=self._team_agent_icon_url(agent),
        )
        thread = SlackThreadRef(
            channel_id=channel_id,
            thread_ts=posted.ts,
            message_ts=posted.ts,
        )
        self.store.upsert_slack_thread_for_session(
            session.provider,
            session.session_id,
            self.team_id,
            thread,
        )
        if channel_notice:
            self._mark_session_channel_notice_posted(session)
        return thread

    def _session_parent_text(self, session: AgentSession, channel_notice: str | None = None) -> str:
        text = format_session_parent(session, self._session_summary(session))
        if channel_notice:
            text = f"{text}\n\n{channel_notice}"
        return text

    def _session_channel_notice(self, session: AgentSession) -> str | None:
        targets = self.terminal_notifier.targets_for_session(session)
        if not targets:
            return None
        if session.provider == Provider.CODEX:
            if any(
                _codex_remote_enabled(target.command, self.codex_app_server_url)
                for target in targets
            ):
                return None
            remote_url = self.codex_app_server_url or DEFAULT_CODEX_APP_SERVER_URL
            return (
                "I can mirror visible output from this Codex session, and Slack replies will still "
                "run through `codex exec resume`, but this terminal was not started against "
                "Slackgentic's Codex app-server. Restart it with "
                f"`codex --remote {remote_url}` if you want Slack replies to appear in the "
                "original Codex terminal."
            )
        if session.provider == Provider.CLAUDE:
            if not any(_slackgentic_channel_enabled(target.command) for target in targets):
                return (
                    "I can mirror visible output from this Claude session, but it was not started "
                    "with Slackgentic's Claude channel enabled, so Slack replies cannot be "
                    "delivered into the live terminal session. Restart it with "
                    "`claude --dangerously-load-development-channels server:slackgentic` if you "
                    "want to chat with it from Slack."
                )
            if claude_session_has_slackgentic_mcp(session):
                return None
            if not is_slackgentic_mcp_server_configured(self.home):
                return (
                    "This Claude terminal has Slackgentic's channel flag, but Claude does not have "
                    "the Slackgentic MCP server registered. Run "
                    "`slackgentic claude-channel --install`, then restart Claude with "
                    "`claude --dangerously-load-development-channels server:slackgentic`."
                )
            return (
                "This Claude terminal has Slackgentic's channel flag, but this session did not "
                "load Slackgentic's MCP server. Restart Claude after "
                "`slackgentic claude-channel --install` so Slack replies and approvals can reach "
                "the live terminal."
            )
        return None

    def _post_session_channel_notice_once(
        self,
        session: AgentSession,
        thread: SlackThreadRef,
    ) -> None:
        if self.store.get_setting(_session_channel_notice_key(session)):
            return
        notice = self._session_channel_notice(session)
        if not notice:
            return
        self.gateway.post_thread_reply(thread, notice)
        self._mark_session_channel_notice_posted(session)

    def _mark_session_channel_notice_posted(self, session: AgentSession) -> None:
        self.store.set_setting(_session_channel_notice_key(session), utc_now().isoformat())

    def _mirror_new_events(
        self,
        provider: AgentProvider,
        session: AgentSession,
        thread: SlackThreadRef,
        agent: TeamAgent,
    ) -> None:
        chunks, max_line = self._new_chunks(provider, session)
        self._post_chunks(session, thread, chunks, agent)
        self._update_parent_summary(session, thread, chunks)
        self._update_cursor_if_needed(session, max_line)

    def _new_chunks(
        self,
        provider: AgentProvider,
        session: AgentSession,
    ) -> tuple[list[RenderedSessionEvent], int]:
        cursor = self.store.get_session_mirror_cursor(
            session.provider,
            session.session_id,
        )
        max_line = cursor
        chunks: list[RenderedSessionEvent] = []
        for event in provider.iter_events(session.transcript_path):
            line_number = event.line_number or 0
            if line_number <= cursor:
                continue
            max_line = max(max_line, line_number)
            rendered = render_session_event_chunk(event)
            if rendered is None:
                continue
            if rendered.author == "user" and self.store.consume_session_bridge_prompt(
                session.provider,
                session.session_id,
                rendered.text,
            ):
                continue
            chunks.extend(
                RenderedSessionEvent(text=chunk, author=rendered.author)
                for chunk in _slack_chunks(rendered.text)
            )
        return chunks, max_line

    def _post_chunks(
        self,
        session: AgentSession,
        thread: SlackThreadRef,
        chunks: list[RenderedSessionEvent],
        agent: TeamAgent,
    ) -> None:
        icon_url = self._team_agent_icon_url(agent)
        for chunk in chunks:
            if chunk.author == "user":
                posted = self.gateway.post_thread_reply(
                    thread,
                    chunk.text,
                    username=self._human_display_name(),
                    icon_url=self._human_image_url(),
                )
                self.store.mark_slack_message_mirrored(
                    thread.channel_id,
                    posted.ts,
                    f"{session.provider.value}:{session.session_id}:user",
                )
            else:
                self.gateway.post_thread_reply(thread, chunk.text, persona=agent, icon_url=icon_url)

    def _update_parent_summary(
        self,
        session: AgentSession,
        thread: SlackThreadRef,
        chunks: list[RenderedSessionEvent],
    ) -> None:
        summary = summarize_session_chunks(chunks, self._session_summary(session))
        if not summary:
            return
        if summary == self._session_summary(session):
            return
        self.store.set_setting(_external_session_summary_key(session), summary)
        if not thread.message_ts:
            return
        try:
            self.gateway.update_message(
                thread.channel_id,
                thread.message_ts,
                self._session_parent_text(session, self._session_channel_notice(session)),
            )
        except Exception:
            LOGGER.debug("failed to update external session parent summary", exc_info=True)

    def _team_agent_for_session(
        self,
        session: AgentSession,
        active_session_keys: set[str],
        channel_id: str,
    ) -> TeamAgent | None:
        setting_key = _external_session_agent_setting_key(session)
        assigned_agent_id = self.store.get_setting(setting_key)
        if assigned_agent_id:
            agent = self.store.get_team_agent(assigned_agent_id)
            if agent is not None:
                self._mark_session_not_pending(session)
                return agent
        active_agents = self.store.list_team_agents()
        if not active_agents:
            self._mark_session_pending(session)
            self._post_or_update_capacity_notice(channel_id, session.provider)
            return None
        assigned_agent_ids = self._active_external_agent_ids(active_session_keys, setting_key)
        idle_agents = self.store.idle_team_agents()
        available = [
            agent
            for agent in idle_agents
            if agent.agent_id not in assigned_agent_ids
            and agent.provider_preference == session.provider
        ]
        if not available:
            self._mark_session_pending(session)
            self._post_or_update_capacity_notice(channel_id, session.provider)
            return None
        agent = available[0]
        self.store.set_setting(setting_key, agent.agent_id)
        self._notify_external_session_occupancy_changed(channel_id)
        self._mark_session_not_pending(session)
        self._update_capacity_notice_if_clear(channel_id, session.provider)
        return agent

    def _external_terminal_session_closed(self, session: AgentSession) -> bool:
        if session.provider not in {Provider.CLAUDE, Provider.CODEX}:
            return False
        if session.status != SessionStatus.ACTIVE:
            return False
        live_target_key = _external_session_live_target_key(session)
        assigned_key = _external_session_agent_setting_key(session)
        was_tracked = bool(
            self.store.get_setting(live_target_key) or self.store.get_setting(assigned_key)
        )
        target = self._live_terminal_target(session)
        if target is not None:
            self.store.set_setting(live_target_key, str(target.pid))
            self.store.delete_setting(_external_session_missing_target_key(session))
            return False
        if not was_tracked:
            return False
        missing_target_key = _external_session_missing_target_key(session)
        if self.store.get_setting(missing_target_key):
            return True
        self.store.set_setting(missing_target_key, utc_now().isoformat())
        return False

    def _live_terminal_target(self, session: AgentSession):
        targets = self.terminal_notifier.targets_for_session(session)
        if len(targets) != 1:
            return None
        return targets[0]

    def _active_external_agent_ids(
        self,
        active_session_keys: set[str],
        current_setting_key: str,
    ) -> set[str]:
        assigned: set[str] = set()
        for key, agent_id in self.store.list_settings(EXTERNAL_SESSION_AGENT_PREFIX).items():
            if key == current_setting_key:
                continue
            session_key = key.removeprefix(EXTERNAL_SESSION_AGENT_PREFIX)
            if session_key in active_session_keys:
                assigned.add(agent_id)
        return assigned

    def _cleanup_inactive_external_sessions(
        self,
        active_session_keys: set[str],
        channel_id: str,
    ) -> None:
        providers_to_refresh: set[Provider] = set()
        for prefix in (
            EXTERNAL_SESSION_AGENT_PREFIX,
            PENDING_EXTERNAL_SESSION_PREFIX,
            EXTERNAL_SESSION_LIVE_TARGET_PREFIX,
            EXTERNAL_SESSION_MISSING_TARGET_PREFIX,
        ):
            for key in list(self.store.list_settings(prefix)):
                session_key = key.removeprefix(prefix)
                if session_key in active_session_keys:
                    continue
                parsed = _provider_session_from_external_key(session_key)
                self.store.delete_setting(key)
                if parsed is None:
                    continue
                provider, session_id = parsed
                providers_to_refresh.add(provider)
                if prefix != EXTERNAL_SESSION_AGENT_PREFIX:
                    continue
                self._notify_external_session_occupancy_changed(channel_id)
                thread = self.store.get_slack_thread_for_session(
                    provider,
                    session_id,
                    self.team_id,
                    channel_id,
                )
                if thread is not None:
                    self.gateway.post_thread_reply(
                        thread,
                        "Session ended; freed up this agent.",
                    )
        for provider in providers_to_refresh:
            self._update_capacity_notice_if_clear(channel_id, provider)

    def _should_sync_session(self, session: AgentSession) -> bool:
        if self.store.get_setting(_ignored_external_session_key(session)):
            return False
        return session.status == SessionStatus.ACTIVE

    def _skip_managed_task_session(self, session: AgentSession, channel_id: str) -> bool:
        managed_task = self._managed_task_for_session(session)
        if managed_task is not None:
            if managed_task.session_id != session.session_id:
                self.store.update_agent_task_session(
                    managed_task.task_id,
                    session.provider,
                    session.session_id,
                )
            changed = self.store.clear_external_session_tracking(
                session.provider,
                session.session_id,
                team_id=self.team_id,
                channel_id=channel_id,
            )
            if changed:
                self._notify_external_session_occupancy_changed(channel_id)
                self._update_capacity_notice_if_clear(channel_id, session.provider)
            return True
        if not self.store.has_agent_task_session(session.provider, session.session_id):
            return False
        changed = self.store.clear_external_session_tracking(
            session.provider,
            session.session_id,
            team_id=self.team_id,
            channel_id=channel_id,
        )
        if changed:
            self._notify_external_session_occupancy_changed(channel_id)
            self._update_capacity_notice_if_clear(channel_id, session.provider)
        return True

    def _managed_task_for_session(self, session: AgentSession) -> AgentTask | None:
        prompt = _managed_prompt_from_session_transcript(session)
        if not prompt:
            return None
        for task in self.store.list_agent_tasks():
            if task.status not in {AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE}:
                continue
            agent = self.store.get_team_agent(task.agent_id)
            if agent is None:
                continue
            provider = agent.provider_preference or Provider.CODEX
            if provider != session.provider:
                continue
            if task.session_id and task.session_id != session.session_id:
                continue
            if build_task_prompt(agent, task) == prompt:
                return task
        return None

    def _notify_external_session_occupancy_changed(self, channel_id: str) -> None:
        if self.on_external_session_occupancy_change is None:
            return
        try:
            self.on_external_session_occupancy_change(channel_id)
        except Exception:
            LOGGER.debug("failed to refresh external session occupancy", exc_info=True)

    def _mark_session_pending(self, session: AgentSession) -> None:
        self.store.set_setting(_pending_external_session_key(session), utc_now().isoformat())

    def _mark_session_not_pending(self, session: AgentSession) -> None:
        self.store.delete_setting(_pending_external_session_key(session))

    def _post_or_update_capacity_notice(self, channel_id: str, provider: Provider) -> None:
        count = self._pending_count(provider)
        text = _capacity_message(provider, count)
        blocks = build_external_session_capacity_blocks(provider, count)
        setting_key = _capacity_notice_ts_key(provider)
        existing_ts = self.store.get_setting(setting_key)
        if existing_ts:
            try:
                self.gateway.update_message(channel_id, existing_ts, text, blocks=blocks)
                return
            except Exception:
                LOGGER.debug("failed to update external capacity notice", exc_info=True)
        posted = self.gateway.post_message(channel_id, text, blocks=blocks)
        self.store.set_setting(setting_key, posted.ts)

    def _update_capacity_notice_if_clear(self, channel_id: str, provider: Provider) -> None:
        if self._pending_count(provider) > 0:
            self._post_or_update_capacity_notice(channel_id, provider)
            return
        setting_key = _capacity_notice_ts_key(provider)
        existing_ts = self.store.get_setting(setting_key)
        if not existing_ts:
            return
        label = provider.value.title()
        try:
            self.gateway.update_message(
                channel_id,
                existing_ts,
                f"{label} capacity for sessions started outside Slack is available now.",
            )
        except Exception:
            LOGGER.debug("failed to clear external capacity notice", exc_info=True)
        self.store.delete_setting(setting_key)

    def _sync_capacity_notices(self, channel_id: str) -> None:
        for provider in Provider:
            if self.store.get_setting(_capacity_notice_ts_key(provider)) is None:
                continue
            if self._pending_count(provider) == 0:
                self._update_capacity_notice_if_clear(channel_id, provider)

    def _pending_count(self, provider: Provider) -> int:
        prefix = f"{PENDING_EXTERNAL_SESSION_PREFIX}{provider.value}."
        return len(self.store.list_settings(prefix))

    def _team_agent_icon_url(self, agent: TeamAgent) -> str | None:
        base_url = (
            self.store.get_setting(SETTING_AGENT_AVATAR_BASE_URL)
            or os.environ.get("SLACKGENTIC_AGENT_AVATAR_BASE_URL")
            or DEFAULT_AGENT_AVATAR_BASE_URL
        ).strip()
        if base_url.lower() in DISABLED_AVATAR_BASE_VALUES:
            return None
        return f"{base_url.rstrip('/')}/{agent.avatar_slug}.png"

    def _human_display_name(self) -> str:
        configured = self.store.get_setting(HUMAN_DISPLAY_NAME_SETTING)
        if configured and configured.strip():
            return configured.strip()
        for value in (
            os.environ.get("SLACKGENTIC_HUMAN_NAME"),
            os.environ.get("USER"),
            getpass.getuser(),
        ):
            if value and value.strip():
                return value.strip()
        return "Local user"

    def _human_image_url(self) -> str | None:
        configured = self.store.get_setting(HUMAN_IMAGE_URL_SETTING)
        if configured and configured.strip():
            return configured.strip()
        return None

    def _update_cursor_if_needed(self, session: AgentSession, max_line: int) -> None:
        cursor = self.store.get_session_mirror_cursor(session.provider, session.session_id)
        if max_line != cursor:
            self.store.set_session_mirror_cursor(
                session.provider,
                session.session_id,
                max_line,
            )

    def _mark_session_seen(self, provider: AgentProvider, session: AgentSession) -> None:
        last_line = 0
        for event in provider.iter_events(session.transcript_path):
            last_line = max(last_line, event.line_number or 0)
        self.store.set_session_mirror_cursor(session.provider, session.session_id, last_line)

    def _channel_id(self) -> str | None:
        return self.channel_id or self.store.get_setting("slack.channel_id")

    def _session_summary(self, session: AgentSession) -> str | None:
        return self.store.get_setting(_external_session_summary_key(session))

    def _cleanup_hidden_external_session_summary(self, session: AgentSession) -> None:
        summary = self._session_summary(session)
        if summary and _has_claude_local_command_block(summary):
            self.store.delete_setting(_external_session_summary_key(session))


def format_session_parent(session: AgentSession, summary: str | None = None) -> str:
    label = session.provider.value.capitalize()
    parts = [f"Started observing {label} session from outside Slack."]
    if summary:
        parts.append(f"Task: {summary}")
    if session.cwd:
        parts.append(f"cwd: `{_short_path(session.cwd)}`")
    if session.git_branch:
        parts.append(f"branch: `{session.git_branch}`")
    if session.model:
        parts.append(f"model: `{session.model}`")
    return "\n".join(parts)


def summarize_session_chunks(
    chunks: list[RenderedSessionEvent],
    existing_summary: str | None = None,
) -> str | None:
    user_texts = [chunk.text for chunk in chunks if chunk.author == "user" and chunk.text.strip()]
    if user_texts:
        return _summary_line(user_texts[0])
    if existing_summary:
        return existing_summary
    assistant_texts = [
        chunk.text for chunk in chunks if chunk.author == "assistant" and chunk.text.strip()
    ]
    if assistant_texts:
        return _summary_line(assistant_texts[0])
    return None


def _external_session_agent_setting_key(session: AgentSession) -> str:
    return f"{EXTERNAL_SESSION_AGENT_PREFIX}{_external_session_key(session.provider, session.session_id)}"


def _external_session_summary_key(session: AgentSession) -> str:
    return f"{EXTERNAL_SESSION_SUMMARY_PREFIX}{_external_session_key(session.provider, session.session_id)}"


def _external_session_live_target_key(session: AgentSession) -> str:
    return f"{EXTERNAL_SESSION_LIVE_TARGET_PREFIX}{_external_session_key(session.provider, session.session_id)}"


def _external_session_missing_target_key(session: AgentSession) -> str:
    return f"{EXTERNAL_SESSION_MISSING_TARGET_PREFIX}{_external_session_key(session.provider, session.session_id)}"


def _ignored_external_session_key(session: AgentSession) -> str:
    return f"{EXTERNAL_SESSION_IGNORED_PREFIX}{_external_session_key(session.provider, session.session_id)}"


def _external_session_key(provider: Provider, session_id: str) -> str:
    return f"{provider.value}.{session_id}"


def _provider_session_from_external_key(value: str) -> tuple[Provider, str] | None:
    provider_text, separator, session_id = value.partition(".")
    if not separator or not session_id:
        return None
    try:
        provider = Provider(provider_text)
    except ValueError:
        return None
    return provider, session_id


def _pending_external_session_key(session: AgentSession) -> str:
    return (
        f"{PENDING_EXTERNAL_SESSION_PREFIX}"
        f"{_external_session_key(session.provider, session.session_id)}"
    )


def _capacity_notice_ts_key(provider: Provider) -> str:
    return f"{CAPACITY_NOTICE_TS_PREFIX}{provider.value}"


def _capacity_message(provider: Provider, waiting_count: int) -> str:
    label = provider.value.title()
    plural = "session is" if waiting_count == 1 else "sessions are"
    return (
        f"No {label} team seat is available for sessions started outside Slack. "
        f"{waiting_count} {label} {plural} waiting. "
        "Hire one matching agent and Slackgentic will backfill the transcript."
    )


def _managed_prompt_from_session_transcript(
    session: AgentSession,
    *,
    max_records: int = 12,
) -> str | None:
    from agent_harness.storage.jsonl import iter_jsonl

    try:
        for index, (_, record) in enumerate(iter_jsonl(session.transcript_path), start=1):
            prompt = _managed_prompt_from_record(session.provider, record)
            if prompt:
                return prompt
            if index >= max_records:
                break
    except OSError:
        return None
    return None


def _managed_prompt_from_record(provider: Provider, record: dict[str, object]) -> str | None:
    if provider == Provider.CLAUDE:
        if record.get("type") == "queue-operation" and record.get("operation") == "enqueue":
            content = record.get("content")
            return content if isinstance(content, str) else None
        if record.get("type") == "user" and record.get("entrypoint") == "sdk-cli":
            message = record.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                return content if isinstance(content, str) else None
    if provider == Provider.CODEX:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return None
        if payload.get("type") == "user_message":
            message = payload.get("message")
            return message if isinstance(message, str) else None
    return None


def _session_channel_notice_key(session: AgentSession) -> str:
    return f"session_channel_notice.{session.provider.value}.{session.session_id}"


def render_session_event(event: AgentEvent) -> str | None:
    rendered = render_session_event_chunk(event)
    return rendered.text if rendered else None


def render_session_event_chunk(event: AgentEvent) -> RenderedSessionEvent | None:
    if event.provider == Provider.CODEX:
        return _render_codex_event(event)
    if event.provider == Provider.CLAUDE:
        return _render_claude_event(event)
    return None


def _render_codex_event(event: AgentEvent) -> RenderedSessionEvent | None:
    payload = event.metadata.get("payload")
    if not isinstance(payload, dict):
        return None
    event_type = payload.get("type")
    if event_type == "agent_message":
        message = payload.get("message") or payload.get("text")
        text = _clean_text(str(message)) if message else ""
        return RenderedSessionEvent(text, "assistant") if text else None
    if event_type == "user_message":
        message = payload.get("message") or payload.get("text")
        text = _clean_text(str(message)) if message else ""
        return RenderedSessionEvent(text, "user") if text else None
    return None


def _render_claude_event(event: AgentEvent) -> RenderedSessionEvent | None:
    if event.event_type not in {"assistant", "user"}:
        return None
    message = event.metadata.get("message")
    if not isinstance(message, dict):
        return None
    text = _claude_message_text(message)
    if not text:
        return None
    if _has_claude_local_command_block(text):
        return None
    if event.event_type == "user" and _has_slackgentic_channel_block(text):
        return None
    if event.event_type == "assistant":
        text = _remove_slackgentic_channel_blocks(text)
        if not text:
            return None
    author = "user" if event.event_type == "user" else "assistant"
    return RenderedSessionEvent(text, author)


def _claude_message_text(message: dict) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return _clean_text(content) or None
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            cleaned = _clean_text(item["text"])
            if cleaned:
                text_parts.append(cleaned)
    return "\n\n".join(text_parts) if text_parts else None


def _slack_chunks(text: str, limit: int = 2800) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    chunks: list[str] = []
    while cleaned:
        chunks.append(cleaned[:limit])
        cleaned = cleaned[limit:]
    return chunks


def _summary_line(text: str, limit: int = 180) -> str:
    cleaned = _clean_text(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^(/status|/usage|status|usage)\b[:\s-]*", "", cleaned, flags=re.I)
    if not cleaned:
        return ""
    sentence_match = re.match(r"(.+?[.!?])(?:\s|$)", cleaned)
    if sentence_match and len(sentence_match.group(1)) >= 24:
        cleaned = sentence_match.group(1)
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 1)].rstrip()}..."


def _clean_text(text: str) -> str:
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


def _has_slackgentic_channel_block(text: str) -> bool:
    return bool(SLACKGENTIC_CHANNEL_BLOCK_RE.search(html.unescape(text)))


def _remove_slackgentic_channel_blocks(text: str) -> str:
    decoded = html.unescape(text)
    if not SLACKGENTIC_CHANNEL_BLOCK_RE.search(decoded):
        return text
    return SLACKGENTIC_CHANNEL_BLOCK_RE.sub("", decoded).strip()


def _has_claude_local_command_block(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "<local-command-caveat>",
            "<command-name>",
            "<command-message>",
            "<command-args>",
            "<local-command-stdout>",
            "<local-command-stderr>",
        )
    )


def _short_path(path: Path) -> str:
    home = Path.home()
    try:
        return f"~/{path.expanduser().resolve().relative_to(home).as_posix()}"
    except ValueError:
        return str(path)
