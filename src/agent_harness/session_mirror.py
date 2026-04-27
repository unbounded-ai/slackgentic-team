from __future__ import annotations

import getpass
import logging
import os
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from agent_harness.codex_app_server import DEFAULT_CODEX_APP_SERVER_URL
from agent_harness.models import (
    AgentEvent,
    AgentSession,
    Provider,
    SessionStatus,
    SlackThreadRef,
    utc_now,
)
from agent_harness.personas import generate_persona
from agent_harness.providers.base import AgentProvider
from agent_harness.session_bridge import _codex_remote_enabled, _slackgentic_channel_enabled
from agent_harness.session_terminal import SessionTerminalNotifier
from agent_harness.slack_client import SlackGateway
from agent_harness.store import Store

LOGGER = logging.getLogger(__name__)
HUMAN_DISPLAY_NAME_SETTING = "slack.human_display_name"
HUMAN_IMAGE_URL_SETTING = "slack.human_image_url"


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
    ):
        self.store = store
        self.gateway = gateway
        self.providers = list(providers)
        self.team_id = team_id
        self.channel_id = channel_id
        self.poll_seconds = poll_seconds
        self.terminal_notifier = terminal_notifier or SessionTerminalNotifier()
        self.codex_app_server_url = codex_app_server_url
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

    def sync_once(self, backfill_new_sessions: bool = True) -> None:
        channel_id = self._channel_id()
        if not channel_id:
            return
        for provider in self.providers:
            for session in provider.discover():
                if session.status != SessionStatus.ACTIVE:
                    continue
                self.store.upsert_session(session)
                thread = self._thread_for_session(session, channel_id)
                if thread is None:
                    if not backfill_new_sessions:
                        self._mark_session_seen(provider, session)
                        continue
                    chunks, max_line = self._new_chunks(provider, session)
                    if not chunks:
                        self._update_cursor_if_needed(session, max_line)
                        continue
                    thread = self._post_session_parent(session, channel_id)
                    self._post_chunks(session, thread, chunks)
                    self._update_cursor_if_needed(session, max_line)
                    continue
                self._post_session_channel_notice_once(session, thread)
                self._mirror_new_events(provider, session, thread)

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

    def _post_session_parent(self, session: AgentSession, channel_id: str) -> SlackThreadRef:
        persona = generate_persona(session.provider, session.session_id)
        self.store.upsert_persona(persona)
        text = format_session_parent(session)
        channel_notice = self._session_channel_notice(session)
        if channel_notice:
            text = f"{text}\n\n{channel_notice}"
        posted = self.gateway.post_session_parent(
            channel_id,
            text,
            persona,
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
            if any(_slackgentic_channel_enabled(target.command) for target in targets):
                return None
            return (
                "I can mirror visible output from this Claude session, but it was not started with "
                "Slackgentic's Claude channel enabled, so Slack replies cannot be delivered into "
                "the live terminal session. Restart it with "
                "`claude --dangerously-load-development-channels server:slackgentic` if you want "
                "to chat with it from Slack."
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
    ) -> None:
        chunks, max_line = self._new_chunks(provider, session)
        self._post_chunks(session, thread, chunks)
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
    ) -> None:
        persona = generate_persona(session.provider, session.session_id)
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
                self.gateway.post_thread_reply(thread, chunk.text, persona=persona)

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


def format_session_parent(session: AgentSession) -> str:
    label = session.provider.value.capitalize()
    parts = [f"Started observing external {label} session."]
    if session.cwd:
        parts.append(f"cwd: `{_short_path(session.cwd)}`")
    if session.git_branch:
        parts.append(f"branch: `{session.git_branch}`")
    if session.model:
        parts.append(f"model: `{session.model}`")
    return "\n".join(parts)


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


def _short_path(path: Path) -> str:
    home = Path.home()
    try:
        return f"~/{path.expanduser().resolve().relative_to(home).as_posix()}"
    except ValueError:
        return str(path)
