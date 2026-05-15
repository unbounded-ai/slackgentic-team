from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent_harness.models import SlackThreadRef, TeamAgent
from agent_harness.slack import normalize_slack_mrkdwn, slack_blocks_for_markdown_table
from agent_harness.team import TeamChatMessage

SLACK_API_RETRY_LIMIT = 3
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostedMessage:
    channel_id: str
    ts: str
    thread_ts: str | None = None


@dataclass(frozen=True)
class SlackUserProfile:
    display_name: str | None = None
    image_url: str | None = None


class SlackGateway:
    def __init__(self, bot_token: str):
        from slack_sdk import WebClient

        self.client = WebClient(token=bot_token)

    def create_channel(self, name: str, is_private: bool) -> str:
        from slack_sdk.errors import SlackApiError

        for attempt in range(5):
            candidate = name if attempt == 0 else f"{name[:74]}-{attempt + 1}"
            try:
                response = self.client.conversations_create(
                    name=candidate,
                    is_private=is_private,
                )
                return response["channel"]["id"]
            except SlackApiError as exc:
                if exc.response.get("error") != "name_taken":
                    raise
        response = self.client.conversations_create(
            name=f"{name[:70]}-{int(time.time())}",
            is_private=is_private,
        )
        return response["channel"]["id"]

    def invite_users(self, channel_id: str, user_ids: list[str]) -> None:
        if user_ids:
            self.client.conversations_invite(channel=channel_id, users=",".join(user_ids))

    def archive_channel(self, channel_id: str) -> bool:
        from slack_sdk.errors import SlackApiError

        try:
            self.client.conversations_archive(channel=channel_id)
        except SlackApiError as exc:
            if exc.response.get("error") == "already_archived":
                return False
            raise
        return True

    def open_view(self, trigger_id: str, view: dict[str, Any]) -> None:
        self.client.views_open(trigger_id=trigger_id, view=view)

    def auth_test(self) -> dict[str, Any]:
        response = self.client.auth_test()
        data = getattr(response, "data", response)
        return dict(data)

    def user_profile(self, user_id: str) -> SlackUserProfile:
        response = self.client.users_info(user=user_id)
        data = getattr(response, "data", response)
        user = dict(data).get("user") or {}
        profile = user.get("profile") or {}
        display_name = None
        for value in (
            profile.get("display_name"),
            profile.get("real_name"),
            user.get("real_name"),
            user.get("name"),
        ):
            if isinstance(value, str) and value.strip():
                display_name = value.strip()
                break
        image_url = None
        for key in ("image_512", "image_192", "image_72", "image_48"):
            value = profile.get(key)
            if isinstance(value, str) and value.strip():
                image_url = value.strip()
                break
        return SlackUserProfile(display_name=display_name, image_url=image_url)

    def user_display_name(self, user_id: str) -> str | None:
        return self.user_profile(user_id).display_name

    def permalink(self, channel_id: str, message_ts: str) -> str | None:
        response = self.client.chat_getPermalink(channel=channel_id, message_ts=message_ts)
        return response.get("permalink")

    def pin_message(self, channel_id: str, message_ts: str) -> None:
        self.client.pins_add(channel=channel_id, timestamp=message_ts)

    def post_message(
        self,
        channel_id: str,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
        thread_ts: str | None = None,
    ) -> PostedMessage:
        kwargs: dict[str, Any] = {
            "channel": channel_id,
            "text": normalize_slack_mrkdwn(text),
        }
        rendered_blocks = blocks if blocks is not None else slack_blocks_for_markdown_table(text)
        auto_rendered_blocks = blocks is None and rendered_blocks is not None
        if rendered_blocks:
            kwargs["blocks"] = rendered_blocks
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        response = self._chat_post_message(kwargs, auto_rendered_blocks=auto_rendered_blocks)
        return PostedMessage(channel_id=channel_id, ts=response["ts"], thread_ts=thread_ts)

    def post_session_parent(
        self,
        channel_id: str,
        text: str,
        persona: TeamAgent,
        icon_url: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> PostedMessage:
        rendered_blocks = blocks if blocks is not None else slack_blocks_for_markdown_table(text)
        kwargs: dict[str, Any] = {
            "channel": channel_id,
            "text": normalize_slack_mrkdwn(text),
            "username": _identity_name(persona),
            "icon_url": icon_url,
            "icon_emoji": None if icon_url else persona.icon_emoji,
        }
        if rendered_blocks is not None:
            kwargs["blocks"] = rendered_blocks
        response = self._chat_post_message(
            kwargs,
            auto_rendered_blocks=blocks is None and rendered_blocks is not None,
        )
        return PostedMessage(channel_id=channel_id, ts=response["ts"])

    def post_task_parent(
        self,
        channel_id: str,
        text: str,
        agent: TeamAgent,
        blocks: list[dict[str, Any]] | None = None,
        icon_url: str | None = None,
    ) -> SlackThreadRef:
        posted = self.post_session_parent(
            channel_id=channel_id,
            text=text,
            persona=agent,
            icon_url=icon_url,
            blocks=blocks,
        )
        return SlackThreadRef(
            channel_id=channel_id,
            thread_ts=posted.ts,
            message_ts=posted.ts,
        )

    def post_team_initialization(
        self,
        channel_id: str,
        agents: list[TeamAgent],
        messages: list[TeamChatMessage],
        icon_url_for: Callable[[TeamAgent], str | None] | None = None,
    ) -> SlackThreadRef:
        if not agents or not messages:
            raise ValueError("team initialization needs at least one agent and one message")
        agent_by_id = {agent.agent_id: agent for agent in agents}
        first = messages[0]
        first_agent = agent_by_id[first.sender_agent_id]
        parent = self.post_session_parent(
            channel_id=channel_id,
            text=first.text,
            persona=first_agent,
            icon_url=icon_url_for(first_agent) if icon_url_for else None,
        )
        thread = SlackThreadRef(
            channel_id=channel_id,
            thread_ts=parent.ts,
            message_ts=parent.ts,
        )
        for message in messages[1:]:
            sender = agent_by_id[message.sender_agent_id]
            self.post_thread_reply(
                thread,
                message.text,
                persona=sender,
                icon_url=icon_url_for(sender) if icon_url_for else None,
            )
        return thread

    def post_thread_reply(
        self,
        thread: SlackThreadRef,
        text: str,
        persona: TeamAgent | None = None,
        username: str | None = None,
        icon_url: str | None = None,
        icon_emoji: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> PostedMessage:
        kwargs = {
            "channel": thread.channel_id,
            "thread_ts": thread.thread_ts,
            "text": normalize_slack_mrkdwn(text),
        }
        rendered_blocks = blocks if blocks is not None else slack_blocks_for_markdown_table(text)
        auto_rendered_blocks = blocks is None and rendered_blocks is not None
        if rendered_blocks:
            kwargs["blocks"] = rendered_blocks
        if persona:
            kwargs["username"] = _identity_name(persona)
            if icon_url:
                kwargs["icon_url"] = icon_url
            else:
                kwargs["icon_emoji"] = persona.icon_emoji
        elif username:
            kwargs["username"] = username
            if icon_url:
                kwargs["icon_url"] = icon_url
            else:
                kwargs["icon_emoji"] = icon_emoji or ":bust_in_silhouette:"
        response = self._chat_post_message(kwargs, auto_rendered_blocks=auto_rendered_blocks)
        return PostedMessage(
            channel_id=thread.channel_id,
            ts=response["ts"],
            thread_ts=thread.thread_ts,
        )

    def thread_messages(
        self,
        channel_id: str,
        thread_ts: str,
        limit: int = 20,
        oldest: str | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        cursor: str | None = None
        remaining = max(1, limit)
        while remaining > 0:
            kwargs: dict[str, Any] = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": min(remaining, 200),
            }
            if oldest:
                kwargs["oldest"] = oldest
                kwargs["inclusive"] = False
            if cursor:
                kwargs["cursor"] = cursor
            response = self.client.conversations_replies(**kwargs)
            page = list(response.get("messages") or [])
            messages.extend(page)
            remaining -= len(page)
            metadata = response.get("response_metadata") or {}
            cursor = metadata.get("next_cursor")
            if not cursor or not page:
                break
        return messages

    def channel_messages(
        self,
        channel_id: str,
        oldest: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        cursor: str | None = None
        remaining = max(1, limit)
        while remaining > 0:
            kwargs: dict[str, Any] = {
                "channel": channel_id,
                "limit": min(remaining, 200),
            }
            if oldest:
                kwargs["oldest"] = oldest
                kwargs["inclusive"] = False
            if cursor:
                kwargs["cursor"] = cursor
            response = self.client.conversations_history(**kwargs)
            page = list(response.get("messages") or [])
            messages.extend(page)
            remaining -= len(page)
            metadata = response.get("response_metadata") or {}
            cursor = metadata.get("next_cursor")
            if not cursor or not page:
                break
        return messages

    def update_message(
        self,
        channel_id: str,
        ts: str,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "channel": channel_id,
            "ts": ts,
            "text": normalize_slack_mrkdwn(text),
        }
        rendered_blocks = blocks if blocks is not None else slack_blocks_for_markdown_table(text)
        auto_rendered_blocks = blocks is None and rendered_blocks is not None
        if rendered_blocks is not None:
            kwargs["blocks"] = rendered_blocks
        self._chat_update(kwargs, auto_rendered_blocks=auto_rendered_blocks)

    def _chat_post_message(
        self,
        kwargs: dict[str, Any],
        *,
        auto_rendered_blocks: bool,
    ):
        return self._call_slack_with_retries(
            self.client.chat_postMessage,
            kwargs,
            auto_rendered_blocks=auto_rendered_blocks,
            invalid_blocks_fallback=None,
        )

    def _chat_update(
        self,
        kwargs: dict[str, Any],
        *,
        auto_rendered_blocks: bool,
    ):
        return self._call_slack_with_retries(
            self.client.chat_update,
            kwargs,
            auto_rendered_blocks=auto_rendered_blocks,
            invalid_blocks_fallback=[],
        )

    def _call_slack_with_retries(
        self,
        call,
        kwargs: dict[str, Any],
        *,
        auto_rendered_blocks: bool,
        invalid_blocks_fallback: list[dict[str, Any]] | None,
    ):
        from slack_sdk.errors import SlackApiError

        attempts = 0
        current_kwargs = dict(kwargs)
        used_plain_text_fallback = False
        while True:
            try:
                return call(**current_kwargs)
            except SlackApiError as exc:
                error = exc.response.get("error")
                if error == "invalid_blocks" and "blocks" in current_kwargs:
                    if used_plain_text_fallback:
                        raise
                    if invalid_blocks_fallback is None:
                        current_kwargs.pop("blocks", None)
                    else:
                        current_kwargs["blocks"] = invalid_blocks_fallback
                    LOGGER.info(
                        "Slack rejected message blocks; retrying with text fallback",
                        extra={"auto_rendered_blocks": auto_rendered_blocks},
                    )
                    used_plain_text_fallback = True
                    continue
                if error == "ratelimited" and attempts < SLACK_API_RETRY_LIMIT:
                    attempts += 1
                    time.sleep(_retry_after_seconds(exc))
                    continue
                raise

    def add_reaction(self, channel_id: str, ts: str, reaction_name: str) -> bool:
        from slack_sdk.errors import SlackApiError

        try:
            self._call_reaction_with_retry(
                self.client.reactions_add,
                channel=channel_id,
                timestamp=ts,
                name=reaction_name,
            )
        except SlackApiError as exc:
            if exc.response.get("error") == "already_reacted":
                return False
            raise
        return True

    def remove_reaction(self, channel_id: str, ts: str, reaction_name: str) -> bool:
        from slack_sdk.errors import SlackApiError

        try:
            self._call_reaction_with_retry(
                self.client.reactions_remove,
                channel=channel_id,
                timestamp=ts,
                name=reaction_name,
            )
        except SlackApiError as exc:
            if exc.response.get("error") in {"no_reaction", "not_reacted"}:
                return False
            raise
        return True

    def _call_reaction_with_retry(self, call, **kwargs):
        # Without this retry, a transient ratelimit on reactions.remove leaves
        # stale status reactions on a message (e.g. :eyes: lingering after the
        # next add(:hourglass:) succeeds) and the message looks stuck in Slack.
        from slack_sdk.errors import SlackApiError

        attempts = 0
        while True:
            try:
                return call(**kwargs)
            except SlackApiError as exc:
                if exc.response.get("error") == "ratelimited" and attempts < SLACK_API_RETRY_LIMIT:
                    attempts += 1
                    time.sleep(_retry_after_seconds(exc))
                    continue
                raise


def _identity_name(identity: TeamAgent) -> str:
    name = getattr(identity, "username", identity.full_name)
    provider = getattr(identity, "provider_preference", None) or getattr(identity, "provider", None)
    if provider is None:
        return name
    return f"{name} [{provider.value}]"


def _retry_after_seconds(exc) -> float:
    headers = getattr(exc.response, "headers", None)
    if headers is None and isinstance(exc.response, dict):
        headers = exc.response.get("headers")
    raw_value = None
    if hasattr(headers, "get"):
        raw_value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(value, 30.0))
