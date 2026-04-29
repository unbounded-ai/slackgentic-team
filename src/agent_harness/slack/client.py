from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent_harness.models import SlackThreadRef, TeamAgent
from agent_harness.slack import normalize_slack_mrkdwn
from agent_harness.team import TeamChatMessage


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
        if blocks:
            kwargs["blocks"] = blocks
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        response = self.client.chat_postMessage(**kwargs)
        return PostedMessage(channel_id=channel_id, ts=response["ts"], thread_ts=thread_ts)

    def post_session_parent(
        self,
        channel_id: str,
        text: str,
        persona: TeamAgent,
        icon_url: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> PostedMessage:
        response = self.client.chat_postMessage(
            channel=channel_id,
            text=normalize_slack_mrkdwn(text),
            blocks=blocks,
            username=_identity_name(persona),
            icon_url=icon_url,
            icon_emoji=None if icon_url else persona.icon_emoji,
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
        if blocks:
            kwargs["blocks"] = blocks
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
        response = self.client.chat_postMessage(**kwargs)
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
    ) -> list[dict[str, Any]]:
        response = self.client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            limit=limit,
        )
        return list(response.get("messages") or [])

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
        if blocks is not None:
            kwargs["blocks"] = blocks
        self.client.chat_update(**kwargs)

    def add_reaction(self, channel_id: str, ts: str, reaction_name: str) -> bool:
        from slack_sdk.errors import SlackApiError

        try:
            self.client.reactions_add(channel=channel_id, timestamp=ts, name=reaction_name)
        except SlackApiError as exc:
            if exc.response.get("error") == "already_reacted":
                return False
            raise
        return True

    def remove_reaction(self, channel_id: str, ts: str, reaction_name: str) -> bool:
        from slack_sdk.errors import SlackApiError

        try:
            self.client.reactions_remove(channel=channel_id, timestamp=ts, name=reaction_name)
        except SlackApiError as exc:
            if exc.response.get("error") in {"no_reaction", "not_reacted"}:
                return False
            raise
        return True


def _identity_name(identity: TeamAgent) -> str:
    name = getattr(identity, "username", identity.full_name)
    provider = getattr(identity, "provider_preference", None) or getattr(identity, "provider", None)
    if provider is None:
        return name
    return f"{name} [{provider.value}]"
