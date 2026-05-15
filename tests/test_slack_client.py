import unittest

from slack_sdk.errors import SlackApiError

from agent_harness.slack.client import SlackGateway
from agent_harness.team import build_initial_model_team, build_initialization_messages


class FakeSlackClient:
    def __init__(self):
        self.messages = []
        self.updates = []
        self.archived_channels = []

    def auth_test(self):
        return type("SlackResponse", (), {"data": {"ok": True, "bot_id": "B1"}})()

    def chat_postMessage(self, **kwargs):
        ts = f"1712345678.{len(self.messages):06d}"
        self.messages.append({"ts": ts, **kwargs})
        return {"ts": ts}

    def chat_update(self, **kwargs):
        self.updates.append(kwargs)
        return {"ok": True}

    def users_info(self, user):
        return {
            "ok": True,
            "user": {
                "id": user,
                "name": "localuser",
                "profile": {
                    "display_name": "Local User",
                    "real_name": "Local",
                    "image_192": "https://example.com/avatar.png",
                },
            },
        }

    def conversations_archive(self, channel):
        self.archived_channels.append(channel)
        return {"ok": True}


class FakeSlackResponse(dict):
    def __init__(self, *args, headers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.headers = headers or {}


class InvalidBlocksOnceClient(FakeSlackClient):
    def __init__(self):
        super().__init__()
        self.calls = []

    def chat_postMessage(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1 and "blocks" in kwargs:
            raise SlackApiError(
                "invalid blocks",
                FakeSlackResponse({"ok": False, "error": "invalid_blocks"}),
            )
        return super().chat_postMessage(**kwargs)


class RateLimitedOnceClient(FakeSlackClient):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def chat_postMessage(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise SlackApiError(
                "rate limited",
                FakeSlackResponse(
                    {"ok": False, "error": "ratelimited"},
                    headers={"Retry-After": "0"},
                ),
            )
        return super().chat_postMessage(**kwargs)


class RateLimitedReactionsClient(FakeSlackClient):
    def __init__(self):
        super().__init__()
        self.add_calls = 0
        self.remove_calls = 0
        self.added = []
        self.removed = []

    def reactions_add(self, **kwargs):
        self.add_calls += 1
        if self.add_calls == 1:
            raise SlackApiError(
                "rate limited",
                FakeSlackResponse(
                    {"ok": False, "error": "ratelimited"},
                    headers={"Retry-After": "0"},
                ),
            )
        self.added.append(kwargs)
        return {"ok": True}

    def reactions_remove(self, **kwargs):
        self.remove_calls += 1
        if self.remove_calls == 1:
            raise SlackApiError(
                "rate limited",
                FakeSlackResponse(
                    {"ok": False, "error": "ratelimited"},
                    headers={"Retry-After": "0"},
                ),
            )
        self.removed.append(kwargs)
        return {"ok": True}


class InvalidUpdateBlocksOnceClient(FakeSlackClient):
    def __init__(self):
        super().__init__()
        self.calls = []

    def chat_update(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1 and "blocks" in kwargs:
            raise SlackApiError(
                "invalid blocks",
                FakeSlackResponse({"ok": False, "error": "invalid_blocks"}),
            )
        return super().chat_update(**kwargs)


class SlackGatewayTests(unittest.TestCase):
    def test_post_team_initialization_uses_agent_identities_in_one_thread(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = FakeSlackClient()
        agents = build_initial_model_team(codex_count=2, claude_count=1)
        messages = build_initialization_messages(agents)

        thread = gateway.post_team_initialization("C1", agents, messages)

        self.assertEqual(thread.channel_id, "C1")
        self.assertEqual(thread.thread_ts, "1712345678.000000")
        self.assertEqual(len(gateway.client.messages), len(messages))
        replies = gateway.client.messages[1:]
        self.assertTrue(all(message["thread_ts"] == thread.thread_ts for message in replies))
        self.assertEqual(gateway.client.messages[0]["username"], "Avery Chen [codex]")

    def test_auth_test_returns_plain_dict(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = FakeSlackClient()

        self.assertEqual(gateway.auth_test()["bot_id"], "B1")

    def test_user_display_name_uses_slack_profile(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = FakeSlackClient()

        self.assertEqual(gateway.user_display_name("U1"), "Local User")
        self.assertEqual(gateway.user_profile("U1").image_url, "https://example.com/avatar.png")

    def test_post_thread_reply_can_customize_human_username(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = FakeSlackClient()

        from agent_harness.models import SlackThreadRef

        gateway.post_thread_reply(SlackThreadRef("C1", "171.000001"), "hello", username="localuser")

        self.assertEqual(gateway.client.messages[0]["username"], "localuser")

    def test_post_thread_reply_renders_markdown_table_as_block(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = FakeSlackClient()

        from agent_harness.models import SlackThreadRef

        gateway.post_thread_reply(
            SlackThreadRef("C1", "171.000001"),
            "| Name | State |\n|---|---|\n| `silas` | busy |",
        )

        blocks = gateway.client.messages[0]["blocks"]
        self.assertEqual([block["type"] for block in blocks], ["table"])
        self.assertEqual(blocks[0]["rows"][1][0]["elements"][0]["elements"][0]["text"], "silas")

    def test_post_thread_reply_falls_back_when_auto_table_blocks_are_invalid(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = InvalidBlocksOnceClient()

        from agent_harness.models import SlackThreadRef

        gateway.post_thread_reply(
            SlackThreadRef("C1", "171.000001"),
            "| Name | State |\n|---|---|\n| `silas` | busy |",
        )

        self.assertIn("blocks", gateway.client.calls[0])
        self.assertNotIn("blocks", gateway.client.calls[1])
        self.assertNotIn("blocks", gateway.client.messages[0])

    def test_post_thread_reply_falls_back_when_explicit_blocks_are_invalid(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = InvalidBlocksOnceClient()

        from agent_harness.models import SlackThreadRef

        gateway.post_thread_reply(
            SlackThreadRef("C1", "171.000001"),
            "hello",
            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "hello"}}],
        )

        self.assertIn("blocks", gateway.client.calls[0])
        self.assertNotIn("blocks", gateway.client.calls[1])
        self.assertNotIn("blocks", gateway.client.messages[0])

    def test_post_thread_reply_retries_after_rate_limit(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = RateLimitedOnceClient()

        from agent_harness.models import SlackThreadRef

        gateway.post_thread_reply(SlackThreadRef("C1", "171.000001"), "hello")

        self.assertEqual(gateway.client.calls, 2)
        self.assertEqual(gateway.client.messages[0]["text"], "hello")

    def test_archive_channel_calls_slack_api(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = FakeSlackClient()

        self.assertTrue(gateway.archive_channel("C1"))

        self.assertEqual(gateway.client.archived_channels, ["C1"])

    def test_update_message_can_send_empty_blocks_to_clear_buttons(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = FakeSlackClient()

        gateway.update_message("C1", "171.000001", "done", blocks=[])

        self.assertEqual(gateway.client.updates[0]["blocks"], [])

    def test_update_message_renders_markdown_table_as_block(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = FakeSlackClient()

        gateway.update_message("C1", "171.000001", "| Name | State |\n|---|---|\n| Nell | free |")

        self.assertEqual(gateway.client.updates[0]["blocks"][0]["type"], "table")

    def test_update_message_clears_blocks_when_rendered_blocks_are_invalid(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = InvalidUpdateBlocksOnceClient()

        gateway.update_message("C1", "171.000001", "| Name | State |\n|---|---|\n| Nell | free |")

        self.assertEqual(gateway.client.calls[1]["blocks"], [])
        self.assertEqual(gateway.client.updates[0]["blocks"], [])

    def test_add_reaction_retries_after_rate_limit(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = RateLimitedReactionsClient()

        self.assertTrue(gateway.add_reaction("C1", "171.000001", "eyes"))

        self.assertEqual(gateway.client.add_calls, 2)
        self.assertEqual(gateway.client.added[0]["name"], "eyes")

    def test_remove_reaction_retries_after_rate_limit(self):
        gateway = object.__new__(SlackGateway)
        gateway.client = RateLimitedReactionsClient()

        self.assertTrue(gateway.remove_reaction("C1", "171.000001", "eyes"))

        self.assertEqual(gateway.client.remove_calls, 2)
        self.assertEqual(gateway.client.removed[0]["name"], "eyes")


if __name__ == "__main__":
    unittest.main()
