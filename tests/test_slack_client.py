import unittest

from agent_harness.slack_client import SlackGateway
from agent_harness.team import build_initial_model_team, build_initialization_messages


class FakeSlackClient:
    def __init__(self):
        self.messages = []

    def auth_test(self):
        return type("SlackResponse", (), {"data": {"ok": True, "bot_id": "B1"}})()

    def chat_postMessage(self, **kwargs):
        ts = f"1712345678.{len(self.messages):06d}"
        self.messages.append({"ts": ts, **kwargs})
        return {"ts": ts}

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


if __name__ == "__main__":
    unittest.main()
