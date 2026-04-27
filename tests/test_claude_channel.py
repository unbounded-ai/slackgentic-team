import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_harness.claude_channel import (
    CHANNEL_NAME,
    ClaudeChannelServer,
    install_claude_mcp_server,
    mcp_config,
)
from agent_harness.models import SlackThreadRef
from agent_harness.slack_agent_requests import SlackAgentRequestHandler
from agent_harness.slack_client import PostedMessage
from agent_harness.store import Store


class FakeGateway:
    def __init__(self):
        self.replies = []
        self.updates = []

    def post_thread_reply(self, thread, text, persona=None, icon_url=None, blocks=None):
        ts = f"1712345678.{len(self.replies):06d}"
        self.replies.append({"thread": thread, "text": text, "blocks": blocks, "ts": ts})
        return PostedMessage(thread.channel_id, ts, thread.thread_ts)

    def update_message(self, channel_id, ts, text, blocks=None):
        self.updates.append({"channel_id": channel_id, "ts": ts, "text": text, "blocks": blocks})


class ClaudeChannelTests(unittest.TestCase):
    def test_initialize_advertises_claude_channel_capability(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            output = io.StringIO()
            try:
                store.init_schema()
                server = ClaudeChannelServer(store, target_pid=123)

                with redirect_stdout(output):
                    server._handle_message(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {"protocolVersion": "2024-11-05"},
                        }
                    )

                response = json.loads(output.getvalue())
                self.assertEqual(response["id"], 1)
                self.assertIn("claude/channel", response["result"]["capabilities"]["experimental"])
                self.assertTrue(server._ready.is_set())
            finally:
                store.close()

    def test_deliver_pending_sends_claude_channel_notification(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            output = io.StringIO()
            try:
                store.init_schema()
                message_id = store.enqueue_claude_channel_message(
                    123,
                    "s1",
                    "continue please",
                    {"slack_channel": "C1", "slack_thread_ts": "171.000001", "bad-key": "ignored"},
                )
                server = ClaudeChannelServer(store, target_pid=123)

                with redirect_stdout(output):
                    server._deliver_pending()

                notification = json.loads(output.getvalue())
                self.assertEqual(notification["method"], "notifications/claude/channel")
                self.assertEqual(notification["params"]["content"], "continue please")
                self.assertEqual(
                    notification["params"]["meta"],
                    {"slack_channel": "C1", "slack_thread_ts": "171.000001"},
                )
                self.assertEqual(server._current_thread, SlackThreadRef("C1", "171.000001"))
                self.assertTrue(store.is_claude_channel_message_delivered(message_id))
            finally:
                store.close()

    def test_tools_list_advertises_slack_input_and_approval_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            output = io.StringIO()
            try:
                store.init_schema()
                server = ClaudeChannelServer(store, target_pid=123)

                with redirect_stdout(output):
                    server._handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

                response = json.loads(output.getvalue())
                tool_names = [tool["name"] for tool in response["result"]["tools"]]
                self.assertIn("request_user_input", tool_names)
                self.assertIn("request_approval", tool_names)
            finally:
                store.close()

    def test_request_approval_tool_posts_slack_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                handler = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=0.01,
                    store=store,
                    provider_label="Claude",
                )
                server = ClaudeChannelServer(store, target_pid=123, request_handler=handler)
                server._current_thread = SlackThreadRef("C1", "171.000001")

                result = server._handle_tool_call(
                    {
                        "name": "request_approval",
                        "arguments": {
                            "kind": "generic",
                            "title": "Run the deployment",
                            "reason": "The user asked for it.",
                        },
                    }
                )

                self.assertIn("Claude requests approval", gateway.replies[0]["text"])
                self.assertIn("abort", result["content"][0]["text"])
            finally:
                store.close()

    def test_mcp_config_points_claude_to_slackgentic_channel_command(self):
        config = mcp_config()

        self.assertEqual(
            config["mcpServers"]["slackgentic"],
            {"command": "slackgentic", "args": ["claude-channel"]},
        )

    def test_install_registers_user_scoped_claude_mcp_server(self):
        calls = []

        def fake_run(args, check):
            calls.append((args, check))

        with patch("subprocess.run", fake_run):
            install_claude_mcp_server("/opt/slackgentic")

        self.assertEqual(
            calls,
            [
                (
                    [
                        "claude",
                        "mcp",
                        "add",
                        "--scope",
                        "user",
                        CHANNEL_NAME,
                        "--",
                        "/opt/slackgentic",
                        "claude-channel",
                    ],
                    True,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
