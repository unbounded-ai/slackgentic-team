import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_harness.models import SlackThreadRef
from agent_harness.sessions.claude_channel import (
    CHANNEL_NAME,
    ClaudeChannelServer,
    install_claude_mcp_server,
    is_slackgentic_mcp_server_configured,
    mcp_config,
    session_transcript_has_slackgentic_mcp,
)
from agent_harness.slack import decode_action_value
from agent_harness.slack.agent_requests import SlackAgentRequestHandler
from agent_harness.slack.client import PostedMessage
from agent_harness.storage.store import Store


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
                self.assertIn("tools", response["result"]["capabilities"])
                self.assertIn("claude/channel", response["result"]["capabilities"]["experimental"])
                self.assertIn(
                    "claude/channel/permission",
                    response["result"]["capabilities"]["experimental"],
                )
                self.assertIn("Slackgentic MCP tools", response["result"]["instructions"])
                self.assertIn("Never quote, repeat", response["result"]["instructions"])
                self.assertIn("request_user_input", response["result"]["instructions"])
                self.assertIn("multiple options", response["result"]["instructions"])
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
                tools = {tool["name"]: tool for tool in response["result"]["tools"]}
                self.assertIn("request_user_input", tools)
                self.assertIn("request_approval", tools)
                self.assertIn("multiple options", tools["request_user_input"]["description"])
                self.assertIn("one concrete action", tools["request_approval"]["description"])
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

    def test_permission_request_notification_posts_slack_request_and_returns_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            output = io.StringIO()
            try:
                store.init_schema()
                handler = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=2,
                    store=store,
                    provider_label="Claude",
                    poll_seconds=0.01,
                )
                server = ClaudeChannelServer(store, target_pid=123, request_handler=handler)
                server._current_thread = SlackThreadRef("C1", "171.000001")

                with redirect_stdout(output):
                    server._handle_message(
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/claude/channel/permission_request",
                            "params": {
                                "request_id": "req-1",
                                "tool_name": "Bash",
                                "description": "List files",
                                "input_preview": "ls ~/code",
                            },
                        }
                    )
                    self.assertTrue(_wait_for(lambda: bool(gateway.replies)))
                    actions = next(
                        block
                        for block in gateway.replies[0]["blocks"]
                        if block["type"] == "actions"
                    )
                    value = actions["elements"][0]["value"]
                    self.assertTrue(
                        handler.handle_block_action(
                            decode_action_value(value),
                            "C1",
                            gateway.replies[0]["ts"],
                        )
                    )
                    self.assertTrue(
                        _wait_for(
                            lambda: "notifications/claude/channel/permission" in output.getvalue()
                        )
                    )

                notification = json.loads(output.getvalue().splitlines()[-1])
                self.assertEqual(notification["method"], "notifications/claude/channel/permission")
                self.assertEqual(
                    notification["params"],
                    {"request_id": "req-1", "behavior": "allow"},
                )
                self.assertIn("Claude requests tool approval", gateway.replies[0]["text"])
                self.assertEqual(gateway.updates[-1]["text"], "Allowed Claude tool request.")
            finally:
                store.close()

    def test_slackgentic_mcp_permission_request_is_auto_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            output = io.StringIO()
            try:
                store.init_schema()
                handler = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=2,
                    store=store,
                    provider_label="Claude",
                    poll_seconds=0.01,
                )
                server = ClaudeChannelServer(store, target_pid=123, request_handler=handler)
                server._current_thread = SlackThreadRef("C1", "171.000001")

                for tool_name in (
                    "mcp__slackgentic__request_approval",
                    "mcp__slackgentic__request_user_input",
                ):
                    with redirect_stdout(output):
                        server._handle_message(
                            {
                                "jsonrpc": "2.0",
                                "method": "notifications/claude/channel/permission_request",
                                "params": {
                                    "request_id": tool_name,
                                    "tool_name": tool_name,
                                    "description": "Ask Slack",
                                    "input_preview": "{}",
                                },
                            }
                        )
                        self.assertTrue(
                            _wait_for(
                                lambda: (
                                    output.getvalue().count(
                                        "notifications/claude/channel/permission"
                                    )
                                    >= 1
                                )
                            )
                        )

                    notification = json.loads(output.getvalue().splitlines()[-1])
                    self.assertEqual(
                        notification["params"],
                        {"request_id": tool_name, "behavior": "allow"},
                    )
                    output.seek(0)
                    output.truncate(0)

                self.assertEqual(gateway.replies, [])
                row_count = store.conn.execute(
                    "SELECT COUNT(*) FROM slack_agent_requests"
                ).fetchone()
                self.assertEqual(row_count[0], 0)
            finally:
                store.close()

    def test_permission_request_preserves_full_tool_input_when_present(self):
        class CapturingHandler:
            def __init__(self):
                self.params = None

            def handle_persistent_request(self, method, params, thread, provider_label=None):
                self.params = params
                return {"behavior": "allow"}

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            handler = CapturingHandler()
            output = io.StringIO()
            try:
                store.init_schema()
                server = ClaudeChannelServer(store, target_pid=123, request_handler=handler)
                server._current_thread = SlackThreadRef("C1", "171.000001")

                with redirect_stdout(output):
                    server._handle_permission_request_worker(
                        "req-1",
                        {
                            "tool_name": "Edit",
                            "description": "Edit file",
                            "input_preview": '{"file_path":"truncated…',
                            "input": {"file_path": "/tmp/README.md", "new_string": "after"},
                            "display": {"diff": "diff text"},
                        },
                    )

                self.assertEqual(
                    handler.params["input"],
                    {"file_path": "/tmp/README.md", "new_string": "after"},
                )
                self.assertEqual(handler.params["display"], {"diff": "diff text"})
                notification = json.loads(output.getvalue())
                self.assertEqual(notification["params"]["behavior"], "allow")
            finally:
                store.close()

    def test_mcp_config_points_claude_to_slackgentic_channel_command(self):
        config = mcp_config()

        self.assertEqual(
            config["mcpServers"]["slackgentic"],
            {"command": "slackgentic", "args": ["claude-channel"]},
        )

    def test_mcp_config_includes_python_module_invocation_args(self):
        config = mcp_config("/usr/bin/python3", ["-m", "agent_harness"])

        self.assertEqual(
            config["mcpServers"]["slackgentic"],
            {"command": "/usr/bin/python3", "args": ["-m", "agent_harness", "claude-channel"]},
        )

    def test_detects_configured_slackgentic_mcp_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".claude.json").write_text(
                json.dumps({"mcpServers": {"slackgentic": {"command": "slackgentic"}}})
            )

            self.assertTrue(is_slackgentic_mcp_server_configured(home))

    def test_detects_loaded_slackgentic_mcp_tools_in_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "claude.jsonl"
            transcript.write_text(
                json.dumps(
                    {
                        "attachment": {
                            "type": "deferred_tools_delta",
                            "addedNames": ["mcp__slackgentic__request_user_input"],
                        }
                    }
                )
                + "\n"
            )

            self.assertTrue(session_transcript_has_slackgentic_mcp(transcript))

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

    def test_install_registers_python_module_when_slackgentic_script_is_unavailable(self):
        calls = []

        def fake_run(args, check):
            calls.append((args, check))

        with (
            patch("agent_harness.sessions.claude_channel.shutil.which", return_value=None),
            patch.object(sys, "argv", ["/repo/src/agent_harness/__main__.py"]),
            patch("subprocess.run", fake_run),
        ):
            install_claude_mcp_server()

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
                        sys.executable,
                        "-m",
                        "agent_harness",
                        "claude-channel",
                    ],
                    True,
                )
            ],
        )


def _wait_for(predicate, attempts=100):
    event = threading.Event()
    for _ in range(attempts):
        if predicate():
            return True
        event.wait(0.01)
    return False


if __name__ == "__main__":
    unittest.main()
