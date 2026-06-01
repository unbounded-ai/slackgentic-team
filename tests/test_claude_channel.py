import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_harness.models import (
    AgentSession,
    PermissionMode,
    Provider,
    SessionStatus,
    SlackThreadRef,
)
from agent_harness.permissions import CLAUDE_CHANNEL_PERMISSION_MODE_ENV
from agent_harness.sessions.claude_channel import (
    CHANNEL_NAME,
    CODEX_MCP_INSTRUCTIONS,
    SLACK_THREAD_CHANNEL_ENV,
    SLACK_THREAD_TS_ENV,
    SLACKGENTIC_MCP_PERMISSION_ALLOW,
    ClaudeChannelServer,
    ensure_claude_mcp_permissions,
    ensure_claude_native_input_hook,
    ensure_codex_mcp_server_registered,
    handle_native_input_hook,
    install_claude_mcp_server,
    install_codex_mcp_server,
    is_codex_mcp_server_configured,
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

    def test_tools_list_advertises_slack_input_approval_and_pr_tools(self):
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
                self.assertIn("create_pull_request", tools)
                self.assertIn("request_user_input", tools)
                self.assertIn("request_approval", tools)
                self.assertIn("GitHub pull request", tools["create_pull_request"]["description"])
                self.assertIn("multiple options", tools["request_user_input"]["description"])
                self.assertIn("one concrete action", tools["request_approval"]["description"])
            finally:
                store.close()

    def test_codex_mcp_instructions_focus_on_pr_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            output = io.StringIO()
            try:
                store.init_schema()
                server = ClaudeChannelServer(
                    store,
                    target_pid=123,
                    instructions=CODEX_MCP_INSTRUCTIONS,
                )

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
                self.assertIn("create_pull_request", response["result"]["instructions"])
                self.assertNotIn("Claude Code session", response["result"]["instructions"])
            finally:
                store.close()

    def test_current_thread_can_be_seeded_from_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                with patch.dict(
                    "os.environ",
                    {
                        SLACK_THREAD_CHANNEL_ENV: "C1",
                        SLACK_THREAD_TS_ENV: "171.000001",
                    },
                ):
                    server = ClaudeChannelServer(store, target_pid=123)

                self.assertEqual(server._current_thread, SlackThreadRef("C1", "171.000001"))
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

    def test_create_pull_request_tool_runs_gh_pr_create(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            if args == ["git", "rev-parse", "--is-inside-work-tree"]:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="not a git repo")
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="https://github.com/example-org/example-repo/pull/12\n",
                stderr="",
            )

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            repo = Path(tmp) / "repo"
            repo.mkdir()
            try:
                store.init_schema()
                server = ClaudeChannelServer(
                    store,
                    target_pid=123,
                    command_runner=fake_run,
                )

                result = server._handle_tool_call(
                    {
                        "name": "create_pull_request",
                        "arguments": {
                            "title": "Update docs",
                            "body": "Summary",
                            "base": "main",
                            "head": "feature/docs",
                            "repo": "example-org/example-repo",
                            "cwd": str(repo),
                            "draft": True,
                        },
                    }
                )

                self.assertEqual(
                    result["content"][0]["text"],
                    "https://github.com/example-org/example-repo/pull/12",
                )
                self.assertEqual(
                    calls[-1][0],
                    [
                        "gh",
                        "pr",
                        "create",
                        "--title",
                        "Update docs",
                        "--body",
                        "Summary",
                        "--base",
                        "main",
                        "--head",
                        "feature/docs",
                        "--repo",
                        "example-org/example-repo",
                        "--draft",
                    ],
                )
                self.assertEqual(calls[-1][1]["cwd"], repo.resolve())
                self.assertTrue(calls[-1][1]["capture_output"])
                self.assertEqual(calls[-1][1]["timeout"], 120)
            finally:
                store.close()

    def test_create_pull_request_tool_pushes_unpushed_current_branch(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            if args == ["git", "rev-parse", "--is-inside-work-tree"]:
                return subprocess.CompletedProcess(args, 0, stdout="true\n", stderr="")
            if args == ["git", "branch", "--show-current"]:
                return subprocess.CompletedProcess(args, 0, stdout="feature/docs\n", stderr="")
            if args == ["git", "remote"]:
                return subprocess.CompletedProcess(args, 0, stdout="origin\n", stderr="")
            if args[:5] == [
                "git",
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "feature/docs@{upstream}",
            ]:
                return subprocess.CompletedProcess(args, 128, stdout="", stderr="no upstream")
            if args[:3] == ["git", "push", "--set-upstream"]:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            if args[:3] == ["gh", "pr", "create"]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="https://github.com/example-org/example-repo/pull/12\n",
                    stderr="",
                )
            self.fail(f"unexpected command: {args}")

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            repo = Path(tmp) / "repo"
            repo.mkdir()
            try:
                store.init_schema()
                server = ClaudeChannelServer(store, target_pid=123, command_runner=fake_run)

                result = server._handle_tool_call(
                    {
                        "name": "create_pull_request",
                        "arguments": {
                            "title": "Update docs",
                            "body": "Summary",
                            "base": "main",
                            "cwd": str(repo),
                        },
                    }
                )

                self.assertEqual(
                    result["content"][0]["text"],
                    "https://github.com/example-org/example-repo/pull/12",
                )
                self.assertIn(
                    (
                        [
                            "git",
                            "push",
                            "--set-upstream",
                            "origin",
                            "refs/heads/feature/docs:refs/heads/feature/docs",
                        ],
                        {
                            "cwd": repo.resolve(),
                            "text": True,
                            "capture_output": True,
                            "check": False,
                            "timeout": 120,
                        },
                    ),
                    calls,
                )
                self.assertIn("--head", calls[-1][0])
                self.assertIn("feature/docs", calls[-1][0])
            finally:
                store.close()

    def test_create_pull_request_tool_pushes_requested_local_head(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            if args == ["git", "rev-parse", "--is-inside-work-tree"]:
                return subprocess.CompletedProcess(args, 0, stdout="true\n", stderr="")
            if args == [
                "git",
                "show-ref",
                "--verify",
                "--quiet",
                "refs/heads/feature/docs",
            ]:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            if args == ["git", "remote"]:
                return subprocess.CompletedProcess(args, 0, stdout="origin\n", stderr="")
            if args[:5] == [
                "git",
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "feature/docs@{upstream}",
            ]:
                return subprocess.CompletedProcess(args, 128, stdout="", stderr="no upstream")
            if args[:3] == ["git", "push", "--set-upstream"]:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            if args[:3] == ["gh", "pr", "create"]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="https://github.com/example-org/example-repo/pull/12\n",
                    stderr="",
                )
            self.fail(f"unexpected command: {args}")

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            repo = Path(tmp) / "repo"
            repo.mkdir()
            try:
                store.init_schema()
                server = ClaudeChannelServer(store, target_pid=123, command_runner=fake_run)

                result = server._handle_tool_call(
                    {
                        "name": "create_pull_request",
                        "arguments": {
                            "title": "Update docs",
                            "head": "feature/docs",
                            "cwd": str(repo),
                        },
                    }
                )

                self.assertNotIn("isError", result)
                self.assertEqual(
                    calls[-2][0],
                    [
                        "git",
                        "push",
                        "--set-upstream",
                        "origin",
                        "refs/heads/feature/docs:refs/heads/feature/docs",
                    ],
                )
                self.assertEqual(
                    calls[-1][0],
                    [
                        "gh",
                        "pr",
                        "create",
                        "--title",
                        "Update docs",
                        "--body",
                        "",
                        "--head",
                        "feature/docs",
                    ],
                )
            finally:
                store.close()

    def test_create_pull_request_tool_uses_upstream_head_name_after_push(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            if args == ["git", "rev-parse", "--is-inside-work-tree"]:
                return subprocess.CompletedProcess(args, 0, stdout="true\n", stderr="")
            if args == ["git", "branch", "--show-current"]:
                return subprocess.CompletedProcess(args, 0, stdout="feature/docs\n", stderr="")
            if args == ["git", "remote"]:
                return subprocess.CompletedProcess(args, 0, stdout="origin\n", stderr="")
            if args[:5] == [
                "git",
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "feature/docs@{upstream}",
            ]:
                return subprocess.CompletedProcess(
                    args, 0, stdout="origin/review/feature-docs\n", stderr=""
                )
            if args[:3] == ["git", "push", "--set-upstream"]:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            if args[:3] == ["gh", "pr", "create"]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="https://github.com/example-org/example-repo/pull/12\n",
                    stderr="",
                )
            self.fail(f"unexpected command: {args}")

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            repo = Path(tmp) / "repo"
            repo.mkdir()
            try:
                store.init_schema()
                server = ClaudeChannelServer(store, target_pid=123, command_runner=fake_run)

                result = server._handle_tool_call(
                    {
                        "name": "create_pull_request",
                        "arguments": {
                            "title": "Update docs",
                            "body": "Summary",
                            "base": "main",
                            "cwd": str(repo),
                        },
                    }
                )

                self.assertNotIn("isError", result)
                self.assertEqual(
                    calls[-2][0],
                    [
                        "git",
                        "push",
                        "--set-upstream",
                        "origin",
                        "refs/heads/feature/docs:refs/heads/review/feature-docs",
                    ],
                )
                self.assertIn("--head", calls[-1][0])
                self.assertIn("review/feature-docs", calls[-1][0])
            finally:
                store.close()

    def test_create_pull_request_tool_reports_branch_push_failure(self):
        def fake_run(args, **kwargs):
            if args == ["git", "rev-parse", "--is-inside-work-tree"]:
                return subprocess.CompletedProcess(args, 0, stdout="true\n", stderr="")
            if args == ["git", "branch", "--show-current"]:
                return subprocess.CompletedProcess(args, 0, stdout="feature/docs\n", stderr="")
            if args == ["git", "remote"]:
                return subprocess.CompletedProcess(args, 0, stdout="origin\n", stderr="")
            if args[:5] == [
                "git",
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "feature/docs@{upstream}",
            ]:
                return subprocess.CompletedProcess(args, 128, stdout="", stderr="no upstream")
            if args[:3] == ["git", "push", "--set-upstream"]:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="rejected\n")
            self.fail(f"unexpected command: {args}")

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            repo = Path(tmp) / "repo"
            repo.mkdir()
            try:
                store.init_schema()
                server = ClaudeChannelServer(store, target_pid=123, command_runner=fake_run)

                result = server._handle_tool_call(
                    {
                        "name": "create_pull_request",
                        "arguments": {"title": "Update docs", "cwd": str(repo)},
                    }
                )

                self.assertTrue(result["isError"])
                self.assertIn("Failed to push branch", result["content"][0]["text"])
                self.assertIn("rejected", result["content"][0]["text"])
            finally:
                store.close()

    def test_create_pull_request_tool_times_out_instead_of_hanging(self):
        def fake_run(args, **kwargs):
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            repo = Path(tmp) / "repo"
            repo.mkdir()
            try:
                store.init_schema()
                server = ClaudeChannelServer(store, target_pid=123, command_runner=fake_run)

                result = server._handle_tool_call(
                    {
                        "name": "create_pull_request",
                        "arguments": {
                            "title": "Update docs",
                            "head": "example-org:feature/docs",
                            "cwd": str(repo),
                        },
                    }
                )

                self.assertTrue(result["isError"])
                self.assertIn("gh pr create", result["content"][0]["text"])
                self.assertIn("timed out after 120 seconds", result["content"][0]["text"])
            finally:
                store.close()

    def test_create_pull_request_tool_reports_missing_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                server = ClaudeChannelServer(store, target_pid=123)

                result = server._handle_tool_call(
                    {"name": "create_pull_request", "arguments": {"body": "Summary"}}
                )

                self.assertTrue(result["isError"])
                self.assertIn("requires a title", result["content"][0]["text"])
            finally:
                store.close()

    def test_create_pull_request_tool_treats_existing_pr_url_as_success(self):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(
                args,
                1,
                stdout="",
                stderr=(
                    "a pull request for branch already exists: "
                    "https://github.com/example-org/example-repo/pull/12\n"
                ),
            )

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                server = ClaudeChannelServer(
                    store,
                    target_pid=123,
                    command_runner=fake_run,
                )

                result = server._handle_tool_call(
                    {
                        "name": "create_pull_request",
                        "arguments": {"title": "Update docs", "cwd": tmp},
                    }
                )

                self.assertNotIn("isError", result)
                self.assertEqual(
                    result["content"][0]["text"],
                    "https://github.com/example-org/example-repo/pull/12",
                )
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

                for tool_name in SLACKGENTIC_MCP_PERMISSION_ALLOW:
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

    def test_dangerous_mode_env_auto_allows_permission_request(self):
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

                from agent_harness.sessions.claude_channel import DANGEROUS_MODE_ENV

                with (
                    redirect_stdout(output),
                    patch.dict(os.environ, {DANGEROUS_MODE_ENV: "1"}),
                ):
                    server._handle_message(
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/claude/channel/permission_request",
                            "params": {
                                "request_id": "req-dangerous",
                                "tool_name": "Bash",
                                "description": "run launchctl",
                                "input_preview": "launchctl kickstart -k ...",
                            },
                        }
                    )
                    self.assertTrue(
                        _wait_for(
                            lambda: "notifications/claude/channel/permission" in output.getvalue()
                        )
                    )

                notification = json.loads(output.getvalue().splitlines()[-1])
                self.assertEqual(
                    notification["params"],
                    {"request_id": "req-dangerous", "behavior": "allow"},
                )
                self.assertEqual(gateway.replies, [], "no Slack approval round-trip expected")
                self.assertEqual(
                    store.conn.execute("SELECT COUNT(*) FROM slack_agent_requests").fetchone()[0],
                    0,
                )
            finally:
                store.close()

    def test_safe_auto_env_auto_allows_read_only_bash_permission_requests(self):
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
                requests = (
                    (
                        "req-git-status",
                        {
                            "command": ("git -C /workspace/repos/example-project status"),
                            "description": "git status of example repo",
                        },
                    ),
                    (
                        "req-ls-wc",
                        {
                            "command": (
                                "ls /tmp/example-provider-analysis.patch "
                                "2>&1 && wc -l "
                                "/tmp/example-provider-analysis.patch"
                            ),
                            "description": "check patch file exists",
                        },
                    ),
                    (
                        "req-git-status-head",
                        {
                            "command": (
                                "git -C /workspace/repos/example-project status 2>&1 | head -30"
                            ),
                            "description": "check example repo status",
                        },
                    ),
                    (
                        "req-ls-grep-sequence",
                        {
                            "command": (
                                "ls /tmp/example-provider-analysis.patch 2>&1; "
                                "ls /tmp/ | grep -i example 2>&1"
                            ),
                            "description": "check for prior patch file",
                        },
                    ),
                    (
                        "req-git-pull-ff-only",
                        {
                            "command": ("git -C /workspace/repos/example-project pull --ff-only"),
                            "description": "fast-forward example repo main",
                        },
                    ),
                    (
                        "req-gh-pr-create",
                        {
                            "command": ("gh pr create --title update --body summary"),
                            "description": "create pull request",
                        },
                    ),
                    (
                        "req-git-config-sequence",
                        {
                            "command": (
                                "git -C /workspace/repos/example-project config user.name; "
                                "git -C /workspace/repos/example-project config user.email"
                            ),
                            "description": "check git author config",
                        },
                    ),
                    (
                        "req-git-add",
                        {
                            "command": (
                                "git -C /workspace/repos/example-project add "
                                "docs/vendors/index.html docs/vendors/modal_embeddings.html "
                                "docs/vendors/vendor-style.css "
                                "docs/vendors/inference_provider.html"
                            ),
                            "description": "stage vendor docs",
                        },
                    ),
                    (
                        "req-git-commit",
                        {
                            "command": (
                                "cd /workspace/repos/example-project && "
                                "git commit -m \"$(cat <<'EOF'\n"
                                "[docs][vendors] Add Qwen3 inference decision\n\n"
                                "Refresh embedding backends.\n"
                                "EOF\n"
                                ')"'
                            ),
                            "description": "commit vendor docs",
                        },
                    ),
                )

                with (
                    redirect_stdout(output),
                    patch.dict(
                        os.environ,
                        {CLAUDE_CHANNEL_PERMISSION_MODE_ENV: PermissionMode.SAFE_AUTO.value},
                    ),
                ):
                    for request_id, input_preview in requests:
                        server._handle_message(
                            {
                                "jsonrpc": "2.0",
                                "method": "notifications/claude/channel/permission_request",
                                "params": {
                                    "request_id": request_id,
                                    "tool_name": "Bash",
                                    "description": input_preview["description"],
                                    "input_preview": json.dumps(input_preview),
                                },
                            }
                        )
                    self.assertTrue(
                        _wait_for(
                            lambda: (
                                output.getvalue().count("notifications/claude/channel/permission")
                                == len(requests)
                            )
                        )
                    )

                notifications = [
                    json.loads(line) for line in output.getvalue().splitlines() if line.strip()
                ]
                self.assertEqual(
                    [notification["params"] for notification in notifications],
                    [
                        {"request_id": "req-git-status", "behavior": "allow"},
                        {"request_id": "req-ls-wc", "behavior": "allow"},
                        {"request_id": "req-git-status-head", "behavior": "allow"},
                        {"request_id": "req-ls-grep-sequence", "behavior": "allow"},
                        {"request_id": "req-git-pull-ff-only", "behavior": "allow"},
                        {"request_id": "req-gh-pr-create", "behavior": "allow"},
                        {"request_id": "req-git-config-sequence", "behavior": "allow"},
                        {"request_id": "req-git-add", "behavior": "allow"},
                        {"request_id": "req-git-commit", "behavior": "allow"},
                    ],
                )
                self.assertEqual(gateway.replies, [], "no Slack approval round-trip expected")
                self.assertEqual(
                    store.conn.execute("SELECT COUNT(*) FROM slack_agent_requests").fetchone()[0],
                    0,
                )
            finally:
                store.close()

    def test_safe_auto_env_still_routes_mutating_bash_permission_requests_to_slack(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            output = io.StringIO()
            try:
                store.init_schema()
                handler = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=0.01,
                    store=store,
                    provider_label="Claude",
                    poll_seconds=0.01,
                )
                server = ClaudeChannelServer(store, target_pid=123, request_handler=handler)
                server._current_thread = SlackThreadRef("C1", "171.000001")

                with (
                    redirect_stdout(output),
                    patch.dict(
                        os.environ,
                        {CLAUDE_CHANNEL_PERMISSION_MODE_ENV: PermissionMode.SAFE_AUTO.value},
                    ),
                ):
                    server._handle_permission_request_worker(
                        "req-commit",
                        {
                            "tool_name": "Bash",
                            "description": "Reset changes",
                            "input_preview": json.dumps({"command": "git -C /repo reset --hard"}),
                        },
                    )

                notification = json.loads(output.getvalue())
                self.assertEqual(
                    notification["params"],
                    {"request_id": "req-commit", "behavior": "deny"},
                )
                self.assertEqual(
                    gateway.replies[0]["text"],
                    "Claude requests command approval: git",
                )
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

    def test_detects_configured_codex_mcp_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            config = home / ".codex" / "config.toml"
            config.parent.mkdir()
            config.write_text(
                '[mcp_servers.slackgentic]\ncommand = "slackgentic"\nargs = ["codex-mcp"]\n',
                encoding="utf-8",
            )

            self.assertTrue(is_codex_mcp_server_configured(home))

    def test_ensure_codex_mcp_server_registers_when_missing(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "agent_harness.sessions.claude_channel.shutil.which",
                return_value="/usr/bin/codex-custom",
            ),
            patch("agent_harness.sessions.claude_channel.install_codex_mcp_server") as install,
        ):
            home = Path(tmp)

            registered = ensure_codex_mcp_server_registered(
                "/opt/slackgentic",
                home,
                codex_binary="codex-custom",
            )

        self.assertTrue(registered)
        install.assert_called_once_with(
            "/opt/slackgentic",
            home,
            codex_binary="codex-custom",
        )

    def test_ensure_codex_mcp_server_skips_when_configured(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("agent_harness.sessions.claude_channel.install_codex_mcp_server") as install,
        ):
            home = Path(tmp)
            config = home / ".codex" / "config.toml"
            config.parent.mkdir()
            config.write_text(
                '[mcp_servers.slackgentic]\ncommand = "slackgentic"\nargs = ["codex-mcp"]\n',
                encoding="utf-8",
            )

            registered = ensure_codex_mcp_server_registered(home=home)

        self.assertFalse(registered)
        install.assert_not_called()

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

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            return type(
                "Completed", (), {"returncode": 0, "stdout": "", "stderr": "", "args": args}
            )()

        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run", fake_run):
            install_claude_mcp_server("/opt/slackgentic", home=Path(tmp))

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
                    {"check": False, "capture_output": True, "text": True},
                )
            ],
        )

    def test_install_registers_codex_mcp_server(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            return type(
                "Completed", (), {"returncode": 0, "stdout": "", "stderr": "", "args": args}
            )()

        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run", fake_run):
            install_codex_mcp_server("/opt/slackgentic", home=Path(tmp))

        self.assertEqual(
            calls[0][0],
            [
                "codex",
                "mcp",
                "add",
                CHANNEL_NAME,
                "--",
                "/opt/slackgentic",
                "codex-mcp",
            ],
        )
        self.assertFalse(calls[0][1]["check"])
        self.assertTrue(calls[0][1]["capture_output"])
        self.assertTrue(calls[0][1]["text"])
        self.assertEqual(calls[0][1]["env"]["CODEX_HOME"], str(Path(tmp) / ".codex"))

    def test_install_replaces_existing_codex_mcp_server(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:3] == ["codex", "mcp", "add"] and len(calls) == 1:
                return type(
                    "Completed",
                    (),
                    {
                        "returncode": 1,
                        "stdout": "MCP server slackgentic already exists",
                        "stderr": "",
                        "args": args,
                    },
                )()
            return type(
                "Completed", (), {"returncode": 0, "stdout": "", "stderr": "", "args": args}
            )()

        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run", fake_run):
            install_codex_mcp_server("/opt/slackgentic", home=Path(tmp))

        self.assertEqual(
            [call[0][:3] for call in calls],
            [
                ["codex", "mcp", "add"],
                ["codex", "mcp", "remove"],
                ["codex", "mcp", "add"],
            ],
        )
        for _args, kwargs in calls:
            self.assertEqual(kwargs["env"]["CODEX_HOME"], str(Path(tmp) / ".codex"))

    def test_install_registers_python_module_when_slackgentic_script_is_unavailable(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            return type(
                "Completed", (), {"returncode": 0, "stdout": "", "stderr": "", "args": args}
            )()

        with (
            patch("agent_harness.sessions.claude_channel.shutil.which", return_value=None),
            patch.object(sys, "argv", ["/repo/src/agent_harness/__main__.py"]),
            patch("subprocess.run", fake_run),
            tempfile.TemporaryDirectory() as tmp,
        ):
            install_claude_mcp_server(home=Path(tmp))

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
                    {"check": False, "capture_output": True, "text": True},
                )
            ],
        )

    def test_install_replaces_existing_mcp_server_and_allows_permissions(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:3] == ["claude", "mcp", "add"] and len(calls) == 1:
                return type(
                    "Completed",
                    (),
                    {
                        "returncode": 1,
                        "stdout": "MCP server slackgentic already exists in user config",
                        "stderr": "",
                        "args": args,
                    },
                )()
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "", "stderr": "", "args": args},
            )()

        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run", fake_run):
            install_claude_mcp_server("/opt/slackgentic", home=Path(tmp))
            settings = Path(tmp) / ".claude" / "settings.local.json"
            config = json.loads(settings.read_text(encoding="utf-8"))

        self.assertEqual(
            [call[0][:3] for call in calls],
            [
                ["claude", "mcp", "add"],
                ["claude", "mcp", "remove"],
                ["claude", "mcp", "add"],
            ],
        )
        for permission in SLACKGENTIC_MCP_PERMISSION_ALLOW:
            self.assertIn(permission, config["permissions"]["allow"])

    def test_install_allows_slackgentic_mcp_tools_in_claude_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            settings = home / ".claude" / "settings.local.json"
            settings.parent.mkdir()
            settings.write_text(
                json.dumps({"permissions": {"allow": ["Bash(ls:*)"]}}),
                encoding="utf-8",
            )

            written = ensure_claude_mcp_permissions(home)

            self.assertEqual(written, settings)
            config = json.loads(settings.read_text(encoding="utf-8"))
            for permission in SLACKGENTIC_MCP_PERMISSION_ALLOW:
                self.assertIn(permission, config["permissions"]["allow"])
            self.assertIn("Bash(ls:*)", config["permissions"]["allow"])

    def test_install_registers_native_input_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            settings = home / ".claude" / "settings.local.json"
            settings.parent.mkdir()
            settings.write_text(
                json.dumps({"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": []}]}}),
                encoding="utf-8",
            )

            written = ensure_claude_native_input_hook("/opt/slackgentic", home)

            self.assertEqual(written, settings)
            config = json.loads(settings.read_text(encoding="utf-8"))
            pre_tool_use = config["hooks"]["PreToolUse"]
            self.assertEqual(pre_tool_use[0]["matcher"], "Bash")
            native = pre_tool_use[1]
            self.assertEqual(native["matcher"], "AskUserQuestion")
            self.assertEqual(
                native["hooks"][0]["command"],
                "/opt/slackgentic claude-channel --native-input-hook",
            )

    def test_native_input_hook_posts_slack_request_and_returns_updated_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                store.upsert_session(session)
                store.upsert_slack_thread_for_session(
                    Provider.CLAUDE,
                    "s1",
                    "T1",
                    SlackThreadRef("C1", "171.000001", "171.000001"),
                )
                gateway = FakeGateway()
                payload = {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "AskUserQuestion",
                    "tool_use_id": "toolu_question",
                    "session_id": "s1",
                    "tool_input": {
                        "questions": [
                            {
                                "header": "Scope",
                                "question": "How much should run?",
                                "options": [
                                    {"label": "Cheap set"},
                                    {"label": "Everything runnable"},
                                ],
                            },
                            {
                                "header": "Validation",
                                "question": "Add result validation?",
                                "options": [{"label": "No"}, {"label": "Yes"}],
                            },
                        ]
                    },
                }
                result = {}

                def run_hook():
                    result["value"] = handle_native_input_hook(
                        payload,
                        store,
                        gateway,
                        poll_seconds=0.01,
                    )

                worker = threading.Thread(target=run_hook)
                worker.start()
                self.assertTrue(_wait_for(lambda: bool(gateway.replies)))
                rows = store.list_pending_slack_agent_requests("item/tool/requestUserInput")
                self.assertEqual(len(rows), 1)
                store.resolve_slack_agent_request(
                    rows[0]["token"],
                    {
                        "answers": {
                            "q0": {"answers": ["Everything runnable"]},
                            "q1": {"answers": ["Yes"]},
                        }
                    },
                )
                worker.join(timeout=1)

                self.assertFalse(worker.is_alive())
                hook_result = result["value"]
                self.assertIsNotNone(hook_result)
                assert hook_result is not None
                self.assertEqual(
                    hook_result["hookSpecificOutput"]["permissionDecision"],
                    "allow",
                )
                updated_input = hook_result["hookSpecificOutput"]["updatedInput"]
                self.assertEqual(
                    updated_input["answers"],
                    {
                        "How much should run?": "Everything runnable",
                        "Add result validation?": "Yes",
                    },
                )
                self.assertIn("Everything runnable", hook_result["systemMessage"])
                self.assertIsNotNone(
                    store.get_setting("claude_native_input_request.claude.s1.toolu_question")
                )
            finally:
                store.close()


def _wait_for(predicate, attempts=100):
    event = threading.Event()
    for _ in range(attempts):
        if predicate():
            return True
        event.wait(0.01)
    return False


if __name__ == "__main__":
    unittest.main()
