import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from agent_harness.agent_skills import (
    bundled_skill_names,
    install_agent_skills,
    skills_dir_for_provider,
)
from agent_harness.cli import main
from agent_harness.loops import (
    AGENT_LOOP_REQUEST_MAX_PENDING,
    describe_agent_loop_create_request,
    enqueue_agent_loop_create_request,
    prepare_agent_loop_create_command,
    wait_for_agent_loop_create_request,
)
from agent_harness.models import (
    LOOP_RESOLUTION_METADATA_KEY,
    LoopCreateRequestStatus,
    LoopStatus,
    LoopVisibility,
    Provider,
)
from agent_harness.sessions.claude_channel import (
    CHANNEL_INSTRUCTIONS,
    CODEX_MCP_INSTRUCTIONS,
    SLACKGENTIC_MCP_PERMISSION_ALLOW,
    ClaudeChannelServer,
)
from agent_harness.slack.app import SETTING_HUMAN_USER_ID, LoopRunner, SlackTeamController
from agent_harness.storage.store import Store
from tests.test_slack_app import FakeGateway, FakeRuntime


class PrepareAgentLoopCreateCommandTests(unittest.TestCase):
    def test_plain_request_becomes_loop_create_command(self):
        result = prepare_agent_loop_create_command(
            "  every weekday at 9am America/New_York   check CI  "
        )

        self.assertIsNone(result.error)
        self.assertEqual(
            result.command, "loop create every weekday at 9am America/New_York check CI"
        )

    def test_existing_create_verb_is_kept(self):
        result = prepare_agent_loop_create_command("create a loop to check CI every hour")

        self.assertEqual(result.command, "create a loop to check CI every hour")

    def test_provider_and_visibility_become_command_options(self):
        result = prepare_agent_loop_create_command(
            "check CI every hour", provider="Claude", visibility=LoopVisibility.PUBLIC
        )

        self.assertEqual(result.command, "loop create check CI every hour provider=claude #public")

    def test_rejects_missing_description_and_bad_options(self):
        self.assertIsNotNone(prepare_agent_loop_create_command("   ").error)
        self.assertIsNotNone(prepare_agent_loop_create_command("loop create").error)
        self.assertIn(
            "codex or claude",
            prepare_agent_loop_create_command("check CI hourly", provider="other").error or "",
        )
        self.assertIn(
            "private or public",
            prepare_agent_loop_create_command("check CI hourly", visibility="team").error or "",
        )

    def test_rejects_dangerous_mode(self):
        result = prepare_agent_loop_create_command("restart the service hourly #dangerous-mode")

        self.assertIsNone(result.command)
        self.assertIn("read-only", result.error or "")


class LoopCreateQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = Store(self.root / "state.sqlite")
        self.store.init_schema()

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_enqueue_claim_and_finish_round_trip(self):
        result = enqueue_agent_loop_create_request(
            self.store, "check CI every hour", provider=Provider.CODEX, source="cli"
        )
        assert result.request is not None
        self.assertEqual(result.request.status, LoopCreateRequestStatus.PENDING)
        self.assertEqual(result.request.text, "loop create check CI every hour provider=codex")

        claimed = self.store.claim_pending_loop_create_requests()
        self.assertEqual([item.request_id for item in claimed], [result.request.request_id])
        self.assertEqual(claimed[0].status, LoopCreateRequestStatus.CLAIMED)
        self.assertEqual(self.store.claim_pending_loop_create_requests(), [])

        self.store.finish_loop_create_request(
            result.request.request_id,
            LoopCreateRequestStatus.POSTED,
            channel_id="CMAIN",
            message_ts="100.000001",
        )
        finished = self.store.get_loop_create_request(result.request.request_id)
        assert finished is not None
        self.assertEqual(finished.status, LoopCreateRequestStatus.POSTED)
        self.assertEqual(finished.message_ts, "100.000001")
        self.assertIn("click Create", describe_agent_loop_create_request(finished))

    def test_enqueue_refuses_when_too_many_requests_are_waiting(self):
        for index in range(AGENT_LOOP_REQUEST_MAX_PENDING):
            self.assertIsNotNone(
                enqueue_agent_loop_create_request(self.store, f"check job {index} hourly").request
            )

        result = enqueue_agent_loop_create_request(self.store, "check one more job hourly")

        self.assertIsNone(result.request)
        self.assertIn("service status", result.error or "")

    def test_wait_returns_pending_request_after_timeout(self):
        request = self.store.enqueue_loop_create_request("loop create check CI hourly")
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        waited = wait_for_agent_loop_create_request(
            self.store,
            request.request_id,
            timeout_seconds=3,
            sleep=sleep,
            monotonic=lambda: clock[0],
        )

        assert waited is not None
        self.assertEqual(waited.status, LoopCreateRequestStatus.PENDING)
        self.assertGreaterEqual(clock[0], 3)
        self.assertIn("not picked it up", describe_agent_loop_create_request(waited))


class QueuedLoopRequestControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = Store(root / "state.sqlite")
        self.store.init_schema()
        self.gateway = FakeGateway()
        self.runtime = FakeRuntime()
        self.controller = SlackTeamController(
            self.store,
            self.gateway,
            default_channel_id="CMAIN",
            runtime=self.runtime,
            home=root,
            default_cwd=root,
            ignored_bot_id="BOWN",
        )

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_loop_runner_posts_queued_request_and_starts_resolver_for_owner(self):
        self.store.set_setting(SETTING_HUMAN_USER_ID, "UOWNER")
        request = enqueue_agent_loop_create_request(
            self.store, "inspect cloud billing every morning at 9am America/Chicago"
        ).request
        assert request is not None

        LoopRunner(self.store, self.controller).sync_once()

        request_post = self.gateway.posts[0]
        self.assertEqual(request_post["channel_id"], "CMAIN")
        self.assertIsNone(request_post["thread_ts"])
        self.assertIn("inspect cloud billing", request_post["text"])
        loops = self.store.list_loops()
        self.assertEqual(len(loops), 1)
        self.assertEqual(loops[0].status, LoopStatus.RESOLVING)
        self.assertEqual(loops[0].owner_slack_user_id, "UOWNER")
        self.assertEqual(loops[0].anchor_thread_ts, request_post["ts"])
        tasks = self.store.list_agent_tasks(include_done=True)
        self.assertTrue(tasks[0].metadata.get(LOOP_RESOLUTION_METADATA_KEY))
        finished = self.store.get_loop_create_request(request.request_id)
        assert finished is not None
        self.assertEqual(finished.status, LoopCreateRequestStatus.POSTED)
        self.assertEqual(finished.message_ts, request_post["ts"])

    def test_queued_request_fails_before_setup_knows_the_owner(self):
        request = enqueue_agent_loop_create_request(self.store, "check CI hourly").request
        assert request is not None

        self.assertEqual(self.controller.process_queued_loop_create_requests(), 1)

        finished = self.store.get_loop_create_request(request.request_id)
        assert finished is not None
        self.assertEqual(finished.status, LoopCreateRequestStatus.FAILED)
        self.assertIn("setup", finished.error or "")
        self.assertEqual(self.gateway.posts, [])
        self.assertEqual(self.store.list_loops(), [])


class CreateLoopMcpToolTests(unittest.TestCase):
    def test_create_loop_tool_is_listed_allowlisted_and_queues_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            output = io.StringIO()
            try:
                store.init_schema()
                server = ClaudeChannelServer(store, target_pid=123, loop_request_wait_seconds=0)

                with redirect_stdout(output):
                    server._handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
                tools = {
                    tool["name"]: tool for tool in json.loads(output.getvalue())["result"]["tools"]
                }
                result = server._handle_tool_call(
                    {
                        "name": "create_loop",
                        "arguments": {"request": "check CI every hour", "visibility": "public"},
                    }
                )

                self.assertIn("create_loop", tools)
                self.assertEqual(tools["create_loop"]["inputSchema"]["required"], ["request"])
                self.assertIn("mcp__slackgentic__create_loop", SLACKGENTIC_MCP_PERMISSION_ALLOW)
                self.assertIn("create_loop", CHANNEL_INSTRUCTIONS)
                self.assertIn("create_loop", CODEX_MCP_INSTRUCTIONS)
                self.assertNotIn("isError", result)
                self.assertIn("queued", result["content"][0]["text"])
                queued = store.claim_pending_loop_create_requests()
                self.assertEqual(
                    [item.text for item in queued], ["loop create check CI every hour #public"]
                )
            finally:
                store.close()

    def test_create_loop_tool_reports_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                server = ClaudeChannelServer(store, target_pid=123, loop_request_wait_seconds=0)

                result = server._handle_tool_call({"name": "create_loop", "arguments": {}})

                self.assertTrue(result["isError"])
                self.assertEqual(store.claim_pending_loop_create_requests(), [])
            finally:
                store.close()


class LoopCliTests(unittest.TestCase):
    def test_loop_create_queues_request_and_list_reports_no_loops(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "loop",
                        "create",
                        "check",
                        "CI every hour",
                        "--provider",
                        "codex",
                        "--no-wait",
                        "--json",
                        "--db",
                        str(db),
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "pending")
            self.assertEqual(payload["text"], "loop create check CI every hour provider=codex")

            status_output = io.StringIO()
            with redirect_stdout(status_output):
                code = main(["loop", "request-status", payload["request_id"], "--db", str(db)])
            self.assertEqual(code, 0)
            self.assertIn("not picked it up", status_output.getvalue())

            list_output = io.StringIO()
            with redirect_stdout(list_output):
                code = main(["loop", "list", "--db", str(db)])
            self.assertEqual(code, 0)
            self.assertIn("No loops", list_output.getvalue())

    def test_loop_create_rejects_dangerous_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            errors = io.StringIO()
            with redirect_stderr(errors):
                code = main(
                    ["loop", "create", "restart prod hourly #dangerous-mode", "--db", str(db)]
                )
            self.assertEqual(code, 2)
            self.assertIn("read-only", errors.getvalue())


class AgentSkillsTests(unittest.TestCase):
    def test_bundled_skills_have_matching_frontmatter(self):
        names = bundled_skill_names()

        self.assertIn("slackgentic", names)
        self.assertIn("slackgentic-loops", names)
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            install_agent_skills(providers=("claude",), home=home)
            for name in names:
                text = (home / ".claude" / "skills" / name / "SKILL.md").read_text()
                self.assertTrue(text.startswith("---\n"))
                frontmatter = text.split("---\n")[1]
                self.assertIn(f"name: {name}\n", frontmatter)
                self.assertIn("description: ", frontmatter)

    def test_install_writes_both_providers_and_replaces_stale_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            stale = home / ".codex" / "skills" / "slackgentic" / "references" / "old.md"
            stale.parent.mkdir(parents=True)
            stale.write_text("stale")

            installed = install_agent_skills(home=home)

            self.assertFalse(stale.exists())
            for provider in ("claude", "codex"):
                root = skills_dir_for_provider(provider, home)
                self.assertTrue((root / "slackgentic-loops" / "SKILL.md").is_file())
                self.assertTrue((root / "slackgentic" / "references" / "task-signals.md").is_file())
            self.assertEqual(len(installed), 2 * len(bundled_skill_names()))

    def test_only_existing_skips_providers_without_a_config_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".claude").mkdir()

            installed = install_agent_skills(home=home, only_existing=True)

            self.assertTrue(installed)
            self.assertTrue(all(".claude" in str(path) for path in installed))
            self.assertFalse((home / ".codex").exists())


if __name__ == "__main__":
    unittest.main()
