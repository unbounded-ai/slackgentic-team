import json
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path

from agent_harness.models import AgentTask, AgentTaskKind, AgentTaskStatus, SlackThreadRef
from agent_harness.slack import decode_action_value
from agent_harness.slack.agent_requests import SlackAgentRequestHandler
from agent_harness.slack.client import PostedMessage
from agent_harness.storage.store import Store
from tests.polling import POLL_TIMEOUT_SECONDS, wait_until


class FakeGateway:
    def __init__(self):
        self.replies = []
        self.updates = []
        self.ephemerals = []

    def post_thread_reply(self, thread, text, persona=None, icon_url=None, blocks=None):
        ts = f"1712345678.{len(self.replies):06d}"
        self.replies.append({"thread": thread, "text": text, "blocks": blocks, "ts": ts})
        return PostedMessage(thread.channel_id, ts, thread.thread_ts)

    def update_message(self, channel_id, ts, text, blocks=None):
        self.updates.append({"channel_id": channel_id, "ts": ts, "text": text, "blocks": blocks})

    def post_ephemeral(self, channel_id, user_id, text):
        self.ephemerals.append((channel_id, user_id, text))
        return True


class FailingFullRequestGateway(FakeGateway):
    def __init__(self):
        super().__init__()
        self.failures = 0

    def post_thread_reply(self, thread, text, persona=None, icon_url=None, blocks=None):
        if self.failures == 0:
            self.failures += 1
            raise RuntimeError("invalid_blocks")
        return super().post_thread_reply(
            thread,
            text,
            persona=persona,
            icon_url=icon_url,
            blocks=blocks,
        )


def _choice_params() -> dict:
    return {
        "questions": [
            {
                "id": "choice",
                "header": "Choice",
                "question": "Pick one",
                "options": [{"label": "A"}, {"label": "B"}],
            }
        ]
    }


def _terminal_button(blocks: list[dict]) -> dict | None:
    for block in blocks:
        if block.get("type") != "actions":
            continue
        for element in block.get("elements", []):
            if decode_action_value(element["value"]).get("decision") == "terminal":
                return element
    return None


class SlackAgentRequestHandlerTests(unittest.TestCase):
    def test_input_request_tags_the_thread_requester(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                store.set_setting("slack.human_user_id", "U_HUMAN")
                now = datetime.now(UTC)
                store.upsert_agent_task(
                    AgentTask(
                        task_id="task-1",
                        agent_id="agent-1",
                        prompt="do the thing",
                        channel_id="C1",
                        kind=AgentTaskKind.WORK,
                        status=AgentTaskStatus.ACTIVE,
                        created_at=now,
                        updated_at=now,
                        requested_by_slack_user="U_REQUESTER",
                        thread_ts="171.000001",
                    )
                )
                handler = SlackAgentRequestHandler(gateway, store=store, provider_label="Claude")

                handler.create_persistent_request(
                    "item/tool/requestUserInput",
                    _choice_params(),
                    SlackThreadRef("C1", "171.000001"),
                )

                # The requester outranks the configured human.
                self.assertEqual(gateway.replies[0]["text"], "<@U_REQUESTER> Claude needs input.")
                self.assertTrue(
                    gateway.replies[0]["blocks"][0]["text"]["text"].startswith(
                        "<@U_REQUESTER> *Claude needs input.*"
                    )
                )
            finally:
                store.close()

    def test_input_request_tags_the_configured_human_for_observed_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                handler = SlackAgentRequestHandler(gateway, store=store, provider_label="Claude")
                thread = SlackThreadRef("C1", "171.000001")

                handler.create_persistent_request(
                    "item/tool/requestUserInput", _choice_params(), thread
                )
                self.assertEqual(gateway.replies[0]["text"], "Claude needs input.")

                store.set_setting("slack.human_user_id", "U_HUMAN")
                handler.create_persistent_request(
                    "item/tool/requestUserInput", _choice_params(), thread
                )
                self.assertEqual(gateway.replies[1]["text"], "<@U_HUMAN> Claude needs input.")
            finally:
                store.close()

    def test_terminal_handoff_button_moves_the_question_to_the_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                handler = SlackAgentRequestHandler(gateway, store=store, provider_label="Claude")
                thread = SlackThreadRef("C1", "171.000001")

                plain = handler.create_persistent_request(
                    "item/tool/requestUserInput", _choice_params(), thread
                )
                plain_blocks = gateway.replies[0]["blocks"]
                self.assertIsNone(_terminal_button(plain_blocks))
                self.assertEqual(plain_blocks[-1]["type"], "actions")

                params = {**_choice_params(), "terminal_handoff": True, "footer": "Press Esc."}
                pending = handler.create_persistent_request(
                    "item/tool/requestUserInput", params, thread
                )
                blocks = gateway.replies[1]["blocks"]
                self.assertEqual(blocks[-1]["type"], "context")
                self.assertEqual(blocks[-1]["elements"][0]["text"], "Press Esc.")
                button = _terminal_button(blocks)
                self.assertIsNotNone(button)
                assert button is not None
                self.assertEqual(button["text"]["text"], "Answer in the terminal")

                handled = handler.handle_block_action(
                    decode_action_value(button["value"]), "C1", gateway.replies[1]["ts"]
                )

                self.assertTrue(handled)
                self.assertEqual(
                    store.get_slack_agent_request_response(pending.token),
                    (True, {"answers": {}, "handoff": "terminal"}),
                )
                self.assertEqual(gateway.updates[-1]["text"], "Moved to the Claude terminal.")
                self.assertEqual(
                    handler.wait_for_persistent_request(pending.token),
                    {"answers": {}, "handoff": "terminal"},
                )
                resolved, _ = store.get_slack_agent_request_response(plain.token)
                self.assertFalse(resolved)
            finally:
                store.close()

    def test_wait_for_persistent_request_returns_when_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                handler = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=POLL_TIMEOUT_SECONDS,
                    store=store,
                    provider_label="Claude",
                    poll_seconds=0.01,
                )
                pending = handler.create_persistent_request(
                    "item/tool/requestUserInput",
                    _choice_params(),
                    SlackThreadRef("C1", "171.000001"),
                )
                stop = threading.Event()
                stop.set()

                self.assertIsNone(handler.wait_for_persistent_request(pending.token, stop=stop))

                # Stopping leaves the request open; the caller decides its fate.
                resolved, _ = store.get_slack_agent_request_response(pending.token)
                self.assertFalse(resolved)
                self.assertEqual(gateway.updates, [])
            finally:
                store.close()

    def test_abandon_persistent_request_closes_an_unanswered_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                handler = SlackAgentRequestHandler(gateway, store=store, provider_label="Claude")
                pending = handler.create_persistent_request(
                    "item/tool/requestUserInput",
                    _choice_params(),
                    SlackThreadRef("C1", "171.000001"),
                )

                self.assertTrue(handler.abandon_persistent_request(pending.token, "Moved on."))

                resolved, response = store.get_slack_agent_request_response(pending.token)
                self.assertTrue(resolved)
                self.assertEqual(response, {"answers": {}})
                self.assertEqual(gateway.updates[-1]["text"], "Moved on.")
                self.assertIsNone(gateway.updates[-1]["blocks"])

                self.assertFalse(handler.abandon_persistent_request(pending.token, "Again."))
                self.assertEqual(len(gateway.updates), 1)
            finally:
                store.close()

    def test_abandon_persistent_request_keeps_an_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                handler = SlackAgentRequestHandler(gateway, store=store, provider_label="Claude")
                pending = handler.create_persistent_request(
                    "item/tool/requestUserInput",
                    _choice_params(),
                    SlackThreadRef("C1", "171.000001"),
                )
                answer = {"answers": {"choice": {"answers": ["A"]}}}
                store.resolve_slack_agent_request(pending.token, answer)

                self.assertFalse(handler.abandon_persistent_request(pending.token, "Moved on."))

                self.assertEqual(
                    store.get_slack_agent_request_response(pending.token), (True, answer)
                )
                self.assertEqual(gateway.updates, [])
            finally:
                store.close()

    def test_persistent_request_can_be_resolved_by_another_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            result = {}
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=POLL_TIMEOUT_SECONDS,
                    store=store,
                    provider_label="Claude",
                )
                responder = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=POLL_TIMEOUT_SECONDS,
                    store=store,
                    provider_label="Claude",
                )
                thread = SlackThreadRef("C1", "171.000001")

                def run_request():
                    result["value"] = requester.handle_persistent_request(
                        "item/tool/requestUserInput",
                        {
                            "questions": [
                                {
                                    "id": "choice",
                                    "header": "Choice",
                                    "question": "Pick one",
                                    "options": [{"label": "A"}, {"label": "B"}],
                                }
                            ]
                        },
                        thread,
                    )

                worker = threading.Thread(target=run_request)
                worker.start()
                self.assertTrue(wait_until(lambda: bool(gateway.replies)))

                value = gateway.replies[0]["blocks"][2]["elements"][1]["value"]
                handled = responder.handle_block_action(
                    decode_action_value(value),
                    "C1",
                    gateway.replies[0]["ts"],
                )

                worker.join(timeout=POLL_TIMEOUT_SECONDS)
                self.assertTrue(handled)
                self.assertEqual(
                    result["value"],
                    {"answers": {"choice": {"answers": ["B"]}}},
                )
                self.assertEqual(gateway.updates[-1]["text"], "Answered Claude input request.")
            finally:
                store.close()

    def test_persistent_command_approval_can_be_resolved_by_another_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            result = {}
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=POLL_TIMEOUT_SECONDS,
                    store=store,
                    provider_label="Claude",
                )
                responder = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=POLL_TIMEOUT_SECONDS,
                    store=store,
                    provider_label="Claude",
                )
                thread = SlackThreadRef("C1", "171.000001")

                def run_request():
                    result["value"] = requester.handle_persistent_request(
                        "item/commandExecution/requestApproval",
                        {"command": ["rm", "-rf", "build"], "reason": "cleanup"},
                        thread,
                    )

                worker = threading.Thread(target=run_request)
                worker.start()
                self.assertTrue(wait_until(lambda: bool(gateway.replies)))

                value = _first_actions_block(gateway.replies[0]["blocks"])["elements"][0]["value"]
                handled = responder.handle_block_action(
                    decode_action_value(value),
                    "C1",
                    gateway.replies[0]["ts"],
                )

                worker.join(timeout=POLL_TIMEOUT_SECONDS)
                self.assertTrue(handled)
                self.assertEqual(result["value"], {"decision": "accept"})
                self.assertEqual(gateway.updates[-1]["text"], "Approved Claude request.")
            finally:
                store.close()

    def test_persistent_claude_permission_can_be_resolved_by_another_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            result = {}
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=POLL_TIMEOUT_SECONDS,
                    store=store,
                    provider_label="Claude",
                )
                responder = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=POLL_TIMEOUT_SECONDS,
                    store=store,
                    provider_label="Claude",
                )
                thread = SlackThreadRef("C1", "171.000001")

                def run_request():
                    result["value"] = requester.handle_persistent_request(
                        "claude/channel/permission",
                        {
                            "request_id": "req-1",
                            "tool_name": "Bash",
                            "description": "List files",
                            "input_preview": "ls ~/code",
                        },
                        thread,
                    )

                worker = threading.Thread(target=run_request)
                worker.start()
                self.assertTrue(wait_until(lambda: bool(gateway.replies)))

                value = _first_actions_block(gateway.replies[0]["blocks"])["elements"][0]["value"]
                handled = responder.handle_block_action(
                    decode_action_value(value),
                    "C1",
                    gateway.replies[0]["ts"],
                )

                worker.join(timeout=POLL_TIMEOUT_SECONDS)
                self.assertTrue(handled)
                self.assertEqual(result["value"], {"behavior": "allow"})
                self.assertIn("Claude requests tool approval", gateway.replies[0]["text"])
                self.assertIn("Input preview", gateway.replies[0]["blocks"][1]["text"]["text"])
                self.assertIn("```ls ~/code```", gateway.replies[0]["blocks"][1]["text"]["text"])
                self.assertEqual(gateway.updates[-1]["text"], "Allowed Claude tool request.")
            finally:
                store.close()

    def test_persistent_claude_permission_retries_compact_message_when_full_blocks_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FailingFullRequestGateway()
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=POLL_TIMEOUT_SECONDS,
                    store=store,
                    provider_label="Claude",
                )
                thread = SlackThreadRef("C1", "171.000001")

                pending = requester.create_persistent_request(
                    "claude/channel/permission",
                    {
                        "request_id": "req-1",
                        "tool_name": "Bash",
                        "description": "List files",
                        "input_preview": "ls ~/code",
                    },
                    thread,
                )

                self.assertEqual(gateway.failures, 1)
                self.assertEqual(len(gateway.replies), 1)
                self.assertEqual(pending.message_ts, gateway.replies[0]["ts"])
                row = store.get_slack_agent_request(pending.token)
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(row["message_ts"], gateway.replies[0]["ts"])
                actions = _first_actions_block(gateway.replies[0]["blocks"])["elements"]
                self.assertEqual(
                    [action["text"]["text"] for action in actions],
                    ["Allow", "Allow Session", "Deny"],
                )
            finally:
                store.close()

    def test_persistent_claude_permission_waiters_share_store_connection_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            start = threading.Event()
            results = []
            errors = []
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=POLL_TIMEOUT_SECONDS,
                    store=store,
                    provider_label="Claude",
                )
                pending = requester.create_persistent_request(
                    "claude/channel/permission",
                    {
                        "request_id": "req-1",
                        "tool_name": "Bash",
                        "description": "List files",
                        "input_preview": "ls ~/code",
                    },
                    SlackThreadRef("C1", "171.000001"),
                )

                def wait_for_request():
                    start.wait()
                    try:
                        results.append(
                            requester.wait_for_persistent_request(
                                pending.token,
                                timeout_seconds=POLL_TIMEOUT_SECONDS,
                            )
                        )
                    except Exception as exc:  # pragma: no cover - asserted through errors
                        errors.append(exc)

                workers = [threading.Thread(target=wait_for_request) for _ in range(8)]
                for worker in workers:
                    worker.start()
                start.set()
                store.resolve_slack_agent_request(pending.token, {"behavior": "allow"})
                for worker in workers:
                    worker.join(timeout=POLL_TIMEOUT_SECONDS)

                self.assertEqual(errors, [])
                self.assertEqual(results, [{"behavior": "allow"}] * 8)
            finally:
                store.close()

    def test_claude_permission_can_be_allowed_for_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=POLL_TIMEOUT_SECONDS,
                    store=store,
                    provider_label="Claude",
                )
                thread = SlackThreadRef("C1", "171.000001")
                pending = requester.create_persistent_request(
                    "claude/channel/permission",
                    {
                        "request_id": "req-1",
                        "tool_name": "Bash",
                        "description": "List files",
                        "input_preview": "ls ~/code",
                        "can_allow_session": True,
                    },
                    thread,
                )

                actions = _first_actions_block(gateway.replies[0]["blocks"])["elements"]
                self.assertEqual(
                    [action["text"]["text"] for action in actions],
                    ["Allow", "Allow Session", "Deny"],
                )

                handled = requester.handle_block_action(
                    decode_action_value(actions[1]["value"]),
                    "C1",
                    gateway.replies[0]["ts"],
                )

                resolved, response = store.get_slack_agent_request_response(pending.token)
                self.assertTrue(handled)
                self.assertTrue(resolved)
                self.assertEqual(response, {"behavior": "allow", "scope": "session"})
                self.assertEqual(
                    gateway.updates[-1]["text"],
                    "Allowed Claude tool request for this session.",
                )
            finally:
                store.close()

    def test_claude_bash_permission_displays_shell_command_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=POLL_TIMEOUT_SECONDS,
                    store=store,
                    provider_label="Claude",
                )
                thread = SlackThreadRef("C1", "171.000001")
                pending = requester.create_persistent_request(
                    "claude/channel/permission",
                    {
                        "request_id": "req-1",
                        "tool_name": "Bash",
                        "description": "Check worktree",
                        "input_preview": json.dumps(
                            {
                                "command": "git -C /workspace/repos/sample-app status",
                                "description": "Check worktree",
                            }
                        ),
                        "can_allow_session": True,
                    },
                    thread,
                )

                self.assertEqual(
                    gateway.replies[0]["text"], "Claude requests command approval: git"
                )
                self.assertIn(
                    "Command: `git`",
                    gateway.replies[0]["blocks"][0]["text"]["text"],
                )
                self.assertNotIn(
                    "Tool: `Bash`",
                    gateway.replies[0]["blocks"][0]["text"]["text"],
                )

                actions = _first_actions_block(gateway.replies[0]["blocks"])["elements"]
                handled = requester.handle_block_action(
                    decode_action_value(actions[1]["value"]),
                    "C1",
                    gateway.replies[0]["ts"],
                )

                resolved, response = store.get_slack_agent_request_response(pending.token)
                self.assertTrue(handled)
                self.assertTrue(resolved)
                self.assertEqual(response, {"behavior": "allow", "scope": "session"})
                self.assertEqual(
                    gateway.updates[-1]["text"],
                    "Allowed Claude command request for this session.",
                )
            finally:
                store.close()

    def test_claude_edit_permission_can_be_allowed_for_session_without_runtime_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=0.01,
                    store=store,
                    provider_label="Claude",
                )

                requester.create_persistent_request(
                    "claude/channel/permission",
                    {
                        "request_id": "req-1",
                        "tool_name": "Edit",
                        "description": "A tool for editing files",
                        "input_preview": (
                            '{"file_path":"/tmp/README.md","old_string":"before",'
                            '"new_string":"after"}'
                        ),
                    },
                    SlackThreadRef("C1", "171.000001"),
                )

                actions = _first_actions_block(gateway.replies[0]["blocks"])["elements"]
                self.assertEqual(
                    [action["text"]["text"] for action in actions],
                    ["Allow", "Allow Session", "Deny"],
                )
            finally:
                store.close()

    def test_claude_edit_permission_preview_is_shown_as_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=0.01,
                    store=store,
                    provider_label="Claude",
                )

                requester.handle_persistent_request(
                    "claude/channel/permission",
                    {
                        "request_id": "req-1",
                        "tool_name": "Edit",
                        "description": "A tool for editing files",
                        "input_preview": (
                            '{"file_path":"/tmp/README.md","old_string":"before",'
                            '"new_string":"after"}'
                        ),
                    },
                    SlackThreadRef("C1", "171.000001"),
                )

                blocks = gateway.replies[0]["blocks"]
                self.assertEqual(blocks[0]["type"], "section")
                self.assertEqual(blocks[1]["type"], "section")
                preview = blocks[1]["text"]["text"]
                self.assertIn("*Proposed diff*", preview)
                self.assertIn("```diff", preview)
                self.assertIn("--- /tmp/README.md (current)", preview)
                self.assertIn("+++ /tmp/README.md (proposed)", preview)
                self.assertIn("-before", preview)
                self.assertIn("+after", preview)
                self.assertNotIn("Input: `", blocks[0]["text"]["text"])
            finally:
                store.close()

    def test_claude_edit_permission_full_input_is_shown_as_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=0.01,
                    store=store,
                    provider_label="Claude",
                )

                requester.handle_persistent_request(
                    "claude/channel/permission",
                    {
                        "request_id": "req-1",
                        "tool_name": "Edit",
                        "description": "A tool for editing files",
                        "input": {
                            "file_path": "/tmp/docs/e2e-slack.md",
                            "old_string": (
                                "threads remain visible and follow-up replies resume "
                                "the persisted task/session\nstate, but Slackgentic "
                                "relaunches the provider process."
                            ),
                            "new_string": (
                                "threads remain visible; follow-up replies resume "
                                "the persisted task/session\nstate, while Slackgentic "
                                "relaunches the provider process."
                            ),
                        },
                    },
                    SlackThreadRef("C1", "171.000001"),
                )

                preview = gateway.replies[0]["blocks"][1]["text"]["text"]
                self.assertIn("*Proposed diff*", preview)
                self.assertIn("-threads remain visible and follow-up replies", preview)
                self.assertIn("+threads remain visible; follow-up replies", preview)
                self.assertIn("-state, but Slackgentic", preview)
                self.assertIn("+state, while Slackgentic", preview)
            finally:
                store.close()

    def test_claude_edit_permission_truncated_preview_shows_unavailable_diff_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=0.01,
                    store=store,
                    provider_label="Claude",
                )

                requester.handle_persistent_request(
                    "claude/channel/permission",
                    {
                        "request_id": "req-1",
                        "tool_name": "Edit",
                        "description": "A tool for editing files",
                        "input_preview": '{"file_path":"/tmp/README.md","old_string":"partial…',
                    },
                    SlackThreadRef("C1", "171.000001"),
                )

                context_blocks = [
                    block for block in gateway.replies[0]["blocks"] if block["type"] == "context"
                ]
                self.assertEqual(len(context_blocks), 1)
                self.assertIn(
                    "Diff unavailable in Slack",
                    gateway.replies[0]["blocks"][1]["text"]["text"],
                )
                self.assertIn(
                    "File: `/tmp/README.md`",
                    gateway.replies[0]["blocks"][1]["text"]["text"],
                )
                self.assertIn(
                    "Restart this Claude session",
                    context_blocks[0]["elements"][0]["text"],
                )
                self.assertNotIn("Input preview", gateway.replies[0]["blocks"][1]["text"]["text"])
            finally:
                store.close()


def _first_actions_block(blocks):
    return next(block for block in blocks if block.get("type") == "actions")


if __name__ == "__main__":
    unittest.main()
