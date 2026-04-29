import tempfile
import threading
import unittest
from pathlib import Path

from agent_harness.models import SlackThreadRef
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


class SlackAgentRequestHandlerTests(unittest.TestCase):
    def test_persistent_request_can_be_resolved_by_another_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            result = {}
            try:
                store.init_schema()
                requester = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=2,
                    store=store,
                    provider_label="Claude",
                )
                responder = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=2,
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
                self.assertTrue(_wait_for(lambda: bool(gateway.replies)))

                value = gateway.replies[0]["blocks"][2]["elements"][1]["value"]
                handled = responder.handle_block_action(
                    decode_action_value(value),
                    "C1",
                    gateway.replies[0]["ts"],
                )

                worker.join(timeout=1)
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
                    timeout_seconds=2,
                    store=store,
                    provider_label="Claude",
                )
                responder = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=2,
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
                self.assertTrue(_wait_for(lambda: bool(gateway.replies)))

                value = _first_actions_block(gateway.replies[0]["blocks"])["elements"][0]["value"]
                handled = responder.handle_block_action(
                    decode_action_value(value),
                    "C1",
                    gateway.replies[0]["ts"],
                )

                worker.join(timeout=1)
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
                    timeout_seconds=2,
                    store=store,
                    provider_label="Claude",
                )
                responder = SlackAgentRequestHandler(
                    gateway,
                    timeout_seconds=2,
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
                self.assertTrue(_wait_for(lambda: bool(gateway.replies)))

                value = _first_actions_block(gateway.replies[0]["blocks"])["elements"][0]["value"]
                handled = responder.handle_block_action(
                    decode_action_value(value),
                    "C1",
                    gateway.replies[0]["ts"],
                )

                worker.join(timeout=1)
                self.assertTrue(handled)
                self.assertEqual(result["value"], {"behavior": "allow"})
                self.assertIn("Claude requests tool approval", gateway.replies[0]["text"])
                self.assertIn("Input preview", gateway.replies[0]["blocks"][1]["text"]["text"])
                self.assertIn("```ls ~/code```", gateway.replies[0]["blocks"][1]["text"]["text"])
                self.assertEqual(gateway.updates[-1]["text"], "Allowed Claude tool request.")
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


def _wait_for(predicate, attempts=50):
    event = threading.Event()
    for _ in range(attempts):
        if predicate():
            return True
        event.wait(0.01)
    return False


def _first_actions_block(blocks):
    return next(block for block in blocks if block.get("type") == "actions")


if __name__ == "__main__":
    unittest.main()
