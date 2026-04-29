import threading
import unittest

from agent_harness.models import SlackThreadRef
from agent_harness.slack import decode_action_value
from agent_harness.slack.client import PostedMessage
from agent_harness.slack.codex_requests import SlackCodexRequestHandler


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


class SlackCodexRequestHandlerTests(unittest.TestCase):
    def test_request_user_input_options_resolve_from_slack_button(self):
        gateway = FakeGateway()
        handler = SlackCodexRequestHandler(gateway, timeout_seconds=2)
        thread = SlackThreadRef("C1", "171.000001")
        result = {}

        def run_request():
            result["value"] = handler.handle_server_request(
                {
                    "id": 7,
                    "method": "item/tool/requestUserInput",
                    "params": {
                        "threadId": "codex-s1",
                        "turnId": "turn-1",
                        "itemId": "item-1",
                        "questions": [
                            {
                                "id": "provider",
                                "header": "Provider",
                                "question": "Choose an agent",
                                "options": [
                                    {"label": "Codex", "description": "Use Codex"},
                                    {"label": "Claude", "description": "Use Claude"},
                                ],
                            }
                        ],
                    },
                },
                thread,
            )

        worker = threading.Thread(target=run_request)
        worker.start()
        self.assertTrue(_wait_for(lambda: bool(gateway.replies)))
        _assert_unique_action_ids(self, gateway.replies[0]["blocks"])

        value = gateway.replies[0]["blocks"][2]["elements"][0]["value"]
        handled = handler.handle_block_action(
            decode_action_value(value),
            "C1",
            gateway.replies[0]["ts"],
        )

        worker.join(timeout=1)
        self.assertTrue(handled)
        self.assertEqual(
            result["value"],
            {"answers": {"provider": {"answers": ["Codex"]}}},
        )
        self.assertEqual(gateway.updates[-1]["text"], "Answered Codex input request.")

    def test_permissions_approval_can_be_granted_for_session(self):
        gateway = FakeGateway()
        handler = SlackCodexRequestHandler(gateway, timeout_seconds=2)
        thread = SlackThreadRef("C1", "171.000001")
        result = {}

        def run_request():
            result["value"] = handler.handle_server_request(
                {
                    "id": 8,
                    "method": "item/permissions/requestApproval",
                    "params": {
                        "threadId": "codex-s1",
                        "turnId": "turn-1",
                        "itemId": "item-1",
                        "cwd": "/repo",
                        "permissions": {"network": {"enabled": True}},
                    },
                },
                thread,
            )

        worker = threading.Thread(target=run_request)
        worker.start()
        self.assertTrue(_wait_for(lambda: bool(gateway.replies)))
        _assert_unique_action_ids(self, gateway.replies[0]["blocks"])

        value = gateway.replies[0]["blocks"][1]["elements"][1]["value"]
        handler.handle_block_action(decode_action_value(value), "C1", gateway.replies[0]["ts"])

        worker.join(timeout=1)
        self.assertEqual(
            result["value"],
            {"permissions": {"network": {"enabled": True}}, "scope": "session"},
        )
        self.assertEqual(
            gateway.updates[-1]["text"],
            "Approved Codex permissions for this session.",
        )

    def test_command_approval_maps_to_app_server_decision(self):
        gateway = FakeGateway()
        handler = SlackCodexRequestHandler(gateway, timeout_seconds=2)
        thread = SlackThreadRef("C1", "171.000001")
        result = {}

        def run_request():
            result["value"] = handler.handle_server_request(
                {
                    "id": 9,
                    "method": "item/commandExecution/requestApproval",
                    "params": {
                        "threadId": "codex-s1",
                        "turnId": "turn-1",
                        "itemId": "item-1",
                        "command": "make test",
                        "cwd": "/repo",
                    },
                },
                thread,
            )

        worker = threading.Thread(target=run_request)
        worker.start()
        self.assertTrue(_wait_for(lambda: bool(gateway.replies)))
        _assert_unique_action_ids(self, gateway.replies[0]["blocks"])

        value = gateway.replies[0]["blocks"][1]["elements"][0]["value"]
        handler.handle_block_action(decode_action_value(value), "C1", gateway.replies[0]["ts"])

        worker.join(timeout=1)
        self.assertEqual(result["value"], {"decision": "accept"})
        self.assertEqual(gateway.updates[-1]["text"], "Approved Codex request.")


def _wait_for(predicate, attempts=50):
    event = threading.Event()
    for _ in range(attempts):
        if predicate():
            return True
        event.wait(0.01)
    return False


def _assert_unique_action_ids(test_case, blocks):
    for block in blocks:
        action_ids = [
            element.get("action_id")
            for element in block.get("elements") or []
            if element.get("action_id")
        ]
        test_case.assertEqual(len(action_ids), len(set(action_ids)))


if __name__ == "__main__":
    unittest.main()
