import tempfile
import threading
import unittest
from pathlib import Path

from agent_harness.models import SlackThreadRef
from agent_harness.slack import decode_action_value
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


def _wait_for(predicate, attempts=50):
    event = threading.Event()
    for _ in range(attempts):
        if predicate():
            return True
        event.wait(0.01)
    return False


if __name__ == "__main__":
    unittest.main()
