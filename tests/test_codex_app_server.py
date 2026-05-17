import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from agent_harness.runtime.codex_app_server import CodexAppServerClient


class FakeRpc:
    instances: ClassVar[list["FakeRpc"]] = []

    def __init__(self, url, timeout_seconds):
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.requests = []
        self.notifications = []
        self.closed = False
        FakeRpc.instances.append(self)

    def connect(self):
        return None

    def request(self, method, params, server_request_handler=None):
        self.requests.append((method, params, server_request_handler))
        return {}

    def notify(self, method, params=None):
        self.notifications.append((method, params))

    def close(self):
        self.closed = True


class CodexAppServerClientTests(unittest.TestCase):
    def test_send_to_thread_forwards_dangerous_permission_overrides(self):
        FakeRpc.instances = []
        with patch("agent_harness.runtime.codex_app_server._JsonRpcWebSocket", FakeRpc):
            client = CodexAppServerClient("ws://127.0.0.1:47684")

            handled = client.send_to_thread(
                "thread-1",
                "continue",
                Path("/tmp/repo"),
                approval_policy="never",
                sandbox="danger-full-access",
                sandbox_policy={"type": "dangerFullAccess"},
            )

        self.assertTrue(handled)
        rpc = FakeRpc.instances[0]
        self.assertEqual(rpc.requests[1][0], "thread/resume")
        self.assertEqual(rpc.requests[1][1]["approvalPolicy"], "never")
        self.assertEqual(rpc.requests[1][1]["sandbox"], "danger-full-access")
        self.assertEqual(rpc.requests[2][0], "turn/start")
        self.assertEqual(rpc.requests[2][1]["approvalPolicy"], "never")
        self.assertEqual(rpc.requests[2][1]["sandboxPolicy"], {"type": "dangerFullAccess"})
        self.assertTrue(rpc.closed)


if __name__ == "__main__":
    unittest.main()
