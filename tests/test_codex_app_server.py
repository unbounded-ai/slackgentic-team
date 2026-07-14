import os
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from agent_harness.config import AgentCommandConfig
from agent_harness.runtime.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerManager,
    CodexAppServerSupervisor,
    CodexInstallation,
    resolve_codex_installation,
)


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


class FakeProcess:
    def __init__(self):
        self.stdout = None
        self.stderr = None
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


class CodexAppServerSupervisorTests(unittest.TestCase):
    def _write_codex(self, path: Path, version: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!/bin/sh\nprintf 'codex-cli {version}\\n'\n")
        path.chmod(0o755)

    def test_resolver_prefers_newest_codex_even_when_older_binary_is_first_on_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_codex = root / "old" / "codex"
            new_codex = root / "new" / "codex"
            self._write_codex(old_codex, "0.144.1")
            self._write_codex(new_codex, "0.144.3")

            installation = resolve_codex_installation(
                "codex",
                environ_path=os.pathsep.join((str(old_codex.parent), str(new_codex.parent))),
            )

        assert installation is not None
        self.assertEqual(installation.executable, new_codex)
        self.assertEqual(installation.version, "0.144.3")

    def test_resolver_honors_an_explicit_codex_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex = Path(tmp) / "pinned" / "codex"
            self._write_codex(codex, "0.144.1")

            installation = resolve_codex_installation(str(codex))

        assert installation is not None
        self.assertEqual(installation.executable, codex)
        self.assertEqual(installation.version, "0.144.1")

    def test_supervisor_restarts_app_server_when_codex_installation_changes(self):
        first = CodexInstallation(Path("/opt/tools/codex-old"), "0.144.1", ("old",))
        second = CodexInstallation(Path("/opt/tools/codex-new"), "0.144.3", ("new",))
        first_process = FakeProcess()
        second_process = FakeProcess()
        supervisor = CodexAppServerSupervisor()

        with (
            patch(
                "agent_harness.runtime.codex_app_server.resolve_codex_installation",
                side_effect=(first, second),
            ),
            patch(
                "agent_harness.runtime.codex_app_server.subprocess.Popen",
                side_effect=(first_process, second_process),
            ) as popen,
            patch("agent_harness.runtime.codex_app_server._health_ready", return_value=True),
        ):
            self.assertTrue(supervisor.reconcile(force=True, snapshot=(("old",),)))
            self.assertTrue(supervisor.reconcile(snapshot=(("new",),)))

        self.assertTrue(first_process.terminated)
        self.assertEqual(popen.call_args_list[0].args[0][0], str(first.executable))
        self.assertEqual(popen.call_args_list[1].args[0][0], str(second.executable))
        supervisor.close()

    def test_embedded_manager_starts_the_newest_resolved_codex(self):
        installation = CodexInstallation(
            Path("/opt/tools/codex-current"),
            "0.144.3",
            ("current",),
        )
        process = FakeProcess()
        manager = CodexAppServerManager(AgentCommandConfig(codex_binary="codex"))

        with (
            patch(
                "agent_harness.runtime.codex_app_server.resolve_codex_installation",
                return_value=installation,
            ),
            patch(
                "agent_harness.runtime.codex_app_server.subprocess.Popen",
                return_value=process,
            ) as popen,
            patch(
                "agent_harness.runtime.codex_app_server._health_ready",
                side_effect=(False, True),
            ),
        ):
            self.assertEqual(manager.start(), "ws://127.0.0.1:47684")

        self.assertEqual(popen.call_args.args[0][0], str(installation.executable))
        manager.close()


if __name__ == "__main__":
    unittest.main()
