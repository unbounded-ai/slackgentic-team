import unittest
from pathlib import Path
from unittest.mock import patch

import pexpect

from agent_harness.models import Provider
from agent_harness.runner import LaunchRequest, ManagedAgentProcess, build_command


class FakeChild:
    def __init__(self):
        self.after = ""
        self.before = ""
        self.calls = 0
        self.sent = []
        self.eof_sent = False

    def expect(self, pattern, timeout):
        self.calls += 1
        if self.calls == 1:
            self.after = "first"
            return 0
        self.before = "tail"
        raise pexpect.EOF("done")

    def send(self, value):
        self.sent.append(value)

    def sendeof(self):
        self.eof_sent = True


class RunnerTests(unittest.TestCase):
    def test_codex_command_uses_argv_not_shell(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CODEX,
                prompt="fix it",
                cwd=Path("/tmp/repo"),
                dangerous=True,
                model="gpt-5.4",
            )
        )
        self.assertEqual(command, "codex")
        self.assertEqual(args[0], "exec")
        self.assertIn("--json", args)
        self.assertIn("--color", args)
        self.assertIn("never", args)
        self.assertIn("--ephemeral", args)
        self.assertIn('projects."/tmp/repo".trust_level="trusted"', args)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", args)
        self.assertEqual(args[-3:], ["-C", "/tmp/repo", "-"])
        self.assertNotIn("fix it", args)

    def test_codex_command_uses_non_interactive_permissions_by_default(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CODEX,
                prompt="fix it",
                cwd=Path("/tmp/repo"),
            )
        )

        self.assertEqual(command, "codex")
        self.assertIn("--sandbox", args)
        self.assertIn("workspace-write", args)
        self.assertNotIn("--ask-for-approval", args)

    def test_claude_command(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="fix it",
                cwd=Path("/tmp/repo"),
                dangerous=True,
                model="opus",
                worktree="feature",
            )
        )
        self.assertEqual(command, "claude")
        self.assertIn("--print", args)
        self.assertIn("--output-format", args)
        self.assertIn("json", args)
        self.assertIn("--no-session-persistence", args)
        self.assertIn("--dangerously-skip-permissions", args)
        self.assertIn("--worktree", args)
        self.assertEqual(args[-1], "-")
        self.assertNotIn("fix it", args)

    def test_managed_process_read_available_keeps_eof_tail(self):
        process = ManagedAgentProcess(
            LaunchRequest(provider=Provider.CODEX, prompt="x", cwd=Path("/tmp/repo"))
        )
        process.child = FakeChild()

        self.assertEqual(process.read_available(), "firsttail")

    def test_managed_process_sends_prompt_over_stdin(self):
        child = FakeChild()
        calls = []

        def fake_spawn(command, args, **kwargs):
            calls.append((command, args, kwargs))
            return child

        process = ManagedAgentProcess(
            LaunchRequest(provider=Provider.CODEX, prompt="hidden prompt", cwd=Path("/tmp/repo"))
        )

        with patch("pexpect.spawn", fake_spawn):
            process.start()

        self.assertEqual(calls[0][0], "codex")
        self.assertNotIn("hidden prompt", calls[0][1])
        self.assertEqual(child.sent, ["hidden prompt", "\n"])
        self.assertTrue(child.eof_sent)

    def test_managed_process_sends_claude_prompt_over_stdin(self):
        child = FakeChild()
        calls = []

        def fake_spawn(command, args, **kwargs):
            calls.append((command, args, kwargs))
            return child

        process = ManagedAgentProcess(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="hidden claude prompt",
                cwd=Path("/tmp/repo"),
            )
        )

        with patch("pexpect.spawn", fake_spawn):
            process.start()

        self.assertEqual(calls[0][0], "claude")
        self.assertEqual(calls[0][1][-1], "-")
        self.assertNotIn("hidden claude prompt", calls[0][1])
        self.assertEqual(child.sent, ["hidden claude prompt", "\n"])
        self.assertTrue(child.eof_sent)


if __name__ == "__main__":
    unittest.main()
