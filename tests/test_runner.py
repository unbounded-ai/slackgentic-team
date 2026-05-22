import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pexpect

from agent_harness.models import PermissionMode, Provider
from agent_harness.permissions import CLAUDE_CHANNEL_PERMISSION_MODE_ENV
from agent_harness.runtime.runner import LaunchRequest, ManagedAgentProcess, build_command


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


class FakeProc:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15


class FakePopenChild(FakeChild):
    def __init__(self, returncode=None):
        super().__init__()
        self.proc = FakeProc(returncode)


class BrokenIsAliveChild(FakeChild):
    def isalive(self):
        raise pexpect.ExceptionPexpect("no child processes")


class RunnerTests(unittest.TestCase):
    def test_codex_command_uses_argv_not_shell(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CODEX,
                prompt="fix it",
                cwd=Path("/tmp/repo"),
                permission_mode=PermissionMode.DANGEROUS,
                model="gpt-5.4",
            )
        )
        self.assertEqual(command, "codex")
        self.assertEqual(args[0], "exec")
        self.assertIn("--json", args)
        self.assertIn("--color", args)
        self.assertIn("never", args)
        self.assertNotIn("--ephemeral", args)
        self.assertIn('projects."/tmp/repo".trust_level="trusted"', args)
        self.assertIn('sandbox_mode="danger-full-access"', args)
        self.assertIn('approval_policy="never"', args)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", args)
        self.assertEqual(args[-3:], ["-C", "/tmp/repo", "-"])
        self.assertNotIn("fix it", args)

    def test_codex_command_uses_safe_auto_sandbox_by_default(self):
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

    def test_codex_command_adds_safe_auto_extra_roots(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CODEX,
                prompt="fix it",
                cwd=Path("/workspace/repos/example-project"),
                safe_auto_extra_roots=(Path("/workspace/repos"),),
            )
        )

        self.assertEqual(command, "codex")
        self.assertIn("--add-dir", args)
        root_index = args.index("--add-dir")
        self.assertEqual(args[root_index + 1], "/workspace/repos")

    def test_codex_command_does_not_add_locked_extra_roots(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CODEX,
                prompt="fix it",
                cwd=Path("/workspace/repos/example-project"),
                permission_mode=PermissionMode.LOCKED,
                safe_auto_extra_roots=(Path("/workspace/repos"),),
            )
        )

        self.assertEqual(command, "codex")
        self.assertNotIn("--add-dir", args)

    def test_codex_command_uses_read_only_sandbox_when_locked(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CODEX,
                prompt="fix it",
                cwd=Path("/tmp/repo"),
                permission_mode=PermissionMode.LOCKED,
            )
        )

        self.assertEqual(command, "codex")
        self.assertIn("--sandbox", args)
        self.assertIn("read-only", args)

    def test_claude_command(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="fix it",
                cwd=Path("/tmp/repo"),
                permission_mode=PermissionMode.DANGEROUS,
                model="opus",
                worktree="feature",
                claude_effort="high",
            )
        )
        self.assertEqual(command, "claude")
        self.assertIn("--print", args)
        self.assertIn("--verbose", args)
        self.assertIn("--output-format", args)
        self.assertIn("--input-format", args)
        input_index = args.index("--input-format")
        self.assertEqual(args[input_index + 1], "stream-json")
        self.assertIn("--effort", args)
        effort_index = args.index("--effort")
        self.assertEqual(args[effort_index + 1], "high")
        self.assertNotIn("--no-session-persistence", args)
        self.assertIn("--dangerously-skip-permissions", args)
        self.assertNotIn("--permission-mode", args)
        self.assertIn("--worktree", args)
        # Initial prompt is delivered over stdin as a stream-json user turn,
        # not as a positional argv. See ManagedAgentProcess.start.
        self.assertNotIn("fix it", args)

    def test_claude_command_uses_safe_auto_permission_mode_by_default(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="review pr",
                cwd=Path("/tmp/repo"),
            )
        )

        self.assertEqual(command, "claude")
        self.assertIn("--permission-mode", args)
        permission_index = args.index("--permission-mode")
        self.assertEqual(args[permission_index + 1], "acceptEdits")
        self.assertNotIn("--effort", args)
        self.assertNotIn("--dangerously-skip-permissions", args)
        self.assertIn("--allowedTools=Bash(git status:*)", args)
        self.assertIn("--allowedTools=Bash(git add:*)", args)
        self.assertIn("--allowedTools=Bash(git commit:*)", args)
        self.assertIn("--allowedTools=Bash(git pull:*)", args)
        self.assertIn("--allowedTools=Bash(git config user.name)", args)
        self.assertIn("--allowedTools=Bash(git config user.email)", args)
        self.assertIn("--allowedTools=Bash(gh pr view:*)", args)
        self.assertIn("--allowedTools=Bash(gh pr create:*)", args)

    def test_claude_command_adds_safe_auto_extra_roots(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="review pr",
                cwd=Path("/workspace/repos/example-project"),
                safe_auto_extra_roots=(Path("/workspace/repos"),),
            )
        )

        self.assertEqual(command, "claude")
        self.assertIn("--add-dir", args)
        root_index = args.index("--add-dir")
        self.assertEqual(args[root_index + 1], "/workspace/repos")
        self.assertIn("--permission-mode", args)
        self.assertLess(root_index, args.index("--permission-mode"))

    def test_claude_safe_auto_allowlist_excludes_mutating_commands(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="review pr",
                cwd=Path("/tmp/repo"),
            )
        )

        self.assertEqual(command, "claude")
        forbidden = (
            "Bash(git branch:*)",
            "Bash(git remote:*)",
            "Bash(gh api:*)",
            "Bash(find:*)",
        )
        for tool in forbidden:
            self.assertNotIn(f"--allowedTools={tool}", args)

    def test_claude_command_locked_skips_permission_flag(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="review pr",
                cwd=Path("/tmp/repo"),
                permission_mode=PermissionMode.LOCKED,
                safe_auto_extra_roots=(Path("/workspace/repos"),),
            )
        )

        self.assertEqual(command, "claude")
        self.assertNotIn("--permission-mode", args)
        self.assertNotIn("--dangerously-skip-permissions", args)
        self.assertNotIn("--allowedTools=Bash(git status:*)", args)
        self.assertNotIn("--add-dir", args)

    def test_claude_command_can_load_slackgentic_channel(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="fix it",
                cwd=Path("/tmp/repo"),
                claude_channel=True,
            )
        )

        self.assertEqual(command, "claude")
        self.assertIn("--dangerously-load-development-channels=server:slackgentic", args)

    def test_claude_command_can_allow_exact_tool(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="fix it",
                cwd=Path("/tmp/repo"),
                allowed_tools=("Bash(gh auth status)",),
            )
        )

        self.assertEqual(command, "claude")
        self.assertIn("--allowedTools=Bash(gh auth status)", args)

    def test_codex_resume_command_uses_existing_session(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CODEX,
                prompt="continue",
                cwd=Path("/tmp/repo"),
                resume_session_id="thread-1",
            )
        )

        self.assertEqual(command, "codex")
        self.assertEqual(args[:2], ["exec", "resume"])
        self.assertIn("thread-1", args)
        self.assertEqual(args[-1], "continue")
        self.assertNotIn("--color", args)
        self.assertNotIn("--sandbox", args)
        self.assertNotIn("-C", args)
        self.assertNotEqual(args[-1], "-")
        self.assertIn('sandbox_mode="workspace-write"', args)

    def test_codex_resume_command_adds_safe_auto_extra_roots(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CODEX,
                prompt="continue",
                cwd=Path("/workspace/repos/example-project"),
                resume_session_id="thread-1",
                safe_auto_extra_roots=(Path("/workspace/repos"),),
            )
        )

        self.assertEqual(command, "codex")
        self.assertNotIn("--add-dir", args)
        self.assertIn('sandbox_workspace_write.writable_roots=["/workspace/repos"]', args)

    def test_codex_resume_command_enforces_locked_sandbox(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CODEX,
                prompt="continue",
                cwd=Path("/tmp/repo"),
                resume_session_id="thread-1",
                permission_mode=PermissionMode.LOCKED,
            )
        )

        self.assertEqual(command, "codex")
        self.assertIn('sandbox_mode="read-only"', args)
        self.assertNotIn("--sandbox", args)

    def test_codex_resume_command_forces_dangerous_policy(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CODEX,
                prompt="continue",
                cwd=Path("/tmp/repo"),
                resume_session_id="thread-1",
                permission_mode=PermissionMode.DANGEROUS,
            )
        )

        self.assertEqual(command, "codex")
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", args)
        self.assertIn('sandbox_mode="danger-full-access"', args)
        self.assertIn('approval_policy="never"', args)

    def test_claude_resume_command_uses_existing_session(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="continue",
                cwd=Path("/tmp/repo"),
                resume_session_id="session-1",
                claude_effort="max",
            )
        )

        self.assertEqual(command, "claude")
        self.assertIn("--resume", args)
        self.assertIn("session-1", args)
        self.assertIn("--input-format", args)
        self.assertIn("--effort", args)
        effort_index = args.index("--effort")
        self.assertEqual(args[effort_index + 1], "max")
        # See above: the prompt is now streamed to stdin, not passed positionally.
        self.assertNotIn("continue", args)

    def test_claude_resume_command_adds_safe_auto_extra_roots(self):
        command, args = build_command(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="continue",
                cwd=Path("/workspace/repos/example-project"),
                resume_session_id="session-1",
                safe_auto_extra_roots=(Path("/workspace/repos"),),
            )
        )

        self.assertEqual(command, "claude")
        self.assertIn("--resume", args)
        self.assertIn("--add-dir", args)
        root_index = args.index("--add-dir")
        self.assertEqual(args[root_index + 1], "/workspace/repos")
        self.assertLess(root_index, args.index("--permission-mode"))

    def test_managed_process_read_available_keeps_eof_tail(self):
        process = ManagedAgentProcess(
            LaunchRequest(provider=Provider.CODEX, prompt="x", cwd=Path("/tmp/repo"))
        )
        process.child = FakeChild()

        self.assertEqual(process.read_available(), "firsttail")

    def test_managed_process_sends_codex_prompt_over_pipe_stdin(self):
        child = FakeChild()
        calls = []
        prompt = "hidden prompt " * 2000

        def fake_popen_spawn(command, **kwargs):
            calls.append((command, kwargs))
            return child

        process = ManagedAgentProcess(
            LaunchRequest(provider=Provider.CODEX, prompt=prompt, cwd=Path("/tmp/repo"))
        )

        def fail_pty_spawn(*_args, **_kwargs):
            raise AssertionError("codex stdin prompt should use pipe-backed spawn")

        with (
            patch("pexpect.spawn", fail_pty_spawn),
            patch("pexpect.popen_spawn.PopenSpawn", fake_popen_spawn),
        ):
            process.start()

        self.assertEqual(calls[0][0][0], "codex")
        self.assertIn("-", calls[0][0])
        self.assertNotIn(prompt, calls[0][0])
        self.assertEqual(child.sent, [prompt, "\n"])
        self.assertTrue(child.eof_sent)

    def test_managed_process_polls_pipe_backed_codex_child(self):
        process = ManagedAgentProcess(
            LaunchRequest(provider=Provider.CODEX, prompt="x", cwd=Path("/tmp/repo"))
        )
        process.child = FakePopenChild()

        self.assertTrue(process.is_alive())
        process.terminate()

        self.assertTrue(process.child.proc.terminated)
        self.assertFalse(process.is_alive())

    def test_managed_process_treats_pexpect_poll_error_as_exited(self):
        process = ManagedAgentProcess(
            LaunchRequest(provider=Provider.CLAUDE, prompt="x", cwd=Path("/tmp/repo"))
        )
        process.child = BrokenIsAliveChild()

        self.assertFalse(process.is_alive())

    def test_managed_codex_resume_process_does_not_send_stdin(self):
        child = FakeChild()

        def fake_spawn(command, args, **kwargs):
            return child

        process = ManagedAgentProcess(
            LaunchRequest(
                provider=Provider.CODEX,
                prompt="hidden prompt",
                cwd=Path("/tmp/repo"),
                resume_session_id="thread-1",
            )
        )

        with patch("pexpect.spawn", fake_spawn):
            process.start()

        self.assertEqual(child.sent, [])
        self.assertFalse(child.eof_sent)

    def test_managed_process_sends_claude_prompt_over_pipe_stdin_as_stream_json(self):
        child = FakeChild()
        calls = []

        def fake_popen_spawn(command, **kwargs):
            calls.append((command, kwargs))
            return child

        process = ManagedAgentProcess(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="hidden claude prompt",
                cwd=Path("/tmp/repo"),
            )
        )

        def fail_pty_spawn(*_args, **_kwargs):
            raise AssertionError("claude --input-format stream-json should use pipe-backed spawn")

        with (
            patch("pexpect.spawn", fail_pty_spawn),
            patch("pexpect.popen_spawn.PopenSpawn", fake_popen_spawn),
        ):
            process.start()

        self.assertEqual(calls[0][0][0], "claude")
        # Prompt is not on argv anymore; it's written to stdin.
        self.assertNotIn("hidden claude prompt", calls[0][0])
        self.assertEqual(len(child.sent), 1)
        payload = json.loads(child.sent[0])
        self.assertEqual(payload["type"], "user")
        self.assertEqual(payload["message"]["role"], "user")
        self.assertEqual(payload["message"]["content"], "hidden claude prompt")
        # Stdin must stay open so follow-up user turns can be appended.
        self.assertFalse(child.eof_sent)

    def test_managed_claude_send_writes_stream_json_user_turn(self):
        child = FakeChild()

        def fake_popen_spawn(command, **kwargs):
            return child

        process = ManagedAgentProcess(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="initial",
                cwd=Path("/tmp/repo"),
            )
        )

        with (
            patch("pexpect.spawn", lambda *a, **k: child),
            patch("pexpect.popen_spawn.PopenSpawn", fake_popen_spawn),
        ):
            process.start()
            process.send("a follow-up from slack")

        # First send is the initial prompt; second is the follow-up.
        self.assertEqual(len(child.sent), 2)
        followup = json.loads(child.sent[1])
        self.assertEqual(followup["type"], "user")
        self.assertEqual(followup["message"]["role"], "user")
        self.assertEqual(followup["message"]["content"], "a follow-up from slack")
        self.assertFalse(child.eof_sent)

    def test_managed_claude_process_passes_slack_thread_env(self):
        child = FakeChild()
        calls = []

        def fake_popen_spawn(command, **kwargs):
            calls.append((command, kwargs))
            return child

        process = ManagedAgentProcess(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="hidden claude prompt",
                cwd=Path("/tmp/repo"),
                slack_channel_id="C1",
                slack_thread_ts="171.000001",
            )
        )

        with patch("pexpect.popen_spawn.PopenSpawn", fake_popen_spawn):
            process.start()

        self.assertEqual(calls[0][1]["env"]["SLACKGENTIC_CLAUDE_CHANNEL_ID"], "C1")
        self.assertEqual(calls[0][1]["env"]["SLACKGENTIC_CLAUDE_THREAD_TS"], "171.000001")

    def test_managed_claude_dangerous_mode_sets_channel_dangerous_env(self):
        child = FakeChild()
        calls = []

        def fake_popen_spawn(command, **kwargs):
            calls.append((command, kwargs))
            return child

        process = ManagedAgentProcess(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="hidden claude prompt",
                cwd=Path("/tmp/repo"),
                permission_mode=PermissionMode.DANGEROUS,
            )
        )

        with patch("pexpect.popen_spawn.PopenSpawn", fake_popen_spawn):
            process.start()

        self.assertEqual(calls[0][1]["env"]["SLACKGENTIC_CLAUDE_DANGEROUS_MODE"], "1")

    def test_managed_claude_safe_auto_omits_channel_dangerous_env(self):
        child = FakeChild()
        calls = []

        def fake_popen_spawn(command, **kwargs):
            calls.append((command, kwargs))
            return child

        process = ManagedAgentProcess(
            LaunchRequest(
                provider=Provider.CLAUDE,
                prompt="hidden claude prompt",
                cwd=Path("/tmp/repo"),
            )
        )

        clean_env = dict(os.environ)
        clean_env.pop("SLACKGENTIC_CLAUDE_DANGEROUS_MODE", None)
        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch("pexpect.popen_spawn.PopenSpawn", fake_popen_spawn),
        ):
            process.start()

        self.assertNotIn(
            "SLACKGENTIC_CLAUDE_DANGEROUS_MODE",
            calls[0][1]["env"],
        )
        self.assertEqual(
            calls[0][1]["env"][CLAUDE_CHANNEL_PERMISSION_MODE_ENV],
            PermissionMode.SAFE_AUTO.value,
        )


if __name__ == "__main__":
    unittest.main()
