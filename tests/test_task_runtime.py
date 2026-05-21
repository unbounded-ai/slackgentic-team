import json
import logging
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from agent_harness.config import AgentCommandConfig
from agent_harness.deferred import AGENT_DEFERRED_SIGNAL_PREFIX
from agent_harness.models import (
    DANGEROUS_MODE_METADATA_KEY,
    PERMISSION_MODE_METADATA_KEY,
    PR_URLS_METADATA_KEY,
    AgentTaskKind,
    AgentTaskStatus,
    PermissionMode,
    Provider,
    SlackThreadRef,
)
from agent_harness.runtime.tasks import (
    AGENT_REACTION_SIGNAL_PREFIX,
    AGENT_ROSTER_STATUS_SIGNAL_PREFIX,
    AGENT_THREAD_DONE_SIGNAL,
    MANAGED_RUN_MAX_RESUME_AGE,
    MANAGED_RUN_MAX_RESUMES,
    MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY,
    MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY,
    MANAGED_RUN_STARTED_METADATA_KEY,
    ManagedTaskRuntime,
    RunningTask,
    _allowed_session_tools_for_claude_denial,
    _allowed_tool_for_claude_denial,
    _allowed_tools_for_claude_denial,
    _append_allowed_tool,
    _claude_missing_resume_session,
    _claude_permission_denials,
    _clean_terminal_output,
    _extract_agent_control_signals,
    _latest_codex_transcript_message,
    _macos_tcc_protected_cwd_issue,
    _process_output_chunks,
    _requested_repo_cwd,
    _session_id_from_output,
    build_task_prompt,
    managed_run_resume_attempts,
    parse_agent_reaction_signal,
    parse_agent_roster_status_signal,
    should_resume_managed_run,
)
from agent_harness.schedules import AGENT_SCHEDULE_SIGNAL_PREFIX
from agent_harness.sessions.claude_channel import SLACKGENTIC_MCP_PERMISSION_ALLOW
from agent_harness.slack.client import PostedMessage
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team, create_agent_task
from agent_harness.timers import AGENT_TIMER_SIGNAL_PREFIX, parse_agent_timer_signal


class FakeGateway:
    def __init__(self):
        self.replies = []
        self.icon_urls = []
        self.blocks = []
        self.updates = []

    def post_thread_reply(self, thread, text, persona=None, icon_url=None, blocks=None):
        self.replies.append(text)
        self.icon_urls.append(icon_url)
        self.blocks.append(blocks)
        ts = f"1712345678.{len(self.replies):06d}"
        return PostedMessage(thread.channel_id, ts, thread.thread_ts)

    def update_message(self, channel_id, ts, text, blocks=None):
        self.updates.append({"channel_id": channel_id, "ts": ts, "text": text, "blocks": blocks})


class FailingGateway(FakeGateway):
    def post_thread_reply(self, thread, text, persona=None, icon_url=None, blocks=None):
        raise RuntimeError("Slack post failed")


class OneShotProcess:
    def __init__(self, request):
        self.request = request
        self.reads = 0

    def start(self):
        pass

    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return '{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}\n'
        return ""

    def is_alive(self):
        return False

    def terminate(self):
        pass


class DuplicateCodexFinalProcess(OneShotProcess):
    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            text = "Final answer"
            return (
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {"type": "agent_message", "message": text},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        },
                    }
                )
                + "\n"
            )
        return ""


class ClaudeOneShotProcess(OneShotProcess):
    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return '{"type":"result","subtype":"success","is_error":false,"result":"Done"}\n'
        return ""


class ClaudePermissionDeniedProcess(OneShotProcess):
    session_id = "claude-denied-session"
    command = "gh pr view 6 --repo unbounded-ai/slackgentic-team --json number,title"
    description = "View PR metadata"

    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            denial = {
                "tool_name": "Bash",
                "tool_use_id": "toolu_1",
                "tool_input": {
                    "command": self.command,
                    "description": self.description,
                },
            }
            return (
                f'{{"type":"system","session_id":"{self.session_id}"}}\n'
                + json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": "The command requires approval.",
                        "permission_denials": [denial],
                        "session_id": self.session_id,
                    }
                )
                + "\n"
            )
        return ""


class ClaudePipedGitStatusPermissionDeniedProcess(ClaudePermissionDeniedProcess):
    session_id = "claude-piped-git-status-session"
    command = "git -C /workspace/repos/example-project status 2>&1 | head -30"
    description = "Check example repo status"


class ClaudeSequencedPatchLookupPermissionDeniedProcess(ClaudePermissionDeniedProcess):
    session_id = "claude-sequenced-patch-lookup-session"
    command = "ls /tmp/example-provider-analysis.patch 2>&1; ls /tmp/ | grep -i example 2>&1"
    description = "Check for prior patch file"


class ClaudeDangerousPermissionDeniedThenFinalProcess(OneShotProcess):
    session_id = "claude-dangerous-denied-session"

    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            denial = {
                "tool_name": "Task",
                "tool_use_id": "toolu_unsupported",
                "tool_input": {"description": "delegate work"},
            }
            return (
                f'{{"type":"system","session_id":"{self.session_id}"}}\n'
                + json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": "The tool requires approval.",
                        "permission_denials": [denial],
                        "session_id": self.session_id,
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": "Done after dangerous-mode bypass.",
                        "session_id": self.session_id,
                    }
                )
                + "\n"
            )
        return ""


class ClaudeSlackgenticMcpPermissionDeniedProcess(OneShotProcess):
    session_id = "claude-slackgentic-mcp-denied-session"
    tool_name = "mcp__slackgentic__request_user_input"

    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            denial = {
                "tool_name": self.tool_name,
                "tool_use_id": "toolu_slackgentic",
                "tool_input": {
                    "question": "Pick a landing path",
                    "options": [{"label": "Open PR"}],
                },
            }
            return (
                f'{{"type":"system","session_id":"{self.session_id}"}}\n'
                + json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": "Slack request tool requires approval.",
                        "permission_denials": [denial],
                        "session_id": self.session_id,
                    }
                )
                + "\n"
            )
        return ""


class ClaudeStreamPermissionDeniedProcess(OneShotProcess):
    session_id = "claude-stream-denied-session"
    command = "cd /workspace/repos/sample-app && git log --oneline -30"
    description = "Show recent commits"

    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            assistant = {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_stream",
                            "name": "Bash",
                            "input": {
                                "command": self.command,
                                "description": self.description,
                            },
                        }
                    ]
                },
            }
            denial = {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_stream",
                            "content": (
                                "This command changes directory before running git, "
                                "which can execute untrusted hooks from the target "
                                "directory. Approve before running."
                            ),
                            "is_error": True,
                        }
                    ]
                },
            }
            return (
                f'{{"type":"system","session_id":"{self.session_id}"}}\n'
                + json.dumps(assistant)
                + "\n"
                + json.dumps(denial)
                + "\n"
            )
        return ""


class ClaudeLivePermissionDeniedProcess(ClaudeStreamPermissionDeniedProcess):
    session_id = "claude-live-denied-session"

    def __init__(self, request):
        super().__init__(request)
        self.terminated = False

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def read_available(self, max_reads=20, timeout=0.05):
        if self.terminated:
            return ""
        if self.reads == 0:
            return super().read_available(max_reads=max_reads, timeout=timeout)
        self.reads += 1
        assistant = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_second",
                        "name": "Bash",
                        "input": {
                            "command": "codex --help",
                            "description": "Try a different command",
                        },
                    }
                ]
            },
        }
        denial = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_second",
                        "content": "This command requires approval",
                        "is_error": True,
                    }
                ]
            },
        }
        return json.dumps(assistant) + "\n" + json.dumps(denial) + "\n"


class ClaudeSecondPermissionDeniedProcess(ClaudePermissionDeniedProcess):
    command = 'gh search prs --author "@me" --state open --json number,title --limit 50'
    description = "Search open PRs"


class ClaudeMultilineCommitPermissionDeniedProcess(ClaudePermissionDeniedProcess):
    session_id = "claude-multiline-commit-session"
    command = """git -C /workspace/repos/sample-app commit -m "$(cat <<'EOF'
[sample-app] Add parity adapter and rollout bridge

Parity for Codex.
EOF
)" """
    description = "Commit codex parity"


class ClaudeMissingResumeProcess(OneShotProcess):
    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return (
                "No conversation found with session ID: missing-session\n"
                + json.dumps(
                    {
                        "type": "result",
                        "subtype": "error_during_execution",
                        "is_error": True,
                        "session_id": "bogus-error-session",
                        "errors": [
                            "No conversation found with session ID: missing-session",
                        ],
                    }
                )
                + "\n"
            )
        return ""


class SessionIdProcess(OneShotProcess):
    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return (
                '{"type":"thread.started","thread_id":"codex-thread-1"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}\n'
            )
        return ""


class ThreadDoneSignalProcess(OneShotProcess):
    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return (
                '{"type":"item.completed","item":{"type":"agent_message",'
                f'"text":"Sounds good.\\n{AGENT_THREAD_DONE_SIGNAL}"'
                "}}\n"
            )
        return ""


class LiveThreadDoneSignalProcess(ThreadDoneSignalProcess):
    instances: ClassVar[list] = []

    def __init__(self, request):
        super().__init__(request)
        self.alive = True
        self.terminated = False
        self.__class__.instances.append(self)

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False


class TimerSignalProcess(OneShotProcess):
    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return (
                '{"type":"item.completed","item":{"type":"agent_message",'
                f'"text":"Wakeup scheduled.\\n{AGENT_TIMER_SIGNAL_PREFIX}'
                '2s | Re-check the PR feedback."}}\n'
            )
        return ""


class ScheduleSignalProcess(OneShotProcess):
    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            signal = (
                f"{AGENT_SCHEDULE_SIGNAL_PREFIX}"
                '{"task":"check CI","target":"somebody","schedule":{"kind":"one_off",'
                '"run_at":"2026-05-16T01:20:00Z"}}'
            )
            return (
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": f"Schedule created.\n{signal}",
                        },
                    }
                )
                + "\n"
            )
        return ""


class RosterStatusSignalProcess(OneShotProcess):
    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return (
                '{"type":"item.completed","item":{"type":"agent_message",'
                f'"text":"Still testing.\\n{AGENT_ROSTER_STATUS_SIGNAL_PREFIX}'
                'PR merge and daemon restart: running E2E."}}\n'
            )
        return ""


class SilentProcess(OneShotProcess):
    def read_available(self, max_reads=20, timeout=0.05):
        self.reads += 1
        return ""


class HoldingProcess(OneShotProcess):
    def __init__(self, request):
        super().__init__(request)
        self.alive = True
        self.interrupts = 0

    def read_available(self, max_reads=20, timeout=0.05):
        return ""

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def interrupt(self):
        self.interrupts += 1


class SessionThenHoldingProcess(HoldingProcess):
    session_id = "codex-thread-1"

    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return (
                f'{{"type":"thread.started","thread_id":"{self.session_id}"}}\n'
                '{"type":"event_msg","payload":{"type":"agent_message","message":"Working"}}\n'
            )
        return ""


class CrashingProcess(OneShotProcess):
    def __init__(self, request):
        super().__init__(request)
        self.alive = True
        self.terminated = False

    def read_available(self, max_reads=20, timeout=0.05):
        raise UnicodeDecodeError("utf-8", b"\x94", 0, 1, "invalid start byte")

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False


class ClaudeSessionOnlyProcess(OneShotProcess):
    session_id = "claude-fallback-session"

    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return f'{{"type":"system","subtype":"init","session_id":"{self.session_id}"}}\n'
        return ""


class ClaudeAssistantOnlyProcess(OneShotProcess):
    session_id = "claude-assistant-only-session"

    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return (
                f'{{"type":"system","subtype":"init","session_id":"{self.session_id}"}}\n'
                '{"type":"assistant","message":{"role":"assistant","content":'
                '[{"type":"text","text":"Working on it"}]}}\n'
                '{"type":"assistant","message":{"role":"assistant","content":'
                '[{"type":"text","text":"PR up: '
                'https://github.com/example/repo/pull/456"}]}}\n'
            )
        return ""


class SessionOnlyProcess(OneShotProcess):
    session_id = "session-fallback"

    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return f'{{"type":"thread.started","thread_id":"{self.session_id}"}}\n'
        return ""


class ProgressOnlySessionProcess(OneShotProcess):
    session_id = "session-progress-final-fallback"

    def read_available(self, max_reads=20, timeout=0.05):
        if self.reads == 0:
            self.reads += 1
            return (
                f'{{"type":"thread.started","thread_id":"{self.session_id}"}}\n'
                '{"type":"event_msg","payload":{"type":"agent_message","message":"Working"}}\n'
            )
        return ""


def _with_permission_mode(task, mode: PermissionMode):
    return replace(
        task,
        metadata={**task.metadata, PERMISSION_MODE_METADATA_KEY: mode.value},
    )


class TaskRuntimeTests(unittest.TestCase):
    def test_build_task_prompt_includes_persona_and_pr_context(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "review the PR", "C1", kind=AgentTaskKind.REVIEW)
        task = replace(task, metadata={"pr_url": "https://github.com/acme/app/pull/42"})

        prompt = build_task_prompt(agent, task)

        self.assertIn(f"@{agent.handle}", prompt)
        self.assertIn("review the PR", prompt)
        self.assertIn("https://github.com/acme/app/pull/42", prompt)

    def test_build_task_prompt_includes_multiple_pr_contexts(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "review the PRs", "C1", kind=AgentTaskKind.REVIEW)
        task = replace(
            task,
            metadata={
                PR_URLS_METADATA_KEY: [
                    "https://github.com/acme/app/pull/42",
                    "https://github.com/acme/app/pull/43",
                ]
            },
        )

        prompt = build_task_prompt(agent, task)

        self.assertIn("Pull request URLs:", prompt)
        self.assertIn("- https://github.com/acme/app/pull/42", prompt)
        self.assertIn("- https://github.com/acme/app/pull/43", prompt)
        self.assertIn("Review the PR(s)", prompt)

    def test_build_task_prompt_includes_thread_context(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "review this", "C1")
        task = replace(task, metadata={"thread_context": "A: original answer\nB: review this"})

        prompt = build_task_prompt(agent, task)

        self.assertIn("Private Slack thread context", prompt)
        self.assertIn("Do not quote this heading", prompt)
        self.assertIn("original answer", prompt)

    def test_build_task_prompt_instructs_agent_requested_reviews(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "work carefully", "C1")

        prompt = build_task_prompt(agent, task)

        self.assertIn("somebody review", prompt)
        self.assertIn("stop and wait", prompt)
        self.assertIn("that request is mandatory", prompt)
        self.assertIn("Do not continue implementation work", prompt)
        self.assertIn("If routing fails or no review context arrives", prompt)

    def test_build_task_prompt_instructs_periodic_slack_updates(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "work carefully", "C1")

        prompt = build_task_prompt(agent, task)

        self.assertIn("at least every 5 minutes", prompt)
        self.assertIn("Slack-visible progress update", prompt)
        self.assertIn(AGENT_ROSTER_STATUS_SIGNAL_PREFIX, prompt)
        self.assertIn("broader goal and current phase", prompt)

    def test_build_task_prompt_instructs_direct_question_handling(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "work carefully", "C1")

        prompt = build_task_prompt(agent, task)

        self.assertIn("Direct Slack questions are not optional", prompt)
        self.assertIn("answer it explicitly in Slack", prompt)

    def test_build_task_prompt_instructs_timer_signal(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "work carefully", "C1")

        prompt = build_task_prompt(agent, task)

        self.assertIn(AGENT_TIMER_SIGNAL_PREFIX, prompt)
        self.assertIn("do not rely on terminal sleeps", prompt)
        self.assertIn("resumes the same agent", prompt)

    def test_build_task_prompt_instructs_reaction_signal(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "work carefully", "C1")

        prompt = build_task_prompt(agent, task)

        self.assertIn(AGENT_REACTION_SIGNAL_PREFIX, prompt)
        self.assertIn("from the user or from another agent", prompt)
        self.assertIn("not on every message", prompt)

    def test_build_task_prompt_prefers_slackgentic_pr_mcp(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "open the PR", "C1")

        prompt = build_task_prompt(agent, task)

        self.assertIn("create_pull_request", prompt)
        self.assertIn("MCP tool", prompt)

    def test_build_task_prompt_instructs_dangerous_mode_no_approval_requests(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "merge and deploy", "C1")
        task = _with_permission_mode(task, PermissionMode.DANGEROUS)

        prompt = build_task_prompt(agent, task)

        self.assertIn("Dangerous mode is enabled", prompt)
        self.assertIn("Do not request Slack approval or host-tool escalation", prompt)
        self.assertIn("no-approval permissions", prompt)

    def test_build_task_prompt_instructs_thread_done_signal(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "work carefully", "C1")

        prompt = build_task_prompt(agent, task)

        self.assertIn(AGENT_THREAD_DONE_SIGNAL, prompt)
        self.assertIn("marks the whole thread done", prompt)

    def test_build_task_prompt_instructs_named_session_agent_callback_format(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "offer options to @nell", "C1")

        prompt = build_task_prompt(agent, task)

        self.assertIn("separate final paragraph", prompt)
        self.assertIn("start of its own final paragraph", prompt)
        self.assertIn("plain text with no backticks", prompt)
        self.assertIn("@nell pick one before I proceed", prompt)
        self.assertIn("Do not put that callback handle inline", prompt)
        self.assertNotIn("`@handle`", prompt)

    def test_build_task_prompt_sanitizes_stored_slack_user_ids(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = replace(
            create_agent_task(agent, "continue", "C1"),
            metadata={
                "thread_context": (
                    "U0ARHEAU9B3: please check <@U0ARHEAU9B3>\nReviewer: @U0ARHEAU9B3 once ready"
                )
            },
        )

        prompt = build_task_prompt(agent, task)

        self.assertIn("Slack user: please check Slack user", prompt)
        self.assertIn("Reviewer: Slack user once ready", prompt)
        self.assertNotIn("U0ARHEAU9B3", prompt)

    def test_macos_tcc_protected_cwd_detects_home_and_volume_paths(self):
        home = Path("/Users/test")
        with patch("agent_harness.runtime.tasks.platform.system", return_value="Darwin"):
            self.assertIn(
                "Documents",
                _macos_tcc_protected_cwd_issue(home / "Documents" / "repo", home=home) or "",
            )
            self.assertIn(
                "mounted volume",
                _macos_tcc_protected_cwd_issue(Path("/Volumes/External/repo"), home=home) or "",
            )
            self.assertIsNone(_macos_tcc_protected_cwd_issue(home / "code" / "repo", home=home))

    def test_runtime_refuses_macos_tcc_protected_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = Store(home / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "work in documents", "C1")
                store.upsert_agent_task(task)
                launched = []

                def process_factory(request):
                    launched.append(request)
                    return OneShotProcess(request)

                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(default_cwd=home / "Documents" / "repo"),
                    process_factory=process_factory,
                    home=home,
                )

                with patch("agent_harness.runtime.tasks.platform.system", return_value="Darwin"):
                    started = runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))

                persisted = store.get_agent_task(task.task_id)
                assert persisted is not None
                self.assertFalse(started)
                self.assertFalse(launched)
                self.assertEqual(persisted.status, AgentTaskStatus.CANCELLED)
                self.assertIn("macOS-protected Documents", runtime.gateway.replies[-1])
            finally:
                store.close()

    def test_runtime_can_opt_into_macos_tcc_protected_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = Store(home / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "work in documents", "C1")
                store.upsert_agent_task(task)
                launched = []

                def process_factory(request):
                    launched.append(request)
                    return OneShotProcess(request)

                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(
                        default_cwd=home / "Documents" / "repo",
                        allow_macos_tcc_protected_paths=True,
                    ),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                    home=home,
                )

                with patch("agent_harness.runtime.tasks.platform.system", return_value="Darwin"):
                    started = runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))

                self.assertTrue(started)
                self.assertEqual(len(launched), 1)
                runtime.stop_all_running_tasks(status=AgentTaskStatus.CANCELLED)
            finally:
                store.close()

    def test_runtime_passes_configured_repo_root_to_codex_safe_auto(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repos"
            project = root / "example-project"
            project.mkdir(parents=True)
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                store.set_setting("slack.repo_root", str(root))
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "in example-project update docs", "C1")
                store.upsert_agent_task(task)
                launched = []

                def process_factory(request):
                    launched.append(request)
                    return OneShotProcess(request)

                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(default_cwd=Path("/workspace/repos")),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )

                self.assertTrue(runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001")))

                self.assertEqual(launched[0].cwd, project)
                self.assertEqual(launched[0].safe_auto_extra_roots, (root.resolve(),))
                runtime.stop_all_running_tasks(status=AgentTaskStatus.CANCELLED)
            finally:
                store.close()

    def test_runtime_passes_configured_repo_root_to_claude_safe_auto(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repos"
            project = root / "example-project"
            project.mkdir(parents=True)
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                store.set_setting("slack.repo_root", str(root))
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "in example-project update docs", "C1")
                store.upsert_agent_task(task)
                launched = []

                def process_factory(request):
                    launched.append(request)
                    return OneShotProcess(request)

                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(default_cwd=Path("/workspace/repos")),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )

                self.assertTrue(runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001")))

                self.assertEqual(launched[0].cwd, project)
                self.assertEqual(launched[0].safe_auto_extra_roots, (root.resolve(),))
                runtime.stop_all_running_tasks(status=AgentTaskStatus.CANCELLED)
            finally:
                store.close()

    def test_runtime_passes_claude_effort_from_user_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / "repo"
            project.mkdir()
            settings_dir = home / ".claude"
            settings_dir.mkdir()
            (settings_dir / "settings.json").write_text(json.dumps({"effortLevel": "xhigh"}))
            store = Store(home / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "work carefully", "C1")
                store.upsert_agent_task(task)
                launched = []

                def process_factory(request):
                    launched.append(request)
                    return OneShotProcess(request)

                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(default_cwd=project),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                    home=home,
                )

                self.assertTrue(runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001")))

                self.assertEqual(launched[0].claude_effort, "xhigh")
                runtime.stop_all_running_tasks(status=AgentTaskStatus.CANCELLED)
            finally:
                store.close()

    def test_runtime_prefers_project_claude_effort_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / "repo"
            (home / ".claude").mkdir()
            (home / ".claude" / "settings.json").write_text(json.dumps({"effortLevel": "low"}))
            (project / ".claude").mkdir(parents=True)
            (project / ".claude" / "settings.json").write_text(json.dumps({"effortLevel": "xhigh"}))
            store = Store(home / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "work carefully", "C1")
                store.upsert_agent_task(task)
                launched = []

                def process_factory(request):
                    launched.append(request)
                    return OneShotProcess(request)

                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(default_cwd=project),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                    home=home,
                )

                self.assertTrue(runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001")))

                self.assertEqual(launched[0].claude_effort, "xhigh")
                runtime.stop_all_running_tasks(status=AgentTaskStatus.CANCELLED)
            finally:
                store.close()

    def test_runtime_defaults_claude_effort_when_settings_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / "repo"
            project.mkdir()
            store = Store(home / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "work carefully", "C1")
                store.upsert_agent_task(task)
                launched = []

                def process_factory(request):
                    launched.append(request)
                    return OneShotProcess(request)

                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(default_cwd=project),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                    home=home,
                )

                self.assertTrue(runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001")))

                self.assertEqual(launched[0].claude_effort, "xhigh")
                runtime.stop_all_running_tasks(status=AgentTaskStatus.CANCELLED)
            finally:
                store.close()

    def test_runtime_defaults_claude_effort_when_settings_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / "repo"
            settings_dir = home / ".claude"
            settings_dir.mkdir()
            (settings_dir / "settings.json").write_text(json.dumps({"effortLevel": "unknown"}))
            store = Store(home / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "work carefully", "C1")
                store.upsert_agent_task(task)
                launched = []

                def process_factory(request):
                    launched.append(request)
                    return OneShotProcess(request)

                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(default_cwd=project),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                    home=home,
                )

                self.assertTrue(runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001")))

                self.assertEqual(launched[0].claude_effort, "xhigh")
                runtime.stop_all_running_tasks(status=AgentTaskStatus.CANCELLED)
            finally:
                store.close()

    def test_requested_repo_cwd_uses_named_sibling_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default = root
            sample_app = root / "sample-app"
            sample_app.mkdir()

            cwd = _requested_repo_cwd("in sample-app summarize the test command", default)

            self.assertEqual(cwd, sample_app)

    def test_requested_repo_cwd_ignores_missing_root(self):
        missing = Path("/tmp/slackgentic-missing-root-for-test")

        self.assertEqual(_requested_repo_cwd("in sample-app summarize tests", missing), missing)

    def test_command_config_accepts_env_aliases(self):
        config = AgentCommandConfig.model_validate(
            {
                "SLACKGENTIC_CODEX_BINARY": "/tmp/codex",
                "SLACKGENTIC_CLAUDE_BINARY": "/tmp/claude",
                "SLACKGENTIC_DEFAULT_CWD": "/tmp",
                "SLACKGENTIC_DANGEROUS_BY_DEFAULT": "true",
                "SLACKGENTIC_AGENT_START_TIMEOUT_SECONDS": "5",
                "SLACKGENTIC_AGENT_PROGRESS_TIMEOUT_SECONDS": "10",
                "SLACKGENTIC_AGENT_STALL_TIMEOUT_SECONDS": "20",
                "SLACKGENTIC_AGENT_STALL_RECOVERY_ATTEMPTS": "1",
            }
        )

        self.assertEqual(config.codex_binary, "/tmp/codex")
        self.assertEqual(config.claude_binary, "/tmp/claude")
        self.assertEqual(config.default_cwd, Path("/tmp"))
        self.assertTrue(config.dangerous_by_default)
        self.assertEqual(config.agent_start_timeout_seconds, 5)
        self.assertEqual(config.agent_progress_timeout_seconds, 10)
        self.assertEqual(config.agent_stall_timeout_seconds, 20)
        self.assertEqual(config.agent_stall_recovery_attempts, 1)

    def test_terminal_output_cleanup_removes_osc_controls(self):
        cleaned = _clean_terminal_output("\x1b]10;?\x1b\\visible\x1b[31m red\x1b[0m")

        self.assertEqual(cleaned, "visible red")

    def test_codex_exec_json_output_renders_agent_message(self):
        chunks, buffer = _process_output_chunks(
            Provider.CODEX,
            (
                '{"type":"thread.started","thread_id":"t1"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}\n'
                '{"type":"turn.completed","usage":{"total_tokens":10}}\n'
            ),
        )

        self.assertEqual(chunks, ["OK"])
        self.assertEqual(buffer, "")

    def test_codex_exec_json_output_renders_event_msg_agent_message(self):
        chunks, buffer = _process_output_chunks(
            Provider.CODEX,
            ('{"type":"event_msg","payload":{"type":"agent_message","message":"Visible final"}}\n'),
        )

        self.assertEqual(chunks, ["Visible final"])
        self.assertEqual(buffer, "")

    def test_codex_exec_json_output_renders_response_item_assistant_message(self):
        chunks, buffer = _process_output_chunks(
            Provider.CODEX,
            (
                '{"type":"response_item","payload":{"type":"message",'
                '"role":"assistant","content":[{"type":"output_text",'
                '"text":"Visible response"}]}}\n'
            ),
        )

        self.assertEqual(chunks, ["Visible response"])
        self.assertEqual(buffer, "")

    def test_codex_exec_json_output_hides_tool_events(self):
        chunks, buffer = _process_output_chunks(
            Provider.CODEX,
            (
                '{"type":"item.completed","item":{"type":"command_execution",'
                '"command":"pwd","output":"/tmp/repo"}}\n'
                '{"type":"item.completed","item":{"type":"tool_call","name":"apply_patch"}}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}\n'
            ),
        )

        self.assertEqual(chunks, ["Done"])
        self.assertEqual(buffer, "")

    def test_codex_exec_json_output_hides_tool_transport_errors(self):
        chunks, buffer = _process_output_chunks(
            Provider.CODEX,
            (
                "2026-05-14T03:13:23.414188Z ERROR "
                "codex_core::tools::router: error=write_stdin failed: "
                "Unknown process id 32391\n"
                '{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}\n'
            ),
        )

        self.assertEqual(chunks, ["Done"])
        self.assertEqual(buffer, "")

    def test_codex_exec_json_output_hides_raw_tool_result_fragments(self):
        chunks, buffer = _process_output_chunks(
            Provider.CODEX,
            (
                '":"        gateway.post_thread_reply(\\n'
                "    secret tool output\\n"
                '","exit_code":0,"status":"completed"}}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}\n'
            ),
        )

        self.assertEqual(chunks, ["Done"])
        self.assertEqual(buffer, "")

    def test_codex_exec_json_output_hides_plain_non_json_noise(self):
        chunks, buffer = _process_output_chunks(
            Provider.CODEX,
            (
                "def _shorten(value: str, limit: int) -> str:\n"
                '    cleaned = " ".join(value.split())\n'
                "    return cleaned\n"
                '{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}\n'
            ),
        )

        self.assertEqual(chunks, ["Done"])
        self.assertEqual(buffer, "")

    def test_codex_exec_json_output_renders_plain_error_lines(self):
        chunks, buffer = _process_output_chunks(
            Provider.CODEX,
            "Error: Codex failed to start\n",
        )

        self.assertEqual(chunks, ["Error: Codex failed to start"])
        self.assertEqual(buffer, "")

    def test_codex_exec_json_output_hides_malformed_json_records(self):
        chunks, buffer = _process_output_chunks(
            Provider.CODEX,
            '{"type":"item.completed","item":{"type":"command_execution","output":"unterminated\n',
            final=True,
        )

        self.assertEqual(chunks, [])
        self.assertEqual(buffer, "")

    def test_codex_exec_json_output_buffers_partial_lines(self):
        chunks, buffer = _process_output_chunks(
            Provider.CODEX,
            '{"type":"item.completed","item":{"type":"agent_message","text":"O',
        )
        self.assertEqual(chunks, [])

        chunks, buffer = _process_output_chunks(
            Provider.CODEX,
            'K"}}\n',
            buffer,
        )

        self.assertEqual(chunks, ["OK"])
        self.assertEqual(buffer, "")

    def test_claude_json_output_renders_final_result_only(self):
        chunks, buffer = _process_output_chunks(
            Provider.CLAUDE,
            (
                '{"type":"system","subtype":"init"}\n'
                '{"type":"rate_limit_event","rate_limit_info":{"status":"allowed"}}\n'
                '{"type":"assistant","message":{"content":[{"type":"thinking","thinking":""}]}}\n'
                '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read"}]}}\n'
                '{"type":"user","message":{"content":'
                '[{"type":"tool_result","content":"secret"}]}}\n'
                '{"type":"result","subtype":"success","is_error":false,"result":"Final"}\n'
            ),
        )

        self.assertEqual(chunks, ["Final"])
        self.assertEqual(buffer, "")

    def test_claude_json_output_renders_assistant_text_before_tool_result(self):
        chunks, buffer = _process_output_chunks(
            Provider.CLAUDE,
            (
                '{"type":"assistant","message":{"content":['
                '{"type":"text","text":"Interim artifact list"},'
                '{"type":"artifact","title":"notes.md","content":"extra"},'
                '{"type":"tool_use","name":"Read"}]}}\n'
                '{"type":"result","subtype":"success","is_error":false,"result":"Final"}\n'
            ),
        )

        self.assertEqual(chunks, ["Interim artifact list", "Final"])
        self.assertEqual(buffer, "")

    def test_claude_json_output_renders_text_before_tool_use(self):
        chunks, buffer = _process_output_chunks(
            Provider.CLAUDE,
            (
                '{"type":"assistant","message":{"content":['
                '{"type":"text","text":"Not stuck - continuing now."},'
                '{"type":"tool_use","name":"Bash"}],'
                '"stop_reason":"tool_use"}}\n'
            ),
        )

        self.assertEqual(chunks, ["Not stuck - continuing now."])
        self.assertEqual(buffer, "")

    def test_claude_json_output_skips_synthetic_no_response_marker(self):
        # Claude's CLI emits a synthetic assistant record whenever a resume
        # has nothing to say. The runtime stream renderer must NOT surface
        # that synthetic text — otherwise the user sees the bot post
        # "No response requested." in their Slack thread out of nowhere.
        chunks, buffer = _process_output_chunks(
            Provider.CLAUDE,
            (
                '{"type":"assistant","message":{"model":"<synthetic>",'
                '"content":[{"type":"text","text":"No response requested."}]}}\n'
                '{"type":"result","subtype":"success","is_error":false,"result":"Done"}\n'
            ),
        )

        self.assertEqual(chunks, ["Done"])
        self.assertEqual(buffer, "")

    def test_codex_long_message_splits_at_paragraph_boundary(self):
        first = "Short opener."
        second = "Body paragraph." + (" filler" * 400)
        message = f"{first}\n\n{second}"
        line = json.dumps(
            {
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": message},
            }
        )

        chunks, buffer = _process_output_chunks(Provider.CODEX, f"{line}\n")

        self.assertEqual(buffer, "")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0], first)
        self.assertTrue(chunks[1].startswith("Body paragraph."))

    def test_claude_json_output_buffers_partial_lines(self):
        chunks, buffer = _process_output_chunks(
            Provider.CLAUDE,
            '{"type":"result","subtype":"success","is_error":false,"result":"O',
        )
        self.assertEqual(chunks, [])

        chunks, buffer = _process_output_chunks(
            Provider.CLAUDE,
            'K"}\n',
            buffer,
        )

        self.assertEqual(chunks, ["OK"])
        self.assertEqual(buffer, "")

    def test_claude_json_output_hides_permission_denial_result(self):
        line = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "The command requires approval.",
                "permission_denials": [
                    {
                        "tool_name": "Bash",
                        "tool_input": {"command": "gh auth status"},
                    }
                ],
            }
        )
        chunks, buffer = _process_output_chunks(Provider.CLAUDE, f"{line}\n")

        self.assertEqual(chunks, [])
        self.assertEqual(buffer, "")

    def test_claude_permission_denials_are_parsed_for_exact_bash_allowlist(self):
        denial = {
            "tool_name": "Bash",
            "tool_input": {
                "command": "gh auth status",
                "description": "Check GitHub auth status",
            },
        }
        line = json.dumps(
            {
                "type": "result",
                "permission_denials": [denial],
            }
        )

        denials, buffer = _claude_permission_denials(f"{line}\n")

        self.assertEqual(buffer, "")
        self.assertEqual(denials, [denial])
        self.assertEqual(_allowed_tool_for_claude_denial(denial), "Bash(gh auth status)")

    def test_claude_permission_denial_uses_safe_gh_pattern_for_comma_args(self):
        denial = {
            "tool_name": "Bash",
            "tool_input": {
                "command": ("gh pr view 6 --repo unbounded-ai/slackgentic-team --json number,title")
            },
        }

        self.assertEqual(_allowed_tool_for_claude_denial(denial), "Bash(gh pr view:*)")

    def test_claude_permission_denial_allows_git_log_after_cd(self):
        denial = {
            "tool_name": "Bash",
            "tool_input": {"command": "cd /workspace/repos/sample-app && git log --oneline -30"},
        }

        self.assertEqual(
            _allowed_tool_for_claude_denial(denial),
            "Bash(cd /workspace/repos/sample-app && git log --oneline -30)",
        )
        self.assertEqual(
            _allowed_tools_for_claude_denial(denial),
            (
                "Bash(cd /workspace/repos/sample-app && git log --oneline -30)",
                "Bash(git log:*)",
                "Bash(git log *)",
            ),
        )

    def test_claude_permission_denial_allows_git_log_after_dash_c(self):
        denial = {
            "tool_name": "Bash",
            "tool_input": {
                "command": ("git -C /workspace/repos/slackgentic-team log --oneline -1")
            },
        }

        allowed_tools = _allowed_tools_for_claude_denial(denial)

        self.assertEqual(
            _allowed_tool_for_claude_denial(denial),
            ("Bash(git -C /workspace/repos/slackgentic-team log --oneline -1)"),
        )
        self.assertIn(
            "Bash(git -C /workspace/repos/slackgentic-team log:*)",
            allowed_tools,
        )
        self.assertIn("Bash(git log:*)", allowed_tools)

    def test_claude_permission_denial_allows_git_pull(self):
        denial = {
            "tool_name": "Bash",
            "tool_input": {"command": "git -C /workspace/repos/example-project pull origin main"},
        }

        allowed_tools = _allowed_tools_for_claude_denial(denial)

        self.assertEqual(
            _allowed_tool_for_claude_denial(denial),
            "Bash(git -C /workspace/repos/example-project pull origin main)",
        )
        self.assertIn("Bash(git pull:*)", allowed_tools)
        self.assertIn("Bash(git -C /workspace/repos/example-project pull:*)", allowed_tools)

    def test_claude_permission_denial_allows_gh_pr_create(self):
        denial = {
            "tool_name": "Bash",
            "tool_input": {"command": "gh pr create --title update --body summary"},
        }

        allowed_tools = _allowed_tools_for_claude_denial(denial)

        self.assertEqual(
            _allowed_tool_for_claude_denial(denial),
            "Bash(gh pr create --title update --body summary)",
        )
        self.assertIn("Bash(gh pr create:*)", allowed_tools)
        self.assertIn("Bash(gh pr create *)", allowed_tools)

    def test_claude_permission_denial_allows_git_config_author_sequence(self):
        command = (
            "git -C /workspace/repos/example-project config user.name; "
            "git -C /workspace/repos/example-project config user.email"
        )
        denial = {"tool_name": "Bash", "tool_input": {"command": command}}

        allowed_tools = _allowed_tools_for_claude_denial(denial)

        self.assertEqual(_allowed_tool_for_claude_denial(denial), f"Bash({command})")
        self.assertEqual(allowed_tools, (f"Bash({command})",))

    def test_claude_permission_denial_allows_file_edit_tools(self):
        self.assertEqual(_allowed_tool_for_claude_denial({"tool_name": "Edit"}), "Edit")
        self.assertEqual(_allowed_tool_for_claude_denial({"tool_name": "Write"}), "Write")

    def test_claude_permission_denial_allows_approved_gh_write_exactly(self):
        denial = {
            "tool_name": "Bash",
            "tool_input": {"command": "gh pr merge 6 --squash"},
        }

        allowed_tools = _allowed_tools_for_claude_denial(denial)

        self.assertEqual(_allowed_tool_for_claude_denial(denial), "Bash(gh pr merge 6 --squash)")
        self.assertNotIn("Bash(gh pr merge:*)", allowed_tools)

    def test_claude_permission_denial_allows_multiline_git_commit_by_prefix(self):
        denial = {
            "tool_name": "Bash",
            "tool_input": {
                "command": ClaudeMultilineCommitPermissionDeniedProcess.command,
                "description": ClaudeMultilineCommitPermissionDeniedProcess.description,
            },
        }

        allowed_tools = _allowed_tools_for_claude_denial(denial)

        self.assertNotIn(
            f"Bash({ClaudeMultilineCommitPermissionDeniedProcess.command.strip()})",
            allowed_tools,
        )
        self.assertIn(
            "Bash(git -C /workspace/repos/sample-app commit:*)",
            allowed_tools,
        )

    def test_claude_session_permission_allows_bash_command_family(self):
        denial = {
            "tool_name": "Bash",
            "tool_input": {"command": "git -C /workspace/repos/sample-app push -u origin feature"},
        }

        self.assertEqual(
            _allowed_session_tools_for_claude_denial(denial),
            ("Bash(git:*)", "Bash(git *)"),
        )

    def test_append_allowed_tool_preserves_prior_approvals(self):
        allowed = _append_allowed_tool(("Bash(gh pr list:*)",), "Bash(gh pr view:*)")

        self.assertEqual(allowed, ("Bash(gh pr list:*)", "Bash(gh pr view:*)"))
        self.assertEqual(_append_allowed_tool(allowed, "Bash(gh pr view:*)"), allowed)

    def test_claude_missing_resume_error_is_detected(self):
        result = json.dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "session_id": "bogus-error-session",
                "errors": ["No conversation found with session ID: old-session"],
            }
        )

        missing, buffer = _claude_missing_resume_session(f"{result}\n")
        session_id, _ = _session_id_from_output(Provider.CLAUDE, f"{result}\n")

        self.assertTrue(missing)
        self.assertEqual(buffer, "")
        self.assertIsNone(session_id)

    def test_agent_control_signal_is_stripped_from_visible_text(self):
        visible, signals = _extract_agent_control_signals(
            f"Sounds good.\n{AGENT_THREAD_DONE_SIGNAL}\n"
        )

        self.assertEqual(visible, "Sounds good.")
        self.assertEqual(signals, [AGENT_THREAD_DONE_SIGNAL])

    def test_agent_timer_signal_is_stripped_from_visible_text(self):
        visible, signals = _extract_agent_control_signals(
            f"Wakeup scheduled.\n{AGENT_TIMER_SIGNAL_PREFIX}10m | Re-check CI.\n"
        )

        self.assertEqual(visible, "Wakeup scheduled.")
        self.assertEqual(signals, [f"{AGENT_TIMER_SIGNAL_PREFIX}10m | Re-check CI."])

    def test_agent_roster_status_signal_is_stripped_from_visible_text(self):
        signal = f"{AGENT_ROSTER_STATUS_SIGNAL_PREFIX}PR merge and daemon restart: running E2E"

        visible, signals = _extract_agent_control_signals(f"Still testing.\n{signal}\n")

        self.assertEqual(visible, "Still testing.")
        self.assertEqual(signals, [signal])
        self.assertEqual(
            parse_agent_roster_status_signal(signal),
            "PR merge and daemon restart: running E2E",
        )

    def test_agent_reaction_signal_is_stripped_from_visible_text(self):
        signal = f"{AGENT_REACTION_SIGNAL_PREFIX}:thumbsup:"

        visible, signals = _extract_agent_control_signals(f"Done.\n{signal}\n")

        self.assertEqual(visible, "Done.")
        self.assertEqual(signals, [signal])
        self.assertEqual(parse_agent_reaction_signal(signal), "thumbsup")

    def test_agent_deferred_signal_is_stripped_from_visible_text(self):
        signal = f'{AGENT_DEFERRED_SIGNAL_PREFIX}{{"task":"follow up","depends_on":[]}}'

        visible, signals = _extract_agent_control_signals(f"Deferred.\n{signal}\n")

        self.assertEqual(visible, "Deferred.")
        self.assertEqual(signals, [signal])

    def test_parse_agent_timer_signal_accepts_delay(self):
        now = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)

        parsed = parse_agent_timer_signal(
            f"{AGENT_TIMER_SIGNAL_PREFIX}10m | Re-check CI.",
            now=now,
        )

        self.assertIsNone(parsed.error)
        self.assertIsNotNone(parsed.request)
        self.assertEqual(parsed.request.due_at, now + timedelta(minutes=10))
        self.assertEqual(parsed.request.prompt, "Re-check CI.")

    def test_parse_agent_timer_signal_accepts_json(self):
        now = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)

        parsed = parse_agent_timer_signal(
            f'{AGENT_TIMER_SIGNAL_PREFIX}{{"delay_seconds": 3, "prompt": "Wake up"}}',
            now=now,
        )

        self.assertIsNone(parsed.error)
        self.assertIsNotNone(parsed.request)
        self.assertEqual(parsed.request.due_at, now + timedelta(seconds=3))
        self.assertEqual(parsed.request.prompt, "Wake up")

    def test_tool_transport_leak_lines_are_stripped_from_visible_text(self):
        visible, signals = _extract_agent_control_signals(
            "Working.\n"
            "2026-05-14T03:13:23.414188Z ERROR "
            "codex_core::tools::router: error=write_stdin failed: "
            "Unknown process id 32391\n"
            "Still working.\n"
        )

        self.assertEqual(visible, "Working.\nStill working.")
        self.assertEqual(signals, [])

    def test_codex_transcript_fallback_uses_latest_agent_message_since_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            session_id = "session-fallback"
            path = (
                home
                / ".codex"
                / "sessions"
                / "2026"
                / "04"
                / "29"
                / f"rollout-2026-04-29T01-17-28-{session_id}.jsonl"
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                "\n".join(
                    [
                        '{"timestamp":"2026-04-29T05:00:00Z","type":"event_msg",'
                        '"payload":{"type":"agent_message","message":"old"}}',
                        '{"timestamp":"2026-04-29T05:01:00Z","type":"event_msg",'
                        '"payload":{"type":"agent_message","message":"new"}}',
                    ]
                )
                + "\n"
            )

            message = _latest_codex_transcript_message(
                session_id,
                home,
                since=datetime.fromisoformat("2026-04-29T05:00:30+00:00"),
            )

            self.assertEqual(message, "new")

    def test_runtime_recovers_only_transcript_messages_added_after_worker_start(self):
        # A worker resuming a session pre-seeds its dedup set with the existing
        # transcript so the recovery branch does not re-post chunks a prior
        # worker already delivered. Recovery still has to fire for messages
        # that genuinely appear during this worker's lifetime but slipped past
        # streaming (transcript-only emissions, or a race between the last
        # chunk and process exit).
        class TranscriptAppendingProcess(OneShotProcess):
            def __init__(self, request):
                super().__init__(request)
                self.transcript_path: Path | None = None
                self.late_message_ts: str | None = None
                self.late_message: str | None = None
                self._appended = False

            def read_available(self, max_reads=20, timeout=0.05):
                if (
                    not self._appended
                    and self.transcript_path is not None
                    and self.late_message is not None
                    and self.late_message_ts is not None
                ):
                    payload = json.dumps(
                        {
                            "timestamp": self.late_message_ts,
                            "type": "event_msg",
                            "payload": {
                                "type": "agent_message",
                                "message": self.late_message,
                            },
                        }
                    )
                    with self.transcript_path.open("a") as fh:
                        fh.write(payload + "\n")
                    self._appended = True
                return ""

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            home = Path(tmp)
            session_id = "session-recover-all"
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "recover transcript", "C1"),
                    session_provider=Provider.CODEX,
                    session_id=session_id,
                )
                store.upsert_agent_task(task)
                path = (
                    home
                    / ".codex"
                    / "sessions"
                    / "2026"
                    / "05"
                    / "20"
                    / f"rollout-2026-05-20T10-00-00-{session_id}.jsonl"
                )
                path.parent.mkdir(parents=True)
                historical_ts = (task.created_at + timedelta(seconds=1)).isoformat()
                late_ts = (task.created_at + timedelta(seconds=5)).isoformat()
                path.write_text(
                    json.dumps(
                        {
                            "timestamp": historical_ts,
                            "type": "event_msg",
                            "payload": {
                                "type": "agent_message",
                                "message": "historical update already in Slack",
                            },
                        }
                    )
                    + "\n"
                )
                processes: list[TranscriptAppendingProcess] = []

                def factory(request):
                    process = TranscriptAppendingProcess(request)
                    process.transcript_path = path
                    process.late_message_ts = late_ts
                    process.late_message = "somebody review this handoff"
                    processes.append(process)
                    return process

                gateway = FakeGateway()
                seen = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=factory,
                    poll_seconds=0.01,
                    on_agent_message=lambda task, agent, thread, text, message_ts: (
                        seen.append(text) or True
                    ),
                    home=home,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    if seen:
                        break
                    time.sleep(0.01)

                # Only the late message — written after the worker started —
                # should reach Slack. The historical entry was already seeded
                # into observed_agent_messages and must not be re-posted.
                self.assertEqual(seen, ["somebody review this handoff"])
                self.assertEqual(gateway.replies, seen)
            finally:
                store.close()

    def test_runtime_does_not_post_process_lifecycle_message_on_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "finish quietly", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=OneShotProcess,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    if gateway.replies:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Done"])
                self.assertEqual(
                    store.get_agent_task(task.task_id).status,
                    AgentTaskStatus.ACTIVE,
                )
                for _ in range(50):
                    current = store.get_agent_task(task.task_id)
                    if current and MANAGED_RUN_STARTED_METADATA_KEY not in current.metadata:
                        break
                    time.sleep(0.01)
                self.assertNotIn(
                    MANAGED_RUN_STARTED_METADATA_KEY,
                    store.get_agent_task(task.task_id).metadata,
                )
            finally:
                store.close()

    def test_should_resume_managed_run_requires_marker(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "no marker", "C1")

        self.assertFalse(should_resume_managed_run(task))

    def test_should_resume_managed_run_rejects_stale_marker(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        now = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
        stale = now - MANAGED_RUN_MAX_RESUME_AGE - timedelta(seconds=30)
        task = create_agent_task(agent, "stale marker", "C1")
        task = replace(
            task,
            metadata={MANAGED_RUN_STARTED_METADATA_KEY: stale.isoformat()},
        )

        self.assertFalse(should_resume_managed_run(task, now=now))

    def test_should_resume_managed_run_accepts_fresh_marker(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        now = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
        fresh = now - timedelta(seconds=30)
        task = create_agent_task(agent, "fresh marker", "C1")
        task = replace(
            task,
            metadata={MANAGED_RUN_STARTED_METADATA_KEY: fresh.isoformat()},
        )

        self.assertTrue(should_resume_managed_run(task, now=now))

    def test_should_resume_managed_run_caps_attempts(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        now = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
        fresh = now - timedelta(seconds=10)
        task = create_agent_task(agent, "too many attempts", "C1")
        task = replace(
            task,
            metadata={
                MANAGED_RUN_STARTED_METADATA_KEY: fresh.isoformat(),
                MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY: MANAGED_RUN_MAX_RESUMES,
            },
        )

        self.assertFalse(should_resume_managed_run(task, now=now))

    def test_resume_orphaned_task_bumps_attempts_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "resume me", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    metadata={
                        MANAGED_RUN_STARTED_METADATA_KEY: "2026-05-13T11:59:50+00:00",
                    },
                )
                store.upsert_agent_task(task)
                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=HoldingProcess,
                    poll_seconds=0.01,
                )

                runtime.resume_orphaned_task(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.thread"),
                )

                current = store.get_agent_task(task.task_id)
                assert current is not None
                self.assertEqual(managed_run_resume_attempts(current), 1)
                runtime.stop_task(task.task_id)
                cleared = store.get_agent_task(task.task_id)
                assert cleared is not None
                self.assertEqual(managed_run_resume_attempts(cleared), 0)
            finally:
                store.close()

    def test_runtime_stop_all_running_tasks_clears_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task_one = create_agent_task(agent, "stay alive", "C1")
                task_two = replace(
                    create_agent_task(agent, "also stay alive", "C1"),
                    thread_ts="171.000002",
                )
                store.upsert_agent_task(task_one)
                store.upsert_agent_task(task_two)
                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=HoldingProcess,
                    poll_seconds=0.01,
                )
                runtime.start_task(task_one, agent, SlackThreadRef("C1", "171.000001"))
                runtime.start_task(task_two, agent, SlackThreadRef("C1", "171.000002"))

                self.assertTrue(runtime.has_running_tasks())
                self.assertEqual(runtime.stop_all_running_tasks(), 2)

                self.assertFalse(runtime.has_running_tasks())
                for task in (task_one, task_two):
                    persisted = store.get_agent_task(task.task_id)
                    assert persisted is not None
                    self.assertNotIn(MANAGED_RUN_STARTED_METADATA_KEY, persisted.metadata)
                    self.assertEqual(persisted.status, AgentTaskStatus.CANCELLED)
            finally:
                store.close()

    def test_runtime_stop_all_running_tasks_can_preserve_restart_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "resume after restart", "C1")
                store.upsert_agent_task(task)
                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=HoldingProcess,
                    poll_seconds=0.01,
                )
                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))

                self.assertEqual(runtime.stop_all_running_tasks(status=None), 1)

                self.assertFalse(runtime.has_running_tasks())
                persisted = store.get_agent_task(task.task_id)
                assert persisted is not None
                self.assertIn(MANAGED_RUN_STARTED_METADATA_KEY, persisted.metadata)
                self.assertEqual(persisted.status, AgentTaskStatus.ACTIVE)
            finally:
                store.close()

    def test_post_agent_chunk_drops_output_after_stop_requested(self):
        # When the user explicitly releases a task (Done/Cancel button →
        # runtime.stop_task), any buffered child-process output that still
        # streams in before the worker thread exits must NOT reach Slack.
        # Previously the chunk loop kept posting until the next stop check.
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "do the thing", "C1")
                store.upsert_agent_task(task)
                thread = SlackThreadRef("C1", "171.release")
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=OneShotProcess,
                    poll_seconds=0.01,
                )
                running = RunningTask(
                    task=task,
                    agent=agent,
                    process=OneShotProcess(None),
                    thread=thread,
                    worker=threading.Thread(),
                )

                runtime._post_agent_chunk(running, "before release")
                self.assertEqual(gateway.replies, ["before release"])

                running.stop_requested = True
                runtime._post_agent_chunk(running, "after release — should not post")
                self.assertEqual(gateway.replies, ["before release"])

                running.stop_requested = False
                runtime._post_agent_chunk(running, "release cleared — posts again")
                self.assertEqual(
                    gateway.replies,
                    ["before release", "release cleared — posts again"],
                )
            finally:
                store.close()

    def test_runtime_stop_task_keeps_unjoined_worker_visible_until_exit(self):
        class BlockingReadProcess(OneShotProcess):
            def __init__(self, request):
                super().__init__(request)
                self.alive = True
                self.reading = threading.Event()
                self.release = threading.Event()

            def read_available(self, max_reads=20, timeout=0.05):
                self.reading.set()
                self.release.wait(timeout=5.0)
                return ""

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.alive = False

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            processes: list[BlockingReadProcess] = []

            def process_factory(request):
                process = BlockingReadProcess(request)
                processes.append(process)
                return process

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "stop carefully", "C1")
                store.upsert_agent_task(task)
                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )
                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))

                self.assertTrue(processes[0].reading.wait(timeout=1.0))
                with self.assertLogs("agent_harness.runtime.tasks", level="WARNING") as logs:
                    self.assertFalse(
                        runtime.stop_task(task.task_id, status=None, join_timeout=0.01)
                    )
                self.assertTrue(
                    any("managed task worker did not stop" in message for message in logs.output)
                )
                self.assertTrue(runtime.has_running_tasks())

                persisted = store.get_agent_task(task.task_id)
                assert persisted is not None
                self.assertIn(MANAGED_RUN_STARTED_METADATA_KEY, persisted.metadata)
                self.assertEqual(persisted.status, AgentTaskStatus.ACTIVE)

                processes[0].release.set()
                deadline = time.monotonic() + 1.0
                while runtime.has_running_tasks() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(runtime.has_running_tasks())
            finally:
                for process in processes:
                    process.release.set()
                store.close()

    def test_runtime_keeps_completed_worker_visible_until_done_callback_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            callback_started = threading.Event()
            release_callback = threading.Event()
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "finish slowly", "C1")
                store.upsert_agent_task(task)

                def on_task_done(task, agent, thread):
                    callback_started.set()
                    release_callback.wait(timeout=1.0)

                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=OneShotProcess,
                    poll_seconds=0.01,
                    on_task_done=on_task_done,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                self.assertTrue(callback_started.wait(timeout=1.0))
                self.assertTrue(runtime.has_running_tasks())

                release_callback.set()
                deadline = time.monotonic() + 1.0
                while runtime.has_running_tasks() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(runtime.has_running_tasks())
            finally:
                release_callback.set()
                store.close()

    def test_runtime_stop_all_uses_shared_join_budget(self):
        class BlockingReadProcess(OneShotProcess):
            instances: ClassVar[list["BlockingReadProcess"]] = []

            def __init__(self, request):
                super().__init__(request)
                self.alive = True
                self.reading = threading.Event()
                self.release = threading.Event()
                self.__class__.instances.append(self)

            def read_available(self, max_reads=20, timeout=0.05):
                self.reading.set()
                self.release.wait(timeout=5.0)
                return ""

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.alive = False

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            BlockingReadProcess.instances = []
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=2, claude_count=0)
                for agent in agents:
                    store.upsert_team_agent(agent)
                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=BlockingReadProcess,
                    poll_seconds=0.01,
                )
                for index, agent in enumerate(agents):
                    task = create_agent_task(agent, f"block {index}", "C1")
                    store.upsert_agent_task(task)
                    runtime.start_task(task, agent, SlackThreadRef("C1", f"171.00000{index}"))

                for process in BlockingReadProcess.instances:
                    self.assertTrue(process.reading.wait(timeout=1.0))

                start = time.monotonic()
                with self.assertLogs("agent_harness.runtime.tasks", level="WARNING"):
                    self.assertEqual(
                        runtime.stop_all_running_tasks(status=None, join_timeout=0.02),
                        0,
                    )
                self.assertLess(time.monotonic() - start, 0.15)
                self.assertTrue(runtime.has_running_tasks())
            finally:
                for process in BlockingReadProcess.instances:
                    process.release.set()
                deadline = time.monotonic() + 1.0
                while (
                    "runtime" in locals()
                    and runtime.has_running_tasks()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                store.close()

    def test_runtime_cancels_codex_run_that_never_starts_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "review a design", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                )
                store.upsert_agent_task(task)
                thread = SlackThreadRef("C1", "171.000001")
                store.upsert_managed_thread_task(task, thread)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=HoldingProcess,
                    poll_seconds=0.01,
                    codex_thread_start_timeout=timedelta(seconds=0),
                )
                running = RunningTask(
                    task=task,
                    agent=agent,
                    process=HoldingProcess(None),
                    thread=thread,
                    worker=threading.Thread(),
                )

                self.assertTrue(runtime._codex_thread_start_timed_out(running))
                runtime._handle_codex_thread_start_timeout(running)

                persisted = store.get_agent_task(task.task_id)
                assert persisted is not None
                self.assertEqual(persisted.status, AgentTaskStatus.CANCELLED)
                self.assertNotIn(MANAGED_RUN_STARTED_METADATA_KEY, persisted.metadata)
                self.assertIsNone(store.get_managed_thread_task("C1", "171.000001"))
                self.assertIn("did not finish starting Codex", gateway.replies[-1])
            finally:
                store.close()

    def test_runtime_cancels_claude_run_that_never_starts_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "review a design", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=HoldingProcess,
                    poll_seconds=0.01,
                    agent_start_timeout=timedelta(seconds=0),
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    current = store.get_agent_task(task.task_id)
                    if current and current.status == AgentTaskStatus.CANCELLED and gateway.replies:
                        break
                    time.sleep(0.01)

                persisted = store.get_agent_task(task.task_id)
                assert persisted is not None
                self.assertEqual(persisted.status, AgentTaskStatus.CANCELLED)
                self.assertIn("did not finish starting Claude", gateway.replies[-1])
            finally:
                store.close()

    def test_runtime_warns_without_interrupting_on_missing_visible_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            processes = []

            def process_factory(request):
                process = SessionThenHoldingProcess(request)
                processes.append(process)
                return process

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "run the long test suite", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                    agent_progress_timeout=timedelta(seconds=0.01),
                    agent_stall_timeout=timedelta(minutes=30),
                    max_stall_recoveries=1,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if any("leaving the run active" in reply for reply in gateway.replies):
                        break
                    time.sleep(0.01)

                self.assertEqual(len(processes), 1)
                self.assertTrue(processes[0].is_alive())
                self.assertIn("Working", gateway.replies)
                self.assertTrue(any("leaving the run active" in reply for reply in gateway.replies))
                current = store.get_agent_task(task.task_id)
                assert current is not None
                self.assertEqual(current.status, AgentTaskStatus.ACTIVE)
            finally:
                if "runtime" in locals():
                    runtime.stop_all_running_tasks(status=AgentTaskStatus.CANCELLED)
                store.close()

    def test_runtime_restarts_stalled_session_with_status_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []
            processes = []

            def process_factory(request):
                requests.append(request)
                process = (
                    SessionThenHoldingProcess(request)
                    if len(requests) == 1
                    else OneShotProcess(request)
                )
                processes.append(process)
                return process

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "open the pull request", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                    agent_progress_timeout=timedelta(minutes=30),
                    agent_stall_timeout=timedelta(seconds=0),
                    max_stall_recoveries=1,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if len(requests) >= 2 and gateway.replies and gateway.replies[-1] == "Done":
                        break
                    time.sleep(0.01)

                self.assertEqual(len(requests), 2)
                self.assertFalse(processes[0].is_alive())
                self.assertEqual(requests[1].resume_session_id, "codex-thread-1")
                self.assertIn("observed no activity", requests[1].prompt)
                self.assertIn(
                    "First post a concise Slack-visible status update", requests[1].prompt
                )
                self.assertIn("Original task: open the pull request", requests[1].prompt)
                self.assertIn("Working", gateway.replies)
                self.assertTrue(
                    any("restarted it with a status request" in reply for reply in gateway.replies)
                )
            finally:
                if "runtime" in locals():
                    runtime.stop_all_running_tasks(status=AgentTaskStatus.CANCELLED)
                store.close()

    def test_runtime_treats_transcript_updates_as_activity_for_stall_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            transcript = (
                home
                / ".codex"
                / "sessions"
                / "2026"
                / "05"
                / "19"
                / "rollout-codex-thread-activity.jsonl"
            )
            transcript.parent.mkdir(parents=True)
            transcript.write_text("{}\n")
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                task = replace(
                    create_agent_task(agent, "keep going", "C1"),
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-activity",
                )
                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=HoldingProcess,
                    agent_stall_timeout=timedelta(seconds=5),
                    home=home,
                )
                stale = time.monotonic() - 60
                running = RunningTask(
                    task=task,
                    agent=agent,
                    process=HoldingProcess(None),
                    thread=SlackThreadRef("C1", "171.000001"),
                    worker=threading.Thread(),
                    started_monotonic=stale,
                    last_output_monotonic=stale,
                    last_activity_monotonic=stale,
                    last_visible_message_monotonic=stale,
                )

                runtime._capture_transcript_activity(running)
                running.last_activity_monotonic = stale
                running.last_transcript_activity_check_monotonic = 0.0
                transcript.write_text("{}\n{}\n")
                runtime._capture_transcript_activity(running)

                self.assertFalse(runtime._managed_task_stall_timed_out(running))
            finally:
                store.close()

    def test_runtime_cancels_stalled_session_after_recovery_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "keep going", "C1"),
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-1",
                    metadata={MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY: 1},
                )
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=HoldingProcess,
                    poll_seconds=0.01,
                    agent_progress_timeout=timedelta(minutes=30),
                    agent_stall_timeout=timedelta(seconds=0),
                    max_stall_recoveries=1,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    current = store.get_agent_task(task.task_id)
                    if current and current.status == AgentTaskStatus.CANCELLED:
                        break
                    time.sleep(0.01)

                persisted = store.get_agent_task(task.task_id)
                assert persisted is not None
                self.assertEqual(persisted.status, AgentTaskStatus.CANCELLED)
                self.assertIn("repeatedly had no observed activity", gateway.replies[-1])
                self.assertNotIn(MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY, persisted.metadata)
            finally:
                store.close()

    def test_runtime_marks_managed_run_while_process_is_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "keep working", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=HoldingProcess,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))

                current = store.get_agent_task(task.task_id)
                self.assertIsNotNone(current)
                assert current is not None
                self.assertIn(MANAGED_RUN_STARTED_METADATA_KEY, current.metadata)

                runtime.stop_task(task.task_id)

                current = store.get_agent_task(task.task_id)
                self.assertIsNotNone(current)
                assert current is not None
                self.assertNotIn(MANAGED_RUN_STARTED_METADATA_KEY, current.metadata)
            finally:
                store.close()

    def test_runtime_interrupt_sends_escape_but_keeps_task_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "stop current run only", "C1")
                store.upsert_agent_task(task)
                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=HoldingProcess,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                running = runtime.running_tasks()[0]

                self.assertTrue(runtime.interrupt_task(task.task_id))
                self.assertTrue(runtime.has_running_tasks())
                self.assertEqual(running.process.interrupts, 1)
                self.assertTrue(running.process.is_alive())
                current = store.get_agent_task(task.task_id)
                assert current is not None
                self.assertEqual(current.status, AgentTaskStatus.ACTIVE)
                self.assertIn(MANAGED_RUN_STARTED_METADATA_KEY, current.metadata)

                runtime.stop_task(task.task_id)
            finally:
                store.close()

    def test_runtime_worker_failure_terminates_orphan_child_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "read will crash", "C1")
                store.upsert_agent_task(task)
                created: list[CrashingProcess] = []

                def factory(request):
                    process = CrashingProcess(request)
                    created.append(process)
                    return process

                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=factory,
                    poll_seconds=0.01,
                )

                logging.disable(logging.CRITICAL)
                try:
                    runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                    for _ in range(50):
                        current = store.get_agent_task(task.task_id)
                        if current and current.status == AgentTaskStatus.CANCELLED:
                            break
                        time.sleep(0.01)
                finally:
                    logging.disable(logging.NOTSET)

                self.assertEqual(len(created), 1)
                self.assertTrue(
                    created[0].terminated,
                    "crashed worker must terminate its child process to avoid leaking it to init",
                )
                self.assertFalse(created[0].is_alive())
            finally:
                store.close()

    def test_runtime_worker_failure_cancels_task_instead_of_leaving_it_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "post will fail", "C1")
                store.upsert_agent_task(task)
                runtime = ManagedTaskRuntime(
                    store,
                    FailingGateway(),
                    AgentCommandConfig(),
                    process_factory=OneShotProcess,
                    poll_seconds=0.01,
                )

                logging.disable(logging.CRITICAL)
                try:
                    runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                    for _ in range(50):
                        current = store.get_agent_task(task.task_id)
                        if current and current.status == AgentTaskStatus.CANCELLED:
                            break
                        time.sleep(0.01)
                finally:
                    logging.disable(logging.NOTSET)

                current = store.get_agent_task(task.task_id)
                self.assertIsNotNone(current)
                assert current is not None
                self.assertEqual(current.status, AgentTaskStatus.CANCELLED)
                self.assertIsNone(store.get_managed_thread_task("C1", "171.000001", agent.agent_id))
            finally:
                store.close()

    def test_runtime_posts_visible_chunks_with_agent_avatar_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "finish with avatar", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=OneShotProcess,
                    poll_seconds=0.01,
                    agent_icon_url=lambda item: f"https://example.com/{item.handle}.png",
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    if gateway.replies:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Done"])
                self.assertEqual(gateway.icon_urls, [f"https://example.com/{agent.handle}.png"])
            finally:
                store.close()

    def test_runtime_cancels_silent_process_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "finish silently", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=SilentProcess,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    current = store.get_agent_task(task.task_id)
                    if current and current.status == AgentTaskStatus.CANCELLED:
                        break
                    time.sleep(0.01)

                self.assertIn("finished without a Slack-visible response", gateway.replies[-1])
                self.assertEqual(
                    store.get_agent_task(task.task_id).status,
                    AgentTaskStatus.CANCELLED,
                )
            finally:
                store.close()

    def test_runtime_captures_codex_session_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "remember this", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=SessionIdProcess,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    current = store.get_agent_task(task.task_id)
                    if current and current.session_id == "codex-thread-1":
                        break
                    time.sleep(0.01)

                current = store.get_agent_task(task.task_id)
                self.assertIsNotNone(current)
                assert current is not None
                self.assertEqual(current.session_provider, Provider.CODEX)
                self.assertEqual(current.session_id, "codex-thread-1")
                for _ in range(50):
                    current = store.get_agent_task(task.task_id)
                    if current and current.status == AgentTaskStatus.ACTIVE:
                        break
                    time.sleep(0.01)
                for _ in range(50):
                    if gateway.replies:
                        break
                    time.sleep(0.01)
            finally:
                store.close()

    def test_runtime_resumes_task_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []

            def process_factory(request):
                requests.append(request)
                return OneShotProcess(request)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "continue", "C1")
                task = replace(
                    task,
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-1",
                )
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))

                self.assertEqual(requests[0].resume_session_id, "codex-thread-1")
                for _ in range(50):
                    current = store.get_agent_task(task.task_id)
                    if current and current.status == AgentTaskStatus.ACTIVE:
                        break
                    time.sleep(0.01)
                for _ in range(50):
                    if gateway.replies:
                        break
                    time.sleep(0.01)
            finally:
                store.close()

    def test_runtime_uses_task_dangerous_mode_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []

            def process_factory(request):
                requests.append(request)
                return OneShotProcess(request)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "rewrite the installer", "C1")
                task = replace(task, metadata={DANGEROUS_MODE_METADATA_KEY: True})
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))

                self.assertTrue(requests[0].dangerous)
                for _ in range(50):
                    if gateway.replies:
                        break
                    time.sleep(0.01)
            finally:
                store.close()

    def test_runtime_records_managed_session_on_capture(self):
        from agent_harness.sessions.managed_session import (
            managed_session_agent_key,
            managed_session_dangerous_key,
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "remember this", "C1")
                task = replace(task, metadata={DANGEROUS_MODE_METADATA_KEY: True})
                store.upsert_agent_task(task)
                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=SessionIdProcess,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if store.get_setting(
                        managed_session_agent_key(Provider.CODEX, "codex-thread-1")
                    ) and store.get_setting(
                        managed_session_dangerous_key(Provider.CODEX, "codex-thread-1")
                    ):
                        break
                    time.sleep(0.01)

                self.assertEqual(
                    store.get_setting(managed_session_agent_key(Provider.CODEX, "codex-thread-1")),
                    agent.agent_id,
                )
                self.assertEqual(
                    store.get_setting(
                        managed_session_dangerous_key(Provider.CODEX, "codex-thread-1")
                    ),
                    "1",
                )
            finally:
                store.close()

    def test_runtime_stop_task_clears_managed_session_setting(self):
        from agent_harness.sessions.managed_session import (
            managed_session_agent_key,
            record_managed_session,
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "free up via button", "C1")
                task = replace(
                    task,
                    session_provider=Provider.CODEX,
                    session_id="codex-stop",
                )
                store.upsert_agent_task(task)
                record_managed_session(
                    store,
                    Provider.CODEX,
                    "codex-stop",
                    agent.agent_id,
                    dangerous_mode=False,
                )
                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=OneShotProcess,
                    poll_seconds=0.01,
                )

                runtime.stop_task(task.task_id, AgentTaskStatus.DONE)

                self.assertIsNone(
                    store.get_setting(managed_session_agent_key(Provider.CODEX, "codex-stop"))
                )
            finally:
                store.close()

    def test_runtime_stop_all_clears_managed_session_settings(self):
        from agent_harness.sessions.managed_session import (
            managed_session_agent_key,
            record_managed_session,
        )

        class HangingProcess(OneShotProcess):
            def __init__(self, request):
                super().__init__(request)
                self.alive = True
                self.reading = threading.Event()
                self.release = threading.Event()

            def read_available(self, max_reads=20, timeout=0.05):
                self.reading.set()
                self.release.wait(timeout=5.0)
                return ""

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.alive = False

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            processes: list[HangingProcess] = []

            def process_factory(request):
                process = HangingProcess(request)
                processes.append(process)
                return process

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "cancel during shutdown", "C1")
                task = replace(
                    task,
                    session_provider=Provider.CODEX,
                    session_id="codex-shutdown",
                )
                store.upsert_agent_task(task)
                record_managed_session(
                    store,
                    Provider.CODEX,
                    "codex-shutdown",
                    agent.agent_id,
                    dangerous_mode=False,
                )
                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )
                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                self.assertTrue(processes[0].reading.wait(timeout=1.0))

                runtime.stop_all_running_tasks(status=AgentTaskStatus.CANCELLED, join_timeout=1.0)

                self.assertIsNone(
                    store.get_setting(managed_session_agent_key(Provider.CODEX, "codex-shutdown"))
                )
            finally:
                for process in processes:
                    process.release.set()
                store.close()

    def test_runtime_ignores_claude_permission_denials_in_dangerous_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []

            def process_factory(request):
                requests.append(request)
                return ClaudeDangerousPermissionDeniedThenFinalProcess(request)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "finish without prompting", "C1")
                task = replace(task, metadata={DANGEROUS_MODE_METADATA_KEY: True})
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if "Done after dangerous-mode bypass." in gateway.replies:
                        break
                    time.sleep(0.01)

                self.assertTrue(requests[0].dangerous)
                self.assertEqual(gateway.replies, ["Done after dangerous-mode bypass."])
                self.assertEqual(
                    store.list_pending_slack_agent_requests("claude/channel/permission"),
                    [],
                )
            finally:
                store.close()

    def test_runtime_loads_claude_channel_for_slack_thread_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = Store(home / "state.sqlite")
            requests = []

            def process_factory(request):
                requests.append(request)
                return ClaudeOneShotProcess(request)

            try:
                store.init_schema()
                (home / ".claude.json").write_text(
                    '{"mcpServers":{"slackgentic":{"command":"slackgentic"}}}\n',
                    encoding="utf-8",
                )
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "check prs", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                    home=home,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))

                self.assertTrue(requests[0].claude_channel)
                self.assertEqual(requests[0].slack_channel_id, "C1")
                self.assertEqual(requests[0].slack_thread_ts, "171.000001")
                for allowed_tool in SLACKGENTIC_MCP_PERMISSION_ALLOW:
                    self.assertIn(allowed_tool, requests[0].allowed_tools)
                for _ in range(50):
                    if gateway.replies:
                        break
                    time.sleep(0.01)
            finally:
                store.close()

    def test_runtime_retries_internal_slackgentic_mcp_permission_without_slack_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []

            def process_factory(request):
                requests.append(request)
                if len(requests) == 1:
                    return ClaudeSlackgenticMcpPermissionDeniedProcess(request)
                return ClaudeOneShotProcess(request)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "ask Slack to choose", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if "Done" in gateway.replies:
                        break
                    time.sleep(0.01)

                self.assertEqual(len(requests), 2)
                self.assertIn(
                    ClaudeSlackgenticMcpPermissionDeniedProcess.tool_name,
                    requests[1].allowed_tools,
                )
                self.assertEqual(gateway.replies, ["Done"])
                self.assertEqual(
                    store.list_pending_slack_agent_requests("claude/channel/permission"),
                    [],
                )
            finally:
                store.close()

    def test_runtime_retries_safe_auto_claude_read_only_denial_without_slack_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []

            def process_factory(request):
                requests.append(request)
                if len(requests) == 1:
                    return ClaudePermissionDeniedProcess(request)
                return ClaudeOneShotProcess(request)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "check PR metadata", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if "Done" in gateway.replies:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Done"])
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[1].resume_session_id, "claude-denied-session")
                self.assertIn("Bash(gh pr view:*)", requests[1].allowed_tools)
                self.assertEqual(
                    store.list_pending_slack_agent_requests("claude/channel/permission"),
                    [],
                )
            finally:
                store.close()

    def test_runtime_retries_safe_auto_piped_read_only_denial_without_slack_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []

            def process_factory(request):
                requests.append(request)
                if len(requests) == 1:
                    return ClaudePipedGitStatusPermissionDeniedProcess(request)
                return ClaudeOneShotProcess(request)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "check Talos repo status", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if "Done" in gateway.replies:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Done"])
                self.assertEqual(len(requests), 2)
                self.assertEqual(
                    requests[1].resume_session_id,
                    "claude-piped-git-status-session",
                )
                self.assertIn(
                    "Bash(git -C /workspace/repos/example-project status 2>&1 | head -30)",
                    requests[1].allowed_tools,
                )
                self.assertEqual(
                    store.list_pending_slack_agent_requests("claude/channel/permission"),
                    [],
                )
            finally:
                store.close()

    def test_runtime_retries_safe_auto_sequenced_read_only_denial_without_slack_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []

            def process_factory(request):
                requests.append(request)
                if len(requests) == 1:
                    return ClaudeSequencedPatchLookupPermissionDeniedProcess(request)
                return ClaudeOneShotProcess(request)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "check for prior patch file", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if "Done" in gateway.replies:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Done"])
                self.assertEqual(len(requests), 2)
                self.assertEqual(
                    requests[1].resume_session_id,
                    "claude-sequenced-patch-lookup-session",
                )
                self.assertIn(
                    "Bash(ls /tmp/example-provider-analysis.patch 2>&1; "
                    "ls /tmp/ | grep -i example 2>&1)",
                    requests[1].allowed_tools,
                )
                self.assertEqual(
                    store.list_pending_slack_agent_requests("claude/channel/permission"),
                    [],
                )
            finally:
                store.close()

    def test_runtime_retries_managed_claude_after_slack_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []
            approved = threading.Event()

            def process_factory(request):
                requests.append(request)
                if len(requests) == 1:
                    return ClaudePermissionDeniedProcess(request)
                return ClaudeOneShotProcess(request)

            def approve_request():
                for _ in range(100):
                    rows = store.list_pending_slack_agent_requests("claude/channel/permission")
                    if rows:
                        params = json.loads(rows[0]["params_json"])
                        self.assertEqual(params["tool_name"], "Bash")
                        self.assertIn("gh pr view 6", params["input_preview"])
                        store.resolve_slack_agent_request(rows[0]["token"], {"behavior": "allow"})
                        approved.set()
                        return
                    time.sleep(0.01)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "check gh auth", "C1")
                task = _with_permission_mode(task, PermissionMode.LOCKED)
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )
                approver = threading.Thread(target=approve_request)
                approver.start()

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if "Done" in gateway.replies:
                        break
                    time.sleep(0.01)
                approver.join(timeout=1)

                self.assertTrue(approved.is_set())
                self.assertIn(
                    "I'm blocked on approval before I can continue: view PR metadata",
                    gateway.replies[0],
                )
                self.assertIn("Claude requests command approval: gh", gateway.replies)
                self.assertIn("Done", gateway.replies)
                self.assertNotIn("The command requires approval.", gateway.replies)
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[1].resume_session_id, "claude-denied-session")
                self.assertEqual(
                    requests[1].allowed_tools[-2:],
                    ("Bash(gh pr view:*)", "Bash(gh pr view *)"),
                )
            finally:
                store.close()

    def test_runtime_broadens_claude_bash_allowlist_for_session_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []
            approved = threading.Event()

            def process_factory(request):
                requests.append(request)
                if len(requests) == 1:
                    return ClaudePermissionDeniedProcess(request)
                return ClaudeOneShotProcess(request)

            def approve_request():
                for _ in range(100):
                    rows = store.list_pending_slack_agent_requests("claude/channel/permission")
                    if rows:
                        store.resolve_slack_agent_request(
                            rows[0]["token"],
                            {"behavior": "allow", "scope": "session"},
                        )
                        approved.set()
                        return
                    time.sleep(0.01)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "check gh auth", "C1")
                task = _with_permission_mode(task, PermissionMode.LOCKED)
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )
                approver = threading.Thread(target=approve_request)
                approver.start()

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if "Done" in gateway.replies:
                        break
                    time.sleep(0.01)
                approver.join(timeout=1)

                self.assertTrue(approved.is_set())
                self.assertEqual(len(requests), 2)
                self.assertIn("Bash(gh pr view:*)", requests[1].allowed_tools)
                self.assertIn("Bash(gh:*)", requests[1].allowed_tools)
                self.assertIn("Bash(gh *)", requests[1].allowed_tools)
            finally:
                store.close()

    def test_runtime_posts_streamed_claude_permission_denial_without_waiting_for_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []
            approved = threading.Event()

            def process_factory(request):
                requests.append(request)
                if len(requests) == 1:
                    return ClaudeStreamPermissionDeniedProcess(request)
                return ClaudeOneShotProcess(request)

            def approve_request():
                for _ in range(100):
                    rows = store.list_pending_slack_agent_requests("claude/channel/permission")
                    if rows:
                        params = json.loads(rows[0]["params_json"])
                        self.assertEqual(params["tool_name"], "Bash")
                        self.assertIn("git log --oneline -30", params["input_preview"])
                        store.resolve_slack_agent_request(rows[0]["token"], {"behavior": "allow"})
                        approved.set()
                        return
                    time.sleep(0.01)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "show recent sample-app commits", "C1")
                task = _with_permission_mode(task, PermissionMode.LOCKED)
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )
                approver = threading.Thread(target=approve_request)
                approver.start()

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if "Done" in gateway.replies:
                        break
                    time.sleep(0.01)
                approver.join(timeout=1)

                self.assertTrue(approved.is_set())
                self.assertIn(
                    "I'm blocked on approval before I can continue: show recent commits",
                    gateway.replies[0],
                )
                self.assertEqual(
                    gateway.replies.count("Claude requests command approval: git"),
                    1,
                )
                self.assertIn("Done", gateway.replies)
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[1].resume_session_id, "claude-stream-denied-session")
                self.assertIn(
                    "Bash(cd /workspace/repos/sample-app && git log --oneline -30)",
                    requests[1].allowed_tools,
                )
                self.assertIn("Bash(git log:*)", requests[1].allowed_tools)
            finally:
                store.close()

    def test_runtime_retries_safe_auto_multiline_commit_without_slack_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []

            def process_factory(request):
                requests.append(request)
                if len(requests) == 1:
                    return ClaudeMultilineCommitPermissionDeniedProcess(request)
                return ClaudeOneShotProcess(request)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "commit staged changes", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if "Done" in gateway.replies:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Done"])
                self.assertNotIn(
                    "Claude requested a tool approval I cannot safely resume automatically.",
                    gateway.replies,
                )
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[1].resume_session_id, "claude-multiline-commit-session")
                self.assertIn(
                    "Bash(git -C /workspace/repos/sample-app commit:*)",
                    requests[1].allowed_tools,
                )
                self.assertEqual(
                    store.list_pending_slack_agent_requests("claude/channel/permission"),
                    [],
                )
            finally:
                store.close()

    def test_runtime_stops_live_claude_after_streamed_permission_denial(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []
            denied_processes = []
            approved = threading.Event()

            def process_factory(request):
                requests.append(request)
                if len(requests) == 1:
                    process = ClaudeLivePermissionDeniedProcess(request)
                    denied_processes.append(process)
                    return process
                return ClaudeOneShotProcess(request)

            def approve_request():
                for _ in range(100):
                    rows = store.list_pending_slack_agent_requests("claude/channel/permission")
                    if rows:
                        store.resolve_slack_agent_request(rows[0]["token"], {"behavior": "allow"})
                        approved.set()
                        return
                    time.sleep(0.01)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "show recent sample-app commits", "C1")
                task = _with_permission_mode(task, PermissionMode.LOCKED)
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )
                approver = threading.Thread(target=approve_request)
                approver.start()

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if "Done" in gateway.replies:
                        break
                    time.sleep(0.01)
                approver.join(timeout=1)

                self.assertTrue(approved.is_set())
                self.assertTrue(denied_processes[0].terminated)
                self.assertEqual(
                    gateway.replies.count("Claude requests command approval: git"),
                    1,
                )
                self.assertIn("Done", gateway.replies)
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[1].resume_session_id, "claude-live-denied-session")
                self.assertIn(
                    "Bash(cd /workspace/repos/sample-app && git log --oneline -30)",
                    requests[1].allowed_tools,
                )
                self.assertIn("Bash(git log:*)", requests[1].allowed_tools)
            finally:
                store.close()

    def test_runtime_accumulates_claude_allowed_tools_across_approvals(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []
            approved_tokens = set()

            def process_factory(request):
                requests.append(request)
                if len(requests) == 1:
                    return ClaudePermissionDeniedProcess(request)
                if len(requests) == 2:
                    return ClaudeSecondPermissionDeniedProcess(request)
                return ClaudeOneShotProcess(request)

            def approve_requests():
                for _ in range(200):
                    rows = store.list_pending_slack_agent_requests("claude/channel/permission")
                    for row in rows:
                        if row["token"] in approved_tokens:
                            continue
                        store.resolve_slack_agent_request(row["token"], {"behavior": "allow"})
                        approved_tokens.add(row["token"])
                    if len(approved_tokens) == 2:
                        return
                    time.sleep(0.01)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "check prs", "C1")
                task = _with_permission_mode(task, PermissionMode.LOCKED)
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )
                approver = threading.Thread(target=approve_requests)
                approver.start()

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(200):
                    if "Done" in gateway.replies:
                        break
                    time.sleep(0.01)
                approver.join(timeout=1)

                self.assertEqual(len(approved_tokens), 2)
                self.assertIn("Done", gateway.replies)
                self.assertEqual(len(requests), 3)
                self.assertEqual(
                    requests[1].allowed_tools[-2:],
                    ("Bash(gh pr view:*)", "Bash(gh pr view *)"),
                )
                self.assertEqual(
                    requests[2].allowed_tools[-4:],
                    (
                        "Bash(gh pr view:*)",
                        "Bash(gh pr view *)",
                        "Bash(gh search prs:*)",
                        "Bash(gh search prs *)",
                    ),
                )
            finally:
                store.close()

    def test_runtime_restarts_managed_claude_when_resume_session_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []

            def process_factory(request):
                requests.append(request)
                if len(requests) == 1:
                    return ClaudeMissingResumeProcess(request)
                return ClaudeOneShotProcess(request)

            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "continue task", "C1")
                task = replace(
                    task,
                    session_provider=Provider.CLAUDE,
                    session_id="missing-session",
                )
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if "Done" in gateway.replies:
                        break
                    time.sleep(0.01)

                self.assertIn("Done", gateway.replies)
                self.assertNotIn("Claude error: error_during_execution", gateway.replies)
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0].resume_session_id, "missing-session")
                self.assertIsNone(requests[1].resume_session_id)
            finally:
                store.close()

    def test_runtime_notifies_agent_message_callback_once_per_visible_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "ask for review", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                seen = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=OneShotProcess,
                    poll_seconds=0.01,
                    on_agent_message=lambda task, agent, thread, text, message_ts: (
                        seen.append(
                            (task.task_id, agent.agent_id, thread.thread_ts, text, message_ts)
                        )
                        or True
                    ),
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    if seen:
                        break
                    time.sleep(0.01)

                self.assertEqual(len(seen), 1)
                self.assertEqual(seen[0][3], "Done")
                self.assertEqual(
                    store.get_agent_task(task.task_id).status,
                    AgentTaskStatus.ACTIVE,
                )
            finally:
                store.close()

    def test_runtime_done_callback_can_restart_same_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "initial", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=OneShotProcess,
                    poll_seconds=0.01,
                )
                done_event = threading.Event()
                callbacks = []
                restart_results = []

                def on_task_done(completed_task, callback_agent, thread):
                    callbacks.append(completed_task.prompt)
                    if len(callbacks) == 1:
                        followup = replace(
                            completed_task,
                            prompt="queued follow-up",
                            status=AgentTaskStatus.ACTIVE,
                        )
                        store.upsert_agent_task(followup)
                        restart_results.append(runtime.start_task(followup, callback_agent, thread))
                    else:
                        done_event.set()

                runtime.on_task_done = on_task_done

                self.assertTrue(runtime.start_task(task, agent, SlackThreadRef("C1", "171.thread")))
                for _ in range(100):
                    if done_event.is_set():
                        break
                    time.sleep(0.01)

                self.assertEqual(restart_results, [True])
                self.assertEqual(callbacks, ["initial", "queued follow-up"])
            finally:
                store.close()

    def test_runtime_posts_duplicate_codex_final_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "finish once", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                seen = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=DuplicateCodexFinalProcess,
                    poll_seconds=0.01,
                    on_agent_message=lambda task, agent, thread, text, message_ts: seen.append(
                        text
                    ),
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if gateway.replies and not runtime.has_running_tasks():
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Final answer"])
                self.assertEqual(seen, ["Final answer"])
            finally:
                store.close()

    def test_runtime_hides_agent_control_signal_and_notifies_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "close the thread", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                seen_controls = []
                done_callbacks = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=ThreadDoneSignalProcess,
                    poll_seconds=0.01,
                    on_agent_control=lambda task, agent, thread, signal: (
                        seen_controls.append(
                            (task.task_id, agent.agent_id, thread.thread_ts, signal)
                        )
                        or True
                    ),
                    on_task_done=lambda task, agent, thread: done_callbacks.append(task.task_id),
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    if seen_controls:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Sounds good."])
                self.assertEqual(
                    seen_controls,
                    [(task.task_id, agent.agent_id, "171.000001", AGENT_THREAD_DONE_SIGNAL)],
                )
                self.assertEqual(done_callbacks, [])
            finally:
                store.close()

    def test_runtime_posts_visible_thread_done_text_before_control_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "close the thread", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                control_order = []

                def handle_control(task, agent, thread, signal):
                    control_order.append((signal, list(gateway.replies)))
                    return True

                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=ThreadDoneSignalProcess,
                    poll_seconds=0.01,
                    on_agent_control=handle_control,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    if control_order:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Sounds good."])
                self.assertEqual(control_order, [(AGENT_THREAD_DONE_SIGNAL, ["Sounds good."])])
            finally:
                store.close()

    def test_runtime_handles_thread_done_signal_before_live_process_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            runtime = None
            LiveThreadDoneSignalProcess.instances = []
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "close the thread", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                seen_controls = []
                done_callbacks = []

                def handle_control(task, agent, thread, signal):
                    seen_controls.append((task.task_id, agent.agent_id, thread.thread_ts, signal))
                    assert runtime is not None
                    runtime.stop_task(task.task_id, AgentTaskStatus.DONE)
                    return True

                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=LiveThreadDoneSignalProcess,
                    poll_seconds=0.01,
                    on_agent_control=handle_control,
                    on_task_done=lambda task, agent, thread: done_callbacks.append(task.task_id),
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if seen_controls and not runtime.has_running_tasks():
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Sounds good."])
                self.assertEqual(
                    seen_controls,
                    [(task.task_id, agent.agent_id, "171.000001", AGENT_THREAD_DONE_SIGNAL)],
                )
                self.assertEqual(store.get_agent_task(task.task_id).status, AgentTaskStatus.DONE)
                self.assertEqual(done_callbacks, [])
                self.assertTrue(LiveThreadDoneSignalProcess.instances[0].terminated)
            finally:
                if runtime is not None:
                    runtime.stop_all_running_tasks()
                store.close()

    def test_runtime_hides_agent_timer_signal_and_notifies_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "check later", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                seen_controls = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=TimerSignalProcess,
                    poll_seconds=0.01,
                    on_agent_control=lambda task, agent, thread, signal: (
                        seen_controls.append(signal) or True
                    ),
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    if seen_controls:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Wakeup scheduled."])
                self.assertEqual(
                    seen_controls,
                    [f"{AGENT_TIMER_SIGNAL_PREFIX}2s | Re-check the PR feedback."],
                )
            finally:
                store.close()

    def test_runtime_timer_signal_does_not_suppress_task_done_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "check later", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                seen_controls = []
                done_callbacks = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=TimerSignalProcess,
                    poll_seconds=0.01,
                    on_agent_control=lambda task, agent, thread, signal: (
                        seen_controls.append(signal) or True
                    ),
                    on_task_done=lambda task, agent, thread: done_callbacks.append(task.task_id),
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    if seen_controls and done_callbacks:
                        break
                    time.sleep(0.01)

                self.assertEqual(
                    seen_controls,
                    [f"{AGENT_TIMER_SIGNAL_PREFIX}2s | Re-check the PR feedback."],
                )
                self.assertEqual(done_callbacks, [task.task_id])
            finally:
                store.close()

    def test_runtime_hides_schedule_success_text_and_notifies_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "schedule work", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                seen_controls = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=ScheduleSignalProcess,
                    poll_seconds=0.01,
                    on_agent_control=lambda task, agent, thread, signal: (
                        seen_controls.append(signal) or True
                    ),
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    if seen_controls:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, [])
                self.assertEqual(len(seen_controls), 1)
                self.assertTrue(seen_controls[0].startswith(AGENT_SCHEDULE_SIGNAL_PREFIX))
            finally:
                store.close()

    def test_runtime_handles_roster_status_signal_immediately_without_deferring_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "ship the roster update", "C1")
                store.upsert_agent_task(task)
                gateway = FakeGateway()
                seen_controls = []
                done_callbacks = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=RosterStatusSignalProcess,
                    poll_seconds=0.01,
                    on_agent_control=lambda task, agent, thread, signal: (
                        seen_controls.append(signal) or True
                    ),
                    on_task_done=lambda task, agent, thread: done_callbacks.append(task.task_id),
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(100):
                    if done_callbacks:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Still testing."])
                self.assertEqual(
                    seen_controls,
                    [
                        f"{AGENT_ROSTER_STATUS_SIGNAL_PREFIX}PR merge and daemon restart: running E2E."
                    ],
                )
                self.assertEqual(done_callbacks, [task.task_id])
            finally:
                store.close()

    def test_runtime_recovers_visible_message_from_codex_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = Store(home / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "help", "C1")
                store.upsert_agent_task(task)
                path = (
                    home
                    / ".codex"
                    / "sessions"
                    / "2026"
                    / "04"
                    / "29"
                    / f"rollout-2026-04-29T01-17-28-{SessionOnlyProcess.session_id}.jsonl"
                )
                path.parent.mkdir(parents=True)
                path.write_text(
                    f'{{"timestamp":"{task.created_at.isoformat()}","type":"event_msg",'
                    '"payload":{"type":"agent_message","message":"Recovered answer"}}\n'
                )
                gateway = FakeGateway()
                done_callbacks = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=SessionOnlyProcess,
                    poll_seconds=0.01,
                    on_task_done=lambda task, agent, thread: done_callbacks.append(task.task_id),
                    home=home,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                deadline = time.monotonic() + 5
                while (not gateway.replies or not done_callbacks) and time.monotonic() < deadline:
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Recovered answer"])
                self.assertEqual(done_callbacks, [task.task_id])
                self.assertEqual(
                    store.get_agent_task(task.task_id).status,
                    AgentTaskStatus.ACTIVE,
                )
            finally:
                store.close()

    def test_runtime_recovers_unseen_final_message_after_progress_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = Store(home / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "help", "C1")
                store.upsert_agent_task(task)
                path = (
                    home
                    / ".codex"
                    / "sessions"
                    / "2026"
                    / "04"
                    / "29"
                    / (f"rollout-2026-04-29T01-17-28-{ProgressOnlySessionProcess.session_id}.jsonl")
                )
                path.parent.mkdir(parents=True)
                path.write_text(
                    f'{{"timestamp":"{task.created_at.isoformat()}","type":"event_msg",'
                    '"payload":{"type":"agent_message","message":"Working"}}\n'
                    f'{{"timestamp":"{task.created_at.isoformat()}","type":"event_msg",'
                    '"payload":{"type":"agent_message","message":"Recovered final"}}\n'
                )
                gateway = FakeGateway()
                done_callbacks = []
                seen = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=ProgressOnlySessionProcess,
                    poll_seconds=0.01,
                    on_agent_message=lambda task, agent, thread, text, message_ts: (
                        seen.append(text) or True
                    ),
                    on_task_done=lambda task, agent, thread: done_callbacks.append(task.task_id),
                    home=home,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                deadline = time.monotonic() + 5
                while (
                    len(gateway.replies) < 2 or not done_callbacks
                ) and time.monotonic() < deadline:
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Working", "Recovered final"])
                self.assertEqual(seen, ["Working", "Recovered final"])
                self.assertEqual(done_callbacks, [task.task_id])
            finally:
                store.close()

    def test_runtime_does_not_repost_transcript_history_when_resuming_session(self):
        # Resuming a session that already has visible messages (e.g. a previous
        # worker on the same session before a daemon restart) must NOT replay
        # the existing transcript to Slack. Without seeding observed messages,
        # the recovery branch at process-exit posts every historical chunk and
        # the thread sees every prior agent reply duplicated.
        session_id = "claude-resume-historical-session"

        class ResumeAssistantProcess(OneShotProcess):
            def read_available(self, max_reads=20, timeout=0.05):
                if self.reads == 0:
                    self.reads += 1
                    return (
                        f'{{"type":"system","subtype":"init","session_id":"{session_id}"}}\n'
                        '{"type":"assistant","message":{"role":"assistant","content":'
                        '[{"type":"text","text":"Yes, codex is simpler than claude here."}]}}\n'
                    )
                return ""

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = Store(home / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "follow-up after restart", "C1"),
                    session_id=session_id,
                    session_provider=Provider.CLAUDE,
                )
                store.upsert_agent_task(task)

                transcript_dir = home / ".claude" / "projects" / "-tmp-repo"
                transcript_dir.mkdir(parents=True)
                transcript_path = transcript_dir / f"{session_id}.jsonl"
                historical = "Took a look at how Slackgentic mirrors Claude into Slack."
                transcript_path.write_text(
                    json.dumps(
                        {
                            "type": "assistant",
                            "timestamp": task.created_at.isoformat(),
                            "sessionId": session_id,
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": historical}],
                            },
                        }
                    )
                    + "\n"
                )
                runtime = ManagedTaskRuntime(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(),
                    process_factory=ResumeAssistantProcess,
                    poll_seconds=0.01,
                    home=home,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                gateway = runtime.gateway
                deadline = time.monotonic() + 5
                while not gateway.replies and time.monotonic() < deadline:
                    time.sleep(0.01)
                # Wait for the recovery branch to also fire.
                time.sleep(0.2)

                self.assertEqual(
                    gateway.replies,
                    ["Yes, codex is simpler than claude here."],
                )
                self.assertNotIn(historical, gateway.replies)
            finally:
                store.close()

    def test_runtime_recovers_visible_message_from_claude_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = Store(home / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "open the PR", "C1")
                store.upsert_agent_task(task)
                transcript_dir = home / ".claude" / "projects" / "-tmp-repo"
                transcript_dir.mkdir(parents=True)
                transcript_path = transcript_dir / f"{ClaudeSessionOnlyProcess.session_id}.jsonl"
                transcript_path.write_text(
                    json.dumps(
                        {
                            "type": "assistant",
                            "timestamp": task.created_at.isoformat(),
                            "sessionId": ClaudeSessionOnlyProcess.session_id,
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": ("PR up: https://github.com/example/repo/pull/123"),
                                    }
                                ],
                            },
                        }
                    )
                    + "\n"
                )
                gateway = FakeGateway()
                done_callbacks = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=ClaudeSessionOnlyProcess,
                    poll_seconds=0.01,
                    on_task_done=lambda task, agent, thread: done_callbacks.append(task.task_id),
                    home=home,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                deadline = time.monotonic() + 5
                while (not gateway.replies or not done_callbacks) and time.monotonic() < deadline:
                    time.sleep(0.01)

                self.assertEqual(
                    gateway.replies,
                    ["PR up: https://github.com/example/repo/pull/123"],
                )
                self.assertEqual(done_callbacks, [task.task_id])
                self.assertNotEqual(
                    store.get_agent_task(task.task_id).status,
                    AgentTaskStatus.CANCELLED,
                )
            finally:
                store.close()

    def test_runtime_posts_claude_assistant_text_without_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = Store(home / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "open the PR", "C1")
                store.upsert_agent_task(task)
                transcript_dir = home / ".claude" / "projects" / "-tmp-repo"
                transcript_dir.mkdir(parents=True)
                transcript_path = transcript_dir / f"{ClaudeAssistantOnlyProcess.session_id}.jsonl"
                started_at = task.created_at.isoformat()
                transcript_path.write_text(
                    json.dumps(
                        {
                            "type": "assistant",
                            "timestamp": started_at,
                            "sessionId": ClaudeAssistantOnlyProcess.session_id,
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "Working on it"}],
                            },
                        }
                    )
                    + "\n"
                    + json.dumps(
                        {
                            "type": "assistant",
                            "timestamp": started_at,
                            "sessionId": ClaudeAssistantOnlyProcess.session_id,
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": ("PR up: https://github.com/example/repo/pull/456"),
                                    }
                                ],
                            },
                        }
                    )
                    + "\n"
                )
                gateway = FakeGateway()
                done_callbacks = []
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=ClaudeAssistantOnlyProcess,
                    poll_seconds=0.01,
                    on_task_done=lambda task, agent, thread: done_callbacks.append(task.task_id),
                    home=home,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000002"))
                deadline = time.monotonic() + 5
                while (
                    len(gateway.replies) < 2 or not done_callbacks
                ) and time.monotonic() < deadline:
                    time.sleep(0.01)

                self.assertEqual(
                    gateway.replies,
                    ["Working on it", "PR up: https://github.com/example/repo/pull/456"],
                )
                self.assertEqual(done_callbacks, [task.task_id])
                self.assertNotEqual(
                    store.get_agent_task(task.task_id).status,
                    AgentTaskStatus.CANCELLED,
                )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
