import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from agent_harness.config import AgentCommandConfig
from agent_harness.models import (
    DANGEROUS_MODE_METADATA_KEY,
    AgentTaskKind,
    AgentTaskStatus,
    Provider,
    SlackThreadRef,
)
from agent_harness.runtime.tasks import (
    AGENT_THREAD_DONE_SIGNAL,
    ManagedTaskRuntime,
    _allowed_tool_for_claude_denial,
    _append_allowed_tool,
    _claude_missing_resume_session,
    _claude_permission_denials,
    _clean_terminal_output,
    _extract_agent_control_signals,
    _latest_codex_transcript_message,
    _process_output_chunks,
    _requested_repo_cwd,
    _session_id_from_output,
    build_task_prompt,
)
from agent_harness.slack.client import PostedMessage
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team, create_agent_task


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


class ClaudeSecondPermissionDeniedProcess(ClaudePermissionDeniedProcess):
    command = 'gh search prs --author "@me" --state open --json number,title --limit 50'
    description = "Search open PRs"


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


class SilentProcess(OneShotProcess):
    def read_available(self, max_reads=20, timeout=0.05):
        self.reads += 1
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


class TaskRuntimeTests(unittest.TestCase):
    def test_build_task_prompt_includes_persona_and_pr_context(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        task = create_agent_task(agent, "review the PR", "C1", kind=AgentTaskKind.REVIEW)
        task = replace(task, metadata={"pr_url": "https://github.com/acme/app/pull/42"})

        prompt = build_task_prompt(agent, task)

        self.assertIn(f"@{agent.handle}", prompt)
        self.assertIn("review the PR", prompt)
        self.assertIn("https://github.com/acme/app/pull/42", prompt)

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
        self.assertIn("@nell pick one before I proceed", prompt)
        self.assertIn("Do not put that callback handle inline", prompt)

    def test_requested_repo_cwd_uses_named_sibling_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default = root
            talos = root / "talos"
            talos.mkdir()

            cwd = _requested_repo_cwd("in talos summarize the test command", default)

            self.assertEqual(cwd, talos)

    def test_requested_repo_cwd_ignores_missing_root(self):
        missing = Path("/tmp/slackgentic-missing-root-for-test")

        self.assertEqual(_requested_repo_cwd("in talos summarize tests", missing), missing)

    def test_command_config_accepts_env_aliases(self):
        config = AgentCommandConfig.model_validate(
            {
                "SLACKGENTIC_CODEX_BINARY": "/tmp/codex",
                "SLACKGENTIC_CLAUDE_BINARY": "/tmp/claude",
                "SLACKGENTIC_DEFAULT_CWD": "/tmp",
                "SLACKGENTIC_DANGEROUS_BY_DEFAULT": "true",
            }
        )

        self.assertEqual(config.codex_binary, "/tmp/codex")
        self.assertEqual(config.claude_binary, "/tmp/claude")
        self.assertEqual(config.default_cwd, Path("/tmp"))
        self.assertTrue(config.dangerous_by_default)

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

    def test_claude_json_output_ignores_assistant_text_and_artifacts_before_result(self):
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

        self.assertEqual(chunks, ["Final"])
        self.assertEqual(buffer, "")

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

        self.assertEqual(_allowed_tool_for_claude_denial(denial), "Bash(gh pr view *)")

    def test_claude_permission_denial_rejects_gh_write_pattern(self):
        denial = {
            "tool_name": "Bash",
            "tool_input": {"command": "gh pr merge 6 --squash"},
        }

        self.assertIsNone(_allowed_tool_for_claude_denial(denial))

    def test_append_allowed_tool_preserves_prior_approvals(self):
        allowed = _append_allowed_tool(("Bash(gh pr list *)",), "Bash(gh pr view *)")

        self.assertEqual(allowed, ("Bash(gh pr list *)", "Bash(gh pr view *)"))
        self.assertEqual(_append_allowed_tool(allowed, "Bash(gh pr view *)"), allowed)

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
                for _ in range(50):
                    if gateway.replies:
                        break
                    time.sleep(0.01)
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
                self.assertIn("Claude requests tool approval: Bash", gateway.replies)
                self.assertIn("Done", gateway.replies)
                self.assertNotIn("The command requires approval.", gateway.replies)
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[1].resume_session_id, "claude-denied-session")
                self.assertEqual(requests[1].allowed_tools, ("Bash(gh pr view *)",))
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
                self.assertEqual(requests[1].allowed_tools, ("Bash(gh pr view *)",))
                self.assertEqual(
                    requests[2].allowed_tools,
                    ("Bash(gh pr view *)", "Bash(gh search prs *)"),
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
                    on_agent_message=lambda task, agent, thread, text: (
                        seen.append((task.task_id, agent.agent_id, thread.thread_ts, text)) or True
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
                for _ in range(50):
                    if gateway.replies:
                        break
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
                    on_agent_message=lambda task, agent, thread, text: seen.append(text) or True,
                    on_task_done=lambda task, agent, thread: done_callbacks.append(task.task_id),
                    home=home,
                )

                runtime.start_task(task, agent, SlackThreadRef("C1", "171.000001"))
                for _ in range(50):
                    if len(gateway.replies) >= 2:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Working", "Recovered final"])
                self.assertEqual(seen, ["Working", "Recovered final"])
                self.assertEqual(done_callbacks, [task.task_id])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
