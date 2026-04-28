import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from agent_harness.config import AgentCommandConfig
from agent_harness.models import AgentTaskKind, AgentTaskStatus, Provider, SlackThreadRef
from agent_harness.store import Store
from agent_harness.task_runtime import (
    ManagedTaskRuntime,
    _clean_terminal_output,
    _process_output_chunks,
    _requested_repo_cwd,
    build_task_prompt,
)
from agent_harness.team import build_initial_model_team, create_agent_task


class FakeGateway:
    def __init__(self):
        self.replies = []

    def post_thread_reply(self, thread, text, persona=None):
        self.replies.append(text)


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

    def test_requested_repo_cwd_uses_named_sibling_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default = root / "slackgentic-team"
            talos = root / "talos"
            default.mkdir()
            talos.mkdir()

            cwd = _requested_repo_cwd("in talos summarize the test command", default)

            self.assertEqual(cwd, talos)

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
                    current = store.get_agent_task(task.task_id)
                    if current and current.status == AgentTaskStatus.DONE:
                        break
                    time.sleep(0.01)

                self.assertEqual(gateway.replies, ["Done"])
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
                    current = store.get_agent_task(task.task_id)
                    if current and current.status == AgentTaskStatus.DONE:
                        break
                    time.sleep(0.01)

                self.assertEqual(len(seen), 1)
                self.assertEqual(seen[0][3], "Done")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
