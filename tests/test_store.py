import tempfile
import unittest
from pathlib import Path

from agent_harness.models import AgentTaskStatus, Provider, SessionDependency, SlackThreadRef
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team, create_agent_task


class StoreTests(unittest.TestCase):
    def test_dependencies_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                store.add_dependency(
                    SessionDependency(
                        blocked_session_id="blocked",
                        blocking_thread=SlackThreadRef(
                            channel_id="C1",
                            thread_ts="171.000001",
                            message_ts="171.000001",
                            permalink="https://example.slack.com/archives/C1/p0000000171000001",
                        ),
                        created_by_slack_user="U1",
                        reason="wait for this",
                    )
                )
                deps = store.dependencies_for("blocked")
                self.assertEqual(len(deps), 1)
                self.assertEqual(deps[0].blocking_thread.channel_id, "C1")
                self.assertEqual(deps[0].created_by_slack_user, "U1")
            finally:
                store.close()

    def test_team_agents_and_tasks_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=1, claude_count=1)
                for agent in agents:
                    store.upsert_team_agent(agent)

                listed = store.list_team_agents()
                self.assertEqual(len(listed), 2)
                self.assertEqual(listed[0].provider_preference, Provider.CODEX)

                task = create_agent_task(listed[0], "do the thing", "C1")
                store.upsert_agent_task(task)
                self.assertEqual(store.active_task_for_agent(listed[0].agent_id), task)
                self.assertEqual(len(store.idle_team_agents()), 1)

                fired = store.fire_team_agent(listed[1].handle)
                self.assertIsNotNone(fired)
                assert fired is not None
                self.assertEqual(fired.status.value, "fired")
                active_task = store.active_task_for_agent(listed[0].agent_id)
                self.assertIsNotNone(active_task)
                assert active_task is not None
                self.assertEqual(active_task.status, AgentTaskStatus.QUEUED)
            finally:
                store.close()

    def test_slack_thread_and_mirror_cursor_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                thread = SlackThreadRef("C1", "171.000001", "171.000001")

                store.upsert_slack_thread_for_session(
                    Provider.CODEX,
                    "s1",
                    "T1",
                    thread,
                )
                stored = store.get_slack_thread_for_session(
                    Provider.CODEX,
                    "s1",
                    "T1",
                    "C1",
                )

                self.assertEqual(stored, thread)
                self.assertEqual(store.get_session_mirror_cursor(Provider.CODEX, "s1"), 0)

                store.set_session_mirror_cursor(Provider.CODEX, "s1", 42)

                self.assertEqual(store.get_session_mirror_cursor(Provider.CODEX, "s1"), 42)
            finally:
                store.close()

    def test_session_bridge_prompt_consumes_prefixed_transcript_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                store.add_session_bridge_prompt(Provider.CLAUDE, "s1", "reply exactly OK")

                consumed = store.consume_session_bridge_prompt(
                    Provider.CLAUDE,
                    "s1",
                    "reply exactly OK\nextra inherited stdin",
                )

                self.assertTrue(consumed)
                self.assertFalse(
                    store.consume_session_bridge_prompt(Provider.CLAUDE, "s1", "reply exactly OK")
                )
            finally:
                store.close()

    def test_claude_channel_message_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                message_id = store.enqueue_claude_channel_message(
                    123,
                    "s1",
                    "hello from Slack",
                    {"slack_channel": "C1"},
                )

                rows = store.pending_claude_channel_messages(123)

                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["id"], message_id)
                self.assertEqual(rows[0]["content"], "hello from Slack")

                store.mark_claude_channel_message_delivered(message_id)

                self.assertTrue(store.is_claude_channel_message_delivered(message_id))
                self.assertEqual(store.pending_claude_channel_messages(123), [])
            finally:
                store.close()

    def test_cancelled_claude_channel_message_is_not_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                message_id = store.enqueue_claude_channel_message(123, "s1", "hello", {})

                store.cancel_claude_channel_message(message_id)

                self.assertFalse(store.is_claude_channel_message_delivered(message_id))
                self.assertEqual(store.pending_claude_channel_messages(123), [])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
