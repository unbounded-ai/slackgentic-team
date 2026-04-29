import tempfile
import unittest
from pathlib import Path

from agent_harness.models import AgentTaskKind, Provider
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team
from agent_harness.team.assignment import assign_channel_work_request


class AssignmentTests(unittest.TestCase):
    def test_assign_channel_work_request_creates_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                for agent in build_initial_model_team(codex_count=1, claude_count=1):
                    store.upsert_team_agent(agent)

                result = assign_channel_work_request(
                    store,
                    "Somebody do update the docs",
                    channel_id="C1",
                    requested_by_slack_user="U1",
                )
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.task.channel_id, "C1")
                self.assertEqual(result.task.requested_by_slack_user, "U1")
                self.assertEqual(len(store.idle_team_agents()), 1)
            finally:
                store.close()

    def test_review_assignment_prefers_opposite_model_from_author(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=1, claude_count=1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                author = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )

                result = assign_channel_work_request(
                    store,
                    f"Somebody review @{author.handle}'s PR",
                    channel_id="C1",
                    requested_by_slack_user="U1",
                )
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.task.kind, AgentTaskKind.REVIEW)
                self.assertEqual(result.agent.provider_preference, Provider.CLAUDE)
                self.assertEqual(result.task.metadata["author_handle"], author.handle)
            finally:
                store.close()

    def test_non_pr_review_assignment_prefers_claude_reviewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                for agent in build_initial_model_team(codex_count=1, claude_count=1):
                    store.upsert_team_agent(agent)

                result = assign_channel_work_request(
                    store,
                    "Somebody review the repo and suggest cleanup improvements",
                    channel_id="C1",
                    requested_by_slack_user="U1",
                )

                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.task.kind, AgentTaskKind.REVIEW)
                self.assertEqual(result.agent.provider_preference, Provider.CLAUDE)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
