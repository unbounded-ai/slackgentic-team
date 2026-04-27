import unittest

from agent_harness.models import AgentTaskKind, AssignmentMode, Provider, WorkRequest
from agent_harness.team import (
    build_initial_model_team,
    build_initialization_messages,
    choose_reaction,
    hire_team_agents,
    least_represented_provider,
    pick_idle_agent,
)


class TeamTests(unittest.TestCase):
    def test_initial_model_team_has_persistent_provider_mapping(self):
        agents = build_initial_model_team(codex_count=2, claude_count=1)
        self.assertEqual(
            [agent.provider_preference for agent in agents],
            [Provider.CODEX, Provider.CODEX, Provider.CLAUDE],
        )
        self.assertEqual(len({agent.handle for agent in agents}), 3)

    def test_hire_auto_picks_less_represented_provider(self):
        agents = build_initial_model_team(codex_count=2, claude_count=1)
        self.assertEqual(least_represented_provider(agents), Provider.CLAUDE)
        hired = hire_team_agents(agents, 1)
        self.assertEqual(hired[0].provider_preference, Provider.CLAUDE)

    def test_initialization_messages_include_intros_and_welcomes(self):
        agents = build_initial_model_team(codex_count=2, claude_count=1)
        messages = build_initialization_messages(agents)
        intros = [message for message in messages if message.kind == "introduction"]
        welcomes = [message for message in messages if message.kind == "welcome"]
        self.assertEqual(len(intros), 3)
        self.assertGreaterEqual(len(welcomes), 1)
        self.assertIn("@", intros[0].text)

    def test_review_assignment_prefers_cross_model_agent(self):
        agents = build_initial_model_team(codex_count=2, claude_count=1)
        author = next(agent for agent in agents if agent.provider_preference == Provider.CODEX)
        request = WorkRequest(
            prompt="review the PR",
            assignment_mode=AssignmentMode.ANYONE,
            task_kind=AgentTaskKind.REVIEW,
            author_handle=author.handle,
        )
        picked = pick_idle_agent(agents, request, author)
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked.provider_preference, Provider.CLAUDE)

    def test_reaction_comes_from_agent_palette(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        reaction = choose_reaction(agent, "tests passed")
        self.assertIn(reaction, agent.reaction_names)


if __name__ == "__main__":
    unittest.main()
