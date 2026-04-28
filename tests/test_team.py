import random
import unittest
from pathlib import Path

from agent_harness.models import AgentTaskKind, AssignmentMode, Provider, WorkRequest
from agent_harness.team import (
    AVATAR_IDENTITY_BANK,
    DEFAULT_AVATAR_BANK_SIZE,
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
        self.assertEqual([agent.handle for agent in agents], ["avery", "jordan", "morgan"])
        self.assertEqual([agent.avatar_slug for agent in agents], ["1", "2", "3"])
        self.assertIn("generalist engineer", agents[0].metadata["backstory"])
        self.assertIn("avatar", agents[0].metadata["avatar_prompt"].lower())
        self.assertIn("cartoon", agents[0].metadata["avatar_prompt"].lower())
        self.assertEqual(
            agents[0].metadata["avatar_path"],
            f"docs/assets/avatars/{agents[0].avatar_slug}.png",
        )

    def test_avatar_identity_bank_has_500_unique_full_names(self):
        self.assertEqual(len(AVATAR_IDENTITY_BANK), DEFAULT_AVATAR_BANK_SIZE)
        self.assertEqual(
            len({identity.full_name for identity in AVATAR_IDENTITY_BANK}),
            DEFAULT_AVATAR_BANK_SIZE,
        )
        self.assertEqual(AVATAR_IDENTITY_BANK[0].full_name, "Avery Chen")
        self.assertEqual(AVATAR_IDENTITY_BANK[0].avatar_index, 1)
        self.assertEqual(AVATAR_IDENTITY_BANK[-1].avatar_index, 500)

    def test_large_team_uses_numbered_avatar_bank_without_repeating_names(self):
        agents = build_initial_model_team(codex_count=250, claude_count=250)

        self.assertEqual(len({agent.full_name for agent in agents}), DEFAULT_AVATAR_BANK_SIZE)
        self.assertEqual(
            {agent.avatar_slug for agent in agents},
            {str(index) for index in range(1, DEFAULT_AVATAR_BANK_SIZE + 1)},
        )

    def test_avatar_assets_exist_for_identity_bank(self):
        assets_dir = Path(__file__).resolve().parents[1] / "docs" / "assets" / "avatars"

        self.assertEqual(len(list(assets_dir.glob("*.png"))), DEFAULT_AVATAR_BANK_SIZE)
        self.assertTrue((assets_dir / "1.png").exists())
        self.assertTrue((assets_dir / "500.png").exists())
        self.assertTrue((assets_dir / "manifest.json").exists())

    def test_team_profiles_are_communication_oriented(self):
        agents = build_initial_model_team(codex_count=6, claude_count=6)
        banned_terms = {
            "security",
            "docs",
            "ux",
            "api",
            "release",
            "deployment",
            "test strategist",
        }
        for agent in agents:
            profile_text = " ".join(
                [
                    agent.role,
                    agent.personality,
                    agent.voice,
                    agent.unique_strength,
                    str(agent.metadata.get("backstory", "")),
                ]
            ).lower()
            self.assertFalse(
                any(term in profile_text for term in banned_terms),
                profile_text,
            )

    def test_hire_auto_picks_less_represented_provider(self):
        agents = build_initial_model_team(codex_count=2, claude_count=1)
        self.assertEqual(least_represented_provider(agents), Provider.CLAUDE)
        hired = hire_team_agents(agents, 1)
        self.assertEqual(hired[0].provider_preference, Provider.CLAUDE)

    def test_randomized_hire_draws_unused_identity_from_bank(self):
        agents = build_initial_model_team(codex_count=2, claude_count=1)
        hired = hire_team_agents(
            agents,
            3,
            Provider.CODEX,
            start_sort_order=3,
            randomize_identities=True,
            rng=random.Random(7),
        )

        existing_names = {agent.full_name for agent in agents}
        existing_avatars = {agent.avatar_slug for agent in agents}
        self.assertTrue(existing_names.isdisjoint({agent.full_name for agent in hired}))
        self.assertTrue(existing_avatars.isdisjoint({agent.avatar_slug for agent in hired}))
        self.assertEqual(len({agent.full_name for agent in hired}), 3)
        self.assertEqual(len({agent.avatar_slug for agent in hired}), 3)

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

    def test_review_assignment_without_author_prefers_claude(self):
        agents = build_initial_model_team(codex_count=2, claude_count=1)
        request = WorkRequest(
            prompt="review the repo",
            assignment_mode=AssignmentMode.ANYONE,
            task_kind=AgentTaskKind.REVIEW,
        )
        picked = pick_idle_agent(agents, request)
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked.provider_preference, Provider.CLAUDE)

    def test_reaction_comes_from_agent_palette(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        reaction = choose_reaction(agent, "tests passed")
        self.assertEqual(reaction, "test_tube")

    def test_reaction_defaults_to_non_random_acknowledgement(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        self.assertEqual(choose_reaction(agent, "please look when you can"), "eyes")


if __name__ == "__main__":
    unittest.main()
