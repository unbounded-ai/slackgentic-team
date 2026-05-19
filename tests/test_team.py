import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent_harness.models import (
    ASSIGNMENT_PROMPT_METADATA_KEY,
    PR_URLS_METADATA_KEY,
    ROSTER_SUMMARY_METADATA_KEY,
    AgentTaskKind,
    AssignmentMode,
    Provider,
    TeamAgentStatus,
    WorkRequest,
)
from agent_harness.storage.store import Store
from agent_harness.team import (
    AGENT_LIMIT_MESSAGE,
    AVATAR_IDENTITY_BANK,
    DEFAULT_AVATAR_BANK_SIZE,
    DEFAULT_TEAM_SIZE,
    build_initial_model_team,
    build_initial_team,
    build_initialization_messages,
    choose_reaction,
    format_agent_handoff_request,
    generate_team_agent,
    hire_team_agents,
    least_represented_provider,
    pick_idle_agent,
)
from agent_harness.team.assignment import assign_work_request


class TeamTests(unittest.TestCase):
    def test_default_initial_team_is_one_agent_per_provider(self):
        agents = build_initial_model_team()
        self.assertEqual(len(agents), DEFAULT_TEAM_SIZE)
        self.assertEqual(
            [agent.provider_preference for agent in agents],
            [Provider.CODEX, Provider.CLAUDE],
        )

    def test_initial_model_team_has_persistent_provider_mapping(self):
        agents = build_initial_model_team(codex_count=2, claude_count=1)
        self.assertEqual(
            [agent.provider_preference for agent in agents],
            [Provider.CODEX, Provider.CODEX, Provider.CLAUDE],
        )
        self.assertEqual(len({agent.handle for agent in agents}), 3)
        self.assertEqual([agent.handle for agent in agents], ["avery", "jordan", "morgan"])
        self.assertEqual([agent.avatar_slug for agent in agents], ["1", "2", "3"])
        self.assertEqual(len(agents[0].metadata["outside_interests"]), 3)
        self.assertEqual(agents[0].metadata["backstory"], agents[0].metadata["personal_context"])
        self.assertIn("avatar", agents[0].metadata["avatar_prompt"].lower())
        self.assertIn("cartoon", agents[0].metadata["avatar_prompt"].lower())
        self.assertEqual(
            agents[0].metadata["avatar_path"],
            f"docs/assets/avatars/{agents[0].avatar_slug}.png",
        )

    def test_default_legacy_initial_team_size_is_two(self):
        agents = build_initial_team()
        self.assertEqual(len(agents), 2)
        self.assertEqual(
            [agent.provider_preference for agent in agents],
            [Provider.CODEX, Provider.CLAUDE],
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

    def test_team_size_cannot_exceed_avatar_bank_limit(self):
        with self.assertRaisesRegex(ValueError, AGENT_LIMIT_MESSAGE):
            build_initial_model_team(codex_count=251, claude_count=250)
        with self.assertRaisesRegex(ValueError, AGENT_LIMIT_MESSAGE):
            build_initial_team(501)

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

    def test_assignment_seeds_roster_summary_from_assignment_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)

                result = assign_work_request(
                    store,
                    WorkRequest(
                        prompt="Which PR should I look at first",
                        assignment_mode=AssignmentMode.ANYONE,
                    ),
                    "C1",
                    extra_metadata={
                        ASSIGNMENT_PROMPT_METADATA_KEY: (
                            "tell me what are my open PRs in talos repo"
                        )
                    },
                )

                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(
                    result.task.metadata[ROSTER_SUMMARY_METADATA_KEY],
                    "tell me what are my open PRs in talos repo",
                )
            finally:
                store.close()

    def test_assignment_seeds_pr_urls_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)

                result = assign_work_request(
                    store,
                    WorkRequest(
                        prompt="review https://github.com/acme/app/pull/42",
                        assignment_mode=AssignmentMode.ANYONE,
                        pr_url="https://github.com/acme/app/pull/42",
                    ),
                    "C1",
                )

                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(
                    result.task.metadata[PR_URLS_METADATA_KEY],
                    ["https://github.com/acme/app/pull/42"],
                )
            finally:
                store.close()

    def test_assignment_merges_request_and_context_pr_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)

                result = assign_work_request(
                    store,
                    WorkRequest(
                        prompt="review https://github.com/acme/app/pull/43",
                        assignment_mode=AssignmentMode.ANYONE,
                        pr_url="https://github.com/acme/app/pull/43",
                    ),
                    "C1",
                    extra_metadata={PR_URLS_METADATA_KEY: ["https://github.com/acme/app/pull/42"]},
                )

                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(
                    result.task.metadata[PR_URLS_METADATA_KEY],
                    [
                        "https://github.com/acme/app/pull/43",
                        "https://github.com/acme/app/pull/42",
                    ],
                )
            finally:
                store.close()

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

    def test_randomized_hire_can_reuse_fired_avatar_capacity(self):
        all_agents = build_initial_model_team(codex_count=250, claude_count=250)

        hired = hire_team_agents(
            all_agents,
            1,
            Provider.CODEX,
            start_sort_order=500,
            avatar_agents=[],
            randomize_identities=True,
            rng=random.Random(11),
        )

        self.assertEqual(len(hired), 1)
        self.assertNotIn(hired[0].handle, {agent.handle for agent in all_agents})

    def test_handle_base_reuses_fired_agent_first_name(self):
        vera_index = next(
            identity.avatar_index
            for identity in AVATAR_IDENTITY_BANK
            if identity.full_name == "Vera Martinez"
        )
        fired_vera = replace(
            generate_team_agent(0, avatar_index=vera_index),
            handle="vera",
            status=TeamAgentStatus.FIRED,
        )

        hired = hire_team_agents(
            [fired_vera],
            1,
            Provider.CODEX,
            start_sort_order=vera_index - 1,
        )

        self.assertEqual(hired[0].handle, "vera")

    def test_handle_uses_last_initial_when_same_first_name_is_active(self):
        vera_martinez_index = next(
            identity.avatar_index
            for identity in AVATAR_IDENTITY_BANK
            if identity.full_name == "Vera Martinez"
        )

        agent = generate_team_agent(
            0,
            {"vera"},
            Provider.CODEX,
            avatar_index=vera_martinez_index,
            active_first_names={"vera"},
        )

        self.assertEqual(agent.handle, "veram")

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

    def test_handoff_request_uses_plain_target_handle_on_new_paragraph(self):
        sender, target = build_initial_model_team(codex_count=2, claude_count=0)

        text = format_agent_handoff_request(sender, target, "continue using my review above.")

        self.assertEqual(
            text,
            (
                f"Passing this back from @{sender.handle}.\n\n"
                f"@{target.handle} please continue using my review above."
            ),
        )
        self.assertNotIn("`@", text)


if __name__ == "__main__":
    unittest.main()
