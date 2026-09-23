import unittest
from dataclasses import replace

from agent_harness.models import (
    ASSIGNMENT_PROMPT_METADATA_KEY,
    ORIGINAL_TASK_METADATA_KEY,
    PR_URLS_METADATA_KEY,
    ROSTER_SUMMARY_METADATA_KEY,
    Provider,
    TeamAgentKind,
)
from agent_harness.slack import (
    SLACK_MAX_MESSAGE_BLOCKS,
    AgentRosterStatus,
    build_external_session_capacity_blocks,
    build_setup_modal,
    build_start_session_modal,
    build_task_thread_blocks,
    build_team_roster_blocks,
    build_update_prompt_blocks,
    dangerous_flag,
    decode_action_value,
    encode_action_value,
    format_external_session_capacity_text,
    is_dependency_intent,
    normalize_slack_mrkdwn,
    pack_slack_ts,
    parse_agent_handles,
    parse_thread_ref,
    slack_blocks_for_markdown_table,
    unpack_slack_permalink_ts,
)
from agent_harness.team import build_initial_model_team


class SlackTests(unittest.TestCase):
    def test_timestamp_pack_round_trip(self):
        ts = "1712345678.901234"
        self.assertEqual(unpack_slack_permalink_ts(pack_slack_ts(ts)), ts)

    def test_parse_permalink(self):
        ref = parse_thread_ref("see https://example.slack.com/archives/C123/p1712345678901234")
        self.assertIsNotNone(ref)
        assert ref is not None
        self.assertEqual(ref.channel_id, "C123")
        self.assertEqual(ref.thread_ts, "1712345678.901234")

    def test_parse_current_thread_fallback(self):
        ref = parse_thread_ref("wait for this", "C1", "171.000001")
        self.assertIsNotNone(ref)
        assert ref is not None
        self.assertEqual(ref.channel_id, "C1")
        self.assertEqual(ref.thread_ts, "171.000001")

    def test_dependency_intent(self):
        self.assertTrue(is_dependency_intent("wait for this to go in and then finish"))
        self.assertFalse(is_dependency_intent("please run tests"))

    def test_dangerous_flags(self):
        self.assertEqual(
            dangerous_flag(Provider.CODEX),
            "--dangerously-bypass-approvals-and-sandbox",
        )
        self.assertEqual(dangerous_flag(Provider.CLAUDE), "--dangerously-skip-permissions")

    def test_start_modal_has_agent_options(self):
        modal = build_start_session_modal()
        self.assertEqual(modal["type"], "modal")
        options = modal["blocks"][0]["element"]["options"]
        self.assertEqual([item["value"] for item in options], ["codex", "claude"])

    def test_setup_modal_asks_for_model_counts(self):
        modal = build_setup_modal(default_repo_root="/tmp/projects")
        block_ids = [block["block_id"] for block in modal["blocks"]]
        self.assertIn("codex_count", block_ids)
        self.assertIn("claude_count", block_ids)
        self.assertIn("repo_root", block_ids)
        counts = {
            block["block_id"]: block["element"]["initial_value"]
            for block in modal["blocks"]
            if block["block_id"] in {"codex_count", "claude_count"}
        }
        self.assertEqual(counts, {"codex_count": "1", "claude_count": "1"})
        repo_root = next(block for block in modal["blocks"] if block["block_id"] == "repo_root")
        self.assertEqual(repo_root["element"]["initial_value"], "/tmp/projects")

    def test_action_value_round_trip(self):
        value = encode_action_value("team.hire", count=2, provider="codex")
        self.assertEqual(
            decode_action_value(value),
            {"v": 1, "action": "team.hire", "count": 2, "provider": "codex"},
        )

    def _cards(self, blocks):
        return {
            str(card["block_id"]).removeprefix("team.agent."): card
            for block in blocks
            if block.get("type") == "carousel"
            for card in block["elements"]
        }

    def _labels(self, card):
        return [element["text"]["text"] for element in card["actions"]]

    def test_roster_blocks_include_fire_buttons(self):
        agents = build_initial_model_team(codex_count=1, claude_count=1)
        blocks = build_team_roster_blocks(agents)
        self.assertIn("Codex 1 / Claude 1", blocks[0]["text"]["text"])
        cards = self._cards(blocks)
        self.assertEqual(
            [[element["action_id"] for element in cards[a.agent_id]["actions"]] for a in agents],
            [["roster.work.assign", "team.fire"], ["roster.work.assign", "team.fire"]],
        )

    def test_roster_blocks_include_free_up_before_fire_for_occupied_task(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        blocks = build_team_roster_blocks(
            [agent],
            {
                agent.agent_id: AgentRosterStatus(
                    "Working",
                    "do the thing",
                    thread_url="https://example.slack.com/archives/C1/p171000001",
                    task_id="task_1",
                )
            },
        )
        card = self._cards(blocks)[agent.agent_id]

        self.assertNotIn("https://example.slack.com/archives/C1/p171000001", card["body"]["text"])
        self.assertEqual(self._labels(card), ["Free up", "Open thread", "Fire"])
        self.assertEqual(
            decode_action_value(card["actions"][0]["value"]),
            {"v": 1, "action": "task.done", "task_id": "task_1"},
        )
        self.assertEqual(
            card["actions"][1]["url"], "https://example.slack.com/archives/C1/p171000001"
        )

    def test_roster_cards_show_status_provider_and_dangerous_mode(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        blocks = build_team_roster_blocks(
            [agent],
            {
                agent.agent_id: AgentRosterStatus(
                    "Working", "repair the installer", dangerous_mode=True
                )
            },
        )
        card = self._cards(blocks)[agent.agent_id]

        self.assertEqual(card["type"], "card")
        self.assertIn(agent.full_name, card["title"]["text"])
        self.assertIn(f"@{agent.handle}", card["title"]["text"])
        self.assertIn("🔨 Working", card["subtitle"]["text"])
        self.assertIn("codex", card["subtitle"]["text"])
        self.assertIn("⚡ dangerous", card["subtitle"]["text"])
        self.assertEqual(card["body"]["text"], "repair the installer")
        self.assertNotIn("Slack task:", str(blocks))

    def test_roster_blocks_show_pr_links_separately_from_status_summary(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        blocks = build_team_roster_blocks(
            [agent],
            {
                agent.agent_id: AgentRosterStatus(
                    "Working",
                    "shipping the status view",
                    pr_urls=(
                        "https://github.com/acme/app/pull/42",
                        "https://github.com/acme/app/pull/43",
                        "https://github.com/acme/app/pull/44",
                    ),
                )
            },
        )
        body = self._cards(blocks)[agent.agent_id]["body"]["text"]
        self.assertTrue(body.startswith("shipping the status view"))
        self.assertIn("*PRs:*", body)
        self.assertIn("<https://github.com/acme/app/pull/42|acme/app#42>", body)
        self.assertIn("+1 more", body)
        self.assertLessEqual(len(body), 200)

    def test_roster_card_bodies_stay_within_slack_limits(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        blocks = build_team_roster_blocks(
            [agent],
            {
                agent.agent_id: AgentRosterStatus(
                    "Working", "x " * 400, pr_urls=("https://github.com/acme/app/pull/1",)
                )
            },
        )
        card = self._cards(blocks)[agent.agent_id]
        self.assertLessEqual(len(card["body"]["text"]), 200)
        self.assertLessEqual(len(card["title"]["text"]), 150)
        self.assertLessEqual(len(card["actions"]), 3)

    def test_roster_blocks_sort_working_first_then_provider_then_name(self):
        agents = build_initial_model_team(codex_count=2, claude_count=2)
        shuffled = [agents[3], agents[2], agents[1], agents[0]]

        blocks = build_team_roster_blocks(
            shuffled,
            {agent.agent_id: AgentRosterStatus("Available") for agent in agents}
            | {
                agents[1].agent_id: AgentRosterStatus("Working", "codex work"),
                agents[2].agent_id: AgentRosterStatus("Working", "claude work"),
            },
        )

        self.assertEqual(
            list(self._cards(blocks)),
            [agents[1].agent_id, agents[2].agent_id, agents[0].agent_id, agents[3].agent_id],
        )
        headings = [block["block_id"] for block in blocks if block.get("type") == "context"]
        self.assertEqual(headings, ["team.section.working", "team.section.available"])

    def test_roster_team_actions_offer_add_work_and_every_hire_option(self):
        blocks = build_team_roster_blocks(build_initial_model_team(codex_count=1, claude_count=1))
        action_block = next(
            block for block in blocks if block.get("block_id") == "team.roster.actions"
        )
        add_work, hire = action_block["elements"]
        self.assertEqual(add_work["action_id"], "roster.work.assign")
        self.assertEqual(hire["type"], "static_select")
        payloads = [decode_action_value(option["value"]) for option in hire["options"]]
        self.assertEqual({payload["action"] for payload in payloads}, {"team.hire"})
        self.assertEqual(
            [(payload.get("provider"), payload.get("kind")) for payload in payloads],
            [
                (None, None),
                (Provider.CODEX.value, None),
                (Provider.CLAUDE.value, None),
                (Provider.CODEX.value, TeamAgentKind.PM.value),
                (Provider.CLAUDE.value, TeamAgentKind.PM.value),
            ],
        )

    def test_roster_groups_engineers_and_pms_separately(self):
        engineers = build_initial_model_team(codex_count=1, claude_count=0)
        pm = replace(engineers[0], agent_id="pm-1", handle="pm-one", kind=TeamAgentKind.PM)
        blocks = build_team_roster_blocks([engineers[0], pm])

        headings = [block.get("block_id") for block in blocks if block.get("type") == "context"]
        self.assertEqual(headings, ["team.section.pms", "team.section.available"])
        cards = self._cards(blocks)
        self.assertNotIn("PM", cards[engineers[0].agent_id]["subtitle"]["text"])
        self.assertIn("PM", cards["pm-1"]["subtitle"]["text"])

    def test_roster_omits_loop_agents_entirely(self):
        engineer = build_initial_model_team(codex_count=1, claude_count=0)[0]
        loop_agent = replace(
            engineer,
            agent_id="loop-1",
            handle="billing-loop",
            full_name="Billing Bot",
            kind=TeamAgentKind.LOOP,
        )

        rendered = str(build_team_roster_blocks([engineer, loop_agent]))

        self.assertIn("1 agent ·", rendered)
        self.assertIn(engineer.handle, rendered)
        self.assertNotIn("billing-loop", rendered)
        self.assertNotIn("Billing Bot", rendered)

    def test_roster_renders_assign_project_for_pms(self):
        engineer = build_initial_model_team(codex_count=1, claude_count=0)[0]
        pm = replace(engineer, agent_id="pm-1", handle="pm-one", kind=TeamAgentKind.PM)
        cards = self._cards(build_team_roster_blocks([engineer, pm]))

        self.assertIn("Assign", self._labels(cards[engineer.agent_id]))
        self.assertNotIn("Assign Project", self._labels(cards[engineer.agent_id]))
        self.assertIn("Assign Project", self._labels(cards["pm-1"]))

    def test_roster_shows_every_agent_within_the_slack_block_limit(self):
        # Slack rejects more than 50 blocks; cards in carousels keep large teams whole.
        for team_size in range(1, 60):
            agents = build_initial_model_team(codex_count=team_size, claude_count=0)
            pm = replace(agents[0], agent_id="pm-1", handle="pm-one", kind=TeamAgentKind.PM)
            blocks = build_team_roster_blocks([*agents, pm])
            self.assertLessEqual(len(blocks), SLACK_MAX_MESSAGE_BLOCKS)
            cards = self._cards(blocks)
            self.assertEqual(len(cards), team_size + 1)
            self.assertIn("pm-1", cards)
            self.assertTrue(
                all(len(block["elements"]) <= 10 for block in blocks if block["type"] == "carousel")
            )

    def test_task_blocks_only_include_finish_button(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        from agent_harness.team import create_agent_task

        task = create_agent_task(agent, "do the thing", "C1")

        blocks = build_task_thread_blocks(task, agent)
        action_block = next(block for block in blocks if block.get("type") == "actions")
        elements = action_block["elements"]

        self.assertEqual(
            [item["text"]["text"] for item in elements], ["Finish and free up this agent"]
        )
        self.assertEqual([item["action_id"] for item in elements], ["task.done"])

    def test_task_blocks_show_original_task_and_latest_summary(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        from agent_harness.team import create_agent_task

        task = replace(
            create_agent_task(agent, "tiny latest prompt", "C1"),
            metadata={
                ASSIGNMENT_PROMPT_METADATA_KEY: "fix the task pickup message",
                ROSTER_SUMMARY_METADATA_KEY: (
                    "Roster UX fix: refreshing Priya's task thread header"
                ),
            },
        )

        rendered = str(build_task_thread_blocks(task, agent))

        self.assertIn("fix the task pickup message", rendered)
        self.assertIn("Roster UX fix: refreshing Priya's task thread header", rendered)
        self.assertNotIn("tiny latest prompt", rendered)

    def test_task_blocks_omit_duplicate_latest_summary(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        from agent_harness.team import create_agent_task

        task = replace(
            create_agent_task(agent, "ship the status view", "C1"),
            metadata={
                ASSIGNMENT_PROMPT_METADATA_KEY: "ship the status view",
                ROSTER_SUMMARY_METADATA_KEY: "ship the status view",
            },
        )

        rendered = str(build_task_thread_blocks(task, agent))

        self.assertIn("ship the status view", rendered)
        self.assertNotIn("Latest:", rendered)

    def test_task_blocks_preserve_first_original_task_when_assignment_changes(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        from agent_harness.team import create_agent_task

        task = replace(
            create_agent_task(agent, "third prompt", "C1"),
            metadata={
                ORIGINAL_TASK_METADATA_KEY: "first original task description",
                ASSIGNMENT_PROMPT_METADATA_KEY: "second assignment prompt",
                ROSTER_SUMMARY_METADATA_KEY: "currently validating the release",
            },
        )

        rendered = str(build_task_thread_blocks(task, agent))

        self.assertIn("first original task description", rendered)
        self.assertIn("Latest: currently validating the release", rendered)
        self.assertNotIn("second assignment prompt", rendered)
        self.assertNotIn("third prompt", rendered)

    def test_task_blocks_show_pr_links_from_metadata(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        from agent_harness.team import create_agent_task

        task = replace(
            create_agent_task(agent, "ship the status view", "C1"),
            metadata={
                PR_URLS_METADATA_KEY: [
                    "https://github.com/acme/app/pull/42",
                    "https://github.com/acme/app/pull/43",
                ]
            },
        )

        rendered = str(build_task_thread_blocks(task, agent))

        sources = build_task_thread_blocks(task, agent)[0]["sources"]
        self.assertEqual(
            sources,
            [
                {
                    "type": "url",
                    "url": "https://github.com/acme/app/pull/42",
                    "text": "acme/app#42",
                },
                {
                    "type": "url",
                    "url": "https://github.com/acme/app/pull/43",
                    "text": "acme/app#43",
                },
            ],
        )
        self.assertIn("acme/app#42", rendered)

    def test_task_blocks_are_a_live_task_card(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        from agent_harness.models import AgentTaskStatus
        from agent_harness.team import create_agent_task

        task = create_agent_task(agent, "Fix the flaky login test\nIt fails on CI only.", "C1")
        card = build_task_thread_blocks(task, agent)[0]
        self.assertEqual(card["type"], "task_card")
        self.assertEqual(card["title"], "Fix the flaky login test")
        self.assertEqual(card["status"], "in_progress")
        self.assertIn("It fails on CI only.", str(card["details"]))

        done = build_task_thread_blocks(
            replace(task, status=AgentTaskStatus.DONE), agent, include_actions=False
        )
        self.assertEqual(done[0]["status"], "complete")
        self.assertIn("✅ done", str(done[1]))

    def test_resolved_task_blocks_omit_finish_button(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        from agent_harness.team import create_agent_task

        task = create_agent_task(agent, "do the thing", "C1")

        blocks = build_task_thread_blocks(task, agent, include_actions=False)

        self.assertFalse(any(block.get("type") == "actions" for block in blocks))

    def test_external_capacity_button_matches_provider(self):
        blocks = build_external_session_capacity_blocks(Provider.CLAUDE, waiting_count=2)
        action_block = next(block for block in blocks if block.get("type") == "actions")
        button = action_block["elements"][0]

        self.assertEqual(button["text"]["text"], "Hire 2 Claude agents")
        self.assertEqual(
            decode_action_value(button["value"]),
            {"v": 1, "action": "team.hire", "count": 2, "provider": "claude"},
        )
        self.assertEqual(
            format_external_session_capacity_text(Provider.CLAUDE, waiting_count=2),
            "No Claude team seat is available for sessions started outside Slack. "
            "2 Claude sessions are waiting. Hire 2 matching agents and Slackgentic "
            "will backfill the transcript.",
        )

    def test_normalize_slack_mrkdwn_converts_double_asterisk_outside_code_blocks(self):
        self.assertEqual(
            normalize_slack_mrkdwn("**Bold** and ```**literal**```"),
            "*Bold* and ```**literal**```",
        )

    def test_normalize_slack_mrkdwn_wraps_markdown_tables(self):
        text = (
            "**Modes**\n\n"
            "| Switch | Off | On |\n"
            "|---|---|---|\n"
            "| `tablet.tablet_mode` | dense | hybrid |\n\n"
            "Done"
        )

        rendered = normalize_slack_mrkdwn(text)

        self.assertIn("*Modes*", rendered)
        self.assertIn("```\n| Switch | Off | On |\n|---|---|---|", rendered)
        self.assertIn("| `tablet.tablet_mode` | dense | hybrid |\n```", rendered)
        self.assertIn("\nDone", rendered)

    def test_slack_blocks_for_markdown_table_renders_native_table_block(self):
        text = (
            "**Modes**\n\n"
            "| Switch | Off | On |\n"
            "|---|---|---|\n"
            "| `tablet.tablet_mode` | dense | hybrid |\n\n"
            "Done"
        )

        blocks = slack_blocks_for_markdown_table(text)

        self.assertIsNotNone(blocks)
        assert blocks is not None
        self.assertEqual([block["type"] for block in blocks], ["section", "table", "section"])
        self.assertEqual(blocks[1]["rows"][0][0]["type"], "rich_text")
        self.assertEqual(
            blocks[1]["column_settings"],
            [
                {"is_wrapped": True},
                {"is_wrapped": True},
                {"is_wrapped": True},
            ],
        )
        first_data_cell = blocks[1]["rows"][1][0]["elements"][0]["elements"][0]
        self.assertEqual(
            first_data_cell,
            {
                "type": "text",
                "text": "tablet.tablet_mode",
                "style": {"code": True},
            },
        )

    def test_slack_blocks_for_markdown_table_renders_empty_cells_as_visible_placeholder(self):
        blocks = slack_blocks_for_markdown_table("| Name | State |\n|---|---|\n| Avery | |")

        self.assertIsNotNone(blocks)
        assert blocks is not None
        empty_cell = blocks[0]["rows"][1][1]["elements"][0]["elements"][0]
        # Slack rejects whitespace-only cells ("must be more than 0 characters"),
        # so the placeholder must be a visible glyph.
        self.assertEqual(empty_cell["text"], "\u2014")
        self.assertNotIn("style", empty_cell)

    def test_slack_blocks_for_markdown_table_renders_empty_code_cells_as_visible_placeholder(self):
        blocks = slack_blocks_for_markdown_table("| Name | State |\n|---|---|\n| `` | busy |")

        self.assertIsNotNone(blocks)
        assert blocks is not None
        empty_code = blocks[0]["rows"][1][0]["elements"][0]["elements"][0]
        self.assertEqual(empty_code["text"], "\u2014")
        # An originally-empty code cell should not keep `code` styling on the
        # placeholder \u2014 that would render as `\u2014` in mono and read as content.
        self.assertNotIn("code", empty_code.get("style", {}))

    def test_parse_agent_handles(self):
        self.assertEqual(parse_agent_handles("please ask @Riley and @sage"), ["riley", "sage"])

    def test_update_prompt_blocks_show_call_to_action_only_before_status(self):
        from agent_harness.updates import ReleaseInfo, UpdateCandidate

        candidate = UpdateCandidate(
            current_version="0.1.0",
            release=ReleaseInfo(
                version="0.1.1",
                tag_name="v0.1.1",
                html_url="https://github.com/example-org/example-repo/releases/tag/v0.1.1",
                body=(
                    "## What's Changed\n"
                    "* Adds safer update restart handling by @contributor in "
                    "https://github.com/example-org/example-repo/pull/1\n"
                    "* Shortens release notes in Slack by @contributor in "
                    "https://github.com/example-org/example-repo/pull/2\n\n"
                    "**Full Changelog**: "
                    "https://github.com/example-org/example-repo/compare/v0.1.0...v0.1.1"
                ),
            ),
            repository="example-org/example-repo",
        )

        prompt = build_update_prompt_blocks(candidate)
        card = prompt[0]
        self.assertEqual(card["type"], "card")
        self.assertIn("Slackgentic 0.1.1 is out", card["title"]["text"])
        self.assertIn("0.1.0", card["subtitle"]["text"])
        self.assertIn("Upgrade now to install the published release", card["body"]["text"])
        self.assertEqual(
            [action["text"]["text"] for action in card["actions"]],
            ["Upgrade now", "What's new", "Not now"],
        )
        release_notes = prompt[1]["text"]
        self.assertIn("What's new", release_notes)
        self.assertIn("- Adds safer update restart handling", release_notes)
        self.assertIn("- Shortens release notes in Slack", release_notes)
        self.assertNotIn("@contributor", release_notes)
        self.assertNotIn("https://", release_notes)
        self.assertNotIn("Full Changelog", release_notes)

        in_progress = build_update_prompt_blocks(
            candidate,
            status_text="Installing Slackgentic v0.1.1 and preparing a restart.",
            include_actions=False,
        )
        in_progress_text = str(in_progress)
        # Once we're past the prompt stage the call-to-action reads wrong, so drop it.
        self.assertNotIn("Upgrade now to install the published release", in_progress_text)
        self.assertIn("Installing Slackgentic v0.1.1", in_progress_text)
        self.assertNotIn("actions", in_progress[0])

        done = build_update_prompt_blocks(
            candidate,
            status_text=":white_check_mark: Installed Slackgentic v0.1.1 and restarted successfully.",
            include_actions=False,
        )
        done_text = str(done)
        self.assertNotIn("Upgrade now to install the published release", done_text)
        self.assertIn(":white_check_mark:", done_text)
        self.assertIn("restarted successfully", done_text)


if __name__ == "__main__":
    unittest.main()
