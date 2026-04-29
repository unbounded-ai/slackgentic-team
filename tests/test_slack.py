import unittest

from agent_harness.models import Provider
from agent_harness.slack import (
    AgentRosterStatus,
    build_external_session_capacity_blocks,
    build_setup_modal,
    build_start_session_modal,
    build_task_thread_blocks,
    build_team_roster_blocks,
    dangerous_flag,
    decode_action_value,
    encode_action_value,
    is_dependency_intent,
    normalize_slack_mrkdwn,
    pack_slack_ts,
    parse_agent_handles,
    parse_thread_ref,
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

    def test_roster_blocks_include_fire_buttons(self):
        blocks = build_team_roster_blocks(build_initial_model_team(codex_count=1, claude_count=1))
        self.assertIn("Codex 1 / Claude 1", blocks[0]["text"]["text"])
        action_ids = [
            element["action_id"]
            for block in blocks
            if block.get("type") == "actions"
            and str(block.get("block_id", "")).startswith("team.agent.")
            for element in block["elements"]
        ]
        self.assertEqual(action_ids, ["team.fire", "team.fire"])

    def test_roster_blocks_include_free_up_before_fire_for_occupied_task(self):
        agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
        blocks = build_team_roster_blocks(
            [agent],
            {
                agent.agent_id: AgentRosterStatus(
                    "Occupied",
                    "Slack task: do the thing",
                    thread_url="https://example.slack.com/archives/C1/p171000001",
                    task_id="task_1",
                )
            },
        )
        action_block = next(
            block
            for block in blocks
            if block.get("block_id") == f"team.agent.actions.{agent.agent_id}"
        )

        status_text = str(blocks[2])
        self.assertNotIn("https://example.slack.com/archives/C1/p171000001", status_text)
        self.assertEqual(
            [element["text"]["text"] for element in action_block["elements"]],
            ["Free up", "Open thread", "Fire"],
        )
        self.assertEqual(
            decode_action_value(action_block["elements"][0]["value"]),
            {"v": 1, "action": "task.done", "task_id": "task_1"},
        )
        self.assertEqual(
            action_block["elements"][1]["url"],
            "https://example.slack.com/archives/C1/p171000001",
        )

    def test_roster_action_block_has_unique_action_ids(self):
        blocks = build_team_roster_blocks(build_initial_model_team(codex_count=1, claude_count=1))
        action_block = next(block for block in blocks if block.get("type") == "actions")
        action_ids = [element["action_id"] for element in action_block["elements"]]

        self.assertEqual(action_ids, ["team.hire.auto", "team.hire.codex", "team.hire.claude"])
        self.assertEqual(len(action_ids), len(set(action_ids)))

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

        self.assertEqual(button["text"]["text"], "Hire 1 Claude agent")
        self.assertEqual(
            decode_action_value(button["value"]),
            {"v": 1, "action": "team.hire", "count": 1, "provider": "claude"},
        )

    def test_normalize_slack_mrkdwn_converts_double_asterisk_outside_code_blocks(self):
        self.assertEqual(
            normalize_slack_mrkdwn("**Bold** and ```**literal**```"),
            "*Bold* and ```**literal**```",
        )

    def test_parse_agent_handles(self):
        self.assertEqual(parse_agent_handles("please ask @Riley and @sage"), ["riley", "sage"])


if __name__ == "__main__":
    unittest.main()
