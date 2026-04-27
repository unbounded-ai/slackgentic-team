import unittest

from agent_harness.models import Provider
from agent_harness.slack import (
    build_setup_modal,
    build_start_session_modal,
    build_team_roster_blocks,
    dangerous_flag,
    decode_action_value,
    encode_action_value,
    is_dependency_intent,
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
        modal = build_setup_modal()
        block_ids = [block["block_id"] for block in modal["blocks"]]
        self.assertIn("codex_count", block_ids)
        self.assertIn("claude_count", block_ids)

    def test_action_value_round_trip(self):
        value = encode_action_value("team.hire", count=2, provider="codex")
        self.assertEqual(
            decode_action_value(value),
            {"v": 1, "action": "team.hire", "count": 2, "provider": "codex"},
        )

    def test_roster_blocks_include_fire_buttons(self):
        blocks = build_team_roster_blocks(build_initial_model_team(codex_count=1, claude_count=1))
        action_ids = [
            block["accessory"]["action_id"]
            for block in blocks
            if block.get("accessory", {}).get("type") == "button"
        ]
        self.assertEqual(action_ids, ["team.fire", "team.fire"])

    def test_roster_action_block_has_unique_action_ids(self):
        blocks = build_team_roster_blocks(build_initial_model_team(codex_count=1, claude_count=1))
        action_block = next(block for block in blocks if block.get("type") == "actions")
        action_ids = [element["action_id"] for element in action_block["elements"]]

        self.assertEqual(action_ids, ["team.hire.auto", "team.hire.codex", "team.hire.claude"])
        self.assertEqual(len(action_ids), len(set(action_ids)))

    def test_parse_agent_handles(self):
        self.assertEqual(parse_agent_handles("please ask @Riley and @sage"), ["riley", "sage"])


if __name__ == "__main__":
    unittest.main()
