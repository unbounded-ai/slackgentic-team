import unittest
from dataclasses import replace

from agent_harness.models import AgentTaskKind, AssignmentMode
from agent_harness.team import build_initial_model_team
from agent_harness.team.routing import (
    canonical_agent_handle,
    canonicalize_agent_mentions,
    parse_lightweight_handles,
    parse_work_request,
)


class RoutingTests(unittest.TestCase):
    def test_parse_anyone_request(self):
        request = parse_work_request("Somebody do update the README", ["riley"])
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.assignment_mode, AssignmentMode.ANYONE)
        self.assertEqual(request.prompt, "do update the README")

    def test_parse_anyone_request_without_helper_verb(self):
        request = parse_work_request("Somebody update the README", ["riley"])
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.assignment_mode, AssignmentMode.ANYONE)
        self.assertEqual(request.prompt, "update the README")

    def test_parse_anyone_request_preserves_do_as_prompt_text(self):
        request = parse_work_request("Somebody do it better", ["riley"])
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.assignment_mode, AssignmentMode.ANYONE)
        self.assertEqual(request.prompt, "do it better")

    def test_parse_anyone_request_preserves_handle_as_prompt_text(self):
        request = parse_work_request("Somebody handle it better", ["riley"])
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.assignment_mode, AssignmentMode.ANYONE)
        self.assertEqual(request.prompt, "handle it better")

    def test_parse_anyone_can_request_preserves_task_phrase_after_can(self):
        request = parse_work_request("Somebody can handle it better", ["riley"])
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.assignment_mode, AssignmentMode.ANYONE)
        self.assertEqual(request.prompt, "handle it better")

    def test_parse_specific_handle_request(self):
        request = parse_work_request("@riley handle the failing test", ["riley"])
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.assignment_mode, AssignmentMode.SPECIFIC)
        self.assertEqual(request.requested_handle, "riley")

    def test_parse_specific_handle_request_without_helper_verb(self):
        request = parse_work_request("@riley update the failing test", ["riley"])
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.assignment_mode, AssignmentMode.SPECIFIC)
        self.assertEqual(request.requested_handle, "riley")
        self.assertEqual(request.prompt, "update the failing test")

    def test_parse_review_request_with_author_and_pr_url(self):
        request = parse_work_request(
            "Somebody review @riley's PR https://github.com/acme/app/pull/42",
            ["riley", "sage"],
        )
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.task_kind, AgentTaskKind.REVIEW)
        self.assertEqual(
            request.prompt,
            "review @riley's PR https://github.com/acme/app/pull/42",
        )
        self.assertEqual(request.author_handle, "riley")
        self.assertEqual(request.pr_url, "https://github.com/acme/app/pull/42")

    def test_parse_non_pr_review_request_as_review_task(self):
        request = parse_work_request(
            "Somebody review the repo and suggest cleanup improvements",
            ["riley", "sage"],
        )
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.assignment_mode, AssignmentMode.ANYONE)
        self.assertEqual(request.task_kind, AgentTaskKind.REVIEW)
        self.assertEqual(request.prompt, "review the repo and suggest cleanup improvements")

    def test_parse_lightweight_handles(self):
        self.assertEqual(parse_lightweight_handles("ask @Riley and @sage"), ["riley", "sage"])

    def test_canonicalize_unique_display_name_alias(self):
        agent = replace(
            build_initial_model_team(1, 0)[0],
            full_name="Mina Adebayo",
            handle="minaa",
        )

        self.assertEqual(
            canonicalize_agent_mentions("@mina try again", [agent]), "@minaa try again"
        )

    def test_canonicalize_single_character_handle_typo_when_unique(self):
        agent = replace(build_initial_model_team(1, 0)[0], handle="minaa")

        self.assertEqual(canonical_agent_handle("miaa", [agent]), "minaa")

    def test_canonicalize_does_not_guess_ambiguous_typo(self):
        agents = build_initial_model_team(2, 0)
        mina = replace(agents[0], handle="minaa")
        mira = replace(agents[1], handle="miraa")

        self.assertIsNone(canonical_agent_handle("miaa", [mina, mira]))


if __name__ == "__main__":
    unittest.main()
