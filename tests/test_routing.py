import unittest

from agent_harness.models import AgentTaskKind, AssignmentMode
from agent_harness.routing import parse_lightweight_handles, parse_work_request


class RoutingTests(unittest.TestCase):
    def test_parse_anyone_request(self):
        request = parse_work_request("Somebody do update the README", ["riley"])
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.assignment_mode, AssignmentMode.ANYONE)
        self.assertEqual(request.prompt, "update the README")

    def test_parse_anyone_request_without_helper_verb(self):
        request = parse_work_request("Somebody update the README", ["riley"])
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.assignment_mode, AssignmentMode.ANYONE)
        self.assertEqual(request.prompt, "update the README")

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
        self.assertEqual(request.author_handle, "riley")
        self.assertEqual(request.pr_url, "https://github.com/acme/app/pull/42")

    def test_parse_lightweight_handles(self):
        self.assertEqual(parse_lightweight_handles("ask @Riley and @sage"), ["riley", "sage"])


if __name__ == "__main__":
    unittest.main()
