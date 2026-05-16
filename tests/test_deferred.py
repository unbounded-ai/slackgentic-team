from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime

from agent_harness.deferred import (
    AGENT_DEFERRED_SIGNAL_PREFIX,
    build_deferred_resolution_prompt,
    is_agent_deferred_signal,
    looks_like_deferred_request,
    parse_agent_deferred_signal,
)
from agent_harness.models import (
    AgentTaskKind,
    AssignmentMode,
    PermissionMode,
    WorkDependencyKind,
)

PERMALINK = "https://example.slack.com/archives/CABC/p1700000000000000"


class DeferredDetectorTests(unittest.TestCase):
    def test_detects_after_with_permalink_and_followon(self):
        text = f"after {PERMALINK} finishes, check the deploy"
        self.assertTrue(looks_like_deferred_request(text))

    def test_detects_wait_for_with_handle_and_followon(self):
        text = "wait for @riley to finish, then summarize the report"
        self.assertTrue(looks_like_deferred_request(text))

    def test_detects_schedule_with_dep(self):
        text = f"schedule @nell to review the dashboard in 20 minutes after {PERMALINK} finishes"
        self.assertTrue(looks_like_deferred_request(text))

    def test_rejects_legacy_wait_for_this_idiom(self):
        text = f"wait for this to go in {PERMALINK}"
        self.assertFalse(looks_like_deferred_request(text))

    def test_rejects_plain_handle_message(self):
        text = "@riley please open the PR"
        self.assertFalse(looks_like_deferred_request(text))

    def test_rejects_schedule_without_gate(self):
        text = "schedule @riley to check the deploy tomorrow at 9am PT"
        self.assertFalse(looks_like_deferred_request(text))

    def test_rejects_after_without_dep_target(self):
        text = "after we ship, do a retro"
        self.assertFalse(looks_like_deferred_request(text))


class DeferredSignalTests(unittest.TestCase):
    def test_is_agent_deferred_signal(self):
        self.assertTrue(is_agent_deferred_signal(f"{AGENT_DEFERRED_SIGNAL_PREFIX}{{}}"))
        self.assertFalse(is_agent_deferred_signal("SLACKGENTIC: SCHEDULE {}"))

    def _signal(self, payload: dict) -> str:
        return AGENT_DEFERRED_SIGNAL_PREFIX + json.dumps(payload)

    def test_parses_thread_dep_with_permalink(self):
        result = parse_agent_deferred_signal(
            self._signal(
                {
                    "task": "check the deploy",
                    "target": "somebody",
                    "depends_on": [{"kind": "thread", "permalink": PERMALINK}],
                }
            ),
            known_handles=["riley"],
            occupied_task_ids={},
        )
        self.assertIsNone(result.error)
        deferred = result.deferred
        assert deferred is not None
        self.assertEqual(deferred.request.prompt, "check the deploy")
        self.assertEqual(deferred.request.assignment_mode, AssignmentMode.ANYONE)
        self.assertEqual(deferred.request.task_kind, AgentTaskKind.WORK)
        self.assertEqual(len(deferred.depends_on), 1)
        dep = deferred.depends_on[0]
        self.assertEqual(dep.kind, WorkDependencyKind.THREAD)
        self.assertEqual(dep.permalink, PERMALINK)

    def test_parses_agent_busy_with_known_task_id(self):
        result = parse_agent_deferred_signal(
            self._signal(
                {
                    "task": "summarize the report",
                    "target": "nell",
                    "depends_on": [{"kind": "agent_busy", "handle": "riley"}],
                }
            ),
            known_handles=["riley", "nell"],
            occupied_task_ids={"riley": "task_xyz"},
        )
        self.assertIsNone(result.error)
        deferred = result.deferred
        assert deferred is not None
        self.assertEqual(deferred.request.requested_handle, "nell")
        self.assertEqual(deferred.depends_on[0].task_id, "task_xyz")

    def test_agent_busy_rejects_idle_agent(self):
        result = parse_agent_deferred_signal(
            self._signal(
                {
                    "task": "summarize",
                    "target": "somebody",
                    "depends_on": [{"kind": "agent_busy", "handle": "riley"}],
                }
            ),
            known_handles=["riley"],
            occupied_task_ids={},
        )
        self.assertIn("not currently occupied", result.error or "")

    def test_dangerous_mode_propagates(self):
        result = parse_agent_deferred_signal(
            self._signal(
                {
                    "task": "ship the patch",
                    "target": "riley",
                    "dangerous_mode": True,
                    "depends_on": [{"kind": "thread", "permalink": PERMALINK}],
                }
            ),
            known_handles=["riley"],
            occupied_task_ids={},
        )
        self.assertIsNone(result.error)
        deferred = result.deferred
        assert deferred is not None
        self.assertEqual(deferred.request.permission_mode, PermissionMode.DANGEROUS)

    def test_delay_and_run_at_are_mutually_exclusive(self):
        result = parse_agent_deferred_signal(
            self._signal(
                {
                    "task": "x",
                    "target": "somebody",
                    "depends_on": [{"kind": "thread", "permalink": PERMALINK}],
                    "delay": {"seconds": 60},
                    "run_at": "2030-01-01T00:00:00Z",
                }
            ),
            known_handles=["riley"],
            occupied_task_ids={},
        )
        self.assertIn("delay or run_at", result.error or "")

    def test_delay_parses_seconds(self):
        result = parse_agent_deferred_signal(
            self._signal(
                {
                    "task": "x",
                    "target": "somebody",
                    "depends_on": [{"kind": "thread", "permalink": PERMALINK}],
                    "delay": {"seconds": 1200},
                }
            ),
            known_handles=["riley"],
            occupied_task_ids={},
        )
        self.assertIsNone(result.error)
        deferred = result.deferred
        assert deferred is not None
        self.assertEqual(deferred.after_dep_delay_seconds, 1200)

    def test_run_at_parses(self):
        run_at = datetime(2030, 1, 1, 12, tzinfo=UTC)
        result = parse_agent_deferred_signal(
            self._signal(
                {
                    "task": "x",
                    "target": "somebody",
                    "depends_on": [{"kind": "thread", "permalink": PERMALINK}],
                    "run_at": run_at.isoformat(),
                }
            ),
            known_handles=["riley"],
            occupied_task_ids={},
        )
        self.assertIsNone(result.error)
        deferred = result.deferred
        assert deferred is not None
        self.assertEqual(deferred.run_at, run_at)

    def test_unknown_target_handle_rejected(self):
        result = parse_agent_deferred_signal(
            self._signal(
                {
                    "task": "x",
                    "target": "stranger",
                    "depends_on": [{"kind": "thread", "permalink": PERMALINK}],
                }
            ),
            known_handles=["riley"],
            occupied_task_ids={},
        )
        self.assertIn("target must be", result.error or "")

    def test_empty_depends_on_rejected(self):
        result = parse_agent_deferred_signal(
            self._signal(
                {
                    "task": "x",
                    "target": "somebody",
                    "depends_on": [],
                }
            ),
            known_handles=["riley"],
            occupied_task_ids={},
        )
        self.assertIn("non-empty list", result.error or "")

    def test_prompt_lists_active_and_occupied(self):
        prompt = build_deferred_resolution_prompt(
            "after @riley finishes, do X",
            ["riley", "nell"],
            occupied=[{"handle": "riley", "task_id": "task_abc"}],
            now=datetime(2030, 1, 1, tzinfo=UTC),
        )
        self.assertIn("@riley, @nell", prompt)
        self.assertIn("@riley is currently working on task_id=task_abc", prompt)
        self.assertIn(AGENT_DEFERRED_SIGNAL_PREFIX, prompt)


if __name__ == "__main__":
    unittest.main()
