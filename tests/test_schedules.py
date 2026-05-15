import json
import unittest
from datetime import UTC, datetime

from agent_harness.models import AssignmentMode, ScheduledWorkKind
from agent_harness.schedules import (
    AGENT_SCHEDULE_SIGNAL_PREFIX,
    build_schedule_resolution_prompt,
    next_run_after,
    parse_agent_schedule_signal,
)


class ScheduleControlTests(unittest.TestCase):
    def test_resolution_prompt_delegates_location_dependent_time_to_agent(self):
        now = datetime(2026, 5, 15, 20, 0, tzinfo=UTC)

        prompt = build_schedule_resolution_prompt(
            "schedule @avery check the patio lights during tomorrow's sunset time in Waco",
            ["avery", "jordan"],
            now=now,
        )

        self.assertIn("tomorrow's sunset time in Waco", prompt)
        self.assertIn("location-dependent wording such as sunset", prompt)
        self.assertIn(AGENT_SCHEDULE_SIGNAL_PREFIX, prompt)
        self.assertIn('"run_at"', prompt)

    def test_one_off_schedule_signal_validates_structured_json(self):
        payload = {
            "task": "check the patio lights",
            "target": "avery",
            "task_kind": "work",
            "schedule": {
                "kind": "one_off",
                "run_at": "2026-05-16T01:20:00Z",
                "timezone": "America/Chicago",
                "description": "tomorrow at sunset in Waco",
            },
        }

        parsed = parse_agent_schedule_signal(
            f"{AGENT_SCHEDULE_SIGNAL_PREFIX}{json.dumps(payload)}",
            known_handles=["avery", "jordan"],
            now=datetime(2026, 5, 15, 20, 0, tzinfo=UTC),
        )

        self.assertIsNone(parsed.error)
        assert parsed.schedule is not None
        self.assertEqual(parsed.schedule.schedule_kind, ScheduledWorkKind.ONE_OFF)
        self.assertEqual(parsed.schedule.request.assignment_mode, AssignmentMode.SPECIFIC)
        self.assertEqual(parsed.schedule.request.requested_handle, "avery")
        self.assertEqual(parsed.schedule.request.prompt, "check the patio lights")
        self.assertEqual(parsed.schedule.description, "tomorrow at sunset in Waco")

    def test_recurring_schedule_signal_can_compute_next_run_when_missing(self):
        payload = {
            "task": "review the nightly report",
            "target": "somebody",
            "schedule": {
                "kind": "recurring",
                "frequency": "daily",
                "time": "17:00",
                "timezone": "America/New_York",
            },
        }

        parsed = parse_agent_schedule_signal(
            f"{AGENT_SCHEDULE_SIGNAL_PREFIX}{json.dumps(payload)}",
            known_handles=["avery", "jordan"],
            now=datetime(2026, 5, 15, 20, 0, tzinfo=UTC),
        )

        self.assertIsNone(parsed.error)
        assert parsed.schedule is not None
        self.assertEqual(parsed.schedule.schedule_kind, ScheduledWorkKind.RECURRING)
        self.assertEqual(
            parsed.schedule.next_run_at,
            datetime(2026, 5, 15, 21, 0, tzinfo=UTC),
        )
        self.assertEqual(parsed.schedule.request.assignment_mode, AssignmentMode.ANYONE)

    def test_schedule_signal_rejects_unknown_target(self):
        payload = {
            "task": "check CI",
            "target": "unknown",
            "schedule": {
                "kind": "one_off",
                "run_at": "2026-05-16T01:20:00Z",
            },
        }

        parsed = parse_agent_schedule_signal(
            f"{AGENT_SCHEDULE_SIGNAL_PREFIX}{json.dumps(payload)}",
            known_handles=["avery", "jordan"],
            now=datetime(2026, 5, 15, 20, 0, tzinfo=UTC),
        )

        self.assertIsNone(parsed.schedule)
        self.assertIn("target must be", parsed.error or "")

    def test_next_run_after_returns_next_weekly_occurrence(self):
        after = datetime(2026, 5, 15, 20, 0, tzinfo=UTC)

        next_run = next_run_after(
            {
                "frequency": "weekly",
                "weekday": 0,
                "time": "09:30",
                "timezone": "America/New_York",
            },
            after=after,
        )

        self.assertEqual(next_run, datetime(2026, 5, 18, 13, 30, tzinfo=UTC))


if __name__ == "__main__":
    unittest.main()
