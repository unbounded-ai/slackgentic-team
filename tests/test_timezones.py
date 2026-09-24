import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from agent_harness.deferred import build_deferred_resolution_prompt
from agent_harness.loops import build_loop_resolution_prompt
from agent_harness.pm import build_pm_resolution_prompt
from agent_harness.providers.usage import day_string
from agent_harness.runtime.tasks import build_task_prompt
from agent_harness.schedules import build_schedule_resolution_prompt
from agent_harness.team import build_initial_model_team, create_agent_task
from agent_harness.timezones import (
    SETTING_USER_TIMEZONE,
    SETTING_USER_TIMEZONE_SOURCE,
    TIMEZONE_SOURCE_MANUAL,
    TIMEZONE_SOURCE_SLACK,
    TIMEZONE_SOURCE_SYSTEM,
    configured_timezone,
    format_user_time,
    normalize_timezone,
    remember_inferred_timezone,
    set_user_timezone,
    system_timezone_name,
    timezone_label,
    timezone_prompt_lines,
)


class MemoryStore:
    def __init__(self):
        self.settings: dict[str, str] = {}

    def get_setting(self, key):
        return self.settings.get(key)

    def set_setting(self, key, value):
        self.settings[key] = value


class TimezoneTests(unittest.TestCase):
    def test_normalize_accepts_iana_names_and_common_spellings(self):
        self.assertEqual(normalize_timezone("America/New_York"), "America/New_York")
        self.assertEqual(normalize_timezone(" america/new_york "), "America/New_York")
        self.assertEqual(normalize_timezone("utc"), "UTC")
        self.assertEqual(normalize_timezone("GMT"), "UTC")
        self.assertIsNone(normalize_timezone("Mars/Olympus"))
        self.assertIsNone(normalize_timezone(""))
        self.assertIsNone(normalize_timezone(None))
        self.assertIsNone(normalize_timezone("../etc/passwd"))

    def test_system_timezone_reads_tz_environment(self):
        with patch.dict("os.environ", {"TZ": "Asia/Tokyo"}):
            self.assertEqual(system_timezone_name(), "Asia/Tokyo")

    def test_slack_profile_zone_is_inferred_first(self):
        store = MemoryStore()
        with patch("agent_harness.timezones.system_timezone_name", return_value="Europe/Paris"):
            zone = remember_inferred_timezone(store, "America/Denver")
        self.assertEqual(zone, "America/Denver")
        self.assertEqual(store.settings[SETTING_USER_TIMEZONE], "America/Denver")
        self.assertEqual(store.settings[SETTING_USER_TIMEZONE_SOURCE], TIMEZONE_SOURCE_SLACK)

    def test_machine_zone_is_the_fallback_when_slack_has_none(self):
        store = MemoryStore()
        with patch("agent_harness.timezones.system_timezone_name", return_value="Europe/Paris"):
            zone = remember_inferred_timezone(store, None)
        self.assertEqual(zone, "Europe/Paris")
        self.assertEqual(store.settings[SETTING_USER_TIMEZONE_SOURCE], TIMEZONE_SOURCE_SYSTEM)

    def test_a_zone_the_owner_picked_is_never_replaced(self):
        store = MemoryStore()
        set_user_timezone(store, "Asia/Kolkata", TIMEZONE_SOURCE_MANUAL)
        self.assertEqual(remember_inferred_timezone(store, "America/Denver"), "Asia/Kolkata")
        self.assertEqual(configured_timezone(store), "Asia/Kolkata")

    def test_an_inferred_zone_is_kept_when_slack_later_has_none(self):
        store = MemoryStore()
        set_user_timezone(store, "America/Denver", TIMEZONE_SOURCE_SLACK)
        with patch("agent_harness.timezones.system_timezone_name", return_value="Europe/Paris"):
            self.assertEqual(remember_inferred_timezone(store, None), "America/Denver")

    def test_format_user_time_labels_the_zone(self):
        moment = datetime(2026, 9, 23, 18, 5, tzinfo=UTC)
        self.assertEqual(format_user_time(moment, "America/New_York"), "Wed Sep 23, 2:05 PM EDT")
        self.assertEqual(format_user_time(moment, None), "Wed Sep 23, 6:05 PM UTC")

    def test_timezone_label_includes_offset(self):
        moment = datetime(2026, 1, 10, tzinfo=UTC)
        self.assertEqual(
            timezone_label("America/New_York", now=moment),
            "America/New_York (EST, UTC-05:00)",
        )
        self.assertEqual(timezone_label("UTC", now=moment), "UTC (UTC+00:00)")

    def test_prompt_lines_tell_the_agent_to_convert_times(self):
        moment = datetime(2026, 9, 23, 18, 5, tzinfo=UTC)
        text = "\n".join(timezone_prompt_lines("America/Chicago", moment))
        self.assertIn("Owner timezone: America/Chicago (CDT, UTC-05:00)", text)
        self.assertIn("Owner's current local time: 2026-09-23 13:05 CDT", text)
        self.assertIn("convert it to America/Chicago", text)
        self.assertEqual(timezone_prompt_lines(None, moment), [])

    def test_prompt_lines_without_now_are_stable(self):
        lines = timezone_prompt_lines("Europe/London")
        self.assertEqual(lines[0], "Owner timezone: Europe/London.")
        self.assertEqual(lines, timezone_prompt_lines("Europe/London"))

    def test_today_follows_the_owner_zone(self):
        late_utc = datetime(2026, 9, 24, 2, 0, tzinfo=UTC)

        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return late_utc.astimezone(tz) if tz else late_utc

        with patch("agent_harness.providers.usage.datetime", FrozenDatetime):
            self.assertEqual(day_string("today"), "2026-09-24")
            self.assertEqual(day_string("today", "America/Los_Angeles"), "2026-09-23")


class PromptTimezoneTests(unittest.TestCase):
    NOW = datetime(2026, 9, 23, 18, 5, tzinfo=UTC)

    def test_resolver_prompts_carry_the_owner_zone(self):
        prompts = {
            "schedule": build_schedule_resolution_prompt(
                "every weekday at 9am check CI", ["avery"], now=self.NOW, timezone="Asia/Tokyo"
            ),
            "deferred": build_deferred_resolution_prompt(
                "after the deploy, check CI at 5pm", ["avery"], now=self.NOW, timezone="Asia/Tokyo"
            ),
            "loop": build_loop_resolution_prompt(
                "every morning at 8 summarize alerts", now=self.NOW, timezone="Asia/Tokyo"
            ),
            "pm": build_pm_resolution_prompt(
                "ship the billing page by Friday",
                ["avery"],
                initiative_id="init_1",
                now=self.NOW,
                timezone="Asia/Tokyo",
            ),
        }
        for name, prompt in prompts.items():
            with self.subTest(prompt=name):
                self.assertIn("Owner timezone: Asia/Tokyo (JST, UTC+09:00)", prompt)
                self.assertIn("Owner's current local time: 2026-09-24 03:05 JST", prompt)
                self.assertIn("convert it to Asia/Tokyo", prompt)
        self.assertIn("When the owner names no timezone, use Asia/Tokyo", prompts["loop"])
        self.assertIn("Asia/Tokyo when it names no timezone", prompts["schedule"])

    def test_resolver_prompts_are_unchanged_without_a_zone(self):
        prompt = build_schedule_resolution_prompt("check CI at 9am", ["avery"], now=self.NOW)
        self.assertNotIn("Owner timezone", prompt)

    def test_task_prompt_asks_the_agent_to_convert_times(self):
        agent = build_initial_model_team(1, 0)[0]
        task = create_agent_task(agent, "look at the deploy logs", "C1")
        prompt = build_task_prompt(agent, task, timezone="Europe/London")
        self.assertIn("Owner timezone: Europe/London.", prompt)
        self.assertIn("convert it to Europe/London", prompt)
        self.assertEqual(prompt, build_task_prompt(agent, task, timezone="Europe/London"))
        self.assertNotIn("Owner timezone", build_task_prompt(agent, task))


if __name__ == "__main__":
    unittest.main()
