import unittest
from pathlib import Path

from agent_harness.models import Provider
from agent_harness.team.commands import (
    SETTING_AUTO_UPDATE,
    SETTING_UPDATE_CHECKS,
    FireCommand,
    FireEveryoneCommand,
    HelpCommand,
    HireCommand,
    RepoRootCommand,
    RosterCommand,
    ScheduledTasksCommand,
    SettingsCommand,
    TimezoneCommand,
    UnassignedExternalSessionsCommand,
    parse_team_command,
)


class TeamCommandTests(unittest.TestCase):
    def test_parse_hire_auto(self):
        self.assertEqual(parse_team_command("hire 3 new agents"), HireCommand(count=3))
        self.assertEqual(parse_team_command("team hire 2"), HireCommand(count=2))

    def test_parse_hire_provider(self):
        self.assertEqual(
            parse_team_command("hire two claude agents"),
            HireCommand(count=2, provider=Provider.CLAUDE),
        )

    def test_parse_fire(self):
        self.assertEqual(parse_team_command("fire @Riley"), FireCommand(handle="riley"))

    def test_parse_fire_everyone(self):
        self.assertEqual(parse_team_command("fire everyone"), FireEveryoneCommand())
        self.assertEqual(parse_team_command("fire all agents"), FireEveryoneCommand())

    def test_parse_roster(self):
        self.assertEqual(parse_team_command("show roster"), RosterCommand())
        self.assertEqual(parse_team_command("roster"), RosterCommand())

    def test_parse_scheduled_tasks(self):
        self.assertEqual(parse_team_command("scheduled tasks"), ScheduledTasksCommand())
        self.assertEqual(parse_team_command("show schedules"), ScheduledTasksCommand())

    def test_parse_unassigned_external_sessions(self):
        self.assertEqual(
            parse_team_command("external sessions"),
            UnassignedExternalSessionsCommand(),
        )
        self.assertEqual(
            parse_team_command("list unassigned sessions"),
            UnassignedExternalSessionsCommand(),
        )

    def test_parse_repo_root(self):
        self.assertEqual(parse_team_command("show repo root"), RepoRootCommand())
        self.assertEqual(
            parse_team_command("repo root /tmp/projects"), RepoRootCommand(Path("/tmp/projects"))
        )
        self.assertEqual(
            parse_team_command('repo root "/tmp/my projects"'),
            RepoRootCommand(Path("/tmp/my projects")),
        )


class HelpAndSessionPhrasingTests(unittest.TestCase):
    def test_help_phrasings(self):
        for text in ("help", "Help", "commands", "?", "show help", "list commands"):
            with self.subTest(text=text):
                self.assertIsInstance(parse_team_command(text), HelpCommand)

    def test_session_phrasings_all_reach_the_same_command(self):
        # "sessions" and "active sessions" are what people type first; before this
        # only the "external"/"unassigned" spellings matched and the rest fell
        # through to being treated as a task.
        for text in (
            "sessions",
            "session",
            "active sessions",
            "live sessions",
            "open sessions",
            "current sessions",
            "show sessions",
            "list sessions",
            "external sessions",
            "unassigned sessions",
            "unclaimed external sessions",
        ):
            with self.subTest(text=text):
                self.assertIsInstance(parse_team_command(text), UnassignedExternalSessionsCommand)

    def test_parse_settings(self):
        for text in ("settings", "show settings", "Preferences", "auto-update", "<@UBOT> settings"):
            with self.subTest(text=text):
                self.assertEqual(parse_team_command(text), SettingsCommand())

    def test_parse_setting_switches(self):
        cases = {
            "auto-update on": SettingsCommand(SETTING_AUTO_UPDATE, True),
            "auto update off": SettingsCommand(SETTING_AUTO_UPDATE, False),
            "settings autoupdate: off": SettingsCommand(SETTING_AUTO_UPDATE, False),
            "set auto-update to on": SettingsCommand(SETTING_AUTO_UPDATE, True),
            "disable auto-updates": SettingsCommand(SETTING_AUTO_UPDATE, False),
            "enable update checks": SettingsCommand(SETTING_UPDATE_CHECKS, True),
            "release checks off": SettingsCommand(SETTING_UPDATE_CHECKS, False),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_team_command(text), expected)

    def test_parse_timezone(self):
        for text, expected in (
            ("timezone", TimezoneCommand()),
            ("show my timezone", TimezoneCommand()),
            ("what is my timezone?", TimezoneCommand()),
            ("timezone America/New_York", TimezoneCommand("America/New_York")),
            ("set timezone to Europe/Berlin", TimezoneCommand("Europe/Berlin")),
            ("time zone: UTC", TimezoneCommand("UTC")),
            ("set timezone mars", TimezoneCommand("mars")),
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_team_command(text), expected)

    def test_near_misses_are_not_commands(self):
        for text in (
            "helpful stuff",
            "session notes",
            "help me fix this",
            "sessions are slow",
            "settings are confusing",
            "auto-update broke again",
            "timezone bug",
            "timezone is wrong on the dashboard",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_team_command(text))


if __name__ == "__main__":
    unittest.main()
