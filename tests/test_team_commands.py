import unittest
from pathlib import Path

from agent_harness.models import Provider
from agent_harness.team.commands import (
    FireCommand,
    FireEveryoneCommand,
    HelpCommand,
    HireCommand,
    RepoRootCommand,
    RosterCommand,
    ScheduledTasksCommand,
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

    def test_near_misses_are_not_commands(self):
        for text in ("helpful stuff", "session notes", "help me fix this", "sessions are slow"):
            with self.subTest(text=text):
                self.assertIsNone(parse_team_command(text))


if __name__ == "__main__":
    unittest.main()
