import unittest
from pathlib import Path

from agent_harness.models import Provider
from agent_harness.team.commands import (
    FireCommand,
    FireEveryoneCommand,
    HireCommand,
    RepoRootCommand,
    RosterCommand,
    parse_team_command,
)


class TeamCommandTests(unittest.TestCase):
    def test_parse_hire_auto(self):
        self.assertEqual(parse_team_command("hire 3 new agents"), HireCommand(count=3))

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

    def test_parse_repo_root(self):
        self.assertEqual(parse_team_command("show repo root"), RepoRootCommand())
        self.assertEqual(
            parse_team_command("repo root /tmp/projects"), RepoRootCommand(Path("/tmp/projects"))
        )
        self.assertEqual(
            parse_team_command('repo root "/tmp/my projects"'),
            RepoRootCommand(Path("/tmp/my projects")),
        )


if __name__ == "__main__":
    unittest.main()
