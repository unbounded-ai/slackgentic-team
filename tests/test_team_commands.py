import unittest

from agent_harness.models import Provider
from agent_harness.team_commands import FireCommand, HireCommand, RosterCommand, parse_team_command


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

    def test_parse_roster(self):
        self.assertEqual(parse_team_command("show roster"), RosterCommand())


if __name__ == "__main__":
    unittest.main()
