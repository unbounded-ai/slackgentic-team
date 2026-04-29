import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_harness.config import load_config_from_env, load_stored_config, save_stored_config


class ConfigTests(unittest.TestCase):
    def test_load_config_merges_file_and_env_with_env_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config.json"
            save_stored_config(
                {
                    "SLACK_BOT_TOKEN": "xoxb-file",
                    "SLACK_APP_TOKEN": "xapp-file",
                    "SLACKGENTIC_INSTANCE": "riley",
                    "SLACKGENTIC_SLASH_COMMAND": "/slackgentic-riley",
                    "SLACKGENTIC_CODEX_AGENTS": "2",
                    "SLACKGENTIC_STATE_DB": str(Path(tmp) / "file.sqlite"),
                },
                config_file,
            )

            with patch.dict(
                os.environ,
                {
                    "SLACK_BOT_TOKEN": "xoxb-env",
                    "SLACKGENTIC_CONFIG_FILE": str(config_file),
                },
                clear=True,
            ):
                config = load_config_from_env()

            self.assertEqual(config.slack.bot_token, "xoxb-env")
            self.assertEqual(config.slack.app_token, "xapp-file")
            self.assertTrue(config.slack.socket_mode_ready)
            self.assertEqual(config.slack.instance_slug, "riley")
            self.assertEqual(config.slack.slash_command, "/slackgentic-riley")
            self.assertEqual(config.team.default_codex_agents, 2)
            self.assertEqual(config.state_db, Path(tmp) / "file.sqlite")
            self.assertEqual(config.config_file, config_file)

    def test_default_team_config_is_one_agent_per_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config.json"
            save_stored_config({}, config_file)

            with patch.dict(
                os.environ,
                {"SLACKGENTIC_CONFIG_FILE": str(config_file)},
                clear=True,
            ):
                config = load_config_from_env()

            self.assertEqual(config.team.default_codex_agents, 1)
            self.assertEqual(config.team.default_claude_agents, 1)

    def test_save_stored_config_preserves_existing_values_and_uses_private_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config.json"

            save_stored_config({"SLACK_BOT_TOKEN": "xoxb-old"}, config_file)
            save_stored_config({"SLACK_APP_TOKEN": "xapp-new"}, config_file)

            self.assertEqual(
                load_stored_config(config_file),
                {"SLACK_BOT_TOKEN": "xoxb-old", "SLACK_APP_TOKEN": "xapp-new"},
            )
            self.assertEqual(stat.S_IMODE(config_file.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
