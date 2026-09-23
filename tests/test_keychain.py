import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_harness import keychain
from agent_harness.config import (
    load_config_from_env,
    load_stored_config,
    move_tokens_to_file,
    move_tokens_to_keychain,
    save_stored_config,
)
from agent_harness.keychain import KEYCHAIN_CONFIG_KEY, KeychainError
from agent_harness.slack.setup import _has_existing_credentials


class FakeSecurity:
    """Stands in for /usr/bin/security with an in-memory keychain."""

    def __init__(self):
        self.items: dict[tuple[str, str], str] = {}
        self.argv: list[list[str]] = []
        self.fail_writes = False

    def run(self, argv, input=None, **_):
        self.argv.append(list(argv))
        if argv[1] == "-i":
            if self.fail_writes:
                return subprocess.CompletedProcess(argv, 1, "", "write failed")
            words = input.split()
            service = words[words.index("-s") + 1]
            account = words[words.index("-a") + 1]
            self.items[(service, account)] = words[words.index("-w") + 1]
            return subprocess.CompletedProcess(argv, 0, "", "")
        service = argv[argv.index("-s") + 1]
        account = argv[argv.index("-a") + 1]
        if argv[1] == "find-generic-password":
            value = self.items.get((service, account))
            if value is None:
                return subprocess.CompletedProcess(argv, 44, "", "not found")
            return subprocess.CompletedProcess(argv, 0, value + "\n", "")
        if argv[1] == "delete-generic-password":
            found = self.items.pop((service, account), None) is not None
            return subprocess.CompletedProcess(argv, 0 if found else 44, "", "")
        raise AssertionError(argv)


class KeychainTokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_file = Path(self.tmp.name) / "config.json"
        self.config_file.write_text(
            json.dumps(
                {
                    "SLACK_APP_ID": "A123",
                    "SLACK_APP_TOKEN": "xapp-1-example",
                    "SLACK_BOT_TOKEN": "xoxb-example",
                    "SLACK_USER_TOKEN": "xoxp-example",
                }
            )
        )
        self.security = FakeSecurity()
        patches = [
            patch.object(keychain.subprocess, "run", self.security.run),
            patch.object(keychain, "keychain_available", lambda: True),
            patch.dict("os.environ", {}, clear=True),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_moving_tokens_to_keychain_removes_them_from_the_file(self):
        moved = move_tokens_to_keychain(self.config_file)

        self.assertEqual(moved, ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_USER_TOKEN"])
        stored = load_stored_config(self.config_file)
        self.assertEqual(stored, {"SLACK_APP_ID": "A123", KEYCHAIN_CONFIG_KEY: True})
        self.assertNotIn("xoxb-example", self.config_file.read_text())
        self.assertEqual(self.config_file.stat().st_mode & 0o777, 0o600)
        # Tokens go through stdin, never the process list.
        self.assertFalse(any("xoxb-example" in arg for argv in self.security.argv for arg in argv))

        config = load_config_from_env(self.config_file)
        self.assertEqual(config.slack.bot_token, "xoxb-example")
        self.assertEqual(config.slack.app_token, "xapp-1-example")
        self.assertEqual(config.slack.user_token, "xoxp-example")
        self.assertTrue(_has_existing_credentials(stored))

    def test_environment_tokens_still_win_over_the_keychain(self):
        move_tokens_to_keychain(self.config_file)
        with patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-from-env"}):
            config = load_config_from_env(self.config_file)
        self.assertEqual(config.slack.bot_token, "xoxb-from-env")

    def test_a_failed_write_leaves_the_file_untouched(self):
        before = self.config_file.read_text()
        self.security.fail_writes = True

        with self.assertRaises(KeychainError):
            move_tokens_to_keychain(self.config_file)

        self.assertEqual(self.config_file.read_text(), before)

    def test_moving_tokens_back_restores_the_file_and_clears_the_keychain(self):
        move_tokens_to_keychain(self.config_file)

        restored = move_tokens_to_file(self.config_file)

        self.assertEqual(set(restored), {"SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_USER_TOKEN"})
        stored = load_stored_config(self.config_file)
        self.assertEqual(stored["SLACK_BOT_TOKEN"], "xoxb-example")
        self.assertNotIn(KEYCHAIN_CONFIG_KEY, stored)
        self.assertEqual(self.security.items, {})

    def test_tokens_saved_later_follow_the_others_into_the_keychain(self):
        move_tokens_to_keychain(self.config_file)

        save_stored_config({"SLACK_BOT_TOKEN": "xoxb-rotated"}, self.config_file)

        self.assertNotIn("SLACK_BOT_TOKEN", load_stored_config(self.config_file))
        self.assertEqual(load_config_from_env(self.config_file).slack.bot_token, "xoxb-rotated")

    def test_values_that_are_not_slack_tokens_are_refused(self):
        with self.assertRaises(KeychainError):
            keychain.write_keychain_secret("SLACK_BOT_TOKEN", "xoxb bad -w x", self.config_file)

    def test_without_the_flag_the_keychain_is_never_consulted(self):
        load_config_from_env(self.config_file)
        self.assertEqual(self.security.argv, [])

    def _run_setup(self, **options):
        from agent_harness.slack import setup

        self.config_file.unlink()
        patches = [
            patch.object(setup, "_configuration_token", return_value="xoxe-config"),
            patch.object(
                setup,
                "_create_slack_app_with_retry",
                return_value=setup.CreatedSlackApp("A999", {}),
            ),
            patch.object(setup, "_bot_token", return_value="xoxb-new"),
            patch.object(setup, "_app_token", return_value="xapp-1-new"),
            patch.object(setup, "_slack_api_post", return_value={"team_id": "T1"}),
            patch.object(setup, "_verify_app_token"),
            patch.object(setup, "_install_claude_channel_if_available"),
            patch.object(setup, "_install_codex_mcp_if_available"),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return setup.run_interactive_setup(
            setup.SlackSetupOptions(
                config_file=self.config_file, open_browser=False, instance="example", **options
            )
        )

    def test_setup_keeps_new_tokens_in_the_keychain_by_default_on_macos(self):
        self.assertEqual(self._run_setup(), 0)

        stored = load_stored_config(self.config_file)
        self.assertTrue(stored[KEYCHAIN_CONFIG_KEY])
        self.assertNotIn("SLACK_BOT_TOKEN", stored)
        self.assertNotIn("xoxb-new", self.config_file.read_text())
        config = load_config_from_env(self.config_file)
        self.assertEqual(config.slack.bot_token, "xoxb-new")
        self.assertEqual(config.slack.app_token, "xapp-1-new")

    def test_setup_can_keep_tokens_in_the_file(self):
        self.assertEqual(self._run_setup(tokens_in_keychain=False), 0)

        stored = load_stored_config(self.config_file)
        self.assertEqual(stored["SLACK_BOT_TOKEN"], "xoxb-new")
        self.assertNotIn(KEYCHAIN_CONFIG_KEY, stored)
        self.assertEqual(self.security.items, {})

    def test_setup_falls_back_to_the_file_when_the_keychain_write_fails(self):
        self.security.fail_writes = True

        self.assertEqual(self._run_setup(), 0)

        stored = load_stored_config(self.config_file)
        self.assertEqual(stored["SLACK_BOT_TOKEN"], "xoxb-new")
        self.assertNotIn(KEYCHAIN_CONFIG_KEY, stored)


if __name__ == "__main__":
    unittest.main()
