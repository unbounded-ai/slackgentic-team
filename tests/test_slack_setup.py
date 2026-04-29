import unittest
from unittest.mock import patch

from agent_harness.slack.setup import (
    APP_TOKEN_RE,
    BOT_SCOPES,
    BOT_TOKEN_RE,
    CreatedSlackApp,
    SlackApiError,
    SlackSetupOptions,
    _configured_slash_command,
    _create_slack_app_with_retry,
    _install_claude_channel_if_available,
    build_socket_mode_manifest,
    extract_token,
    recursive_token_search,
    resolve_instance_slug,
    slash_command_for_instance,
)


class SlackSetupTests(unittest.TestCase):
    def test_manifest_matches_socket_mode_runtime_needs(self):
        manifest = build_socket_mode_manifest("riley")

        self.assertTrue(manifest["settings"]["socket_mode_enabled"])
        self.assertNotIn("redirect_urls", manifest["oauth_config"])
        self.assertIn("commands", manifest["oauth_config"]["scopes"]["bot"])
        self.assertIn("message.groups", manifest["settings"]["event_subscriptions"]["bot_events"])
        self.assertEqual(manifest["oauth_config"]["scopes"]["bot"], BOT_SCOPES)
        self.assertEqual(manifest["display_information"]["name"], "slackgentic-riley")
        commands = manifest["features"]["slash_commands"]
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["command"], "/slackgentic-riley")

    def test_manifest_has_no_http_request_urls(self):
        manifest = build_socket_mode_manifest()

        self.assertNotIn("request_url", manifest["settings"]["event_subscriptions"])
        self.assertNotIn("request_url", manifest["settings"]["interactivity"])
        for command in manifest["features"]["slash_commands"]:
            self.assertNotIn("url", command)

    def test_instance_slug_is_safe_for_slack_command_names(self):
        self.assertEqual(resolve_instance_slug("Riley Smith!"), "riley-smith")
        self.assertEqual(slash_command_for_instance("Riley Smith!"), "/slackgentic-riley-smith")

    def test_manifest_can_preserve_legacy_slash_command_name(self):
        manifest = build_socket_mode_manifest("local", slash_command_override="/slackgentic")

        self.assertEqual(manifest["features"]["slash_commands"][0]["command"], "/slackgentic")
        self.assertEqual(len(manifest["features"]["slash_commands"]), 1)

    def test_configured_slash_command_preserves_legacy_default(self):
        self.assertEqual(_configured_slash_command({}, "local"), "/slackgentic")
        self.assertEqual(
            _configured_slash_command({"SLACKGENTIC_INSTANCE": "riley"}, "riley"),
            "/slackgentic-riley",
        )

    def test_extract_token_matches_slack_token_prefixes(self):
        self.assertEqual(extract_token("token xoxb-123-abc", BOT_TOKEN_RE), "xoxb-123-abc")
        self.assertEqual(extract_token("token xapp-1-A-B", APP_TOKEN_RE), "xapp-1-A-B")

    def test_recursive_token_search_finds_nested_tokens(self):
        payload = {"credentials": {"nested": ["nope", {"token": "secret xapp-1-A-B"}]}}

        self.assertEqual(recursive_token_search(payload, APP_TOKEN_RE), "xapp-1-A-B")

    def test_create_app_retries_after_revoked_slack_cli_token(self):
        calls = []

        def fake_create(token, manifest):
            calls.append(token)
            if token == "old-token":
                raise SlackApiError("apps.manifest.create", "token_revoked")
            return CreatedSlackApp(app_id="A123", raw={})

        with (
            patch("agent_harness.slack.setup.create_slack_app", side_effect=fake_create),
            patch(
                "agent_harness.slack.setup._slack_cli_config_token",
                return_value="new-token",
            ) as refresh,
        ):
            created = _create_slack_app_with_retry(
                "old-token",
                build_socket_mode_manifest("riley"),
                SlackSetupOptions(),
            )

        self.assertEqual(created.app_id, "A123")
        self.assertEqual(calls, ["old-token", "new-token"])
        refresh.assert_called_once()
        self.assertTrue(refresh.call_args.kwargs["force_login"])

    def test_create_app_does_not_retry_unrelated_slack_api_errors(self):
        with (
            patch(
                "agent_harness.slack.setup.create_slack_app",
                side_effect=SlackApiError("apps.manifest.create", "invalid_auth"),
            ),
            patch("agent_harness.slack.setup._slack_cli_config_token") as refresh,
            self.assertRaises(SlackApiError),
        ):
            _create_slack_app_with_retry(
                "bad-token",
                build_socket_mode_manifest("riley"),
                SlackSetupOptions(),
            )

        refresh.assert_not_called()

    def test_initial_setup_installs_claude_channel_when_claude_is_available(self):
        with (
            patch("agent_harness.slack.setup.shutil.which", return_value="/usr/bin/claude"),
            patch("agent_harness.sessions.claude_channel.install_claude_mcp_server") as install,
        ):
            _install_claude_channel_if_available()

        install.assert_called_once_with()

    def test_initial_setup_skips_claude_channel_when_claude_is_missing(self):
        with (
            patch("agent_harness.slack.setup.shutil.which", return_value=None),
            patch("agent_harness.sessions.claude_channel.install_claude_mcp_server") as install,
        ):
            _install_claude_channel_if_available()

        install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
