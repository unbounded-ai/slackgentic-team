import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_harness.cli import main
from agent_harness.service import UnsafeServiceRestartError
from agent_harness.slack.app import SETTING_CHANNEL_ID, SETTING_ROSTER_TS
from agent_harness.storage.store import Store


class CliTests(unittest.TestCase):
    def test_reset_state_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            config = Path(tmp) / "config.json"
            db.write_text("state")

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "slack",
                        "reset-state",
                        "--config-file",
                        str(config),
                        "--db",
                        str(db),
                    ]
                )

            self.assertEqual(code, 2)
            self.assertTrue(db.exists())
            self.assertIn("--yes", output.getvalue())

    def test_reset_state_removes_sqlite_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            config = Path(tmp) / "config.json"
            paths = [db, Path(f"{db}-wal"), Path(f"{db}-shm"), Path(f"{db}-journal")]
            for path in paths:
                path.write_text("state")

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "slack",
                        "reset-state",
                        "--config-file",
                        str(config),
                        "--db",
                        str(db),
                        "--yes",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue(all(not path.exists() for path in paths))
            self.assertIn("setup", output.getvalue())

    def test_close_channel_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            config = Path(tmp) / "config.json"
            config.write_text('{"SLACK_BOT_TOKEN": "xoxb-test"}')
            store = Store(db)
            try:
                store.init_schema()
                store.set_setting(SETTING_CHANNEL_ID, "C1")
            finally:
                store.close()

            output = io.StringIO()
            with (
                patch("agent_harness.slack.client.SlackGateway") as gateway,
                redirect_stdout(output),
            ):
                code = main(
                    [
                        "slack",
                        "close-channel",
                        "--config-file",
                        str(config),
                        "--db",
                        str(db),
                    ]
                )

            self.assertEqual(code, 2)
            gateway.assert_not_called()
            self.assertIn("--yes", output.getvalue())

    def test_close_channel_archives_configured_channel_and_clears_state(self):
        archived = []

        class FakeSlackGateway:
            def __init__(self, bot_token):
                self.bot_token = bot_token

            def archive_channel(self, channel_id):
                archived.append((self.bot_token, channel_id))
                return True

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            config = Path(tmp) / "config.json"
            config.write_text('{"SLACK_BOT_TOKEN": "xoxb-test"}')
            store = Store(db)
            try:
                store.init_schema()
                store.set_setting(SETTING_CHANNEL_ID, "C1")
                store.set_setting(SETTING_ROSTER_TS, "171.000001")
            finally:
                store.close()

            output = io.StringIO()
            with (
                patch("agent_harness.slack.client.SlackGateway", FakeSlackGateway),
                redirect_stdout(output),
            ):
                code = main(
                    [
                        "slack",
                        "close-channel",
                        "--config-file",
                        str(config),
                        "--db",
                        str(db),
                        "--yes",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(archived, [("xoxb-test", "C1")])
            store = Store(db)
            try:
                store.init_schema()
                self.assertIsNone(store.get_setting(SETTING_CHANNEL_ID))
                self.assertIsNone(store.get_setting(SETTING_ROSTER_TS))
            finally:
                store.close()
            self.assertIn("archived Slack channel C1", output.getvalue())

    def test_service_restart_reports_unsafe_restart(self):
        output = io.StringIO()
        with (
            patch(
                "agent_harness.service.restart_service",
                side_effect=UnsafeServiceRestartError("daemon owns app-server"),
            ),
            redirect_stdout(output),
        ):
            code = main(["service", "restart"])

        self.assertEqual(code, 2)
        self.assertIn("daemon owns app-server", output.getvalue())

    def test_service_restart_passes_force_flag(self):
        output = io.StringIO()
        with (
            patch("agent_harness.service.restart_service", return_value=0) as restart,
            redirect_stdout(output),
        ):
            code = main(["service", "restart", "--force"])

        self.assertEqual(code, 0)
        restart.assert_called_once_with("slackgentic-team", force=True)

    def test_service_start_starts_managed_services(self):
        output = io.StringIO()
        with (
            patch("agent_harness.service.start_services", return_value=[0, 0]) as start,
            redirect_stdout(output),
        ):
            code = main(["service", "start"])

        self.assertEqual(code, 0)
        start.assert_called_once_with("slackgentic-team", include_codex_app_server=True)
        self.assertIn("started services", output.getvalue())

    def test_service_start_can_skip_codex_app_server(self):
        output = io.StringIO()
        with (
            patch("agent_harness.service.start_services", return_value=[0]) as start,
            redirect_stdout(output),
        ):
            code = main(["service", "start", "--no-codex-app-server"])

        self.assertEqual(code, 0)
        start.assert_called_once_with("slackgentic-team", include_codex_app_server=False)

    def test_slack_serve_refuses_python_314_runtime(self):
        output = io.StringIO()
        with (
            patch("sys.version_info", (3, 14, 0)),
            patch("agent_harness.cli.platform.system", return_value="Darwin"),
            redirect_stdout(output),
        ):
            code = main(["slack", "serve"])

        self.assertEqual(code, 2)
        self.assertIn("Python 3.14+ on macOS", output.getvalue())

    def test_slack_serve_passes_ignored_external_session_cwd_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config.json"
            config_file.write_text("{}")

            with (
                patch("sys.version_info", (3, 13, 0)),
                patch("agent_harness.slack.app.run_slack_app", return_value=0) as run_app,
            ):
                code = main(
                    [
                        "slack",
                        "serve",
                        "--config-file",
                        str(config_file),
                        "--ignore-external-session-cwd",
                        "example-project/.local",
                    ]
                )

            self.assertEqual(code, 0)
            config = run_app.call_args.args[0]
            self.assertEqual(
                config.sessions.ignored_external_session_cwds,
                ("example-project/.local",),
            )

    def test_slack_serve_passes_allowed_external_session_cwd_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config.json"
            config_file.write_text("{}")

            with (
                patch("sys.version_info", (3, 13, 0)),
                patch("agent_harness.slack.app.run_slack_app", return_value=0) as run_app,
            ):
                code = main(
                    [
                        "slack",
                        "serve",
                        "--config-file",
                        str(config_file),
                        "--allow-external-session-cwd-prefix",
                        "/workspace/repos",
                    ]
                )

            self.assertEqual(code, 0)
            config = run_app.call_args.args[0]
            self.assertEqual(
                config.sessions.allowed_external_session_cwd_prefixes,
                ("/workspace/repos",),
            )

    def test_claude_channel_refuses_python_314_runtime(self):
        output = io.StringIO()
        with (
            patch("sys.version_info", (3, 14, 0)),
            patch("agent_harness.cli.platform.system", return_value="Darwin"),
            redirect_stdout(output),
        ):
            code = main(["claude-channel", "--print-mcp-config"])

        self.assertEqual(code, 2)
        self.assertIn("TCC privacy prompts", output.getvalue())

    def test_codex_mcp_install_registers_server(self):
        output = io.StringIO()
        with (
            patch("agent_harness.sessions.claude_channel.install_codex_mcp_server") as install,
            redirect_stdout(output),
        ):
            code = main(["codex-mcp", "--install"])

        self.assertEqual(code, 0)
        install.assert_called_once_with()
        self.assertIn("registered Codex MCP server", output.getvalue())

    def test_service_install_refuses_python_314_runtime(self):
        output = io.StringIO()
        with (
            patch("sys.version_info", (3, 14, 0)),
            patch("agent_harness.cli.platform.system", return_value="Darwin"),
            patch("agent_harness.service.install_services") as install,
            redirect_stdout(output),
        ):
            code = main(["service", "install"])

        self.assertEqual(code, 2)
        install.assert_not_called()
        self.assertIn("reinstall the service", output.getvalue())

    def test_service_install_installs_codex_app_server_and_daemon(self):
        output = io.StringIO()
        with (
            patch("sys.version_info", (3, 13, 0)),
            patch("agent_harness.cli.platform.system", return_value="Darwin"),
            patch(
                "agent_harness.service._current_slackgentic_executable",
                return_value=Path("/tmp/slackgentic"),
            ),
            patch(
                "agent_harness.service.install_services",
                return_value=[Path("/tmp/codex.plist"), Path("/tmp/daemon.plist")],
            ) as install,
            redirect_stdout(output),
        ):
            code = main(["service", "install", "--codex-binary", "/tmp/codex"])

        self.assertEqual(code, 0)
        specs = install.call_args.args[0]
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].args, ["slack", "serve"])
        self.assertEqual(
            specs[1].args,
            [
                "codex-app-server",
                "--listen",
                "ws://127.0.0.1:47684",
                "--codex-binary",
                "/tmp/codex",
            ],
        )
        self.assertEqual(specs[1].executable, Path("/tmp/slackgentic"))
        self.assertIn("codex.plist", output.getvalue())
        self.assertIn("daemon.plist", output.getvalue())

    def test_service_install_can_skip_codex_app_server(self):
        output = io.StringIO()
        with (
            patch("sys.version_info", (3, 13, 0)),
            patch("agent_harness.cli.platform.system", return_value="Darwin"),
            patch(
                "agent_harness.service._current_slackgentic_executable",
                return_value=Path("/tmp/slackgentic"),
            ),
            patch(
                "agent_harness.service.install_services",
                return_value=[Path("/tmp/daemon.plist")],
            ) as install,
            redirect_stdout(output),
        ):
            code = main(["service", "install", "--no-codex-app-server"])

        self.assertEqual(code, 0)
        specs = install.call_args.args[0]
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].args, ["slack", "serve"])

    def test_internal_codex_app_server_command_runs_supervisor(self):
        with patch(
            "agent_harness.runtime.codex_app_server.run_codex_app_server_supervisor",
            return_value=0,
        ) as run:
            code = main(
                [
                    "codex-app-server",
                    "--listen",
                    "ws://127.0.0.1:9999",
                    "--codex-binary",
                    "/opt/tools/codex",
                    "--version-check-interval",
                    "3",
                ]
            )

        self.assertEqual(code, 0)
        run.assert_called_once_with(
            "/opt/tools/codex",
            "ws://127.0.0.1:9999",
            check_interval_seconds=3.0,
        )

    def test_service_install_can_set_ignored_external_session_cwd_argument(self):
        output = io.StringIO()
        with (
            patch("sys.version_info", (3, 13, 0)),
            patch("agent_harness.cli.platform.system", return_value="Darwin"),
            patch(
                "agent_harness.service._current_slackgentic_executable",
                return_value=Path("/tmp/slackgentic"),
            ),
            patch(
                "agent_harness.service.install_services",
                return_value=[Path("/tmp/daemon.plist")],
            ) as install,
            redirect_stdout(output),
        ):
            code = main(
                [
                    "service",
                    "install",
                    "--no-codex-app-server",
                    "--ignore-external-session-cwd",
                    "example-project/.local",
                ]
            )

        self.assertEqual(code, 0)
        specs = install.call_args.args[0]
        self.assertEqual(
            specs[0].args,
            [
                "slack",
                "serve",
                "--ignore-external-session-cwd",
                "example-project/.local",
            ],
        )

    def test_service_install_can_set_allowed_external_session_cwd_prefix(self):
        output = io.StringIO()
        with (
            patch("sys.version_info", (3, 13, 0)),
            patch("agent_harness.cli.platform.system", return_value="Darwin"),
            patch(
                "agent_harness.service._current_slackgentic_executable",
                return_value=Path("/tmp/slackgentic"),
            ),
            patch(
                "agent_harness.service.install_services",
                return_value=[Path("/tmp/daemon.plist")],
            ) as install,
            redirect_stdout(output),
        ):
            code = main(
                [
                    "service",
                    "install",
                    "--no-codex-app-server",
                    "--allow-external-session-cwd-prefix",
                    "/workspace/repos",
                ]
            )

        self.assertEqual(code, 0)
        specs = install.call_args.args[0]
        self.assertEqual(
            specs[0].args,
            [
                "slack",
                "serve",
                "--allow-external-session-cwd-prefix",
                "/workspace/repos",
            ],
        )


if __name__ == "__main__":
    unittest.main()
