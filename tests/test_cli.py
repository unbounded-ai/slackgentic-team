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

    def test_service_install_installs_codex_app_server_and_daemon(self):
        output = io.StringIO()
        with (
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
        self.assertEqual(specs[1].args[:3], ["app-server", "--listen", "ws://127.0.0.1:47684"])
        self.assertIn("codex.plist", output.getvalue())
        self.assertIn("daemon.plist", output.getvalue())

    def test_service_install_can_skip_codex_app_server(self):
        output = io.StringIO()
        with (
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


if __name__ == "__main__":
    unittest.main()
