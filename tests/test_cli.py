import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_harness.cli import main
from agent_harness.service import UnsafeServiceRestartError


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
