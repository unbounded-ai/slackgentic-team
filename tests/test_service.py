import plistlib
import tempfile
import unittest
from pathlib import Path

from agent_harness.service import (
    ServiceSpec,
    render_launchd_plist,
    render_systemd_unit,
    service_environment_path,
)


class ServiceTests(unittest.TestCase):
    def test_render_launchd_plist_runs_slack_serve(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = ServiceSpec(
                name="slackgentic-team",
                executable=Path(tmp) / "slackgentic",
                args=["slack", "serve"],
                working_directory=Path(tmp),
                log_dir=Path(tmp) / "logs",
            )

            payload = plistlib.loads(render_launchd_plist(spec))

            self.assertTrue(payload["RunAtLoad"])
            self.assertTrue(payload["KeepAlive"])
            self.assertEqual(payload["ProgramArguments"][-2:], ["slack", "serve"])
            self.assertNotIn(".codex/tmp", payload["EnvironmentVariables"]["PATH"])

    def test_render_systemd_unit_runs_slack_serve(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = ServiceSpec(
                name="slackgentic-team",
                executable=Path(tmp) / "slackgentic",
                args=["slack", "serve"],
                working_directory=Path(tmp),
                log_dir=Path(tmp) / "logs",
            )

            unit = render_systemd_unit(spec)

            self.assertIn("ExecStart=", unit)
            self.assertIn("slack serve", unit)
            self.assertIn('Environment="PATH=', unit)
            self.assertIn("Restart=always", unit)

    def test_service_environment_path_filters_transient_entries(self):
        path = service_environment_path(
            "/tmp/bin:"
            "/Users/test/.codex/tmp/codex-abc:"
            "/opt/homebrew/bin:"
            "/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin"
        )

        self.assertIn("/opt/homebrew/bin", path)
        self.assertNotIn(".codex/tmp", path)
        self.assertNotIn("cryptexd", path)


if __name__ == "__main__":
    unittest.main()
