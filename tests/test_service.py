import plistlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from agent_harness.service import (
    MACOS_CODEX_APP_SERVER_LABEL,
    ServiceSpec,
    UnsafeServiceRestartError,
    build_codex_app_server_service_spec,
    install_services,
    render_launchd_plist,
    render_systemd_unit,
    restart_service,
    service_environment_path,
    start_services,
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
            self.assertEqual(
                payload["EnvironmentVariables"]["SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART"],
                "false",
            )

    def test_render_launchd_plist_runs_codex_app_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = build_codex_app_server_service_spec(
                executable=Path(tmp) / "codex",
                working_directory=Path(tmp),
                url="ws://127.0.0.1:9999",
            )

            payload = plistlib.loads(render_launchd_plist(spec))

            self.assertEqual(payload["Label"], MACOS_CODEX_APP_SERVER_LABEL)
            self.assertEqual(
                payload["ProgramArguments"][-3:],
                ["app-server", "--listen", "ws://127.0.0.1:9999"],
            )
            self.assertNotIn(
                "SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART",
                payload["EnvironmentVariables"],
            )

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
            self.assertIn('Environment="SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART=false"', unit)
            self.assertIn("Restart=always", unit)

    def test_render_systemd_unit_runs_codex_app_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = build_codex_app_server_service_spec(
                executable=Path(tmp) / "codex",
                working_directory=Path(tmp),
                url="ws://127.0.0.1:9999",
            )

            unit = render_systemd_unit(spec)

            self.assertIn("Description=Slackgentic Team Codex app-server", unit)
            self.assertIn("app-server --listen ws://127.0.0.1:9999", unit)
            self.assertNotIn("SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART", unit)

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

    def test_restart_service_on_macos_uses_launchd_label_target(self):
        with (
            patch("agent_harness.service.platform.system", return_value="Darwin"),
            patch("agent_harness.service.os.getuid", return_value=501),
            patch("agent_harness.service._launchd_restart_safety_issue", return_value=None),
            patch("agent_harness.service.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess([], 0)

            result = restart_service()

        self.assertEqual(result, 0)
        run.assert_called_once_with(
            ["launchctl", "kickstart", "-k", "gui/501/com.slackgentic-team.daemon"],
            check=False,
        )

    def test_start_services_on_macos_starts_codex_then_daemon(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("agent_harness.service.platform.system", return_value="Darwin"),
            patch("agent_harness.service.os.getuid", return_value=501),
            patch(
                "agent_harness.service._launchd_path",
                side_effect=lambda label: Path(tmp) / f"{label}.plist",
            ),
            patch("agent_harness.service.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess([], 0)

            result = start_services()

        self.assertEqual(result, [0, 0])
        run.assert_has_calls(
            [
                call(
                    [
                        "launchctl",
                        "kickstart",
                        "gui/501/com.slackgentic-team.codex-app-server",
                    ],
                    check=False,
                ),
                call(
                    ["launchctl", "kickstart", "gui/501/com.slackgentic-team.daemon"],
                    check=False,
                ),
            ]
        )

    def test_install_services_on_macos_replaces_then_loads_codex_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            daemon = ServiceSpec(
                name="slackgentic-team",
                executable=Path(tmp) / "slackgentic",
                args=["slack", "serve"],
                working_directory=Path(tmp),
                log_dir=Path(tmp) / "logs",
            )
            codex = build_codex_app_server_service_spec(
                executable=Path(tmp) / "codex",
                working_directory=Path(tmp),
            )
            with (
                patch("agent_harness.service.platform.system", return_value="Darwin"),
                patch("agent_harness.service.os.getuid", return_value=501),
                patch(
                    "agent_harness.service._launchd_path",
                    side_effect=lambda label: Path(tmp) / f"{label}.plist",
                ),
                patch("agent_harness.service.subprocess.run") as run,
            ):
                run.return_value = subprocess.CompletedProcess([], 0)

                paths = install_services([daemon, codex])

        self.assertEqual(len(paths), 2)
        run.assert_has_calls(
            [
                call(
                    ["launchctl", "bootout", "gui/501/com.slackgentic-team.daemon"],
                    check=False,
                    capture_output=True,
                    text=True,
                ),
                call(
                    [
                        "launchctl",
                        "bootout",
                        "gui/501/com.slackgentic-team.codex-app-server",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                ),
                call(
                    ["launchctl", "bootstrap", "gui/501", str(paths[1])],
                    check=False,
                    capture_output=True,
                    text=True,
                ),
                call(
                    ["launchctl", "bootstrap", "gui/501", str(paths[0])],
                    check=False,
                    capture_output=True,
                    text=True,
                ),
            ]
        )

    def test_install_services_on_macos_retries_transient_bootstrap_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            daemon = ServiceSpec(
                name="slackgentic-team",
                executable=Path(tmp) / "slackgentic",
                args=["slack", "serve"],
                working_directory=Path(tmp),
                log_dir=Path(tmp) / "logs",
            )
            codex = build_codex_app_server_service_spec(
                executable=Path(tmp) / "codex",
                working_directory=Path(tmp),
            )

            def fake_run(command, **kwargs):
                if command[:2] == ["launchctl", "bootstrap"]:
                    bootstrap_calls.append(command)
                    if len(bootstrap_calls) == 1:
                        return subprocess.CompletedProcess(
                            command,
                            5,
                            "",
                            "Bootstrap failed: 5: Input/output error",
                        )
                if command[:2] == ["launchctl", "print"]:
                    if len(bootstrap_calls) >= 2:
                        return subprocess.CompletedProcess(command, 0, "", "")
                    return subprocess.CompletedProcess(command, 3, "", "not loaded")
                return subprocess.CompletedProcess(command, 0, "", "")

            bootstrap_calls = []
            with (
                patch("agent_harness.service.platform.system", return_value="Darwin"),
                patch("agent_harness.service.os.getuid", return_value=501),
                patch("agent_harness.service.time.sleep"),
                patch(
                    "agent_harness.service._launchd_path",
                    side_effect=lambda label: Path(tmp) / f"{label}.plist",
                ),
                patch("agent_harness.service.subprocess.run", side_effect=fake_run),
            ):
                paths = install_services([daemon, codex])

        self.assertEqual(len(paths), 2)
        self.assertEqual(len(bootstrap_calls), 5)

    def test_install_services_on_macos_reconciles_missing_label_after_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            daemon = ServiceSpec(
                name="slackgentic-team",
                executable=Path(tmp) / "slackgentic",
                args=["slack", "serve"],
                working_directory=Path(tmp),
                log_dir=Path(tmp) / "logs",
            )
            codex = build_codex_app_server_service_spec(
                executable=Path(tmp) / "codex",
                working_directory=Path(tmp),
            )

            kickstart_calls = []

            def fake_run(command, **kwargs):
                if command[:2] == ["launchctl", "kickstart"]:
                    kickstart_calls.append(command)
                if command[:2] == ["launchctl", "print"]:
                    if command[-1].endswith("codex-app-server") and not kickstart_calls:
                        return subprocess.CompletedProcess(command, 3, "", "not loaded")
                    return subprocess.CompletedProcess(command, 0, "", "")
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                patch("agent_harness.service.platform.system", return_value="Darwin"),
                patch("agent_harness.service.os.getuid", return_value=501),
                patch("agent_harness.service.time.sleep"),
                patch(
                    "agent_harness.service._launchd_path",
                    side_effect=lambda label: Path(tmp) / f"{label}.plist",
                ),
                patch("agent_harness.service.subprocess.run", side_effect=fake_run),
            ):
                paths = install_services([daemon, codex])

        self.assertEqual(len(paths), 2)
        self.assertIn(
            [
                "launchctl",
                "kickstart",
                "gui/501/com.slackgentic-team.codex-app-server",
            ],
            kickstart_calls,
        )

    def test_restart_service_on_macos_refuses_unsafe_launchd_unit(self):
        with (
            patch("agent_harness.service.platform.system", return_value="Darwin"),
            patch(
                "agent_harness.service._launchd_restart_safety_issue",
                return_value="unsafe restart",
            ),
            patch("agent_harness.service.subprocess.run") as run,
            self.assertRaisesRegex(UnsafeServiceRestartError, "unsafe restart"),
        ):
            restart_service()

        run.assert_not_called()

    def test_restart_service_on_macos_force_bypasses_safety_check(self):
        with (
            patch("agent_harness.service.platform.system", return_value="Darwin"),
            patch("agent_harness.service.os.getuid", return_value=501),
            patch("agent_harness.service._launchd_restart_safety_issue") as safety,
            patch("agent_harness.service.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess([], 0)

            result = restart_service(force=True)

        self.assertEqual(result, 0)
        safety.assert_not_called()
        run.assert_called_once_with(
            ["launchctl", "kickstart", "-k", "gui/501/com.slackgentic-team.daemon"],
            check=False,
        )

    def test_restart_service_on_linux_uses_systemd_unit(self):
        with (
            patch("agent_harness.service.platform.system", return_value="Linux"),
            patch("agent_harness.service._systemd_restart_safety_issue", return_value=None),
            patch("agent_harness.service.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess([], 0)

            result = restart_service("slackgentic-custom")

        self.assertEqual(result, 0)
        run.assert_called_once_with(
            ["systemctl", "--user", "restart", "slackgentic-custom.service"],
            check=False,
        )

    def test_start_services_on_linux_starts_codex_then_daemon(self):
        with (
            patch("agent_harness.service.platform.system", return_value="Linux"),
            patch("agent_harness.service.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess([], 0)

            result = start_services("slackgentic-custom")

        self.assertEqual(result, [0, 0])
        run.assert_has_calls(
            [
                call(
                    [
                        "systemctl",
                        "--user",
                        "start",
                        "slackgentic-custom-codex-app-server.service",
                    ],
                    check=False,
                ),
                call(
                    ["systemctl", "--user", "start", "slackgentic-custom.service"],
                    check=False,
                ),
            ]
        )

    def test_install_services_on_linux_replaces_then_starts_codex_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            daemon = ServiceSpec(
                name="slackgentic-team",
                executable=Path(tmp) / "slackgentic",
                args=["slack", "serve"],
                working_directory=Path(tmp),
                log_dir=Path(tmp) / "logs",
            )
            codex = build_codex_app_server_service_spec(
                executable=Path(tmp) / "codex",
                working_directory=Path(tmp),
            )
            with (
                patch("agent_harness.service.platform.system", return_value="Linux"),
                patch(
                    "agent_harness.service._systemd_path",
                    side_effect=lambda name: Path(tmp) / f"{name}.service",
                ),
                patch("agent_harness.service.subprocess.run") as run,
            ):
                run.return_value = subprocess.CompletedProcess([], 0)

                install_services([daemon, codex])

        run.assert_has_calls(
            [
                call(["systemctl", "--user", "daemon-reload"], check=True),
                call(["systemctl", "--user", "stop", "slackgentic-team.service"], check=False),
                call(
                    [
                        "systemctl",
                        "--user",
                        "stop",
                        "slackgentic-team-codex-app-server.service",
                    ],
                    check=False,
                ),
                call(
                    [
                        "systemctl",
                        "--user",
                        "enable",
                        "--now",
                        "slackgentic-team-codex-app-server.service",
                    ],
                    check=True,
                ),
                call(
                    ["systemctl", "--user", "enable", "--now", "slackgentic-team.service"],
                    check=True,
                ),
            ]
        )


if __name__ == "__main__":
    unittest.main()
