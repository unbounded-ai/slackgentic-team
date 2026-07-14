import json
import os
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
    _stable_python_executable,
    build_codex_app_server_service_spec,
    ensure_service_python_shims,
    install_services,
    installed_services_match,
    render_launchd_plist,
    render_systemd_unit,
    restart_service,
    service_environment_path,
    start_services,
    start_update_helper,
)
from agent_harness.storage.store import Store
from agent_harness.updates import SETTING_UPDATE_RESTART_HELPER


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
            self.assertEqual(payload["ExitTimeOut"], 30)
            self.assertEqual(payload["ProgramArguments"][-2:], ["slack", "serve"])
            self.assertNotIn(".codex/tmp", payload["EnvironmentVariables"]["PATH"])
            self.assertEqual(
                payload["EnvironmentVariables"]["PYTHONPATH"],
                str(Path(tmp) / "src"),
            )
            self.assertEqual(
                payload["EnvironmentVariables"]["SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART"],
                "false",
            )

    def test_render_launchd_plist_runs_codex_app_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = build_codex_app_server_service_spec(
                executable=Path(tmp) / "codex",
                supervisor_executable=Path(tmp) / "slackgentic",
                working_directory=Path(tmp),
                url="ws://127.0.0.1:9999",
            )

            payload = plistlib.loads(render_launchd_plist(spec))

            self.assertEqual(payload["Label"], MACOS_CODEX_APP_SERVER_LABEL)
            self.assertEqual(
                payload["ProgramArguments"],
                [
                    str(Path(tmp) / "slackgentic"),
                    "codex-app-server",
                    "--listen",
                    "ws://127.0.0.1:9999",
                    "--codex-binary",
                    str(Path(tmp) / "codex"),
                ],
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
            self.assertIn(f'Environment="PYTHONPATH={Path(tmp) / "src"}"', unit)
            self.assertIn('Environment="SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART=false"', unit)
            self.assertIn("Restart=always", unit)
            self.assertIn("TimeoutStopSec=30", unit)

    def test_render_systemd_unit_runs_codex_app_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = build_codex_app_server_service_spec(
                executable=Path(tmp) / "codex",
                supervisor_executable=Path(tmp) / "slackgentic",
                working_directory=Path(tmp),
                url="ws://127.0.0.1:9999",
            )

            unit = render_systemd_unit(spec)

            self.assertIn("Description=Slackgentic Team Codex app-server", unit)
            self.assertIn("codex-app-server --listen ws://127.0.0.1:9999", unit)
            self.assertIn(f"--codex-binary {Path(tmp) / 'codex'}", unit)
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

    def test_service_environment_path_prefers_python_shim_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            shim_dir = Path(tmp) / "bin"
            with patch(
                "agent_harness.service._stable_python_executable",
                return_value=Path("/opt/homebrew/bin/python3.13"),
            ):
                path = service_environment_path("/opt/homebrew/bin", service_bin_dir=shim_dir)

        self.assertEqual(path.split(os.pathsep)[0], str(shim_dir))

    def test_ensure_service_python_shims_points_python_to_stable_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            stable = Path(tmp) / "python3.13"
            stable.write_text("#!/bin/sh\n")
            spec = ServiceSpec(
                name="slackgentic-team",
                executable=Path(tmp) / "slackgentic",
                args=["slack", "serve"],
                working_directory=Path(tmp),
                log_dir=Path(tmp) / "state" / "logs",
            )

            with patch("agent_harness.service._stable_python_executable", return_value=stable):
                bin_dir = ensure_service_python_shims(spec)

            assert bin_dir is not None
            self.assertEqual((bin_dir / "python").resolve(), stable.resolve())
            self.assertEqual((bin_dir / "python3").resolve(), stable.resolve())

    def test_stable_python_falls_back_to_current_stable_interpreter(self):
        with (
            patch("agent_harness.service.shutil.which", return_value=None),
            patch("agent_harness.service.sys.version_info", (3, 13, 0)),
            patch("agent_harness.service.sys.executable", "/tmp/venv/bin/python"),
        ):
            self.assertEqual(
                _stable_python_executable(),
                Path("/tmp/venv/bin/python").resolve(),
            )

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
                    ["launchctl", "bootout", "gui/501/com.slackgentic-team.daemon"],
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

    def test_installed_services_match_compares_rendered_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = ServiceSpec(
                name="slackgentic-team",
                executable=Path(tmp) / "slackgentic",
                args=["slack", "serve"],
                working_directory=Path(tmp),
                log_dir=Path(tmp) / "logs",
            )
            path = Path(tmp) / "daemon.plist"
            with (
                patch("agent_harness.service.platform.system", return_value="Darwin"),
                patch("agent_harness.service._launchd_path", return_value=path),
            ):
                path.write_bytes(render_launchd_plist(spec))
                self.assertTrue(installed_services_match([spec]))
                path.write_text("different")
                self.assertFalse(installed_services_match([spec]))

    def test_install_services_on_macos_does_not_bootout_unchanged_services(self):
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

            def launchd_path(label):
                return Path(tmp) / f"{label}.plist"

            with (
                patch("agent_harness.service.platform.system", return_value="Darwin"),
                patch("agent_harness.service.os.getuid", return_value=501),
                patch("agent_harness.service._launchd_path", side_effect=launchd_path),
            ):
                launchd_path(daemon.label).write_bytes(render_launchd_plist(daemon))
                launchd_path(codex.label).write_bytes(render_launchd_plist(codex))
                with patch("agent_harness.service.subprocess.run") as run:
                    run.return_value = subprocess.CompletedProcess([], 0)
                    install_services([daemon, codex])

            bootouts = [
                call_args.args[0]
                for call_args in run.call_args_list
                if call_args.args[0][:2] == ["launchctl", "bootout"]
            ]
        self.assertEqual(bootouts, [])

    def test_install_services_on_macos_rolls_back_plist_when_bootstrap_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            daemon = ServiceSpec(
                name="slackgentic-team",
                executable=Path(tmp) / "slackgentic",
                args=["slack", "serve"],
                working_directory=Path(tmp),
                log_dir=Path(tmp) / "logs",
            )
            old_payload = {
                "Label": daemon.label,
                "ProgramArguments": ["/old/slackgentic", "slack", "serve"],
            }
            daemon_path = Path(tmp) / f"{daemon.label}.plist"
            daemon_path.write_bytes(plistlib.dumps(old_payload))

            def fake_run(command, **kwargs):
                if command[:2] == ["launchctl", "bootstrap"] and command[-1] == str(daemon_path):
                    return subprocess.CompletedProcess(
                        command,
                        5,
                        "",
                        "Bootstrap failed: 5: Input/output error",
                    )
                if command[:2] == ["launchctl", "print"]:
                    return subprocess.CompletedProcess(command, 3, "", "not loaded")
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                patch("agent_harness.service.platform.system", return_value="Darwin"),
                patch("agent_harness.service.os.getuid", return_value=501),
                patch("agent_harness.service.time.sleep"),
                patch(
                    "agent_harness.service._launchd_path",
                    side_effect=lambda label: daemon_path,
                ),
                patch("agent_harness.service.subprocess.run", side_effect=fake_run),
                self.assertRaises(RuntimeError),
            ):
                install_services([daemon])

            self.assertEqual(plistlib.loads(daemon_path.read_bytes()), old_payload)

    def test_start_update_helper_on_macos_uses_one_shot_launch_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper_plist = Path(tmp) / "helper.plist"
            state_db = Path(tmp) / "state.sqlite"
            with (
                patch("agent_harness.service.platform.system", return_value="Darwin"),
                patch("agent_harness.service.os.getuid", return_value=501),
                patch("agent_harness.service._launchd_path", return_value=helper_plist),
                patch("agent_harness.service.subprocess.run") as run,
            ):
                run.return_value = subprocess.CompletedProcess([], 0)
                log_file = start_update_helper(
                    executable=Path("/opt/example/bin/slackgentic"),
                    state_db=state_db,
                    version="0.2.0",
                    command=["slackgentic", "service", "install"],
                    log_dir=Path(tmp) / "logs",
                    working_directory=Path(tmp),
                )

            payload = plistlib.loads(helper_plist.read_bytes())
            store = Store(state_db)
            try:
                helper_state = json.loads(store.get_setting(SETTING_UPDATE_RESTART_HELPER))
            finally:
                store.close()

        self.assertEqual(payload["Label"], "com.slackgentic-team.update-helper")
        self.assertFalse(payload["KeepAlive"])
        self.assertIn("update-helper", payload["ProgramArguments"])
        self.assertIn("slackgentic", payload["ProgramArguments"])
        self.assertEqual(log_file.parent.name, "logs")
        self.assertEqual(helper_state["phase"], "scheduled")
        self.assertEqual(helper_state["version"], "0.2.0")
        self.assertEqual(helper_state["command"], ["slackgentic", "service", "install"])

    def test_install_services_on_macos_bootstraps_codex_before_daemon_bootout(self):
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
            commands = []

            def fake_run(command, **kwargs):
                commands.append(command)
                if command == [
                    "launchctl",
                    "bootout",
                    "gui/501/com.slackgentic-team.daemon",
                ]:
                    return subprocess.CompletedProcess(
                        command,
                        5,
                        "",
                        "Bootstrap failed: 5: Input/output error",
                    )
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                patch("agent_harness.service.platform.system", return_value="Darwin"),
                patch("agent_harness.service.os.getuid", return_value=501),
                patch(
                    "agent_harness.service._launchd_path",
                    side_effect=lambda label: Path(tmp) / f"{label}.plist",
                ),
                patch("agent_harness.service.subprocess.run", side_effect=fake_run),
                self.assertRaises(RuntimeError),
            ):
                install_services([daemon, codex])

            codex_bootstrap = [
                "launchctl",
                "bootstrap",
                "gui/501",
                str(Path(tmp) / f"{codex.label}.plist"),
            ]
            daemon_bootout = [
                "launchctl",
                "bootout",
                "gui/501/com.slackgentic-team.daemon",
            ]

        self.assertLess(
            commands.index(codex_bootstrap),
            commands.index(daemon_bootout),
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
                call(["systemctl", "--user", "stop", "slackgentic-team.service"], check=False),
                call(
                    ["systemctl", "--user", "enable", "--now", "slackgentic-team.service"],
                    check=True,
                ),
            ]
        )


if __name__ == "__main__":
    unittest.main()
