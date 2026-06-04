import json
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_harness.models import utc_now
from agent_harness.storage.store import Store
from agent_harness.updates import (
    SETTING_UPDATE_INSTALLED_VERSION,
    SETTING_UPDATE_RESTART_HELPER,
    SETTING_UPDATE_RESTART_PENDING,
    GitHubReleaseSource,
    ReleaseInfo,
    SelfUpdater,
    SlackgenticUpdateRunner,
    UpdateCandidate,
    UpgradePlan,
    UpgradeResult,
    github_tag_tarball_url,
    is_newer_version,
    run_update_helper,
)


class UpdateVersionTests(unittest.TestCase):
    def test_semver_compare_handles_multi_digit_versions(self):
        self.assertTrue(is_newer_version("0.10.0", "0.9.9"))
        self.assertFalse(is_newer_version("1.0.0", "1.0"))
        self.assertFalse(is_newer_version("1.0.0-rc1", "1.0.0"))
        self.assertTrue(is_newer_version("1.0.0", "1.0.0-rc1"))

    def test_package_version_matches_project_metadata(self):
        from agent_harness import __version__

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        metadata = tomllib.loads(pyproject.read_text())

        self.assertEqual(__version__, metadata["project"]["version"])


class GitHubReleaseSourceTests(unittest.TestCase):
    def test_latest_release_parses_github_payload(self):
        payload = {
            "tag_name": "v0.2.0",
            "html_url": "https://github.com/example-org/example-repo/releases/tag/v0.2.0",
            "tarball_url": "https://api.github.com/repos/example-org/example-repo/tarball/v0.2.0",
            "name": "v0.2.0",
            "published_at": "2026-05-20T18:00:00Z",
            "body": "## Changes\n- Improve service restart handling",
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(payload).encode()

        with patch("agent_harness.updates.urllib.request.urlopen", return_value=Response()):
            release = GitHubReleaseSource("example-org/example-repo").latest_release()

        assert release is not None
        self.assertEqual(release.version, "0.2.0")
        self.assertEqual(release.tag_name, "v0.2.0")
        self.assertEqual(release.tarball_url, payload["tarball_url"])
        self.assertEqual(release.body, payload["body"])


class SelfUpdaterTests(unittest.TestCase):
    def test_pip_plan_uses_release_tarball_when_not_running_from_source(self):
        release = ReleaseInfo(version="0.2.0", tag_name="v0.2.0")
        with patch("agent_harness.updates.detect_source_root", return_value=None):
            plan = SelfUpdater(
                repository="example-org/example-repo",
                python_executable="/venv/bin/python",
            ).plan(release)

        self.assertEqual(plan.description, "install the published Slackgentic release")
        self.assertEqual(
            plan.commands[0].args,
            (
                "/venv/bin/python",
                "-m",
                "pip",
                "install",
                "--upgrade",
                github_tag_tarball_url("example-org/example-repo", "v0.2.0"),
            ),
        )

    def test_source_plan_fetches_tag_and_reinstalls_editable_checkout(self):
        release = ReleaseInfo(version="0.2.0", tag_name="v0.2.0")
        with patch("agent_harness.updates.detect_source_root", return_value=Path("/repo")):
            plan = SelfUpdater(
                repository="example-org/example-repo",
                python_executable="/venv/bin/python",
            ).plan(release)

        commands = [command.display_args() for command in plan.commands]
        self.assertEqual(plan.description, "update the local Slackgentic source checkout")
        self.assertIn("https://github.com/example-org/example-repo.git", commands[0])
        self.assertEqual(commands[1][-2:], ("--detach", "v0.2.0"))
        self.assertEqual(commands[2][-2:], ("-e", "<slackgentic-checkout>"))

    def test_install_refuses_dirty_source_checkout(self):
        release = ReleaseInfo(version="0.2.0", tag_name="v0.2.0")
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=" M src/file.py\n", stderr="")

        with patch("agent_harness.updates.detect_source_root", return_value=Path("/repo")):
            result = SelfUpdater(run=fake_run).install(release)

        self.assertFalse(result.succeeded)
        self.assertIn("local changes", result.failure_message)
        self.assertEqual(len(calls), 1)

    def test_install_runs_source_commands_after_clean_check(self):
        release = ReleaseInfo(version="0.2.0", tag_name="v0.2.0")
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with patch("agent_harness.updates.detect_source_root", return_value=Path("/repo")):
            result = SelfUpdater(run=fake_run, python_executable="/venv/bin/python").install(
                release
            )

        self.assertTrue(result.succeeded)
        self.assertEqual(calls[0][-2:], ["status", "--porcelain"])
        self.assertIn("fetch", calls[1])
        self.assertEqual(calls[2][-2:], ["--detach", "v0.2.0"])
        self.assertEqual(calls[3][:4], ["/venv/bin/python", "-m", "pip", "install"])

    def test_install_falls_back_to_uv_when_python_has_no_pip(self):
        release = ReleaseInfo(
            version="0.2.0",
            tag_name="v0.2.0",
            tarball_url="https://example.com/release.tar.gz",
        )
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            if args[:4] == ["/venv/bin/python", "-m", "pip", "install"]:
                return subprocess.CompletedProcess(
                    args,
                    1,
                    stdout="",
                    stderr="/venv/bin/python: No module named pip\n",
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with (
            patch("agent_harness.updates.detect_source_root", return_value=None),
            patch("agent_harness.updates.shutil.which", return_value="/usr/bin/uv"),
        ):
            result = SelfUpdater(
                run=fake_run,
                python_executable="/venv/bin/python",
            ).install(release)

        self.assertTrue(result.succeeded)
        self.assertEqual(
            calls,
            [
                [
                    "/venv/bin/python",
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "https://example.com/release.tar.gz",
                ],
                [
                    "/usr/bin/uv",
                    "pip",
                    "install",
                    "--python",
                    "/venv/bin/python",
                    "--upgrade",
                    "https://example.com/release.tar.gz",
                ],
            ],
        )
        self.assertEqual(
            result.commands[-1].command,
            (
                "uv",
                "pip",
                "install",
                "--python",
                "/venv/bin/python",
                "--upgrade",
                "https://example.com/release.tar.gz",
            ),
        )


class UpdateRunnerTests(unittest.TestCase):
    def test_sync_once_prompts_once_for_new_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                candidate = UpdateCandidate(
                    current_version="0.1.0",
                    release=ReleaseInfo(version="0.2.0", tag_name="v0.2.0"),
                    repository="example-org/example-repo",
                )
                prompts = []

                class Checker:
                    release_source = GitHubReleaseSource("example-org/example-repo")

                    def check(self):
                        return candidate

                runner = SlackgenticUpdateRunner(
                    store=store,
                    checker=Checker(),
                    updater=object(),
                    channel_id=lambda: "C1",
                    prompt=lambda channel_id, update: prompts.append((channel_id, update)) or "171",
                    update_message=lambda channel_id, ts, text, blocks: None,
                    status_blocks=lambda update, status, include_actions: [],
                )

                self.assertEqual(runner.sync_once(), candidate)
                self.assertEqual(runner.sync_once(), candidate)

                self.assertEqual(prompts, [("C1", candidate)])
            finally:
                store.close()

    def test_start_upgrade_installs_and_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                candidate = UpdateCandidate(
                    current_version="0.1.0",
                    release=ReleaseInfo(version="0.2.0", tag_name="v0.2.0"),
                    repository="example-org/example-repo",
                )
                store.set_setting("slackgentic.update.candidate.0.2.0", candidate.to_json())
                updates = []
                restarts = []

                class Checker:
                    release_source = GitHubReleaseSource("example-org/example-repo")

                    def check(self):
                        return None

                class Updater:
                    def install(self, release):
                        return UpgradeResult(True, UpgradePlan("test upgrade", ()))

                runner = SlackgenticUpdateRunner(
                    store=store,
                    checker=Checker(),
                    updater=Updater(),
                    channel_id=lambda: "C1",
                    prompt=lambda channel_id, update: "171",
                    update_message=lambda channel_id, ts, text, blocks: updates.append(text),
                    status_blocks=lambda update, status, include_actions: [],
                    restart=lambda: restarts.append(True),
                )

                thread = runner.start_upgrade("0.2.0", "C1", "171")
                assert thread is not None
                thread.join(timeout=2)

                self.assertFalse(thread.is_alive())
                self.assertEqual(restarts, [True])
                self.assertIn("Installed Slackgentic v0.2.0", updates[-1])
                self.assertIsNone(store.get_setting(SETTING_UPDATE_INSTALLED_VERSION))
                # The pre-restart message should hand off the post-restart
                # ack to the next daemon by recording where to update.
                pending = store.get_setting("slackgentic.update.restart_pending")
                self.assertIsNotNone(pending)
                payload = json.loads(pending)
                self.assertEqual(payload["channel_id"], "C1")
                self.assertIn("created_at", payload)
                self.assertEqual(payload["message_ts"], "171")
                self.assertEqual(payload["version"], "0.2.0")
            finally:
                store.close()

    def test_start_posts_post_restart_ack_when_pending_matches_current_version(self):
        from agent_harness import __version__

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                candidate = UpdateCandidate(
                    current_version="0.0.0",
                    release=ReleaseInfo(version=__version__, tag_name=f"v{__version__}"),
                    repository="example-org/example-repo",
                )
                store.set_setting(
                    f"slackgentic.update.candidate.{__version__}", candidate.to_json()
                )
                store.set_setting(
                    SETTING_UPDATE_RESTART_PENDING,
                    json.dumps(
                        {
                            "channel_id": "C1",
                            "created_at": utc_now().isoformat(),
                            "message_ts": "999",
                            "version": __version__,
                        }
                    ),
                )
                updates: list[tuple[str, str, str]] = []

                class Checker:
                    release_source = GitHubReleaseSource("example-org/example-repo")

                    def check(self):
                        return None

                runner = SlackgenticUpdateRunner(
                    store=store,
                    checker=Checker(),
                    updater=object(),
                    channel_id=lambda: "C1",
                    prompt=lambda channel_id, update: None,
                    update_message=lambda channel_id, ts, text, blocks: updates.append(
                        (channel_id, ts, text)
                    ),
                    status_blocks=lambda update, status, include_actions: [],
                    # Disable the background check loop so the test stays
                    # focused on the startup ack behavior.
                    enabled=True,
                )
                runner.start()
                try:
                    self.assertEqual(len(updates), 1)
                    channel_id, ts, text = updates[0]
                    self.assertEqual(channel_id, "C1")
                    self.assertEqual(ts, "999")
                    self.assertIn(":white_check_mark:", text)
                    self.assertIn(f"v{__version__}", text)
                    self.assertIn("restarted successfully", text)
                    self.assertIsNone(store.get_setting("slackgentic.update.restart_pending"))
                    self.assertEqual(
                        store.get_setting(SETTING_UPDATE_INSTALLED_VERSION),
                        __version__,
                    )
                finally:
                    runner.stop()
            finally:
                store.close()

    def test_post_restart_ack_waits_for_matching_update_helper(self):
        from agent_harness import __version__

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                candidate = UpdateCandidate(
                    current_version="0.0.0",
                    release=ReleaseInfo(version=__version__, tag_name=f"v{__version__}"),
                    repository="example-org/example-repo",
                )
                store.set_setting(
                    f"slackgentic.update.candidate.{__version__}", candidate.to_json()
                )
                store.set_setting(
                    SETTING_UPDATE_RESTART_PENDING,
                    json.dumps(
                        {
                            "channel_id": "C1",
                            "created_at": utc_now().isoformat(),
                            "message_ts": "999",
                            "version": __version__,
                        }
                    ),
                )
                store.set_setting(
                    SETTING_UPDATE_RESTART_HELPER,
                    json.dumps({"phase": "running", "version": __version__}),
                )
                updates: list[tuple[str, str, str]] = []

                class Checker:
                    release_source = GitHubReleaseSource("example-org/example-repo")

                    def check(self):
                        return None

                runner = SlackgenticUpdateRunner(
                    store=store,
                    checker=Checker(),
                    updater=object(),
                    channel_id=lambda: "C1",
                    prompt=lambda channel_id, update: None,
                    update_message=lambda channel_id, ts, text, blocks: updates.append(
                        (channel_id, ts, text)
                    ),
                    status_blocks=lambda update, status, include_actions: [],
                )

                runner._confirm_pending_restart_once()

                self.assertEqual(updates, [])
                self.assertIsNotNone(store.get_setting(SETTING_UPDATE_RESTART_PENDING))

                store.set_setting(
                    SETTING_UPDATE_RESTART_HELPER,
                    json.dumps({"phase": "succeeded", "version": __version__}),
                )
                runner._confirm_pending_restart_once()

                self.assertEqual(len(updates), 1)
                self.assertIn(":white_check_mark:", updates[0][2])
                self.assertIsNone(store.get_setting(SETTING_UPDATE_RESTART_PENDING))
            finally:
                store.close()

    def test_post_restart_ack_retains_pending_marker_when_slack_update_fails(self):
        from agent_harness import __version__

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                candidate = UpdateCandidate(
                    current_version="0.0.0",
                    release=ReleaseInfo(version=__version__, tag_name=f"v{__version__}"),
                    repository="example-org/example-repo",
                )
                store.set_setting(
                    f"slackgentic.update.candidate.{__version__}", candidate.to_json()
                )
                store.set_setting(
                    SETTING_UPDATE_RESTART_PENDING,
                    json.dumps(
                        {
                            "channel_id": "C1",
                            "created_at": utc_now().isoformat(),
                            "message_ts": "999",
                            "version": __version__,
                        }
                    ),
                )
                attempts = 0
                updates: list[tuple[str, str, str]] = []

                class Checker:
                    release_source = GitHubReleaseSource("example-org/example-repo")

                    def check(self):
                        return None

                def update_message(channel_id, ts, text, blocks):
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        raise RuntimeError("temporary Slack failure")
                    updates.append((channel_id, ts, text))

                runner = SlackgenticUpdateRunner(
                    store=store,
                    checker=Checker(),
                    updater=object(),
                    channel_id=lambda: "C1",
                    prompt=lambda channel_id, update: None,
                    update_message=update_message,
                    status_blocks=lambda update, status, include_actions: [],
                )

                runner._confirm_pending_restart_once()

                self.assertEqual(updates, [])
                self.assertIsNotNone(store.get_setting(SETTING_UPDATE_RESTART_PENDING))

                runner._confirm_pending_restart_once()

                self.assertEqual(len(updates), 1)
                self.assertEqual(attempts, 2)
                self.assertIsNone(store.get_setting(SETTING_UPDATE_RESTART_PENDING))
            finally:
                store.close()

    def test_start_does_not_ack_when_pending_version_does_not_match(self):
        # If the kickstart somehow brought up a daemon at the OLD version
        # (e.g. the install step was rolled back), do not lie to the user by
        # claiming a successful upgrade. Leave the pending marker alone so
        # the operator can investigate.
        from agent_harness import __version__

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                pending_payload = {
                    "channel_id": "C1",
                    "created_at": utc_now().isoformat(),
                    "message_ts": "999",
                    "version": f"{__version__}-rollback-target",
                }
                store.set_setting(SETTING_UPDATE_RESTART_PENDING, json.dumps(pending_payload))
                updates: list[tuple[str, str, str]] = []

                class Checker:
                    release_source = GitHubReleaseSource("example-org/example-repo")

                    def check(self):
                        return None

                runner = SlackgenticUpdateRunner(
                    store=store,
                    checker=Checker(),
                    updater=object(),
                    channel_id=lambda: "C1",
                    prompt=lambda channel_id, update: None,
                    update_message=lambda channel_id, ts, text, blocks: updates.append(
                        (channel_id, ts, text)
                    ),
                    status_blocks=lambda update, status, include_actions: [],
                )
                runner.start()
                try:
                    self.assertEqual(updates, [])
                    # Marker is intentionally retained — we don't want to
                    # quietly drop the breadcrumb when the version mismatch
                    # might mean a partial rollback.
                    self.assertEqual(
                        json.loads(store.get_setting("slackgentic.update.restart_pending")),
                        pending_payload,
                    )
                finally:
                    runner.stop()
            finally:
                store.close()

    def test_start_warns_when_pending_restart_is_stale(self):
        from agent_harness import __version__

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                candidate = UpdateCandidate(
                    current_version="0.0.0",
                    release=ReleaseInfo(version=__version__, tag_name=f"v{__version__}"),
                    repository="example-org/example-repo",
                )
                store.set_setting(
                    f"slackgentic.update.candidate.{__version__}", candidate.to_json()
                )
                store.set_setting(
                    SETTING_UPDATE_RESTART_PENDING,
                    json.dumps(
                        {
                            "channel_id": "C1",
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "message_ts": "999",
                            "version": __version__,
                        }
                    ),
                )
                updates: list[tuple[str, str, str]] = []

                class Checker:
                    release_source = GitHubReleaseSource("example-org/example-repo")

                    def check(self):
                        return None

                runner = SlackgenticUpdateRunner(
                    store=store,
                    checker=Checker(),
                    updater=object(),
                    channel_id=lambda: "C1",
                    prompt=lambda channel_id, update: None,
                    update_message=lambda channel_id, ts, text, blocks: updates.append(
                        (channel_id, ts, text)
                    ),
                    status_blocks=lambda update, status, include_actions: [],
                )
                runner.start()
                try:
                    self.assertEqual(len(updates), 1)
                    self.assertIn("automatic service restart did not confirm", updates[0][2])
                    self.assertIsNone(store.get_setting(SETTING_UPDATE_RESTART_PENDING))
                    self.assertEqual(
                        store.get_setting(SETTING_UPDATE_INSTALLED_VERSION),
                        __version__,
                    )
                finally:
                    runner.stop()
            finally:
                store.close()

    def test_update_helper_records_exit_code_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_db = Path(tmp) / "state.sqlite"
            log_file = Path(tmp) / "update.log"

            def fake_run(args, **kwargs):
                kwargs["stdout"].write("helper output\n")
                return subprocess.CompletedProcess(args, 7)

            result = run_update_helper(
                state_db=state_db,
                log_file=log_file,
                version="0.2.0",
                command=["--", "slackgentic", "service", "install"],
                run=fake_run,
            )

            store = Store(state_db)
            try:
                payload = json.loads(store.get_setting(SETTING_UPDATE_RESTART_HELPER))
            finally:
                store.close()

            self.assertEqual(result, 7)
            self.assertEqual(payload["phase"], "failed")
            self.assertEqual(payload["exit_code"], 7)
            self.assertEqual(payload["command"], ["slackgentic", "service", "install"])
            self.assertIn("helper output", log_file.read_text())


if __name__ == "__main__":
    unittest.main()
