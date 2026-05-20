import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_harness.storage.store import Store
from agent_harness.updates import (
    GitHubReleaseSource,
    ReleaseInfo,
    SelfUpdater,
    SlackgenticUpdateRunner,
    UpdateCandidate,
    UpgradePlan,
    UpgradeResult,
    github_tag_tarball_url,
    is_newer_version,
)


class UpdateVersionTests(unittest.TestCase):
    def test_semver_compare_handles_multi_digit_versions(self):
        self.assertTrue(is_newer_version("0.10.0", "0.9.9"))
        self.assertFalse(is_newer_version("1.0.0", "1.0"))
        self.assertFalse(is_newer_version("1.0.0-rc1", "1.0.0"))
        self.assertTrue(is_newer_version("1.0.0", "1.0.0-rc1"))


class GitHubReleaseSourceTests(unittest.TestCase):
    def test_latest_release_parses_github_payload(self):
        payload = {
            "tag_name": "v0.2.0",
            "html_url": "https://github.com/example-org/example-repo/releases/tag/v0.2.0",
            "tarball_url": "https://api.github.com/repos/example-org/example-repo/tarball/v0.2.0",
            "name": "v0.2.0",
            "published_at": "2026-05-20T18:00:00Z",
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
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
