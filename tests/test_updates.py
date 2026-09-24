import json
import subprocess
import tempfile
import threading
import tomllib
import types
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from agent_harness.models import utc_now
from agent_harness.storage.store import Store
from agent_harness.updates import (
    SETTING_UPDATE_AUTO_ATTEMPTED_VERSION,
    SETTING_UPDATE_AUTO_INSTALL,
    SETTING_UPDATE_CANDIDATE_PREFIX,
    SETTING_UPDATE_CHECKS_ENABLED,
    SETTING_UPDATE_DISMISSED_VERSION,
    SETTING_UPDATE_INSTALLED_VERSION,
    SETTING_UPDATE_INSTALLING_VERSION,
    SETTING_UPDATE_LAST_ERROR,
    SETTING_UPDATE_PROMPT_MESSAGE,
    SETTING_UPDATE_PROMPTED_VERSION,
    SETTING_UPDATE_RESTART_HELPER,
    SETTING_UPDATE_RESTART_PENDING,
    UPDATE_HELPER_FOLLOWUP_SECONDS,
    GitHubReleaseSource,
    ReleaseInfo,
    SelfUpdater,
    SlackgenticUpdateRunner,
    UpdateCandidate,
    UpdateChecker,
    UpgradePlan,
    UpgradeResult,
    finalize_restart_pending,
    github_tag_tarball_url,
    is_newer_version,
    run_update_helper,
)
from tests.polling import POLL_TIMEOUT_SECONDS, wait_until


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

    def test_current_version_matches_running_module(self):
        from agent_harness import __version__
        from agent_harness.updates import current_package_version

        self.assertEqual(current_package_version(), __version__)


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


class UpdateCheckerTests(unittest.TestCase):
    def test_check_collects_notes_from_every_release_the_upgrade_spans(self):
        def release(version):
            return ReleaseInfo(
                version=version, tag_name=f"v{version}", body=f"- Change in {version}"
            )

        class Source:
            repository = "example-org/example-repo"

            def latest_release(self):
                return release("0.4.0")

            def releases(self):
                return [release(v) for v in ("0.4.0", "0.3.0", "0.2.0", "0.1.0")]

        candidate = UpdateChecker(Source(), current_version=lambda: "0.2.0").check()

        assert candidate is not None
        self.assertEqual(candidate.release.body, "- Change in 0.4.0\n\n- Change in 0.3.0")


class SelfUpdaterTests(unittest.TestCase):
    def test_pip_plan_uses_installable_archive_when_not_running_from_source(self):
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

    def test_pip_plan_rewrites_github_api_tarball_url(self):
        release = ReleaseInfo(
            version="0.2.0",
            tag_name="v0.2.0",
            tarball_url="https://api.github.com/repos/example-org/example-repo/tarball/v0.2.0",
        )
        with patch("agent_harness.updates.detect_source_root", return_value=None):
            plan = SelfUpdater(
                repository="example-org/example-repo",
                python_executable="/venv/bin/python",
            ).plan(release)

        self.assertEqual(
            plan.commands[0].args[-1],
            github_tag_tarball_url("example-org/example-repo", "v0.2.0"),
        )

    def test_pip_plan_preserves_supported_release_archive_url(self):
        release = ReleaseInfo(
            version="0.2.0",
            tag_name="v0.2.0",
            tarball_url="https://example.com/release.tar.gz",
        )
        with patch("agent_harness.updates.detect_source_root", return_value=None):
            plan = SelfUpdater(
                repository="example-org/example-repo",
                python_executable="/venv/bin/python",
            ).plan(release)

        self.assertEqual(plan.commands[0].args[-1], "https://example.com/release.tar.gz")

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

    def test_install_retries_transient_published_release_failure(self):
        release = ReleaseInfo(
            version="0.2.0",
            tag_name="v0.2.0",
            tarball_url="https://example.com/release.tar.gz",
        )
        calls = []
        sleeps = []

        def fake_run(args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                return subprocess.CompletedProcess(
                    args,
                    2,
                    stdout="",
                    stderr="error: request failed\n  Caused by: operation timed out\n",
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with patch("agent_harness.updates.detect_source_root", return_value=None):
            result = SelfUpdater(
                run=fake_run,
                python_executable="/venv/bin/python",
                sleep=sleeps.append,
            ).install(release)

        self.assertTrue(result.succeeded)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(sleeps, [2.0])
        self.assertEqual([command.returncode for command in result.commands], [2, 0])

    def test_install_does_not_retry_non_transient_failure(self):
        release = ReleaseInfo(
            version="0.2.0",
            tag_name="v0.2.0",
            tarball_url="https://example.com/release.tar.gz",
        )
        calls = []
        sleeps = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="permission denied")

        with patch("agent_harness.updates.detect_source_root", return_value=None):
            result = SelfUpdater(
                run=fake_run,
                python_executable="/venv/bin/python",
                sleep=sleeps.append,
            ).install(release)

        self.assertFalse(result.succeeded)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sleeps, [])
        self.assertIn("permission denied", result.failure_message)

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

    def test_install_falls_back_to_uv_with_supported_archive_url(self):
        release = ReleaseInfo(
            version="0.2.0",
            tag_name="v0.2.0",
            tarball_url="https://api.github.com/repos/example-org/example-repo/tarball/v0.2.0",
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
                repository="example-org/example-repo",
                run=fake_run,
                python_executable="/venv/bin/python",
            ).install(release)

        archive_url = github_tag_tarball_url("example-org/example-repo", "v0.2.0")
        self.assertTrue(result.succeeded)
        self.assertEqual(calls[0][-1], archive_url)
        self.assertEqual(calls[1][-1], archive_url)


class _StaticChecker:
    release_source = GitHubReleaseSource("example-org/example-repo")

    def __init__(self, candidate):
        self.candidate = candidate
        self.calls = 0

    def check(self):
        self.calls += 1
        return self.candidate


class AutoUpdateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._tmp.name) / "state.sqlite")
        self.store.init_schema()
        self.candidate = UpdateCandidate(
            current_version="0.1.0",
            release=ReleaseInfo(version="99.0.0", tag_name="v99.0.0"),
            repository="example-org/example-repo",
        )
        self.checker = _StaticChecker(self.candidate)
        self.updates = []
        self.upgrades = []
        self.busy = False

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _runner(self, **kwargs):
        runner = SlackgenticUpdateRunner(
            store=self.store,
            checker=self.checker,
            updater=object(),
            channel_id=lambda: "C1",
            prompt=lambda channel_id, update: "171",
            update_message=lambda channel_id, ts, text, blocks: self.updates.append(
                (channel_id, ts, text)
            ),
            status_blocks=lambda update, status, include_actions: [],
            is_busy=lambda: self.busy,
            **kwargs,
        )
        runner.start_upgrade = lambda version, channel_id, ts: self.upgrades.append(
            (version, channel_id, ts)
        )
        return runner

    def test_auto_install_upgrades_from_the_posted_card(self):
        runner = self._runner(auto_install=True)

        runner.sync_once()
        runner.sync_once()

        self.assertEqual(self.upgrades, [("99.0.0", "C1", "171")])
        self.assertEqual(self.store.get_setting(SETTING_UPDATE_AUTO_ATTEMPTED_VERSION), "99.0.0")

    def test_auto_install_off_only_prompts(self):
        runner = self._runner(auto_install=False)

        runner.sync_once()

        self.assertEqual(self.upgrades, [])

    def test_auto_install_does_not_wait_for_an_idle_team(self):
        # The upgrade drains running agents itself; waiting here for a fully
        # idle team could postpone the update forever.
        runner = self._runner(auto_install=True)
        self.busy = True

        runner.sync_once()

        self.assertEqual(self.upgrades, [("99.0.0", "C1", "171")])

    def test_auto_install_skips_dismissed_version(self):
        self.store.set_setting(SETTING_UPDATE_DISMISSED_VERSION, "99.0.0")
        runner = self._runner(auto_install=True)

        runner.sync_once()

        self.assertEqual(self.upgrades, [])

    def test_slack_setting_overrides_configured_default(self):
        runner = self._runner(auto_install=True)
        runner.set_auto_install_enabled(False)

        runner.sync_once()

        self.assertFalse(runner.auto_install_enabled())
        self.assertEqual(self.store.get_setting(SETTING_UPDATE_AUTO_INSTALL), "off")
        self.assertEqual(self.upgrades, [])

    def test_turning_auto_install_on_retries_a_previously_attempted_release(self):
        self.store.set_setting(SETTING_UPDATE_AUTO_ATTEMPTED_VERSION, "99.0.0")
        runner = self._runner(auto_install=False)

        runner.sync_once()
        self.assertEqual(self.upgrades, [])

        runner.set_auto_install_enabled(True)
        runner.sync_once()

        self.assertEqual(self.upgrades, [("99.0.0", "C1", "171")])

    def test_release_checks_off_skips_the_network(self):
        self.store.set_setting(SETTING_UPDATE_CHECKS_ENABLED, "off")
        runner = self._runner(auto_install=True)

        self.assertIsNone(runner.sync_once())

        self.assertEqual(self.checker.calls, 0)
        self.assertEqual(self.upgrades, [])


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

    def test_newer_prompt_retires_the_previous_prompt_buttons(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                store.set_setting(
                    SETTING_UPDATE_PROMPT_MESSAGE,
                    json.dumps({"version": "0.1.9", "channel_id": "C1", "ts": "160"}),
                )
                candidate = UpdateCandidate(
                    current_version="0.1.0",
                    release=ReleaseInfo(version="0.2.0", tag_name="v0.2.0"),
                    repository="example-org/example-repo",
                )
                updates = []
                statuses = []

                class Checker:
                    release_source = GitHubReleaseSource("example-org/example-repo")

                    def check(self):
                        return candidate

                runner = SlackgenticUpdateRunner(
                    store=store,
                    checker=Checker(),
                    updater=object(),
                    channel_id=lambda: "C1",
                    prompt=lambda channel_id, update: "171",
                    update_message=lambda channel_id, ts, text, blocks: updates.append(
                        (channel_id, ts, text)
                    ),
                    status_blocks=lambda update, status, include_actions: (
                        statuses.append((update.version, include_actions)) or []
                    ),
                )

                runner.sync_once()

                self.assertEqual(
                    updates,
                    [("C1", "160", "Superseded by Slackgentic v0.2.0; use the newer card.")],
                )
                self.assertEqual(statuses, [("0.1.9", False)])
                self.assertEqual(
                    json.loads(store.get_setting(SETTING_UPDATE_PROMPT_MESSAGE)),
                    {"version": "0.2.0", "channel_id": "C1", "ts": "171"},
                )
            finally:
                store.close()

    def test_newer_prompt_keeps_the_outcome_of_an_installed_or_dismissed_card(self):
        for setting, value in (
            (None, None),
            ("slackgentic.update.dismissed_version", "0.1.9"),
        ):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as tmp:
                store = Store(Path(tmp) / "state.sqlite")
                try:
                    store.init_schema()
                    if setting:
                        store.set_setting(setting, value)
                    store.set_setting(
                        SETTING_UPDATE_PROMPT_MESSAGE,
                        json.dumps({"version": "0.1.9", "channel_id": "C1", "ts": "160"}),
                    )
                    # Without a dismissal the old card's release is the one running.
                    candidate = UpdateCandidate(
                        current_version="0.1.0" if setting else "0.1.9",
                        release=ReleaseInfo(version="0.2.0", tag_name="v0.2.0"),
                        repository="example-org/example-repo",
                    )
                    updates = []

                    checker = types.SimpleNamespace(
                        release_source=GitHubReleaseSource("example-org/example-repo"),
                        check=lambda candidate=candidate: candidate,
                    )

                    runner = SlackgenticUpdateRunner(
                        store=store,
                        checker=checker,
                        updater=object(),
                        channel_id=lambda: "C1",
                        prompt=lambda channel_id, update: "171",
                        update_message=lambda *args, updates=updates: updates.append(args),
                        status_blocks=lambda update, status, include_actions: [],
                    )

                    runner.sync_once()

                    self.assertEqual(updates, [])
                finally:
                    store.close()

    def test_sync_once_reprompts_same_version_after_failed_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                candidate = UpdateCandidate(
                    current_version="0.1.0",
                    release=ReleaseInfo(version="0.2.0", tag_name="v0.2.0"),
                    repository="example-org/example-repo",
                )
                store.set_setting(SETTING_UPDATE_PROMPTED_VERSION, "0.2.0")
                store.set_setting(SETTING_UPDATE_LAST_ERROR, "previous install failed")
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

                self.assertEqual(prompts, [("C1", candidate)])
                self.assertEqual(store.get_setting(SETTING_UPDATE_PROMPTED_VERSION), "0.2.0")
                self.assertIsNone(store.get_setting(SETTING_UPDATE_LAST_ERROR))
            finally:
                store.close()

    def test_start_upgrade_installs_and_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                candidate = UpdateCandidate(
                    current_version="0.1.0",
                    release=ReleaseInfo(version="99.0.0", tag_name="v99.0.0"),
                    repository="example-org/example-repo",
                )
                store.set_setting("slackgentic.update.candidate.99.0.0", candidate.to_json())
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

                thread = runner.start_upgrade("99.0.0", "C1", "171")
                assert thread is not None
                thread.join(timeout=POLL_TIMEOUT_SECONDS)

                self.assertFalse(thread.is_alive())
                self.assertEqual(restarts, [True])
                self.assertIn("Installed Slackgentic v99.0.0", updates[-1])
                self.assertIsNone(store.get_setting(SETTING_UPDATE_INSTALLED_VERSION))
                # The pre-restart message should hand off the post-restart
                # ack to the next daemon by recording where to update.
                pending = store.get_setting("slackgentic.update.restart_pending")
                self.assertIsNotNone(pending)
                payload = json.loads(pending)
                self.assertEqual(payload["channel_id"], "C1")
                self.assertIn("created_at", payload)
                self.assertEqual(payload["message_ts"], "171")
                self.assertEqual(payload["version"], "99.0.0")
            finally:
                store.close()

    def test_start_upgrade_rejects_older_target_without_installing(self):
        from agent_harness import __version__

        stale_version = "0.1.67"
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                candidate = UpdateCandidate(
                    current_version="0.1.0",
                    release=ReleaseInfo(version=stale_version, tag_name=f"v{stale_version}"),
                    repository="example-org/example-repo",
                )
                store.set_setting(
                    f"slackgentic.update.candidate.{stale_version}", candidate.to_json()
                )
                store.set_setting(SETTING_UPDATE_PROMPTED_VERSION, stale_version)
                installs = []
                updates = []
                action_flags = []

                class Checker:
                    release_source = GitHubReleaseSource("example-org/example-repo")

                    def check(self):
                        return None

                class Updater:
                    def install(self, release):
                        installs.append(release)
                        return UpgradeResult(True, UpgradePlan("test upgrade", ()))

                runner = SlackgenticUpdateRunner(
                    store=store,
                    checker=Checker(),
                    updater=Updater(),
                    channel_id=lambda: "C1",
                    prompt=lambda channel_id, update: "171",
                    update_message=lambda channel_id, ts, text, blocks: updates.append(text),
                    status_blocks=lambda update, status, include_actions: (
                        action_flags.append(include_actions) or []
                    ),
                )

                thread = runner.start_upgrade(stale_version, "C1", "171")

                self.assertIsNone(thread)
                self.assertEqual(installs, [])
                self.assertIn("is no longer newer than the running version", updates[-1])
                self.assertIn(__version__, updates[-1])
                self.assertFalse(action_flags[-1])
                self.assertIsNone(store.get_setting(SETTING_UPDATE_INSTALLING_VERSION))
                self.assertIsNone(store.get_setting(SETTING_UPDATE_PROMPTED_VERSION))
            finally:
                store.close()

    def test_start_upgrade_failure_keeps_retry_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                candidate = UpdateCandidate(
                    current_version="0.1.0",
                    release=ReleaseInfo(version="99.0.0", tag_name="v99.0.0"),
                    repository="example-org/example-repo",
                )
                store.set_setting("slackgentic.update.candidate.99.0.0", candidate.to_json())
                store.set_setting(SETTING_UPDATE_PROMPTED_VERSION, "99.0.0")
                updates = []
                action_flags = []

                class Checker:
                    release_source = GitHubReleaseSource("example-org/example-repo")

                    def check(self):
                        return None

                class Updater:
                    def install(self, release):
                        return UpgradeResult(
                            False,
                            UpgradePlan("test upgrade", ()),
                            failure_message="install failed",
                        )

                runner = SlackgenticUpdateRunner(
                    store=store,
                    checker=Checker(),
                    updater=Updater(),
                    channel_id=lambda: "C1",
                    prompt=lambda channel_id, update: "171",
                    update_message=lambda channel_id, ts, text, blocks: updates.append(text),
                    status_blocks=lambda update, status, include_actions: (
                        action_flags.append(include_actions) or []
                    ),
                )

                thread = runner.start_upgrade("99.0.0", "C1", "171")
                assert thread is not None
                thread.join(timeout=POLL_TIMEOUT_SECONDS)

                self.assertFalse(thread.is_alive())
                self.assertEqual(updates[-1], "install failed")
                self.assertTrue(action_flags[-1])
                self.assertIsNone(store.get_setting(SETTING_UPDATE_PROMPTED_VERSION))
                self.assertEqual(store.get_setting(SETTING_UPDATE_LAST_ERROR), "install failed")
            finally:
                store.close()

    def test_start_posts_post_restart_ack_after_slow_startup(self):
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
                            "created_at": (utc_now() - timedelta(minutes=3)).isoformat(),
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

    def test_update_loop_rechecks_running_helper_before_release_poll(self):
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
                    f"{SETTING_UPDATE_CANDIDATE_PREFIX}{__version__}",
                    candidate.to_json(),
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

                    def __init__(self):
                        self.calls = 0

                    def check(self):
                        self.calls += 1
                        return None

                checker = Checker()
                runner = SlackgenticUpdateRunner(
                    store=store,
                    checker=checker,
                    updater=object(),
                    channel_id=lambda: "C1",
                    prompt=lambda channel_id, update: None,
                    update_message=lambda channel_id, ts, text, blocks: updates.append(
                        (channel_id, ts, text)
                    ),
                    status_blocks=lambda update, status, include_actions: [],
                    poll_seconds=300,
                )

                self.assertEqual(runner._run_once(), UPDATE_HELPER_FOLLOWUP_SECONDS)
                self.assertEqual(checker.calls, 0)
                self.assertEqual(updates, [])

                store.set_setting(
                    SETTING_UPDATE_RESTART_HELPER,
                    json.dumps({"phase": "succeeded", "version": __version__}),
                )
                self.assertEqual(runner._run_once(), 300)

                self.assertEqual(checker.calls, 1)
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
                            "created_at": (utc_now() - timedelta(minutes=5)).isoformat(),
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
                    self.assertIn("within 5 minutes", updates[0][2])
                    self.assertNotIn("after a later start", updates[0][2])
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


class _FakeClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, tzinfo=UTC)
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self._now += timedelta(seconds=seconds)


class _Checker:
    release_source = GitHubReleaseSource("example-org/example-repo")

    def check(self):
        return None


class RestartDrainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._tmp.name) / "state.sqlite")
        self.store.init_schema()
        candidate = UpdateCandidate(
            current_version="0.1.0",
            release=ReleaseInfo(version="99.0.0", tag_name="v99.0.0"),
            repository="example-org/example-repo",
        )
        self.store.set_setting(f"{SETTING_UPDATE_CANDIDATE_PREFIX}99.0.0", candidate.to_json())
        self.busy = threading.Event()
        self.updates = []
        self.draining_cards = []
        self.installs = []
        self.restarts = []
        self.paused = []
        self.install_succeeds = True

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _runner(self, **kwargs):
        test = self

        class Updater:
            def install(self, release):
                test.installs.append(release.version)
                if test.install_succeeds:
                    return UpgradeResult(True, UpgradePlan("test upgrade", ()))
                return UpgradeResult(
                    False, UpgradePlan("test upgrade", ()), failure_message="install failed"
                )

        def status_blocks(update, status, include_actions, *, draining=False):
            if draining:
                self.draining_cards.append(status)
            return []

        return SlackgenticUpdateRunner(
            store=self.store,
            checker=_Checker(),
            updater=Updater(),
            channel_id=lambda: "C1",
            prompt=lambda channel_id, update: "171",
            update_message=lambda channel_id, ts, text, blocks: self.updates.append(text),
            status_blocks=status_blocks,
            restart=lambda: self.restarts.append(True),
            is_busy=self.busy.is_set,
            pause_new_work=self.paused.append,
            drain_poll_seconds=0.01,
            **kwargs,
        )

    def _join(self, thread):
        assert thread is not None
        thread.join(timeout=POLL_TIMEOUT_SECONDS)
        self.assertFalse(thread.is_alive())

    def test_idle_team_installs_without_draining(self):
        thread = self._runner().start_upgrade("99.0.0", "C1", "171")
        self._join(thread)

        self.assertEqual(self.draining_cards, [])
        self.assertEqual(self.installs, ["99.0.0"])
        self.assertEqual(self.restarts, [True])
        # New work stays paused into the restart; the next daemon accepts work.
        self.assertEqual(self.paused, [True])

    def test_waits_for_mid_turn_agents_before_installing(self):
        self.busy.set()
        runner = self._runner()

        thread = runner.start_upgrade("99.0.0", "C1", "171")

        self.assertTrue(wait_until(lambda: self.draining_cards))
        self.assertEqual(self.paused, [True])
        self.assertEqual(self.installs, [])
        self.busy.clear()
        self._join(thread)
        self.assertEqual(self.installs, ["99.0.0"])
        self.assertEqual(self.restarts, [True])

    def test_drain_timeout_restarts_anyway(self):
        self.busy.set()
        runner = self._runner(drain_timeout_seconds=0.05)

        self._join(runner.start_upgrade("99.0.0", "C1", "171"))

        self.assertEqual(len(self.draining_cards), 1)
        self.assertEqual(self.installs, ["99.0.0"])
        self.assertEqual(self.restarts, [True])

    def test_install_now_skips_the_wait(self):
        self.busy.set()
        runner = self._runner()
        thread = runner.start_upgrade("99.0.0", "C1", "171")
        self.assertTrue(wait_until(lambda: self.draining_cards))

        self.assertIsNone(runner.start_upgrade("99.0.0", "C1", "171", force=True))

        self._join(thread)
        self.assertEqual(self.installs, ["99.0.0"])
        self.assertEqual(self.restarts, [True])

    def test_forced_upgrade_never_drains(self):
        self.busy.set()

        self._join(self._runner().start_upgrade("99.0.0", "C1", "171", force=True))

        self.assertEqual(self.draining_cards, [])
        self.assertEqual(self.restarts, [True])

    def test_failed_install_resumes_new_work(self):
        self.install_succeeds = False

        self._join(self._runner().start_upgrade("99.0.0", "C1", "171"))

        self.assertEqual(self.paused, [True, False])
        self.assertEqual(self.restarts, [])
        self.assertIsNone(self.store.get_setting(SETTING_UPDATE_INSTALLING_VERSION))

    def test_stopping_mid_drain_does_not_install(self):
        self.busy.set()
        runner = self._runner()
        thread = runner.start_upgrade("99.0.0", "C1", "171")
        self.assertTrue(wait_until(lambda: self.draining_cards))

        runner.stop()

        self._join(thread)
        self.assertEqual(self.installs, [])
        self.assertEqual(self.restarts, [])


class UpdateHelperConfirmTests(unittest.TestCase):
    def _store(self, tmp: str) -> Store:
        store = Store(Path(tmp) / "state.sqlite")
        store.init_schema()
        return store

    def _set_pending(self, store: Store, version: str) -> None:
        store.set_setting(
            SETTING_UPDATE_RESTART_PENDING,
            json.dumps(
                {
                    "channel_id": "C1",
                    "created_at": utc_now().isoformat(),
                    "message_ts": "171",
                    "version": version,
                }
            ),
        )
        candidate = UpdateCandidate(
            current_version="0.1.0",
            release=ReleaseInfo(version=version, tag_name=f"v{version}"),
            repository="example-org/example-repo",
        )
        store.set_setting(f"{SETTING_UPDATE_CANDIDATE_PREFIX}{version}", candidate.to_json())

    def test_finalize_posts_terminal_status_when_daemon_never_confirms(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            try:
                self._set_pending(store, "0.2.0")
                clock = _FakeClock()
                posts: list[tuple] = []

                outcome = finalize_restart_pending(
                    store,
                    version="0.2.0",
                    helper_phase="succeeded",
                    update_message=lambda *args: posts.append(args),
                    status_blocks=lambda candidate, status, actions: [{"type": "section"}],
                    poll_seconds=5,
                    sleep=clock.sleep,
                    now=clock.now,
                )

                self.assertEqual(outcome, "notified")
                self.assertEqual(sum(clock.slept), 6 * 60)
                self.assertEqual(len(posts), 1)
                channel_id, message_ts, text, blocks = posts[0]
                self.assertEqual((channel_id, message_ts), ("C1", "171"))
                self.assertIn("did not confirm", text)
                self.assertIn("within 5 minutes", text)
                self.assertEqual(blocks, [{"type": "section"}])
                self.assertIsNone(store.get_setting(SETTING_UPDATE_RESTART_PENDING))
                self.assertEqual(store.get_setting(SETTING_UPDATE_INSTALLED_VERSION), "0.2.0")
            finally:
                store.close()

    def test_finalize_waits_for_slow_daemon_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            try:
                self._set_pending(store, "0.2.0")
                clock = _FakeClock()
                posts: list[tuple] = []

                def clearing_sleep(seconds: float) -> None:
                    clock.sleep(seconds)
                    # Startup reconciliation can outlast the old helper timeout.
                    if sum(clock.slept) >= 4 * 60:
                        store.delete_setting(SETTING_UPDATE_RESTART_PENDING)

                outcome = finalize_restart_pending(
                    store,
                    version="0.2.0",
                    helper_phase="succeeded",
                    update_message=lambda *args: posts.append(args),
                    status_blocks=lambda *args: [],
                    poll_seconds=5,
                    sleep=clearing_sleep,
                    now=clock.now,
                )

                self.assertEqual(outcome, "confirmed")
                self.assertEqual(sum(clock.slept), 4 * 60)
                self.assertEqual(posts, [])
            finally:
                store.close()

    def test_finalize_uses_reinstall_failure_text_when_helper_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            try:
                self._set_pending(store, "0.2.0")
                clock = _FakeClock()
                posts: list[tuple] = []

                finalize_restart_pending(
                    store,
                    version="0.2.0",
                    helper_phase="failed",
                    update_message=lambda *args: posts.append(args),
                    timeout_seconds=0,
                    poll_seconds=5,
                    sleep=clock.sleep,
                    now=clock.now,
                )

                self.assertEqual(len(posts), 1)
                self.assertIn("reinstall failed", posts[0][2])
            finally:
                store.close()

    def test_finalize_retains_marker_when_slack_update_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            try:
                self._set_pending(store, "0.2.0")

                def boom(*args):
                    raise RuntimeError("slack down")

                outcome = finalize_restart_pending(
                    store,
                    version="0.2.0",
                    helper_phase="succeeded",
                    update_message=boom,
                    timeout_seconds=0,
                    poll_seconds=5,
                )

                self.assertIsNone(outcome)
                self.assertIsNotNone(store.get_setting(SETTING_UPDATE_RESTART_PENDING))
                self.assertIsNone(store.get_setting(SETTING_UPDATE_INSTALLED_VERSION))
            finally:
                store.close()

    def test_finalize_noop_without_pending_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            try:
                posts: list[tuple] = []
                outcome = finalize_restart_pending(
                    store,
                    version="0.2.0",
                    helper_phase="succeeded",
                    update_message=lambda *args: posts.append(args),
                )
                self.assertIsNone(outcome)
                self.assertEqual(posts, [])
            finally:
                store.close()

    def test_run_update_helper_finalizes_with_install_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_db = Path(tmp) / "state.sqlite"
            log_file = Path(tmp) / "update.log"
            captured: dict = {}

            def fake_finalize(*, state_db, version, helper_phase, config_file):
                captured.update(
                    state_db=state_db,
                    version=version,
                    helper_phase=helper_phase,
                    config_file=config_file,
                )
                return "notified"

            def fake_run(args, **kwargs):
                kwargs["stdout"].write("ok\n")
                return subprocess.CompletedProcess(args, 0)

            result = run_update_helper(
                state_db=state_db,
                log_file=log_file,
                version="0.2.0",
                command=["--", "slackgentic", "service", "install"],
                config_file=Path("/tmp/example-config.json"),
                run=fake_run,
                finalize_restart=fake_finalize,
            )

            self.assertEqual(result, 0)
            self.assertEqual(captured["helper_phase"], "succeeded")
            self.assertEqual(captured["version"], "0.2.0")
            self.assertEqual(captured["config_file"], Path("/tmp/example-config.json"))
            self.assertIn("post-restart status: notified", log_file.read_text())


if __name__ == "__main__":
    unittest.main()
