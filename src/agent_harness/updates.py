from __future__ import annotations

import importlib.metadata
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

from agent_harness import __version__
from agent_harness.models import utc_now

LOGGER = logging.getLogger(__name__)

DEFAULT_UPDATE_REPOSITORY = "unbounded-ai/slackgentic-team"
DEFAULT_UPDATE_CHECK_INTERVAL_SECONDS = 5 * 60
DEFAULT_UPGRADE_TIMEOUT_SECONDS = 10 * 60
SETTING_UPDATE_PROMPTED_VERSION = "slackgentic.update.prompted_version"
SETTING_UPDATE_DISMISSED_VERSION = "slackgentic.update.dismissed_version"
SETTING_UPDATE_INSTALLING_VERSION = "slackgentic.update.installing_version"
SETTING_UPDATE_INSTALLED_VERSION = "slackgentic.update.installed_version"
SETTING_UPDATE_LAST_CHECK_AT = "slackgentic.update.last_check_at"
SETTING_UPDATE_LAST_ERROR = "slackgentic.update.last_error"
SETTING_UPDATE_CANDIDATE_PREFIX = "slackgentic.update.candidate."
# Records the prompt message the running daemon should update once it comes
# back up on the newly-installed version. Cleared after the post-restart ack
# is posted (or after one failed attempt, to avoid retry storms).
SETTING_UPDATE_RESTART_PENDING = "slackgentic.update.restart_pending"


class UpdateCheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    tag_name: str
    html_url: str | None = None
    tarball_url: str | None = None
    name: str | None = None
    published_at: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "tag_name": self.tag_name,
                "html_url": self.html_url,
                "tarball_url": self.tarball_url,
                "name": self.name,
                "published_at": self.published_at,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, value: str) -> ReleaseInfo:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("release payload must be an object")
        version = payload.get("version")
        tag_name = payload.get("tag_name")
        if not isinstance(version, str) or not isinstance(tag_name, str):
            raise ValueError("release payload is missing version or tag_name")
        return cls(
            version=version,
            tag_name=tag_name,
            html_url=_optional_string(payload.get("html_url")),
            tarball_url=_optional_string(payload.get("tarball_url")),
            name=_optional_string(payload.get("name")),
            published_at=_optional_string(payload.get("published_at")),
        )


@dataclass(frozen=True)
class UpdateCandidate:
    current_version: str
    release: ReleaseInfo
    repository: str = DEFAULT_UPDATE_REPOSITORY

    @property
    def version(self) -> str:
        return self.release.version

    def to_json(self) -> str:
        return json.dumps(
            {
                "current_version": self.current_version,
                "release": json.loads(self.release.to_json()),
                "repository": self.repository,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, value: str) -> UpdateCandidate:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("update candidate payload must be an object")
        current_version = payload.get("current_version")
        release_payload = payload.get("release")
        repository = payload.get("repository", DEFAULT_UPDATE_REPOSITORY)
        if not isinstance(current_version, str) or not isinstance(release_payload, dict):
            raise ValueError("update candidate payload is incomplete")
        return cls(
            current_version=current_version,
            release=ReleaseInfo.from_json(json.dumps(release_payload)),
            repository=repository if isinstance(repository, str) else DEFAULT_UPDATE_REPOSITORY,
        )


class GitHubReleaseSource:
    def __init__(
        self,
        repository: str = DEFAULT_UPDATE_REPOSITORY,
        *,
        timeout_seconds: float = 10.0,
    ):
        self.repository = normalize_repository(repository)
        self.timeout_seconds = timeout_seconds

    def latest_release(self) -> ReleaseInfo | None:
        url = f"https://api.github.com/repos/{self.repository}/releases/latest"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "slackgentic-update-check",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise UpdateCheckError(f"GitHub release check failed with HTTP {exc.code}") from exc
        except OSError as exc:
            raise UpdateCheckError(f"GitHub release check failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise UpdateCheckError("GitHub release response was not an object")
        tag_name = payload.get("tag_name")
        if not isinstance(tag_name, str) or not tag_name:
            raise UpdateCheckError("GitHub release response did not include a tag name")
        return ReleaseInfo(
            version=version_from_tag(tag_name),
            tag_name=tag_name,
            html_url=_optional_string(payload.get("html_url")),
            tarball_url=_optional_string(payload.get("tarball_url")),
            name=_optional_string(payload.get("name")),
            published_at=_optional_string(payload.get("published_at")),
        )


def current_package_version() -> str:
    try:
        return importlib.metadata.version("slackgentic-team")
    except importlib.metadata.PackageNotFoundError:
        return __version__


class UpdateChecker:
    def __init__(
        self,
        release_source: GitHubReleaseSource,
        *,
        current_version: Callable[[], str] = current_package_version,
    ):
        self.release_source = release_source
        self.current_version = current_version

    def check(self) -> UpdateCandidate | None:
        release = self.release_source.latest_release()
        if release is None:
            return None
        current = self.current_version()
        if not is_newer_version(release.version, current):
            return None
        return UpdateCandidate(
            current_version=current,
            release=release,
            repository=self.release_source.repository,
        )


@dataclass(frozen=True)
class UpgradeCommand:
    args: tuple[str, ...]
    cwd: Path | None = None
    redacted_args: tuple[str, ...] | None = None

    def display_args(self) -> tuple[str, ...]:
        return self.redacted_args or self.args


@dataclass(frozen=True)
class UpgradePlan:
    description: str
    commands: tuple[UpgradeCommand, ...]
    source_root: Path | None = None


@dataclass(frozen=True)
class UpgradeCommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class UpgradeResult:
    succeeded: bool
    plan: UpgradePlan
    commands: tuple[UpgradeCommandResult, ...] = ()
    failure_message: str | None = None


class SelfUpdater:
    def __init__(
        self,
        *,
        repository: str = DEFAULT_UPDATE_REPOSITORY,
        package_file: Path | None = None,
        python_executable: str | None = None,
        timeout_seconds: float = DEFAULT_UPGRADE_TIMEOUT_SECONDS,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.repository = normalize_repository(repository)
        self.package_file = Path(package_file).resolve() if package_file is not None else None
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = timeout_seconds
        self.run = run

    def plan(self, release: ReleaseInfo) -> UpgradePlan:
        source_root = detect_source_root(self.package_file)
        if source_root is not None:
            repo_url = github_repo_url(self.repository)
            return UpgradePlan(
                description="update the local Slackgentic source checkout",
                source_root=source_root,
                commands=(
                    UpgradeCommand(
                        (
                            "git",
                            "-C",
                            str(source_root),
                            "fetch",
                            "--tags",
                            "--force",
                            repo_url,
                            f"refs/tags/{release.tag_name}:refs/tags/{release.tag_name}",
                        ),
                        redacted_args=(
                            "git",
                            "-C",
                            "<slackgentic-checkout>",
                            "fetch",
                            "--tags",
                            "--force",
                            repo_url,
                            f"refs/tags/{release.tag_name}:refs/tags/{release.tag_name}",
                        ),
                    ),
                    UpgradeCommand(
                        ("git", "-C", str(source_root), "checkout", "--detach", release.tag_name),
                        redacted_args=(
                            "git",
                            "-C",
                            "<slackgentic-checkout>",
                            "checkout",
                            "--detach",
                            release.tag_name,
                        ),
                    ),
                    UpgradeCommand(
                        (self.python_executable, "-m", "pip", "install", "-e", str(source_root)),
                        redacted_args=(
                            self.python_executable,
                            "-m",
                            "pip",
                            "install",
                            "-e",
                            "<slackgentic-checkout>",
                        ),
                    ),
                ),
            )
        install_target = release.tarball_url or github_tag_tarball_url(
            self.repository,
            release.tag_name,
        )
        return UpgradePlan(
            description="install the published Slackgentic release",
            commands=(
                UpgradeCommand(
                    (
                        self.python_executable,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        install_target,
                    ),
                ),
            ),
        )

    def install(self, release: ReleaseInfo) -> UpgradeResult:
        plan = self.plan(release)
        if plan.source_root is not None:
            dirty_message = self._dirty_source_message(plan.source_root)
            if dirty_message:
                return UpgradeResult(False, plan, failure_message=dirty_message)

        results: list[UpgradeCommandResult] = []
        for command in plan.commands:
            try:
                completed = self.run(
                    list(command.args),
                    cwd=command.cwd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return UpgradeResult(
                    False,
                    plan,
                    tuple(results),
                    f"Timed out while trying to {plan.description}.",
                )
            except OSError as exc:
                return UpgradeResult(
                    False,
                    plan,
                    tuple(results),
                    f"Could not start the update command: {exc}",
                )
            result = UpgradeCommandResult(
                command=command.display_args(),
                returncode=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
            )
            results.append(result)
            if completed.returncode != 0:
                fallback = self._pip_install_fallback(command, result)
                if fallback is not None:
                    try:
                        completed = self.run(
                            list(fallback.args),
                            cwd=fallback.cwd,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=self.timeout_seconds,
                        )
                    except subprocess.TimeoutExpired:
                        return UpgradeResult(
                            False,
                            plan,
                            tuple(results),
                            f"Timed out while trying to {plan.description}.",
                        )
                    except OSError as exc:
                        return UpgradeResult(
                            False,
                            plan,
                            tuple(results),
                            f"Could not start the update command: {exc}",
                        )
                    result = UpgradeCommandResult(
                        command=fallback.display_args(),
                        returncode=completed.returncode,
                        stdout=completed.stdout or "",
                        stderr=completed.stderr or "",
                    )
                    results.append(result)
                    if completed.returncode == 0:
                        continue
                return UpgradeResult(
                    False,
                    plan,
                    tuple(results),
                    _command_failure_message(result, plan),
                )
        return UpgradeResult(True, plan, tuple(results))

    def _pip_install_fallback(
        self,
        command: UpgradeCommand,
        result: UpgradeCommandResult,
    ) -> UpgradeCommand | None:
        if not _pip_module_missing(result):
            return None
        args = command.args
        if len(args) < 5 or args[:4] != (self.python_executable, "-m", "pip", "install"):
            return None
        uv = shutil.which("uv")
        if not uv:
            return None
        install_args = args[4:]
        redacted_install_args = (
            command.redacted_args[4:] if command.redacted_args is not None else install_args
        )
        return UpgradeCommand(
            (uv, "pip", "install", "--python", self.python_executable, *install_args),
            cwd=command.cwd,
            redacted_args=(
                "uv",
                "pip",
                "install",
                "--python",
                self.python_executable,
                *redacted_install_args,
            ),
        )

    def _dirty_source_message(self, source_root: Path) -> str | None:
        try:
            completed = self.run(
                ["git", "-C", str(source_root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"Could not inspect the Slackgentic source checkout before upgrading: {exc}"
        if completed.returncode != 0:
            return "Could not inspect the Slackgentic source checkout before upgrading."
        if completed.stdout.strip():
            return (
                "The Slackgentic source checkout has local changes, so self-upgrade "
                "will not overwrite it. Commit, stash, or move those changes, then "
                "run the upgrade again."
            )
        return None


class SlackgenticUpdateRunner:
    def __init__(
        self,
        *,
        store,
        checker: UpdateChecker,
        updater: SelfUpdater,
        channel_id: Callable[[], str | None],
        prompt: Callable[[str, UpdateCandidate], str | None],
        update_message: Callable[[str, str, str, list[dict[str, Any]] | None], None],
        status_blocks: Callable[[UpdateCandidate, str, bool], list[dict[str, Any]]],
        restart: Callable[[], None] | None = None,
        enabled: bool = True,
        poll_seconds: float = DEFAULT_UPDATE_CHECK_INTERVAL_SECONDS,
    ):
        self.store = store
        self.checker = checker
        self.updater = updater
        self.channel_id = channel_id
        self.prompt = prompt
        self.update_message = update_message
        self.status_blocks = status_blocks
        self.restart = restart
        self.enabled = enabled
        self.poll_seconds = max(60.0, poll_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._upgrade_lock = threading.Lock()
        self._upgrade_thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self.store.delete_setting(SETTING_UPDATE_INSTALLING_VERSION)
        self._confirm_pending_restart_once()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="slackgentic-update-check",
        )
        self._thread.start()

    def _confirm_pending_restart_once(self) -> None:
        # When the previous daemon process kickstarted itself after a
        # successful pip install, it recorded the prompt message it wanted
        # the newly-loaded daemon to update with a success ack. Run that ack
        # exactly once. Clear the marker after a matching-version attempt
        # (success or Slack failure) so a transient API error doesn't loop
        # on every subsequent restart. Leave the marker untouched on a
        # version mismatch — that means the kickstart did not actually
        # adopt the new build and an operator should look at it.
        raw = self.store.get_setting(SETTING_UPDATE_RESTART_PENDING)
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            self.store.delete_setting(SETTING_UPDATE_RESTART_PENDING)
            return
        channel_id = payload.get("channel_id")
        message_ts = payload.get("message_ts")
        version = payload.get("version")
        if (
            not isinstance(channel_id, str)
            or not isinstance(message_ts, str)
            or not isinstance(version, str)
        ):
            self.store.delete_setting(SETTING_UPDATE_RESTART_PENDING)
            return
        if version != __version__:
            return
        candidate = self._candidate_for_version(version) or self._fallback_candidate(version)
        tag_name = candidate.release.tag_name
        text = f":white_check_mark: Installed Slackgentic {tag_name} and restarted successfully."
        self.store.set_setting(SETTING_UPDATE_INSTALLED_VERSION, version)
        self.store.delete_setting(SETTING_UPDATE_LAST_ERROR)
        try:
            self.update_message(
                channel_id,
                message_ts,
                text,
                self.status_blocks(candidate, text, False),
            )
        except Exception:
            LOGGER.exception("failed to post post-restart upgrade ack")
        self.store.delete_setting(SETTING_UPDATE_RESTART_PENDING)

    def stop(self) -> bool:
        self._stop.set()
        stopped = True
        if self._thread:
            self._thread.join(timeout=2)
            stopped = not self._thread.is_alive()
        upgrade_thread = self._upgrade_thread
        if upgrade_thread and upgrade_thread.is_alive():
            upgrade_thread.join(timeout=2)
            stopped = stopped and not upgrade_thread.is_alive()
        return stopped

    def sync_once(self) -> UpdateCandidate | None:
        if not self.enabled:
            return None
        self.store.set_setting(SETTING_UPDATE_LAST_CHECK_AT, utc_now().isoformat())
        candidate = self.checker.check()
        if candidate is None:
            return None
        self._remember_candidate(candidate)
        prompted = self.store.get_setting(SETTING_UPDATE_PROMPTED_VERSION)
        if prompted == candidate.version:
            return candidate
        channel_id = self.channel_id()
        if not channel_id:
            return candidate
        message_ts = self.prompt(channel_id, candidate)
        if message_ts:
            self.store.set_setting(SETTING_UPDATE_PROMPTED_VERSION, candidate.version)
            self.store.delete_setting(SETTING_UPDATE_LAST_ERROR)
        return candidate

    def dismiss(self, version: str, channel_id: str, message_ts: str) -> None:
        candidate = self._candidate_for_version(version)
        if candidate is None:
            candidate = self._fallback_candidate(version)
        self.store.set_setting(SETTING_UPDATE_DISMISSED_VERSION, version)
        text = (
            f"Skipped Slackgentic {candidate.release.tag_name}. "
            "I will ask again when another version is published."
        )
        self.update_message(
            channel_id,
            message_ts,
            text,
            self.status_blocks(candidate, text, False),
        )

    def start_upgrade(
        self,
        version: str,
        channel_id: str,
        message_ts: str,
    ) -> threading.Thread | None:
        candidate = self._candidate_for_version(version)
        if candidate is None:
            candidate = self._fallback_candidate(version)
        with self._upgrade_lock:
            installing = self.store.get_setting(SETTING_UPDATE_INSTALLING_VERSION)
            if installing:
                text = f"Slackgentic {installing} is already installing."
                self.update_message(
                    channel_id,
                    message_ts,
                    text,
                    self.status_blocks(candidate, text, False),
                )
                return None
            self.store.set_setting(SETTING_UPDATE_INSTALLING_VERSION, candidate.version)
        text = f"Installing Slackgentic {candidate.release.tag_name} and preparing a restart."
        self.update_message(
            channel_id,
            message_ts,
            text,
            self.status_blocks(candidate, text, False),
        )
        thread = threading.Thread(
            target=self._upgrade_in_background,
            args=(candidate, channel_id, message_ts),
            daemon=True,
            name="slackgentic-self-upgrade",
        )
        self._upgrade_thread = thread
        thread.start()
        return thread

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            try:
                self.sync_once()
            except Exception:
                LOGGER.exception("failed to check for Slackgentic updates")
                self.store.set_setting(SETTING_UPDATE_LAST_ERROR, "update check failed")
            if self._stop.wait(self.poll_seconds):
                break

    def _upgrade_in_background(
        self,
        candidate: UpdateCandidate,
        channel_id: str,
        message_ts: str,
    ) -> None:
        try:
            result = self.updater.install(candidate.release)
            if not result.succeeded:
                message = result.failure_message or "Slackgentic upgrade failed."
                self.store.set_setting(SETTING_UPDATE_LAST_ERROR, message)
                self.update_message(
                    channel_id,
                    message_ts,
                    message,
                    self.status_blocks(candidate, message, False),
                )
                return
            self.store.delete_setting(SETTING_UPDATE_LAST_ERROR)
            message = (
                f"Installed Slackgentic {candidate.release.tag_name}. "
                "Restarting the service to load it."
            )
            self.update_message(
                channel_id,
                message_ts,
                message,
                self.status_blocks(candidate, message, False),
            )
            # Hand the post-restart success ack off to whatever daemon comes
            # up after the kickstart. The setting is the only thing that
            # survives this process exiting in a few hundred milliseconds.
            self.store.set_setting(
                SETTING_UPDATE_RESTART_PENDING,
                json.dumps(
                    {
                        "channel_id": channel_id,
                        "message_ts": message_ts,
                        "version": candidate.version,
                    },
                    sort_keys=True,
                ),
            )
            if self.restart is not None:
                self.restart()
        except Exception as exc:
            LOGGER.exception("failed to install Slackgentic update")
            message = f"Slackgentic upgrade failed: {exc}"
            self.store.set_setting(SETTING_UPDATE_LAST_ERROR, message)
            self.update_message(
                channel_id,
                message_ts,
                message,
                self.status_blocks(candidate, message, False),
            )
        finally:
            self.store.delete_setting(SETTING_UPDATE_INSTALLING_VERSION)

    def _remember_candidate(self, candidate: UpdateCandidate) -> None:
        self.store.set_setting(
            f"{SETTING_UPDATE_CANDIDATE_PREFIX}{candidate.version}",
            candidate.to_json(),
        )

    def _candidate_for_version(self, version: str) -> UpdateCandidate | None:
        raw = self.store.get_setting(f"{SETTING_UPDATE_CANDIDATE_PREFIX}{version}")
        if raw:
            try:
                return UpdateCandidate.from_json(raw)
            except (json.JSONDecodeError, ValueError):
                LOGGER.debug("stored Slackgentic update candidate was invalid", exc_info=True)
        try:
            candidate = self.checker.check()
        except Exception:
            LOGGER.debug("failed to refresh Slackgentic update candidate", exc_info=True)
            candidate = None
        if candidate is not None:
            self._remember_candidate(candidate)
            if candidate.version == version:
                return candidate
        return None

    def _fallback_candidate(self, version: str) -> UpdateCandidate:
        tag_name = version if version.startswith("v") else f"v{version}"
        return UpdateCandidate(
            current_version=current_package_version(),
            release=ReleaseInfo(
                version=version_from_tag(tag_name),
                tag_name=tag_name,
                html_url=f"https://github.com/{self.checker.release_source.repository}/releases/tag/{tag_name}",
                tarball_url=github_tag_tarball_url(
                    self.checker.release_source.repository, tag_name
                ),
            ),
            repository=self.checker.release_source.repository,
        )


def normalize_repository(value: str) -> str:
    stripped = value.strip().removeprefix("https://github.com/").removesuffix(".git")
    parts = [part for part in stripped.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"invalid GitHub repository: {value}")
    return f"{parts[-2]}/{parts[-1]}"


def github_repo_url(repository: str) -> str:
    return f"https://github.com/{normalize_repository(repository)}.git"


def github_tag_tarball_url(repository: str, tag_name: str) -> str:
    return (
        f"https://github.com/{normalize_repository(repository)}/archive/refs/tags/{tag_name}.tar.gz"
    )


def version_from_tag(tag_name: str) -> str:
    return tag_name[1:] if tag_name.startswith("v") else tag_name


def is_newer_version(candidate: str, current: str) -> bool:
    return _compare_versions(candidate, current) > 0


def detect_source_root(package_file: Path | None = None) -> Path | None:
    file_path = Path(package_file or __file__).resolve()
    candidates: list[Path] = []
    if file_path.parent.parent.name == "src":
        candidates.append(file_path.parents[2])
    candidates.extend(file_path.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not ((candidate / ".git").exists() or (candidate / ".git").is_file()):
            continue
        try:
            completed = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0 and completed.stdout.strip():
            return Path(completed.stdout.strip()).resolve()
    return None


def _command_failure_message(result: UpgradeCommandResult, plan: UpgradePlan) -> str:
    detail = (result.stderr or result.stdout).strip().splitlines()
    suffix = f": {detail[-1]}" if detail else ""
    return f"Could not {plan.description}; command exited {result.returncode}{suffix}"


def _pip_module_missing(result: UpgradeCommandResult) -> bool:
    output = f"{result.stderr}\n{result.stdout}".lower()
    return "no module named pip" in output or "no module named 'pip'" in output


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _compare_versions(left: str, right: str) -> int:
    left_key = _parse_version(left)
    right_key = _parse_version(right)
    release_cmp = _compare_release(left_key[0], right_key[0])
    if release_cmp:
        return release_cmp
    return _compare_prerelease(left_key[1], right_key[1])


def _parse_version(value: str) -> tuple[tuple[int, ...], tuple[tuple[int, int, str], ...] | None]:
    normalized = value.strip().lstrip("v")
    match = re.match(r"^(?P<release>\d+(?:\.\d+)*)(?P<suffix>.*)$", normalized)
    if not match:
        return (0,), ((1, 0, normalized.lower()),)
    release = tuple(int(part) for part in match.group("release").split("."))
    release = _trim_trailing_zeroes(release)
    suffix = (match.group("suffix") or "").strip()
    if not suffix or suffix.startswith("+"):
        return release, None
    return release, _parse_prerelease(suffix)


def _trim_trailing_zeroes(parts: tuple[int, ...]) -> tuple[int, ...]:
    trimmed = list(parts)
    while len(trimmed) > 1 and trimmed[-1] == 0:
        trimmed.pop()
    return tuple(trimmed)


def _parse_prerelease(value: str) -> tuple[tuple[int, int, str], ...]:
    cleaned = value.lstrip("-._+")
    parts: list[tuple[int, int, str]] = []
    for token in re.findall(r"\d+|[A-Za-z]+", cleaned):
        if token.isdigit():
            parts.append((0, int(token), ""))
        else:
            parts.append((1, 0, token.lower()))
    return tuple(parts) or ((1, 0, cleaned.lower()),)


def _compare_release(left: Sequence[int], right: Sequence[int]) -> int:
    for left_part, right_part in zip_longest(left, right, fillvalue=0):
        if left_part != right_part:
            return 1 if left_part > right_part else -1
    return 0


def _compare_prerelease(
    left: tuple[tuple[int, int, str], ...] | None,
    right: tuple[tuple[int, int, str], ...] | None,
) -> int:
    if left is None and right is None:
        return 0
    if left is None:
        return 1
    if right is None:
        return -1
    for left_part, right_part in zip_longest(left, right):
        if left_part is None:
            return -1
        if right_part is None:
            return 1
        if left_part != right_part:
            return 1 if left_part > right_part else -1
    return 0
