"""Keep Slack tokens in the macOS login keychain instead of the config file.

Items are written and read only through Apple's ``/usr/bin/security`` tool. The
tool that creates an item is on its access list, so the daemon reads its tokens
without an "allow access" prompt, including after upgrades that replace the
Python interpreter. The trade-off: any process running as the user can read the
items the same way. This keeps tokens out of plain-text files, backups and greps;
it is not a barrier against the user's own processes.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SECURITY_BINARY = "/usr/bin/security"
KEYCHAIN_SERVICE = "slackgentic-team"
# Config flag: secret keys missing from the file are looked up in the keychain.
KEYCHAIN_CONFIG_KEY = "SLACKGENTIC_TOKENS_IN_KEYCHAIN"
SECRET_CONFIG_KEYS = ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_USER_TOKEN")
_TIMEOUT_SECONDS = 10
# Slack tokens are ASCII letters, digits and dashes; anything else is refused so
# the value can be passed to `security -i` without quoting surprises.
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")
_ACCOUNT_PATTERN = re.compile(r"^[^\s\"'\\]+$")


class KeychainError(RuntimeError):
    pass


def keychain_available() -> bool:
    return sys.platform == "darwin" and Path(SECURITY_BINARY).exists()


def keychain_account(key: str, config_file: Path) -> str:
    """One item per token per config file, so separate instances never collide."""
    return f"{key}@{config_file.expanduser().resolve()}"


def read_keychain_secret(key: str, config_file: Path) -> str | None:
    if not keychain_available():
        return None
    try:
        result = subprocess.run(
            [
                SECURITY_BINARY,
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                keychain_account(key, config_file),
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def write_keychain_secret(key: str, value: str, config_file: Path) -> None:
    if not keychain_available():
        raise KeychainError("the macOS keychain is not available on this system")
    if not _TOKEN_PATTERN.fullmatch(value):
        raise KeychainError(f"{key} does not look like a Slack token")
    account = keychain_account(key, config_file)
    if not _ACCOUNT_PATTERN.fullmatch(account):
        # Paths with spaces or quotes would need escaping inside `security -i`.
        raise KeychainError(f"config path {config_file} contains unsupported characters")
    # `security -i` reads the command from stdin, so the token never appears in
    # the process list the way a -w argument would.
    command = f"add-generic-password -U -s {KEYCHAIN_SERVICE} -a {account} -w {value}\n"
    try:
        result = subprocess.run(
            [SECURITY_BINARY, "-i"],
            input=command,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise KeychainError(f"could not write {key} to the keychain: {exc}") from exc
    if result.returncode != 0 or result.stderr.strip():
        raise KeychainError(
            f"could not write {key} to the keychain: {result.stderr.strip() or result.returncode}"
        )
    if read_keychain_secret(key, config_file) != value:
        raise KeychainError(f"{key} did not read back from the keychain")


def delete_keychain_secret(key: str, config_file: Path) -> bool:
    if not keychain_available():
        return False
    try:
        result = subprocess.run(
            [
                SECURITY_BINARY,
                "delete-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                keychain_account(key, config_file),
            ],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
