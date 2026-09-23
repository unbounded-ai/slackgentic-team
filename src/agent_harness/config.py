from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_harness.keychain import (
    KEYCHAIN_CONFIG_KEY,
    SECRET_CONFIG_KEYS,
    KeychainError,
    delete_keychain_secret,
    read_keychain_secret,
    write_keychain_secret,
)
from agent_harness.team import DEFAULT_CLAUDE_TEAM_SIZE, DEFAULT_CODEX_TEAM_SIZE
from agent_harness.updates import (
    DEFAULT_UPDATE_CHECK_INTERVAL_SECONDS,
    DEFAULT_UPDATE_REPOSITORY,
)


class SlackConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    bot_token: str | None = Field(default=None, validation_alias="SLACK_BOT_TOKEN")
    app_token: str | None = Field(default=None, validation_alias="SLACK_APP_TOKEN")
    # Optional token that acts as the owner (User Token Scope chat:write). Used only
    # to delete the owner's own replies when a quiet loop's run log is cleared.
    user_token: str | None = Field(default=None, validation_alias="SLACK_USER_TOKEN")
    team_id: str | None = Field(default=None, validation_alias="SLACK_TEAM_ID")
    channel_id: str | None = Field(default=None, validation_alias="SLACK_CHANNEL_ID")
    app_id: str | None = Field(default=None, validation_alias="SLACK_APP_ID")
    instance_slug: str | None = Field(default=None, validation_alias="SLACKGENTIC_INSTANCE")
    slash_command: str = Field(
        default="/slackgentic",
        validation_alias="SLACKGENTIC_SLASH_COMMAND",
    )

    @property
    def socket_mode_ready(self) -> bool:
        return bool(self.bot_token and self.app_token)


class AgentCommandConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    codex_binary: str = Field(default="codex", validation_alias="SLACKGENTIC_CODEX_BINARY")
    claude_binary: str = Field(default="claude", validation_alias="SLACKGENTIC_CLAUDE_BINARY")
    codex_app_server_url: str | None = Field(
        default="ws://127.0.0.1:47684",
        validation_alias="SLACKGENTIC_CODEX_APP_SERVER_URL",
    )
    codex_app_server_autostart: bool = Field(
        default=True,
        validation_alias="SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART",
    )
    default_cwd: Path = Field(default_factory=Path.cwd, validation_alias="SLACKGENTIC_DEFAULT_CWD")
    dangerous_by_default: bool = Field(
        default=False, validation_alias="SLACKGENTIC_DANGEROUS_BY_DEFAULT"
    )
    allow_macos_tcc_protected_paths: bool = Field(
        default=False,
        validation_alias="SLACKGENTIC_ALLOW_MACOS_TCC_PROTECTED_PATHS",
    )
    agent_start_timeout_seconds: float = Field(
        default=120.0,
        validation_alias="SLACKGENTIC_AGENT_START_TIMEOUT_SECONDS",
    )
    agent_progress_timeout_seconds: float = Field(
        default=300.0,
        validation_alias="SLACKGENTIC_AGENT_PROGRESS_TIMEOUT_SECONDS",
    )
    agent_stall_timeout_seconds: float = Field(
        default=900.0,
        validation_alias="SLACKGENTIC_AGENT_STALL_TIMEOUT_SECONDS",
    )
    agent_stall_recovery_attempts: int = Field(
        default=2,
        validation_alias="SLACKGENTIC_AGENT_STALL_RECOVERY_ATTEMPTS",
    )


class TeamConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    default_codex_agents: int = Field(
        default=DEFAULT_CODEX_TEAM_SIZE, validation_alias="SLACKGENTIC_CODEX_AGENTS"
    )
    default_claude_agents: int = Field(
        default=DEFAULT_CLAUDE_TEAM_SIZE, validation_alias="SLACKGENTIC_CLAUDE_AGENTS"
    )


DEFAULT_EXTERNAL_SESSION_IDLE_RELEASE_SECONDS = 2 * 60 * 60.0


class SessionConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ignored_external_session_cwds: tuple[str, ...] = Field(
        default=(),
        validation_alias="SLACKGENTIC_EXTERNAL_SESSION_IGNORED_CWDS",
    )
    allowed_external_session_cwd_prefixes: tuple[str, ...] = Field(
        default=(),
        validation_alias="SLACKGENTIC_EXTERNAL_SESSION_ALLOWED_CWD_PREFIXES",
    )
    external_session_mirror_poll_seconds: float = Field(
        default=15.0,
        validation_alias="SLACKGENTIC_EXTERNAL_SESSION_MIRROR_POLL_SECONDS",
    )
    # A session started outside Slack stops occupying its agent after this long
    # without conversational activity, and claims one again when it resumes.
    # Zero disables idle release.
    external_session_idle_release_seconds: float = Field(
        default=DEFAULT_EXTERNAL_SESSION_IDLE_RELEASE_SECONDS,
        validation_alias="SLACKGENTIC_EXTERNAL_SESSION_IDLE_RELEASE_SECONDS",
    )

    @field_validator(
        "ignored_external_session_cwds",
        "allowed_external_session_cwd_prefixes",
        mode="before",
    )
    @classmethod
    def _parse_string_tuple(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ()
            if stripped.startswith("["):
                decoded = json.loads(stripped)
                return cls._parse_string_tuple(decoded)
            separator = os.pathsep if os.pathsep in stripped and "," not in stripped else ","
            return tuple(part.strip() for part in stripped.split(separator) if part.strip())
        if isinstance(value, (list, tuple, set)):
            return tuple(str(part).strip() for part in value if str(part).strip())
        return (str(value).strip(),) if str(value).strip() else ()


class UpdateConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = Field(default=True, validation_alias="SLACKGENTIC_UPDATE_CHECK_ENABLED")
    # Install new releases without waiting for a click. The `settings` card in
    # Slack overrides this per install.
    auto_install: bool = Field(default=True, validation_alias="SLACKGENTIC_UPDATE_AUTO_INSTALL")
    repository: str = Field(
        default=DEFAULT_UPDATE_REPOSITORY,
        validation_alias="SLACKGENTIC_UPDATE_REPOSITORY",
    )
    check_interval_seconds: float = Field(
        default=DEFAULT_UPDATE_CHECK_INTERVAL_SECONDS,
        validation_alias="SLACKGENTIC_UPDATE_CHECK_INTERVAL_SECONDS",
    )


class AppConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    config_file: Path = Field(default_factory=lambda: default_config_file())
    state_db: Path = Field(
        default_factory=lambda: Path.home() / ".slackgentic-team" / "state.sqlite"
    )
    home: Path = Field(default_factory=Path.home)
    poll_seconds: float = 1.0
    slack: SlackConfig = Field(default_factory=SlackConfig)
    commands: AgentCommandConfig = Field(default_factory=AgentCommandConfig)
    sessions: SessionConfig = Field(default_factory=SessionConfig)
    team: TeamConfig = Field(default_factory=TeamConfig)
    updates: UpdateConfig = Field(default_factory=UpdateConfig)


def default_config_file() -> Path:
    configured = os.environ.get("SLACKGENTIC_CONFIG_FILE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".slackgentic-team" / "config.json"


def load_stored_config(config_file: Path | None = None) -> dict[str, Any]:
    path = config_file or default_config_file()
    if not path.exists():
        return {}
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"Slackgentic config must be a JSON object: {path}")
    return {str(key): value for key, value in values.items() if value is not None}


def save_stored_config(values: dict[str, Any], config_file: Path | None = None) -> Path:
    path = config_file or default_config_file()
    existing = load_stored_config(path)
    merged = {**existing, **{key: value for key, value in values.items() if value is not None}}
    _write_stored_config(merged, path)
    if merged.get(KEYCHAIN_CONFIG_KEY) is True and any(
        merged.get(key) for key in SECRET_CONFIG_KEYS
    ):
        # Tokens saved after the move (a re-run of setup) follow the others in.
        move_tokens_to_keychain(path)
    return path


def _write_stored_config(values: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp_path, 0o600)
    tmp_path.replace(path)
    os.chmod(path, 0o600)


def tokens_in_keychain(values: dict[str, Any]) -> bool:
    return values.get(KEYCHAIN_CONFIG_KEY) is True


def resolve_keychain_tokens(values: dict[str, Any], config_file: Path) -> dict[str, Any]:
    """Fill secret keys the file and environment leave empty from the keychain."""
    if not tokens_in_keychain(values):
        return values
    resolved = dict(values)
    for key in SECRET_CONFIG_KEYS:
        if not resolved.get(key):
            secret = read_keychain_secret(key, config_file)
            if secret:
                resolved[key] = secret
    return resolved


def move_tokens_to_keychain(config_file: Path | None = None) -> list[str]:
    """Move Slack tokens from the config file into the login keychain.

    Every token is written and read back before any is removed from the file, so
    a failure part-way leaves the file untouched."""
    path = config_file or default_config_file()
    stored = load_stored_config(path)
    moved = [key for key in SECRET_CONFIG_KEYS if stored.get(key)]
    for key in moved:
        write_keychain_secret(key, str(stored[key]), path)
    remaining = {key: value for key, value in stored.items() if key not in moved}
    remaining[KEYCHAIN_CONFIG_KEY] = True
    _write_stored_config(remaining, path)
    return moved


def move_tokens_to_file(config_file: Path | None = None) -> list[str]:
    """Undo move_tokens_to_keychain: write the tokens back to the file."""
    path = config_file or default_config_file()
    stored = load_stored_config(path)
    restored: dict[str, Any] = {}
    for key in SECRET_CONFIG_KEYS:
        if stored.get(key):
            continue
        secret = read_keychain_secret(key, path)
        if secret:
            restored[key] = secret
    if (
        stored.get(KEYCHAIN_CONFIG_KEY)
        and not restored
        and not any(stored.get(key) for key in SECRET_CONFIG_KEYS)
    ):
        raise KeychainError("no Slackgentic tokens were found in the keychain")
    updated = {**stored, **restored}
    updated.pop(KEYCHAIN_CONFIG_KEY, None)
    _write_stored_config(updated, path)
    for key in restored:
        delete_keychain_secret(key, path)
    return list(restored)


def load_config_from_env(config_file: Path | None = None) -> AppConfig:
    resolved_config_file = config_file or default_config_file()
    stored_values = load_stored_config(resolved_config_file)
    env_values = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("SLACK_", "SLACKGENTIC_"))
    }
    merged_values = resolve_keychain_tokens({**stored_values, **env_values}, resolved_config_file)
    state_db = merged_values.get("SLACKGENTIC_STATE_DB")
    home = merged_values.get("SLACKGENTIC_HOME")
    config_values: dict[str, Any] = {"config_file": resolved_config_file}
    if state_db:
        config_values["state_db"] = Path(state_db)
    if home:
        config_values["home"] = Path(home)
    config_values["slack"] = SlackConfig.model_validate(merged_values)
    config_values["sessions"] = SessionConfig.model_validate(merged_values)
    config_values["team"] = TeamConfig.model_validate(merged_values)
    config_values["commands"] = AgentCommandConfig.model_validate(merged_values)
    config_values["updates"] = UpdateConfig.model_validate(merged_values)
    return AppConfig.model_validate(config_values)
