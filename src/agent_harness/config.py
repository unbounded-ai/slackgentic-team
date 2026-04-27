from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_harness.team import DEFAULT_CLAUDE_TEAM_SIZE, DEFAULT_CODEX_TEAM_SIZE


class SlackConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    bot_token: str | None = Field(default=None, validation_alias="SLACK_BOT_TOKEN")
    app_token: str | None = Field(default=None, validation_alias="SLACK_APP_TOKEN")
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


class TeamConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    default_codex_agents: int = Field(
        default=DEFAULT_CODEX_TEAM_SIZE, validation_alias="SLACKGENTIC_CODEX_AGENTS"
    )
    default_claude_agents: int = Field(
        default=DEFAULT_CLAUDE_TEAM_SIZE, validation_alias="SLACKGENTIC_CLAUDE_AGENTS"
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
    team: TeamConfig = Field(default_factory=TeamConfig)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_stored_config(path)
    merged = {**existing, **{key: value for key, value in values.items() if value is not None}}
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp_path, 0o600)
    tmp_path.replace(path)
    os.chmod(path, 0o600)
    return path


def load_config_from_env(config_file: Path | None = None) -> AppConfig:
    resolved_config_file = config_file or default_config_file()
    stored_values = load_stored_config(resolved_config_file)
    env_values = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("SLACK_", "SLACKGENTIC_"))
    }
    merged_values = {**stored_values, **env_values}
    state_db = merged_values.get("SLACKGENTIC_STATE_DB")
    home = merged_values.get("SLACKGENTIC_HOME")
    config_values: dict[str, Any] = {"config_file": resolved_config_file}
    if state_db:
        config_values["state_db"] = Path(state_db)
    if home:
        config_values["home"] = Path(home)
    config_values["slack"] = SlackConfig.model_validate(merged_values)
    config_values["team"] = TeamConfig.model_validate(merged_values)
    config_values["commands"] = AgentCommandConfig.model_validate(merged_values)
    return AppConfig.model_validate(config_values)
