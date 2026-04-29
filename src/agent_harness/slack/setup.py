from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_harness.config import default_config_file, load_stored_config, save_stored_config

BOT_SCOPES = [
    "app_mentions:read",
    "channels:history",
    "channels:manage",
    "channels:read",
    "channels:write.invites",
    "chat:write",
    "chat:write.customize",
    "commands",
    "pins:write",
    "groups:history",
    "groups:read",
    "groups:write",
    "groups:write.invites",
    "reactions:write",
    "users:read",
]

BOT_TOKEN_RE = re.compile(r"\bxoxb-[A-Za-z0-9-]+\b")
APP_TOKEN_RE = re.compile(r"\bxapp-[A-Za-z0-9-]+\b")

CONFIG_TOKEN_URL = "https://api.slack.com/apps"
SLACK_CLI_INSTALL_URL = "https://downloads.slack-edge.com/slack-cli/install.sh"
SLACK_API_BASE = "https://slack.com/api"
INSTANCE_SLUG_MAX_LENGTH = 16


@dataclass(frozen=True)
class SlackSetupOptions:
    config_file: Path | None = None
    open_browser: bool = True
    timeout_seconds: int = 600
    force: bool = False
    bootstrap_tools: bool = True
    instance: str | None = None


@dataclass(frozen=True)
class SlackManifestUpdateOptions:
    config_file: Path | None = None
    app_id: str | None = None
    instance: str | None = None
    bootstrap_tools: bool = True


@dataclass(frozen=True)
class CreatedSlackApp:
    app_id: str | None
    raw: dict[str, Any]


class SlackApiError(RuntimeError):
    def __init__(
        self,
        label: str,
        error: str | None,
        errors: Any | None = None,
    ) -> None:
        self.label = label
        self.error = error
        self.errors = errors
        suffix = f" ({errors})" if errors else ""
        super().__init__(f"Slack {label} failed: {error}{suffix}")


def build_slack_manifest(
    instance_slug: str | None = None,
    slash_command_override: str | None = None,
) -> dict[str, Any]:
    instance = resolve_instance_slug(instance_slug)
    app_name = f"slackgentic-{instance}"
    slash_command = slash_command_override or slash_command_for_instance(instance)
    return {
        "_metadata": {"major_version": 1, "minor_version": 1},
        "display_information": {
            "name": app_name,
            "description": "Slack control plane for local Codex and Claude agent sessions.",
            "background_color": "#2457A6",
        },
        "features": {
            "bot_user": {
                "display_name": app_name,
                "always_online": True,
            },
            "slash_commands": [
                {
                    "command": slash_command,
                    "description": "Set up and manage your local agent team.",
                    "usage_hint": (
                        "setup | status | show roster | repo root ~/code | hire 2 claude agents"
                    ),
                    "should_escape": True,
                }
            ],
        },
        "oauth_config": {"scopes": {"bot": BOT_SCOPES}},
        "settings": {
            "event_subscriptions": {
                "bot_events": ["app_mention", "message.channels", "message.groups"]
            },
            "interactivity": {"is_enabled": True},
            "org_deploy_enabled": False,
            "socket_mode_enabled": True,
            "token_rotation_enabled": False,
            "is_hosted": False,
        },
    }


def build_socket_mode_manifest(
    instance_slug: str | None = None,
    slash_command_override: str | None = None,
) -> dict[str, Any]:
    return build_slack_manifest(instance_slug, slash_command_override)


def slash_command_for_instance(instance_slug: str | None = None) -> str:
    return f"/slackgentic-{resolve_instance_slug(instance_slug)}"


def resolve_instance_slug(value: str | None = None) -> str:
    raw = value or os.environ.get("SLACKGENTIC_INSTANCE") or getpass.getuser() or "local"
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return (slug or "local")[:INSTANCE_SLUG_MAX_LENGTH].strip("-") or "local"


def run_interactive_setup(options: SlackSetupOptions | None = None) -> int:
    options = options or SlackSetupOptions()
    existing = load_stored_config(options.config_file)
    if _has_existing_credentials(existing) and not options.force:
        path = options.config_file or default_config_file()
        print(f"Slackgentic is already configured at {path}. Use --force to recreate it.")
        return 0

    print("Starting Slackgentic Socket Mode setup.")
    instance = resolve_instance_slug(options.instance)
    slash_command = slash_command_for_instance(instance)
    print(f"Using Slack app instance `{instance}` with slash command `{slash_command}`.")
    config_token = _configuration_token(options, existing)
    print("Creating the Slack app from the bundled Socket Mode manifest...")
    created = _create_slack_app_with_retry(
        config_token,
        build_socket_mode_manifest(instance),
        options,
    )
    print(f"Created Slack app {created.app_id or '(unknown app id)'}.")

    bot_token = _bot_token(created, options)
    app_token = _app_token(created, options)
    auth_response = _slack_api_post("auth.test", bot_token, {})
    _verify_app_token(app_token)

    saved_path = save_stored_config(
        {
            "SLACK_BOT_TOKEN": bot_token,
            "SLACK_APP_TOKEN": app_token,
            "SLACK_TEAM_ID": auth_response.get("team_id"),
            "SLACK_APP_ID": created.app_id,
            "SLACKGENTIC_INSTANCE": instance,
            "SLACKGENTIC_SLASH_COMMAND": slash_command,
        },
        options.config_file,
    )
    print(f"Saved Slackgentic credentials to {saved_path}.")
    _install_claude_channel_if_available()
    print("Sessions started outside Slack will be mirrored with this Slack app identity.")
    command = _slackgentic_executable()
    print("Next, install and start the background service:")
    print(f"  {command} service install && {command} service status")
    print(f"Then use `{slash_command} setup` in Slack.")
    return 0


def _install_claude_channel_if_available() -> None:
    if shutil.which("claude") is None:
        print("Claude CLI not found on PATH; skipping Claude Slack channel registration.")
        return
    try:
        from agent_harness.sessions.claude_channel import install_claude_mcp_server

        install_claude_mcp_server()
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Warning: failed to register Claude Slack channel: {exc}")
        print("Run `slackgentic claude-channel --install` after Claude is available.")
        return
    print("Registered Claude Slack channel MCP server.")


def update_slack_app_manifest(options: SlackManifestUpdateOptions | None = None) -> int:
    options = options or SlackManifestUpdateOptions()
    existing = load_stored_config(options.config_file)
    app_id = options.app_id or existing.get("SLACK_APP_ID")
    if not app_id:
        raise RuntimeError("SLACK_APP_ID is required to update the Slack app manifest.")
    instance = options.instance or existing.get("SLACKGENTIC_INSTANCE") or "local"
    slash_command = _configured_slash_command(existing, instance)
    config_token = _configuration_token(
        SlackSetupOptions(
            config_file=options.config_file,
            open_browser=False,
            bootstrap_tools=options.bootstrap_tools,
            instance=instance,
        ),
        existing,
    )
    manifest = build_socket_mode_manifest(instance, slash_command_override=slash_command)
    _update_slack_app_manifest_with_retry(
        config_token,
        str(app_id),
        manifest,
        options=SlackSetupOptions(
            config_file=options.config_file,
            open_browser=False,
            bootstrap_tools=options.bootstrap_tools,
            instance=instance,
        ),
    )
    save_stored_config(
        {
            "SLACK_APP_ID": str(app_id),
            "SLACKGENTIC_INSTANCE": instance,
            "SLACKGENTIC_SLASH_COMMAND": slash_command,
        },
        options.config_file,
    )
    print(f"Updated Slack app manifest {app_id} with slash command `{slash_command}`.")
    return 0


def create_slack_app(config_token: str, manifest: dict[str, Any]) -> CreatedSlackApp:
    response = _slack_api_post(
        "apps.manifest.create",
        config_token,
        {"manifest": json.dumps(manifest)},
    )
    app = _nested_dict(response, "app")
    return CreatedSlackApp(
        app_id=response.get("app_id") or app.get("app_id") or app.get("id"),
        raw=response,
    )


def _create_slack_app_with_retry(
    config_token: str,
    manifest: dict[str, Any],
    options: SlackSetupOptions,
) -> CreatedSlackApp:
    try:
        return create_slack_app(config_token, manifest)
    except SlackApiError as exc:
        refreshed = _refreshed_config_token_after_revocation(exc, config_token, options)
        if not refreshed:
            raise
        return create_slack_app(refreshed, manifest)


def _update_slack_app_manifest_with_retry(
    config_token: str,
    app_id: str,
    manifest: dict[str, Any],
    *,
    options: SlackSetupOptions,
) -> None:
    payload = {"app_id": app_id, "manifest": json.dumps(manifest)}
    try:
        _slack_api_post("apps.manifest.update", config_token, payload)
    except SlackApiError as exc:
        refreshed = _refreshed_config_token_after_revocation(exc, config_token, options)
        if not refreshed:
            raise
        _slack_api_post("apps.manifest.update", refreshed, payload)


def _refreshed_config_token_after_revocation(
    exc: SlackApiError,
    old_token: str,
    options: SlackSetupOptions,
) -> str | None:
    if exc.error != "token_revoked":
        return None
    print("Slack CLI workspace authorization was revoked. Starting `slack login` again.")
    refreshed = _slack_cli_config_token(options, force_login=True)
    if not refreshed:
        return None
    if refreshed == old_token:
        return None
    print("Captured refreshed Slack CLI workspace authorization. Retrying Slack API call.")
    return refreshed


def _configured_slash_command(existing: dict[str, Any], instance: str) -> str:
    configured = existing.get("SLACKGENTIC_SLASH_COMMAND")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if not existing.get("SLACKGENTIC_INSTANCE"):
        return "/slackgentic"
    return slash_command_for_instance(instance)


def extract_token(text: str | None, pattern: re.Pattern[str]) -> str | None:
    if not text:
        return None
    match = pattern.search(text)
    return match.group(0) if match else None


def recursive_token_search(value: Any, pattern: re.Pattern[str]) -> str | None:
    if isinstance(value, str):
        return extract_token(value, pattern)
    if isinstance(value, dict):
        for item in value.values():
            found = recursive_token_search(item, pattern)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = recursive_token_search(item, pattern)
            if found:
                return found
    return None


def read_clipboard() -> str | None:
    commands = [
        ["pbpaste"],
        ["wl-paste", "--no-newline"],
        ["xclip", "-selection", "clipboard", "-o"],
        ["xsel", "--clipboard", "--output"],
    ]
    for command in commands:
        if not shutil.which(command[0]):
            continue
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0:
            return completed.stdout.strip()
    return None


def _configuration_token(options: SlackSetupOptions, existing: dict[str, Any]) -> str:
    token = (
        os.environ.get("SLACKGENTIC_CONFIG_TOKEN")
        or os.environ.get("SLACK_CONFIG_TOKEN")
        or existing.get("SLACKGENTIC_CONFIG_TOKEN")
        or existing.get("SLACK_CONFIG_TOKEN")
    )
    if token:
        return str(token)
    token = _slack_cli_config_token(options)
    if token:
        return token
    raise RuntimeError(
        "Slack CLI workspace authorization is required for app creation. "
        f"Run `slack login` or open {CONFIG_TOKEN_URL} through Slack's CLI flow."
    )


def _bot_token(created: CreatedSlackApp, options: SlackSetupOptions) -> str:
    token = recursive_token_search(created.raw, BOT_TOKEN_RE)
    if token:
        return token
    _open_url(_app_oauth_page(created.app_id), options.open_browser)
    print(
        "Install the app in Slack, then click Copy next to Bot User OAuth Token. "
        "I will read the xoxb token from the clipboard."
    )
    return _wait_for_clipboard_token(BOT_TOKEN_RE, "Slack bot token", options.timeout_seconds)


def _app_token(created: CreatedSlackApp, options: SlackSetupOptions) -> str:
    token = recursive_token_search(created.raw, APP_TOKEN_RE)
    if token:
        return token
    _open_url(_app_basic_page(created.app_id), options.open_browser)
    print(
        "Under App-Level Tokens, click `Generate Token and Scopes`, name it "
        "`slackgentic` (any name is fine), add the `connections:write` scope, "
        "generate the token, then click Copy. I will read the xapp token from "
        "the clipboard."
    )
    return _wait_for_clipboard_token(
        APP_TOKEN_RE,
        "Slack app-level token",
        options.timeout_seconds,
    )


def _verify_app_token(app_token: str) -> None:
    _slack_api_post("apps.connections.open", app_token, {})


def _slack_cli_config_token(
    options: SlackSetupOptions,
    *,
    force_login: bool = False,
) -> str | None:
    if not force_login:
        credentials = _read_slack_cli_credentials()
        token = _token_from_slack_cli_credentials(credentials)
        if token:
            print("Using Slack CLI workspace authorization for app creation.")
            return token
    if not options.bootstrap_tools:
        return None
    slack_cli = _ensure_slack_cli()
    if not slack_cli:
        return None
    if force_login:
        print("Refreshing Slack CLI authorization with `slack login`.")
    else:
        print("Slack CLI is not authorized yet. Starting `slack login`.")
    print(
        "Send the printed /slackauthticket command in Slack, approve the modal, "
        "and enter the challenge code here."
    )
    completed = subprocess.run([slack_cli, "login"], check=False)
    if completed.returncode != 0:
        return None
    credentials = _read_slack_cli_credentials()
    token = _token_from_slack_cli_credentials(credentials)
    if token:
        print("Captured Slack CLI workspace authorization.")
    return token


def _read_slack_cli_credentials() -> dict[str, Any]:
    path = Path.home() / ".slack" / "credentials.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _token_from_slack_cli_credentials(credentials: dict[str, Any]) -> str | None:
    now = int(time.time())
    for item in credentials.values():
        if not isinstance(item, dict):
            continue
        token = item.get("token")
        expires_at = item.get("exp")
        if isinstance(token, str) and token.startswith("xoxe"):
            if isinstance(expires_at, int) and expires_at <= now + 300:
                continue
            return token
    return None


def _ensure_slack_cli() -> str | None:
    existing = _find_slack_cli()
    if existing:
        return existing
    print("Installing Slack CLI into your user-local Slack directory...")
    try:
        script = _download_bytes(SLACK_CLI_INSTALL_URL)
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "slack-cli-install.sh"
            script_path.write_bytes(script)
            script_path.chmod(0o755)
            completed = subprocess.run(["bash", str(script_path)], check=False)
        if completed.returncode != 0:
            return None
    except Exception as exc:
        print(f"Could not install Slack CLI automatically: {exc}")
        return None
    return _find_slack_cli()


def _find_slack_cli() -> str | None:
    configured = os.environ.get("SLACKGENTIC_SLACK_CLI")
    candidates = [
        configured,
        shutil.which("slack"),
        str(Path.home() / ".local" / "bin" / "slack"),
        str(Path.home() / ".slack" / "bin" / "slack"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "slackgentic-team"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _slack_api_post(method: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{SLACK_API_BASE}/{method}",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    return _open_slack_request(request, method)


def _open_slack_request(request: urllib.request.Request, label: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Slack {label} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Slack {label} request failed: {exc}") from exc
    if not data.get("ok"):
        raise SlackApiError(label, data.get("error"), data.get("errors"))
    return data


def _wait_for_clipboard_token(
    pattern: re.Pattern[str],
    label: str,
    timeout_seconds: int,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    seen = read_clipboard()
    seen_token = extract_token(seen, pattern)
    if seen_token:
        print(
            f"The clipboard already contains a {label}. "
            "Click Slack's Copy button again so setup can capture the current value."
        )
    while time.monotonic() < deadline:
        value = read_clipboard()
        token = extract_token(value, pattern)
        if token and token != seen_token:
            print(f"Captured {label} from clipboard.")
            return token
        if token and not seen_token:
            print(f"Captured {label} from clipboard.")
            return token
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {label} on the clipboard")


def _open_url(url: str, open_browser: bool) -> None:
    if open_browser:
        webbrowser.open(url)
    print(f"Open: {url}")


def _has_existing_credentials(values: dict[str, Any]) -> bool:
    return bool(values.get("SLACK_BOT_TOKEN") and values.get("SLACK_APP_TOKEN"))


def _slackgentic_executable() -> str:
    if shutil.which("slackgentic"):
        return "slackgentic"
    executable = Path(sys.argv[0])
    if executable.name == "slackgentic":
        return str(executable)
    return "slackgentic"


def _nested_dict(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def _app_basic_page(app_id: str | None) -> str:
    if app_id:
        return f"https://api.slack.com/apps/{app_id}/general"
    return "https://api.slack.com/apps"


def _app_oauth_page(app_id: str | None) -> str:
    if app_id:
        return f"https://api.slack.com/apps/{app_id}/oauth"
    return "https://api.slack.com/apps"
