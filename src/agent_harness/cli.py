from __future__ import annotations

import argparse
import contextlib
import json
import platform
import shutil
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agent_harness.models import AgentTaskKind, Provider, TeamAgentKind
from agent_harness.providers import ClaudeProvider, CodexProvider
from agent_harness.providers.quota import probe_claude_quota, read_claude_sign_ins
from agent_harness.providers.usage import (
    collect_daily_usage,
    collect_weekly_usage,
    day_string,
    format_daily_usage,
)
from agent_harness.slack import (
    build_setup_modal,
    build_start_session_modal,
    build_team_roster_blocks,
    parse_thread_ref,
)
from agent_harness.storage.store import Store
from agent_harness.team import (
    AGENT_LIMIT_MESSAGE,
    DEFAULT_CLAUDE_TEAM_SIZE,
    DEFAULT_CODEX_TEAM_SIZE,
    MAX_TEAM_AGENTS,
    build_initial_model_team,
    build_initialization_messages,
    hire_team_agents,
    runtime_personality_prompt,
)
from agent_harness.team.assignment import assign_channel_work_request

SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 14)
UV_TOOL_REINSTALL_COMMAND = (
    "uv tool install --python 3.13 --reinstall --with pip "
    "git+https://github.com/unbounded-ai/slackgentic-team.git"
)
SOURCE_VENV_RECREATE_COMMAND = (
    "rm -rf .venv && python3.13 -m venv .venv && source .venv/bin/activate && pip install -e ."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="slackgentic")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Discover local agent sessions")
    scan.add_argument("--provider", choices=["all", "codex", "claude"], default="all")
    scan.add_argument("--home", type=Path)
    scan.add_argument("--json", action="store_true")

    usage = sub.add_parser("usage", help="Summarize token usage for a day")
    usage.add_argument("--date", default="today")
    usage.add_argument("--home", type=Path)
    usage.add_argument("--json", action="store_true")

    modal = sub.add_parser("modal", help="Print Slack modal payloads")
    modal.add_argument("kind", choices=["start", "setup"])

    ref = sub.add_parser("thread-ref", help="Parse a Slack thread reference")
    ref.add_argument("text")
    ref.add_argument("--channel")
    ref.add_argument("--thread-ts")

    init_db = sub.add_parser("init-db", help="Initialize local SQLite state")
    init_db.add_argument("--db", type=Path, required=True)

    claude_channel = sub.add_parser("claude-channel", help="Run the Claude Code channel server")
    claude_channel.add_argument("--db", type=Path)
    claude_channel.add_argument(
        "--print-mcp-config",
        action="store_true",
        help="Print the MCP server config instead of running the server",
    )
    claude_channel.add_argument(
        "--install",
        action="store_true",
        help="Register the Slackgentic Claude channel in user-level Claude MCP config",
    )
    claude_channel.add_argument(
        "--native-input-hook",
        action="store_true",
        help="Run the Slackgentic Claude native input hook",
    )

    codex_mcp = sub.add_parser("codex-mcp", help="Run the Slackgentic MCP server for Codex")
    codex_mcp.add_argument("--db", type=Path)
    codex_mcp.add_argument(
        "--install",
        action="store_true",
        help="Register the Slackgentic MCP server in user-level Codex config",
    )

    loop = sub.add_parser("loop", help="Request and inspect Slackgentic loops")
    loop_sub = loop.add_subparsers(dest="loop_command", required=True)
    loop_create = loop_sub.add_parser(
        "create",
        help="Ask the running service to post a loop request for the owner to approve",
    )
    loop_create.add_argument("text", nargs="+", help="The loop's task and schedule")
    loop_create.add_argument("--provider", choices=[item.value for item in Provider])
    loop_visibility = loop_create.add_mutually_exclusive_group()
    loop_visibility.add_argument(
        "--public", dest="visibility", action="store_const", const="public"
    )
    loop_visibility.add_argument(
        "--private", dest="visibility", action="store_const", const="private"
    )
    loop_notify = loop_create.add_mutually_exclusive_group()
    loop_notify.add_argument(
        "--quiet",
        dest="quiet",
        action="store_const",
        const=True,
        help="Log all-clear runs silently in the panel thread (default)",
    )
    loop_notify.add_argument(
        "--every-run",
        dest="quiet",
        action="store_const",
        const=False,
        help="Post every run's report card to the channel",
    )
    loop_create.add_argument(
        "--no-wait",
        action="store_true",
        help="Return right after queueing instead of waiting for the service",
    )
    loop_create.add_argument("--db", type=Path)
    loop_create.add_argument("--json", action="store_true")
    loop_request = loop_sub.add_parser(
        "request-status", help="Show what happened to a queued loop request"
    )
    loop_request.add_argument("request_id")
    loop_request.add_argument("--db", type=Path)
    loop_request.add_argument("--json", action="store_true")
    loop_list = loop_sub.add_parser("list", help="List loops")
    loop_list.add_argument("--all", action="store_true", help="Include stopped loops")
    loop_list.add_argument("--db", type=Path)
    loop_list.add_argument("--json", action="store_true")

    skills = sub.add_parser("skills", help="Install agent skills that teach Slackgentic usage")
    skills_sub = skills.add_subparsers(dest="skills_command", required=True)
    skills_install = skills_sub.add_parser(
        "install", help="Install Slackgentic skills for Claude Code and Codex"
    )
    skills_install.add_argument("--claude", action="store_true", help="Only install for Claude")
    skills_install.add_argument("--codex", action="store_true", help="Only install for Codex")

    team = sub.add_parser("team", help="Manage the lightweight agent team")
    team_sub = team.add_subparsers(dest="team_command", required=True)

    team_init = team_sub.add_parser("init", help="Create the initial agent roster")
    team_init.add_argument("--db", type=Path, required=True)
    team_init.add_argument("--codex", type=int, default=DEFAULT_CODEX_TEAM_SIZE)
    team_init.add_argument("--claude", type=int, default=DEFAULT_CLAUDE_TEAM_SIZE)
    team_init.add_argument("--json", action="store_true")

    team_list = team_sub.add_parser("list", help="List team agents")
    team_list.add_argument("--db", type=Path, required=True)
    team_list.add_argument("--all", action="store_true")
    team_list.add_argument("--json", action="store_true")

    team_hire = team_sub.add_parser("hire", help="Hire more agents")
    team_hire.add_argument("count", type=int, nargs="?", default=1)
    team_hire.add_argument("--db", type=Path, required=True)
    team_hire.add_argument("--provider", choices=["auto", "codex", "claude"], default="auto")
    team_hire.add_argument(
        "--kind",
        choices=["engineer", "pm"],
        default="engineer",
        help="Agent persona kind. 'pm' agents always act as program managers.",
    )
    team_hire.add_argument("--json", action="store_true")

    team_fire = team_sub.add_parser("fire", help="Fire an agent by handle")
    team_fire.add_argument("handle")
    team_fire.add_argument("--db", type=Path, required=True)
    team_fire.add_argument("--json", action="store_true")

    team_intro = team_sub.add_parser("intros", help="Print initialization intro messages")
    team_intro.add_argument("--db", type=Path, required=True)
    team_intro.add_argument("--json", action="store_true")

    team_prompt = team_sub.add_parser("prompt", help="Print an agent runtime persona prompt")
    team_prompt.add_argument("handle")
    team_prompt.add_argument("--db", type=Path, required=True)

    team_assign = team_sub.add_parser("assign", help="Parse and assign a channel work request")
    team_assign.add_argument("text")
    team_assign.add_argument("--db", type=Path, required=True)
    team_assign.add_argument("--channel", required=True)
    team_assign.add_argument("--user")
    team_assign.add_argument("--json", action="store_true")

    team_roster = team_sub.add_parser("roster-blocks", help="Print Slack roster blocks")
    team_roster.add_argument("--db", type=Path, required=True)

    service = sub.add_parser("service", help="Install or manage the Slackgentic daemon service")
    service_sub = service.add_subparsers(dest="service_command", required=True)
    service_install = service_sub.add_parser("install", help="Install and start the daemon service")
    service_install.add_argument("--name", default="slackgentic-team")
    service_install.add_argument("--config-file", type=Path)
    service_install.add_argument("--workdir", type=Path)
    service_install.add_argument("--print-only", action="store_true")
    service_install.add_argument("--no-codex-app-server", action="store_true")
    service_install.add_argument("--codex-app-server-url", default="ws://127.0.0.1:47684")
    service_install.add_argument("--codex-binary", type=Path)
    service_install.add_argument(
        "--ignore-external-session-cwd",
        action="append",
        dest="ignored_external_session_cwds",
        help=(
            "Ignore observed external agent sessions whose cwd contains this path or path "
            "segment pattern; e.g. .local matches any .local directory."
        ),
    )
    service_install.add_argument(
        "--allow-external-session-cwd-prefix",
        action="append",
        dest="allowed_external_session_cwd_prefixes",
        help=(
            "Only mirror observed external agent sessions whose cwd is under this path prefix. "
            "May be passed more than once."
        ),
    )
    service_uninstall = service_sub.add_parser("uninstall", help="Stop and remove the service")
    service_uninstall.add_argument("--name", default="slackgentic-team")
    service_start = service_sub.add_parser("start", help="Start installed services")
    service_start.add_argument("--name", default="slackgentic-team")
    service_start.add_argument("--no-codex-app-server", action="store_true")
    service_restart = service_sub.add_parser("restart", help="Restart the daemon service")
    service_restart.add_argument("--name", default="slackgentic-team")
    service_restart.add_argument(
        "--force",
        action="store_true",
        help="Restart even if the installed service may own the Codex app-server",
    )
    service_status = service_sub.add_parser("status", help="Show service status")
    service_status.add_argument("--name", default="slackgentic-team")
    service_status.add_argument("--config-file", type=Path)
    service_print = service_sub.add_parser("print", help="Print the service definition")
    service_print.add_argument("--name", default="slackgentic-team")
    service_print.add_argument("--config-file", type=Path)
    service_print.add_argument("--workdir", type=Path)
    service_print.add_argument("--no-codex-app-server", action="store_true")
    service_print.add_argument("--codex-app-server-url", default="ws://127.0.0.1:47684")
    service_print.add_argument("--codex-binary", type=Path)
    service_print.add_argument(
        "--ignore-external-session-cwd",
        action="append",
        dest="ignored_external_session_cwds",
        help=(
            "Ignore observed external agent sessions whose cwd contains this path or path "
            "segment pattern; e.g. .local matches any .local directory."
        ),
    )
    service_print.add_argument(
        "--allow-external-session-cwd-prefix",
        action="append",
        dest="allowed_external_session_cwd_prefixes",
        help=(
            "Only mirror observed external agent sessions whose cwd is under this path prefix. "
            "May be passed more than once."
        ),
    )

    run_once = sub.add_parser("index-once", help="Index local sessions into SQLite")
    run_once.add_argument("--db", type=Path)
    run_once.add_argument("--home", type=Path)

    slack = sub.add_parser("slack", help="Run Slack integrations")
    slack_sub = slack.add_subparsers(dest="slack_command", required=True)
    slack_setup = slack_sub.add_parser("setup", help="Interactively create and install Slack app")
    slack_setup.add_argument("--config-file", type=Path)
    slack_setup.add_argument("--force", action="store_true")
    slack_setup.add_argument("--no-browser", action="store_true")
    slack_setup.add_argument("--timeout", type=int, default=600)
    slack_setup.add_argument("--serve", action="store_true")
    slack_setup.add_argument("--no-bootstrap-tools", action="store_true")
    slack_setup.add_argument(
        "--tokens-in-file",
        action="store_true",
        help="Keep Slack tokens in the config file instead of the macOS login keychain",
    )
    slack_setup.add_argument(
        "--instance",
        help="Unique local Slack app suffix, defaulting to the current OS user",
    )
    slack_update_manifest = slack_sub.add_parser(
        "update-manifest",
        help="Update the configured Slack app manifest in place",
    )
    slack_update_manifest.add_argument("--config-file", type=Path)
    slack_update_manifest.add_argument("--app-id")
    slack_update_manifest.add_argument(
        "--instance",
        help="Slack app suffix to use in generated display names",
    )
    slack_update_manifest.add_argument("--no-bootstrap-tools", action="store_true")
    slack_serve = slack_sub.add_parser("serve", help="Run the Socket Mode Slack app")
    slack_serve.add_argument("--config-file", type=Path)
    slack_serve.add_argument("--db", type=Path)
    slack_serve.add_argument("--home", type=Path)
    slack_serve.add_argument(
        "--ignore-external-session-cwd",
        action="append",
        dest="ignored_external_session_cwds",
        help=(
            "Ignore observed external agent sessions whose cwd contains this path or path "
            "segment pattern; e.g. .local matches any .local directory."
        ),
    )
    slack_serve.add_argument(
        "--allow-external-session-cwd-prefix",
        action="append",
        dest="allowed_external_session_cwd_prefixes",
        help=(
            "Only mirror observed external agent sessions whose cwd is under this path prefix. "
            "May be passed more than once."
        ),
    )
    slack_reset_state = slack_sub.add_parser(
        "reset-state",
        help="Delete local SQLite runtime state while preserving Slack credentials",
    )
    slack_reset_state.add_argument("--config-file", type=Path)
    slack_reset_state.add_argument("--db", type=Path)
    slack_reset_state.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm deletion of the configured state database",
    )
    slack_close_channel = slack_sub.add_parser(
        "close-channel",
        help="Archive the configured Slack agent channel",
    )
    slack_close_channel.add_argument("--config-file", type=Path)
    slack_close_channel.add_argument("--db", type=Path)
    slack_close_channel.add_argument(
        "--channel",
        help="Slack channel id to archive, defaulting to the configured agent channel",
    )
    slack_close_channel.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm archiving the Slack channel",
    )
    slack_tokens = slack_sub.add_parser(
        "tokens",
        help="Show or change where the Slack tokens are stored (config file or macOS keychain)",
    )
    slack_tokens.add_argument(
        "action",
        nargs="?",
        choices=("status", "keychain", "file"),
        default="status",
        help="keychain: move tokens into the login keychain; file: move them back",
    )
    slack_tokens.add_argument("--config-file", type=Path)
    slack_doctor = slack_sub.add_parser("doctor", help="Check local Slack E2E config")
    slack_doctor.add_argument("--config-file", type=Path)
    slack_doctor.add_argument("--db", type=Path)
    slack_doctor.add_argument("--home", type=Path)

    update_helper = sub.add_parser("update-helper", help=argparse.SUPPRESS)
    update_helper.add_argument("--state-db", type=Path, required=True)
    update_helper.add_argument("--log-file", type=Path, required=True)
    update_helper.add_argument("--version", required=True)
    update_helper.add_argument("--config-file", type=Path, default=None)
    update_helper.add_argument("command_args", nargs=argparse.REMAINDER)

    codex_app_server = sub.add_parser("codex-app-server", help=argparse.SUPPRESS)
    codex_app_server.add_argument("--listen", default="ws://127.0.0.1:47684")
    codex_app_server.add_argument("--codex-binary", default="codex")
    codex_app_server.add_argument("--version-check-interval", type=float, default=2.0)

    args = parser.parse_args(argv)
    runtime_python_issue = _managed_runtime_python_issue(args)
    if runtime_python_issue:
        print(runtime_python_issue)
        return 2
    if args.command == "scan":
        return _scan(args)
    if args.command == "usage":
        return _usage(args)
    if args.command == "modal":
        payload = build_start_session_modal() if args.kind == "start" else build_setup_modal()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "thread-ref":
        parsed = parse_thread_ref(args.text, args.channel, args.thread_ts)
        print(json.dumps(_jsonable(parsed), indent=2, sort_keys=True))
        return 0
    if args.command == "init-db":
        store = Store(args.db)
        try:
            store.init_schema()
        finally:
            store.close()
        print(f"initialized {args.db}")
        return 0
    if args.command == "claude-channel":
        from agent_harness.sessions.claude_channel import (
            _current_slackgentic_invocation,
            install_claude_mcp_server,
            mcp_config,
            run_channel_server,
            run_native_input_hook,
        )

        if args.native_input_hook:
            return run_native_input_hook(args.db)
        if args.print_mcp_config:
            command, command_args = _current_slackgentic_invocation()
            print(json.dumps(mcp_config(command, command_args), indent=2, sort_keys=True))
            return 0
        if args.install:
            install_claude_mcp_server()
            print("registered Claude MCP server: slackgentic")
            _install_skills_quietly("claude")
            print(
                "Start Claude with: "
                "claude --dangerously-load-development-channels server:slackgentic"
            )
            return 0
        return run_channel_server(args.db)
    if args.command == "codex-mcp":
        from agent_harness.sessions.claude_channel import (
            install_codex_mcp_server,
            run_channel_server,
        )

        if args.install:
            install_codex_mcp_server()
            print("registered Codex MCP server: slackgentic")
            _install_skills_quietly("codex")
            return 0
        return run_channel_server(args.db, provider_label="Codex")
    if args.command == "loop":
        return _loop(args)
    if args.command == "skills":
        return _skills(args)
    if args.command == "team":
        return _team(args)
    if args.command == "service":
        return _service(args)
    if args.command == "codex-app-server":
        from agent_harness.runtime.codex_app_server import run_codex_app_server_supervisor

        return run_codex_app_server_supervisor(
            args.codex_binary,
            args.listen,
            check_interval_seconds=args.version_check_interval,
        )
    if args.command == "update-helper":
        from agent_harness.updates import run_update_helper

        return run_update_helper(
            state_db=args.state_db,
            log_file=args.log_file,
            version=args.version,
            command=args.command_args,
            config_file=args.config_file,
        )
    if args.command == "index-once":
        from agent_harness.config import AppConfig
        from agent_harness.sessions.indexer import AgentDaemon

        config = AppConfig.model_validate(
            {
                key: value
                for key, value in {
                    "state_db": args.db,
                    "home": args.home,
                }.items()
                if value is not None
            }
        )
        import asyncio

        count = asyncio.run(AgentDaemon(config).index_once())
        print(f"indexed {count} sessions")
        return 0
    if args.command == "slack":
        from agent_harness.config import AppConfig, load_config_from_env
        from agent_harness.slack.app import run_slack_app

        if args.slack_command == "tokens":
            return _slack_tokens(args.action, args.config_file)
        if args.slack_command == "setup":
            from agent_harness.slack.setup import SlackSetupOptions, run_interactive_setup

            result = run_interactive_setup(
                SlackSetupOptions(
                    config_file=args.config_file,
                    open_browser=not args.no_browser,
                    timeout_seconds=args.timeout,
                    force=args.force,
                    bootstrap_tools=not args.no_bootstrap_tools,
                    instance=args.instance,
                    tokens_in_keychain=not args.tokens_in_file,
                )
            )
            if result != 0 or not args.serve:
                return result
        if args.slack_command == "update-manifest":
            from agent_harness.slack.setup import (
                SlackManifestUpdateOptions,
                update_slack_app_manifest,
            )

            return update_slack_app_manifest(
                SlackManifestUpdateOptions(
                    config_file=args.config_file,
                    app_id=args.app_id,
                    instance=args.instance,
                    bootstrap_tools=not args.no_bootstrap_tools,
                )
            )

        config = load_config_from_env(args.config_file)
        overrides = {
            key: value
            for key, value in {
                "state_db": getattr(args, "db", None),
                "home": getattr(args, "home", None),
            }.items()
            if value is not None
        }
        if overrides:
            config = AppConfig.model_validate({**config.model_dump(), **overrides})
        ignored_external_session_cwds = getattr(args, "ignored_external_session_cwds", None)
        allowed_external_session_cwd_prefixes = getattr(
            args,
            "allowed_external_session_cwd_prefixes",
            None,
        )
        if (
            ignored_external_session_cwds is not None
            or allowed_external_session_cwd_prefixes is not None
        ):
            session_updates = {}
            if ignored_external_session_cwds is not None:
                session_updates["ignored_external_session_cwds"] = tuple(
                    cwd.strip() for cwd in ignored_external_session_cwds if cwd and cwd.strip()
                )
            if allowed_external_session_cwd_prefixes is not None:
                session_updates["allowed_external_session_cwd_prefixes"] = tuple(
                    cwd.strip()
                    for cwd in allowed_external_session_cwd_prefixes
                    if cwd and cwd.strip()
                )
            config = config.model_copy(
                update={"sessions": config.sessions.model_copy(update=session_updates)}
            )
        if args.slack_command == "reset-state":
            return _reset_slack_state(config, yes=args.yes)
        if args.slack_command == "close-channel":
            return _close_slack_channel(config, channel_id=args.channel, yes=args.yes)
        if args.slack_command == "doctor":
            return _slack_doctor(config)
        if args.slack_command == "serve":
            _enable_stack_dumps()
            _raise_open_file_limit()
            return run_slack_app(config)
        if args.slack_command == "setup" and args.serve:
            _enable_stack_dumps()
            _raise_open_file_limit()
            return run_slack_app(config)
    raise AssertionError(args.command)


DAEMON_OPEN_FILE_TARGET = 4096


def _raise_open_file_limit(target: int = DAEMON_OPEN_FILE_TARGET) -> int | None:
    """Lift the daemon's soft open-file limit toward ``target``.

    launchd starts services with a 256-descriptor soft limit. Every managed
    task needs pipes to its agent process plus a SQLite connection, so a busy
    daemon can exhaust that budget. Returns the soft limit left in effect, or
    None when the platform does not expose it.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX platform
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return None
    if soft == resource.RLIM_INFINITY or soft >= target:
        return soft
    wanted = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if wanted <= soft:
        return soft
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
    except (OSError, ValueError):
        return soft
    return wanted


def _enable_stack_dumps() -> None:
    """`kill -USR1 <daemon pid>` writes every thread's stack to the service log."""
    import faulthandler
    import signal

    with contextlib.suppress(AttributeError, ValueError, RuntimeError):
        faulthandler.register(signal.SIGUSR1, all_threads=True)


def _managed_runtime_python_issue(args: argparse.Namespace) -> str | None:
    if not _command_starts_managed_runtime(args):
        return None
    version = tuple(sys.version_info[:2])
    if platform.system().lower() != "darwin" or version < SUPPORTED_PYTHON_MAX_EXCLUSIVE:
        return None
    if _installed_with_uv_tool():
        fix = (
            "This copy was installed with `uv tool install`; reinstall it on Python 3.13:\n\n"
            f"  {UV_TOOL_REINSTALL_COMMAND}\n"
        )
    else:
        fix = (
            "From a source checkout, recreate the venv on Python 3.13:\n\n"
            f"  {SOURCE_VENV_RECREATE_COMMAND}\n\n"
            "If this copy was installed with `uv tool install` instead:\n\n"
            f"  {UV_TOOL_REINSTALL_COMMAND}\n"
        )
    return (
        "Refusing to run Slackgentic managed services under Python 3.14+ on macOS "
        f"(this command is running under Python {version[0]}.{version[1]} "
        f"at {sys.executable}). Homebrew framework Python 3.14 is shown in TCC "
        'privacy prompts as "Python 3.14", which can block unattended Slackgentic '
        "agents.\n\n"
        f"{fix}\n"
        "Then run `slackgentic service install` and "
        "`slackgentic claude-channel --install` again."
    )


def _installed_with_uv_tool() -> bool:
    """uv writes a receipt into every tool environment it manages."""
    return (Path(sys.prefix) / "uv-receipt.toml").is_file()


def _command_starts_managed_runtime(args: argparse.Namespace) -> bool:
    if args.command in {"claude-channel", "codex-app-server"}:
        return True
    if args.command == "service":
        return args.service_command in {"install", "print"}
    if args.command != "slack":
        return False
    if args.slack_command == "serve":
        return True
    return args.slack_command == "setup" and bool(getattr(args, "serve", False))


def _service(args: argparse.Namespace) -> int:
    from agent_harness.service import (
        UnsafeServiceRestartError,
        build_codex_app_server_service_spec,
        build_service_spec,
        install_services,
        render_services,
        restart_service,
        service_statuses,
        start_services,
        uninstall_services,
    )

    if args.service_command in {"install", "print"}:
        daemon_spec = build_service_spec(
            name=args.name,
            working_directory=args.workdir,
            config_file=args.config_file,
            ignored_external_session_cwds=args.ignored_external_session_cwds,
            allowed_external_session_cwd_prefixes=args.allowed_external_session_cwd_prefixes,
        )
        specs = [daemon_spec]
        if not args.no_codex_app_server:
            specs.append(
                build_codex_app_server_service_spec(
                    name=args.name,
                    executable=args.codex_binary,
                    supervisor_executable=daemon_spec.executable,
                    working_directory=args.workdir,
                    url=args.codex_app_server_url,
                ),
            )
        if args.service_command == "print" or args.print_only:
            for path, content in render_services(specs):
                print(f"# {path}")
                if isinstance(content, bytes):
                    print(content.decode())
                else:
                    print(content)
            return 0
        paths = install_services(specs)
        for path in paths:
            print(f"installed service at {path}")
        return 0
    if args.service_command == "uninstall":
        for path in uninstall_services(args.name):
            print(f"removed service at {path}")
        return 0
    if args.service_command == "start":
        statuses = start_services(
            args.name,
            include_codex_app_server=not args.no_codex_app_server,
        )
        if all(status == 0 for status in statuses):
            print(f"started services for {args.name}")
            return 0
        return 1
    if args.service_command == "restart":
        try:
            result = restart_service(args.name, force=args.force)
        except UnsafeServiceRestartError as exc:
            print(f"refusing unsafe service restart: {exc}")
            return 2
        if result == 0:
            print(f"restarted service {args.name}")
        return result
    if args.service_command == "status":
        statuses = service_statuses(args.name)
        _print_stale_update_restart_warning(args.config_file)
        return 0 if all(status == 0 for status in statuses) else 1
    raise AssertionError(args.service_command)


def _print_stale_update_restart_warning(config_file: Path | None) -> None:
    from agent_harness.config import load_config_from_env
    from agent_harness.storage.store import Store
    from agent_harness.updates import SETTING_UPDATE_RESTART_PENDING, stale_restart_pending_status

    try:
        config = load_config_from_env(config_file)
    except Exception:
        return
    if not config.state_db.exists():
        return
    store = Store(config.state_db)
    try:
        warning = stale_restart_pending_status(store.get_setting(SETTING_UPDATE_RESTART_PENDING))
    except Exception:
        warning = None
    finally:
        store.close()
    if warning:
        print(warning)


def _scan(args: argparse.Namespace) -> int:
    providers = []
    if args.provider in ("all", "codex"):
        providers.append(CodexProvider(home=args.home))
    if args.provider in ("all", "claude"):
        providers.append(ClaudeProvider(home=args.home))
    sessions = [session for provider in providers for session in provider.discover()]
    if args.json:
        print(json.dumps(_jsonable(sessions), indent=2, sort_keys=True))
        return 0
    for session in sessions:
        cwd = f" cwd={session.cwd}" if session.cwd else ""
        print(
            f"{session.provider.value} {session.status.value} "
            f"{session.session_id}{cwd} path={session.transcript_path}"
        )
    return 0


def _usage(args: argparse.Namespace) -> int:
    day = day_string(args.date)
    snapshots = collect_daily_usage(day, home=args.home)
    if args.json:
        print(json.dumps(_jsonable(snapshots), indent=2, sort_keys=True))
    else:
        print(
            format_daily_usage(
                day,
                snapshots,
                collect_weekly_usage(day, home=args.home),
                claude_quota=probe_claude_quota(),
                claude_sign_ins=read_claude_sign_ins(args.home),
            )
        )
    return 0


def _loop_store(args: argparse.Namespace) -> Store:
    db_path = args.db
    if db_path is None:
        from agent_harness.config import load_config_from_env

        db_path = load_config_from_env().state_db
    store = Store(db_path)
    store.init_schema()
    return store


def _loop(args: argparse.Namespace) -> int:
    from agent_harness.loops import (
        describe_agent_loop_create_request,
        enqueue_agent_loop_create_request,
        wait_for_agent_loop_create_request,
    )
    from agent_harness.models import LoopCreateRequestStatus, LoopStatus

    store = _loop_store(args)
    try:
        if args.loop_command == "create":
            result = enqueue_agent_loop_create_request(
                store,
                " ".join(args.text),
                provider=args.provider,
                visibility=args.visibility,
                quiet=args.quiet,
                source="cli",
            )
            if result.request is None:
                print(f"error: {result.error}", file=sys.stderr)
                return 2
            request = result.request
            if not args.no_wait:
                request = wait_for_agent_loop_create_request(store, request.request_id) or request
            if args.json:
                print(json.dumps(_jsonable(request), indent=2, sort_keys=True))
            else:
                print(f"request: {request.request_id}")
                print(describe_agent_loop_create_request(request))
            return 1 if request.status == LoopCreateRequestStatus.FAILED else 0
        if args.loop_command == "request-status":
            request = store.get_loop_create_request(args.request_id)
            if args.json:
                print(json.dumps(_jsonable(request), indent=2, sort_keys=True))
            else:
                print(describe_agent_loop_create_request(request))
            return 0 if request is not None else 1
        statuses = None if args.all else tuple(s for s in LoopStatus if s != LoopStatus.CANCELLED)
        loops = store.list_loops(statuses=statuses)
        if args.json:
            print(json.dumps(_jsonable(loops), indent=2, sort_keys=True))
            return 0
        if not loops:
            print("No loops. Create one with `slackgentic loop create <task and schedule>`.")
            return 0
        for item in loops:
            channel = f"#{item.channel_name}" if item.channel_name else "(no channel yet)"
            next_run = item.next_run_at.isoformat() if item.next_run_at else "-"
            print(
                f"{item.loop_id}\t{item.status.value}\t{channel}\t{item.title}\t"
                f"next={next_run}\truns={item.run_count}"
            )
        return 0
    finally:
        store.close()


def _skills(args: argparse.Namespace) -> int:
    from agent_harness.agent_skills import install_agent_skills

    providers: tuple[str, ...] = tuple(
        name for name, selected in (("claude", args.claude), ("codex", args.codex)) if selected
    ) or ("claude", "codex")
    for path in install_agent_skills(providers=providers):
        print(f"installed {path}")
    return 0


def _install_skills_quietly(provider: str) -> None:
    from agent_harness.agent_skills import install_agent_skills

    try:
        for path in install_agent_skills(providers=(provider,)):
            print(f"installed skill: {path}")
    except OSError as exc:
        print(f"Warning: failed to install Slackgentic skills: {exc}")


def _team(args: argparse.Namespace) -> int:
    store = Store(args.db)
    try:
        store.init_schema()
        if args.team_command == "init":
            if args.codex + args.claude > MAX_TEAM_AGENTS:
                print(f"{AGENT_LIMIT_MESSAGE} Max team size is {MAX_TEAM_AGENTS}.")
                return 2
            existing = store.list_team_agents(include_fired=True)
            if existing:
                agents = store.list_team_agents()
            else:
                agents = build_initial_model_team(args.codex, args.claude)
                for agent in agents:
                    store.upsert_team_agent(agent)
            return _print_team_agents(agents, args.json)
        if args.team_command == "list":
            return _print_team_agents(store.list_team_agents(include_fired=args.all), args.json)
        if args.team_command == "hire":
            provider = None if args.provider == "auto" else Provider(args.provider)
            all_agents = store.list_team_agents(include_fired=True)
            active_agents = store.list_team_agents()
            if len(active_agents) + args.count > MAX_TEAM_AGENTS:
                print(f"{AGENT_LIMIT_MESSAGE} Max team size is {MAX_TEAM_AGENTS}.")
                return 2
            hired = hire_team_agents(
                all_agents,
                args.count,
                provider,
                start_sort_order=store.next_team_sort_order(),
                balance_agents=active_agents,
                avatar_agents=active_agents,
                randomize_identities=True,
                kind=TeamAgentKind(args.kind),
            )
            for agent in hired:
                store.upsert_team_agent(agent)
            return _print_team_agents(hired, args.json)
        if args.team_command == "fire":
            fired = store.fire_team_agent(args.handle)
            if fired is None:
                print(f"agent not found: {args.handle}")
                return 2
            if args.json:
                print(json.dumps(_jsonable(fired), indent=2, sort_keys=True))
            else:
                print(f"fired @{fired.handle} ({fired.full_name})")
            return 0
        if args.team_command == "intros":
            agents = store.list_team_agents()
            messages = build_initialization_messages(agents)
            if args.json:
                print(json.dumps(_jsonable(messages), indent=2, sort_keys=True))
            else:
                agent_by_id = {agent.agent_id: agent for agent in agents}
                for message in messages:
                    sender = agent_by_id[message.sender_agent_id]
                    print(f"@{sender.handle}: {message.text}")
            return 0
        if args.team_command == "prompt":
            agent = store.get_team_agent(args.handle)
            if agent is None:
                print(f"agent not found: {args.handle}")
                return 2
            print(runtime_personality_prompt(agent))
            return 0
        if args.team_command == "assign":
            return _assign_team_task(store, args)
        if args.team_command == "roster-blocks":
            blocks = build_team_roster_blocks(store.list_team_agents())
            print(json.dumps(blocks, indent=2, sort_keys=True))
            return 0
    finally:
        store.close()
    raise AssertionError(args.team_command)


def _assign_team_task(store: Store, args: argparse.Namespace) -> int:
    result = assign_channel_work_request(store, args.text, args.channel, args.user)
    if result is None:
        print("no idle matching agent available")
        return 2
    if args.json:
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    else:
        label = "review" if result.request.task_kind == AgentTaskKind.REVIEW else "task"
        print(f"assigned {label} to @{result.agent.handle}: {result.request.prompt}")
    return 0


def _print_team_agents(agents: list[Any], as_json: bool) -> int:
    if as_json:
        print(json.dumps(_jsonable(agents), indent=2, sort_keys=True))
        return 0
    for agent in agents:
        provider = agent.provider_preference.value if agent.provider_preference else "unmapped"
        print(f"{agent.full_name} [{provider}] @{agent.handle}")
    return 0


def _reset_slack_state(config, yes: bool = False) -> int:
    state_db = config.state_db.expanduser()
    if not yes:
        print(f"This will delete local Slackgentic runtime state: {state_db}")
        print("Slack credentials and app configuration are preserved.")
        print("Re-run with `slackgentic slack reset-state --yes` to confirm.")
        return 2

    removed: list[Path] = []
    for path in _sqlite_state_paths(state_db):
        if path.exists():
            path.unlink()
            removed.append(path)

    if removed:
        for path in removed:
            print(f"removed {path}")
    else:
        print(f"state database did not exist: {state_db}")
    print(f"Run `slackgentic service restart`, then use `{config.slack.slash_command} setup`.")
    return 0


def _close_slack_channel(config, channel_id: str | None = None, yes: bool = False) -> int:
    from slack_sdk.errors import SlackApiError

    from agent_harness.slack.app import SETTING_CHANNEL_ID, SETTING_ROSTER_TS
    from agent_harness.slack.client import SlackGateway

    if not config.slack.bot_token:
        print("SLACK_BOT_TOKEN is required to archive a Slack channel.")
        return 2

    store = Store(config.state_db)
    try:
        store.init_schema()
        resolved_channel_id = (
            channel_id
            or config.slack.channel_id
            or store.get_setting(SETTING_CHANNEL_ID)
            or store.get_setting("slack_channel_id")
        )
        if not resolved_channel_id:
            print("No Slack channel is configured. Pass `--channel C123...` to archive one.")
            return 2
        if not yes:
            print(f"This will archive Slack channel {resolved_channel_id}.")
            print("Re-run with `slackgentic slack close-channel --yes` to confirm.")
            return 2

        try:
            archived = SlackGateway(config.slack.bot_token).archive_channel(resolved_channel_id)
        except SlackApiError as exc:
            error = exc.response.get("error") if exc.response else str(exc)
            print(f"failed to archive Slack channel {resolved_channel_id}: {error}")
            return 1

        cleared = []
        for key in (SETTING_CHANNEL_ID, "slack_channel_id"):
            if store.get_setting(key) == resolved_channel_id:
                store.delete_setting(key)
                cleared.append(key)
        if cleared and store.get_setting(SETTING_ROSTER_TS):
            store.delete_setting(SETTING_ROSTER_TS)
            cleared.append(SETTING_ROSTER_TS)

        if archived:
            print(f"archived Slack channel {resolved_channel_id}")
        else:
            print(f"Slack channel {resolved_channel_id} was already archived")
        if cleared:
            print(f"cleared local channel state from {config.state_db}")
        return 0
    finally:
        store.close()


def _sqlite_state_paths(path: Path) -> list[Path]:
    return [
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        Path(f"{path}-journal"),
    ]


def _slack_tokens(action: str, config_file: Path | None) -> int:
    from agent_harness.config import (
        default_config_file,
        load_stored_config,
        move_tokens_to_file,
        move_tokens_to_keychain,
    )
    from agent_harness.keychain import KeychainError, keychain_available

    path = config_file or default_config_file()
    try:
        if action == "keychain":
            if not keychain_available():
                print("The macOS keychain is not available on this system.")
                return 1
            moved = move_tokens_to_keychain(path)
            print(
                f"Moved {', '.join(moved)} into the login keychain and out of {path}."
                if moved
                else f"No tokens left in {path}; reading them from the login keychain."
            )
            print("Restart the daemon to pick this up: slackgentic service restart")
            return 0
        if action == "file":
            restored = move_tokens_to_file(path)
            print(
                f"Moved {', '.join(restored)} back into {path} and out of the keychain."
                if restored
                else f"Tokens are already stored in {path}."
            )
            print("Restart the daemon to pick this up: slackgentic service restart")
            return 0
    except KeychainError as exc:
        print(f"Nothing changed: {exc}.")
        return 1
    print(f"Slack tokens: {_token_storage_text(load_stored_config(path), path)}")
    return 0


def _token_storage_text(stored: dict, path: Path) -> str:
    from agent_harness.config import tokens_in_keychain

    if tokens_in_keychain(stored):
        return "macOS login keychain (read without prompts via /usr/bin/security)"
    return f"config file {path}"


def _slack_doctor(config) -> int:
    from agent_harness.runtime.power import format_power_doctor_lines, inspect_macos_power
    from agent_harness.slack.client import SlackGateway
    from agent_harness.storage.store import Store

    checks = [
        ("SLACK_BOT_TOKEN", bool(config.slack.bot_token)),
        ("SLACK_APP_TOKEN", bool(config.slack.app_token)),
        ("codex binary", bool(shutil.which(config.commands.codex_binary))),
        ("claude binary", bool(shutil.which(config.commands.claude_binary))),
    ]
    ok = True
    for name, passed in checks:
        print(f"{'ok' if passed else 'missing'} {name}")
        ok = ok and passed
    print(
        f"{'ok' if config.slack.user_token else 'optional'} SLACK_USER_TOKEN "
        "(lets quiet loops delete your replies when clearing their run log)"
    )
    print(f"config file {config.config_file}")
    from agent_harness.config import load_stored_config

    print(
        "token storage "
        + _token_storage_text(load_stored_config(config.config_file), config.config_file)
    )
    print("delivery mode socket")
    print(f"slash command {config.slack.slash_command}")
    print(f"state db {config.state_db}")
    print(f"default cwd {config.commands.default_cwd}")
    print("power")
    for line in format_power_doctor_lines(inspect_macos_power()):
        print(f"  {line}")
    gateway = None
    if config.slack.bot_token:
        gateway = SlackGateway(config.slack.bot_token)
        try:
            scopes = gateway.auth_scopes()
        except Exception as exc:
            print(f"failed Slack scope check: {exc}")
            ok = False
        else:
            customize = "chat:write.customize" in scopes
            print(f"{'ok' if customize else 'missing'} loop persona scope chat:write.customize")
            print(
                f"{'ok' if 'files:write' in scopes else 'optional missing'} "
                "loop badge scope files:write"
            )
            ok = ok and customize
    loops = []
    if config.state_db.exists():
        store = Store(config.state_db)
        try:
            loops = store.list_loops(limit=1_000)
        except Exception:
            loops = []
        finally:
            store.close()
    if loops:
        print(f"loops {len(loops)} configured")
    if gateway is not None:
        for loop in loops:
            if not loop.channel_id or loop.status.value not in {"active", "paused"}:
                continue
            try:
                channel = gateway.channel_info(loop.channel_id)
            except Exception as exc:
                print(f"failed loop channel check {loop.loop_id}: {exc}")
                ok = False
                continue
            available = channel is not None and not channel.get("is_archived")
            print(f"{'ok' if available else 'unavailable'} loop channel {loop.loop_id}")
            ok = ok and available
    return 0 if ok else 2


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)
