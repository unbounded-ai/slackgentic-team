from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agent_harness.assignment import assign_channel_work_request
from agent_harness.models import AgentTaskKind, Provider
from agent_harness.personas import avatar_svg, generate_persona
from agent_harness.providers import ClaudeProvider, CodexProvider
from agent_harness.slack import (
    build_setup_modal,
    build_start_session_modal,
    build_team_roster_blocks,
    parse_thread_ref,
)
from agent_harness.store import Store
from agent_harness.team import (
    build_initial_model_team,
    build_initialization_messages,
    hire_team_agents,
    runtime_personality_prompt,
)
from agent_harness.usage import collect_daily_usage, day_string, format_daily_usage


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

    persona = sub.add_parser("persona", help="Generate deterministic session persona")
    persona.add_argument("provider", choices=[item.value for item in Provider])
    persona.add_argument("session_id")
    persona.add_argument("--svg", action="store_true")

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

    team = sub.add_parser("team", help="Manage the lightweight agent team")
    team_sub = team.add_subparsers(dest="team_command", required=True)

    team_init = team_sub.add_parser("init", help="Create the initial agent roster")
    team_init.add_argument("--db", type=Path, required=True)
    team_init.add_argument("--codex", type=int, default=5)
    team_init.add_argument("--claude", type=int, default=5)
    team_init.add_argument("--json", action="store_true")

    team_list = team_sub.add_parser("list", help="List team agents")
    team_list.add_argument("--db", type=Path, required=True)
    team_list.add_argument("--all", action="store_true")
    team_list.add_argument("--json", action="store_true")

    team_hire = team_sub.add_parser("hire", help="Hire more agents")
    team_hire.add_argument("count", type=int, nargs="?", default=1)
    team_hire.add_argument("--db", type=Path, required=True)
    team_hire.add_argument("--provider", choices=["auto", "codex", "claude"], default="auto")
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
    service_print = service_sub.add_parser("print", help="Print the service definition")
    service_print.add_argument("--name", default="slackgentic-team")
    service_print.add_argument("--config-file", type=Path)
    service_print.add_argument("--workdir", type=Path)
    service_print.add_argument("--no-codex-app-server", action="store_true")
    service_print.add_argument("--codex-app-server-url", default="ws://127.0.0.1:47684")
    service_print.add_argument("--codex-binary", type=Path)

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
    slack_doctor = slack_sub.add_parser("doctor", help="Check local Slack E2E config")
    slack_doctor.add_argument("--config-file", type=Path)
    slack_doctor.add_argument("--db", type=Path)
    slack_doctor.add_argument("--home", type=Path)

    args = parser.parse_args(argv)
    if args.command == "scan":
        return _scan(args)
    if args.command == "usage":
        return _usage(args)
    if args.command == "persona":
        return _persona(args)
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
        from agent_harness.claude_channel import (
            _current_slackgentic_invocation,
            install_claude_mcp_server,
            mcp_config,
            run_channel_server,
        )

        if args.print_mcp_config:
            command, command_args = _current_slackgentic_invocation()
            print(json.dumps(mcp_config(command, command_args), indent=2, sort_keys=True))
            return 0
        if args.install:
            install_claude_mcp_server()
            print("registered Claude MCP server: slackgentic")
            print(
                "Start Claude with: "
                "claude --dangerously-load-development-channels server:slackgentic"
            )
            return 0
        return run_channel_server(args.db)
    if args.command == "team":
        return _team(args)
    if args.command == "service":
        return _service(args)
    if args.command == "index-once":
        from agent_harness.config import AppConfig
        from agent_harness.daemon import AgentDaemon

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
        from agent_harness.slack_app import run_slack_app

        if args.slack_command == "setup":
            from agent_harness.slack_setup import SlackSetupOptions, run_interactive_setup

            result = run_interactive_setup(
                SlackSetupOptions(
                    config_file=args.config_file,
                    open_browser=not args.no_browser,
                    timeout_seconds=args.timeout,
                    force=args.force,
                    bootstrap_tools=not args.no_bootstrap_tools,
                    instance=args.instance,
                )
            )
            if result != 0 or not args.serve:
                return result
        if args.slack_command == "update-manifest":
            from agent_harness.slack_setup import (
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
        if args.slack_command == "reset-state":
            return _reset_slack_state(config, yes=args.yes)
        if args.slack_command == "doctor":
            return _slack_doctor(config)
        if args.slack_command == "serve":
            return run_slack_app(config)
        if args.slack_command == "setup" and args.serve:
            return run_slack_app(config)
    raise AssertionError(args.command)


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
        )
        specs = [daemon_spec]
        if not args.no_codex_app_server:
            specs.append(
                build_codex_app_server_service_spec(
                    name=args.name,
                    executable=args.codex_binary,
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
        return 0 if all(status == 0 for status in statuses) else 1
    raise AssertionError(args.service_command)


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
        print(format_daily_usage(day, snapshots))
    return 0


def _persona(args: argparse.Namespace) -> int:
    persona = generate_persona(args.provider, args.session_id)
    if args.svg:
        print(avatar_svg(persona))
    else:
        print(json.dumps(_jsonable(persona), indent=2, sort_keys=True))
    return 0


def _team(args: argparse.Namespace) -> int:
    store = Store(args.db)
    try:
        store.init_schema()
        if args.team_command == "init":
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
            hired = hire_team_agents(
                all_agents,
                args.count,
                provider,
                start_sort_order=store.next_team_sort_order(),
                balance_agents=active_agents,
                randomize_identities=True,
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
        print(f"@{agent.handle} {agent.full_name} [{provider}] {agent.role}")
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
    print(
        "Run `slackgentic slack serve`, then use "
        f"`{config.slack.slash_command} setup` in Slack to initialize fresh state."
    )
    return 0


def _sqlite_state_paths(path: Path) -> list[Path]:
    return [
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        Path(f"{path}-journal"),
    ]


def _slack_doctor(config) -> int:
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
    print(f"config file {config.config_file}")
    print("delivery mode socket")
    print(f"slash command {config.slack.slash_command}")
    print(f"state db {config.state_db}")
    print(f"default cwd {config.commands.default_cwd}")
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
