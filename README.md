# slackgentic-team

Slackgentic is a local Slack control plane for Codex and Claude Code. It makes
local coding-agent sessions visible in Slack, routes work to lightweight agent
personas, mirrors external terminal sessions, and keeps all runtime state on the
developer's machine.

## Highlights

- Socket Mode Slack app: no public callback URL, tunnel, or relay server.
- Local-first state in SQLite under `~/.slackgentic-team`.
- Per-teammate Slack app instances such as `/slackgentic-riley`.
- Persistent roster with human-style handles, personas, provider preferences,
  and hire/fire controls.
- Work routing from natural Slack messages such as `Somebody do update docs` or
  `@riley review this PR`.
- Managed Codex and Claude task processes with streamed Slack replies.
- Mirrored external Codex and Claude sessions from local transcript files.
- Live external-session delivery through Codex app-server or the Slackgentic
  Claude Code channel.
- Provider-neutral Slack buttons for agent input and approval requests.

## How It Works

Slackgentic runs as a local daemon. The daemon connects outbound to Slack over
Socket Mode, launches or discovers local Codex and Claude processes, stores
session metadata in SQLite, and posts into Slack with per-message persona
identity.

Slack only talks to the local daemon. Agent transcripts and credentials are not
sent through a hosted Slackgentic service.

## Requirements

- Python 3.11 or newer.
- Slack workspace permissions to create and install a Socket Mode app.
- `codex` and/or `claude` on `PATH` for the providers you want to use.
- macOS or Linux for optional user-service installation.

## Installation

```sh
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/slackgentic slack doctor
```

The first `doctor` run should report missing Slack credentials. Setup writes
those credentials after the Slack app is created.

Override provider binaries only when needed:

```sh
export SLACKGENTIC_CODEX_BINARY=/path/to/codex
export SLACKGENTIC_CLAUDE_BINARY=/path/to/claude
```

## Slack Setup

Create and configure a local Socket Mode app:

```sh
.venv/bin/slackgentic slack setup
```

Setup:

- installs Slack's official CLI into `~/.slack/bin/slack` when missing;
- reuses Slack CLI workspace authorization to create the app;
- creates a Socket Mode manifest with one teammate-specific slash command;
- opens Slack-hosted pages for the required token copy buttons;
- stores credentials in `~/.slackgentic-team/config.json` with mode `0600`.

Use an explicit teammate suffix when the default OS username is not right:

```sh
.venv/bin/slackgentic slack setup --instance riley
```

Run the daemon:

```sh
.venv/bin/slackgentic slack serve
```

Or combine setup and serving:

```sh
.venv/bin/slackgentic slack setup --serve
```

## Slack Usage

Open the setup modal from Slack:

```text
/slackgentic-<you> setup
```

The modal creates or selects the agent channel, initializes the Codex/Claude
roster, posts the roster, and starts the introduction thread.

Common controls:

- roster buttons: `Hire Auto`, `Hire Codex`, `Hire Claude`, `Fire`;
- slash command: `/slackgentic-<you> hire 2 claude agents`;
- slash command: `/slackgentic-<you> fire @riley`;
- slash command: `/slackgentic-<you> usage`;
- slash command: `/slackgentic-<you> status`;
- channel text: `hire 3 new agents`;
- channel text: `fire @riley`;
- channel text: `show roster`;
- channel text: `usage` or `status`.

Start work from the agent channel:

```text
Somebody do inspect the repo and summarize the test command
@riley do update the README with install steps
Somebody review @riley's PR https://github.com/org/repo/pull/42
```

When work is accepted, Slackgentic creates a task thread, launches the selected
provider, streams visible output into the thread, and forwards thread replies
back into the managed process while it is running.

Task buttons:

- `Mark Done` stops any attached process and marks the task done.
- `Pause` stops the process and puts the task back in queued state.
- `Cancel` stops the process and marks the task cancelled.

## External Sessions

Slackgentic discovers external Codex and Claude sessions from local transcript
files and mirrors visible output into Slack threads.

### Codex App Server

The Slack daemon starts a local Codex app-server by default:

```text
ws://127.0.0.1:47684
```

Start Codex against that server when you want Slack replies to appear in the
original terminal:

```sh
codex --remote ws://127.0.0.1:47684
```

If the app-server is unavailable, Slackgentic falls back to `codex exec resume`.
Codex app-server requests for user input, command approval, file-change
approval, and permission approval are shown as Slack buttons in the mirrored
thread.

### Claude Code Channel

Install the Slackgentic Claude channel and start Claude with the channel flag:

```sh
slackgentic claude-channel --install
claude --dangerously-load-development-channels server:slackgentic
```

The channel delivers Slack thread replies into the live Claude session through
Claude Code's channel notification path. It also exposes `request_user_input`
and `request_approval` MCP tools so Claude can ask the Slack thread for a choice
or approval and receive the button response back in-session.

Without the channel flag, Slackgentic falls back to Claude's supported resume
command.

### Exit Handling

Slackgentic does not provide a `command <provider-command>` shim. Replies in a
mirrored session thread are forwarded as text. The only special case is an exact
`/exit` reply, which is sent through the live channel when possible and then
falls back to terminating the matched live process.

## Service Install

Install the daemon as a user service:

```sh
slackgentic service install
```

On macOS this writes a launchd agent at
`~/Library/LaunchAgents/com.slackgentic-team.daemon.plist`. On Linux this writes
a systemd user unit at `~/.config/systemd/user/slackgentic-team.service`.

Service commands:

```sh
slackgentic service status
slackgentic service print
slackgentic service uninstall
```

## Configuration

The interactive setup stores config in `~/.slackgentic-team/config.json`.
Environment variables override stored values:

| Variable | Purpose |
| --- | --- |
| `SLACK_BOT_TOKEN` | Slack bot token for Web API calls. |
| `SLACK_APP_TOKEN` | Slack app-level Socket Mode token. |
| `SLACK_TEAM_ID` | Slack workspace team id. |
| `SLACK_CHANNEL_ID` | Default agent channel id. |
| `SLACKGENTIC_CONFIG_FILE` | Alternate stored config path. |
| `SLACKGENTIC_STATE_DB` | Alternate SQLite state path. |
| `SLACKGENTIC_HOME` | Alternate home directory for provider discovery. |
| `SLACKGENTIC_CODEX_BINARY` | Codex executable path. |
| `SLACKGENTIC_CLAUDE_BINARY` | Claude executable path. |
| `SLACKGENTIC_CODEX_APP_SERVER_URL` | Codex app-server URL; use `off` to disable. |
| `SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART` | Whether the daemon starts Codex app-server. |
| `SLACKGENTIC_CODEX_AGENTS` | Default initial Codex roster count. |
| `SLACKGENTIC_CLAUDE_AGENTS` | Default initial Claude roster count. |

## Development

Install development dependencies:

```sh
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Run the local checks:

```sh
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m compileall src tests
python -m ruff check src tests
python -m ruff format --check src tests
```

Useful local commands:

```sh
PYTHONPATH=src python -m agent_harness scan --provider all --json
PYTHONPATH=src python -m agent_harness usage --date today
PYTHONPATH=src python -m agent_harness index-once --db /tmp/slackgentic.sqlite
PYTHONPATH=src python -m agent_harness team init --db /tmp/slackgentic.sqlite
PYTHONPATH=src python -m agent_harness team assign \
  "Somebody do update the README" \
  --db /tmp/slackgentic.sqlite \
  --channel C123
```

More Slack-specific validation steps are in [docs/e2e-slack.md](docs/e2e-slack.md).
The high-level design is documented in [docs/architecture.md](docs/architecture.md).

## Current Limits

- Socket Mode apps are intended for internal workspace use, not Slack
  Marketplace distribution.
- Setup still requires clicking Slack's token copy buttons because Slack does
  not expose a public API for minting app-level `xapp-` tokens.
- Managed process state is in memory. If the daemon restarts, old task threads
  remain visible but cannot receive managed-process input.
- PR review execution passes the PR URL to Codex/Claude; it does not fetch and
  cache GitHub diffs itself.

## License

MIT. See [LICENSE](LICENSE).
