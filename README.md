# slackgentic-team

Local Codex and Claude agents, coordinated from Slack.

Slackgentic turns one Slack channel into a control room for local coding agents:
start work with a normal message, watch agent output stream back into a thread,
track sessions started outside Slack, and keep all credentials and transcripts
strictly local.

Use agents exactly the way you already do. They still run on your laptop, with
the same tools, deployment access, repo access, and approval levels. Slack just
becomes the place where you can step away, touch some grass, and reply when an
agent needs you.

## Why It Exists

- One Slack channel becomes the surface for local agent work.
- Everything stays on your machine.
- The agent channel is private by default, so only you see what your agents are
  up to.
- Codex and Claude work under one roster.
- No API-key billing detour: it drives the Codex and Claude Code you already use.
- Cross-model reviews are easy: ask for one directly, or let an agent pull in
  another agent when it wants a second opinion.
- Thread subtasks are natural: reply with `somebody ...`, another agent picks
  up that slice, then the original agent resumes with the added context.
- Sessions started outside Slack still get tracked in Slack threads.
- Hire/fire keeps parallelism intentional.

## Quick Install

```sh
git clone https://github.com/unbounded-ai/slackgentic-team.git && cd slackgentic-team
python3 -m venv .venv && source .venv/bin/activate && pip install -e . && slackgentic slack setup
slackgentic service install && slackgentic service status
```

Then open Slack and run:

```text
/slackgentic-<you> setup
```

Requirements: Python 3.11+, permission to create a Slack app, and `codex` or
`claude` on `PATH`.

## Setup Notes

`slackgentic slack setup` creates a teammate-specific Slack app, for example
`/slackgentic-riley`. It uses Slack's official CLI for app creation, opens
Slack-hosted token pages, then saves credentials in
`~/.slackgentic-team/config.json` with mode `0600`.

Choose a suffix explicitly when needed:

```sh
slackgentic slack setup --instance riley
```

## Slack Usage

The setup modal creates a private agent channel by default, posts a short
command guide, then starts with one Codex agent and one Claude agent.
The roster stays updated in-place and pinned when Slack grants the bundled
pinning scope.

Run commands with the slash command:

```text
/slackgentic-<you> <command>
```

Inside the agent channel, you can also type the command body directly, for
example `status`, `show roster`, or `hire 3 agents`.

Useful commands:

```text
/slackgentic-<you> status
/slackgentic-<you> show roster
/slackgentic-<you> show repo root
/slackgentic-<you> repo root ~/code
/slackgentic-<you> hire 3 agents
/slackgentic-<you> hire 1 claude agent
/slackgentic-<you> fire @riley
/slackgentic-<you> fire everyone
status
show roster
hire 3 agents
```

Start work by typing in the agent channel:

```text
Inspect the repo and summarize the test command
@riley update the README with install steps
Somebody review @riley's PR https://github.com/org/repo/pull/42
```

Slackgentic reacts right away, replies in your thread, launches the selected
provider, and streams the visible result back. Thread replies continue from the
same Slack context.

In the main channel, write anything to hand it to an available agent. Use
`@agentname ...` when you want a specific agent. Inside a task thread, reply
with `somebody ...` to bring in another agent for a subtask; the original agent
then picks the thread back up with that new context.

### Thread = Context Boundary

A Slack thread is the continuity boundary. If the same agent continues in that
thread, Slackgentic resumes the same provider session whenever Codex or Claude
has exposed a session id. The thread transcript is also included as backup
context, so handoffs and follow-ups stay grounded in what happened there.

Sessions started outside Slack keep their original Codex or Claude Code session
too. Replies from Slack are sent into that live session when possible, or resumed
against that exact session id.

Setup asks for a repos root. Agents launch there by default, and phrases like
`in talos ...` can select a sibling repo under that root.

Agents can also talk to each other. When one is unsure, it can ask another agent
for a review in the same thread; the router prefers a different provider when a
cross-model review is useful.

Task threads have one control: `Finish and free up this agent`.

## Why Hire/Fire Agents?

Different levels of parallelism work for different people. Hire/fire lets you
operate at the level that is optimal for you while staying disciplined about not
spreading yourself too thin.

If you do not know your level yet, start with the default team of 2. Slowly hire
agents until it feels like too many, then back down. That is your sweet spot.

The hard limit is 500 active agents. You definitely do not need that many
agents.

## Sessions Started Outside Slack

Codex:

```sh
codex --remote ws://127.0.0.1:47684
```

Claude:

```sh
slackgentic claude-channel --install
claude --dangerously-load-development-channels server:slackgentic
```

`slackgentic slack setup` tries to register the Claude channel automatically
when `claude` is on `PATH`. Run `slackgentic claude-channel --install` manually
if setup could not reach Claude, and restart any already-open Claude sessions
after installing. The launch flag alone is not enough; Claude must also load the
Slackgentic MCP server so Slack replies and approval requests can reach the live
terminal, including native Claude tool approval prompts. No extra MCP flag is
needed after registration unless you start Claude with `--strict-mcp-config` or
another custom MCP config that excludes user-level servers.

Those commands create tracked Slack threads for sessions you started outside
Slack. If all matching team seats are occupied, Slackgentic posts a
provider-specific hire button and waits without advancing the transcript cursor,
so visible output is backfilled after you add capacity.

If you revive an ended session from its Slack thread, Slackgentic uses a free
matching Codex or Claude seat to resume that exact session. If no matching seat
is free, hire or free that provider; `somebody ...` can still start a new
session in the same thread using the Slack context.

## Service

Install the background services:

```sh
slackgentic service install
```

Common service commands:

```sh
slackgentic service start
slackgentic service restart
slackgentic service status
slackgentic service uninstall
```

On macOS, active managed sessions and sessions started outside Slack keep the
machine awake with the built-in `caffeinate` command until the work is done.

`slackgentic slack doctor` also reports macOS network-wake readiness: Wake for
network access, TCP keepalive, Wi-Fi wake support, and scheduled wakes. Network
wake can help with Wake-on-LAN/Bonjour-style wakeups, but it is best-effort and
does not make Slack a reliable way to wake a closed-lid or deep-sleeping laptop.

## Reset

Archive the current Slack channel and clear the local pointer:

```sh
slackgentic slack close-channel --yes
```

Reset local runtime state while keeping Slack app credentials:

```sh
slackgentic slack reset-state --yes
```

Recreate Slack credentials only when needed:

```sh
slackgentic slack setup --force
```

## Configuration

The config file lives at `~/.slackgentic-team/config.json`. Environment
variables override stored values.

| Variable | Purpose |
| --- | --- |
| `SLACK_BOT_TOKEN` | Slack bot Web API token. |
| `SLACK_APP_TOKEN` | Slack Socket Mode app token. |
| `SLACK_TEAM_ID` | Slack workspace team id. |
| `SLACK_CHANNEL_ID` | Default agent channel id. |
| `SLACKGENTIC_CONFIG_FILE` | Alternate config path. |
| `SLACKGENTIC_STATE_DB` | Alternate SQLite state path. |
| `SLACKGENTIC_HOME` | Alternate home directory for transcript discovery. |
| `SLACKGENTIC_CODEX_BINARY` | Codex executable path. |
| `SLACKGENTIC_CLAUDE_BINARY` | Claude executable path. |
| `SLACKGENTIC_CODEX_APP_SERVER_URL` | Codex app-server URL. |
| `SLACKGENTIC_CODEX_AGENTS` | Default initial Codex count. |
| `SLACKGENTIC_CLAUDE_AGENTS` | Default initial Claude count. |
| `SLACKGENTIC_AGENT_AVATAR_BASE_URL` | Public HTTPS avatar directory, or `off`. |

## Development

```sh
python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'
```

Checks:

```sh
python -m unittest discover -s tests
python -m compileall src tests
ruff check src tests
ruff format --check src tests
```

Foreground daemon debugging:

```sh
slackgentic slack serve
```

More validation steps are in [docs/e2e-slack.md](docs/e2e-slack.md). The design
notes are in [docs/architecture.md](docs/architecture.md).

## Current Limits

- Slack still requires clicking token copy buttons for the app-level `xapp-`
  token.
- Managed process state is in memory. After daemon restart, thread follow-ups
  start fresh managed tasks with the same Slack context.
- PR review execution passes the PR URL to Codex/Claude; it does not cache
  GitHub diffs itself.

## Coming Next

- Stabilization.
- Link threads and wait for one thread to finish before automatically starting
  a follow-up task.
- Scheduled tasks for recurring or one-off future work, such as nightly repo
  checks, delayed follow-ups, and periodic review sweeps.
- More features.

## License

MIT. See [LICENSE](LICENSE).
