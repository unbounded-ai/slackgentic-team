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

Recommended — installs `slackgentic` on `PATH` globally via
[uv](https://docs.astral.sh/uv/) so the command works in any shell:

```sh
uv tool install --with pip git+https://github.com/unbounded-ai/slackgentic-team.git
slackgentic slack setup
slackgentic service install && slackgentic service status
```

`--with pip` lets the in-app updater apply new releases in place.

Alternatively, install from a source checkout — needed if you want to track
`main` or hack on the code:

```sh
git clone https://github.com/unbounded-ai/slackgentic-team.git && cd slackgentic-team
python3.13 -m venv .venv && source .venv/bin/activate && pip install -e . && slackgentic slack setup
slackgentic service install && slackgentic service status
```

Then the CLI will ask you to message yourself in Slack something like:

```text
/slackgentic-<you> setup
```

Proceed with the rest of the CLI instructions.

Requirements: Python 3.11+, permission to create a Slack app, and `codex` or
`claude` on `PATH`. On macOS, install managed Slackgentic services from Python
3.11-3.13 so Homebrew `python3` drift cannot put long-running agents behind a
new TCC prompt identity.

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
#dangerous-mode @riley repair the local service installer
```

Slackgentic reacts right away, replies in your thread, launches the selected
provider, and streams the visible result back. Thread replies continue from the
same Slack context.

The roster also has `Assign Work` and `Schedule Work` buttons. They open a
Slack form for a task, work/review kind, timing, dangerous mode, repeat cadence,
and an optional dependency on a currently busy agent finishing its active task or
external session.

In the main channel, write anything to hand it to an available agent. Use
`@agentname ...` when you want a specific agent. Inside a task thread, reply
with `somebody ...` to bring in another agent for a subtask; the original agent
then picks the thread back up with that new context.

By default tasks launch in **safe-auto** permission mode: Codex runs with
`--sandbox workspace-write` and Claude runs with `--permission-mode acceptEdits`
plus an allowlist for read-only inspection commands such as `git status`,
`git log`, `git diff`, `gh pr view`, `rg`, and `ls`. That keeps reviews and
investigations unblocked without auto-approving anything destructive. Safe-auto
Codex and Claude tasks also receive the configured repo root as an additional
directory so Git worktrees under that root can update their shared metadata
during normal edits and commits.

Add `#dangerous-mode` to a task when you want that managed agent process to run
with Codex `--dangerously-bypass-approvals-and-sandbox` or Claude
`--dangerously-skip-permissions`. Slackgentic strips the tag from the task
prompt and marks active dangerous-mode tasks on the roster.

### Thread = Context Boundary

A Slack thread is the continuity boundary. If the same agent continues in that
thread, Slackgentic resumes the same provider session whenever Codex or Claude
has exposed a session id. The thread transcript is also included as backup
context, so handoffs and follow-ups stay grounded in what happened there.

Sessions started outside Slack keep their original Codex or Claude Code session
too. Replies from Slack are sent into that live session when possible, or resumed
against that exact session id.

Setup asks for a repos root. Agents launch there by default, and phrases like
`in sample-app ...` can select a sibling repo under that root.

Agents can also talk to each other. When one is unsure, it can ask another agent
for a review in the same thread; the router prefers a different provider when a
cross-model review is useful.

Agents can schedule one-off follow-ups in the same thread. Slackgentic stores
those timers in the daemon state database, so the wakeup is not tied to a
provider process or an open terminal session.

### PM Initiatives

For larger projects, hand the team a project description and let a
PM-flavored agent break it into a DAG of subtasks. There are two entry
points:

```text
pm: ship the new logging stack
pm plan the migration to FastAPI
@alice plan the storage refactor
```

The first two use the `pm:` / `pm <verb>` shortcut and route to whichever
PM-kind agent is hired (or fall back to a worker if none are). The third
addresses a specific PM-kind agent by handle — tagging a PM agent always
activates PM mode, even without the `pm` keyword. Hire a PM with
`team hire --kind pm` (use `--provider claude` or `--provider codex` to
pick the model).

A PM-kind agent owns its initiative end-to-end: it can ask one or two
clarifying questions in the thread before committing to a plan, answers
status questions in the same thread (every reply receives a fresh
`[PM HARNESS: initiative state]` snapshot of the subtask DAG), and stays
assigned for the life of the initiative. PM-kind agents are reserved
for PM duties — the worker router never picks them up for regular
subtasks.

The PM emits a structured plan (title, summary, up to 20 subtasks with
sibling dependencies). Subtasks may optionally include a `co_designers`
list of two or more handles, in which case the PM fans the subtask out
into one draft per co-designer running in parallel and follows up with
an automatic synthesis stage. The plan is parked for approval — the
approval message includes a rough cost estimate (subtask count, critical
path depth, dangerous-mode count, and a wall-clock band) so a user can
trim before approving. No subtask runs until the user clicks *Start
executing* on the plan message (or *Cancel* to drop it). Once approved,
each subtask becomes a deferred-work row, dispatched the moment its
dependencies satisfy. A watchdog polls active initiatives and surfaces
blockers in the initiative thread — stalled approvals, stuck tasks, or
plan failures — but never starts or stops work on its own. When every
subtask finishes, the watchdog posts a recap and closes the initiative.

At any time during a live initiative, drop `pm status` (or `pm plan`,
`status`, `dag`) into the initiative thread to get an instant DAG render
without going through the PM agent. The reply lists each subtask with
its current deferred-work status, the worker that owns it, and its
dependencies. The PM agent also receives the same snapshot as a
`[PM HARNESS: initiative state]` prefix on every follow-up.

Need to course-correct after a subtask fails? Reply with
`pm replan: <new context>` in the initiative thread (PM-owned initiatives
only). The PM agent receives the failure surface and the remaining
subtask state, re-plans from there, and parks a fresh approval gate.

Marking the PM thread done cancels every non-terminal subtask in the
initiative.

Users can schedule future work directly from Slack too:

```text
schedule @riley to check CI tomorrow at 9am PT
schedule somebody review the nightly report every day at 5pm ET
schedule @riley inspect the deploy every Monday at 10:30am America/New_York
schedule somebody check the deploy every 2 hours
schedule @riley check the patio lights during tomorrow's sunset time in Waco
```

Slackgentic asks an agent to resolve the human-language schedule into a validated
hidden control message, so richer requests can use the model instead of a local
regex parser. The validated schedule is stored in SQLite and claimed by the
daemon when due. One-off items run once; recurring items are rescheduled after
each due occurrence.

Task threads have two controls: `Finish and free up this agent` ends the task
and frees the agent; replying in the thread sends the follow-up into the active
task, interrupting the current managed run first when live delivery is not
available. Replying `stop` sends only the interrupt without closing the task
thread or replaying fallback-queued replies. Reacting to an agent-authored
message in an active task thread is delivered back to that agent as lightweight
feedback.

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
slackgentic codex-mcp --install
codex --remote ws://127.0.0.1:47684
```

Claude:

```sh
slackgentic claude-channel --install
claude --dangerously-load-development-channels server:slackgentic
```

`slackgentic slack setup` tries to register the Codex MCP server and Claude
channel automatically when those CLIs are on `PATH`. Run
`slackgentic codex-mcp --install` or `slackgentic claude-channel --install`
manually if setup could not reach a CLI, and restart any already-open sessions
after installing. The Claude installer adds Slackgentic's Slack request and
thread-link tools to Claude's allowlist so they do not require their own
approval. No extra MCP flag is needed after registration unless you start Claude
with `--strict-mcp-config` or another custom MCP config that excludes user-level
servers.

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

The Codex app-server service is supervised so `codex --remote` stays compatible
across Codex upgrades. With the default `codex` command, Slackgentic checks all
Codex executables visible on the service `PATH`, starts the newest version, and
restarts the app-server when that executable or its upgrade symlink changes. An
explicit `SLACKGENTIC_CODEX_BINARY` pins selection to that path, while still
detecting upgrades installed at the same path.

On macOS, active managed sessions and sessions started outside Slack keep the
machine awake with the built-in `caffeinate` command until the work is done.

`slackgentic slack doctor` also reports macOS network-wake readiness: Wake for
network access, TCP keepalive, Wi-Fi wake support, and scheduled wakes. Network
wake can help with Wake-on-LAN/Bonjour-style wakeups, but it is best-effort and
does not make Slack a reliable way to wake a closed-lid or deep-sleeping laptop.

## Updating

You do not need to `git pull` or rerun `pip install` to update Slackgentic.
The running daemon checks GitHub Releases of `unbounded-ai/slackgentic-team`
every five minutes. When a newer published version appears it posts one
prompt in the agent channel:

> *Slackgentic update available*
> Current: `0.1.0`  Latest: `0.1.1`
> [Upgrade now] [Skip]

Clicking *Upgrade now* fetches and checks out the release tag in the local
source checkout, reinstalls the editable package, restarts the service, and
edits the same Slack message in place with a green checkmark once the new
daemon is back up. Non-source installs upgrade from the release tarball.

Dirty source checkouts are left untouched until you commit, stash, or move
the local changes — Slackgentic refuses to overwrite work in progress.

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
| `SLACKGENTIC_EXTERNAL_SESSION_ALLOWED_CWD_PREFIXES` | Comma- or path-list of cwd prefixes to mirror for sessions started outside Slack, for example `/workspace/repos`. When set, sessions outside these prefixes are ignored. |
| `SLACKGENTIC_EXTERNAL_SESSION_IGNORED_CWDS` | Comma- or path-list of cwd path segments or patterns to ignore for sessions started outside Slack. |
| `SLACKGENTIC_EXTERNAL_SESSION_MIRROR_POLL_SECONDS` | Seconds between scans for sessions started outside Slack, default `15`. |
| `SLACKGENTIC_AGENT_AVATAR_BASE_URL` | Public HTTPS avatar directory, or `off`. |
| `SLACKGENTIC_ALLOW_MACOS_TCC_PROTECTED_PATHS` | Allow managed tasks to start in macOS privacy-protected locations after you have granted OS access. |
| `SLACKGENTIC_UPDATE_CHECK_ENABLED` | Enable or disable release checks, default `true`. |
| `SLACKGENTIC_UPDATE_CHECK_INTERVAL_SECONDS` | Seconds between release checks, default `300`. |
| `SLACKGENTIC_UPDATE_REPOSITORY` | GitHub repository used for update checks, default `unbounded-ai/slackgentic-team`. |

## Development

```sh
python3.13 -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'
```

Checks:

```sh
PYTHONPATH=src python -m pytest -n auto tests
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

## Publishing Releases

1. Bump the version in `pyproject.toml` and `src/agent_harness/__init__.py`.
2. Merge the change to `main`.

The release workflow runs on every push to `main`. When it sees a version that
does not yet have a matching `vX.Y.Z` tag, it creates the tag and a GitHub
Release with auto-generated notes. Running Slackgentic services consume that
release metadata during their next update check.

## Current Limits

- Slack still requires clicking token copy buttons for the app-level `xapp-`
  token.
- Managed child processes cannot be reattached after daemon restart. Follow-up
  replies resume the persisted task/session state, while Slackgentic relaunches
  the provider process.
- PR review execution passes the PR URL to Codex/Claude; it does not cache
  GitHub diffs itself.

## Coming Next

- Stabilization.
- Link threads and wait for one thread to finish before automatically starting
  a follow-up task.
- More features.

## License

MIT. See [LICENSE](LICENSE).
