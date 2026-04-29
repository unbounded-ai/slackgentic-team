# End-to-End Slack Test

Slackgentic uses Socket Mode only. The local daemon connects outbound to Slack
over WebSocket, so there is no public request URL, no local HTTPS callback, and
no browser certificate warning.

## 1. Install locally

```sh
python -m venv .venv && source .venv/bin/activate && pip install -e .
slackgentic slack doctor
```

The first doctor run will report missing Slack credentials. That is expected
until setup finishes.

The `codex` and `claude` binaries must be on `PATH`. Override them with env vars
only if needed:

```sh
export SLACKGENTIC_CODEX_BINARY=/path/to/codex
export SLACKGENTIC_CLAUDE_BINARY=/path/to/claude
```

## 2. Self-register the Slack app

```sh
slackgentic slack setup
```

The setup command:

- installs Slack's official CLI in `~/.slack/bin/slack` if needed;
- starts Slack CLI authorization if no workspace authorization exists yet;
- creates a Socket Mode Slack app from the bundled manifest;
- uses a teammate-specific app suffix and slash command, such as
  `/slackgentic-riley`, defaulting to the current OS user;
- opens Slack-hosted app pages only;
- watches the clipboard after you click Slack's Copy buttons for the bot token
  and app-level `connections:write` token;
- saves credentials to `~/.slackgentic-team/config.json` with mode `0600`.

There is no token paste step and no localhost HTTPS page.

If you do not want setup to install Slack CLI automatically, use:

```sh
slackgentic slack setup --no-bootstrap-tools
```

If the default suffix is not what you want, choose one explicitly:

```sh
slackgentic slack setup --instance riley
```

## 3. Install the local service

```sh
slackgentic slack doctor
slackgentic service install && slackgentic service status
```

This installs user-level services for the Slackgentic Slack daemon and Codex
app-server.

On macOS, `slackgentic slack doctor` also reports whether active sessions can
use `caffeinate` and whether the current system settings support network wake.
Network wake is advisory only; active Slackgentic work is kept awake, but a
closed-lid or deep-sleeping laptop may still need you to open it.

Agent posts use numbered cartoon PNGs from `docs/assets/avatars` by default.
When the repository is public, Slack fetches them from GitHub's raw asset URLs.
Set `SLACKGENTIC_AGENT_AVATAR_BASE_URL` to a different public HTTPS directory,
or set it to `off` to disable custom agent icons.

`slackgentic slack setup` attempts to register Slackgentic's Claude channel MCP
server automatically when `claude` is on `PATH`. If setup prints a warning,
run `slackgentic claude-channel --install` after Claude is available, then
restart any already-open Claude sessions before using
`claude --dangerously-load-development-channels server:slackgentic`. The channel
relays Slack replies and native Claude tool approval prompts. The installer also
adds Slackgentic's own Slack request tools to Claude's allowlist so those tools
do not need a separate local approval. No extra MCP flag is needed after
registration unless you use `--strict-mcp-config` or a custom MCP config that
excludes user-level servers.

## 4. Set up from Slack

In any channel or DM where the app is available:

```text
/slackgentic-<you> setup
```

Choose the channel name, public/private visibility, Codex agent count, and
Claude agent count, and repos root. The default repos root is two directories
above the service working directory when possible. The app creates the channel,
posts a short usage note first, then posts the roster.

## 5. Start real work

In the agent channel:

```text
Inspect the repo and summarize the test command
```

The harness assigns an idle agent, replies in the user's message thread,
launches that agent's mapped runtime, and streams terminal output into the
thread. Later thread replies continue with the same Slack context.

Specific agent:

```text
@riley do update the README with install steps
```

Dangerous mode for one managed task:

```text
#dangerous-mode @riley repair the local service installer
```

Slackgentic strips the tag from the prompt, launches that task with Codex
`--dangerously-bypass-approvals-and-sandbox` or Claude
`--dangerously-skip-permissions`, and marks active dangerous-mode tasks on the
roster.

PR review:

```text
Somebody review @riley's PR https://github.com/org/repo/pull/42
```

The reviewer picker prefers an idle agent on the opposite provider from the
author agent.

## 6. Manage the team

Use roster buttons or commands. In the agent channel, command bodies also work
as plain messages, such as `hire 2 claude agents` or `show roster`.

```text
/slackgentic-<you> hire 2 claude agents
/slackgentic-<you> fire @riley
/slackgentic-<you> fire everyone
/slackgentic-<you> show roster
/slackgentic-<you> show repo root
/slackgentic-<you> repo root ~/code
/slackgentic-<you> usage
/slackgentic-<you> status
hire 2 claude agents
show roster
status
```

Wipe local runtime state and start fresh:

```sh
slackgentic slack reset-state --yes
```

This keeps Slack credentials in `~/.slackgentic-team/config.json`.

Task thread buttons:

- `Finish and free up this agent` terminates any attached managed process and
  releases the agent for new work.

Usage:

```text
/slackgentic-<you> usage
/slackgentic-<you> status
status
```

The app keeps one usage message per day and edits that same message when you run
the command again.

## Current Limits

- Slack can see Slack messages; no extra relay is in the message path.
- Slack does not expose a public API for minting the required app-level `xapp-`
  token, so setup still requires clicking Slack's Copy button once for that
  token.
- Managed child processes cannot be reattached after daemon restart. Old
  threads remain visible and follow-up replies resume the persisted task/session
  state, but Slackgentic relaunches the provider process.
- PR review execution passes the PR URL to Codex/Claude; it does not yet fetch
  and cache GitHub diffs itself.
- Socket Mode apps are intended for workspace/internal use, not Slack
  Marketplace distribution.

## Development Debugging

Run the Slack daemon in the foreground when debugging service startup locally:

```sh
slackgentic slack serve
```
