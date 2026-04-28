# End-to-End Slack Test

Slackgentic uses Socket Mode only. The local daemon connects outbound to Slack
over WebSocket, so there is no public request URL, no local HTTPS callback, and
no browser certificate warning.

## 1. Install locally

```sh
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/slackgentic slack doctor
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
.venv/bin/slackgentic slack setup
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
.venv/bin/slackgentic slack setup --no-bootstrap-tools
```

If the default suffix is not what you want, choose one explicitly:

```sh
.venv/bin/slackgentic slack setup --instance riley
```

## 3. Install the local service

```sh
.venv/bin/slackgentic slack doctor
.venv/bin/slackgentic service install
.venv/bin/slackgentic service status
```

This installs user-level services for the Slackgentic Slack daemon and Codex
app-server. For foreground debugging instead of a persistent service, run:

```sh
.venv/bin/slackgentic slack serve
```

Agent posts use numbered cartoon PNGs from `docs/assets/avatars` by default.
When the repository is public, Slack fetches them from GitHub's raw asset URLs.
Set `SLACKGENTIC_AGENT_AVATAR_BASE_URL` to a different public HTTPS directory,
or set it to `off` to disable custom agent icons.

## 4. Set up from Slack

In any channel or DM where the app is available:

```text
/slackgentic-<you> setup
```

Choose the channel name, public/private visibility, Codex agent count, and
Claude agent count. The app creates the channel, posts the roster, and starts an
introduction thread.

## 5. Start real work

In the agent channel:

```text
Somebody do inspect the repo and summarize the test command
```

The harness assigns an idle agent, creates a task thread, launches that agent's
mapped runtime, and streams terminal output into the thread. Replies in the
thread are sent back into the managed process.

Specific agent:

```text
@riley do update the README with install steps
```

PR review:

```text
Somebody review @riley's PR https://github.com/org/repo/pull/42
```

The reviewer picker prefers an idle agent on the opposite provider from the
author agent.

## 6. Manage the team

Use roster buttons or commands:

```text
/slackgentic-<you> hire 2 claude agents
/slackgentic-<you> fire @riley
/slackgentic-<you> show roster
/slackgentic-<you> usage
/slackgentic-<you> status
```

Wipe local runtime state and start fresh:

```sh
.venv/bin/slackgentic slack reset-state --yes
```

This keeps Slack credentials in `~/.slackgentic-team/config.json`.

Task thread buttons:

- `Mark Done` terminates any attached managed process and marks the task done.
- `Pause` terminates the process and puts the task back in queued state.
- `Cancel` terminates the process and marks the task cancelled.

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
- Managed processes are in-memory. If `slackgentic slack serve` restarts, old
  threads remain visible but cannot receive input until task resumption is
  implemented.
- PR review execution passes the PR URL to Codex/Claude; it does not yet fetch
  and cache GitHub diffs itself.
- Socket Mode apps are intended for workspace/internal use, not Slack
  Marketplace distribution.
