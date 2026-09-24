# Slackgentic CLI

The `slackgentic` command is installed on `PATH`. Commands that change
Slackgentic's local state write under `~/.slackgentic-team/`, so a sandboxed
agent may need escalation to run them.

## Loops

```sh
slackgentic loop create "<task and schedule>" [--every-run] [--provider claude|codex] [--public]
slackgentic loop create "<task and schedule>" --no-wait   # queue and return immediately
slackgentic loop request-status <request-id>
slackgentic loop list [--all] [--json]
```

`loop create` queues a request for the running service, which posts it in the
agent channel. The owner must approve the preview in Slack. Agent requests
always start read-only and quiet; `--every-run` posts every run's card instead.

## Service

```sh
slackgentic service status      # installed and running?
slackgentic service start
slackgentic service restart
slackgentic service install     # install and start the background services
slackgentic service uninstall
slackgentic slack serve         # run the daemon in the foreground for debugging
```

If queued loop requests, schedules, or timers are not firing, check
`slackgentic service status` first.

## Setup and diagnostics

```sh
slackgentic slack setup                 # create the Slack app and save credentials
slackgentic slack doctor                # check the local Slack configuration
slackgentic slack update-manifest       # refresh app scopes after an upgrade
slackgentic claude-channel --install    # connect Claude Code terminal sessions to Slack
slackgentic codex-mcp --install         # connect Codex terminal sessions to Slack
slackgentic skills install              # (re)install these skills for Claude Code and Codex
```

Configuration lives in `~/.slackgentic-team/config.json`; environment
variables such as `SLACK_BOT_TOKEN` override it. On macOS the tokens may
instead be in the login keychain (`slackgentic slack tokens` shows where).
Treat that file, the keychain items, and the Slack tokens as secrets: never
print, copy, or commit them, and never read them out of the keychain.

## Team

```sh
slackgentic team list --db ~/.slackgentic-team/state.sqlite
slackgentic team hire 1 --provider claude --db ~/.slackgentic-team/state.sqlite
slackgentic team hire --kind pm --db ~/.slackgentic-team/state.sqlite
```

Prefer the Slack `hire` and `fire` messages when the service is running, so the
pinned roster updates immediately.

## Updates and reset

The service checks GitHub releases every five minutes and posts an
**Upgrade now** prompt in the agent channel; no manual `git pull` is needed.

```sh
slackgentic slack close-channel --yes   # archive the agent channel, clear the pointer
slackgentic slack reset-state --yes     # reset local runtime state, keep credentials
slackgentic slack setup --force         # recreate Slack credentials
```

The reset commands are destructive; run them only when the user asks.
