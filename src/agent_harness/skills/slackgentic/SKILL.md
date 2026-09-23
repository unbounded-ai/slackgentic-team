---
name: slackgentic
description: How to use Slackgentic, which runs local Codex and Claude Code agents from a Slack channel. Use when the user mentions Slackgentic or its Slack agent channel, or asks you to do something through it — schedule future or recurring work, set a follow-up or reminder in a Slack thread, hand work to another agent or ask one for a review, hire or fire agents, start a PM initiative, check status or quotas, connect a terminal session to Slack, or install, update, or troubleshoot the Slackgentic service. For recurring loops with their own channel, also use the slackgentic-loops skill.
---

# Slackgentic

Slackgentic turns one private Slack channel (the **agent channel**) into a
control room for the user's local agents. A roster of named Codex and Claude
agents (for example `@riley`) takes work from Slack messages, runs on this
machine with the user's normal tools and credentials, and streams results back
into Slack threads. A Slack thread is the unit of context: follow-ups in the
same thread continue the same agent session.

## First, work out where you are running

How you drive Slackgentic depends on your context:

1. **Managed Slack task.** Your prompt says you are working from Slack and
   describes hidden `SLACKGENTIC: ...` lines. Slackgentic launched you for a
   Slack thread. Your visible replies go to that thread, and you can use the
   hidden control lines in [references/task-signals.md](references/task-signals.md)
   (follow-up timers, scheduling, reactions, roster status, closing the thread).
2. **Terminal session connected to Slack.** A Claude Code session started with
   the Slackgentic channel, or a Codex session started with
   `codex --remote ws://127.0.0.1:47684`. Slackgentic mirrors your visible
   output to a Slack thread and forwards Slack replies to you. Hidden control
   lines do **nothing** here; use the MCP tools and CLI instead.
3. **Anywhere else.** Use the `slackgentic` CLI and any Slackgentic MCP tools
   you have. For actions only Slack can do, give the user the exact text to
   send in the agent channel.

In every context, never write to Slackgentic's SQLite database or config
directly, and never post to Slack with raw tokens.

## Tools available to agents

Slackgentic MCP tools (named `mcp__slackgentic__<tool>` in Claude Code), when
present:

| Tool | Use it to |
|---|---|
| `create_loop` | Request a recurring loop; the owner approves the preview in Slack. See the `slackgentic-loops` skill |
| `create_pull_request` | Open a GitHub PR via `gh pr create`, even when your shell is sandboxed or offline |
| `read_thread` | Read a Slack thread from a permalink in the agent channel |
| `request_approval` | Ask the Slack thread for one concrete yes/no approval (connected sessions) |
| `request_user_input` | Ask the Slack thread to pick among options (connected sessions) |

In a connected Claude Code session, use `request_approval` and
`request_user_input` instead of the built-in question tool for decisions that
should reach Slack.

CLI commands that are useful to agents are in
[references/cli.md](references/cli.md). The most common:

```sh
slackgentic loop create "<task and schedule>"   # request a loop (owner approves in Slack)
slackgentic loop list                           # see loops
slackgentic service status                      # is the daemon running?
slackgentic slack doctor                        # check the Slack setup
```

## Feature map

| The user wants to… | Do this |
|---|---|
| Recurring check or report in its own channel | Loop: follow the `slackgentic-loops` skill |
| Resume this thread's work later ("check CI in 20 minutes") | Managed task: `SLACKGENTIC: TIMER` line. Otherwise ask the user to reply in the thread later, or suggest a `schedule` message |
| Schedule a one-off or recurring task for an agent | Managed task: `SLACKGENTIC: SCHEDULE` line. Otherwise give the user a `schedule ...` message to send |
| Hand part of the work to another agent or get a review | In a Slack thread, send a message starting `somebody ...` (for example `somebody review the migration in PR ...`), or start the final paragraph with a specific `@handle` |
| Break a large project into a plan of subtasks | Give the user a `pm: <project>` message to send in the agent channel |
| Add or remove agents | Give the user `hire 2 agents`, `hire 1 claude agent`, `fire @handle` |
| See quotas, token usage, the roster, or loops | Give the user `status`, `show roster`, or `loops` |
| Connect this terminal session to Slack | See "Sessions started outside Slack" in [references/slack-commands.md](references/slack-commands.md) |
| Install, update, or repair Slackgentic | See [references/cli.md](references/cli.md) |

The full list of messages the user can send is in
[references/slack-commands.md](references/slack-commands.md).

## Giving the user Slack commands

Many Slackgentic actions can only be taken by the human in Slack. When that is
the case, give the exact message in a code block and say where to send it: the
**agent channel** (top-level message), a **task thread** (reply), or a **loop
channel**. Commands can also be sent as `/slackgentic-<name> <command>` from
anywhere in Slack, where `<name>` is the user's install suffix. In the agent
channel the command text alone works.

Useful message options:

- `@handle <task>` sends work to a specific agent; plain text goes to any free
  agent.
- `#dangerous-mode` runs that task without approval prompts. Only suggest it
  when the user asks for it.
- `model=<name>` launches the agent with a specific model.
- `in <repo-name> <task>` starts the agent in a sibling repo under the
  configured repos root.
