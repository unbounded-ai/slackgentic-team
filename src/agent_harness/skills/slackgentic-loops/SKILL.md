---
name: slackgentic-loops
description: Create and manage Slackgentic loops — recurring, read-only agent tasks that run on a schedule and post report cards in their own Slack channel. Use when the user asks for a Slackgentic loop, a recurring or scheduled check/report/monitor/watch/digest posted to Slack ("every morning check X and tell me in Slack", "watch CI and ping me when it breaks"), or wants to pause, edit, run, silence, or stop an existing loop.
---

# Slackgentic loops

A loop is a standing mission plus a recurring schedule. Slackgentic gives each
loop its own bot name, emoji, and Slack channel, starts a fresh agent session
for every due run, and posts each run as a report card (status badge, headline,
metrics with deltas, optional chart, full report). A pinned control panel in
the loop channel shows state, schedule, next run, and recent results.

Loops are for recurring work. For a single future task use a Slackgentic
schedule instead (see the `slackgentic` skill).

## Create a loop

### 1. Pin down the request

Before creating anything, make sure you know:

- **Mission**: what each run inspects and what it reports. Be concrete: the
  repo, dashboard, command, query, or URL to check, and what counts as a
  problem.
- **Schedule**: it must recur. Daily or weekly needs a time and a timezone
  (ask for the timezone if the user's location is unknown); intervals must be
  at least 5 minutes.
- **Quiet or not**: quiet loops post nothing on all-clear runs and only post
  (and notify) when a run finds something or fails. Default to quiet when the
  user says things like "only tell me when", "alert me if", or "ping me when".
- Optional: `provider` (`claude` or `codex`) and `public` visibility. Loop
  channels are private by default.

Ask one short question only if the mission or schedule is genuinely missing.
Do not ask about the name, channel, or emoji; Slackgentic chooses them.

### 2. Submit it

Write one self-contained request in plain language: schedule first, then the
mission, then the reporting rule. Example:

> every weekday at 9:00 America/New_York, check GitHub Actions on main in
> example-org/example-repo, list failing workflows with the failing step and a
> likely cause, and only post when something is failing

Then use the first mechanism that is available:

1. **MCP tool** (preferred): call Slackgentic's `create_loop` tool
   (`mcp__slackgentic__create_loop` in Claude Code) with
   `{"request": "<request>"}`, plus optional `"provider"` and `"visibility"`.
2. **CLI**: run

   ```sh
   slackgentic loop create "<request>"
   ```

   Add `--provider claude|codex` or `--public` when asked. The command waits
   up to 20 seconds for the Slackgentic service to pick the request up. If a
   sandbox blocks the command from writing Slackgentic's local state, request
   escalation for that one command or use the MCP tool.
3. **No tool or CLI access**: tell the user to type this in the Slackgentic
   agent channel: `loop create <request>`.

Never try to create the loop channel yourself, post to Slack directly, or edit
Slackgentic's database.

### 3. Tell the user what happens next

Creating a loop always needs the owner's approval in Slack. Slackgentic posts
the request in the agent channel, resolves it into a preview (title, bot name,
channel, schedule, next run, emoji), and waits. The user must click
**Create** on that preview; then Slackgentic creates the channel, invites the
owner, pins the control panel, and runs on schedule. Say this plainly, for
example: "I've sent the loop request to Slackgentic. Approve the preview in
your agent channel and it will create #loop-... and start at the next run."

If the result says the request is still queued, the Slackgentic service is not
running. Suggest `slackgentic service status` and `slackgentic service start`;
the request posts automatically once the service runs. Check a request later
with `slackgentic loop request-status <request-id>`.

Agent-requested loops always start read-only. Do not add `#dangerous-mode` to a
request; the owner can change permissions from the loop channel after creation.

## What a loop can and cannot do

- Runs are **read-only by default**. A guard checks every tool call: reads
  run, anything that could change state (edits outside the loop's scratch
  directory, destructive shell commands, mutating HTTP or SQL, cloud writes) is
  blocked, and unclear calls go to a judge that fails closed. Blocked calls are
  counted on the report card. Each run works in a private scratch directory
  with the configured directory attached read-only.
- Loops keep a journal of run summaries and owner notes, so later runs see
  earlier results and trends. 👍/👎 on a report teaches the loop what the owner
  finds useful.
- Loops cannot delegate to roster agents, start PM initiatives, or read Slack
  files and attachments. Only the owner can instruct a loop bot; other members'
  messages are withheld from the agent.
- A loop that must change things (open PRs, restart a service) needs the owner
  to switch it to `safe-auto` or confirmed `dangerous` in the loop channel.
  Tell the user when the mission they describe needs this.
- Up to 25 loops can be active or pending at once. Each loop gets its own
  dedicated bot; it does not take one of the user's roster agents.

## Manage existing loops

List loops from a terminal:

```sh
slackgentic loop list          # active, paused, and pending loops
slackgentic loop list --json
```

Loop control happens in Slack, and only the owner can do it. Give the user the
exact text to send, in the **loop's channel**:

| Send in the loop channel | Effect |
|---|---|
| `loop status` | Show the control panel |
| `loop pause` / `loop resume` | Pause or resume scheduled runs |
| `loop run now` | Run once now without changing the schedule |
| `loop schedule: <text>` | Change only the schedule |
| `loop task: <text>` | Change only the mission |
| `loop name: <text>` | Rename the loop bot |
| `loop icon: :emoji:` | Change the emoji (a URL or `regenerate` also works) |
| `loop cwd: <path>` | Set the directory future runs read |
| `loop permissions: <mode>` | `read-only` (default), `safe-auto`, `locked`, or `dangerous` (asks for confirmation) |
| `loop quiet: on` / `loop quiet: off` | Only post when a run needs attention, or post every run |
| `loop compact now` | Compact the loop's memory |
| `loop stop` / `loop stop archive` | Stop the loop, optionally archiving its channel |
| `loop help` | Show the in-channel command reference |

The pinned panel also has Run now, Pause/Resume, and **Edit** (mission,
schedule, permissions, reference directory, quiet) buttons. In the main agent
channel, `loops` (or `loop list`) shows every loop as a card.

## If you are a loop run

If your prompt says you are running a Slackgentic loop, follow the reporting
instructions in that prompt (the `SLACKGENTIC: LOOP_SUMMARY` line and friends);
this skill is for creating and managing loops, not for running them.
