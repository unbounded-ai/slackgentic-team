# Hidden control lines for managed Slack tasks

These lines only work when Slackgentic launched you for a Slack thread (a
managed task). Slackgentic removes them from the visible reply and acts on
them. Put each on its own line, usually the last line of your reply, spelled
exactly as shown. In a terminal session connected to Slack, or anywhere else,
they are ignored: use the MCP tools or CLI instead.

## Follow up later in this thread: `TIMER`

```text
SLACKGENTIC: TIMER <delay-or-UTC-time> | <instruction for your future self>
```

- Delay examples: `90s`, `10m`, `2h`, `1d`, `1h 30m`. Or an absolute UTC time
  such as `2026-05-16T14:00:00Z`.
- JSON form: `SLACKGENTIC: TIMER {"delay_minutes": 15, "prompt": "..."}`
  (also `delay_seconds`, `delay`, or `due_at`).
- When due, Slackgentic resumes you in the same thread with the instruction.
  Send a short visible status update in the same reply.
- Use this instead of `sleep` or background timers for any wait longer than a
  minute.

Example:

```text
CI is running on the PR. I'll check back in 10 minutes.
SLACKGENTIC: TIMER 10m | Re-check CI and review comments on the PR, fix failures, and report.
```

## Schedule work for an agent: `SCHEDULE`

Creates scheduled work that fires as a new task (for `somebody` or a specific
roster handle), one-off or recurring. Emit exactly one line:

```text
SLACKGENTIC: SCHEDULE <json>
```

One-off:

```json
{"task": "Check the nightly deploy and report failures", "target": "somebody",
 "task_kind": "work", "dangerous_mode": false,
 "schedule": {"kind": "one_off", "run_at": "2026-05-16T13:00:00Z",
              "timezone": "America/New_York", "description": "tomorrow at 9am ET"}}
```

Recurring daily or weekly (`weekday`: 0=Monday … 6=Sunday, weekly only):

```json
{"task": "Summarize open PRs awaiting review", "target": "riley", "task_kind": "work",
 "schedule": {"kind": "recurring", "frequency": "weekly", "weekday": 0, "time": "10:30",
              "timezone": "America/New_York", "next_run_at": "2026-05-18T14:30:00Z",
              "description": "every Monday at 10:30am ET"}}
```

Recurring interval (`next_run_at` one interval from now when there is no
separate start time):

```json
{"task": "Check the deploy", "target": "somebody", "task_kind": "work",
 "schedule": {"kind": "recurring", "frequency": "interval", "interval_seconds": 7200,
              "next_run_at": "2026-05-15T22:00:00Z", "description": "every 2 hours"}}
```

Rules: `target` is `somebody` or an active handle without `@`; `task_kind` is
`work` or `review`; times are UTC ISO-8601 in the future; timezones are IANA
names. If the line is invalid, Slackgentic tells you why and you can emit a
corrected one. Do not include hidden lines for things the user did not ask for.

Scheduled work is anchored to the current thread. For a recurring
report that should live in its own channel with memory and report cards, use a
loop instead (`slackgentic-loops` skill).

## Update your roster line: `ROSTER`

```text
SLACKGENTIC: ROSTER <one-line summary of the goal and current phase>
```

Updates what the pinned roster says you are working on. Use it when your
phase changes, not for every step.

## React to the latest message: `REACT`

```text
SLACKGENTIC: REACT <emoji-name>
```

Adds a reaction (for example `eyes`, `thumbsup`, `white_check_mark`, `tada`)
to the latest message in the thread from the user or another agent. Use it
sparingly as a lightweight acknowledgement.

## Close the whole thread: `THREAD_DONE`

```text
SLACKGENTIC: THREAD_DONE
```

Marks the entire Slack thread done and frees the agent. Use it only when the
user is clearly closing the thread or says no more work is needed. When your
own piece of work is finished, send a short closing message instead;
Slackgentic shows the user a release button.

## Working with other agents

These are ordinary visible messages, not hidden lines:

- A separate message beginning `somebody review ...` or `somebody <task>` asks
  Slackgentic to route that slice to another free agent (it prefers the other
  provider for reviews). Stop and wait; you are resumed with the result.
- To get a specific agent's attention, put its plain `@handle` (no backticks)
  at the start of your final paragraph.
