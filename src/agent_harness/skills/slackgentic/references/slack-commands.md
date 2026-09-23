# Slack messages the user can send

Send these in the Slackgentic **agent channel** as plain messages, or from
anywhere in Slack as `/slackgentic-<name> <message>`. Replies inside a thread
are noted separately.

## Start and route work

| Message | Effect |
|---|---|
| `<task>` | A free agent picks it up in a new thread |
| `@riley <task>` | That agent picks it up |
| `somebody review https://github.com/example-org/example-repo/pull/42` | A free agent reviews the PR |
| `in example-project <task>` | Start in a sibling repo under the repos root |
| `<task> #dangerous-mode` | Run without approval prompts (user's explicit choice only) |
| `<task> model=<name>` | Launch with a specific model; later turns in the thread keep it |

The roster message also has an **Add work** button, and each available agent's
card has an **Assign** button. Both open a form for the same options: run now or
once at a set time, optionally waiting for a busy agent to finish. The form has
no repeat options; recurring work belongs in a loop.

## Inside a task thread

| Reply | Effect |
|---|---|
| `<follow-up>` | Continues the same agent session (interrupts the current run if needed) |
| `somebody <subtask>` | Another agent takes that slice; the original agent resumes with the result |
| `stop` | Interrupt the current run without closing the thread |
| Reaction on an agent message | Delivered to the agent as lightweight feedback |
| **Finish and free up this agent** button | Ends the task and frees the agent |

## Schedule work

```text
schedule @riley to check CI tomorrow at 9am PT
schedule somebody review the nightly report every day at 5pm ET
schedule @riley inspect the deploy every Monday at 10:30am America/New_York
schedule somebody check the deploy every 2 hours
```

`remind ...` also works. An agent resolves the wording into a validated
schedule, stored locally and fired by the daemon.

## Loops

| Where | Message | Effect |
|---|---|---|
| Agent channel | `loop create` | Open the guided loop form |
| Agent channel | `loop create <task and schedule>` | Resolve a loop preview directly (add `#public`, `provider=claude\|codex`, `model=<name>`) |
| Agent channel | `loops` or `loop list` | Every loop as a card, with run, pause, resume, and stop controls |
| Loop channel | `loop status`, `loop pause`, `loop resume`, `loop run now`, `loop schedule: …`, `loop task: …`, `loop quiet: on\|off`, `loop permissions: …`, `loop stop`, `loop help` | Manage that loop (owner only) |

Details are in the `slackgentic-loops` skill.

## PM initiatives

```text
pm: ship the new logging stack
pm plan the migration to FastAPI
@alice plan the storage refactor        (alice is a PM-kind agent)
```

A PM agent may ask clarifying questions, then posts a plan of up to 20
subtasks with dependencies and a cost estimate. Nothing runs until the user
clicks **Start executing**. In the initiative thread: `pm status` (or `status`,
`dag`) shows progress; `pm replan: <context>` re-plans after a failure. Hire a
PM agent from a terminal with `slackgentic team hire --kind pm --provider claude --db ~/.slackgentic-team/state.sqlite`.

## Team and status

| Message | Effect |
|---|---|
| `status` | Quota windows and token usage per signed-in account |
| `show roster` | Post the roster |
| `hire 3 agents`, `hire 1 claude agent`, `hire 1 codex agent` | Add agents |
| `fire @riley`, `fire everyone` | Remove agents |
| `show repo root`, `repo root ~/code` | Show or set where agents start |

## Sessions started outside Slack

Terminal sessions can be mirrored into Slack threads, and Slack replies are sent
back into the live session:

```sh
slackgentic claude-channel --install
claude --dangerously-load-development-channels server:slackgentic
```

```sh
slackgentic codex-mcp --install
codex --remote ws://127.0.0.1:47684
```

If every matching agent is busy, Slackgentic posts a hire button and backfills
the thread once capacity frees up.
