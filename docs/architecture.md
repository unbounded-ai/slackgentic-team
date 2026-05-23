# Architecture

Slackgentic is organized around four boundaries: provider adapters, durable
state, Slack control handling, and runtime delivery.

## Provider Adapters

Provider adapters discover local sessions and normalize provider-specific
transcript formats.

- `CodexProvider` reads `~/.codex/sessions/**/*.jsonl`.
- `ClaudeProvider` reads `~/.claude/projects/**/*.jsonl`.

Both adapters expose session discovery, transcript event iteration, and per-day
usage extraction. Launching and sending prompts is handled by the runtime and
session bridge layers because those paths differ between managed tasks and
external sessions.

## Durable State

SQLite is the source of truth for local control-plane state:

- discovered sessions;
- Slack channel and thread mappings;
- persistent team agents;
- identity-to-provider mappings for Codex and Claude;
- channel-originated tasks and PR review tasks;
- managed thread-to-task ownership for restart-safe follow-ups;
- cross-thread dependencies;
- daily usage snapshots;
- Claude channel delivery queues;
- pending Slack-mediated agent input and approval requests.

The schema lives in `agent_harness.storage.store.SCHEMA`. The synchronous
`Store` is used by the Slack daemon and tests; `AsyncStore` is used by the
transcript index watcher.

## Slack App

Slack is the operator interface. The app uses Socket Mode only, so the local
daemon opens an outbound WebSocket to Slack and does not need an inbound request
URL.

Supported Slack surfaces:

- setup, team-management, and usage slash commands;
- roster and task Block Kit buttons;
- provider-neutral input/approval buttons;
- message, app-mention, and reaction events in the configured agent channel.

The live dispatch path is:

- `block_actions` -> roster buttons, task buttons, or agent request buttons;
- `slash_commands` -> setup, usage, or team commands;
- `events_api` -> mirrored external session replies, usage/team commands, task
  thread replies, emoji reaction relay, or new work assignment.

## Identity Model

Slack has one bot user per app. Slackgentic keeps one app per teammate and uses
per-message identity customization for roster agents:

- deterministic full name and handle;
- deterministic avatar slug and emoji fallback;
- `chat.postMessage(username=..., icon_url=...)` when Slack grants
  `chat:write.customize`;
- reaction choice based on the acting agent's personality.

Reactions still come from the single real Slack bot user because Slack does not
support persona-authored reactions.

## Team Roster

The roster is separate from discovered provider sessions. A roster member has:

- a stable handle such as `@riley`;
- a display name and avatar metadata;
- a provider preference, currently `codex` or `claude`;
- outside-work context and reaction preferences;
- active or fired status.

Hiring accepts a provider or `auto`. Auto-hiring picks the less represented
provider among active agents. Firing marks a member inactive instead of deleting
history.

## Work Assignment

Channel messages are parsed into provider-neutral work requests. Assignment
filters the active roster to idle agents, creates a queued task row, and posts a
parent message using the selected roster identity. That parent message becomes
the task thread.

The managed runtime launches the selected provider process, streams visible
assistant output into the thread, and forwards user replies back into the live
process when the provider supports interactive input.

When live delivery is unavailable for a running managed task, same-thread user
replies interrupt the current provider turn and are sent into the same task
right away. Slackgentic only persists queued follow-ups as a recovery path when
interrupt delivery fails, or when replaying older queued rows after restart.
Each Slack reply keeps its own in-progress reaction and is cleared by its own
timestamp when the provider run completes.

Replying `stop` in a task thread sends an Esc-style interrupt into the current
managed provider process without marking the task done or terminating the
process. This mirrors an interactive interrupt: fallback-queued replies are
cleared, the thread stays active, and the next reply can continue from a new
direction.

For review requests, the parser records the author handle when text contains
phrasing such as `@riley's PR` or `PR by @riley`. The picker first prefers
reviewers whose provider differs from the author agent, then falls back to the
whole idle pool.

## External Sessions

External sessions are discoverable from provider transcript files and mirrored
into Slack.

Codex live delivery uses the local Codex app-server. Slackgentic sends
`thread/resume` and `turn/start` JSON-RPC calls to the app-server and listens
for input or approval requests from the server.

Claude live delivery uses a Claude Code channel server. Slackgentic queues
Slack replies in SQLite, and the channel process emits
`notifications/claude/channel` messages to the live Claude session. The same
channel advertises MCP tools for Slack-mediated user input and approval, and it
also advertises Claude's channel permission relay so native tool approval prompts
can be handled in Slack. Setup tries to register this MCP server automatically;
if that registration is missing or an already-open Claude session did not load
it, Slackgentic warns in the session thread instead of treating the channel
launch flag as sufficient.

If a live channel is unavailable, replies fall back to the provider's supported
resume command. The exception is `/exit`: Slackgentic tries the live channel
first, then terminates the matched process directly when it can identify exactly
one process for the session.

## Dependencies Between Threads

An agent thread can block on another thread by posting dependency language such
as:

- `wait for this to go in`;
- `after this lands`;
- `once that goes in`;
- a Slack permalink to another thread.

The daemon stores the blocked task, blocking thread reference, and status. When
the blocking thread is marked done, Slackgentic resumes the waiting task with
context from the blocking thread.

## Deferred Work (DAG Dependencies)

Users can stage new work that fires only after one or more existing pieces of
work finish. From the agent channel:

- `after <permalink> finishes, summarize the deploy`
- `wait for @riley to finish, then update the dashboard`
- `schedule @nell to review the dashboard 20 minutes after <permalink> finishes`
- `after <permalink> and @riley finish, post the release notes`

The Slack controller only detects that the message is a dependency-gated work
request. It then starts a normal managed agent task whose prompt asks the LLM
to resolve the request into a hidden `SLACKGENTIC: DEPEND {...}` control line.
The controller validates the structured JSON, snapshots the active task id for
each occupied agent dependency, and retries the same agent with the validation
error when the control line is malformed (up to three attempts before
cancelling with a Slack-visible message).

After validation, Slackgentic stores the resulting `WorkRequest` plus the
dependency set, optional post-satisfaction delay, and optional absolute run-at
in `deferred_work_requests` and acknowledges the deferred entry in the
originating thread. A daemon poller evaluates waiting rows on every tick and
when every dependency reaches `done` or `cancelled` it promotes the row to
`ready`, computes `fire_at = max(now, run_at)` (or `now + delay` when a
post-satisfaction delay was set), and the same poller fires the work when
`fire_at` arrives. Firing reuses the scheduled-work path: claim → assign →
start the task in the originating thread, or queue it as pending work if no
matching agent is available.

Dependency satisfaction is also re-evaluated whenever a task in the system
finishes (managed task done, finish button, or thread-done control signal),
so newly-satisfied deferred rows promote promptly without waiting for the next
poll tick. Marking the originating thread done cancels its deferred rows
before they fire.

## Agent Timers

Managed agents cannot depend on provider-local sleeps for delayed Slack work:
the provider process may exit, be restarted, or lose its terminal. Timer requests
therefore use the same hidden control-signal pattern as thread completion. The
runtime strips `SLACKGENTIC: TIMER ...` lines from Slack-visible text and passes
the control signal to the Slack controller.

The controller parses the timer into an absolute UTC due time plus a follow-up
prompt, then stores it in `scheduled_timers`. A daemon poller claims due timers
and resumes the same agent in the same thread using the existing same-thread
continuation path, preserving the provider session id when one is known. Marking
a thread done cancels pending timers for that thread.

## PM Initiatives

PM-flavored agents are a distinct roster persona. Hire one with
`team hire --kind pm` (provider can be `claude` or `codex`). PM-kind agents
are stored on the same `team_agents` table with a `kind` column, and the
work router filters them out of the ANYONE candidate pool so they are never
picked up for generic worker tasks. The schema migration backfills
`kind = 'engineer'` for pre-existing rows.

Two entry points trigger PM mode:

- `pm: ship the new logging stack` / `pm plan the migration to FastAPI` —
  the `pm` keyword shortcut routes to an idle PM-kind agent (or falls back
  to a worker when no PM is hired).
- `@alice plan the storage refactor` — tagging a PM-kind agent by handle
  always activates PM mode, even without the `pm` keyword. The Slack
  controller compares the leading handle against the PM roster and pins
  assignment with `AssignmentMode.SPECIFIC`.

In a thread that already owns a PM initiative, follow-up replies stay with
the same PM agent (the thread lookup in `_handle_pm_request` defers to
`_handle_task_thread_reply` so a new initiative is not created). The
controller wraps each follow-up with a `[PM HARNESS: initiative state]`
snapshot — initiative id/title/status plus every subtask's status and owner
handle — before delivery, so the PM can answer status questions or replan
without re-reading thread history.

The PM owns the initiative for its full life cycle. After emitting the
plan, the PM resolver task is not marked `DONE` for PM-kind agents — the
agent stays assigned to the thread so it can field clarifying questions,
status checks, and replans. Because PM-kind agents are excluded from the
worker router, this does not consume a worker slot.

The plan itself is a hidden `SLACKGENTIC: PM_PLAN {...}` control line.
The JSON includes a title, summary, and up to 20 subtasks with sibling
`depends_on` ids, target (`somebody` or a specific handle), task kind
(`work`/`review`), and an optional `after_delay_seconds` gate. The PM
persona prompt also instructs the agent to ask at most one or two
clarifying questions in the thread before committing to a plan.

A subtask may include a `co_designers: [handles…]` list of two or more
worker handles. `expand_codesign_plan` fans the subtask out into one draft
per co-designer (each new `local_id` is `<original_id>--<handle>`, pinned
to that handle with `AssignmentMode.SPECIFIC`) plus a synthesis subtask
that takes the original `local_id` and depends on every draft. Downstream
subtasks that referenced the original id continue to depend on the
synthesis stage, so fan-out is transparent to the rest of the DAG.

After validation, the controller does *not* immediately dispatch the
plan. Instead, the initiative is moved to `awaiting_approval` with the
expanded plan serialized into `pm_initiatives.pending_plan_json`, and the
plan ack message is posted with a *Start executing* (primary) and
*Cancel* (danger) button row. The ack also includes a rough plan
estimate (`estimate_pm_plan`) — subtask count, critical-path depth,
dangerous-mode count, co-design fan-outs, and a wall-clock band — so the
user has a sense of the cost before approving. The watchdog only polls
initiatives in `active`, so nothing escalates until a human starts
execution.

When the user clicks *Start executing*, the block-action handler
deserializes the parked plan, promotes the initiative to `active`,
persists each subtask as a `deferred_work` row (with sibling references
stored as `WorkDependency(kind=SUBTASK)`) plus one `pm_subtasks` row per
subtask, and inserts root subtasks (no plan-level deps) in `ready` status
with `fire_at = now()` so the existing deferred-work poller dispatches
them on the next tick. As each subtask finishes,
`evaluate_pending_deferred_work` promotes downstream subtasks — no new
dispatch code is needed. If subtask persistence fails partway through,
the initiative is rolled back to `cancelled` and the surfaced error is
posted to the thread. *Cancel* moves the initiative straight to
`cancelled` without persisting any subtasks. Both buttons rewrite the
plan message to drop the action row and append a `_Plan {status}._`
footer so the approval cannot be re-clicked.

A `PMInitiativeRunner` polls active initiatives every ~30 s and acts as
a watchdog. It surfaces blockers in the initiative thread once per
fingerprint:

- `STALLED_APPROVAL` — an unresolved `slack_agent_requests` row in a
  subtask's thread older than 5 minutes.
- `STALLED_TASK` — a subtask task in `active` for longer than 30 minutes
  with no managed runtime activity.
- terminal recap — when every subtask reaches a terminal status, the
  watchdog posts a per-subtask outcome list, sets the initiative to `done`
  or `cancelled`, and clears the dedup settings.

The watchdog only reads state and posts in Slack. Dispatch belongs to the
deferred-work poller; this separation keeps the watchdog safe even when it
sees a transient false positive. When a post to the initiative thread
fails with `channel_not_found`, `thread_not_found`, `message_not_found`,
`is_archived`, or `not_in_channel`, the watchdog interprets the thread as
permanently gone and moves the initiative to `cancelled` instead of
looping against the dead thread. The watchdog's `last_run_at` heartbeat
is only stamped after a successful evaluation, so a monitoring check on a
stale heartbeat still notices a thrown exception.

In-thread commands a user can drop into an initiative thread:

- `pm status` (also `pm plan`, `pm dag`, `pm state`, `/status`) — the
  controller renders the current DAG with per-subtask status, owner, and
  deps directly in Slack without consulting the PM agent. The same
  renderer feeds the `[PM HARNESS: initiative state]` prefix that PM
  agents see on every follow-up.
- `pm replan: <new context>` — cancels every non-DONE subtask, resets
  the initiative to `planning`, and re-runs the PM resolver with the
  prior plan snapshot plus the user's new instructions. Only available
  when a PM-kind agent is still attached to the initiative.
- `pm extend: <new work>` — keeps the existing subtasks running and asks
  the PM to plan an extension. The new plan may reference existing
  subtask ids in `depends_on` (allowed via
  `parse_agent_pm_plan_signal(..., allowed_external_dep_ids=…)`). The
  extension still goes through the normal approval gate.

Marking the initiative thread done cascades through
`cancel_pm_initiative_for_thread` and the existing
`cancel_deferred_work_for_thread`, so cancelling a PM thread stops all
non-terminal subtasks.

## User Scheduled Work

Users can create one-off or recurring scheduled work from the agent channel with
natural-language phrases such as `tomorrow at 9am PT`, `in 30 minutes`,
`every day at 5pm ET`, `every Monday at 10:30am America/New_York`,
`every 2 hours`, or `during tomorrow's sunset time in Waco`.

The Slack controller only detects that the message is asking to schedule work.
It then starts a normal managed agent task whose prompt asks the LLM to resolve
the schedule into a hidden `SLACKGENTIC: SCHEDULE {...}` control line. This keeps
interpretation with the agent instead of a local regex parser. The controller
validates the structured JSON and retries the same agent with the validation
error when the control line is malformed.

After validation, Slackgentic stores the resulting `WorkRequest` plus recurrence
metadata in `scheduled_work_requests` and acknowledges the schedule in the
original thread. A daemon poller claims due rows and starts normal Slackgentic
work in that same thread. If no matching agent is available at the due time, the
due occurrence is queued as pending work so it can resume when capacity opens.
One-off schedules move to `done` after their due occurrence is launched or
queued; recurring schedules compute their next run after each due occurrence.
Marking a thread done cancels pending schedules for that thread.
