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
- message and app-mention events in the configured agent channel.

The live dispatch path is:

- `block_actions` -> roster buttons, task buttons, or agent request buttons;
- `slash_commands` -> setup, usage, or team commands;
- `events_api` -> mirrored external session replies, usage/team commands, task
  thread replies, or new work assignment.

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
