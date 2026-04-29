import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent_harness.models import (
    DANGEROUS_MODE_METADATA_KEY,
    AgentSession,
    AgentTaskStatus,
    AssignmentMode,
    ControlMode,
    Provider,
    SessionStatus,
    SlackThreadRef,
    WorkRequest,
)
from agent_harness.runtime.tasks import AGENT_THREAD_DONE_SIGNAL
from agent_harness.slack import encode_action_value
from agent_harness.slack.app import (
    AUTO_ALLOWED_CLAUDE_PERMISSION_TEXT,
    CLAUDE_CHANNEL_PERMISSION_METHOD,
    DEFAULT_AGENT_AVATAR_BASE_URL,
    SETTING_ROSTER_TS,
    SETTING_SLACK_BACKFILL_LAST_AWAKE,
    ClaudePermissionAutoResolver,
    SlackMessageBackfill,
    SlackReplyTarget,
    SlackTeamController,
)
from agent_harness.slack.client import PostedMessage
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team, create_agent_task
from agent_harness.team.commands import FireCommand, FireEveryoneCommand, HireCommand, RosterCommand


class FakeGateway:
    def __init__(self):
        self.posts = []
        self.updates = []
        self.thread_replies = []
        self.channels = []
        self.invites = []
        self.views = []
        self.reactions = []
        self.removed_reactions = []
        self.pins = []
        self.history_messages = []
        self.thread_history_messages = {}

    def create_channel(self, name, is_private):
        self.channels.append((name, is_private))
        return "CNEW"

    def invite_users(self, channel_id, user_ids):
        self.invites.append((channel_id, user_ids))

    def open_view(self, trigger_id, view):
        self.views.append((trigger_id, view))

    def post_message(self, channel_id, text, blocks=None, thread_ts=None):
        ts = f"1712345678.{len(self.posts):06d}"
        self.posts.append(
            {
                "channel_id": channel_id,
                "text": text,
                "blocks": blocks,
                "thread_ts": thread_ts,
                "ts": ts,
            }
        )
        return PostedMessage(channel_id=channel_id, ts=ts, thread_ts=thread_ts)

    def update_message(self, channel_id, ts, text, blocks=None):
        self.updates.append({"channel_id": channel_id, "ts": ts, "text": text, "blocks": blocks})

    def permalink(self, channel_id, message_ts):
        return f"https://example.slack.com/archives/{channel_id}/p{message_ts.replace('.', '')}"

    def pin_message(self, channel_id, message_ts):
        self.pins.append((channel_id, message_ts))

    def post_thread_reply(self, thread, text, persona=None, icon_url=None, blocks=None):
        ts = f"1712345679.{len(self.thread_replies):06d}"
        self.thread_replies.append(
            {
                "thread": thread,
                "text": text,
                "persona": persona,
                "icon_url": icon_url,
                "blocks": blocks,
                "ts": ts,
            }
        )
        return PostedMessage(thread.channel_id, ts, thread.thread_ts)

    def post_session_parent(self, channel_id, text, persona, icon_url=None, blocks=None):
        posted = self.post_message(channel_id, text, blocks=blocks)
        self.posts[-1]["icon_url"] = icon_url
        return posted

    def post_task_parent(self, channel_id, text, agent, blocks=None, icon_url=None):
        posted = self.post_message(channel_id, text, blocks=blocks)
        self.posts[-1]["icon_url"] = icon_url
        from agent_harness.models import SlackThreadRef

        return SlackThreadRef(channel_id, posted.ts, posted.ts)

    def post_team_initialization(self, channel_id, agents, messages, icon_url_for=None):
        self.posts.append(
            {
                "channel_id": channel_id,
                "text": "initialization",
                "agents": agents,
                "messages": messages,
            }
        )

    def add_reaction(self, channel_id, ts, reaction_name):
        self.reactions.append((channel_id, ts, reaction_name))
        return True

    def remove_reaction(self, channel_id, ts, reaction_name):
        self.removed_reactions.append((channel_id, ts, reaction_name))
        return True

    def thread_messages(self, channel_id, thread_ts, limit=20, oldest=None):
        messages = [
            {
                "username": getattr(item.get("persona"), "full_name", None),
                "text": item["text"],
                "ts": item["ts"],
                "thread_ts": item["thread"].thread_ts,
            }
            for item in self.thread_replies
            if item["thread"].channel_id == channel_id and item["thread"].thread_ts == thread_ts
        ]
        messages.extend(self.thread_history_messages.get((channel_id, thread_ts), []))
        parents = [
            {"username": None, "text": item["text"], "ts": item["ts"]}
            for item in self.posts
            if item["channel_id"] == channel_id and item["ts"] == thread_ts
        ]
        combined = parents + messages
        if oldest:
            combined = [item for item in combined if float(item.get("ts", 0)) > float(oldest)]
        return combined[-limit:]

    def channel_messages(self, channel_id, oldest=None, limit=200):
        messages = [
            item for item in self.history_messages if item.get("channel", channel_id) == channel_id
        ]
        if oldest:
            messages = [item for item in messages if float(item.get("ts", 0)) > float(oldest)]
        return messages[:limit]


class FakeRuntime:
    def __init__(self):
        self.started = []
        self.sent = []
        self.stopped = []

    def start_task(self, task, agent, thread):
        self.started.append((task, agent, thread))
        return True

    def send_to_task(self, task_id, message):
        self.sent.append((task_id, message))
        return True

    def stop_task(self, task_id, status=AgentTaskStatus.CANCELLED):
        self.stopped.append((task_id, status))
        return True


class FakeSessionBridge:
    def __init__(self):
        self.sent = []
        self.agent_request_actions = []

    def send_to_session(self, session, text, thread, slack_user=None):
        self.sent.append((session, text, thread, slack_user))
        return True

    def handle_agent_request_block_action(self, payload, channel_id, message_ts):
        self.agent_request_actions.append((payload, channel_id, message_ts))
        return True


class DetachedRuntime(FakeRuntime):
    def send_to_task(self, task_id, message):
        self.sent.append((task_id, message))
        return False


class SlackAppTests(unittest.TestCase):
    def test_hire_button_adds_agent_and_refreshes_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 1):
                    store.upsert_team_agent(agent)
                store.set_setting(SETTING_ROSTER_TS, "171.000001")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.000001"},
                        "actions": [
                            {
                                "value": encode_action_value(
                                    "team.hire", count=1, provider=Provider.CLAUDE.value
                                )
                            }
                        ],
                    }
                )

                self.assertEqual(len(store.list_team_agents()), 3)
                self.assertEqual(store.list_team_agents()[-1].provider_preference, Provider.CLAUDE)
                self.assertEqual(len(gateway.updates), 1)
                self.assertEqual(len(gateway.thread_replies), 1)
            finally:
                store.close()

    def test_fire_button_marks_agent_inactive_and_refreshes_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                store.set_setting(SETTING_ROSTER_TS, "171.000001")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.000001"},
                        "actions": [
                            {
                                "value": encode_action_value(
                                    "team.fire", agent_id=agent.agent_id, handle=agent.handle
                                )
                            }
                        ],
                    }
                )

                self.assertEqual(store.list_team_agents(), [])
                self.assertEqual(len(gateway.updates), 1)
            finally:
                store.close()

    def test_agent_request_button_routes_to_session_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            try:
                store.init_schema()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    session_bridge=bridge,
                )

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.000001"},
                        "actions": [{"value": encode_action_value("agent.request", token="t1")}],
                    }
                )

                self.assertEqual(len(bridge.agent_request_actions), 1)
                self.assertEqual(bridge.agent_request_actions[0][1:], ("C1", "171.000001"))
            finally:
                store.close()

    def test_auto_resolves_internal_claude_slackgentic_permission_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                thread = SlackThreadRef("C1", "171.000001")
                store.create_slack_agent_request(
                    "internal",
                    "Claude",
                    CLAUDE_CHANNEL_PERMISSION_METHOD,
                    {
                        "request_id": "req-1",
                        "tool_name": "mcp__slackgentic__request_user_input",
                        "description": "Ask Slack",
                        "input_preview": "{}",
                    },
                    thread,
                    message_ts="171.000002",
                )
                store.create_slack_agent_request(
                    "bash",
                    "Claude",
                    CLAUDE_CHANNEL_PERMISSION_METHOD,
                    {
                        "request_id": "req-2",
                        "tool_name": "Bash",
                        "description": "List files",
                        "input_preview": "ls",
                    },
                    thread,
                    message_ts="171.000003",
                )
                resolver = ClaudePermissionAutoResolver(store, gateway, poll_seconds=0.01)

                self.assertEqual(resolver.resolve_once(), 1)

                resolved, response = store.get_slack_agent_request_response("internal")
                self.assertTrue(resolved)
                self.assertEqual(response, {"behavior": "allow"})
                resolved, _ = store.get_slack_agent_request_response("bash")
                self.assertFalse(resolved)
                self.assertEqual(gateway.updates[0]["text"], AUTO_ALLOWED_CLAUDE_PERMISSION_TEXT)
                self.assertEqual(gateway.updates[0]["blocks"], [])
                self.assertEqual(gateway.updates[1]["text"], "Claude requests tool approval: Bash")
                self.assertIn("```ls```", gateway.updates[1]["blocks"][1]["text"]["text"])

                self.assertEqual(resolver.resolve_once(), 0)
                self.assertEqual(len(gateway.updates), 2)
            finally:
                store.close()

    def test_hire_command_works_in_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 1):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_team_command(
                    HireCommand(count=2, provider=None),
                    SlackReplyTarget(channel_id="C1", thread_ts="171.000001"),
                )

                self.assertEqual(len(store.list_team_agents()), 4)
                self.assertEqual(len(gateway.thread_replies), 2)
                self.assertTrue(gateway.posts)
            finally:
                store.close()

    def test_hire_resumes_pending_channel_work_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "#dangerous-mode write a tiny validation note",
                            "ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(len(store.list_pending_work_requests()), 1)
                self.assertTrue(
                    store.list_pending_work_requests()[0].extra_metadata[
                        DANGEROUS_MODE_METADATA_KEY
                    ]
                )
                self.assertIn(
                    "resume this thread automatically",
                    gateway.thread_replies[-1]["text"],
                )

                restarted_controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                restarted_controller.handle_team_command(
                    HireCommand(count=1, provider=Provider.CODEX),
                    SlackReplyTarget(channel_id="C1", thread_ts="171.hire"),
                )

                self.assertEqual(store.list_pending_work_requests(), [])
                self.assertEqual(len(runtime.started), 1)
                task, agent, thread = runtime.started[0]
                self.assertEqual(task.prompt, "write a tiny validation note")
                self.assertTrue(task.metadata[DANGEROUS_MODE_METADATA_KEY])
                self.assertEqual(agent.provider_preference, Provider.CODEX)
                self.assertEqual(thread.thread_ts, "171.000001")
                self.assertTrue(
                    any(
                        "Capacity is available now." in item["text"]
                        for item in gateway.thread_replies
                    )
                )
            finally:
                store.close()

    def test_fire_command_works_in_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_team_command(
                    FireCommand(handle=agent.handle),
                    SlackReplyTarget(channel_id="C1", thread_ts="171.000001"),
                )

                self.assertEqual(store.list_team_agents(), [])
                self.assertTrue(gateway.posts)
            finally:
                store.close()

    def test_fire_everyone_command_fires_all_active_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                for agent in build_initial_model_team(2, 1):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_team_command(
                    FireEveryoneCommand(),
                    SlackReplyTarget(channel_id="C1", thread_ts="171.000001"),
                )

                self.assertEqual(store.list_team_agents(), [])
                self.assertTrue(any("Fired 3 agent(s)." in post["text"] for post in gateway.posts))
                self.assertTrue(gateway.updates or gateway.posts)
            finally:
                store.close()

    def test_roster_command_posts_visible_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 1):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_team_command(
                    RosterCommand(),
                    SlackReplyTarget(channel_id="C1"),
                )

                self.assertEqual(
                    gateway.posts[-1]["text"],
                    "Agent roster: 2 active lightweight handles, 2 available, 0 occupied",
                )
                self.assertIsNotNone(gateway.posts[-1]["blocks"])
                self.assertEqual(gateway.pins, [("C1", gateway.posts[-1]["ts"])])
            finally:
                store.close()

    def test_roster_shows_active_task_occupancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                task = create_agent_task(agents[0], "investigate flaky tests", "C1")
                store.upsert_agent_task(task)
                store.update_agent_task_thread(task.task_id, "171.000001", "171.000001")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.post_roster("C1")

                self.assertIn("1 available, 1 occupied", gateway.posts[-1]["text"])
                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn("Queued: Slack task: investigate flaky tests", blocks)
                self.assertNotIn("<https://example.slack.com/archives/C1/p", blocks)
                self.assertIn("'text': {'type': 'plain_text', 'text': 'Open thread'}", blocks)
                self.assertIn("'url': 'https://example.slack.com/archives/C1/p", blocks)
                self.assertIn("Free up", blocks)
                self.assertIn("Available", blocks)
            finally:
                store.close()

    def test_roster_marks_dangerous_mode_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "rewrite installer", "C1"),
                    metadata={DANGEROUS_MODE_METADATA_KEY: True},
                )
                store.upsert_agent_task(task)
                store.update_agent_task_thread(task.task_id, "171.000001", "171.000001")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.post_roster("C1")

                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn("Queued: Dangerous mode: Slack task: rewrite installer", blocks)
            finally:
                store.close()

    def test_runtime_task_exit_refreshes_roster_while_agent_stays_occupied(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "validate ownership", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.000001",
                    parent_message_ts="171.000001",
                )
                store.upsert_agent_task(task)
                store.set_setting(SETTING_ROSTER_TS, "171.000099")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_runtime_task_done(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.000001"),
                )

                self.assertEqual(gateway.updates[-1]["ts"], "171.000099")
                self.assertIn("0 available, 1 occupied", gateway.updates[-1]["text"])
                self.assertEqual(store.get_agent_task(task.task_id).status, AgentTaskStatus.ACTIVE)
            finally:
                store.close()

    def test_runtime_subtask_exit_marks_request_message_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "review subtask", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.000001",
                    parent_message_ts="171.bot",
                    metadata={
                        "parent_task_id": "task_parent",
                        "request_message_ts": "171.user",
                    },
                )
                store.upsert_agent_task(task)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_runtime_task_done(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.000001"),
                )

                self.assertIn(
                    ("C1", "171.user", "hourglass_flowing_sand"), gateway.removed_reactions
                )
                self.assertIn(("C1", "171.user", "eyes"), gateway.removed_reactions)
                self.assertIn(("C1", "171.user", "white_check_mark"), gateway.reactions)
                self.assertEqual(store.get_agent_task(task.task_id).status, AgentTaskStatus.DONE)
            finally:
                store.close()

    def test_runtime_external_thread_helper_exit_marks_task_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "find options for external agent", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.000001",
                    parent_message_ts="171.bot",
                    metadata={
                        "external_session_provider": Provider.CODEX.value,
                        "external_session_id": "codex-external",
                        "request_message_ts": "171.user",
                    },
                )
                store.upsert_agent_task(task)
                store.upsert_managed_thread_task(task, SlackThreadRef("C1", "171.000001"))
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_runtime_task_done(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.000001"),
                )

                self.assertEqual(store.get_agent_task(task.task_id).status, AgentTaskStatus.DONE)
                self.assertIsNone(store.get_managed_thread_task("C1", "171.000001", agent.agent_id))
                self.assertIn(("C1", "171.user", "white_check_mark"), gateway.reactions)
            finally:
                store.close()

    def test_roster_shows_external_session_occupancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 0)
                for agent in agents:
                    store.upsert_team_agent(agent)
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CODEX,
                        session_id="s1",
                        transcript_path=Path(tmp) / "codex.jsonl",
                        status=SessionStatus.ACTIVE,
                    )
                )
                store.set_setting("external_session_agent.codex.s1", agents[0].agent_id)
                store.set_setting("external_session_summary.codex.s1", "update the docs")
                store.upsert_slack_thread_for_session(
                    Provider.CODEX,
                    "s1",
                    "local",
                    SlackThreadRef("C1", "171.000001", "171.000001"),
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.post_roster("C1")

                self.assertIn("0 available, 1 occupied", gateway.posts[-1]["text"])
                self.assertIn(
                    "codex session outside Slack: update the docs", str(gateway.posts[-1]["blocks"])
                )
                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn("Free up", blocks)
                self.assertIn("Open thread", blocks)
                self.assertIn("'url': 'https://example.slack.com/archives/C1/p", blocks)
                self.assertLess(blocks.index("Free up"), blocks.index("Open thread"))
                self.assertLess(blocks.index("Open thread"), blocks.index("Fire"))
            finally:
                store.close()

    def test_roster_shows_idle_tracked_external_session_occupancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agents = build_initial_model_team(0, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CLAUDE,
                        session_id="s1",
                        transcript_path=Path(tmp) / "claude.jsonl",
                        status=SessionStatus.IDLE,
                    )
                )
                store.set_setting("external_session_agent.claude.s1", agents[0].agent_id)
                store.set_setting("external_session_summary.claude.s1", "waiting on approval")
                store.upsert_slack_thread_for_session(
                    Provider.CLAUDE,
                    "s1",
                    "local",
                    SlackThreadRef("C1", "171.000001", "171.000001"),
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.post_roster("C1")

                self.assertIn("0 available, 1 occupied", gateway.posts[-1]["text"])
                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn("claude session outside Slack: waiting on approval", blocks)
                action_blocks = [
                    block
                    for block in gateway.posts[-1]["blocks"]
                    if block.get("block_id") == f"team.agent.actions.{agents[0].agent_id}"
                ]
                self.assertEqual(len(action_blocks), 1)
                actions = action_blocks[0]["elements"]
                self.assertEqual(
                    [action["text"]["text"] for action in actions],
                    [
                        "Free up",
                        "Open thread",
                        "Fire",
                    ],
                )
                self.assertEqual(actions[0]["action_id"], "external.session.finish")
                self.assertIn("'url': 'https://example.slack.com/archives/C1/p", blocks)
            finally:
                store.close()

    def test_roster_shows_external_session_free_up_without_thread_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agents = build_initial_model_team(0, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CLAUDE,
                        session_id="s1",
                        transcript_path=Path(tmp) / "claude.jsonl",
                        status=SessionStatus.ACTIVE,
                    )
                )
                store.set_setting("external_session_agent.claude.s1", agents[0].agent_id)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.post_roster("C1")

                action_blocks = [
                    block
                    for block in gateway.posts[-1]["blocks"]
                    if block.get("block_id") == f"team.agent.actions.{agents[0].agent_id}"
                ]
                self.assertEqual(len(action_blocks), 1)
                actions = action_blocks[0]["elements"]
                self.assertEqual(
                    [action["text"]["text"] for action in actions],
                    [
                        "Free up",
                        "Fire",
                    ],
                )
                self.assertEqual(actions[0]["action_id"], "external.session.finish")
            finally:
                store.close()

    def test_external_session_free_up_button_ignores_session_and_refreshes_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CODEX,
                        session_id="s1",
                        transcript_path=Path(tmp) / "codex.jsonl",
                        status=SessionStatus.ACTIVE,
                    )
                )
                store.set_setting("external_session_agent.codex.s1", agent.agent_id)
                store.upsert_slack_thread_for_session(
                    Provider.CODEX,
                    "s1",
                    "local",
                    SlackThreadRef("C1", "171.000001", "171.000001"),
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.000002"},
                        "actions": [
                            {
                                "value": encode_action_value(
                                    "external.session.finish",
                                    provider=Provider.CODEX.value,
                                    session_id="s1",
                                )
                            }
                        ],
                    }
                )

                self.assertIsNone(store.get_setting("external_session_agent.codex.s1"))
                self.assertIsNotNone(store.get_setting("external_session_ignored.codex.s1"))
                self.assertIn(
                    "Finished and freed up this agent.", gateway.thread_replies[-1]["text"]
                )
                self.assertIn("1 available, 0 occupied", gateway.posts[-1]["text"])
            finally:
                store.close()

    def test_specific_request_for_external_busy_agent_resumes_when_agent_frees(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(0, 1)[0]
                store.upsert_team_agent(agent)
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CLAUDE,
                        session_id="s1",
                        transcript_path=Path(tmp) / "claude.jsonl",
                        status=SessionStatus.ACTIVE,
                    )
                )
                store.set_setting("external_session_agent.claude.s1", agent.agent_id)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{agent.handle} hi",
                            "ts": "171.000002",
                            "thread_ts": "171.000001",
                        }
                    }
                )

                pending = store.list_pending_work_requests(channel_id="C1")
                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0].request.requested_handle, agent.handle)
                self.assertIn("That specific agent is busy", gateway.thread_replies[-1]["text"])

                store.delete_setting("external_session_agent.claude.s1")
                controller.handle_external_session_occupancy_change("C1")

                row = store.conn.execute(
                    "SELECT status FROM pending_work_requests WHERE pending_id = ?",
                    (pending[0].pending_id,),
                ).fetchone()
                self.assertEqual(row["status"], "assigned")
                self.assertEqual(len(runtime.started), 1)
                self.assertEqual(runtime.started[0][1].agent_id, agent.agent_id)
                self.assertIn("Capacity is available now", gateway.thread_replies[-1]["text"])
            finally:
                store.close()

    def test_channel_request_does_not_assign_idle_tracked_external_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(0, 1)[0]
                store.upsert_team_agent(agent)
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CLAUDE,
                        session_id="s1",
                        transcript_path=Path(tmp) / "claude.jsonl",
                        status=SessionStatus.IDLE,
                    )
                )
                store.set_setting("external_session_agent.claude.s1", agent.agent_id)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "tablet service should be multi-tablet",
                            "ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(store.list_agent_tasks(), [])
                self.assertEqual(runtime.started, [])
                self.assertIn("No agents are available", gateway.thread_replies[-1]["text"])
            finally:
                store.close()

    def test_external_session_free_up_button_resumes_pending_specific_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(0, 1)[0]
                store.upsert_team_agent(agent)
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CLAUDE,
                        session_id="s1",
                        transcript_path=Path(tmp) / "claude.jsonl",
                        status=SessionStatus.ACTIVE,
                    )
                )
                store.set_setting("external_session_agent.claude.s1", agent.agent_id)
                store.upsert_slack_thread_for_session(
                    Provider.CLAUDE,
                    "s1",
                    "local",
                    SlackThreadRef("C1", "171.000010", "171.000010"),
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{agent.handle} hi",
                            "ts": "171.000002",
                            "thread_ts": "171.000001",
                        }
                    }
                )
                pending = store.list_pending_work_requests(channel_id="C1")
                self.assertEqual(len(pending), 1)

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.000003"},
                        "actions": [
                            {
                                "value": encode_action_value(
                                    "external.session.finish",
                                    provider=Provider.CLAUDE.value,
                                    session_id="s1",
                                )
                            }
                        ],
                    }
                )

                row = store.conn.execute(
                    "SELECT status FROM pending_work_requests WHERE pending_id = ?",
                    (pending[0].pending_id,),
                ).fetchone()
                self.assertEqual(row["status"], "assigned")
                self.assertEqual(len(runtime.started), 1)
                self.assertEqual(runtime.started[0][1].agent_id, agent.agent_id)
            finally:
                store.close()

    def test_resume_pending_work_uses_stored_channel_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(0, 1)[0]
                store.upsert_team_agent(agent)
                store.set_setting("slack.channel_id", "C1")
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id=None,
                    runtime=runtime,
                )
                store.create_pending_work_request(
                    SlackThreadRef("C1", "171.000001"),
                    WorkRequest(
                        prompt="hi",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    requested_by_slack_user="U1",
                )

                resumed = controller.resume_pending_work_requests_for_configured_channel()

                self.assertEqual(resumed, 1)
                self.assertEqual(len(runtime.started), 1)
            finally:
                store.close()

    def test_roster_channel_message_posts_in_channel_not_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 1):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "show roster",
                            "ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(gateway.posts[-1]["thread_ts"], None)
                self.assertIn("Agent roster", gateway.posts[-1]["text"])
            finally:
                store.close()

    def test_setup_submission_creates_channel_team_roster_and_invite(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(store, gateway)

                controller.handle_view_submission(
                    {
                        "type": "view_submission",
                        "user": {"id": "U1"},
                        "view": {
                            "callback_id": "setup.initial",
                            "state": {
                                "values": {
                                    "channel_name": {"value": {"value": "Agent Team"}},
                                    "visibility": {
                                        "value": {"selected_option": {"value": "private"}}
                                    },
                                    "codex_count": {"value": {"value": "1"}},
                                    "claude_count": {"value": {"value": "1"}},
                                    "repo_root": {"value": {"value": str(Path(tmp))}},
                                }
                            },
                        },
                    }
                )

                self.assertEqual(gateway.channels, [("agent-team", True)])
                self.assertEqual(gateway.invites, [("CNEW", ["U1"])])
                self.assertEqual(store.get_setting("slack.repo_root"), str(Path(tmp).resolve()))
                self.assertEqual(len(store.list_team_agents()), 2)
                self.assertTrue(gateway.posts)
            finally:
                store.close()

    def test_setup_submission_defaults_to_two_agent_team(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(store, gateway)

                controller.handle_view_submission(
                    {
                        "type": "view_submission",
                        "view": {
                            "callback_id": "setup.initial",
                            "state": {"values": {}},
                        },
                    }
                )

                agents = store.list_team_agents()
                self.assertEqual(len(agents), 2)
                self.assertEqual(
                    [agent.provider_preference for agent in agents],
                    [Provider.CODEX, Provider.CLAUDE],
                )
            finally:
                store.close()

    def test_setup_slash_command_opens_setup_modal(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_slash_command(
                    {"text": "setup", "channel_id": "C1", "trigger_id": "T1"}
                )

                self.assertEqual(gateway.views[0][0], "T1")
                self.assertEqual(gateway.views[0][1]["callback_id"], "setup.initial")
            finally:
                store.close()

    def test_setup_slash_modal_defaults_repo_root_two_directories_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                default_cwd = Path(tmp) / "org" / "slackgentic-team"
                default_cwd.mkdir(parents=True)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    default_cwd=default_cwd,
                )

                controller.handle_slash_command(
                    {"text": "setup", "channel_id": "C1", "trigger_id": "T1"}
                )

                view = gateway.views[0][1]
                repo_block = next(
                    block for block in view["blocks"] if block["block_id"] == "repo_root"
                )
                self.assertEqual(repo_block["element"]["initial_value"], str(Path(tmp).resolve()))
            finally:
                store.close()

    def test_setup_submission_rejects_missing_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(store, gateway)

                response = controller.handle_view_submission(
                    {
                        "type": "view_submission",
                        "view": {
                            "callback_id": "setup.initial",
                            "state": {
                                "values": {
                                    "repo_root": {"value": {"value": str(Path(tmp) / "missing")}}
                                }
                            },
                        },
                    }
                )

                self.assertEqual(response["response_action"], "errors")
                self.assertIn("repo_root", response["errors"])
            finally:
                store.close()

    def test_channel_overview_uses_configured_slash_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    slash_command="/slackgentic-ilshat",
                )

                controller.post_channel_overview("C1")

                text = gateway.posts[-1]["text"]
                self.assertIn("@agentname", text)
                self.assertIn("somebody ...", text)
                self.assertIn("/slackgentic-ilshat <command>", text)
                self.assertIn("/slackgentic-ilshat hire 3 agents", text)
                self.assertIn("or just `status`", text)
                self.assertIn("#dangerous-mode", text)
                self.assertIn("Codex outside Slack", text)
                self.assertIn("Claude outside Slack", text)
                self.assertIn("slackgentic claude-channel --install", text)
                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn("Write anything in this channel", blocks)
                self.assertIn("type them directly in this channel", blocks)
                self.assertIn("Thread subtasks", blocks)
                self.assertIn("Dangerous mode", blocks)
                self.assertIn("Sessions started outside Slack", blocks)
                self.assertIn("slackgentic claude-channel --install", blocks)
                self.assertIn(("C1", gateway.posts[-1]["ts"]), gateway.pins)
            finally:
                store.close()

    def test_hire_slash_command_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_slash_command({"text": "hire 2 codex agents", "channel_id": "C1"})

                self.assertEqual(len(store.list_team_agents()), 2)
                self.assertTrue(gateway.posts)
            finally:
                store.close()

    def test_hire_slash_command_rejects_more_than_500_active_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                for agent in build_initial_model_team(250, 250):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_slash_command({"text": "hire 1 codex agent", "channel_id": "C1"})

                self.assertEqual(len(store.list_team_agents()), 500)
                self.assertIn(
                    "You definitely do not need that many agents.", gateway.posts[-1]["text"]
                )
            finally:
                store.close()

    def test_repo_root_command_updates_default_work_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                root = Path(tmp) / "projects"
                root.mkdir()
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_slash_command({"text": f"repo root {root}", "channel_id": "C1"})
                controller.handle_slash_command({"text": "show repo root", "channel_id": "C1"})

                self.assertEqual(store.get_setting("slack.repo_root"), str(root.resolve()))
                self.assertIn(f"Repo root set to `{root.resolve()}`.", gateway.posts[-2]["text"])
                self.assertIn(f"Repo root: `{root.resolve()}`", gateway.posts[-1]["text"])
            finally:
                store.close()

    def test_repo_root_command_rejects_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_slash_command(
                    {"text": f"repo root {Path(tmp) / 'missing'}", "channel_id": "C1"}
                )

                self.assertIsNone(store.get_setting("slack.repo_root"))
                self.assertIn("Use an existing local folder path.", gateway.posts[-1]["text"])
            finally:
                store.close()

    def test_setup_modal_rejects_more_than_500_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(store, gateway)

                response = controller.handle_view_submission(
                    {
                        "type": "view_submission",
                        "view": {
                            "callback_id": "setup.initial",
                            "state": {
                                "values": {
                                    "codex_count": {"value": {"value": "500"}},
                                    "claude_count": {"value": {"value": "1"}},
                                }
                            },
                        },
                    }
                )

                self.assertEqual(response["response_action"], "errors")
                self.assertIn("You definitely do not need that many agents.", str(response))
                self.assertEqual(store.list_team_agents(), [])
            finally:
                store.close()

    def test_usage_command_posts_then_updates_daily_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    home=Path(tmp),
                )

                controller.handle_slash_command({"text": "usage", "channel_id": "C1"})
                controller.handle_slash_command({"text": "usage", "channel_id": "C1"})

                self.assertEqual(len(gateway.posts), 1)
                self.assertEqual(len(gateway.updates), 3)
                self.assertEqual(
                    gateway.posts[0]["text"], ":hourglass_flowing_sand: Getting status..."
                )
                self.assertIn("Agent status", gateway.updates[-1]["text"])
            finally:
                store.close()

    def test_status_slash_command_posts_loading_then_updates_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    home=Path(tmp),
                )

                controller.handle_slash_command(
                    {"command": "/slackgentic-riley", "text": "status", "channel_id": "C1"}
                )

                self.assertEqual(len(gateway.posts), 1)
                self.assertEqual(
                    gateway.posts[0]["text"], ":hourglass_flowing_sand: Getting status..."
                )
                self.assertEqual(len(gateway.updates), 1)
                self.assertIn("Agent status", gateway.updates[0]["text"])
            finally:
                store.close()

    def test_slash_command_targets_configured_agent_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                store.set_setting("slack.channel_id", "CAGENTS")
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="CDEFAULT",
                    home=Path(tmp),
                )

                controller.handle_slash_command(
                    {"command": "/slackgentic-riley", "text": "status", "channel_id": "COTHER"}
                )

                self.assertEqual(gateway.posts[0]["channel_id"], "CAGENTS")
                self.assertEqual(gateway.updates[0]["channel_id"], "CAGENTS")
            finally:
                store.close()

    def test_channel_work_request_creates_thread_and_starts_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 1):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "Update the README",
                            "ts": "171.000001",
                        }
                    }
                )

                tasks = store.list_agent_tasks()
                self.assertEqual(len(tasks), 1)
                self.assertEqual(len(runtime.started), 1)
                self.assertEqual(tasks[0].thread_ts, "171.000001")
                self.assertEqual(gateway.thread_replies[0]["thread"].thread_ts, "171.000001")
                self.assertIn(("C1", "171.000001", "hourglass_flowing_sand"), gateway.reactions)
            finally:
                store.close()

    def test_repeated_slack_message_event_is_processed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                payload = {
                    "event": {
                        "type": "message",
                        "channel": "C1",
                        "user": "U1",
                        "text": "Update the README",
                        "ts": "171.000001",
                    }
                }

                controller.handle_event(payload)
                controller.handle_event(payload)

                self.assertEqual(len(store.list_agent_tasks()), 1)
                self.assertEqual(len(runtime.started), 1)
            finally:
                store.close()

    def test_slack_message_backfill_processes_channel_messages_after_sleep_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                store.set_setting(SETTING_SLACK_BACKFILL_LAST_AWAKE, "171.000000")
                gateway.history_messages.append(
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "U1",
                        "text": "Update docs after wake",
                        "ts": "171.000010",
                    }
                )
                backfill = SlackMessageBackfill(
                    store,
                    gateway,
                    controller,
                    team_id="local",
                    sleep_gap_seconds=5.0,
                    grace_seconds=0.0,
                    now=lambda: 181.0,
                )

                recovered = backfill.sync_once()

                self.assertEqual(recovered, 1)
                self.assertEqual(len(store.list_agent_tasks()), 1)
                self.assertEqual(store.list_agent_tasks()[0].prompt, "Update docs after wake")
                self.assertEqual(len(runtime.started), 1)
                self.assertEqual(store.get_setting(SETTING_SLACK_BACKFILL_LAST_AWAKE), "181.000000")
            finally:
                store.close()

    def test_slack_message_backfill_processes_known_thread_replies(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "Somebody summarize pyproject",
                            "ts": "171.000001",
                        }
                    }
                )
                parent_task = store.list_agent_tasks()[0]
                gateway.thread_history_messages[("C1", parent_task.thread_ts)] = [
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "U1",
                        "text": "continue with setup notes",
                        "ts": "171.000010",
                        "thread_ts": parent_task.thread_ts,
                    }
                ]
                backfill = SlackMessageBackfill(
                    store,
                    gateway,
                    controller,
                    team_id="local",
                    sleep_gap_seconds=5.0,
                    grace_seconds=0.0,
                    now=lambda: 181.0,
                )

                recovered = backfill.recover_since("171.000005")

                self.assertEqual(recovered, 1)
                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 1)
                self.assertEqual(tasks[0].prompt, "continue with setup notes")
                self.assertEqual(len(runtime.started), 2)
            finally:
                store.close()

    def test_channel_work_request_accepts_dangerous_mode_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "#dangerous-mode Update the README",
                            "ts": "171.000001",
                        }
                    }
                )

                tasks = store.list_agent_tasks()
                self.assertEqual(len(tasks), 1)
                self.assertEqual(tasks[0].prompt, "Update the README")
                self.assertTrue(tasks[0].metadata[DANGEROUS_MODE_METADATA_KEY])
                started_task, _, _ = runtime.started[0]
                self.assertTrue(started_task.metadata[DANGEROUS_MODE_METADATA_KEY])
                self.assertNotIn("#dangerous-mode", gateway.thread_replies[0]["text"])
            finally:
                store.close()

    def test_specific_busy_agent_request_says_to_wait_or_ask_someone_else(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{agent.handle} first task",
                            "ts": "171.000001",
                        }
                    }
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{agent.handle} second task",
                            "ts": "171.000002",
                        }
                    }
                )

                self.assertIn("specific agent is busy", gateway.thread_replies[-1]["text"])
                self.assertIn("ask someone else", gateway.thread_replies[-1]["text"])
                self.assertNotIn("hire more agents", gateway.thread_replies[-1]["text"])
                pending = store.list_pending_work_requests()
                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0].request.requested_handle, agent.handle)
            finally:
                store.close()

    def test_agent_can_reengage_in_same_thread_when_mentioned_by_another_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                cruz, dante = build_initial_model_team(1, 1)
                # Keep the names unimportant; the handles are what routing uses.
                cruz = replace(cruz, handle="cruz", provider_preference=Provider.CLAUDE)
                dante = replace(dante, handle="dante", provider_preference=Provider.CODEX)
                store.upsert_team_agent(cruz)
                store.upsert_team_agent(dante)
                original = create_agent_task(cruz, "first pass", "C1")
                original = replace(
                    original,
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CLAUDE,
                    session_id="claude-session-1",
                )
                store.upsert_agent_task(original)
                reviewer = create_agent_task(dante, "help @cruz", "C1")
                reviewer = replace(
                    reviewer,
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.thread",
                    parent_message_ts="171.review",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-1",
                )
                store.upsert_agent_task(reviewer)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                handled = controller.handle_runtime_agent_message(
                    reviewer,
                    dante,
                    SlackThreadRef("C1", "171.thread"),
                    "@cruz do this better",
                )

                self.assertTrue(handled)
                self.assertEqual(gateway.thread_replies, [])
                self.assertEqual(len(runtime.started), 1)
                task, agent, thread = runtime.started[0]
                self.assertEqual(task.task_id, original.task_id)
                self.assertIn("this better", task.prompt)
                self.assertEqual(agent.handle, "cruz")
                self.assertEqual(thread.thread_ts, "171.thread")
                self.assertEqual(task.session_provider, Provider.CLAUDE)
                self.assertEqual(task.session_id, "claude-session-1")
            finally:
                store.close()

    def test_agent_can_callback_assigned_external_session_agent_in_same_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                nell, asha = build_initial_model_team(1, 1)
                nell = replace(nell, handle="nell", provider_preference=Provider.CODEX)
                asha = replace(asha, handle="asha", provider_preference=Provider.CLAUDE)
                store.upsert_team_agent(nell)
                store.upsert_team_agent(asha)
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="codex-external",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    control_mode=ControlMode.OBSERVED,
                )
                thread = SlackThreadRef("C1", "171.thread", "171.parent")
                store.upsert_session(session)
                store.set_setting("external_session_agent.codex.codex-external", nell.agent_id)
                store.upsert_slack_thread_for_session(
                    Provider.CODEX,
                    "codex-external",
                    "T1",
                    thread,
                )
                helper_task = replace(
                    create_agent_task(
                        asha,
                        "tell @nell ideas of what to work on",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts=thread.thread_ts,
                    parent_message_ts="171.helper",
                    session_provider=Provider.CLAUDE,
                    session_id="claude-helper",
                    metadata={
                        "external_session_provider": Provider.CODEX.value,
                        "external_session_id": "codex-external",
                    },
                )
                store.upsert_agent_task(helper_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                    session_bridge=bridge,
                    team_id="T1",
                )

                handled = controller.handle_runtime_agent_message(
                    helper_task,
                    asha,
                    thread,
                    "@nell here are five project ideas",
                )

                self.assertTrue(handled)
                self.assertEqual(len(bridge.sent), 1)
                sent_session, prompt, sent_thread, slack_user = bridge.sent[0]
                self.assertEqual(sent_session.session_id, "codex-external")
                self.assertEqual(prompt, "here are five project ideas")
                self.assertEqual(sent_thread.thread_ts, "171.thread")
                self.assertEqual(slack_user, "U1")
                self.assertEqual(gateway.thread_replies, [])
                self.assertEqual(store.list_pending_work_requests(), [])
                self.assertEqual(len(store.list_agent_tasks(include_done=True)), 1)
            finally:
                store.close()

    def test_agent_can_callback_assigned_external_session_agent_on_final_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                nell, asha = build_initial_model_team(1, 1)
                nell = replace(nell, handle="nell", provider_preference=Provider.CODEX)
                asha = replace(asha, handle="asha", provider_preference=Provider.CLAUDE)
                store.upsert_team_agent(nell)
                store.upsert_team_agent(asha)
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="codex-external",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    control_mode=ControlMode.OBSERVED,
                )
                thread = SlackThreadRef("C1", "171.thread", "171.parent")
                store.upsert_session(session)
                store.set_setting("external_session_agent.codex.codex-external", nell.agent_id)
                store.upsert_slack_thread_for_session(
                    Provider.CODEX,
                    "codex-external",
                    "T1",
                    thread,
                )
                helper_task = replace(
                    create_agent_task(
                        asha,
                        "give nell options",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts=thread.thread_ts,
                    parent_message_ts="171.helper",
                    session_provider=Provider.CLAUDE,
                    session_id="claude-helper",
                    metadata={
                        "external_session_provider": Provider.CODEX.value,
                        "external_session_id": "codex-external",
                    },
                )
                store.upsert_agent_task(helper_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                    session_bridge=bridge,
                    team_id="T1",
                )

                handled = controller.handle_runtime_agent_message(
                    helper_task,
                    asha,
                    thread,
                    "Here are five project ideas.\n\n@nell - pick one before I proceed",
                )

                self.assertTrue(handled)
                self.assertEqual(len(bridge.sent), 1)
                sent_session, prompt, sent_thread, slack_user = bridge.sent[0]
                self.assertEqual(sent_session.session_id, "codex-external")
                self.assertEqual(
                    prompt,
                    "Here are five project ideas.\n\n@nell - pick one before I proceed",
                )
                self.assertEqual(sent_thread.thread_ts, "171.thread")
                self.assertEqual(slack_user, "U1")
                self.assertEqual(gateway.thread_replies, [])
                self.assertEqual(len(store.list_agent_tasks(include_done=True)), 1)
            finally:
                store.close()

    def test_agent_contextual_mention_of_external_session_agent_does_not_hijack_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 2)
                codex_agents = agents[:1]
                claude_agents = agents[1:]
                nell = replace(codex_agents[0], handle="nell", provider_preference=Provider.CODEX)
                asha = replace(claude_agents[0], handle="asha", provider_preference=Provider.CLAUDE)
                mika = replace(claude_agents[1], handle="mika", provider_preference=Provider.CLAUDE)
                for agent in (nell, asha, mika):
                    store.upsert_team_agent(agent)
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="codex-external",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    control_mode=ControlMode.OBSERVED,
                )
                thread = SlackThreadRef("C1", "171.thread", "171.parent")
                store.upsert_session(session)
                store.set_setting("external_session_agent.codex.codex-external", nell.agent_id)
                store.upsert_slack_thread_for_session(
                    Provider.CODEX,
                    "codex-external",
                    "T1",
                    thread,
                )
                helper_task = replace(
                    create_agent_task(
                        asha,
                        "find a helper for nell",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts=thread.thread_ts,
                    parent_message_ts="171.helper",
                )
                store.upsert_agent_task(helper_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                    session_bridge=bridge,
                    team_id="T1",
                )

                handled = controller.handle_runtime_agent_message(
                    helper_task,
                    asha,
                    thread,
                    "ask @mika to prepare options for @nell before we hand it back",
                )

                self.assertTrue(handled)
                self.assertEqual(bridge.sent, [])
                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                delegated = tasks[-1]
                self.assertEqual(delegated.agent_id, mika.agent_id)
                self.assertIn("prepare options for @nell", delegated.prompt)
            finally:
                store.close()

    def test_user_specific_same_thread_followup_uses_requested_agents_prior_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                original = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )
                requested = next(
                    agent for agent in agents if agent.provider_preference == Provider.CLAUDE
                )
                original = replace(original, handle="taylor")
                requested = replace(requested, handle="mina", full_name="Mina Adebayo")
                store.upsert_team_agent(original)
                store.upsert_team_agent(requested)
                original_task = replace(
                    create_agent_task(
                        original,
                        "tell me I'm pretty",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-original",
                )
                requested_prior_task = replace(
                    create_agent_task(
                        requested,
                        "make it better",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.thread",
                    parent_message_ts="171.mina",
                    session_provider=Provider.CLAUDE,
                    session_id="claude-mina",
                )
                store.upsert_agent_task(original_task)
                store.upsert_agent_task(requested_prior_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "@mina make it even better",
                            "ts": "171.000003",
                            "thread_ts": "171.thread",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                task, agent, thread = runtime.started[0]
                self.assertEqual(agent.handle, "mina")
                self.assertEqual(thread.thread_ts, "171.thread")
                self.assertEqual(task.session_provider, Provider.CLAUDE)
                self.assertEqual(task.session_id, "claude-mina")
                self.assertEqual(task.metadata["parent_task_id"], requested_prior_task.task_id)
                self.assertNotEqual(task.session_id, "codex-original")
            finally:
                store.close()

    def test_review_request_displays_full_command_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 1):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "Somebody review the repo",
                            "ts": "171.000001",
                        }
                    }
                )

                tasks = store.list_agent_tasks()
                self.assertEqual(tasks[0].prompt, "review the repo")
                rendered = gateway.thread_replies[0]["blocks"][0]["text"]["text"]
                self.assertIn("*Task:* review the repo", rendered)
            finally:
                store.close()

    def test_agent_avatar_base_url_is_used_for_agent_posts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                store.set_setting(
                    "slack.agent_avatar_base_url", "https://cdn.example.test/avatars/"
                )
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "Somebody do update the README",
                            "ts": "171.000001",
                        }
                    }
                )

                agent = store.list_team_agents()[0]
                self.assertEqual(
                    gateway.thread_replies[0]["icon_url"],
                    f"https://cdn.example.test/avatars/{agent.avatar_slug}.png",
                )
            finally:
                store.close()

    def test_agent_avatar_base_url_defaults_to_github_raw_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "Somebody do update the README",
                            "ts": "171.000001",
                        }
                    }
                )

                agent = store.list_team_agents()[0]
                self.assertEqual(
                    gateway.thread_replies[0]["icon_url"],
                    f"{DEFAULT_AGENT_AVATAR_BASE_URL}/{agent.avatar_slug}.png",
                )
            finally:
                store.close()

    def test_channel_work_request_accepts_other_bot_posting_mechanisms(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                    ignored_bot_id="BSLACKGENTIC",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "bot_id": "BPOSTER",
                            "text": "Somebody do update the README",
                            "ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(len(store.list_agent_tasks()), 1)
                self.assertEqual(len(runtime.started), 1)
            finally:
                store.close()

    def test_channel_work_request_ignores_own_bot_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                    ignored_bot_id="BSLACKGENTIC",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "bot_id": "BSLACKGENTIC",
                            "text": "Somebody do update the README",
                            "ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(store.list_agent_tasks(), [])
                self.assertEqual(runtime.started, [])
            finally:
                store.close()

    def test_task_thread_reply_is_sent_to_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "Somebody do update docs",
                            "ts": "171.000001",
                        }
                    }
                )
                task = store.list_agent_tasks()[0]

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "also include install notes",
                            "ts": "171.000002",
                            "thread_ts": task.thread_ts,
                        }
                    }
                )

                self.assertEqual(runtime.sent, [(task.task_id, "also include install notes")])
            finally:
                store.close()

    def test_status_in_task_thread_posts_usage_not_runtime_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                    home=Path(tmp),
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "Somebody do update docs",
                            "ts": "171.000001",
                        }
                    }
                )
                task = store.list_agent_tasks()[0]

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "/status",
                            "ts": "171.000002",
                            "thread_ts": task.thread_ts,
                        }
                    }
                )

                self.assertEqual(runtime.sent, [])
                self.assertEqual(
                    gateway.posts[-1]["text"], ":hourglass_flowing_sand: Getting status..."
                )
                self.assertIn("Agent status", gateway.updates[-1]["text"])
            finally:
                store.close()

    def test_external_session_thread_reply_is_sent_to_session_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    control_mode=ControlMode.OBSERVED,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(session)
                store.upsert_slack_thread_for_session(Provider.CLAUDE, "s1", "T1", thread)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    session_bridge=bridge,
                    team_id="T1",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "what are you working on?",
                            "ts": "171.000002",
                            "thread_ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(len(bridge.sent), 1)
                self.assertEqual(bridge.sent[0][0].session_id, "s1")
                self.assertEqual(bridge.sent[0][1], "what are you working on?")
                self.assertEqual(bridge.sent[0][3], "U1")
            finally:
                store.close()

    def test_external_session_thread_somebody_review_routes_to_claude_reviewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(codex_count=1, claude_count=1):
                    store.upsert_team_agent(agent)
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    control_mode=ControlMode.OBSERVED,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(session)
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                    session_bridge=bridge,
                    team_id="T1",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "sOmEbOdY ReVieW the repo cleanup plan",
                            "ts": "171.000002",
                            "thread_ts": "171.000001",
                        }
                    }
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 1)
                task = tasks[0]
                reviewer = store.get_team_agent(task.agent_id)
                self.assertEqual(task.thread_ts, "171.000001")
                self.assertEqual(task.kind.value, "review")
                self.assertEqual(reviewer.provider_preference, Provider.CLAUDE)
                self.assertEqual(task.metadata["external_session_provider"], Provider.CODEX.value)
                self.assertEqual(task.metadata["external_session_id"], "s1")
                self.assertEqual(len(runtime.started), 1)
                self.assertEqual(bridge.sent, [])
            finally:
                store.close()

    def test_external_session_thread_explicit_agent_request_routes_by_handle_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=1, claude_count=1)
                target_agent = replace(
                    agents[1],
                    agent_id="agent_another_agent_name",
                    handle="another-agent-name",
                )
                store.upsert_team_agent(agents[0])
                store.upsert_team_agent(target_agent)
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    control_mode=ControlMode.OBSERVED,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(session)
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                    session_bridge=bridge,
                    team_id="T1",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "@ANOTHER-AGENT-NAME do inspect the setup docs",
                            "ts": "171.000002",
                            "thread_ts": "171.000001",
                        }
                    }
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 1)
                self.assertEqual(tasks[0].agent_id, target_agent.agent_id)
                self.assertEqual(tasks[0].prompt, "inspect the setup docs")
                self.assertEqual(len(runtime.started), 1)
                self.assertEqual(runtime.started[0][1].handle, "another-agent-name")
                self.assertEqual(bridge.sent, [])
            finally:
                store.close()

    def test_external_session_thread_request_to_assigned_agent_goes_to_live_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agent = replace(
                    build_initial_model_team(codex_count=1, claude_count=0)[0], handle="nell"
                )
                store.upsert_team_agent(agent)
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    control_mode=ControlMode.OBSERVED,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(session)
                store.set_setting("external_session_agent.codex.s1", agent.agent_id)
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                    session_bridge=bridge,
                    team_id="T1",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "@nell pick one of these ideas",
                            "ts": "171.000002",
                            "thread_ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(len(bridge.sent), 1)
                sent_session, prompt, sent_thread, slack_user = bridge.sent[0]
                self.assertEqual(sent_session.session_id, "s1")
                self.assertEqual(prompt, "pick one of these ideas")
                self.assertEqual(sent_thread.thread_ts, "171.000001")
                self.assertEqual(slack_user, "U1")
                self.assertEqual(store.list_agent_tasks(include_done=True), [])
                self.assertEqual(store.list_pending_work_requests(), [])
                self.assertEqual(gateway.thread_replies, [])
            finally:
                store.close()

    def test_external_session_thread_command_text_forwards_literal_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    control_mode=ControlMode.OBSERVED,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(session)
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    session_bridge=bridge,
                    team_id="T1",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "command exit",
                            "ts": "171.000002",
                            "thread_ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(len(bridge.sent), 1)
                self.assertEqual(bridge.sent[0][0].session_id, "s1")
                self.assertEqual(bridge.sent[0][1], "command exit")
                self.assertEqual(bridge.sent[0][3], "U1")
            finally:
                store.close()

    def test_external_session_empty_command_text_forwards_literal_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    control_mode=ControlMode.OBSERVED,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(session)
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    session_bridge=bridge,
                    team_id="T1",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "command",
                            "ts": "171.000002",
                            "thread_ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(len(bridge.sent), 1)
                self.assertEqual(bridge.sent[0][1], "command")
                self.assertEqual(gateway.thread_replies, [])
            finally:
                store.close()

    def test_external_session_status_text_is_sent_to_session_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    control_mode=ControlMode.OBSERVED,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(session)
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    session_bridge=bridge,
                    team_id="T1",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "/status",
                            "ts": "171.000002",
                            "thread_ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(len(bridge.sent), 1)
                self.assertEqual(bridge.sent[0][1], "/status")
                self.assertEqual(gateway.posts, [])
            finally:
                store.close()

    def test_inactive_external_session_thread_reply_reserves_agent_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.DONE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(session)
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    session_bridge=bridge,
                    team_id="T1",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "revive this session",
                            "ts": "171.000002",
                            "thread_ts": thread.thread_ts,
                        }
                    }
                )

                self.assertEqual(len(bridge.sent), 1)
                self.assertEqual(bridge.sent[0][0].session_id, "s1")
                self.assertEqual(
                    store.get_setting("external_session_agent.codex.s1"),
                    agent.agent_id,
                )
            finally:
                store.close()

    def test_inactive_external_session_thread_reply_waits_without_matching_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            try:
                store.init_schema()
                for agent in build_initial_model_team(0, 1):
                    store.upsert_team_agent(agent)
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.DONE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(session)
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    session_bridge=bridge,
                    team_id="T1",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "revive this session",
                            "ts": "171.000002",
                            "thread_ts": thread.thread_ts,
                        }
                    }
                )

                self.assertEqual(bridge.sent, [])
                self.assertIn("No available codex agent", gateway.thread_replies[-1]["text"])
                self.assertIn("somebody ...", gateway.thread_replies[-1]["text"])
                self.assertIsNone(store.get_setting("external_session_agent.codex.s1"))
            finally:
                store.close()

    def test_inactive_external_session_thread_reply_does_not_steal_busy_external_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                active_session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="active",
                    transcript_path=Path(tmp) / "active.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                ended_session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="ended",
                    transcript_path=Path(tmp) / "ended.jsonl",
                    status=SessionStatus.DONE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(active_session)
                store.upsert_session(ended_session)
                store.set_setting("external_session_agent.codex.active", agent.agent_id)
                store.upsert_slack_thread_for_session(Provider.CODEX, "ended", "T1", thread)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    session_bridge=bridge,
                    team_id="T1",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "revive this session",
                            "ts": "171.000002",
                            "thread_ts": thread.thread_ts,
                        }
                    }
                )

                self.assertEqual(bridge.sent, [])
                self.assertIsNone(store.get_setting("external_session_agent.codex.ended"))
                self.assertIn("No available codex agent", gateway.thread_replies[-1]["text"])
            finally:
                store.close()

    def test_mirrored_slack_message_is_not_forwarded_back_to_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            bridge = FakeSessionBridge()
            try:
                store.init_schema()
                store.mark_slack_message_mirrored("C1", "171.000002", "codex:s1:user")
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    session_bridge=bridge,
                    team_id="T1",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "local terminal prompt",
                            "ts": "171.000002",
                            "thread_ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(bridge.sent, [])
            finally:
                store.close()

    def test_thread_somebody_request_starts_cross_model_task_with_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                codex_agent = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{codex_agent.handle} summarize pyproject",
                            "ts": "171.000001",
                        }
                    }
                )
                parent_task = store.list_agent_tasks(include_done=True)[0]
                store.update_agent_task_status(parent_task.task_id, AgentTaskStatus.DONE)

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "somebody review this answer",
                            "ts": "171.000002",
                            "thread_ts": parent_task.thread_ts,
                        }
                    }
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                followup = tasks[-1]
                reviewer = store.get_team_agent(followup.agent_id)
                self.assertEqual(reviewer.provider_preference, Provider.CLAUDE)
                self.assertEqual(followup.thread_ts, parent_task.thread_ts)
                self.assertEqual(followup.metadata["parent_task_id"], parent_task.task_id)
                self.assertIn("summarize pyproject", followup.metadata["thread_context"])
                self.assertEqual(len(runtime.started), 2)

                store.update_agent_task_status(followup.task_id, AgentTaskStatus.DONE)
                gateway.post_thread_reply(
                    runtime.started[-1][2],
                    "review feedback: include the package version",
                    persona=reviewer,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "apply that review feedback",
                            "ts": "171.000003",
                            "thread_ts": parent_task.thread_ts,
                        }
                    }
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                continued = store.get_agent_task(parent_task.task_id)
                self.assertIsNotNone(continued)
                self.assertEqual(continued.agent_id, parent_task.agent_id)
                self.assertEqual(continued.status, AgentTaskStatus.ACTIVE)
                self.assertEqual(continued.prompt, "apply that review feedback")
                self.assertIn("review feedback", continued.metadata["thread_context"])
            finally:
                store.close()

    def test_thread_somebody_mentions_parent_agent_uses_other_available_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                original = next(
                    agent for agent in agents if agent.provider_preference == Provider.CLAUDE
                )
                other = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{original.handle} answer casually",
                            "ts": "171.000001",
                        }
                    }
                )
                parent_task = store.list_agent_tasks(include_done=True)[0]
                store.update_agent_task_status(parent_task.task_id, AgentTaskStatus.DONE)

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"somebody teach @{original.handle} how to do it better",
                            "ts": "171.000002",
                            "thread_ts": parent_task.thread_ts,
                        }
                    }
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                self.assertEqual(tasks[-1].agent_id, other.agent_id)
                self.assertIn(f"teach @{original.handle}", tasks[-1].prompt)
                self.assertEqual(store.list_pending_work_requests(), [])
            finally:
                store.close()

    def test_thread_somebody_mentions_parent_agent_queues_when_other_agent_busy(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                original = next(
                    agent for agent in agents if agent.provider_preference == Provider.CLAUDE
                )
                other = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CODEX,
                        session_id="busy-codex",
                        transcript_path=Path(tmp) / "codex.jsonl",
                        status=SessionStatus.ACTIVE,
                    )
                )
                store.set_setting(
                    "external_session_agent.codex.busy-codex",
                    other.agent_id,
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{original.handle} answer casually",
                            "ts": "171.000001",
                        }
                    }
                )
                parent_task = store.list_agent_tasks(include_done=True)[0]
                store.update_agent_task_status(parent_task.task_id, AgentTaskStatus.DONE)

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"somebody teach @{original.handle} how to do it better",
                            "ts": "171.000002",
                            "thread_ts": parent_task.thread_ts,
                        }
                    }
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 1)
                pending = store.list_pending_work_requests()
                self.assertEqual(len(pending), 1)
                self.assertIn(original.agent_id, pending[0].exclude_agent_ids)
                self.assertNotIn(other.agent_id, pending[0].exclude_agent_ids)
                self.assertIn(f"teach @{original.handle}", pending[0].request.prompt)
                self.assertIn("No agents are available", gateway.thread_replies[-1]["text"])
            finally:
                store.close()

    def test_thread_agent_can_delegate_task_back_to_original_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                codex_agent = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{codex_agent.handle} summarize pyproject",
                            "ts": "171.000001",
                        }
                    }
                )
                parent_task = store.list_agent_tasks(include_done=True)[0]
                store.update_agent_task_status(parent_task.task_id, AgentTaskStatus.DONE)

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "somebody ask the original agent to reply with DELEGATED_OK",
                            "ts": "171.000002",
                            "thread_ts": parent_task.thread_ts,
                        }
                    }
                )

                delegated_by_reviewer = store.list_agent_tasks(include_done=True)[-1]
                reviewer = store.get_team_agent(delegated_by_reviewer.agent_id)
                self.assertEqual(
                    delegated_by_reviewer.metadata["delegate_to_agent_id"],
                    parent_task.agent_id,
                )
                self.assertEqual(
                    delegated_by_reviewer.metadata["delegate_prompt"],
                    "reply with DELEGATED_OK",
                )
                gateway.post_thread_reply(
                    runtime.started[-1][2],
                    "I am asking the original agent to reply with DELEGATED_OK.",
                    persona=reviewer,
                )

                controller.handle_runtime_task_done(
                    delegated_by_reviewer,
                    reviewer,
                    SlackThreadRef("C1", parent_task.thread_ts),
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                self.assertEqual(len(runtime.started), 3)
                continued_task = runtime.started[-1][0]
                self.assertEqual(continued_task.task_id, parent_task.task_id)
                self.assertEqual(continued_task.agent_id, parent_task.agent_id)
                self.assertEqual(continued_task.prompt, "reply with DELEGATED_OK")
                self.assertIn("DELEGATED_OK", continued_task.metadata["thread_context"])
                self.assertIn(
                    f"@{codex_agent.handle}, please reply with DELEGATED_OK",
                    gateway.thread_replies[-1]["text"],
                )
                self.assertNotIn("for <@U1>", gateway.thread_replies[-1]["text"])
            finally:
                store.close()

    def test_thread_agent_can_delegate_task_to_explicit_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 2)
                for agent in agents:
                    store.upsert_team_agent(agent)
                codex_agent = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )
                target_agent = next(
                    agent for agent in agents if agent.provider_preference == Provider.CLAUDE
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{codex_agent.handle} reply with BASE_OK",
                            "ts": "171.000001",
                        }
                    }
                )
                parent_task = store.list_agent_tasks(include_done=True)[0]
                store.update_agent_task_status(parent_task.task_id, AgentTaskStatus.DONE)

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"somebody give @{target_agent.handle} some work to do",
                            "ts": "171.000002",
                            "thread_ts": parent_task.thread_ts,
                        }
                    }
                )

                delegated_by_reviewer = store.list_agent_tasks(include_done=True)[-1]
                reviewer = store.get_team_agent(delegated_by_reviewer.agent_id)
                self.assertNotEqual(reviewer.agent_id, target_agent.agent_id)
                self.assertEqual(
                    delegated_by_reviewer.metadata["delegate_to_agent_id"],
                    target_agent.agent_id,
                )
                gateway.post_thread_reply(
                    runtime.started[-1][2],
                    f"@{target_agent.handle}, please reply with NAMED_DELEGATION_OK.",
                    persona=reviewer,
                )

                controller.handle_runtime_task_done(
                    delegated_by_reviewer,
                    reviewer,
                    SlackThreadRef("C1", parent_task.thread_ts),
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 3)
                self.assertEqual(tasks[-1].agent_id, target_agent.agent_id)
                self.assertIn(f"@{reviewer.handle} assigned", tasks[-1].prompt)
                self.assertIn("NAMED_DELEGATION_OK", tasks[-1].metadata["thread_context"])
                self.assertEqual(len(runtime.started), 3)
                self.assertIn(
                    f"@{target_agent.handle}, please take the task I assigned above.",
                    gateway.thread_replies[-2]["text"],
                )
            finally:
                store.close()

    def test_agent_authored_somebody_review_starts_reviewer_and_returns_to_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                original = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{original.handle} make a risky change",
                            "ts": "171.000001",
                        }
                    }
                )
                parent_task = store.list_agent_tasks(include_done=True)[0]

                handled = controller.handle_runtime_agent_message(
                    parent_task,
                    original,
                    SlackThreadRef("C1", parent_task.thread_ts),
                    "somebody review the proposed change before I proceed",
                )

                self.assertTrue(handled)
                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                review_task = tasks[-1]
                reviewer = store.get_team_agent(review_task.agent_id)
                self.assertEqual(reviewer.provider_preference, Provider.CLAUDE)
                self.assertEqual(review_task.metadata["delegate_to_agent_id"], original.agent_id)
                self.assertIn(
                    "Continue the original task",
                    review_task.metadata["delegate_prompt"],
                )
                self.assertEqual(len(runtime.started), 2)
                gateway.post_thread_reply(
                    runtime.started[-1][2],
                    "Review: proceed, but mention test coverage.",
                    persona=reviewer,
                )

                controller.handle_runtime_task_done(
                    review_task,
                    reviewer,
                    SlackThreadRef("C1", parent_task.thread_ts),
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                followup = runtime.started[-1][0]
                self.assertEqual(followup.task_id, parent_task.task_id)
                self.assertEqual(followup.agent_id, original.agent_id)
                self.assertIn("Continue the original task", followup.prompt)
                self.assertIn("test coverage", followup.metadata["thread_context"])
            finally:
                store.close()

    def test_give_it_back_to_named_agent_survives_review_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 2)
                claude_agents = [
                    agent for agent in agents if agent.provider_preference == Provider.CLAUDE
                ]
                codex_agent = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )
                mika = replace(claude_agents[0], handle="mika", full_name="Mika Dlamini")
                owen = replace(claude_agents[1], handle="owen", full_name="Owen Kowalski")
                evan = replace(codex_agent, handle="evan", full_name="Evan Silva")
                for agent in (mika, owen, evan):
                    store.upsert_team_agent(agent)
                original_task = replace(
                    create_agent_task(
                        mika,
                        "hi",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CLAUDE,
                    session_id="claude-mika",
                )
                store.upsert_agent_task(original_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": (
                                "somebody help @mika do a better job at a greeting. "
                                "Then ask somebody else to improve it further. "
                                "Then give it back to mika"
                            ),
                            "ts": "171.000002",
                            "thread_ts": "171.thread",
                        }
                    }
                )

                helper_task = store.list_agent_tasks(include_done=True)[-1]
                helper = store.get_team_agent(helper_task.agent_id)
                self.assertIsNotNone(helper)
                self.assertNotEqual(helper_task.agent_id, mika.agent_id)
                self.assertEqual(helper_task.metadata["delegate_to_agent_id"], mika.agent_id)
                self.assertIn("Slack thread context", helper_task.metadata["delegate_prompt"])

                handled = controller.handle_runtime_agent_message(
                    helper_task,
                    helper,
                    SlackThreadRef("C1", "171.thread"),
                    "somebody review this improved greeting for @mika",
                )

                self.assertTrue(handled)
                review_task = store.list_agent_tasks(include_done=True)[-1]
                reviewer = store.get_team_agent(review_task.agent_id)
                self.assertIsNotNone(reviewer)
                self.assertEqual(review_task.metadata["delegate_to_agent_id"], mika.agent_id)
                self.assertIn("Slack thread context", review_task.metadata["delegate_prompt"])

                reviewer_text = "Here's a warmer version.\n\n@mika - over to you."
                gateway.post_thread_reply(
                    SlackThreadRef("C1", "171.thread"),
                    reviewer_text,
                    persona=reviewer,
                )
                handled = controller.handle_runtime_agent_message(
                    review_task,
                    reviewer,
                    SlackThreadRef("C1", "171.thread"),
                    reviewer_text,
                )

                self.assertFalse(handled)
                self.assertEqual(len(store.list_agent_tasks(include_done=True)), 3)

                controller.handle_runtime_task_done(
                    review_task,
                    reviewer,
                    SlackThreadRef("C1", "171.thread"),
                )

                self.assertEqual(len(store.list_agent_tasks(include_done=True)), 3)
                continued_task, continued_agent, _ = runtime.started[-1]
                self.assertEqual(continued_task.task_id, original_task.task_id)
                self.assertEqual(continued_agent.agent_id, mika.agent_id)
                self.assertIn("Continue the original task", continued_task.prompt)
                self.assertIn("warmer version", continued_task.metadata["thread_context"])
            finally:
                store.close()

    def test_agent_authored_review_after_newline_starts_reviewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                original = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )
                reviewer = next(
                    agent for agent in agents if agent.provider_preference == Provider.CLAUDE
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                parent_task = replace(
                    create_agent_task(
                        original,
                        "tell me I'm pretty in two parts",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    thread_ts="171.thread",
                    parent_message_ts="171.000001",
                )
                store.upsert_agent_task(parent_task)

                handled = controller.handle_runtime_agent_message(
                    parent_task,
                    original,
                    SlackThreadRef("C1", "171.thread"),
                    (
                        "Part one: you're pretty in a quiet, specific way.\n\n"
                        "somebody review my part one above and add part two, "
                        f"then hand it back to @{original.handle} for the closer."
                    ),
                )

                self.assertTrue(handled)
                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                review_task = tasks[-1]
                self.assertEqual(review_task.agent_id, reviewer.agent_id)
                self.assertEqual(review_task.metadata["delegate_to_agent_id"], original.agent_id)
                self.assertEqual(
                    review_task.prompt,
                    (
                        "review my part one above and add part two, "
                        f"then hand it back to @{original.handle} for the closer"
                    ),
                )
                self.assertEqual(len(runtime.started), 1)
                self.assertEqual(runtime.started[0][1].agent_id, reviewer.agent_id)
            finally:
                store.close()

    def test_agent_authored_review_mention_does_not_double_delegate_to_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                original = next(
                    agent for agent in agents if agent.provider_preference == Provider.CLAUDE
                )
                reviewer = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{original.handle} answer casually",
                            "ts": "171.000001",
                        }
                    }
                )
                parent_task = store.list_agent_tasks(include_done=True)[0]
                handled = controller.handle_runtime_agent_message(
                    parent_task,
                    original,
                    SlackThreadRef("C1", parent_task.thread_ts),
                    "somebody review my reply",
                )
                self.assertTrue(handled)
                review_task = store.list_agent_tasks(include_done=True)[-1]
                self.assertEqual(review_task.agent_id, reviewer.agent_id)

                handled = controller.handle_runtime_agent_message(
                    review_task,
                    reviewer,
                    SlackThreadRef("C1", parent_task.thread_ts),
                    f"@{original.handle} the issue was the tone. Try a warmer rewrite.",
                )

                self.assertFalse(handled)
                self.assertEqual(len(store.list_agent_tasks(include_done=True)), 2)
                gateway.post_thread_reply(
                    runtime.started[-1][2],
                    f"@{original.handle} the issue was the tone. Try a warmer rewrite.",
                    persona=reviewer,
                )
                controller.handle_runtime_task_done(
                    review_task,
                    reviewer,
                    SlackThreadRef("C1", parent_task.thread_ts),
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                followup = runtime.started[-1][0]
                self.assertEqual(followup.task_id, parent_task.task_id)
                self.assertEqual(followup.agent_id, original.agent_id)
                self.assertIn("Continue the original task", followup.prompt)
            finally:
                store.close()

    def test_agent_authored_non_review_request_is_not_control_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 1):
                    store.upsert_team_agent(agent)
                original = next(
                    agent
                    for agent in store.list_team_agents()
                    if agent.provider_preference == Provider.CODEX
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{original.handle} make a risky change",
                            "ts": "171.000001",
                        }
                    }
                )
                parent_task = store.list_agent_tasks(include_done=True)[0]

                handled = controller.handle_runtime_agent_message(
                    parent_task,
                    original,
                    SlackThreadRef("C1", parent_task.thread_ts),
                    "somebody do the follow-up later",
                )

                self.assertFalse(handled)
                self.assertEqual(len(store.list_agent_tasks(include_done=True)), 1)
            finally:
                store.close()

    def test_agent_authored_specific_request_routes_to_named_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                sender = next(
                    agent for agent in agents if agent.provider_preference == Provider.CLAUDE
                )
                target = next(
                    agent for agent in agents if agent.provider_preference == Provider.CODEX
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                parent_task = create_agent_task(
                    sender,
                    "review the change",
                    "C1",
                    requested_by_slack_user="U1",
                )
                parent_task = replace(parent_task, thread_ts="171.000001")
                store.upsert_agent_task(parent_task)

                handled = controller.handle_runtime_agent_message(
                    parent_task,
                    sender,
                    SlackThreadRef("C1", "171.000001"),
                    f"@{target.handle} please update the patch based on my review",
                )

                self.assertTrue(handled)
                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                delegated = tasks[-1]
                self.assertEqual(delegated.agent_id, target.agent_id)
                self.assertIn("update the patch", delegated.prompt)
                self.assertEqual(len(runtime.started), 1)
            finally:
                store.close()

    def test_agent_authored_specific_request_routes_unique_alias_to_active_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                base = build_initial_model_team(1, 1)
                old_mina = replace(
                    base[0],
                    agent_id="agent_old_mina",
                    handle="mina",
                    full_name="Mina Adebayo",
                    provider_preference=Provider.CLAUDE,
                )
                mina = replace(
                    base[0],
                    agent_id="agent_minaa",
                    handle="minaa",
                    full_name="Mina Adebayo",
                    provider_preference=Provider.CLAUDE,
                )
                taylor = replace(
                    base[1],
                    agent_id="agent_taylor",
                    handle="taylor",
                    full_name="Taylor Nielsen",
                    provider_preference=Provider.CODEX,
                )
                store.upsert_team_agent(old_mina)
                store.fire_team_agent("mina")
                store.upsert_team_agent(mina)
                store.upsert_team_agent(taylor)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                mina_task = replace(
                    create_agent_task(
                        mina,
                        "hi tell me I'm pretty",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    thread_ts="171.thread",
                    parent_message_ts="171.000001",
                    session_provider=Provider.CLAUDE,
                    session_id="claude-session-mina",
                )
                review_task = replace(
                    create_agent_task(
                        taylor,
                        "help @minaa say it in a more believable way",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    thread_ts="171.thread",
                    parent_message_ts="171.000002",
                )
                store.upsert_agent_task(mina_task)
                store.upsert_agent_task(review_task)

                handled = controller.handle_runtime_agent_message(
                    review_task,
                    taylor,
                    SlackThreadRef("C1", "171.thread"),
                    "@mina try: say it like a person, not a template",
                )

                self.assertTrue(handled)
                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                self.assertEqual(len(runtime.started), 1)
                followup, agent, _ = runtime.started[0]
                self.assertEqual(followup.task_id, mina_task.task_id)
                self.assertEqual(followup.agent_id, mina.agent_id)
                self.assertEqual(followup.session_id, "claude-session-mina")
                self.assertIn("say it like a person", followup.prompt)
                self.assertEqual(agent.handle, "minaa")
            finally:
                store.close()

    def test_thread_anyone_request_canonicalizes_unique_agent_alias_for_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                base = build_initial_model_team(1, 1)
                mina = replace(
                    base[0],
                    agent_id="agent_minaa",
                    handle="minaa",
                    full_name="Mina Adebayo",
                    provider_preference=Provider.CLAUDE,
                )
                taylor = replace(
                    base[1],
                    agent_id="agent_taylor",
                    handle="taylor",
                    full_name="Taylor Nielsen",
                    provider_preference=Provider.CODEX,
                )
                store.upsert_team_agent(mina)
                store.upsert_team_agent(taylor)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                parent_task = replace(
                    create_agent_task(
                        mina,
                        "hi tell me I'm pretty",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    thread_ts="171.thread",
                    parent_message_ts="171.000001",
                )
                store.upsert_agent_task(parent_task)

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "somebody help @mina say it in a more believable way",
                            "ts": "171.000003",
                            "thread_ts": "171.thread",
                        }
                    }
                )

                helper_task = store.list_agent_tasks(include_done=True)[-1]
                self.assertEqual(helper_task.agent_id, taylor.agent_id)
                self.assertIn("@minaa", helper_task.prompt)
                self.assertEqual(len(runtime.started), 1)
                self.assertEqual(runtime.started[0][1].handle, "taylor")
            finally:
                store.close()

    def test_agent_authored_multi_way_handoff_starts_each_named_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                base = build_initial_model_team(2, 1)
                sender = replace(base[0], agent_id="agent_sender", handle="taylor")
                mina = replace(
                    base[1],
                    agent_id="agent_minaa",
                    handle="minaa",
                    full_name="Mina Adebayo",
                    provider_preference=Provider.CLAUDE,
                )
                cruz = replace(base[2], agent_id="agent_cruz", handle="cruz")
                for agent in (sender, mina, cruz):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                sender_task = replace(
                    create_agent_task(sender, "coordinate feedback", "C1"),
                    thread_ts="171.thread",
                    parent_message_ts="171.000001",
                )
                store.upsert_agent_task(sender_task)

                handled = controller.handle_runtime_agent_message(
                    sender_task,
                    sender,
                    SlackThreadRef("C1", "171.thread"),
                    "@mina and @cruz review this answer before I send it",
                )

                self.assertTrue(handled)
                started_handles = [agent.handle for _, agent, _ in runtime.started]
                self.assertEqual(started_handles, ["minaa", "cruz"])
                prompts = [task.prompt for task, _, _ in runtime.started]
                self.assertEqual(prompts, ["review this answer before I send it"] * 2)
            finally:
                store.close()

    def test_plain_thread_reply_after_completion_continues_original_agent_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 1):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "#dangerous-mode Somebody summarize pyproject",
                            "ts": "171.000001",
                        }
                    }
                )
                parent_task = store.list_agent_tasks(include_done=True)[0]
                self.assertTrue(parent_task.metadata[DANGEROUS_MODE_METADATA_KEY])
                store.update_agent_task_status(parent_task.task_id, AgentTaskStatus.DONE)
                resolved_task = store.get_agent_task(parent_task.task_id)
                self.assertIsNotNone(resolved_task)
                controller._remove_task_action_buttons_if_resolved(resolved_task)
                gateway.updates.clear()
                gateway.thread_replies.clear()

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "let's do it!",
                            "ts": "171.000002",
                            "thread_ts": parent_task.thread_ts,
                        }
                    }
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 1)
                self.assertEqual(tasks[0].task_id, parent_task.task_id)
                self.assertEqual(tasks[0].agent_id, parent_task.agent_id)
                self.assertEqual(tasks[0].prompt, "let's do it!")
                self.assertEqual(tasks[0].status, AgentTaskStatus.ACTIVE)
                self.assertEqual(tasks[0].thread_ts, parent_task.thread_ts)
                self.assertTrue(tasks[0].metadata[DANGEROUS_MODE_METADATA_KEY])
                self.assertIn(
                    "Original task: summarize pyproject",
                    tasks[0].metadata["thread_context"],
                )
                self.assertEqual(len(runtime.started), 2)
                self.assertEqual(runtime.started[-1][0].task_id, parent_task.task_id)
                self.assertEqual(len(gateway.updates), 1)
                update = gateway.updates[0]
                self.assertEqual(update["channel_id"], "C1")
                self.assertEqual(update["ts"], parent_task.parent_message_ts)
                self.assertIn("summarize pyproject", update["text"])
                self.assertNotIn("let's do it", update["text"])
                actions = [block for block in update["blocks"] if block.get("type") == "actions"]
                self.assertEqual(len(actions), 1)
                self.assertEqual(
                    actions[0]["elements"][0]["text"]["text"],
                    "Finish and free up this agent",
                )
                self.assertNotIn(
                    "I'll take this",
                    "\n".join(reply["text"] for reply in gateway.thread_replies),
                )

                store.update_agent_task_status(parent_task.task_id, AgentTaskStatus.DONE)
                continued_task = store.get_agent_task(parent_task.task_id)
                self.assertIsNotNone(continued_task)
                gateway.updates.clear()

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "try again",
                            "ts": "171.000003",
                            "thread_ts": parent_task.thread_ts,
                        }
                    }
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 1)
                self.assertEqual(tasks[0].task_id, parent_task.task_id)
                self.assertEqual(tasks[0].prompt, "try again")
                self.assertEqual(tasks[0].status, AgentTaskStatus.ACTIVE)
                thread_context = tasks[0].metadata["thread_context"]
                self.assertIn("Original task: summarize pyproject", thread_context)
                self.assertNotIn("Original task: let's do it!", thread_context)
                self.assertEqual(len(runtime.started), 3)
                self.assertEqual(len(gateway.updates), 1)
                self.assertIn("summarize pyproject", gateway.updates[0]["text"])
                self.assertNotIn("try again", gateway.updates[0]["text"])
            finally:
                store.close()

    def test_roster_command_in_task_thread_is_not_sent_to_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "Somebody do update docs",
                            "ts": "171.000001",
                        }
                    }
                )
                task = store.list_agent_tasks()[0]

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "show roster",
                            "ts": "171.000002",
                            "thread_ts": task.thread_ts,
                        }
                    }
                )

                self.assertEqual(runtime.sent, [])
                self.assertEqual(gateway.posts[-1]["thread_ts"], task.thread_ts)
                self.assertIn("Agent roster", gateway.posts[-1]["text"])
            finally:
                store.close()

    def test_active_thread_ownership_survives_startup_reconcile(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "Somebody do update docs",
                            "ts": "171.000001",
                        }
                    }
                )
                task = store.list_agent_tasks()[0]
                store.update_agent_task_status(task.task_id, AgentTaskStatus.ACTIVE)

                cancelled = controller.cancel_orphaned_active_tasks()

                self.assertEqual(cancelled, 0)
                self.assertEqual(
                    store.get_agent_task(task.task_id).status,
                    AgentTaskStatus.ACTIVE,
                )
            finally:
                store.close()

    def test_plain_reply_to_persisted_active_task_restarts_same_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "summarize pyproject", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-1",
                )
                store.upsert_agent_task(task)
                store.upsert_managed_thread_task(task, SlackThreadRef("C1", "171.thread"))
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "also include the package version",
                            "ts": "171.000002",
                            "thread_ts": "171.thread",
                        }
                    }
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 1)
                self.assertEqual(runtime.sent, [(task.task_id, "also include the package version")])
                self.assertEqual(len(runtime.started), 1)
                restarted = runtime.started[0][0]
                self.assertEqual(restarted.task_id, task.task_id)
                self.assertEqual(restarted.session_id, "codex-thread-1")
                self.assertEqual(restarted.prompt, "also include the package version")
                current = store.get_agent_task(task.task_id)
                self.assertIsNotNone(current)
                assert current is not None
                self.assertEqual(current.prompt, "also include the package version")
            finally:
                store.close()

    def test_task_thread_dependency_link_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "Somebody do update docs",
                            "ts": "171.000001",
                        }
                    }
                )
                task = store.list_agent_tasks()[0]

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": (
                                "wait for this to go in "
                                "https://example.slack.com/archives/C2/p1712345678901234"
                            ),
                            "ts": "171.000002",
                            "thread_ts": task.thread_ts,
                        }
                    }
                )

                self.assertEqual(len(store.dependencies_for(task.task_id)), 1)
                self.assertEqual(runtime.sent, [])
            finally:
                store.close()

    def test_task_button_stops_runtime_and_updates_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "Somebody do update docs",
                            "ts": "171.000001",
                        }
                    }
                )
                task = store.list_agent_tasks()[0]

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": task.thread_ts},
                        "actions": [
                            {"value": encode_action_value("task.done", task_id=task.task_id)}
                        ],
                    }
                )

                self.assertEqual(runtime.stopped, [(task.task_id, AgentTaskStatus.DONE)])
                self.assertIn(
                    ("C1", task.thread_ts, "hourglass_flowing_sand"),
                    gateway.removed_reactions,
                )
                self.assertIn(("C1", task.thread_ts, "eyes"), gateway.removed_reactions)
                self.assertIn(("C1", task.thread_ts, "white_check_mark"), gateway.reactions)
                self.assertIn(
                    "Finished and freed up this agent.", gateway.thread_replies[-1]["text"]
                )
            finally:
                store.close()

    def test_task_done_button_frees_all_same_thread_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                first = replace(
                    create_agent_task(agent, "first pass", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.000001",
                    parent_message_ts="171.000001",
                )
                second = replace(
                    create_agent_task(agent, "follow-up", "C1"),
                    status=AgentTaskStatus.QUEUED,
                    thread_ts="171.000001",
                    parent_message_ts="171.000002",
                )
                store.upsert_agent_task(first)
                store.upsert_agent_task(second)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.000001"},
                        "actions": [
                            {"value": encode_action_value("task.done", task_id=first.task_id)}
                        ],
                    }
                )

                statuses = {
                    task.task_id: task.status for task in store.list_agent_tasks(include_done=True)
                }
                self.assertEqual(statuses[first.task_id], AgentTaskStatus.DONE)
                self.assertEqual(statuses[second.task_id], AgentTaskStatus.DONE)
                self.assertEqual(
                    {update["ts"] for update in gateway.updates},
                    {"171.000001", "171.000002"},
                )
                for update in gateway.updates:
                    self.assertFalse(
                        any(block.get("type") == "actions" for block in update["blocks"])
                    )
                self.assertIn(
                    "Finished and freed up this agent.", gateway.thread_replies[-1]["text"]
                )
            finally:
                store.close()

    def test_agent_thread_done_signal_frees_whole_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                first_agent, second_agent = build_initial_model_team(1, 1)
                store.upsert_team_agent(first_agent)
                store.upsert_team_agent(second_agent)
                first = replace(
                    create_agent_task(first_agent, "initial task", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={"request_message_ts": "171.user"},
                )
                second = replace(
                    create_agent_task(second_agent, "follow-up task", "C1"),
                    status=AgentTaskStatus.QUEUED,
                    thread_ts="171.thread",
                    parent_message_ts="171.bot2",
                    metadata={"request_message_ts": "171.user2"},
                )
                outside = replace(
                    create_agent_task(second_agent, "other thread", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.other",
                    parent_message_ts="171.other",
                )
                store.upsert_agent_task(first)
                store.upsert_agent_task(second)
                store.upsert_agent_task(outside)
                pending = store.create_pending_work_request(
                    SlackThreadRef("C1", "171.thread", "171.pending"),
                    WorkRequest(
                        prompt="queued same thread",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=first_agent.handle,
                    ),
                    requested_by_slack_user="U1",
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                handled = controller.handle_runtime_agent_control(
                    first,
                    first_agent,
                    SlackThreadRef("C1", "171.thread"),
                    AGENT_THREAD_DONE_SIGNAL,
                )

                self.assertTrue(handled)
                statuses = {
                    task.task_id: task.status for task in store.list_agent_tasks(include_done=True)
                }
                self.assertEqual(statuses[first.task_id], AgentTaskStatus.DONE)
                self.assertEqual(statuses[second.task_id], AgentTaskStatus.DONE)
                self.assertEqual(statuses[outside.task_id], AgentTaskStatus.ACTIVE)
                row = store.conn.execute(
                    "SELECT status FROM pending_work_requests WHERE pending_id = ?",
                    (pending.pending_id,),
                ).fetchone()
                self.assertEqual(row["status"], "cancelled")
                self.assertEqual(
                    {update["ts"] for update in gateway.updates},
                    {"171.parent", "171.bot2"},
                )
                for update in gateway.updates:
                    self.assertFalse(
                        any(block.get("type") == "actions" for block in update["blocks"])
                    )
                self.assertIn(("C1", "171.thread", "white_check_mark"), gateway.reactions)
                self.assertIn(("C1", "171.user2", "white_check_mark"), gateway.reactions)
            finally:
                store.close()

    def test_unknown_agent_control_signal_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "initial task", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                )
                store.upsert_agent_task(task)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                handled = controller.handle_runtime_agent_control(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.thread"),
                    "SLACKGENTIC: UNKNOWN",
                )

                self.assertFalse(handled)
                self.assertEqual(
                    store.get_agent_task(task.task_id).status,
                    AgentTaskStatus.ACTIVE,
                )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
