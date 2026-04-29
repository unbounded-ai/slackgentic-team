import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent_harness.models import (
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
    DEFAULT_AGENT_AVATAR_BASE_URL,
    SETTING_ROSTER_TS,
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

    def thread_messages(self, channel_id, thread_ts, limit=20):
        messages = [
            {
                "username": getattr(item.get("persona"), "full_name", None),
                "text": item["text"],
            }
            for item in self.thread_replies
            if item["thread"].channel_id == channel_id and item["thread"].thread_ts == thread_ts
        ]
        parents = [
            {"username": None, "text": item["text"]}
            for item in self.posts
            if item["channel_id"] == channel_id and item["ts"] == thread_ts
        ]
        return (parents + messages)[-limit:]


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
                            "text": "write a tiny validation note",
                            "ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(len(store.list_pending_work_requests()), 1)
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
                self.assertIn("Codex outside Slack", text)
                self.assertIn("Claude outside Slack", text)
                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn("Write anything in this channel", blocks)
                self.assertIn("type them directly in this channel", blocks)
                self.assertIn("Thread subtasks", blocks)
                self.assertIn("Sessions started outside Slack", blocks)
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
                self.assertNotIn("specific agent is busy", gateway.thread_replies[-1]["text"])
                self.assertEqual(len(runtime.started), 1)
                task, agent, thread = runtime.started[0]
                self.assertEqual(agent.handle, "cruz")
                self.assertEqual(thread.thread_ts, "171.thread")
                self.assertEqual(task.session_provider, Provider.CLAUDE)
                self.assertEqual(task.session_id, "claude-session-1")
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
                self.assertEqual(len(tasks), 3)
                self.assertEqual(tasks[-1].agent_id, parent_task.agent_id)
                self.assertIn("review feedback", tasks[-1].metadata["thread_context"])
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
                self.assertEqual(len(tasks), 3)
                self.assertEqual(tasks[-1].agent_id, parent_task.agent_id)
                self.assertEqual(tasks[-1].prompt, "reply with DELEGATED_OK")
                self.assertIn("DELEGATED_OK", tasks[-1].metadata["thread_context"])
                self.assertEqual(len(runtime.started), 3)
                self.assertIn(
                    f"@{codex_agent.handle}, please reply with DELEGATED_OK",
                    gateway.thread_replies[-2]["text"],
                )
                self.assertIn(
                    f"@{reviewer.handle}",
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
                self.assertEqual(len(tasks), 3)
                followup = tasks[-1]
                self.assertEqual(followup.agent_id, original.agent_id)
                self.assertIn("Continue the original task", followup.prompt)
                self.assertIn("test coverage", followup.metadata["thread_context"])
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
                self.assertEqual(len(tasks), 3)
                self.assertEqual(tasks[-1].agent_id, original.agent_id)
                self.assertIn("Continue the original task", tasks[-1].prompt)
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
                followup = tasks[-1]
                self.assertEqual(followup.agent_id, mina.agent_id)
                self.assertEqual(followup.session_id, "claude-session-mina")
                self.assertIn("say it like a person", followup.prompt)
                self.assertEqual(len(runtime.started), 1)
                self.assertEqual(runtime.started[0][1].handle, "minaa")
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

    def test_plain_thread_reply_after_completion_starts_followup_with_original_agent(self):
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
                            "text": "Somebody summarize pyproject",
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
                            "text": "also include the package version",
                            "ts": "171.000002",
                            "thread_ts": parent_task.thread_ts,
                        }
                    }
                )

                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(len(tasks), 2)
                self.assertEqual(tasks[-1].agent_id, parent_task.agent_id)
                self.assertEqual(tasks[-1].prompt, "also include the package version")
                self.assertEqual(tasks[-1].thread_ts, parent_task.thread_ts)
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
