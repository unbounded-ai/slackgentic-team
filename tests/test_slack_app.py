import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent_harness.models import (
    AgentSession,
    AgentTaskStatus,
    ControlMode,
    Provider,
    SessionStatus,
    SlackThreadRef,
)
from agent_harness.slack import encode_action_value
from agent_harness.slack_app import (
    DEFAULT_AGENT_AVATAR_BASE_URL,
    SETTING_ROSTER_TS,
    SlackReplyTarget,
    SlackTeamController,
)
from agent_harness.slack_client import PostedMessage
from agent_harness.store import Store
from agent_harness.team import build_initial_model_team
from agent_harness.team_commands import FireCommand, HireCommand, RosterCommand


class FakeGateway:
    def __init__(self):
        self.posts = []
        self.updates = []
        self.thread_replies = []
        self.channels = []
        self.invites = []
        self.views = []
        self.reactions = []

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

    def post_thread_reply(self, thread, text, persona=None, icon_url=None, blocks=None):
        ts = f"1712345679.{len(self.thread_replies):06d}"
        self.thread_replies.append(
            {
                "thread": thread,
                "text": text,
                "persona": persona,
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
                    "Agent roster: 2 active lightweight handles",
                )
                self.assertIsNotNone(gateway.posts[-1]["blocks"])
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
                                }
                            },
                        },
                    }
                )

                self.assertEqual(gateway.channels, [("agent-team", True)])
                self.assertEqual(gateway.invites, [("CNEW", ["U1"])])
                self.assertEqual(len(store.list_team_agents()), 2)
                self.assertTrue(gateway.posts)
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
                self.assertEqual(len(gateway.updates), 1)
                self.assertIn("Agent usage for", gateway.posts[0]["text"])
            finally:
                store.close()

    def test_status_slash_command_posts_usage(self):
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
                self.assertIn("Agent usage for", gateway.posts[0]["text"])
            finally:
                store.close()

    def test_channel_work_request_creates_thread_and_starts_runtime(self):
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
                            "text": "Somebody do update the README",
                            "ts": "171.000001",
                        }
                    }
                )

                tasks = store.list_agent_tasks()
                self.assertEqual(len(tasks), 1)
                self.assertEqual(len(runtime.started), 1)
                self.assertEqual(tasks[0].thread_ts, "1712345678.000000")
                self.assertTrue(gateway.reactions)
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
                rendered = gateway.posts[0]["blocks"][0]["text"]["text"]
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
                store.set_setting("slack.agent_avatar_base_url", "https://cdn.example.test/avatars/")
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
                    gateway.posts[0]["icon_url"],
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
                    gateway.posts[0]["icon_url"],
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
                self.assertIn("Agent usage for", gateway.posts[-1]["text"])
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

    def test_orphaned_active_tasks_are_cancelled_on_startup_reconcile(self):
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

                self.assertEqual(cancelled, 1)
                self.assertEqual(
                    store.get_agent_task(task.task_id).status,
                    AgentTaskStatus.CANCELLED,
                )
                self.assertIn("daemon restarted", gateway.thread_replies[-1]["text"])
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
                            {"value": encode_action_value("task.cancel", task_id=task.task_id)}
                        ],
                    }
                )

                self.assertEqual(runtime.stopped, [(task.task_id, AgentTaskStatus.CANCELLED)])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
