import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from agent_harness.deferred import (
    AGENT_DEFERRED_SIGNAL_PREFIX,
    DEFERRED_RESOLUTION_ATTEMPTS_METADATA_KEY,
    DEFERRED_RESOLUTION_METADATA_KEY,
    DEFERRED_RESOLUTION_OCCUPIED_HANDLES_METADATA_KEY,
)
from agent_harness.models import (
    ASSIGNMENT_PROMPT_METADATA_KEY,
    DANGEROUS_MODE_METADATA_KEY,
    PR_URLS_METADATA_KEY,
    ROSTER_SUMMARY_METADATA_KEY,
    AgentSession,
    AgentTaskKind,
    AgentTaskStatus,
    AssignmentMode,
    ControlMode,
    DeferredWorkStatus,
    Provider,
    ScheduledWorkKind,
    SessionStatus,
    SlackThreadRef,
    TeamAgentStatus,
    WorkDependency,
    WorkDependencyKind,
    WorkRequest,
    deferred_work_dependency_id,
    external_session_dependency_id,
    scheduled_work_dependency_id,
    utc_now,
)
from agent_harness.runtime.tasks import (
    AGENT_REACTION_SIGNAL_PREFIX,
    AGENT_ROSTER_STATUS_SIGNAL_PREFIX,
    AGENT_THREAD_DONE_SIGNAL,
    MANAGED_RUN_MAX_RESUME_AGE,
    MANAGED_RUN_MAX_RESUMES,
    MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY,
    MANAGED_RUN_STARTED_METADATA_KEY,
    managed_run_resume_attempts,
)
from agent_harness.schedules import (
    AGENT_SCHEDULE_SIGNAL_PREFIX,
    SCHEDULE_RESOLUTION_ATTEMPTS_METADATA_KEY,
    SCHEDULE_RESOLUTION_METADATA_KEY,
)
from agent_harness.slack import encode_action_value
from agent_harness.slack.app import (
    AUTO_ALLOWED_CLAUDE_PERMISSION_TEXT,
    CLAUDE_CHANNEL_PERMISSION_METHOD,
    DEFAULT_AGENT_AVATAR_BASE_URL,
    SETTING_ROSTER_TS,
    SETTING_SLACK_BACKFILL_LAST_AWAKE,
    ClaudePermissionAutoResolver,
    DeferredWorkRunner,
    ScheduledTimerRunner,
    ScheduledWorkRunner,
    SlackMessageBackfill,
    SlackReplyTarget,
    SlackTeamController,
    SocketModeSlackApp,
)
from agent_harness.slack.client import PostedMessage
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team, create_agent_task
from agent_harness.team.commands import (
    FireCommand,
    FireEveryoneCommand,
    HireCommand,
    RosterCommand,
    ScheduledTasksCommand,
)
from agent_harness.timers import AGENT_TIMER_SIGNAL_PREFIX


class FakeGateway:
    bot_user_id_value = "UBOT"

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

    def bot_user_id(self):
        return self.bot_user_id_value

    def _current_reactions(self, channel_id, message_ts):
        active: dict[str, int] = {}
        for ch, ts, name in self.reactions:
            if ch == channel_id and ts == message_ts:
                active[name] = active.get(name, 0) + 1
        for ch, ts, name in self.removed_reactions:
            if ch == channel_id and ts == message_ts and active.get(name, 0) > 0:
                active[name] -= 1
        return [
            {"name": name, "count": count, "users": [self.bot_user_id_value]}
            for name, count in active.items()
            if count > 0
        ]

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
        result = []
        for item in combined[-limit:]:
            decorated = dict(item)
            reactions = self._current_reactions(channel_id, item.get("ts"))
            if reactions:
                decorated["reactions"] = reactions
            result.append(decorated)
        return result

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
        self.interrupted_sent = []
        self.stopped = []
        self.interrupted = []
        self.resumed = []
        self.running_task_ids: set[str] = set()

    def start_task(self, task, agent, thread):
        self.started.append((task, agent, thread))
        return True

    def send_to_task(self, task_id, message):
        self.sent.append((task_id, message))
        return True

    def send_to_interrupted_task(self, task_id, message):
        self.interrupted_sent.append((task_id, message))
        return True

    def is_task_running(self, task_id: str) -> bool:
        return task_id in self.running_task_ids

    def stop_task(self, task_id, status=AgentTaskStatus.CANCELLED):
        self.stopped.append((task_id, status))
        return True

    def interrupt_task(self, task_id):
        self.interrupted.append(task_id)
        return True

    def resume_orphaned_task(self, task, agent, thread):
        attempts = managed_run_resume_attempts(task) + 1
        metadata = dict(task.metadata)
        metadata[MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY] = attempts
        bumped = replace(task, metadata=metadata)
        self.resumed.append((bumped, agent, thread))
        return self.start_task(bumped, agent, thread)


class FakeSessionBridge:
    def __init__(self):
        self.sent = []
        self.live_sent = []
        self.agent_request_actions = []

    def send_to_session(self, session, text, thread, slack_user=None):
        self.sent.append((session, text, thread, slack_user))
        return True

    def send_live_to_session(self, session, text, thread, slack_user=None):
        self.live_sent.append((session, text, thread, slack_user))
        return True

    def handle_agent_request_block_action(self, payload, channel_id, message_ts):
        self.agent_request_actions.append((payload, channel_id, message_ts))
        return True


class DetachedRuntime(FakeRuntime):
    def send_to_task(self, task_id, message):
        self.sent.append((task_id, message))
        return False


def _roster_work_submission_payload(
    agent=None,
    *,
    prompt: str,
    timing: str,
    dangerous: bool = False,
    repeat_time: str = "",
    timezone: str = "",
    weekday: str | None = None,
    run_at: str = "",
    dependency: str = "none",
    delay: str = "",
):
    metadata = {"channel_id": "C1", "message_ts": "171.roster"}
    if agent is not None:
        metadata["agent_id"] = agent.agent_id
        metadata["handle"] = agent.handle
    values = {
        "roster_work_prompt": {"value": {"value": prompt}},
        "roster_work_kind": {"value": {"selected_option": {"value": "work"}}},
        "roster_work_timing": {"value": {"selected_option": {"value": timing}}},
        "roster_work_run_at": {"value": {"value": run_at}},
        "roster_work_time": {"value": {"value": repeat_time}},
        "roster_work_timezone": {"value": {"value": timezone}},
        "roster_work_weekday": {
            "value": {"selected_option": {"value": weekday} if weekday is not None else None}
        },
        "roster_work_dependency": {"value": {"selected_option": {"value": dependency}}},
        "roster_work_delay": {"value": {"value": delay}},
        "roster_work_permissions": {
            "dangerous": {
                "selected_options": [{"value": "dangerous"}] if dangerous else [],
            }
        },
    }
    return {
        "type": "view_submission",
        "user": {"id": "U1"},
        "view": {
            "callback_id": "roster.work",
            "private_metadata": json.dumps(metadata),
            "state": {"values": values},
        },
    }


class SlackAppTests(unittest.TestCase):
    def test_socket_mode_connect_retries_before_process_exit(self):
        connect_calls = []
        closed_clients = []
        closed_apps = []

        class FakeSocketModeClient:
            def __init__(self, app_token, web_client):
                self.app_token = app_token
                self.web_client = web_client
                self.socket_mode_request_listeners = []

            def connect(self):
                connect_calls.append("connect")
                if len(connect_calls) == 1:
                    raise RuntimeError("network unavailable")
                raise KeyboardInterrupt()

            def close(self):
                closed_clients.append(True)

        class FakeSocketModeResponse:
            def __init__(self, envelope_id, payload=None):
                self.envelope_id = envelope_id
                self.payload = payload

        class FakeBackoff:
            def __init__(self, **kwargs):
                self.failures = 0
                self.next_delay = 0.0

            def wait(self, stop_event):
                self.failures += 1
                return False

        socket_mode_module = types.ModuleType("slack_sdk.socket_mode")
        socket_mode_module.SocketModeClient = FakeSocketModeClient
        response_module = types.ModuleType("slack_sdk.socket_mode.response")
        response_module.SocketModeResponse = FakeSocketModeResponse
        app = object.__new__(SocketModeSlackApp)
        app.config = types.SimpleNamespace(slack=types.SimpleNamespace(app_token="xapp-test"))
        app.gateway = types.SimpleNamespace(client=object())
        app.close = lambda: closed_apps.append(True)
        app.handle_request = lambda request: None

        with (
            patch.dict(
                sys.modules,
                {
                    "slack_sdk.socket_mode": socket_mode_module,
                    "slack_sdk.socket_mode.response": response_module,
                },
            ),
            patch("agent_harness.slack.app.LoopBackoff", FakeBackoff),
            patch("agent_harness.slack.app.log_loop_failure"),
        ):
            app.run_forever()

        self.assertEqual(connect_calls, ["connect", "connect"])
        self.assertEqual(closed_clients, [True])
        self.assertEqual(closed_apps, [True])

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

    def test_refresh_roster_tolerates_slack_update_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()

            def fail_update(channel_id, ts, text, blocks=None):
                raise RuntimeError("rate limited")

            gateway.update_message = fail_update
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 1):
                    store.upsert_team_agent(agent)
                store.set_setting(SETTING_ROSTER_TS, "171.000001")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                roster_ts = controller.refresh_or_post_roster("C1")

                self.assertEqual(roster_ts, "171.000001")
            finally:
                store.close()

    def test_refresh_roster_updates_all_remembered_roster_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                first_roster_ts = controller.post_roster("C1")
                second_roster_ts = controller.post_roster("C1")
                task = create_agent_task(agents[0], "investigate flaky tests", "C1")
                store.upsert_agent_task(task)

                returned_ts = controller.refresh_or_post_roster("C1")

                self.assertEqual(returned_ts, second_roster_ts)
                updated = {item["ts"]: item["text"] for item in gateway.updates}
                self.assertEqual(set(updated), {first_roster_ts, second_roster_ts})
                self.assertTrue(all("1 available, 1 occupied" in text for text in updated.values()))
            finally:
                store.close()

    def test_refresh_roster_discovers_recent_untracked_roster_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                store.set_setting(SETTING_ROSTER_TS, "171.000003")
                gateway.history_messages.extend(
                    [
                        {
                            "type": "message",
                            "channel": "C1",
                            "text": (
                                "Agent roster: 2 active lightweight handles, "
                                "2 available, 0 occupied"
                            ),
                            "ts": "171.000001",
                        },
                        {
                            "type": "message",
                            "channel": "C1",
                            "text": "not a roster",
                            "ts": "171.000002",
                        },
                        {
                            "type": "message",
                            "channel": "C1",
                            "text": (
                                "Agent roster: 2 active lightweight handles, "
                                "2 available, 0 occupied"
                            ),
                            "ts": "171.000003",
                        },
                    ]
                )
                task = create_agent_task(agents[0], "investigate flaky tests", "C1")
                store.upsert_agent_task(task)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                returned_ts = controller.refresh_or_post_roster("C1")

                self.assertEqual(returned_ts, "171.000003")
                updated = {item["ts"]: item["text"] for item in gateway.updates}
                self.assertEqual(set(updated), {"171.000001", "171.000003"})
                self.assertTrue(all("1 available, 1 occupied" in text for text in updated.values()))
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

    def test_fire_button_detaches_external_session_first(self):
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
                store.set_setting("external_session_live_target.codex.s1", "123")
                store.set_setting("external_session_summary.codex.s1", "update the docs")
                store.upsert_slack_thread_for_session(
                    Provider.CODEX,
                    "s1",
                    "local",
                    SlackThreadRef("C1", "171.000010", "171.000010"),
                )
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
                fired = store.get_team_agent(agent.agent_id, include_fired=True)
                self.assertIsNotNone(fired)
                self.assertEqual(fired.status if fired else None, TeamAgentStatus.FIRED)
                self.assertIsNone(store.get_setting("external_session_agent.codex.s1"))
                self.assertIsNone(store.get_setting("external_session_live_target.codex.s1"))
                self.assertIsNone(store.get_setting("external_session_summary.codex.s1"))
                self.assertIsNotNone(store.get_setting("external_session_ignored.codex.s1"))
                self.assertIsNone(
                    store.get_slack_thread_for_session(Provider.CODEX, "s1", "local", "C1")
                )
                self.assertIn(
                    f"Detached external session and removed @{agent.handle}.",
                    gateway.thread_replies[-1]["text"],
                )
            finally:
                store.close()

    def test_hiring_reclaims_fired_agent_handle_for_active_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                old_vera = replace(
                    build_initial_model_team(1, 0)[0],
                    agent_id="agent_old_vera",
                    handle="vera",
                    full_name="Vera Aoki",
                    status=TeamAgentStatus.FIRED,
                )
                new_vera = replace(
                    build_initial_model_team(1, 0)[0],
                    agent_id="agent_new_vera",
                    handle="vera",
                    full_name="Vera Martinez",
                )
                store.upsert_team_agent(old_vera)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller._release_inactive_handle(new_vera.handle)
                store.upsert_team_agent(new_vera)

                current = store.get_team_agent("vera")
                self.assertIsNotNone(current)
                assert current is not None
                self.assertEqual(current.agent_id, new_vera.agent_id)
                retired = store.get_team_agent("agent_old_vera", include_fired=True)
                self.assertIsNotNone(retired)
                assert retired is not None
                self.assertNotEqual(retired.handle, "vera")
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

    def test_roster_assign_button_opens_work_modal_for_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.roster"},
                        "trigger_id": "T1",
                        "actions": [
                            {
                                "value": encode_action_value(
                                    "roster.work.open",
                                    mode="now",
                                    agent_id=agent.agent_id,
                                    handle=agent.handle,
                                )
                            }
                        ],
                    }
                )

                self.assertEqual(gateway.views[0][0], "T1")
                view = gateway.views[0][1]
                self.assertEqual(view["callback_id"], "roster.work")
                self.assertIn(f'"handle":"{agent.handle}"', view["private_metadata"])
                timing = next(
                    block
                    for block in view["blocks"]
                    if block.get("block_id") == "roster_work_timing"
                )
                self.assertEqual(timing["element"]["initial_option"]["value"], "now")
                timezone = next(
                    block
                    for block in view["blocks"]
                    if block.get("block_id") == "roster_work_timezone"
                )
                self.assertTrue(timezone["element"].get("initial_value"))
                self.assertIn("Required for daily or weekly", timezone["hint"]["text"])
            finally:
                store.close()

    def test_roster_work_modal_shows_dependency_task_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                blocker, target = build_initial_model_team(2, 0)
                store.upsert_team_agent(blocker)
                store.upsert_team_agent(target)
                task = create_agent_task(blocker, "finish the blocker branch", "C1")
                store.upsert_agent_task(task)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.roster"},
                        "trigger_id": "T1",
                        "actions": [
                            {
                                "value": encode_action_value(
                                    "roster.work.open",
                                    mode="now",
                                    agent_id=target.agent_id,
                                    handle=target.handle,
                                )
                            }
                        ],
                    }
                )

                dependency = next(
                    block
                    for block in gateway.views[0][1]["blocks"]
                    if block.get("block_id") == "roster_work_dependency"
                )
                options = dependency["element"]["options"]
                blocker_option = next(
                    option for option in options if option["value"] == blocker.handle
                )
                self.assertEqual(
                    blocker_option["description"]["text"],
                    "finish the blocker branch",
                )
                self.assertNotIn(task.task_id, str(options))
            finally:
                store.close()

    def test_roster_work_button_for_stale_occupied_agent_does_not_open_modal(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                store.upsert_agent_task(create_agent_task(agent, "already busy", "C1"))
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.roster"},
                        "trigger_id": "T1",
                        "actions": [
                            {
                                "value": encode_action_value(
                                    "roster.work.open",
                                    mode="now",
                                    agent_id=agent.agent_id,
                                    handle=agent.handle,
                                )
                            }
                        ],
                    }
                )

                self.assertEqual(gateway.views, [])
                self.assertIn("already occupied", gateway.thread_replies[-1]["text"])
            finally:
                store.close()

    def test_roster_work_submission_starts_dangerous_specific_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
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

                response = controller.handle_view_submission(
                    _roster_work_submission_payload(
                        agent,
                        prompt="repair the deploy script",
                        timing="now",
                        dangerous=True,
                    )
                )

                self.assertIsNone(response)
                self.assertEqual(len(runtime.started), 1)
                task, started_agent, thread = runtime.started[0]
                self.assertEqual(started_agent.agent_id, agent.agent_id)
                self.assertEqual(task.prompt, "repair the deploy script")
                self.assertTrue(task.metadata[DANGEROUS_MODE_METADATA_KEY])
                self.assertEqual(
                    task.metadata[ROSTER_SUMMARY_METADATA_KEY],
                    "repair the deploy script",
                )
                self.assertEqual(thread.thread_ts, gateway.posts[0]["ts"])
                self.assertIn("*:zap: Dangerous mode*", gateway.posts[0]["text"])
            finally:
                store.close()

    def test_roster_work_submission_creates_recurring_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                response = controller.handle_view_submission(
                    _roster_work_submission_payload(
                        agent,
                        prompt="check CI",
                        timing="daily",
                        repeat_time="17:00",
                        timezone="America/Chicago",
                    )
                )

                self.assertIsNone(response)
                scheduled = store.list_scheduled_work()
                self.assertEqual(len(scheduled), 1)
                self.assertEqual(scheduled[0].requested_handle, agent.handle)
                self.assertEqual(scheduled[0].schedule_kind, ScheduledWorkKind.RECURRING)
                self.assertEqual(scheduled[0].recurrence["frequency"], "daily")
                self.assertIn(f"Scheduled: @{agent.handle} `check CI`", gateway.updates[-1]["text"])
                self.assertIn("daily at 17:00 America/Chicago", gateway.updates[-1]["text"])
                self.assertNotIn("schedule_", gateway.updates[-1]["text"])
            finally:
                store.close()

    def test_roster_work_submission_for_busy_specific_agent_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                store.upsert_agent_task(create_agent_task(agent, "busy", "C1"))
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                response = controller.handle_view_submission(
                    _roster_work_submission_payload(
                        agent,
                        prompt="check CI",
                        timing="daily",
                        repeat_time="17:00",
                        timezone="America/Chicago",
                    )
                )

                self.assertEqual(response["response_action"], "errors")
                self.assertIn("already occupied", response["errors"]["roster_work_prompt"])
                self.assertEqual(store.list_scheduled_work(), [])
            finally:
                store.close()

    def test_roster_work_success_can_run_after_view_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                callbacks = []
                controller._run_after_view_ack = lambda label, callback: callbacks.append(
                    (label, callback)
                )
                run_at = (utc_now() + timedelta(hours=1)).isoformat()

                response = controller.handle_view_submission(
                    _roster_work_submission_payload(
                        agent,
                        prompt="check CI later",
                        timing="once",
                        run_at=run_at,
                    ),
                    async_success=True,
                )

                self.assertIsNone(response)
                self.assertEqual(store.list_scheduled_work(), [])
                self.assertEqual(callbacks[0][0], "roster-work-scheduled")

                callbacks[0][1]()

                scheduled = store.list_scheduled_work()
                self.assertEqual(len(scheduled), 1)
                self.assertEqual(scheduled[0].requested_handle, agent.handle)
            finally:
                store.close()

    def test_roster_work_submission_defers_until_busy_agent_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                blocker, target = build_initial_model_team(2, 0)
                store.upsert_team_agent(blocker)
                store.upsert_team_agent(target)
                active_task = create_agent_task(blocker, "finish blocker", "C1")
                store.upsert_agent_task(active_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                response = controller.handle_view_submission(
                    _roster_work_submission_payload(
                        target,
                        prompt="follow up afterward",
                        timing="now",
                        dependency=blocker.handle,
                    )
                )

                self.assertIsNone(response)
                self.assertEqual(runtime.started, [])
                deferred = store.list_deferred_work()
                self.assertEqual(len(deferred), 1)
                self.assertEqual(deferred[0].status, DeferredWorkStatus.WAITING_DEPS)
                self.assertEqual(deferred[0].requested_handle, target.handle)
                self.assertEqual(deferred[0].depends_on[0].handle, blocker.handle)
                self.assertEqual(deferred[0].depends_on[0].task_id, active_task.task_id)
                ack = gateway.updates[-1]["text"]
                self.assertIn("Waiting on", ack)
                self.assertIn(f"- @{blocker.handle}: finish blocker", ack)
                self.assertNotIn(active_task.task_id, ack)
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
                self.assertIn("*Queued:* investigate flaky tests", blocks)
                self.assertNotIn("Slack task:", blocks)
                self.assertNotIn("<https://example.slack.com/archives/C1/p", blocks)
                self.assertIn("'text': {'type': 'plain_text', 'text': 'Open thread'}", blocks)
                self.assertIn("'url': 'https://example.slack.com/archives/C1/p", blocks)
                self.assertIn("Free up", blocks)
                self.assertIn("Available", blocks)
            finally:
                store.close()

    def test_roster_frees_agent_and_prunes_stale_managed_session_without_active_task(self):
        from agent_harness.sessions.managed_session import (
            managed_session_agent_key,
            record_managed_session,
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 0)
                store.upsert_team_agent(agents[0])
                # A previous task left a managed_session record behind even
                # though every task for the agent is now in a terminal state.
                # The roster must not surface this as "Occupied" — every
                # Occupied slot has to be backed by a real deferred or
                # active task — and the stale record should be cleaned up.
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="codex-orphan",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    started_at=utc_now(),
                    last_seen_at=utc_now(),
                    status=SessionStatus.IDLE,
                    control_mode=ControlMode.OBSERVED,
                )
                store.upsert_session(session)
                record_managed_session(
                    store,
                    Provider.CODEX,
                    "codex-orphan",
                    agents[0].agent_id,
                    dangerous_mode=True,
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.post_roster("C1")

                self.assertIn("1 available, 0 occupied", gateway.posts[-1]["text"])
                blocks = str(gateway.posts[-1]["blocks"])
                self.assertNotIn("session not yet released", blocks)
                self.assertIsNone(
                    store.get_setting(managed_session_agent_key(Provider.CODEX, "codex-orphan"))
                )
            finally:
                store.close()

    def test_roster_clears_managed_session_after_task_done(self):
        from agent_harness.sessions.managed_session import (
            managed_session_agent_key,
            record_managed_session,
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 0)
                store.upsert_team_agent(agents[0])
                task = create_agent_task(agents[0], "ship it", "C1")
                task = replace(
                    task,
                    session_provider=Provider.CODEX,
                    session_id="codex-finished",
                )
                store.upsert_agent_task(task)
                record_managed_session(
                    store,
                    Provider.CODEX,
                    "codex-finished",
                    agents[0].agent_id,
                    dangerous_mode=True,
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_runtime_task_done(
                    task,
                    agents[0],
                    SlackThreadRef("C1", "171.000001"),
                )

                self.assertIsNone(
                    store.get_setting(managed_session_agent_key(Provider.CODEX, "codex-finished"))
                )
            finally:
                store.close()

    def test_roster_prefers_agent_roster_summary_over_latest_task_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "rerun tests", "C1"),
                    metadata={
                        ROSTER_SUMMARY_METADATA_KEY: (
                            "Roster UX fix: validating E2E before PR merge"
                        )
                    },
                )
                store.upsert_agent_task(task)
                store.update_agent_task_thread(task.task_id, "171.000001", "171.000001")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.post_roster("C1")

                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn(
                    "*Queued:* Roster UX fix: validating E2E before PR merge",
                    blocks,
                )
                self.assertNotIn("Slack task:", blocks)
            finally:
                store.close()

    def test_roster_falls_back_to_original_assignment_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "rerun tests", "C1"),
                    metadata={
                        ASSIGNMENT_PROMPT_METADATA_KEY: (
                            "Improve roster summaries and dangerous-mode display"
                        )
                    },
                )
                store.upsert_agent_task(task)
                store.update_agent_task_thread(task.task_id, "171.000001", "171.000001")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.post_roster("C1")

                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn(
                    ("*Queued:* Improve roster summaries and dangerous-mode display"),
                    blocks,
                )
                self.assertNotIn("Slack task:", blocks)
            finally:
                store.close()

    def test_channel_task_assignment_refreshes_existing_roster_to_occupied(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                store.set_setting(SETTING_ROSTER_TS, "171.roster")
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
                            "text": "write the status update",
                            "ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                self.assertEqual(gateway.updates[-1]["ts"], "171.roster")
                self.assertIn("0 available, 1 occupied", gateway.updates[-1]["text"])
            finally:
                store.close()

    def test_roster_refresh_skips_unchanged_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                roster_ts = controller.post_roster("C1")
                controller.refresh_or_post_roster("C1")

                self.assertEqual(gateway.updates, [])
                self.assertEqual(gateway.pins, [("C1", roster_ts)])

                task = create_agent_task(agent, "ship the fix", "C1")
                store.upsert_agent_task(task)
                store.update_agent_task_thread(task.task_id, "171.thread", "171.thread")
                controller.refresh_or_post_roster("C1")
                controller.refresh_or_post_roster("C1")

                self.assertEqual(len(gateway.updates), 1)
                self.assertIn("0 available, 1 occupied", gateway.updates[0]["text"])
                self.assertEqual(gateway.pins, [("C1", roster_ts)])
            finally:
                store.close()

    def test_task_status_reaction_transition_removes_only_known_previous_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller._mark_message_acknowledged("C1", "171.000001")
                gateway.removed_reactions.clear()

                controller._mark_message_queued("C1", "171.000001")

                self.assertEqual(gateway.reactions[-1], ("C1", "171.000001", "inbox_tray"))
                self.assertEqual(
                    gateway.removed_reactions,
                    [("C1", "171.000001", "eyes")],
                )
                gateway.removed_reactions.clear()

                controller._mark_message_in_progress("C1", "171.000001")
                controller._mark_message_in_progress("C1", "171.000001")

                self.assertEqual(
                    gateway.reactions[-1], ("C1", "171.000001", "hourglass_flowing_sand")
                )
                self.assertEqual(
                    gateway.removed_reactions,
                    [("C1", "171.000001", "inbox_tray")],
                )
            finally:
                store.close()

    def test_user_reaction_to_agent_message_is_relayed_to_active_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "ship the fix", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                )
                store.upsert_agent_task(task)
                store.update_agent_task_thread(task.task_id, "171.thread", "171.agent")
                task = store.get_agent_task(task.task_id)
                assert task is not None
                runtime.running_task_ids.add(task.task_id)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller._remember_agent_authored_message(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.thread", "171.agent"),
                    "171.agent",
                    "Final status",
                )
                event = {
                    "event": {
                        "type": "reaction_added",
                        "user": "U1",
                        "reaction": "thumbsup",
                        "item": {
                            "type": "message",
                            "channel": "C1",
                            "ts": "171.agent",
                        },
                        "event_ts": "172.000001",
                    }
                }

                controller.handle_event(event)
                controller.handle_event(event)

                self.assertEqual(len(runtime.sent), 1)
                self.assertEqual(runtime.sent[0][0], task.task_id)
                self.assertIn("reacted with :thumbsup:", runtime.sent[0][1])
                self.assertIn("lightweight feedback", runtime.sent[0][1])
            finally:
                store.close()

    def test_bot_reaction_to_agent_message_is_not_relayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "ship the fix", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                )
                store.upsert_agent_task(task)
                store.update_agent_task_thread(task.task_id, "171.thread", "171.agent")
                task = store.get_agent_task(task.task_id)
                assert task is not None
                runtime.running_task_ids.add(task.task_id)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller._remember_agent_authored_message(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.thread", "171.agent"),
                    "171.agent",
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "reaction_added",
                            "user": "UBOT",
                            "reaction": "eyes",
                            "item": {
                                "type": "message",
                                "channel": "C1",
                                "ts": "171.agent",
                            },
                            "event_ts": "172.000002",
                        }
                    }
                )

                self.assertEqual(runtime.sent, [])
            finally:
                store.close()

    def test_agent_reaction_signal_reacts_to_latest_user_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "ship the fix", "C1"),
                    metadata={"request_message_ts": "171.user"},
                )
                store.upsert_agent_task(task)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                handled = controller.handle_runtime_agent_control(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.thread"),
                    f"{AGENT_REACTION_SIGNAL_PREFIX}:eyes:",
                )

                self.assertTrue(handled)
                self.assertIn(("C1", "171.user", "eyes"), gateway.reactions)
            finally:
                store.close()

    def test_roster_shows_runtime_running_task_occupancy(self):
        class RuntimeWithRunningTask(FakeRuntime):
            def __init__(self, running):
                super().__init__()
                self._running = running

            def running_tasks(self):
                return self._running

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "drive PR 23", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                )
                running = type(
                    "Running",
                    (),
                    {
                        "agent": agent,
                        "task": task,
                        "thread": SlackThreadRef("C1", "171.thread"),
                    },
                )()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=RuntimeWithRunningTask([running]),
                )

                controller.post_roster("C1")

                self.assertIn("0 available, 1 occupied", gateway.posts[-1]["text"])
                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn("*Working:* drive PR 23", blocks)
                self.assertNotIn("Occupied: Slack task:", blocks)
            finally:
                store.close()

    def test_roster_shows_idle_active_task_as_open_thread_not_working(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "waiting for follow-up", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                )
                store.upsert_agent_task(task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=DetachedRuntime(),
                )

                controller.post_roster("C1")

                self.assertIn("0 available, 1 occupied", gateway.posts[-1]["text"])
                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn("*Occupied:* Open thread: waiting for follow-up", blocks)
                self.assertNotIn("*Working:* waiting for follow-up", blocks)
            finally:
                store.close()

    def test_roster_ignores_terminal_runtime_running_task(self):
        class RuntimeWithRunningTask(FakeRuntime):
            def __init__(self, running):
                super().__init__()
                self._running = running

            def running_tasks(self):
                return self._running

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "closing cleanup", "C1"),
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.thread",
                )
                store.upsert_agent_task(task)
                running = type(
                    "Running",
                    (),
                    {
                        "agent": agent,
                        "task": task,
                        "thread": SlackThreadRef("C1", "171.thread"),
                    },
                )()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=RuntimeWithRunningTask([running]),
                )

                controller.post_roster("C1")

                self.assertIn("1 available, 0 occupied", gateway.posts[-1]["text"])
                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn("Available", blocks)
                self.assertNotIn("closing cleanup", blocks)
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
                self.assertIn("*Queued:* rewrite installer", blocks)
                self.assertNotIn("Slack task:", blocks)
                self.assertIn("*Mode:* :zap: Dangerous", blocks)
            finally:
                store.close()

    def test_roster_status_signal_updates_active_task_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "small follow-up", "C1")
                store.upsert_agent_task(task)
                store.update_agent_task_thread(task.task_id, "171.000001", "171.000001")
                store.set_setting(SETTING_ROSTER_TS, "171.roster")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                handled = controller.handle_runtime_agent_control(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.000001"),
                    f"{AGENT_ROSTER_STATUS_SIGNAL_PREFIX}Roster UX fix: opening PR after E2E",
                )

                current = store.get_agent_task(task.task_id)
                assert current is not None
                self.assertTrue(handled)
                self.assertEqual(
                    current.metadata[ROSTER_SUMMARY_METADATA_KEY],
                    "Roster UX fix: opening PR after E2E",
                )
                header_updates = [
                    update for update in gateway.updates if update["ts"] == "171.000001"
                ]
                self.assertEqual(len(header_updates), 1)
                self.assertIn(
                    "Roster UX fix: opening PR after E2E",
                    header_updates[0]["text"],
                )
                self.assertIn(
                    "Roster UX fix: opening PR after E2E",
                    str(header_updates[0]["blocks"]),
                )
                roster_updates = [
                    update for update in gateway.updates if update["ts"] == "171.roster"
                ]
                self.assertEqual(len(roster_updates), 1)
                self.assertIn(
                    "*Queued:* Roster UX fix: opening PR after E2E",
                    str(roster_updates[0]["blocks"]),
                )
                self.assertNotIn("Slack task:", str(roster_updates[0]["blocks"]))
            finally:
                store.close()

    def test_runtime_agent_messages_update_task_and_roster_pr_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "ship the task view", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                )
                store.upsert_agent_task(task)
                store.update_agent_task_thread(task.task_id, "171.thread", "171.parent")
                task = store.get_agent_task(task.task_id)
                assert task is not None
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                controller.post_roster("C1")
                roster_ts = gateway.posts[-1]["ts"]

                first = controller.handle_runtime_agent_message(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.thread"),
                    "PR up: https://github.com/acme/app/pull/42",
                    "171.agent1",
                )
                current = store.get_agent_task(task.task_id)
                assert current is not None
                second = controller.handle_runtime_agent_message(
                    current,
                    agent,
                    SlackThreadRef("C1", "171.thread"),
                    "Second PR: https://github.com/acme/app/pull/43",
                    "171.agent2",
                )

                current = store.get_agent_task(task.task_id)
                assert current is not None
                self.assertFalse(first)
                self.assertFalse(second)
                self.assertEqual(
                    current.metadata[PR_URLS_METADATA_KEY],
                    [
                        "https://github.com/acme/app/pull/42",
                        "https://github.com/acme/app/pull/43",
                    ],
                )
                header_updates = [
                    update for update in gateway.updates if update["ts"] == "171.parent"
                ]
                self.assertGreaterEqual(len(header_updates), 2)
                self.assertIn(
                    "<https://github.com/acme/app/pull/42|acme/app#42>",
                    str(header_updates[-1]["blocks"]),
                )
                self.assertIn(
                    "<https://github.com/acme/app/pull/43|acme/app#43>",
                    str(header_updates[-1]["blocks"]),
                )
                roster_updates = [update for update in gateway.updates if update["ts"] == roster_ts]
                self.assertGreaterEqual(len(roster_updates), 2)
                self.assertIn("*PRs:*", str(roster_updates[-1]["blocks"]))
                self.assertIn(
                    "<https://github.com/acme/app/pull/42|acme/app#42>",
                    str(roster_updates[-1]["blocks"]),
                )
                self.assertIn(
                    "<https://github.com/acme/app/pull/43|acme/app#43>",
                    str(roster_updates[-1]["blocks"]),
                )
            finally:
                store.close()

    def test_roster_status_signal_refreshes_thread_header_when_summary_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                summary = "Roster UX fix: refreshing Priya's task thread header"
                task = replace(
                    create_agent_task(agent, "small follow-up", "C1"),
                    metadata={ROSTER_SUMMARY_METADATA_KEY: summary},
                )
                store.upsert_agent_task(task)
                store.update_agent_task_thread(task.task_id, "171.000001", "171.000001")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                handled = controller.handle_runtime_agent_control(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.000001"),
                    f"{AGENT_ROSTER_STATUS_SIGNAL_PREFIX}{summary}",
                )

                self.assertTrue(handled)
                self.assertEqual(len(gateway.updates), 1)
                self.assertEqual(gateway.updates[0]["ts"], "171.000001")
                self.assertIn(summary, gateway.updates[0]["text"])
                self.assertIn(summary, str(gateway.updates[0]["blocks"]))
            finally:
                store.close()

    def test_runtime_task_exit_refreshes_roster_and_keeps_root_task_occupied(self):
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
                    metadata={
                        PR_URLS_METADATA_KEY: ["https://github.com/acme/app/pull/42"],
                        ROSTER_SUMMARY_METADATA_KEY: "validate ownership PR",
                        MANAGED_RUN_STARTED_METADATA_KEY: utc_now().isoformat(),
                    },
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
                current = store.get_agent_task(task.task_id)
                assert current is not None
                self.assertEqual(current.status, AgentTaskStatus.ACTIVE)
                self.assertNotIn(MANAGED_RUN_STARTED_METADATA_KEY, current.metadata)
                self.assertIn("validate ownership PR", str(gateway.updates[-1]["blocks"]))
                self.assertIn(
                    "<https://github.com/acme/app/pull/42|acme/app#42>",
                    str(gateway.updates[-1]["blocks"]),
                )
                self.assertNotIn(("C1", "171.000001", "white_check_mark"), gateway.reactions)
            finally:
                store.close()

    def test_runtime_task_exit_clears_request_message_hourglass(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "put up the PR", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.bot",
                    metadata={"request_message_ts": "171.user"},
                )
                store.upsert_agent_task(task)
                gateway.reactions.append(("C1", "171.user", "eyes"))
                gateway.reactions.append(("C1", "171.user", "hourglass_flowing_sand"))
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_runtime_task_done(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.thread", "171.bot"),
                )

                self.assertEqual(store.get_agent_task(task.task_id).status, AgentTaskStatus.ACTIVE)
                self.assertIn(("C1", "171.user", "eyes"), gateway.removed_reactions)
                self.assertIn(
                    ("C1", "171.user", "hourglass_flowing_sand"),
                    gateway.removed_reactions,
                )
                self.assertNotIn(("C1", "171.user", "white_check_mark"), gateway.reactions)
            finally:
                store.close()

    def test_runtime_self_parented_followup_keeps_agent_occupied_after_run_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "follow up in same thread", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.000001",
                    parent_message_ts="171.000001",
                )
                task = replace(
                    task,
                    metadata={
                        "parent_task_id": task.task_id,
                        "parent_agent_id": agent.agent_id,
                        "request_message_ts": "171.user",
                    },
                )
                store.upsert_agent_task(task)
                store.set_setting(SETTING_ROSTER_TS, "171.000099")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_runtime_task_done(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.000001"),
                )

                self.assertEqual(store.get_agent_task(task.task_id).status, AgentTaskStatus.ACTIVE)
                self.assertIn("0 available, 1 occupied", gateway.updates[-1]["text"])
                self.assertIn(
                    ("C1", "171.user", "hourglass_flowing_sand"), gateway.removed_reactions
                )
                self.assertIn(("C1", "171.user", "eyes"), gateway.removed_reactions)
                self.assertNotIn(("C1", "171.user", "white_check_mark"), gateway.reactions)
            finally:
                store.close()

    def test_runtime_self_parented_review_followup_frees_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(
                        agent,
                        "review the updated design",
                        "C1",
                        kind=AgentTaskKind.REVIEW,
                    ),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.000001",
                    parent_message_ts="171.bot",
                )
                task = replace(
                    task,
                    metadata={
                        "parent_task_id": task.task_id,
                        "parent_agent_id": agent.agent_id,
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

                self.assertEqual(store.get_agent_task(task.task_id).status, AgentTaskStatus.DONE)
                self.assertEqual(
                    [item.agent_id for item in store.idle_team_agents()],
                    [agent.agent_id],
                )
                self.assertEqual(gateway.updates[-1]["ts"], "171.bot")
                self.assertNotIn("actions", str(gateway.updates[-1]["blocks"]))
                self.assertIn(
                    ("C1", "171.user", "hourglass_flowing_sand"),
                    gateway.removed_reactions,
                )
            finally:
                store.close()

    def test_agent_final_handle_line_routes_callback_without_action_button(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                ravi, leah = build_initial_model_team(1, 1)
                ravi = replace(ravi, handle="ravi", provider_preference=Provider.CODEX)
                leah = replace(leah, handle="leah", provider_preference=Provider.CLAUDE)
                store.upsert_team_agent(ravi)
                store.upsert_team_agent(leah)
                parent = replace(
                    create_agent_task(leah, "drive the design to completion", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    requested_by_slack_user="U1",
                )
                store.upsert_agent_task(parent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                handled = controller.handle_runtime_agent_message(
                    parent,
                    leah,
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                    "If that's green, I'll move to implementation.\n\n@ravi",
                    "171.agent",
                )

                self.assertTrue(handled)
                self.assertEqual(len(runtime.started), 1)
                callback_task, callback_agent, callback_thread = runtime.started[0]
                self.assertEqual(callback_agent.agent_id, ravi.agent_id)
                self.assertEqual(callback_task.kind, AgentTaskKind.REVIEW)
                self.assertIn("respond in the same thread", callback_task.prompt)
                self.assertEqual(callback_thread.thread_ts, "171.thread")
                self.assertNotIn("Finish and free up", str(gateway.thread_replies[-1]["blocks"]))
                self.assertEqual(
                    store.get_agent_task(parent.task_id).status,
                    AgentTaskStatus.ACTIVE,
                )
            finally:
                store.close()

    def test_runtime_subtask_exit_clears_request_message_status(self):
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
                self.assertIn(("C1", "171.user", "white_check_mark"), gateway.removed_reactions)
                self.assertNotIn(("C1", "171.user", "white_check_mark"), gateway.reactions)
                self.assertEqual(store.get_agent_task(task.task_id).status, AgentTaskStatus.DONE)
            finally:
                store.close()

    def test_runtime_agent_message_clears_answered_request_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "answer follow-ups", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        "request_message_ts": "171.user2",
                        "request_message_ts_history": ["171.user1"],
                    },
                )
                store.upsert_agent_task(task)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                handled = controller.handle_runtime_agent_message(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.thread"),
                    "Answered both questions.",
                    "171.agent",
                )

                self.assertFalse(handled)
                for message_ts in ("171.user1", "171.user2"):
                    self.assertIn(
                        ("C1", message_ts, "hourglass_flowing_sand"),
                        gateway.removed_reactions,
                    )
                    self.assertIn(("C1", message_ts, "eyes"), gateway.removed_reactions)
            finally:
                store.close()

    def test_runtime_agent_message_keeps_top_level_task_hourglass_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "keep the parent active", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={"request_message_ts": "171.thread"},
                )
                store.upsert_agent_task(task)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                handled = controller.handle_runtime_agent_message(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.thread"),
                    "I am still working on this.",
                    "171.agent",
                )

                self.assertFalse(handled)
                self.assertNotIn(
                    ("C1", "171.thread", "hourglass_flowing_sand"),
                    gateway.removed_reactions,
                )
                self.assertNotIn(("C1", "171.thread", "eyes"), gateway.removed_reactions)
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
                self.assertIn(("C1", "171.user", "white_check_mark"), gateway.removed_reactions)
                self.assertNotIn(("C1", "171.user", "white_check_mark"), gateway.reactions)
            finally:
                store.close()

    def test_mark_task_complete_surfaces_lingering_thread_reactions(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "wrap up", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.thread",
                    metadata={"request_message_ts": "171.user1"},
                )
                store.upsert_agent_task(task)
                gateway.thread_history_messages[("C1", "171.thread")] = [
                    {
                        "ts": "171.000099",
                        "text": "follow-up the agent forgot",
                        "thread_ts": "171.thread",
                    },
                ]
                gateway.reactions.append(("C1", "171.000099", "eyes"))
                gateway.reactions.append(("C1", "171.000099", "inbox_tray"))
                gateway.reactions.append(("C1", "171.000099", "hourglass_flowing_sand"))
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller._mark_task_complete(
                    task,
                    SlackThreadRef("C1", "171.thread"),
                    include_thread=True,
                )

                warnings = [
                    reply
                    for reply in gateway.thread_replies
                    if "never marked complete" in reply["text"]
                ]
                self.assertEqual(len(warnings), 1, gateway.thread_replies)
                self.assertIn("p171000099", warnings[0]["text"])
                self.assertIn(("C1", "171.000099", "eyes"), gateway.removed_reactions)
                self.assertIn(("C1", "171.000099", "inbox_tray"), gateway.removed_reactions)
                self.assertIn(
                    ("C1", "171.000099", "hourglass_flowing_sand"),
                    gateway.removed_reactions,
                )
            finally:
                store.close()

    def test_mark_task_complete_skips_warning_when_thread_is_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "clean exit", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.thread",
                    metadata={"request_message_ts": "171.user1"},
                )
                store.upsert_agent_task(task)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller._mark_task_complete(
                    task,
                    SlackThreadRef("C1", "171.thread"),
                    include_thread=True,
                )

                warnings = [
                    reply
                    for reply in gateway.thread_replies
                    if "never marked complete" in reply["text"]
                ]
                self.assertEqual(warnings, [])
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
                    "T1",
                    SlackThreadRef("C1", "171.000001", "171.000001"),
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.post_roster("C1")

                self.assertIn("0 available, 1 occupied", gateway.posts[-1]["text"])
                self.assertIn(
                    "codex session outside Slack: update the docs", str(gateway.posts[-1]["blocks"])
                )
                blocks = str(gateway.posts[-1]["blocks"])
                self.assertIn("Detach", blocks)
                self.assertIn("Open thread", blocks)
                self.assertIn("'url': 'https://example.slack.com/archives/C1/p", blocks)
                action_block = next(
                    block
                    for block in gateway.posts[-1]["blocks"]
                    if block.get("block_id") == f"team.agent.actions.{agents[0].agent_id}"
                )
                self.assertEqual(
                    [action["text"]["text"] for action in action_block["elements"]],
                    ["Detach", "Open thread", "Fire"],
                )
                self.assertEqual(
                    action_block["elements"][0]["action_id"], "external.session.detach"
                )
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
                        "Detach",
                        "Open thread",
                        "Fire",
                    ],
                )
                self.assertEqual(actions[0]["action_id"], "external.session.detach")
                self.assertIn("'url': 'https://example.slack.com/archives/C1/p", blocks)
            finally:
                store.close()

    def test_roster_shows_external_session_detach_without_thread_link(self):
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
                        "Detach",
                        "Fire",
                    ],
                )
                self.assertEqual(actions[0]["action_id"], "external.session.detach")
            finally:
                store.close()

    def test_external_session_detach_button_ignores_session_and_refreshes_roster(self):
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
                store.set_setting("external_session_pending.codex.s1", "now")
                store.set_setting("external_session_live_target.codex.s1", "123")
                store.set_setting("external_session_missing_target.codex.s1", "123")
                store.set_setting("external_session_summary.codex.s1", "update the docs")
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
                                    "external.session.detach",
                                    provider=Provider.CODEX.value,
                                    session_id="s1",
                                )
                            }
                        ],
                    }
                )

                self.assertIsNone(store.get_setting("external_session_agent.codex.s1"))
                self.assertIsNone(store.get_setting("external_session_pending.codex.s1"))
                self.assertIsNone(store.get_setting("external_session_live_target.codex.s1"))
                self.assertIsNone(store.get_setting("external_session_missing_target.codex.s1"))
                self.assertIsNone(store.get_setting("external_session_summary.codex.s1"))
                self.assertIsNotNone(store.get_setting("external_session_ignored.codex.s1"))
                self.assertIsNone(
                    store.get_slack_thread_for_session(Provider.CODEX, "s1", "local", "C1")
                )
                self.assertIn(
                    "Detached this agent from the external session.",
                    gateway.thread_replies[-1]["text"],
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

    def test_external_session_detach_button_resumes_pending_specific_request(self):
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
                                    "external.session.detach",
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
                    slash_command="/slackgentic-dev",
                )

                controller.post_channel_overview("C1")

                text = gateway.posts[-1]["text"]
                self.assertIn("@agentname", text)
                self.assertIn("somebody ...", text)
                self.assertIn("/slackgentic-dev <command>", text)
                self.assertIn("/slackgentic-dev hire 3 agents", text)
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
                self.assertEqual(
                    gateway.reactions,
                    [
                        ("C1", "171.000001", "eyes"),
                        ("C1", "171.000001", "hourglass_flowing_sand"),
                    ],
                )
                self.assertIn(("C1", "171.000001", "eyes"), gateway.removed_reactions)
                self.assertIn(
                    ("C1", "171.000001", "white_check_mark"),
                    gateway.removed_reactions,
                )
            finally:
                store.close()

    def test_channel_work_request_prompt_keywords_do_not_checkmark_until_done(self):
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
                            "text": (
                                "Can you explain how this works with "
                                "https://github.com/acme/app/pull/42?"
                            ),
                            "ts": "171.000001",
                        }
                    }
                )

                self.assertIn(("C1", "171.000001", "eyes"), gateway.reactions)
                self.assertIn(
                    ("C1", "171.000001", "hourglass_flowing_sand"),
                    gateway.reactions,
                )
                self.assertNotIn(
                    ("C1", "171.000001", "white_check_mark"),
                    gateway.reactions,
                )
            finally:
                store.close()

    def test_channel_work_request_includes_linked_slack_thread_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                linked_url = (
                    "https://example.slack.com/archives/C2/"
                    "p1712345680000002?thread_ts=1712345680.000001&cid=C2"
                )
                gateway.thread_history_messages[("C2", "1712345680.000001")] = [
                    {
                        "type": "message",
                        "channel": "C2",
                        "user": "U2",
                        "text": "The issue: agent missed the linked PR request.",
                        "ts": "1712345680.000001",
                    },
                    {
                        "type": "message",
                        "channel": "C2",
                        "username": "agent",
                        "text": "I only see naming artifacts, not code.",
                        "ts": "1712345680.000002",
                        "thread_ts": "1712345680.000001",
                    },
                ]
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
                            "text": f"Fix this issue {linked_url}",
                            "ts": "171.000001",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                task = runtime.started[0][0]
                thread_context = task.metadata["thread_context"]
                self.assertIn("Linked Slack thread from task prompt", thread_context)
                self.assertIn("The issue: agent missed the linked PR request.", thread_context)
                self.assertIn("I only see naming artifacts, not code.", thread_context)
            finally:
                store.close()

    def test_top_level_request_with_same_channel_link_keeps_new_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                parent_ts = "1712345670.000001"
                reply_ts = "1712345670.000002"
                gateway.thread_history_messages[("C1", parent_ts)] = [
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "U2",
                        "text": "Original request kicked off here.",
                        "ts": parent_ts,
                    },
                    {
                        "type": "message",
                        "channel": "C1",
                        "username": "agent",
                        "text": "Earlier reply from the agent.",
                        "ts": reply_ts,
                        "thread_ts": parent_ts,
                    },
                ]
                linked_url = (
                    "https://example.slack.com/archives/C1/"
                    f"p{reply_ts.replace('.', '')}?thread_ts={parent_ts}&cid=C1"
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
                            "text": f"Fix the threading bug from {linked_url}",
                            "ts": "1712345680.000001",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                task = runtime.started[0][0]
                self.assertEqual(task.thread_ts, "1712345680.000001")
                self.assertEqual(
                    gateway.thread_replies[0]["thread"].thread_ts,
                    "1712345680.000001",
                )
                self.assertEqual(
                    gateway.thread_replies[0]["thread"].message_ts,
                    "1712345680.000001",
                )
                self.assertIn(
                    "Original request kicked off here.",
                    task.metadata["thread_context"],
                )
            finally:
                store.close()

    def test_top_level_request_with_route_intent_uses_linked_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                parent_ts = "1712345670.000001"
                reply_ts = "1712345670.000002"
                gateway.thread_history_messages[("C1", parent_ts)] = [
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "U2",
                        "text": "Original request kicked off here.",
                        "ts": parent_ts,
                    },
                    {
                        "type": "message",
                        "channel": "C1",
                        "username": "agent",
                        "text": "Earlier reply from the agent.",
                        "ts": reply_ts,
                        "thread_ts": parent_ts,
                    },
                ]
                linked_url = (
                    "https://example.slack.com/archives/C1/"
                    f"p{reply_ts.replace('.', '')}?thread_ts={parent_ts}&cid=C1"
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
                            "text": f"somebody continue there and fix the threading bug {linked_url}",
                            "ts": "1712345680.000001",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                task = runtime.started[0][0]
                self.assertEqual(task.thread_ts, parent_ts)
                self.assertEqual(
                    gateway.thread_replies[0]["thread"].thread_ts,
                    parent_ts,
                )
                self.assertEqual(
                    gateway.thread_replies[0]["thread"].message_ts,
                    "1712345680.000001",
                )
            finally:
                store.close()

    def test_top_level_request_resolves_reply_permalink_context_to_parent_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                parent_ts = "1712345670.000001"
                reply_ts = "1712345670.000002"
                # Permalink with no ?thread_ts= query — points at the reply.
                # Slack's conversations.replies returns the whole thread with the
                # parent first when called with either parent or reply ts.
                gateway.thread_history_messages[("C1", reply_ts)] = [
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "U2",
                        "text": "Original request kicked off here.",
                        "ts": parent_ts,
                    },
                    {
                        "type": "message",
                        "channel": "C1",
                        "username": "agent",
                        "text": "Earlier reply from the agent.",
                        "ts": reply_ts,
                        "thread_ts": parent_ts,
                    },
                ]
                linked_url = f"https://example.slack.com/archives/C1/p{reply_ts.replace('.', '')}"
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
                            "text": f"Fix the threading bug from {linked_url}",
                            "ts": "1712345680.000001",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                task = runtime.started[0][0]
                self.assertEqual(task.thread_ts, "1712345680.000001")
                self.assertEqual(
                    gateway.thread_replies[0]["thread"].thread_ts,
                    "1712345680.000001",
                )
                self.assertIn(
                    "Original request kicked off here.",
                    task.metadata["thread_context"],
                )
            finally:
                store.close()

    def test_top_level_request_with_cross_channel_link_does_not_redirect(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                foreign_parent = "1712345670.000003"
                gateway.thread_history_messages[("C2", foreign_parent)] = [
                    {
                        "type": "message",
                        "channel": "C2",
                        "user": "U2",
                        "text": "Different channel context.",
                        "ts": foreign_parent,
                    }
                ]
                linked_url = (
                    "https://example.slack.com/archives/C2/"
                    f"p{foreign_parent.replace('.', '')}?thread_ts={foreign_parent}&cid=C2"
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
                            "text": f"Look at {linked_url} and fix it",
                            "ts": "1712345680.000002",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                task = runtime.started[0][0]
                self.assertEqual(task.thread_ts, "1712345680.000002")
                self.assertEqual(
                    gateway.thread_replies[0]["thread"].thread_ts,
                    "1712345680.000002",
                )
            finally:
                store.close()

    def test_thread_reply_with_link_keeps_user_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(1, 0):
                    store.upsert_team_agent(agent)
                other_parent = "1712345670.000004"
                gateway.thread_history_messages[("C1", other_parent)] = [
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "U2",
                        "text": "An unrelated thread to reference.",
                        "ts": other_parent,
                    }
                ]
                linked_url = (
                    "https://example.slack.com/archives/C1/"
                    f"p{other_parent.replace('.', '')}?thread_ts={other_parent}&cid=C1"
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
                            "thread_ts": "1712345680.000003",
                            "text": f"Please look at {linked_url}",
                            "ts": "1712345680.000004",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                task = runtime.started[0][0]
                self.assertEqual(task.thread_ts, "1712345680.000003")
                self.assertEqual(
                    gateway.thread_replies[0]["thread"].thread_ts,
                    "1712345680.000003",
                )
            finally:
                store.close()

    def test_thread_work_request_with_original_thread_link_routes_to_linked_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(2, 0)
                for agent in agents:
                    store.upsert_team_agent(agent)
                current_parent = "1712345680.000003"
                original_parent = "1712345670.000004"
                parent_task = replace(
                    create_agent_task(agents[0], "triage the stuck thread", "C1"),
                    status=AgentTaskStatus.DONE,
                    thread_ts=current_parent,
                    parent_message_ts=current_parent,
                )
                store.upsert_agent_task(parent_task)
                gateway.thread_history_messages[("C1", original_parent)] = [
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "U2",
                        "text": "Original vendor analysis thread.",
                        "ts": original_parent,
                    }
                ]
                linked_url = (
                    "https://example.slack.com/archives/C1/"
                    f"p{original_parent.replace('.', '')}?thread_ts={original_parent}&cid=C1"
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
                            "thread_ts": current_parent,
                            "text": (
                                "somebody figure out why this unrelated thread got replies "
                                f"instead of the original {linked_url} one"
                            ),
                            "ts": "1712345680.000004",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                task, _, thread = runtime.started[0]
                self.assertEqual(task.thread_ts, original_parent)
                self.assertEqual(thread.thread_ts, original_parent)
                self.assertEqual(
                    gateway.thread_replies[0]["thread"].thread_ts,
                    original_parent,
                )
                self.assertIn(
                    "Original vendor analysis thread.",
                    task.metadata["thread_context"],
                )
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

    def test_slack_message_backfill_reprocesses_acknowledged_message_without_task(self):
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
                controller._mark_slack_message_processed("C1", "171.000010")
                gateway.history_messages.append(
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "U1",
                        "text": "Update docs after interrupted restart",
                        "ts": "171.000010",
                        "reactions": [{"name": "white_check_mark"}],
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

                recovered = backfill.recover_since("171.000005")

                self.assertEqual(recovered, 1)
                self.assertEqual(len(store.list_agent_tasks()), 1)
                self.assertEqual(
                    store.list_agent_tasks()[0].prompt,
                    "Update docs after interrupted restart",
                )
                self.assertEqual(len(runtime.started), 1)
            finally:
                store.close()

    def test_slack_message_backfill_skips_queued_followup_message_with_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "initial task", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        "queued_thread_followups": [
                            {
                                "prompt": "Queued follow-up",
                                "message_ts": "171.000010",
                                "requested_by_slack_user": "U1",
                                "created_at": utc_now().isoformat(),
                            }
                        ]
                    },
                )
                store.upsert_agent_task(task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller._mark_slack_message_processed("C1", "171.000010")
                gateway.history_messages.append(
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "U1",
                        "text": "Queued follow-up",
                        "ts": "171.000010",
                        "thread_ts": "171.thread",
                        "reactions": [{"name": "inbox_tray"}],
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

                recovered = backfill.recover_since("171.000005")

                self.assertEqual(recovered, 0)
                self.assertEqual(runtime.started, [])
                self.assertEqual(len(store.list_agent_tasks()), 1)
            finally:
                store.close()

    def test_slack_message_backfill_skips_stored_agent_request_message(self):
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
                store.create_slack_agent_request(
                    "approval-1",
                    "Claude",
                    "agent/requestApproval",
                    {"title": "Somebody update docs"},
                    SlackThreadRef("C1", "171.thread"),
                    message_ts="171.000010",
                )
                gateway.history_messages.append(
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "USLACKGENTIC",
                        "text": "Somebody update docs",
                        "ts": "171.000010",
                        "thread_ts": "171.thread",
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

                recovered = backfill.recover_since("171.000005")

                self.assertEqual(recovered, 0)
                self.assertEqual(store.list_agent_tasks(), [])
                self.assertEqual(runtime.started, [])
            finally:
                store.close()

    def test_slack_message_backfill_skips_pending_work_request_message(self):
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
                            "text": "Update docs when someone is available",
                            "ts": "171.000010",
                        }
                    }
                )
                pending = store.list_pending_work_requests(channel_id="C1")
                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0].extra_metadata["request_message_ts"], "171.000010")
                gateway.history_messages.append(
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "U1",
                        "text": "Update docs when someone is available",
                        "ts": "171.000010",
                        "reactions": [{"name": "white_check_mark"}],
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

                recovered = backfill.recover_since("171.000005")

                self.assertEqual(recovered, 0)
                self.assertEqual(len(store.list_pending_work_requests(channel_id="C1")), 1)
                self.assertEqual(store.list_agent_tasks(), [])
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

    def test_slack_message_backfill_skips_existing_task_thread_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "Good bye", "C1"),
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.000001",
                    parent_message_ts="171.000002",
                    metadata={
                        "assignment_prompt": "Somebody summarize pyproject",
                        "request_message_ts": "171.000010",
                    },
                )
                store.upsert_agent_task(task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                gateway.history_messages.append(
                    {
                        "type": "message",
                        "channel": "C1",
                        "user": "U1",
                        "text": "Somebody summarize pyproject",
                        "ts": "171.000001",
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

                recovered = backfill.recover_since("171.000000")

                self.assertEqual(recovered, 0)
                self.assertEqual(len(store.list_agent_tasks(include_done=True)), 1)
                self.assertEqual(runtime.started, [])
            finally:
                store.close()

    def test_pending_duplicate_thread_root_is_cancelled_when_capacity_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "Good bye", "C1"),
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.000001",
                    parent_message_ts="171.000002",
                    metadata={
                        "assignment_prompt": "Somebody summarize pyproject",
                        "request_message_ts": "171.000010",
                    },
                )
                store.upsert_agent_task(task)
                pending = store.create_pending_work_request(
                    SlackThreadRef("C1", "171.000001"),
                    WorkRequest(
                        prompt="Somebody summarize pyproject",
                        assignment_mode=AssignmentMode.ANYONE,
                    ),
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                resumed = controller.resume_pending_work_requests("C1")

                self.assertEqual(resumed, 0)
                self.assertEqual(runtime.started, [])
                self.assertEqual(len(store.list_agent_tasks(include_done=True)), 1)
                row = store.conn.execute(
                    "SELECT status FROM pending_work_requests WHERE pending_id = ?",
                    (pending.pending_id,),
                ).fetchone()
                self.assertEqual(row["status"], "cancelled")
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
                self.assertIn(
                    "*:zap: Dangerous mode*",
                    gateway.thread_replies[0]["text"],
                )
                self.assertIn(
                    "*:zap: Dangerous mode*",
                    gateway.thread_replies[0]["blocks"][0]["text"]["text"],
                )
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
                    "171.agent",
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
                self.assertEqual(task.metadata["request_message_ts"], "171.agent")
                self.assertIn(("C1", "171.agent", "eyes"), gateway.reactions)
                self.assertIn(("C1", "171.agent", "hourglass_flowing_sand"), gateway.reactions)

                controller.handle_runtime_task_done(task, agent, thread)

                self.assertIn(("C1", "171.agent", "eyes"), gateway.removed_reactions)
                self.assertIn(
                    ("C1", "171.agent", "hourglass_flowing_sand"),
                    gateway.removed_reactions,
                )
                self.assertIn(("C1", "171.agent", "white_check_mark"), gateway.removed_reactions)
                self.assertNotIn(("C1", "171.agent", "white_check_mark"), gateway.reactions)
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

    def test_same_thread_followup_waits_when_agent_is_busy_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                agent = replace(agent, handle="milo")
                store.upsert_team_agent(agent)
                prior_task = replace(
                    create_agent_task(
                        agent,
                        "explain the first bug",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-prior",
                )
                busy_task = replace(
                    create_agent_task(
                        agent,
                        "ship another fix",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="172.thread",
                    parent_message_ts="172.parent",
                )
                store.upsert_agent_task(prior_task)
                store.upsert_agent_task(busy_task)
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
                            "text": "what happened next?",
                            "ts": "171.followup",
                            "thread_ts": "171.thread",
                        }
                    }
                )

                self.assertEqual(runtime.started, [])
                pending = store.list_pending_work_requests(channel_id="C1")
                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0].thread_ts, "171.thread")
                self.assertEqual(pending[0].request.requested_handle, "milo")
                self.assertIn("what happened next?", pending[0].request.prompt)
                self.assertIn("That specific agent is busy", gateway.thread_replies[-1]["text"])
            finally:
                store.close()

    def test_same_thread_review_continuation_frees_reviewer_after_exit(self):
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
                reviewer = next(
                    agent for agent in agents if agent.provider_preference == Provider.CLAUDE
                )
                original = replace(original, handle="ravi")
                reviewer = replace(reviewer, handle="livia", full_name="Livia Singh")
                store.upsert_team_agent(original)
                store.upsert_team_agent(reviewer)
                original_task = replace(
                    create_agent_task(
                        original,
                        "design the install flow",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                )
                prior_review = replace(
                    create_agent_task(
                        reviewer,
                        "review v1",
                        "C1",
                        requested_by_slack_user="U1",
                        kind=AgentTaskKind.REVIEW,
                    ),
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.thread",
                    parent_message_ts="171.review",
                    metadata={
                        "parent_task_id": original_task.task_id,
                        "parent_agent_id": original.agent_id,
                    },
                )
                store.upsert_agent_task(original_task)
                store.upsert_agent_task(prior_review)
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
                            "text": "@livia review the v2 patch",
                            "ts": "171.000003",
                            "thread_ts": "171.thread",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                task, agent, thread = runtime.started[0]
                self.assertEqual(task.task_id, prior_review.task_id)
                self.assertEqual(task.metadata["parent_task_id"], prior_review.task_id)
                self.assertEqual(task.kind, AgentTaskKind.REVIEW)

                controller.handle_runtime_task_done(task, agent, thread)

                self.assertEqual(
                    store.get_agent_task(prior_review.task_id).status,
                    AgentTaskStatus.DONE,
                )
                self.assertIsNone(store.active_task_for_agent(reviewer.agent_id))
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

    def test_thread_work_request_includes_linked_slack_thread_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                for agent in build_initial_model_team(2, 0):
                    store.upsert_team_agent(agent)
                linked_url = (
                    "https://example.slack.com/archives/C2/"
                    "p1712345680000002?thread_ts=1712345680.000001&cid=C2"
                )
                gateway.thread_history_messages[("C2", "1712345680.000001")] = [
                    {
                        "type": "message",
                        "channel": "C2",
                        "user": "U2",
                        "text": "Linked issue root: preserve permalink context.",
                        "ts": "1712345680.000001",
                    }
                ]
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
                parent_task = runtime.started[0][0]

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"somebody fix this issue {linked_url}",
                            "ts": "171.000002",
                            "thread_ts": parent_task.thread_ts,
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 2)
                followup = runtime.started[-1][0]
                thread_context = followup.metadata["thread_context"]
                self.assertIn("Original task: summarize pyproject", thread_context)
                self.assertIn("Linked Slack thread from task prompt", thread_context)
                self.assertIn("Linked issue root: preserve permalink context.", thread_context)
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
                current = store.get_agent_task(task.task_id)
                assert current is not None
                self.assertEqual(current.metadata["request_message_ts"], "171.000002")
                self.assertIn("171.000001", current.metadata["request_message_ts_history"])
                agent = store.get_team_agent(task.agent_id)
                assert agent is not None

                controller.handle_runtime_task_done(
                    current,
                    agent,
                    SlackThreadRef("C1", task.thread_ts),
                )

                self.assertIn(
                    ("C1", "171.000002", "hourglass_flowing_sand"),
                    gateway.removed_reactions,
                )
                self.assertIn(("C1", "171.000002", "eyes"), gateway.removed_reactions)
                self.assertNotIn(("C1", "171.000002", "white_check_mark"), gateway.reactions)
            finally:
                store.close()

    def test_detached_task_thread_followups_preserve_request_message_history(self):
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
                            "text": "Are you stuck?",
                            "ts": "171.000002",
                            "thread_ts": task.thread_ts,
                        }
                    }
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "What is happening?",
                            "ts": "171.000003",
                            "thread_ts": task.thread_ts,
                        }
                    }
                )

                current = store.get_agent_task(task.task_id)
                assert current is not None
                self.assertEqual(current.metadata["request_message_ts"], "171.000003")
                self.assertCountEqual(
                    current.metadata["request_message_ts_history"],
                    ["171.000001", "171.000002"],
                )
                agent = store.get_team_agent(task.agent_id)
                assert agent is not None

                controller.handle_runtime_agent_message(
                    current,
                    agent,
                    SlackThreadRef("C1", task.thread_ts),
                    "Answered the follow-ups.",
                    "171.agent",
                )

                for message_ts in ("171.000002", "171.000003"):
                    self.assertIn(
                        ("C1", message_ts, "hourglass_flowing_sand"),
                        gateway.removed_reactions,
                    )
                    self.assertIn(("C1", message_ts, "eyes"), gateway.removed_reactions)
            finally:
                store.close()

    def test_task_thread_question_is_sent_to_runtime_with_answer_first_instruction(self):
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
                            "text": "Is there a better way to do this?",
                            "ts": "171.000002",
                            "thread_ts": task.thread_ts,
                        }
                    }
                )

                self.assertEqual(len(runtime.sent), 1)
                task_id, message = runtime.sent[0]
                self.assertEqual(task_id, task.task_id)
                self.assertIn("answer it explicitly in Slack before continuing", message)
                self.assertIn("Is there a better way to do this?", message)
                current = store.get_agent_task(task.task_id)
                assert current is not None
                self.assertEqual(current.metadata["request_message_ts"], "171.000002")
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
                    f"\n\n@{codex_agent.handle} please reply with DELEGATED_OK",
                    gateway.thread_replies[-1]["text"],
                )
                self.assertNotIn("for <@U1>", gateway.thread_replies[-1]["text"])
            finally:
                store.close()

    def test_thread_reply_during_running_task_does_not_spawn_parallel_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(1, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                claude_agent = next(
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
                            "text": f"@{claude_agent.handle} long-running work",
                            "ts": "171.000001",
                        }
                    }
                )
                parent_task = store.list_agent_tasks(include_done=True)[0]
                runtime.running_task_ids.add(parent_task.task_id)
                started_before = len(runtime.started)

                for ping_ts in ("171.000002", "171.000003"):
                    controller.handle_event(
                        {
                            "event": {
                                "type": "message",
                                "channel": "C1",
                                "user": "U1",
                                "text": "status?",
                                "ts": ping_ts,
                                "thread_ts": parent_task.thread_ts,
                            }
                        }
                    )

                self.assertEqual(
                    len(runtime.started),
                    started_before,
                    "follow-up pings must not spawn parallel start_task calls "
                    "while the previous run is still executing",
                )
                tasks = store.list_agent_tasks(include_done=True)
                self.assertEqual(
                    [task.task_id for task in tasks],
                    [parent_task.task_id],
                    "no new agent task rows should be created for in-flight pings",
                )
            finally:
                store.close()

    def test_delegate_to_original_posts_handoff_once_when_continuation_fails(self):
        class UnresumableRuntime(DetachedRuntime):
            def start_task(self, task, agent, thread):
                self.started.append((task, agent, thread))
                return False

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = UnresumableRuntime()
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
                gateway.post_thread_reply(
                    runtime.started[-1][2],
                    "I am asking the original agent to reply with DELEGATED_OK.",
                    persona=reviewer,
                )

                replies_before = len(gateway.thread_replies)

                controller.handle_runtime_task_done(
                    delegated_by_reviewer,
                    reviewer,
                    SlackThreadRef("C1", parent_task.thread_ts),
                )

                new_replies = gateway.thread_replies[replies_before:]
                handoff_request_marker = f"@{codex_agent.handle} please reply with DELEGATED_OK"
                handoff_request_count = sum(
                    1 for reply in new_replies if handoff_request_marker in reply["text"]
                )
                self.assertEqual(
                    handoff_request_count,
                    1,
                    f"expected one handoff request, got {handoff_request_count}: "
                    f"{[r['text'] for r in new_replies]}",
                )
                handoff_assignment_count = sum(
                    1
                    for reply in new_replies
                    if reply["text"].startswith(f"Got it, @{reviewer.handle}.")
                )
                self.assertEqual(handoff_assignment_count, 1)
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
                    f"\n\n@{target_agent.handle} please take the task I assigned above.",
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

    def test_parent_review_handoff_keeps_original_request_in_progress(self):
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
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                parent_task = replace(
                    create_agent_task(
                        original,
                        "draft the install flow",
                        "C1",
                        requested_by_slack_user="U1",
                    ),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.assignment",
                    metadata={"request_message_ts": "171.user"},
                )
                store.upsert_agent_task(parent_task)

                handled = controller.handle_runtime_agent_message(
                    parent_task,
                    original,
                    SlackThreadRef("C1", "171.thread"),
                    "somebody review my install-flow draft",
                    "171.agent",
                )
                controller.handle_runtime_task_done(
                    parent_task,
                    original,
                    SlackThreadRef("C1", "171.thread"),
                )

                self.assertTrue(handled)
                self.assertNotIn(
                    ("C1", "171.user", "hourglass_flowing_sand"),
                    gateway.removed_reactions,
                )
                self.assertNotIn(("C1", "171.user", "eyes"), gateway.removed_reactions)
                self.assertIn(("C1", "171.agent", "hourglass_flowing_sand"), gateway.reactions)
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

    def test_thread_context_resolves_known_human_user_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                controller._remember_human_user(
                    "U12345678",
                    {
                        "display_name": "Ilshat",
                        "image_72": "https://example.com/avatar.png",
                    },
                )
                gateway.thread_history_messages[("C1", "171.thread")] = [
                    {
                        "user": "U12345678",
                        "text": "please check <@U12345678> and @U12345678",
                        "ts": "171.000001",
                    }
                ]

                context = controller._thread_context("C1", "171.thread")

                self.assertEqual(context, "Ilshat: please check Ilshat and Ilshat")
                self.assertNotIn("U12345678", context)
            finally:
                store.close()

    def test_thread_context_hides_unresolved_raw_human_user_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                gateway.thread_history_messages[("C1", "171.thread")] = [
                    {
                        "user": "UUNKNOWN1",
                        "text": "@UUNKNOWN1 once the PR is updated, this is good to merge",
                        "ts": "171.000001",
                    }
                ]

                context = controller._thread_context("C1", "171.thread")

                self.assertEqual(
                    context,
                    "Slack user: Slack user once the PR is updated, this is good to merge",
                )
                self.assertNotIn("UUNKNOWN1", context)
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

    def test_startup_reconcile_does_not_restart_idle_active_thread_without_run_marker(self):
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

                resumed = controller.cancel_orphaned_active_tasks()

                self.assertEqual(resumed, 0)
                self.assertEqual(runtime.started, [])
            finally:
                store.close()

    def test_startup_reconcile_replays_stranded_queued_followups_without_run_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "initial task", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-queued",
                    metadata={
                        "queued_thread_followups": [
                            {
                                "prompt": "What happened to this queued question?",
                                "message_ts": "171.user",
                                "requested_by_slack_user": "U1",
                                "created_at": utc_now().isoformat(),
                            }
                        ]
                    },
                )
                store.upsert_agent_task(task)
                store.upsert_managed_thread_task(task, SlackThreadRef("C1", "171.thread"))
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                controller._mark_message_queued("C1", "171.user")
                gateway.removed_reactions.clear()

                resumed = controller.cancel_orphaned_active_tasks()

                self.assertEqual(resumed, 1)
                self.assertEqual(len(runtime.started), 1)
                restarted, restarted_agent, restarted_thread = runtime.started[0]
                self.assertEqual(restarted.task_id, task.task_id)
                self.assertEqual(restarted_agent.agent_id, agent.agent_id)
                self.assertEqual(restarted_thread.thread_ts, "171.thread")
                self.assertIn("What happened to this queued question?", restarted.prompt)
                current_task = store.get_agent_task(task.task_id)
                assert current_task is not None
                self.assertEqual(
                    current_task.metadata.get("active_thread_followup_message_ts_values"),
                    ["171.user"],
                )
                self.assertNotIn("queued_thread_followups", current_task.metadata)
                self.assertIn(("C1", "171.user", "inbox_tray"), gateway.removed_reactions)
                self.assertIn(("C1", "171.user", "hourglass_flowing_sand"), gateway.reactions)
            finally:
                store.close()

    def test_startup_reconcile_restarts_interrupted_managed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "continue after restart", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-1",
                    metadata={MANAGED_RUN_STARTED_METADATA_KEY: utc_now().isoformat()},
                )
                store.upsert_agent_task(task)
                store.upsert_managed_thread_task(task, SlackThreadRef("C1", "171.thread"))
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                resumed = controller.cancel_orphaned_active_tasks()

                self.assertEqual(resumed, 1)
                self.assertEqual(len(runtime.started), 1)
                restarted, restarted_agent, restarted_thread = runtime.started[0]
                self.assertEqual(restarted.task_id, task.task_id)
                self.assertEqual(restarted.session_id, "codex-thread-1")
                self.assertEqual(restarted_agent.agent_id, agent.agent_id)
                self.assertEqual(restarted_thread.thread_ts, "171.thread")
                self.assertEqual(managed_run_resume_attempts(restarted), 1)
            finally:
                store.close()

    def test_restarted_managed_run_exit_keeps_root_task_occupied(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "continue after restart", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-1",
                    metadata={
                        MANAGED_RUN_STARTED_METADATA_KEY: utc_now().isoformat(),
                        "request_message_ts": "171.user",
                    },
                )
                store.upsert_agent_task(task)
                store.upsert_managed_thread_task(task, SlackThreadRef("C1", "171.thread"))
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                resumed = controller.cancel_orphaned_active_tasks()
                restarted, restarted_agent, restarted_thread = runtime.started[0]
                controller.handle_runtime_task_done(
                    restarted,
                    restarted_agent,
                    restarted_thread,
                )

                self.assertEqual(resumed, 1)
                current = store.get_agent_task(task.task_id)
                assert current is not None
                self.assertEqual(current.status, AgentTaskStatus.ACTIVE)
                self.assertNotIn(MANAGED_RUN_STARTED_METADATA_KEY, current.metadata)
                self.assertIn(
                    ("C1", "171.user", "hourglass_flowing_sand"), gateway.removed_reactions
                )
                self.assertIn(("C1", "171.user", "eyes"), gateway.removed_reactions)
                self.assertNotIn(("C1", "171.user", "white_check_mark"), gateway.reactions)
            finally:
                store.close()

    def test_startup_reconcile_skips_stale_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                stale_marker = (
                    utc_now() - MANAGED_RUN_MAX_RESUME_AGE - timedelta(seconds=30)
                ).isoformat()
                task = replace(
                    create_agent_task(agent, "stale orphan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-stale",
                    metadata={MANAGED_RUN_STARTED_METADATA_KEY: stale_marker},
                )
                store.upsert_agent_task(task)
                store.upsert_managed_thread_task(task, SlackThreadRef("C1", "171.thread"))
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                resumed = controller.cancel_orphaned_active_tasks()

                self.assertEqual(resumed, 0)
                self.assertEqual(runtime.started, [])
                persisted = store.get_agent_task(task.task_id)
                assert persisted is not None
                self.assertEqual(persisted.status, AgentTaskStatus.CANCELLED)
                self.assertNotIn(MANAGED_RUN_STARTED_METADATA_KEY, persisted.metadata)
            finally:
                store.close()

    def test_startup_reconcile_skips_when_resume_attempts_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "exhausted attempts", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-exhausted",
                    metadata={
                        MANAGED_RUN_STARTED_METADATA_KEY: utc_now().isoformat(),
                        MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY: MANAGED_RUN_MAX_RESUMES,
                    },
                )
                store.upsert_agent_task(task)
                store.upsert_managed_thread_task(task, SlackThreadRef("C1", "171.thread"))
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                resumed = controller.cancel_orphaned_active_tasks()

                self.assertEqual(resumed, 0)
                self.assertEqual(runtime.started, [])
                persisted = store.get_agent_task(task.task_id)
                assert persisted is not None
                self.assertEqual(persisted.status, AgentTaskStatus.CANCELLED)
            finally:
                store.close()

    def test_startup_reconcile_keeps_stale_marker_when_session_still_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                stale_marker = (
                    utc_now() - MANAGED_RUN_MAX_RESUME_AGE - timedelta(seconds=30)
                ).isoformat()
                task = replace(
                    create_agent_task(agent, "long running orphan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-alive",
                    metadata={
                        MANAGED_RUN_STARTED_METADATA_KEY: stale_marker,
                        DANGEROUS_MODE_METADATA_KEY: True,
                    },
                )
                store.upsert_agent_task(task)
                store.upsert_managed_thread_task(task, SlackThreadRef("C1", "171.thread"))
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CODEX,
                        session_id="codex-thread-alive",
                        transcript_path=Path(tmp) / "codex.jsonl",
                        cwd=Path(tmp),
                        started_at=utc_now(),
                        last_seen_at=utc_now(),
                        status=SessionStatus.ACTIVE,
                        control_mode=ControlMode.MANAGED,
                    )
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                resumed = controller.cancel_orphaned_active_tasks()

                self.assertEqual(resumed, 0)
                self.assertEqual(runtime.started, [])
                persisted = store.get_agent_task(task.task_id)
                assert persisted is not None
                self.assertEqual(persisted.status, AgentTaskStatus.ACTIVE)
                self.assertIn(MANAGED_RUN_STARTED_METADATA_KEY, persisted.metadata)
            finally:
                store.close()

    def test_startup_reconcile_keeps_exhausted_attempts_when_session_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "still working", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-resilient",
                    metadata={
                        MANAGED_RUN_STARTED_METADATA_KEY: utc_now().isoformat(),
                        MANAGED_RUN_RESUME_ATTEMPTS_METADATA_KEY: MANAGED_RUN_MAX_RESUMES,
                    },
                )
                store.upsert_agent_task(task)
                store.upsert_managed_thread_task(task, SlackThreadRef("C1", "171.thread"))
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CODEX,
                        session_id="codex-thread-resilient",
                        transcript_path=Path(tmp) / "codex.jsonl",
                        cwd=Path(tmp),
                        started_at=utc_now(),
                        last_seen_at=utc_now(),
                        status=SessionStatus.IDLE,
                        control_mode=ControlMode.MANAGED,
                    )
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                resumed = controller.cancel_orphaned_active_tasks()

                self.assertEqual(resumed, 0)
                self.assertEqual(runtime.started, [])
                persisted = store.get_agent_task(task.task_id)
                assert persisted is not None
                self.assertEqual(persisted.status, AgentTaskStatus.ACTIVE)
            finally:
                store.close()

    def test_startup_reconcile_roster_shows_agent_busy_when_task_kept_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = DetachedRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                stale_marker = (
                    utc_now() - MANAGED_RUN_MAX_RESUME_AGE - timedelta(seconds=30)
                ).isoformat()
                task = replace(
                    create_agent_task(agent, "summarize pyproject", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-roster",
                    metadata={MANAGED_RUN_STARTED_METADATA_KEY: stale_marker},
                )
                store.upsert_agent_task(task)
                store.upsert_managed_thread_task(task, SlackThreadRef("C1", "171.thread"))
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CODEX,
                        session_id="codex-thread-roster",
                        transcript_path=Path(tmp) / "codex.jsonl",
                        cwd=Path(tmp),
                        started_at=utc_now(),
                        last_seen_at=utc_now(),
                        status=SessionStatus.ACTIVE,
                        control_mode=ControlMode.MANAGED,
                    )
                )
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.cancel_orphaned_active_tasks()
                controller.post_roster("C1")

                self.assertIn("0 available, 1 occupied", gateway.posts[-1]["text"])
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
                self.assertIn(("C1", task.thread_ts, "link"), gateway.removed_reactions)
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
                timer = store.create_scheduled_timer(
                    first,
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                    prompt="wake later",
                    due_at=utc_now() + timedelta(minutes=5),
                )
                scheduled = store.create_scheduled_work_request(
                    SlackThreadRef("C1", "171.thread", "171.schedule"),
                    WorkRequest(
                        prompt="queued scheduled work",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=first_agent.handle,
                    ),
                    schedule_kind=ScheduledWorkKind.RECURRING,
                    next_run_at=utc_now() + timedelta(minutes=5),
                    recurrence={
                        "frequency": "daily",
                        "time": "17:00",
                        "timezone": "America/New_York",
                    },
                    timezone="America/New_York",
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
                timer_row = store.conn.execute(
                    "SELECT status FROM scheduled_timers WHERE timer_id = ?",
                    (timer.timer_id,),
                ).fetchone()
                self.assertEqual(timer_row["status"], "cancelled")
                scheduled_row = store.conn.execute(
                    "SELECT status FROM scheduled_work_requests WHERE schedule_id = ?",
                    (scheduled.schedule_id,),
                ).fetchone()
                self.assertEqual(scheduled_row["status"], "cancelled")
                self.assertEqual(
                    {update["ts"] for update in gateway.updates},
                    {"171.parent", "171.bot2"},
                )
                for update in gateway.updates:
                    self.assertFalse(
                        any(block.get("type") == "actions" for block in update["blocks"])
                    )
                self.assertIn(("C1", "171.thread", "white_check_mark"), gateway.reactions)
                self.assertIn(("C1", "171.user2", "white_check_mark"), gateway.removed_reactions)
                self.assertNotIn(("C1", "171.user2", "white_check_mark"), gateway.reactions)
            finally:
                store.close()

    def test_same_thread_replies_while_managed_task_running_are_replayed_as_batch(self):
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
                runtime = DetachedRuntime()
                runtime.running_task_ids.add(task.task_id)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                for text, ts in (
                    ("Do not use manual parsing; use the agent LLM.", "171.user1"),
                    ("Keep each queued follow-up pending until answered.", "171.user2"),
                ):
                    controller.handle_event(
                        {
                            "event": {
                                "type": "message",
                                "channel": "C1",
                                "user": "U1",
                                "text": text,
                                "ts": ts,
                                "thread_ts": "171.thread",
                            }
                        }
                    )

                queued_task = store.get_agent_task(task.task_id)
                assert queued_task is not None
                queued = queued_task.metadata.get("queued_thread_followups")
                self.assertIsInstance(queued, list)
                self.assertEqual(len(queued), 2)
                self.assertEqual(runtime.started, [])
                self.assertIn(("C1", "171.user1", "inbox_tray"), gateway.reactions)
                self.assertIn(("C1", "171.user2", "inbox_tray"), gateway.reactions)
                self.assertNotIn(
                    ("C1", "171.user1", "hourglass_flowing_sand"),
                    gateway.reactions,
                )
                self.assertNotIn(
                    ("C1", "171.user2", "hourglass_flowing_sand"),
                    gateway.reactions,
                )
                self.assertEqual(len(gateway.thread_replies), 1)
                self.assertIn(
                    "could not deliver that follow-up live",
                    gateway.thread_replies[0]["text"],
                )
                self.assertIn("say `stop`", gateway.thread_replies[0]["text"])

                runtime.running_task_ids.clear()
                controller.handle_runtime_task_done(
                    queued_task,
                    agent,
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                )

                self.assertEqual(len(runtime.started), 1)
                resumed_task = runtime.started[0][0]
                self.assertIn("Do not use manual parsing", resumed_task.prompt)
                self.assertIn("Keep each queued follow-up", resumed_task.prompt)
                current_task = store.get_agent_task(task.task_id)
                assert current_task is not None
                self.assertEqual(
                    current_task.metadata.get("active_thread_followup_message_ts_values"),
                    ["171.user1", "171.user2"],
                )
                self.assertNotIn("queued_thread_followups", current_task.metadata)
                self.assertIn(("C1", "171.user1", "inbox_tray"), gateway.removed_reactions)
                self.assertIn(("C1", "171.user2", "inbox_tray"), gateway.removed_reactions)
                self.assertIn(("C1", "171.user1", "hourglass_flowing_sand"), gateway.reactions)
                self.assertIn(("C1", "171.user2", "hourglass_flowing_sand"), gateway.reactions)
                self.assertNotIn(("C1", "171.user1", "white_check_mark"), gateway.reactions)
                self.assertNotIn(("C1", "171.user2", "white_check_mark"), gateway.reactions)

                controller.handle_runtime_task_done(
                    resumed_task,
                    agent,
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                )

                self.assertIn(("C1", "171.user1", "white_check_mark"), gateway.reactions)
                self.assertIn(("C1", "171.user2", "white_check_mark"), gateway.reactions)
                self.assertEqual(len(runtime.started), 1)
                current_task = store.get_agent_task(task.task_id)
                assert current_task is not None
                self.assertNotIn("active_thread_followup_message_ts", current_task.metadata)
                self.assertNotIn("active_thread_followup_message_ts_values", current_task.metadata)
                self.assertNotIn("queued_thread_followups", current_task.metadata)
            finally:
                store.close()

    def test_same_thread_specific_mention_live_delivers_to_running_managed_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                parent_agent, target_agent = build_initial_model_team(1, 1)
                target_agent = replace(
                    target_agent,
                    handle="livia",
                    full_name="Livia Singh",
                    provider_preference=Provider.CLAUDE,
                )
                store.upsert_team_agent(parent_agent)
                store.upsert_team_agent(target_agent)
                parent_task = replace(
                    create_agent_task(parent_agent, "initial task", "C1"),
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                )
                target_task = replace(
                    create_agent_task(target_agent, "take this over", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.livia",
                    session_provider=Provider.CLAUDE,
                    session_id="claude-live-1",
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="claude-live-1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    control_mode=ControlMode.MANAGED,
                )
                store.upsert_agent_task(parent_task)
                store.upsert_agent_task(target_task)
                store.upsert_session(session)
                runtime = DetachedRuntime()
                runtime.running_task_ids.add(target_task.task_id)
                bridge = FakeSessionBridge()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                    session_bridge=bridge,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "@livia look above",
                            "ts": "171.user",
                            "thread_ts": "171.thread",
                        }
                    }
                )

                self.assertEqual(len(bridge.live_sent), 1)
                sent_session, sent_text, sent_thread, sent_user = bridge.live_sent[0]
                self.assertEqual(sent_session.session_id, "claude-live-1")
                self.assertEqual(sent_text, "look above")
                self.assertEqual(sent_thread, SlackThreadRef("C1", "171.thread"))
                self.assertEqual(sent_user, "U1")
                current_task = store.get_agent_task(target_task.task_id)
                assert current_task is not None
                self.assertNotIn("queued_thread_followups", current_task.metadata)
                self.assertEqual(current_task.metadata.get("request_message_ts"), "171.user")
                self.assertEqual(runtime.started, [])
                self.assertIn(("C1", "171.user", "hourglass_flowing_sand"), gateway.reactions)
            finally:
                store.close()

    def test_stop_in_task_thread_interrupts_running_managed_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "long running task", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                )
                store.upsert_agent_task(task)
                runtime = FakeRuntime()
                runtime.running_task_ids.add(task.task_id)
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
                            "text": "stop",
                            "ts": "171.stop",
                            "thread_ts": "171.thread",
                        }
                    }
                )

                self.assertEqual(runtime.interrupted, [task.task_id])
                self.assertEqual(runtime.sent, [])
                self.assertEqual(runtime.interrupted_sent, [])
                self.assertEqual(runtime.started, [])
                self.assertIn(task.task_id, runtime.running_task_ids)
                self.assertIn(("C1", "171.stop", "white_check_mark"), gateway.reactions)
                self.assertIn(("C1", "171.stop", "eyes"), gateway.removed_reactions)
                self.assertIn("Interrupted the current run", gateway.thread_replies[-1]["text"])
            finally:
                store.close()

    def test_stop_in_task_thread_replays_queued_followups_after_interrupt(self):
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
                runtime = DetachedRuntime()
                runtime.running_task_ids.add(task.task_id)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                for text, ts in (
                    ("Switch to the LLM schedule resolver.", "171.user1"),
                    ("Also keep the stop command.", "171.user2"),
                ):
                    controller.handle_event(
                        {
                            "event": {
                                "type": "message",
                                "channel": "C1",
                                "user": "U1",
                                "text": text,
                                "ts": ts,
                                "thread_ts": "171.thread",
                            }
                        }
                    )

                self.assertIn(("C1", "171.user1", "inbox_tray"), gateway.reactions)
                self.assertIn(("C1", "171.user2", "inbox_tray"), gateway.reactions)
                self.assertNotIn(
                    ("C1", "171.user1", "hourglass_flowing_sand"),
                    gateway.reactions,
                )
                self.assertNotIn(
                    ("C1", "171.user2", "hourglass_flowing_sand"),
                    gateway.reactions,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "stop",
                            "ts": "171.stop",
                            "thread_ts": "171.thread",
                        }
                    }
                )

                self.assertEqual(runtime.interrupted, [task.task_id])
                self.assertEqual(runtime.started, [])
                self.assertEqual(len(runtime.interrupted_sent), 1)
                sent_task_id, sent_prompt = runtime.interrupted_sent[0]
                self.assertEqual(sent_task_id, task.task_id)
                self.assertIn("Switch to the LLM schedule resolver", sent_prompt)
                self.assertIn("Also keep the stop command", sent_prompt)
                self.assertIn(task.task_id, runtime.running_task_ids)
                self.assertIn(("C1", "171.stop", "white_check_mark"), gateway.reactions)
                current_task = store.get_agent_task(task.task_id)
                assert current_task is not None
                self.assertEqual(
                    current_task.metadata.get("active_thread_followup_message_ts_values"),
                    ["171.user1", "171.user2"],
                )
                self.assertNotIn("queued_thread_followups", current_task.metadata)
                self.assertIn(("C1", "171.user1", "inbox_tray"), gateway.removed_reactions)
                self.assertIn(("C1", "171.user2", "inbox_tray"), gateway.removed_reactions)
                self.assertIn(("C1", "171.user1", "hourglass_flowing_sand"), gateway.reactions)
                self.assertIn(("C1", "171.user2", "hourglass_flowing_sand"), gateway.reactions)
                self.assertIn("sent the queued follow-ups", gateway.thread_replies[-1]["text"])
            finally:
                store.close()

    def test_user_schedule_message_starts_agent_resolution_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                runtime = FakeRuntime()
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
                            "text": "schedule @avery to check CI every day at 5pm ET",
                            "ts": "171.schedule",
                        }
                    }
                )

                rows = store.conn.execute("SELECT * FROM scheduled_work_requests").fetchall()
                self.assertEqual(rows, [])
                self.assertEqual(len(runtime.started), 1)
                task, started_agent, thread = runtime.started[0]
                self.assertEqual(started_agent.agent_id, agent.agent_id)
                self.assertEqual(thread.thread_ts, "171.schedule")
                self.assertIn(AGENT_SCHEDULE_SIGNAL_PREFIX, task.prompt)
                self.assertIn("schedule @avery to check CI every day at 5pm ET", task.prompt)
                self.assertTrue(task.metadata[SCHEDULE_RESOLUTION_METADATA_KEY])
                self.assertEqual(gateway.thread_replies[-1]["text"], f"@{agent.handle} scheduling.")
                self.assertIn(("C1", "171.schedule", "eyes"), gateway.reactions)
                self.assertIn(("C1", "171.schedule", "hourglass_flowing_sand"), gateway.reactions)
            finally:
                store.close()

    def test_agent_schedule_control_signal_creates_scheduled_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "resolve schedule", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    requested_by_slack_user="U1",
                    metadata={SCHEDULE_RESOLUTION_METADATA_KEY: True},
                )
                store.upsert_agent_task(task)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                payload = {
                    "task": "check the patio lights",
                    "target": agent.handle,
                    "schedule": {
                        "kind": "one_off",
                        "run_at": (utc_now() + timedelta(hours=1)).isoformat(),
                        "timezone": "America/Chicago",
                        "description": "tomorrow at sunset in Waco",
                    },
                }

                handled = controller.handle_runtime_agent_control(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                    f"{AGENT_SCHEDULE_SIGNAL_PREFIX}{json.dumps(payload)}",
                )

                self.assertTrue(handled)
                rows = store.conn.execute("SELECT * FROM scheduled_work_requests").fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["prompt"], "check the patio lights")
                self.assertEqual(rows[0]["requested_handle"], agent.handle)
                self.assertEqual(rows[0]["schedule_kind"], "one_off")
                self.assertEqual(rows[0]["status"], "pending")
                self.assertIn(
                    f"Scheduled: @{agent.handle} `check the patio lights`; "
                    "tomorrow at sunset in Waco",
                    gateway.thread_replies[-1]["text"],
                )
                self.assertIn("next `", gateway.thread_replies[-1]["text"])
                self.assertNotIn("schedule_", gateway.thread_replies[-1]["text"])
                self.assertEqual(store.get_agent_task(task.task_id).status, AgentTaskStatus.DONE)
            finally:
                store.close()

    def test_user_schedule_request_e2e_posts_concise_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                runtime = FakeRuntime()
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
                            "text": "schedule @avery to check CI tomorrow at 9am UTC",
                            "ts": "171.schedule",
                        }
                    }
                )

                task, started_agent, thread = runtime.started[0]
                self.assertEqual(started_agent.agent_id, agent.agent_id)
                self.assertEqual(gateway.thread_replies[-1]["text"], f"@{agent.handle} scheduling.")

                run_at = utc_now() + timedelta(hours=1)
                payload = {
                    "task": "check CI",
                    "target": agent.handle,
                    "schedule": {
                        "kind": "one_off",
                        "run_at": run_at.isoformat(),
                        "timezone": "UTC",
                        "description": "tomorrow at 9am UTC",
                    },
                }

                handled = controller.handle_runtime_agent_control(
                    task,
                    agent,
                    thread,
                    f"{AGENT_SCHEDULE_SIGNAL_PREFIX}{json.dumps(payload)}",
                )

                self.assertTrue(handled)
                rows = store.conn.execute("SELECT * FROM scheduled_work_requests").fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["prompt"], "check CI")
                self.assertEqual(rows[0]["requested_handle"], agent.handle)
                self.assertEqual(
                    gateway.thread_replies[-1]["text"],
                    (
                        f"Scheduled: @{agent.handle} `check CI`; tomorrow at 9am UTC; "
                        f"next `{run_at.isoformat(timespec='minutes')}`."
                    ),
                )
            finally:
                store.close()

    def test_roster_shows_specific_pending_schedule_as_occupied(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                store.create_scheduled_work_request(
                    SlackThreadRef("C1", "171.thread", "171.schedule"),
                    WorkRequest(
                        prompt="check CI",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    schedule_kind=ScheduledWorkKind.RECURRING,
                    next_run_at=utc_now() + timedelta(hours=1),
                    recurrence={
                        "frequency": "daily",
                        "time": "17:00",
                        "timezone": "America/New_York",
                    },
                    timezone="America/New_York",
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.post_roster("C1")

                self.assertIn("0 available, 1 occupied", gateway.posts[-1]["text"])
                blocks_text = str(gateway.posts[-1]["blocks"])
                self.assertIn("Scheduled task: check CI", blocks_text)
                self.assertIn("daily at 17:00 America/New_York", blocks_text)
            finally:
                store.close()

    def test_roster_shows_specific_pending_deferred_as_occupied(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.deferred", "171.deferred"),
                    WorkRequest(
                        prompt="check CI after the deploy",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    depends_on=(
                        WorkDependency(
                            kind=WorkDependencyKind.THREAD,
                            channel_id="C1",
                            thread_ts="171.blocker",
                        ),
                    ),
                    description="after deploy finishes",
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.post_roster("C1")

                self.assertIn("0 available, 1 occupied", gateway.posts[-1]["text"])
                blocks_text = str(gateway.posts[-1]["blocks"])
                self.assertIn("Deferred task: check CI after the deploy", blocks_text)
                self.assertIn("waiting", blocks_text)
                self.assertIn("after deploy finishes", blocks_text)
                self.assertNotIn("'text': {'type': 'plain_text', 'text': 'Assign'}", blocks_text)
            finally:
                store.close()

    def test_deferred_agent_is_not_assigned_new_channel_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.deferred", "171.deferred"),
                    WorkRequest(
                        prompt="reserved future work",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    depends_on=(
                        WorkDependency(
                            kind=WorkDependencyKind.THREAD,
                            channel_id="C1",
                            thread_ts="171.blocker",
                        ),
                    ),
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
                            "text": "Somebody update the docs",
                            "ts": "171.user",
                        }
                    }
                )

                self.assertEqual(store.list_agent_tasks(), [])
                self.assertEqual(runtime.started, [])
                self.assertIn(
                    "No agents are available right now", gateway.thread_replies[-1]["text"]
                )
            finally:
                store.close()

    def test_future_work_appears_in_deferred_dependency_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                deferred_agent, scheduled_agent = [
                    replace(agent, handle=handle, agent_id=f"agent_{handle}", sort_order=index)
                    for index, (agent, handle) in enumerate(
                        zip(build_initial_model_team(2, 0), ("deferred", "scheduled"), strict=True)
                    )
                ]
                store.upsert_team_agent(deferred_agent)
                store.upsert_team_agent(scheduled_agent)
                deferred = store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.deferred", "171.deferred"),
                    WorkRequest(
                        prompt="run after blocker",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=deferred_agent.handle,
                    ),
                    depends_on=(
                        WorkDependency(
                            kind=WorkDependencyKind.THREAD,
                            channel_id="C1",
                            thread_ts="171.blocker",
                        ),
                    ),
                )
                scheduled = store.create_scheduled_work_request(
                    SlackThreadRef("C1", "171.scheduled", "171.scheduled"),
                    WorkRequest(
                        prompt="run later",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=scheduled_agent.handle,
                    ),
                    schedule_kind=ScheduledWorkKind.ONE_OFF,
                    next_run_at=utc_now() + timedelta(hours=1),
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                occupied = dict(controller._occupied_handle_task_ids())

                self.assertEqual(
                    occupied[deferred_agent.handle],
                    deferred_work_dependency_id(deferred.deferred_id),
                )
                self.assertEqual(
                    occupied[scheduled_agent.handle],
                    scheduled_work_dependency_id(scheduled.schedule_id),
                )
            finally:
                store.close()

    def test_scheduled_agent_is_not_assigned_new_channel_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                store.create_scheduled_work_request(
                    SlackThreadRef("C1", "171.thread", "171.schedule"),
                    WorkRequest(
                        prompt="check CI",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    schedule_kind=ScheduledWorkKind.ONE_OFF,
                    next_run_at=utc_now() + timedelta(hours=1),
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
                            "text": "Somebody update docs",
                            "ts": "171.user",
                        }
                    }
                )

                self.assertEqual(store.list_agent_tasks(), [])
                self.assertEqual(runtime.started, [])
                self.assertIn(
                    "No agents are available right now", gateway.thread_replies[-1]["text"]
                )
            finally:
                store.close()

    def test_scheduled_tasks_command_posts_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                scheduled = store.create_scheduled_work_request(
                    SlackThreadRef("C1", "171.thread", "171.schedule"),
                    WorkRequest(
                        prompt="check CI",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    schedule_kind=ScheduledWorkKind.ONE_OFF,
                    next_run_at=utc_now() + timedelta(hours=1),
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_team_command(
                    ScheduledTasksCommand(),
                    SlackReplyTarget(channel_id="C1"),
                )

                self.assertIn("Scheduled tasks: 1 active", gateway.posts[-1]["text"])
                action_block = next(
                    block
                    for block in gateway.posts[-1]["blocks"]
                    if block.get("block_id") == f"schedule.actions.{scheduled.schedule_id}"
                )
                self.assertEqual(
                    [element["action_id"] for element in action_block["elements"]],
                    ["schedule.cancel", "schedule.change", "thread.open"],
                )
            finally:
                store.close()

    def test_scheduled_tasks_command_formats_interval_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                store.create_scheduled_work_request(
                    SlackThreadRef("C1", "171.thread", "171.schedule"),
                    WorkRequest(
                        prompt="check CI",
                        assignment_mode=AssignmentMode.ANYONE,
                    ),
                    schedule_kind=ScheduledWorkKind.RECURRING,
                    next_run_at=utc_now() + timedelta(hours=2),
                    recurrence={
                        "frequency": "interval",
                        "interval": {"value": 2, "unit": "hours"},
                    },
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_team_command(
                    ScheduledTasksCommand(),
                    SlackReplyTarget(channel_id="C1"),
                )

                self.assertIn("every 2 hours", gateway.posts[-1]["text"])
                self.assertIn("every 2 hours", str(gateway.posts[-1]["blocks"]))
            finally:
                store.close()

    def test_schedule_cancel_button_deschedules_and_updates_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                scheduled = store.create_scheduled_work_request(
                    SlackThreadRef("C1", "171.thread", "171.schedule"),
                    WorkRequest(
                        prompt="check CI",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    schedule_kind=ScheduledWorkKind.ONE_OFF,
                    next_run_at=utc_now() + timedelta(hours=1),
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.list"},
                        "actions": [
                            {
                                "value": encode_action_value(
                                    "schedule.cancel",
                                    schedule_id=scheduled.schedule_id,
                                )
                            }
                        ],
                    }
                )

                current = store.get_scheduled_work(scheduled.schedule_id)
                assert current is not None
                self.assertEqual(current.status.value, "cancelled")
                self.assertIn("Descheduled", gateway.thread_replies[-1]["text"])
                self.assertEqual(gateway.updates[-1]["text"], "Scheduled tasks: none.")
            finally:
                store.close()

    def test_schedule_change_button_opens_modal_and_submission_updates_next_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                scheduled = store.create_scheduled_work_request(
                    SlackThreadRef("C1", "171.thread", "171.schedule"),
                    WorkRequest(
                        prompt="check CI",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    schedule_kind=ScheduledWorkKind.ONE_OFF,
                    next_run_at=utc_now() + timedelta(hours=1),
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.list"},
                        "trigger_id": "TRIGGER",
                        "actions": [
                            {
                                "value": encode_action_value(
                                    "schedule.change",
                                    schedule_id=scheduled.schedule_id,
                                )
                            }
                        ],
                    }
                )

                self.assertEqual(gateway.views[-1][0], "TRIGGER")
                modal = gateway.views[-1][1]
                changed_run_at = utc_now() + timedelta(days=2)
                response = controller.handle_view_submission(
                    {
                        "type": "view_submission",
                        "view": {
                            "callback_id": "schedule.change",
                            "private_metadata": modal["private_metadata"],
                            "state": {
                                "values": {
                                    "schedule_next_run": {
                                        "value": {"value": changed_run_at.isoformat()}
                                    }
                                }
                            },
                        },
                    }
                )

                self.assertIsNone(response)
                current = store.get_scheduled_work(scheduled.schedule_id)
                assert current is not None
                self.assertEqual(current.next_run_at, changed_run_at)
                self.assertIn("Changed schedule", gateway.thread_replies[-1]["text"])
                self.assertIn("Scheduled tasks: 1 active", gateway.updates[-1]["text"])
            finally:
                store.close()

    def test_invalid_agent_schedule_control_retries_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "resolve schedule", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        SCHEDULE_RESOLUTION_METADATA_KEY: True,
                        SCHEDULE_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                    },
                )
                store.upsert_agent_task(task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                handled = controller.handle_runtime_agent_control(
                    task,
                    agent,
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                    f"{AGENT_SCHEDULE_SIGNAL_PREFIX}{{not json",
                )

                self.assertTrue(handled)
                self.assertEqual(len(runtime.started), 1)
                retry_task = runtime.started[0][0]
                self.assertIn("previous schedule control line was invalid", retry_task.prompt)
                self.assertEqual(retry_task.metadata[SCHEDULE_RESOLUTION_ATTEMPTS_METADATA_KEY], 1)
            finally:
                store.close()

    def test_due_scheduled_work_starts_task_and_marks_one_off_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                scheduled = store.create_scheduled_work_request(
                    SlackThreadRef("C1", "171.thread", "171.schedule"),
                    WorkRequest(
                        prompt="check CI",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    schedule_kind=ScheduledWorkKind.ONE_OFF,
                    next_run_at=utc_now() - timedelta(seconds=1),
                    requested_by_slack_user="U1",
                )
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                runner = ScheduledWorkRunner(store, controller, poll_seconds=0.01)

                fired = runner.sync_once()

                self.assertEqual(fired, 1)
                self.assertEqual(len(runtime.started), 1)
                task, started_agent, thread = runtime.started[0]
                self.assertEqual(task.prompt, "check CI")
                self.assertEqual(started_agent.agent_id, agent.agent_id)
                self.assertEqual(thread.thread_ts, "171.thread")
                row = store.conn.execute(
                    "SELECT status, last_task_id FROM scheduled_work_requests WHERE schedule_id = ?",
                    (scheduled.schedule_id,),
                ).fetchone()
                self.assertEqual(row["status"], "done")
                self.assertEqual(row["last_task_id"], task.task_id)
                self.assertIn("is due now", gateway.thread_replies[-1]["text"])
            finally:
                store.close()

    def test_due_recurring_scheduled_work_reschedules_next_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                scheduled = store.create_scheduled_work_request(
                    SlackThreadRef("C1", "171.thread", "171.schedule"),
                    WorkRequest(
                        prompt="check CI",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    schedule_kind=ScheduledWorkKind.RECURRING,
                    next_run_at=utc_now() - timedelta(seconds=1),
                    recurrence={
                        "frequency": "daily",
                        "time": "17:00",
                        "timezone": "America/New_York",
                    },
                    timezone="America/New_York",
                    requested_by_slack_user="U1",
                )
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                runner = ScheduledWorkRunner(store, controller, poll_seconds=0.01)

                fired = runner.sync_once()

                self.assertEqual(fired, 1)
                row = store.conn.execute(
                    "SELECT status, next_run_at, last_task_id FROM scheduled_work_requests "
                    "WHERE schedule_id = ?",
                    (scheduled.schedule_id,),
                ).fetchone()
                self.assertEqual(row["status"], "pending")
                self.assertGreater(row["next_run_at"], utc_now().isoformat())
                self.assertEqual(row["last_task_id"], runtime.started[0][0].task_id)
            finally:
                store.close()

    def test_timer_control_signal_creates_scheduled_timer(self):
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
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                    f"{AGENT_TIMER_SIGNAL_PREFIX}5m | Re-check the PR feedback.",
                )

                self.assertTrue(handled)
                rows = store.conn.execute("SELECT * FROM scheduled_timers").fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "pending")
                self.assertEqual(rows[0]["task_id"], task.task_id)
                self.assertEqual(rows[0]["agent_id"], agent.agent_id)
                self.assertEqual(rows[0]["prompt"], "Re-check the PR feedback.")
                self.assertEqual(gateway.thread_replies, [])
            finally:
                store.close()

    def test_due_timer_resumes_same_thread_agent_task(self):
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
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-1",
                )
                store.upsert_agent_task(task)
                store.create_scheduled_timer(
                    task,
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                    prompt="Re-check the PR feedback.",
                    due_at=utc_now() - timedelta(seconds=1),
                )
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                runner = ScheduledTimerRunner(store, controller, poll_seconds=0.01)

                fired = runner.sync_once()

                self.assertEqual(fired, 1)
                self.assertEqual(len(runtime.started), 1)
                started_task, started_agent, started_thread = runtime.started[0]
                self.assertEqual(started_task.task_id, task.task_id)
                self.assertEqual(started_task.prompt, "Re-check the PR feedback.")
                self.assertEqual(started_task.session_id, "codex-thread-1")
                self.assertEqual(started_agent.agent_id, agent.agent_id)
                self.assertEqual(started_thread.thread_ts, "171.thread")
                rows = store.conn.execute("SELECT status FROM scheduled_timers").fetchall()
                self.assertEqual([row["status"] for row in rows], ["fired"])
            finally:
                store.close()

    def test_due_timer_resumes_finished_same_thread_agent_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                task = replace(
                    create_agent_task(agent, "initial task", "C1"),
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-1",
                )
                store.upsert_agent_task(task)
                store.create_scheduled_timer(
                    task,
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                    prompt="Re-check the PR feedback.",
                    due_at=utc_now() - timedelta(seconds=1),
                )
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                runner = ScheduledTimerRunner(store, controller, poll_seconds=0.01)

                fired = runner.sync_once()

                self.assertEqual(fired, 1)
                self.assertEqual(len(runtime.started), 1)
                started_task, started_agent, started_thread = runtime.started[0]
                self.assertEqual(started_task.task_id, task.task_id)
                self.assertEqual(started_task.status, AgentTaskStatus.ACTIVE)
                self.assertEqual(started_task.prompt, "Re-check the PR feedback.")
                self.assertEqual(started_task.session_id, "codex-thread-1")
                self.assertEqual(started_agent.agent_id, agent.agent_id)
                self.assertEqual(started_thread.thread_ts, "171.thread")
                rows = store.conn.execute("SELECT status FROM scheduled_timers").fetchall()
                self.assertEqual([row["status"] for row in rows], ["fired"])
            finally:
                store.close()

    def test_due_timer_waits_when_agent_is_busy_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                timer_task = replace(
                    create_agent_task(agent, "initial task", "C1"),
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    session_provider=Provider.CODEX,
                    session_id="codex-thread-1",
                )
                busy_task = replace(
                    create_agent_task(agent, "other work", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="172.thread",
                    parent_message_ts="172.parent",
                )
                store.upsert_agent_task(timer_task)
                store.upsert_agent_task(busy_task)
                store.create_scheduled_timer(
                    timer_task,
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                    prompt="Re-check the PR feedback.",
                    due_at=utc_now() - timedelta(seconds=1),
                )
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                runner = ScheduledTimerRunner(store, controller, poll_seconds=0.01)

                fired = runner.sync_once()

                self.assertEqual(fired, 0)
                self.assertEqual(runtime.started, [])
                rows = store.conn.execute("SELECT status FROM scheduled_timers").fetchall()
                self.assertEqual([row["status"] for row in rows], ["pending"])
            finally:
                store.close()

    def test_due_timer_waits_while_same_task_is_running(self):
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
                store.create_scheduled_timer(
                    task,
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                    prompt="Re-check the PR feedback.",
                    due_at=utc_now() - timedelta(seconds=1),
                )
                runtime = FakeRuntime()
                runtime.running_task_ids.add(task.task_id)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                runner = ScheduledTimerRunner(store, controller, poll_seconds=0.01)

                fired = runner.sync_once()

                self.assertEqual(fired, 0)
                self.assertEqual(runtime.started, [])
                rows = store.conn.execute("SELECT status FROM scheduled_timers").fetchall()
                self.assertEqual([row["status"] for row in rows], ["pending"])
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


class DeferredWorkFlowTests(unittest.TestCase):
    PERMALINK = "https://example.slack.com/archives/C1/p0000000171000001"

    def _setup_controller(self, tmp: str):
        store = Store(Path(tmp) / "state.sqlite")
        gateway = FakeGateway()
        runtime = FakeRuntime()
        store.init_schema()
        agents = build_initial_model_team(1, 1)
        for agent in agents:
            store.upsert_team_agent(agent)
        controller = SlackTeamController(
            store,
            gateway,
            default_channel_id="C1",
            runtime=runtime,
        )
        return store, gateway, runtime, agents, controller

    def test_user_message_starts_deferred_resolver_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _gateway, runtime, _agents, controller = self._setup_controller(tmp)
            try:
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"after {self.PERMALINK} finishes, summarize the deploy",
                            "ts": "171.user",
                        }
                    }
                )
                self.assertEqual(len(runtime.started), 1)
                task, _agent, thread = runtime.started[0]
                self.assertIn(AGENT_DEFERRED_SIGNAL_PREFIX, task.prompt)
                self.assertIn("summarize the deploy", task.prompt)
                self.assertTrue(task.metadata[DEFERRED_RESOLUTION_METADATA_KEY])
                self.assertEqual(thread.thread_ts, "171.user")
            finally:
                store.close()

    def test_deferred_after_external_busy_agent_routes_to_explicit_future_assignee(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                eli, matt, mila = [
                    replace(agent, handle=handle, agent_id=f"agent_{handle}", sort_order=index)
                    for index, (agent, handle) in enumerate(
                        zip(
                            build_initial_model_team(3, 0),
                            ("eli", "matt", "mila"),
                            strict=True,
                        )
                    )
                ]
                store.upsert_team_agent(eli)
                store.upsert_team_agent(matt)
                store.upsert_team_agent(mila)
                store.upsert_session(
                    AgentSession(
                        provider=Provider.CODEX,
                        session_id="busy-eli",
                        transcript_path=Path(tmp) / "codex.jsonl",
                        status=SessionStatus.ACTIVE,
                    )
                )
                store.set_setting("external_session_agent.codex.busy-eli", eli.agent_id)
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
                                "after @Eli is done @matt put up a PR and get @mila to review"
                            ),
                            "ts": "171.user",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                resolver_task, resolver_agent, resolver_thread = runtime.started[0]
                external_dep = external_session_dependency_id(Provider.CODEX, "busy-eli")
                self.assertIn(
                    f"@eli is currently working on task_id={external_dep}", resolver_task.prompt
                )
                self.assertNotEqual(resolver_agent.handle, "eli")

                payload = {
                    "task": "@matt put up a PR and get @mila to review",
                    "target": "somebody",
                    "depends_on": [{"kind": "agent_busy", "handle": "eli"}],
                }
                handled = controller.handle_runtime_agent_control(
                    resolver_task,
                    resolver_agent,
                    resolver_thread,
                    f"{AGENT_DEFERRED_SIGNAL_PREFIX}{json.dumps(payload)}",
                )
                self.assertTrue(handled)
                ack = gateway.thread_replies[-1]["text"]
                self.assertIn("- @eli: codex external session", ack)
                self.assertNotIn(external_dep, ack)
                deferred_rows = store.list_deferred_work()
                self.assertEqual(len(deferred_rows), 1)
                row = deferred_rows[0]
                self.assertEqual(row.requested_handle, "matt")
                self.assertEqual(row.prompt, "put up a PR and get @mila to review")
                self.assertEqual(row.depends_on[0].task_id, external_dep)
                self.assertEqual(controller.evaluate_pending_deferred_work(row.deferred_id), 0)

                store.delete_setting("external_session_agent.codex.busy-eli")
                self.assertEqual(controller.evaluate_pending_deferred_work(row.deferred_id), 1)
                runner = DeferredWorkRunner(store, controller, poll_seconds=0.01)
                runner.sync_once()

                self.assertEqual(len(runtime.started), 2)
                final_task, final_agent, _final_thread = runtime.started[-1]
                self.assertEqual(final_agent.handle, "matt")
                self.assertEqual(final_task.prompt, "put up a PR and get @mila to review")
                final = store.get_deferred_work(row.deferred_id)
                assert final is not None
                self.assertEqual(final.status, DeferredWorkStatus.DONE)
            finally:
                store.close()

    def test_deferred_signal_creates_row_and_fires_after_dep_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _gateway, runtime, agents, controller = self._setup_controller(tmp)
            try:
                blocking_agent = agents[0]
                blocking_task = create_agent_task(blocking_agent, "blocker", "C1")
                store.upsert_agent_task(blocking_task)
                store.update_agent_task_thread(blocking_task.task_id, "171.blocking", "171.parent")
                store.update_agent_task_status(blocking_task.task_id, AgentTaskStatus.ACTIVE)

                resolver_agent = agents[1]
                resolver_task = replace(
                    create_agent_task(resolver_agent, "resolve deferred", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.resolver",
                    parent_message_ts="171.resolver",
                    requested_by_slack_user="U1",
                    metadata={
                        DEFERRED_RESOLUTION_METADATA_KEY: True,
                        DEFERRED_RESOLUTION_OCCUPIED_HANDLES_METADATA_KEY: [],
                    },
                )
                store.upsert_agent_task(resolver_task)
                payload = {
                    "task": "summarize the deploy thread",
                    "target": blocking_agent.handle,
                    "depends_on": [
                        {
                            "kind": "thread",
                            "permalink": (
                                "https://example.slack.com/archives/C1/p0000000171000001"
                            ),
                            "channel_id": "C1",
                            "thread_ts": "171.blocking",
                        }
                    ],
                }
                handled = controller.handle_runtime_agent_control(
                    resolver_task,
                    resolver_agent,
                    SlackThreadRef("C1", "171.resolver", "171.resolver"),
                    f"{AGENT_DEFERRED_SIGNAL_PREFIX}{json.dumps(payload)}",
                )
                self.assertTrue(handled)
                deferred_rows = store.list_deferred_work()
                self.assertEqual(len(deferred_rows), 1)
                row = deferred_rows[0]
                self.assertEqual(row.status, DeferredWorkStatus.WAITING_DEPS)
                self.assertEqual(row.depends_on[0].thread_ts, "171.blocking")

                # Dependency unsatisfied — fire path stays empty.
                started_before = list(runtime.started)
                runner = DeferredWorkRunner(store, controller, poll_seconds=0.01)
                runner.sync_once()
                self.assertEqual(runtime.started, started_before)

                # Mark the blocking task done; runner should promote and fire.
                store.update_agent_task_status(blocking_task.task_id, AgentTaskStatus.DONE)
                runner.sync_once()  # promote waiting -> ready
                runner.sync_once()  # fire ready -> claim/start
                started_tasks = [task for task, _, _ in runtime.started]
                self.assertTrue(
                    any(task.prompt == "summarize the deploy thread" for task in started_tasks),
                    runtime.started,
                )
                final = store.get_deferred_work(row.deferred_id)
                assert final is not None
                self.assertEqual(final.status, DeferredWorkStatus.DONE)
            finally:
                store.close()

    def test_deferred_with_after_delay_anchors_on_dep_satisfaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _gateway, _runtime, agents, controller = self._setup_controller(tmp)
            try:
                blocking_task = create_agent_task(agents[0], "blocker", "C1")
                store.upsert_agent_task(blocking_task)
                store.update_agent_task_thread(blocking_task.task_id, "171.blocking", "p")
                deferred = store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.user", "171.user"),
                    WorkRequest(prompt="follow up", assignment_mode=AssignmentMode.ANYONE),
                    depends_on=(
                        WorkDependency(
                            kind=WorkDependencyKind.THREAD,
                            channel_id="C1",
                            thread_ts="171.blocking",
                            task_id=blocking_task.task_id,
                        ),
                    ),
                    after_dep_delay_seconds=1200,
                )
                # While dep not satisfied: promotion is a no-op.
                self.assertEqual(controller.evaluate_pending_deferred_work(), 0)
                # Complete dep, then promote.
                store.update_agent_task_status(blocking_task.task_id, AgentTaskStatus.DONE)
                before = utc_now()
                self.assertEqual(controller.evaluate_pending_deferred_work(), 1)
                refreshed = store.get_deferred_work(deferred.deferred_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, DeferredWorkStatus.READY)
                self.assertIsNotNone(refreshed.fire_at)
                assert refreshed.fire_at is not None
                delta = refreshed.fire_at - before
                # Window has slack to cover test execution jitter.
                self.assertGreaterEqual(delta, timedelta(seconds=1199))
                self.assertLessEqual(delta, timedelta(seconds=1205))
            finally:
                store.close()

    def test_deferred_with_run_at_waits_for_max_of_dep_and_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _gateway, _runtime, agents, controller = self._setup_controller(tmp)
            try:
                blocking_task = create_agent_task(agents[0], "blocker", "C1")
                store.upsert_agent_task(blocking_task)
                store.update_agent_task_thread(blocking_task.task_id, "171.blocking", "p")
                future_run_at = utc_now() + timedelta(hours=2)
                deferred = store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.user", "171.user"),
                    WorkRequest(prompt="follow up", assignment_mode=AssignmentMode.ANYONE),
                    depends_on=(
                        WorkDependency(
                            kind=WorkDependencyKind.THREAD,
                            channel_id="C1",
                            thread_ts="171.blocking",
                            task_id=blocking_task.task_id,
                        ),
                    ),
                    run_at=future_run_at,
                )
                store.update_agent_task_status(blocking_task.task_id, AgentTaskStatus.DONE)
                self.assertEqual(controller.evaluate_pending_deferred_work(), 1)
                refreshed = store.get_deferred_work(deferred.deferred_id)
                assert refreshed is not None and refreshed.fire_at is not None
                self.assertEqual(refreshed.fire_at, future_run_at)
            finally:
                store.close()

    def test_multi_dependency_waits_until_all_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _gateway, _runtime, agents, controller = self._setup_controller(tmp)
            try:
                first = create_agent_task(agents[0], "first", "C1")
                second = create_agent_task(agents[1], "second", "C1")
                store.upsert_agent_task(first)
                store.upsert_agent_task(second)
                store.update_agent_task_thread(first.task_id, "171.first", "p1")
                store.update_agent_task_thread(second.task_id, "171.second", "p2")
                deferred = store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.user", "171.user"),
                    WorkRequest(prompt="finalize", assignment_mode=AssignmentMode.ANYONE),
                    depends_on=(
                        WorkDependency(
                            kind=WorkDependencyKind.THREAD,
                            channel_id="C1",
                            thread_ts="171.first",
                            task_id=first.task_id,
                        ),
                        WorkDependency(
                            kind=WorkDependencyKind.AGENT_BUSY,
                            handle=agents[1].handle,
                            task_id=second.task_id,
                        ),
                    ),
                )
                store.update_agent_task_status(first.task_id, AgentTaskStatus.DONE)
                self.assertEqual(controller.evaluate_pending_deferred_work(), 0)
                store.update_agent_task_status(second.task_id, AgentTaskStatus.DONE)
                self.assertEqual(controller.evaluate_pending_deferred_work(), 1)
                refreshed = store.get_deferred_work(deferred.deferred_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, DeferredWorkStatus.READY)
            finally:
                store.close()

    def test_deferred_can_wait_on_another_deferred_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                agents = build_initial_model_team(2, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                blocker_agent, first_target, second_target = agents[:3]
                blocking_task = create_agent_task(blocker_agent, "first blocker", "C1")
                store.upsert_agent_task(blocking_task)
                store.update_agent_task_thread(blocking_task.task_id, "171.blocking", "p")
                first_deferred = store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.first", "171.first"),
                    WorkRequest(
                        prompt="first future task",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=first_target.handle,
                    ),
                    depends_on=(
                        WorkDependency(
                            kind=WorkDependencyKind.THREAD,
                            channel_id="C1",
                            thread_ts="171.blocking",
                            task_id=blocking_task.task_id,
                        ),
                    ),
                    description=f"after @{blocker_agent.handle} finishes",
                )
                resolver_agent = second_target
                resolver_task = replace(
                    create_agent_task(resolver_agent, "resolve deferred", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.resolver",
                    parent_message_ts="171.resolver",
                    metadata={
                        DEFERRED_RESOLUTION_METADATA_KEY: True,
                        DEFERRED_RESOLUTION_OCCUPIED_HANDLES_METADATA_KEY: [
                            {
                                "handle": first_target.handle,
                                "task_id": deferred_work_dependency_id(first_deferred.deferred_id),
                            }
                        ],
                    },
                )
                store.upsert_agent_task(resolver_task)
                payload = {
                    "task": "second future task",
                    "target": second_target.handle,
                    "depends_on": [{"kind": "agent_busy", "handle": first_target.handle}],
                }

                handled = controller.handle_runtime_agent_control(
                    resolver_task,
                    resolver_agent,
                    SlackThreadRef("C1", "171.resolver", "171.resolver"),
                    f"{AGENT_DEFERRED_SIGNAL_PREFIX}{json.dumps(payload)}",
                )

                self.assertTrue(handled)
                deferred_rows = store.list_deferred_work()
                second_deferred = next(
                    item for item in deferred_rows if item.deferred_id != first_deferred.deferred_id
                )
                self.assertEqual(
                    second_deferred.depends_on[0].task_id,
                    deferred_work_dependency_id(first_deferred.deferred_id),
                )

                store.update_agent_task_status(blocking_task.task_id, AgentTaskStatus.DONE)
                runner = DeferredWorkRunner(store, controller, poll_seconds=0.01)
                runner.sync_once()
                runner.sync_once()
                first_started = next(
                    task for task, agent, _thread in runtime.started if agent == first_target
                )
                self.assertEqual(first_started.prompt, "first future task")
                self.assertEqual(
                    store.get_deferred_work(second_deferred.deferred_id).status,
                    DeferredWorkStatus.WAITING_DEPS,
                )

                store.update_agent_task_status(first_started.task_id, AgentTaskStatus.DONE)
                promoted = controller.evaluate_pending_deferred_work(second_deferred.deferred_id)

                self.assertEqual(promoted, 1)
                refreshed = store.get_deferred_work(second_deferred.deferred_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, DeferredWorkStatus.READY)
            finally:
                store.close()

    def test_invalid_deferred_signal_retries_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _gateway, runtime, agents, controller = self._setup_controller(tmp)
            try:
                resolver_agent = agents[0]
                task = replace(
                    create_agent_task(resolver_agent, "resolve deferred", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        DEFERRED_RESOLUTION_METADATA_KEY: True,
                        DEFERRED_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        DEFERRED_RESOLUTION_OCCUPIED_HANDLES_METADATA_KEY: [],
                    },
                )
                store.upsert_agent_task(task)
                handled = controller.handle_runtime_agent_control(
                    task,
                    resolver_agent,
                    SlackThreadRef("C1", "171.thread", "171.parent"),
                    f"{AGENT_DEFERRED_SIGNAL_PREFIX}{{not json",
                )
                self.assertTrue(handled)
                self.assertEqual(len(runtime.started), 1)
                retry_task = runtime.started[0][0]
                self.assertIn("previous deferred control line was invalid", retry_task.prompt)
                self.assertEqual(retry_task.metadata[DEFERRED_RESOLUTION_ATTEMPTS_METADATA_KEY], 1)
            finally:
                store.close()

    def test_deferred_cancelled_when_originating_thread_is_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _gateway, _runtime, agents, controller = self._setup_controller(tmp)
            try:
                blocking_task = create_agent_task(agents[0], "blocker", "C1")
                store.upsert_agent_task(blocking_task)
                store.update_agent_task_thread(blocking_task.task_id, "171.blocking", "p")
                user_task = create_agent_task(agents[1], "deferred holder", "C1")
                store.upsert_agent_task(user_task)
                store.update_agent_task_thread(user_task.task_id, "171.user", "171.user")
                deferred = store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.user", "171.user"),
                    WorkRequest(prompt="follow up", assignment_mode=AssignmentMode.ANYONE),
                    depends_on=(
                        WorkDependency(
                            kind=WorkDependencyKind.THREAD,
                            channel_id="C1",
                            thread_ts="171.blocking",
                            task_id=blocking_task.task_id,
                        ),
                    ),
                )
                self.assertEqual(deferred.status, DeferredWorkStatus.WAITING_DEPS)
                # Simulate the user marking the deferred holder thread done.
                completed = controller._complete_task_thread("C1", "171.user")
                self.assertTrue(completed)
                refreshed = store.get_deferred_work(deferred.deferred_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, DeferredWorkStatus.CANCELLED)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
