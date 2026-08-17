import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent_harness.loops import AGENT_LOOP_SIGNAL_PREFIX
from agent_harness.models import (
    LOOP_RESOLUTION_ATTEMPTS_METADATA_KEY,
    AgentTaskStatus,
    LoopStatus,
    LoopVisibility,
    PermissionMode,
    Provider,
    SlackThreadRef,
    TeamAgentStatus,
)
from agent_harness.slack import encode_action_value
from agent_harness.slack.app import SlackTeamController
from agent_harness.storage.store import Store
from tests.test_slack_app import FakeGateway, FakeRuntime


class LoopCreationFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = Store(root / "state.sqlite")
        self.store.init_schema()
        self.gateway = FakeGateway()
        self.runtime = FakeRuntime()
        self.controller = SlackTeamController(
            self.store,
            self.gateway,
            default_channel_id="CMAIN",
            runtime=self.runtime,
            home=root,
            default_cwd=root,
        )

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _request_loop(self, text: str | None = None):
        self.controller.handle_event(
            {
                "event": {
                    "type": "message",
                    "channel": "CMAIN",
                    "ts": "100.000001",
                    "user": "UOWNER",
                    "text": text
                    or (
                        "loop create inspect cloud billing every five minutes #private "
                        "provider=codex model=example-model"
                    ),
                }
            }
        )
        loops = self.store.list_loops()
        self.assertEqual(len(loops), 1)
        loop = loops[0]
        task = self.store.list_agent_tasks(include_done=True)[0]
        agent = self.store.get_team_agent(loop.agent_id, include_fired=True)
        assert agent is not None
        return loop, task, agent

    def _resolve_loop(self):
        loop, task, agent = self._request_loop()
        signal = AGENT_LOOP_SIGNAL_PREFIX + json.dumps(
            {
                "title": "Cloud Billing Watch",
                "bot_name": "Billing Bot",
                "channel_name": "loop-cloud-billing",
                "mission": "Inspect cloud billing and report material anomalies.",
                "schedule": {"frequency": "interval", "interval_seconds": 300},
                "icon": {
                    "emoji": "money_with_wings",
                    "badge": {
                        "background": "#0B6E4F",
                        "glyph": "$",
                        "glyph_color": "#F4FFF9",
                        "shape": "circle",
                    },
                },
            }
        )
        handled = self.controller.handle_runtime_agent_control(
            task,
            agent,
            SlackThreadRef(loop.anchor_channel_id, loop.anchor_thread_ts),
            signal,
        )
        self.assertTrue(handled)
        resolved = self.store.get_loop(loop.loop_id)
        assert resolved is not None
        return resolved

    def _action_payload(self, action: str, loop_id: str, *, user: str = "UOWNER"):
        return {
            "actions": [
                {
                    "value": encode_action_value(
                        action,
                        loop_id=loop_id,
                    )
                }
            ],
            "channel": {"id": "CMAIN"},
            "message": {"ts": "100.preview"},
            "user": {"id": user},
        }

    def test_creation_resolves_previews_and_approves_channel(self):
        loop = self._resolve_loop()

        self.assertEqual(loop.status, LoopStatus.AWAITING_APPROVAL)
        self.assertIsNotNone(loop.pending_spec_json)
        self.assertIsNotNone(loop.preview_message_ts)
        agent = self.store.get_team_agent(loop.agent_id)
        assert agent is not None
        self.assertEqual(agent.full_name, "Billing Bot")
        self.assertEqual(agent.handle, "billing-bot")
        self.assertEqual(agent.icon_emoji, ":money_with_wings:")
        self.assertFalse(agent.metadata["provisional"])
        self.assertEqual(len(self.gateway.uploads), 1)
        self.assertTrue(Path(self.gateway.uploads[0][1]).is_file())
        self.assertIn("loop.approve", str(self.gateway.thread_replies[-1]["blocks"]))

        self.controller.handle_block_action(self._action_payload("loop.approve", loop.loop_id))

        created = self.store.get_loop(loop.loop_id)
        assert created is not None
        self.assertEqual(created.status, LoopStatus.ACTIVE)
        self.assertEqual(created.channel_id, "CNEW")
        self.assertIsNone(created.pending_spec_json)
        self.assertIsNotNone(created.next_run_at)
        self.assertEqual(self.gateway.channels, [("loop-cloud-billing", True)])
        self.assertEqual(self.gateway.invites, [("CNEW", ["UOWNER"])])
        self.assertEqual(self.gateway.topics[0][0], "CNEW")
        self.assertEqual(self.gateway.pins[0][0], "CNEW")
        self.assertEqual(len(self.gateway.uploads), 2)
        self.assertIn("Loop created", self.gateway.updates[-1]["text"])

    def test_malformed_spec_retries_then_cleans_up(self):
        loop, task, agent = self._request_loop()
        thread = SlackThreadRef(loop.anchor_channel_id, loop.anchor_thread_ts)
        malformed = f'{AGENT_LOOP_SIGNAL_PREFIX}{{"title":"missing fields"}}'

        self.assertTrue(
            self.controller.handle_runtime_agent_control(task, agent, thread, malformed)
        )
        retried = self.store.get_agent_task(task.task_id)
        assert retried is not None
        self.assertEqual(retried.metadata[LOOP_RESOLUTION_ATTEMPTS_METADATA_KEY], 1)
        self.assertIn("previous loop control line was invalid", retried.prompt)

        retried = replace(
            retried,
            metadata={
                **retried.metadata,
                LOOP_RESOLUTION_ATTEMPTS_METADATA_KEY: 3,
            },
        )
        self.store.upsert_agent_task(retried)
        self.controller.handle_runtime_agent_control(retried, agent, thread, malformed)

        cancelled = self.store.get_loop(loop.loop_id)
        assert cancelled is not None
        self.assertEqual(cancelled.status, LoopStatus.CANCELLED)
        self.assertEqual(
            self.store.get_team_agent(loop.agent_id, include_fired=True).status,
            TeamAgentStatus.FIRED,
        )
        self.assertEqual(
            self.store.get_agent_task(task.task_id).status,
            AgentTaskStatus.CANCELLED,
        )

    def test_non_owner_cannot_approve_or_edit_preview(self):
        loop = self._resolve_loop()
        before = loop.pending_spec_json

        self.controller.handle_block_action(
            self._action_payload("loop.approve", loop.loop_id, user="UBYSTANDER")
        )
        self.controller.handle_event(
            {
                "event": {
                    "type": "message",
                    "channel": "CMAIN",
                    "thread_ts": loop.anchor_thread_ts,
                    "ts": "100.000002",
                    "user": "UBYSTANDER",
                    "text": "name: Hijacked Bot",
                }
            }
        )

        unchanged = self.store.get_loop(loop.loop_id)
        assert unchanged is not None
        self.assertEqual(unchanged.status, LoopStatus.AWAITING_APPROVAL)
        self.assertEqual(unchanged.pending_spec_json, before)
        self.assertEqual(self.gateway.channels, [])
        self.assertEqual(
            self.gateway.ephemerals,
            [("CMAIN", "UBYSTANDER", "Only the loop owner can do this.")],
        )

    def test_owner_can_cancel_preview_without_creating_channel(self):
        loop = self._resolve_loop()

        self.controller.handle_block_action(self._action_payload("loop.cancel", loop.loop_id))

        cancelled = self.store.get_loop(loop.loop_id)
        assert cancelled is not None
        self.assertEqual(cancelled.status, LoopStatus.CANCELLED)
        self.assertIsNone(cancelled.pending_spec_json)
        self.assertEqual(self.gateway.channels, [])
        self.assertIn("Loop creation cancelled", self.gateway.updates[-1]["text"])
        agent = self.store.get_team_agent(loop.agent_id, include_fired=True)
        assert agent is not None
        self.assertEqual(agent.status, TeamAgentStatus.FIRED)

    def test_owner_can_edit_preview_and_reresolve_schedule(self):
        loop = self._resolve_loop()
        edits = (
            ("name: Cost Guard Bot", "100.000010"),
            ("visibility: public", "100.000011"),
            ("permissions: locked", "100.000012"),
            ("provider: claude", "100.000013"),
            ("model: example-model-v2", "100.000014"),
            ("icon: :chart_with_upwards_trend:", "100.000015"),
            (f"cwd: {self.temp_dir.name}", "100.000016"),
        )
        for text, ts in edits:
            self.controller.handle_event(
                {
                    "event": {
                        "type": "message",
                        "channel": "CMAIN",
                        "thread_ts": loop.anchor_thread_ts,
                        "ts": ts,
                        "user": "UOWNER",
                        "text": text,
                    }
                }
            )

        edited = self.store.get_loop(loop.loop_id)
        assert edited is not None
        edited_agent = self.store.get_team_agent(loop.agent_id)
        assert edited_agent is not None
        self.assertEqual(edited.visibility, LoopVisibility.PUBLIC)
        self.assertEqual(edited.permission_mode, PermissionMode.LOCKED)
        self.assertEqual(edited.provider, Provider.CLAUDE)
        self.assertEqual(edited.model, "example-model-v2")
        self.assertEqual(edited.cwd, str(Path(self.temp_dir.name).resolve()))
        self.assertEqual(edited_agent.full_name, "Cost Guard Bot")
        self.assertEqual(edited_agent.icon_emoji, ":chart_with_upwards_trend:")

        self.controller.handle_event(
            {
                "event": {
                    "type": "message",
                    "channel": "CMAIN",
                    "thread_ts": loop.anchor_thread_ts,
                    "ts": "100.000017",
                    "user": "UOWNER",
                    "text": "schedule: every ten minutes",
                }
            }
        )

        resolving = self.store.get_loop(loop.loop_id)
        assert resolving is not None
        self.assertEqual(resolving.status, LoopStatus.RESOLVING)
        self.assertIn("OWNER OVERRIDE (SCHEDULE)", self.runtime.started[-1][0].prompt)
        self.assertNotIn("loop.approve", str(self.gateway.updates[-1]["blocks"]))


if __name__ == "__main__":
    unittest.main()
