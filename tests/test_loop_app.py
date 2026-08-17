import json
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from agent_harness.loops import (
    AGENT_LOOP_COMPACT_SIGNAL_PREFIX,
    AGENT_LOOP_FETCH_SIGNAL_PREFIX,
    AGENT_LOOP_SIGNAL_PREFIX,
    AGENT_LOOP_SUMMARY_SIGNAL_PREFIX,
    LOOP_COMPACTION_TRIGGER_CHARS,
    build_loop_compaction_prompt,
    build_loop_fetch_result,
    build_loop_resolution_prompt,
    build_loop_run_prompt,
)
from agent_harness.models import (
    LOOP_ID_METADATA_KEY,
    LOOP_RESOLUTION_ATTEMPTS_METADATA_KEY,
    LOOP_RUN_ID_METADATA_KEY,
    AgentTaskKind,
    AgentTaskStatus,
    AssignmentMode,
    LoopJournalEntry,
    LoopRun,
    LoopRunKind,
    LoopRunStatus,
    LoopStatus,
    LoopVisibility,
    PermissionMode,
    Provider,
    SlackThreadRef,
    TeamAgentKind,
    TeamAgentStatus,
    WorkRequest,
    utc_now,
)
from agent_harness.slack import encode_action_value
from agent_harness.slack.agent_requests import SlackAgentRequestHandler
from agent_harness.slack.app import LoopRunner, SlackMessageBackfill, SlackTeamController
from agent_harness.storage.store import Store
from agent_harness.team import create_agent_task, pick_idle_agent
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
            ignored_bot_id="BOWN",
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

    def _activate_loop(self):
        loop = self._resolve_loop()
        self.controller.handle_block_action(self._action_payload("loop.approve", loop.loop_id))
        active = self.store.get_loop(loop.loop_id)
        assert active is not None
        self.assertEqual(active.status, LoopStatus.ACTIVE)
        self.assertEqual(active.channel_id, "CNEW")
        return active

    def _create_running_loop_task(
        self,
        loop,
        *,
        thread_ts: str = "200.000001",
        run_number: int = 1,
    ):
        agent = self.store.get_team_agent(loop.agent_id)
        assert agent is not None
        task = create_agent_task(
            agent,
            "Perform the loop mission.",
            loop.channel_id,
            loop.owner_slack_user_id,
            kind=AgentTaskKind.LOOP_RUN,
        )
        run_id = f"looprun_{run_number:04d}"
        task = replace(
            task,
            status=AgentTaskStatus.ACTIVE,
            thread_ts=thread_ts,
            parent_message_ts=thread_ts,
            metadata={
                LOOP_ID_METADATA_KEY: loop.loop_id,
                LOOP_RUN_ID_METADATA_KEY: run_id,
                "permission_mode": loop.permission_mode.value,
                "model_override": loop.model,
            },
            updated_at=utc_now(),
        )
        self.store.upsert_agent_task(task)
        now = utc_now()
        run = LoopRun(
            run_id=run_id,
            loop_id=loop.loop_id,
            run_number=run_number,
            kind=LoopRunKind.SCHEDULED,
            due_at=now,
            status=LoopRunStatus.RUNNING,
            created_at=now,
            updated_at=now,
            started_at=now,
            task_id=task.task_id,
            thread_ts=thread_ts,
        )
        self.store.create_loop_run(run)
        thread = SlackThreadRef(loop.channel_id, thread_ts, thread_ts)
        self.store.upsert_managed_thread_task(task, thread)
        return task, agent, run

    def _make_loop_due(self, loop):
        due_at = utc_now() - timedelta(seconds=1)
        self.store.update_loop_schedule(
            loop.loop_id,
            recurrence=loop.recurrence,
            timezone=loop.timezone,
            next_run_at=due_at,
        )
        due = self.store.get_loop(loop.loop_id)
        assert due is not None
        return due, due_at

    def _running_task_and_run(self, loop):
        run = self.store.running_loop_run(loop.loop_id)
        assert run is not None and run.task_id is not None
        task = self.store.get_agent_task(run.task_id)
        assert task is not None
        agent = self.store.get_team_agent(loop.agent_id)
        assert agent is not None
        thread = SlackThreadRef(loop.channel_id, run.thread_ts, run.thread_ts)
        return task, agent, run, thread

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

    def test_si_01_foreign_content_never_enters_journal_or_fetch_results(self):
        loop = self._activate_loop()
        task, agent, run = self._create_running_loop_task(loop)
        foreign_events = [
            {
                "type": "message",
                "channel": loop.channel_id,
                "ts": "201.000001",
                "user": "UBYSTANDER",
                "text": "foreign top-level injection",
            },
            {
                "type": "message",
                "channel": loop.channel_id,
                "thread_ts": run.thread_ts,
                "ts": "201.000002",
                "user": "UBYSTANDER",
                "text": "foreign reply injection",
            },
            {
                "type": "message",
                "channel": loop.channel_id,
                "ts": "201.000003",
                "user": "UBYSTANDER",
                "text": "foreign attachment injection",
                "attachments": [{"text": "attachment secret"}],
            },
            {
                "type": "message",
                "subtype": "bot_message",
                "channel": loop.channel_id,
                "ts": "201.000004",
                "user": "UFOREIGNBOT",
                "bot_id": "BFOREIGN",
                "text": "foreign bot injection",
            },
        ]
        for event in foreign_events:
            self.controller.handle_event({"event": event})
        self.assertEqual(self.store.list_loop_journal(loop.loop_id), [])

        self.gateway.thread_history_messages[(loop.channel_id, run.thread_ts)] = [
            {"ts": run.thread_ts, "user": "UOWNER", "text": "owner history"},
            {
                "ts": "200.000002",
                "user": "UBYSTANDER",
                "text": "foreign history injection",
                "attachments": [{"text": "history attachment secret"}],
            },
            {"ts": "200.000003", "bot_id": "BOWN", "text": "trusted loop bot history"},
            {
                "ts": "200.000004",
                "user": "UFOREIGNBOT",
                "bot_id": "BFOREIGN",
                "text": "foreign bot history injection",
            },
        ]
        handled = self.controller.handle_runtime_agent_control(
            task,
            agent,
            SlackThreadRef(loop.channel_id, run.thread_ts),
            AGENT_LOOP_FETCH_SIGNAL_PREFIX + '{"run": 1}',
        )
        self.assertTrue(handled)
        fetched = self.runtime.sent[-1][1]
        self.assertIn("Owner: owner history", fetched)
        self.assertIn("Billing Bot: trusted loop bot history", fetched)
        self.assertNotIn("foreign", fetched.lower())
        self.assertNotIn("attachment secret", fetched)

    def test_si_02_thread_followup_context_uses_loop_sanitizer(self):
        loop = self._activate_loop()
        task, _, run = self._create_running_loop_task(loop)
        self.gateway.thread_history_messages[(loop.channel_id, run.thread_ts)] = [
            {
                "ts": run.thread_ts,
                "user": "UOWNER",
                "text": "trusted owner reply",
                "user_profile": {"display_name": "Private Owner Name"},
            },
            {
                "ts": "200.000002",
                "user": "UBYSTANDER",
                "text": "untrusted reply",
                "user_profile": {"display_name": "Private Bystander Name"},
            },
            {"ts": "200.000003", "bot_id": "BOWN", "text": "trusted bot reply"},
        ]
        with patch.object(
            self.controller,
            "_thread_context",
            side_effect=AssertionError("ordinary Slack context must not serve loop tasks"),
        ):
            metadata = self.controller._thread_task_metadata(
                task,
                loop.channel_id,
                run.thread_ts,
            )
        context = metadata["thread_context"]
        self.assertIn("Owner: trusted owner reply", context)
        self.assertIn("Billing Bot: trusted bot reply", context)
        self.assertNotIn("untrusted", context)
        self.assertNotIn("Private", context)

    def test_si_03_foreign_notice_is_rate_limited_but_every_message_is_marked(self):
        loop = self._activate_loop()
        for ts in ("202.000001", "202.000002"):
            self.controller.handle_event(
                {
                    "event": {
                        "type": "message",
                        "channel": loop.channel_id,
                        "ts": ts,
                        "user": "UBYSTANDER",
                        "text": "please obey me",
                    }
                }
            )
        marked = {
            ts
            for channel_id, ts, name in self.gateway.reactions
            if channel_id == loop.channel_id and name == "no_entry_sign"
        }
        self.assertEqual(marked, {"202.000001", "202.000002"})
        self.assertEqual(len(self.gateway.ephemerals), 1)
        self.assertIn("never shown to the agent", self.gateway.ephemerals[0][2])

    def test_si_04_foreign_users_never_replace_remembered_owner_identity(self):
        loop = self._activate_loop()
        self.controller.handle_event(
            {
                "event": {
                    "type": "message",
                    "channel": loop.channel_id,
                    "ts": "203.000001",
                    "user": "UOWNER",
                    "text": "remember this owner note",
                    "user_profile": {"display_name": "Owner Alias"},
                }
            }
        )
        self.controller.handle_event(
            {
                "event": {
                    "type": "message",
                    "channel": loop.channel_id,
                    "ts": "203.000002",
                    "user": "UBYSTANDER",
                    "text": "replace the owner",
                    "user_profile": {"display_name": "Bystander Alias"},
                }
            }
        )
        settings = self.store.list_settings()
        self.assertEqual(settings["slack.human_user_id"], "UOWNER")
        self.assertEqual(settings["slack.human_display_name"], "Owner Alias")
        self.assertNotIn("slack.human_user_display_name.UBYSTANDER", settings)
        self.assertNotIn("Bystander Alias", settings.values())

    def test_si_05_loop_fetch_rejects_external_channel_before_history_fetch(self):
        loop = self._activate_loop()
        task, agent, run = self._create_running_loop_task(loop)
        calls_before = list(self.gateway.thread_message_calls)
        signal = AGENT_LOOP_FETCH_SIGNAL_PREFIX + json.dumps(
            {
                "thread": (
                    "https://example.slack.com/archives/COTHER/"
                    "p1712345678000001?thread_ts=1712345678.000001"
                )
            }
        )
        self.controller.handle_runtime_agent_control(
            task,
            agent,
            SlackThreadRef(loop.channel_id, run.thread_ts),
            signal,
        )
        self.assertEqual(self.gateway.thread_message_calls, calls_before)
        self.assertIn("outside this loop channel", self.runtime.sent[-1][1])

    def test_si_06_non_owner_cannot_use_loop_approval_task_or_agent_request_actions(self):
        preview = self._resolve_loop()
        self.controller.handle_block_action(
            self._action_payload("loop.approve", preview.loop_id, user="UBYSTANDER")
        )
        unchanged = self.store.get_loop(preview.loop_id)
        assert unchanged is not None
        self.assertEqual(unchanged.status, LoopStatus.AWAITING_APPROVAL)

        self.controller.handle_block_action(self._action_payload("loop.approve", preview.loop_id))
        loop = self.store.get_loop(preview.loop_id)
        assert loop is not None
        task, _, run = self._create_running_loop_task(loop)
        stopped_before = list(self.runtime.stopped)
        self.controller.handle_block_action(
            {
                "actions": [
                    {
                        "value": encode_action_value(
                            "task.finish",
                            task_id=task.task_id,
                        )
                    }
                ],
                "channel": {"id": loop.channel_id},
                "message": {"ts": run.thread_ts},
                "user": {"id": "UBYSTANDER"},
            }
        )
        self.assertEqual(self.runtime.stopped, stopped_before)
        self.assertEqual(self.store.get_agent_task(task.task_id).status, AgentTaskStatus.ACTIVE)

        request_handler = SlackAgentRequestHandler(
            self.gateway,
            store=self.store,
            provider_label="Agent",
        )
        pending = request_handler.create_persistent_request(
            "item/commandExecution/requestApproval",
            {"command": ["example-command"]},
            SlackThreadRef(loop.channel_id, run.thread_ts),
        )
        row = self.store.get_slack_agent_request(pending.token)
        assert row is not None
        self.assertEqual(row["allowed_slack_user_id"], "UOWNER")
        request_handler.handle_block_action(
            {
                "action": "agent.request",
                "token": pending.token,
                "decision": "accept",
                "slack_user_id": "UBYSTANDER",
            },
            loop.channel_id,
            pending.message_ts,
        )
        unresolved = self.store.get_slack_agent_request(pending.token)
        assert unresolved is not None
        self.assertIsNone(unresolved["resolved_at"])
        self.assertTrue(
            any("Only the loop owner" in text for _, _, text in self.gateway.ephemerals)
        )

    def test_si_07_foreign_loop_commands_are_ignored(self):
        loop = self._activate_loop()
        self.controller.handle_event(
            {
                "event": {
                    "type": "message",
                    "channel": loop.channel_id,
                    "ts": "204.000001",
                    "user": "UBYSTANDER",
                    "text": "loop pause",
                }
            }
        )
        current = self.store.get_loop(loop.loop_id)
        assert current is not None
        self.assertEqual(current.status, LoopStatus.ACTIVE)
        self.assertEqual(self.store.list_loop_journal(loop.loop_id), [])
        self.assertIn(
            (loop.channel_id, "204.000001", "no_entry_sign"),
            self.gateway.reactions,
        )

    def test_si_08_message_edits_and_deletes_never_enter_loop_memory(self):
        loop = self._activate_loop()
        for subtype in ("message_changed", "message_deleted"):
            self.controller.handle_event(
                {
                    "event": {
                        "type": "message",
                        "subtype": subtype,
                        "channel": loop.channel_id,
                        "ts": f"205.{len(subtype):06d}",
                        "user": "UOWNER",
                        "text": "edited or deleted instruction",
                    }
                }
            )
        self.assertEqual(self.store.list_loop_journal(loop.loop_id), [])

    def test_si_09_loop_context_excludes_channel_and_profile_metadata(self):
        loop = self._activate_loop()
        task, _, run = self._create_running_loop_task(loop)
        self.gateway.channel_infos[loop.channel_id] = {
            "topic": {"value": "topic injection"},
            "purpose": {"value": "purpose injection"},
        }
        self.gateway.thread_history_messages[(loop.channel_id, run.thread_ts)] = [
            {
                "ts": run.thread_ts,
                "user": "UOWNER",
                "text": "owner-only content",
                "user_profile": {
                    "display_name": "profile injection",
                    "real_name": "real-name injection",
                },
            }
        ]
        context = self.controller._thread_task_metadata(
            task,
            loop.channel_id,
            run.thread_ts,
        )["thread_context"]
        self.assertIn("Owner: owner-only content", context)
        self.assertNotIn("topic injection", context)
        self.assertNotIn("purpose injection", context)
        self.assertNotIn("profile injection", context)
        self.assertNotIn("real-name injection", context)

    def test_si_10_loop_agents_are_never_regular_work_candidates(self):
        loop = self._activate_loop()
        agent = self.store.get_team_agent(loop.agent_id)
        assert agent is not None
        self.assertEqual(agent.kind, TeamAgentKind.LOOP)
        anyone = WorkRequest("regular work", AssignmentMode.ANYONE)
        specific = WorkRequest(
            "regular work",
            AssignmentMode.SPECIFIC,
            requested_handle=agent.handle,
        )
        self.assertIsNone(pick_idle_agent([agent], anyone))
        self.assertIsNone(pick_idle_agent([agent], specific))
        self.assertNotIn(agent, self.controller._regular_team_agents())

    def test_si_11_public_and_private_loops_enforce_the_same_boundary(self):
        loop = self._activate_loop()
        self.controller.handle_event(
            {
                "event": {
                    "type": "message",
                    "channel": loop.channel_id,
                    "ts": "206.000001",
                    "user": "UPRIVATE",
                    "text": "private-channel injection",
                }
            }
        )
        self.store.update_loop_identity(
            loop.loop_id,
            title=loop.title,
            mission=loop.mission,
            channel_name=loop.channel_name,
            visibility=LoopVisibility.PUBLIC,
            provider=loop.provider,
            model=loop.model,
            permission_mode=loop.permission_mode,
            cwd=loop.cwd,
        )
        self.controller.handle_event(
            {
                "event": {
                    "type": "message",
                    "channel": loop.channel_id,
                    "ts": "206.000002",
                    "user": "UPUBLIC",
                    "text": "public-channel injection",
                }
            }
        )
        marked = {
            ts
            for channel_id, ts, name in self.gateway.reactions
            if channel_id == loop.channel_id and name == "no_entry_sign"
        }
        self.assertEqual(marked, {"206.000001", "206.000002"})
        self.assertEqual(
            {(channel_id, user_id) for channel_id, user_id, _ in self.gateway.ephemerals},
            {(loop.channel_id, "UPRIVATE"), (loop.channel_id, "UPUBLIC")},
        )
        self.assertEqual(self.store.list_loop_journal(loop.loop_id), [])

    def test_si_12_every_loop_prompt_repeats_the_withheld_content_boundary(self):
        loop = self._activate_loop()
        _, agent, run = self._create_running_loop_task(loop)
        prompts = [
            build_loop_resolution_prompt("create a recurring cost monitor", now=utc_now()),
            build_loop_run_prompt(
                loop,
                run,
                journal_rendered="Standing owner instructions:\n- none",
                now=utc_now(),
                bot_name=agent.full_name,
            ),
            build_loop_fetch_result(run_number=1, rendered_messages="Owner: trusted"),
            build_loop_compaction_prompt(
                loop,
                journal_rendered="Standing owner instructions:\n- none",
                bot_name=agent.full_name,
            ),
        ]
        for prompt in prompts:
            self.assertIn("withhold", prompt.lower())

    def test_backfill_scans_main_and_active_loop_channels(self):
        loop = self._activate_loop()
        backfill = SlackMessageBackfill(
            self.store,
            self.gateway,
            self.controller,
            team_id="TEXAMPLE",
        )
        recovered = backfill.recover_since("100.000000", include_threads=False)
        self.assertEqual(recovered, 0)
        scanned_channels = [channel_id for channel_id, _, _ in self.gateway.channel_message_calls]
        self.assertEqual(scanned_channels, ["CMAIN", loop.channel_id])

    def test_loop_slash_commands_stay_in_channel_and_reject_nonowners(self):
        loop = self._activate_loop()
        self.controller.handle_slash_command(
            {
                "channel_id": loop.channel_id,
                "user_id": "UBYSTANDER",
                "user_name": "bystander",
                "text": "loop status",
            }
        )
        self.assertEqual(self.store.get_setting("slack.human_user_id"), "UOWNER")
        self.assertEqual(self.gateway.ephemerals[-1][:2], (loop.channel_id, "UBYSTANDER"))
        self.assertEqual(
            self.controller._slash_command_target_channel_id({"channel_id": loop.channel_id}),
            loop.channel_id,
        )

    def test_loop_reactions_are_relayed_only_for_the_owner(self):
        loop = self._activate_loop()
        base_event = {
            "type": "reaction_added",
            "reaction": "eyes",
            "item": {"type": "message", "channel": loop.channel_id, "ts": "207.000001"},
        }
        with patch.object(
            self.controller,
            "_relay_user_reaction_to_agent",
            return_value=True,
        ) as relay:
            self.controller.handle_event(
                {"event": {**base_event, "user": "UBYSTANDER", "event_ts": "207.1"}}
            )
            relay.assert_not_called()
            self.controller.handle_event(
                {"event": {**base_event, "user": "UOWNER", "event_ts": "207.2"}}
            )
            relay.assert_called_once()

    def test_loop_runner_claims_due_occurrence_and_starts_fresh_run(self):
        loop = self._activate_loop()
        _, due_at = self._make_loop_due(loop)
        self.runtime.started.clear()
        runner = LoopRunner(self.store, self.controller, poll_seconds=0.01)

        self.assertEqual(runner.sync_once(), 1)

        current = self.store.get_loop(loop.loop_id)
        assert current is not None
        self.assertEqual(current.run_count, 1)
        self.assertGreater(current.next_run_at, due_at)
        task, agent, run, thread = self._running_task_and_run(current)
        self.assertEqual(run.kind, LoopRunKind.SCHEDULED)
        self.assertEqual(run.due_at, due_at)
        self.assertEqual(task.kind, AgentTaskKind.LOOP_RUN)
        self.assertEqual(task.channel_id, loop.channel_id)
        self.assertEqual(task.thread_ts, run.thread_ts)
        self.assertEqual(task.requested_by_slack_user, "UOWNER")
        self.assertEqual(task.metadata[LOOP_ID_METADATA_KEY], loop.loop_id)
        self.assertEqual(task.metadata[LOOP_RUN_ID_METADATA_KEY], run.run_id)
        self.assertEqual(task.metadata["permission_mode"], loop.permission_mode.value)
        self.assertEqual(task.metadata["model_override"], loop.model)
        self.assertIsNone(task.session_id)
        self.assertIn("[LOOP HARNESS: state]", task.prompt)
        self.assertEqual(self.runtime.started, [(task, agent, thread)])
        self.assertIn("▶ Run #1", self.gateway.posts[-1]["text"])

    def test_scheduled_overlap_advances_schedule_and_records_skipped_run(self):
        loop = self._activate_loop()
        first_due, _ = self._make_loop_due(loop)
        self.controller.fire_due_loop(first_due)
        running = self.store.running_loop_run(loop.loop_id)
        assert running is not None
        current = self.store.get_loop(loop.loop_id)
        assert current is not None
        second_due, due_at = self._make_loop_due(current)
        starts_before = len(self.runtime.started)

        self.assertTrue(self.controller.fire_due_loop(second_due))

        runs = self.store.list_loop_runs(loop.loop_id)
        skipped = next(item for item in runs if item.status == LoopRunStatus.SKIPPED)
        self.assertEqual(skipped.run_number, 2)
        self.assertEqual(skipped.kind, LoopRunKind.SCHEDULED)
        self.assertEqual(skipped.due_at, due_at)
        self.assertIn(f"run #{running.run_number}", skipped.error)
        self.assertEqual(len(self.runtime.started), starts_before)
        advanced = self.store.get_loop(loop.loop_id)
        assert advanced is not None
        self.assertGreater(advanced.next_run_at, due_at)
        self.assertIn("Skipped this occurrence", self.gateway.posts[-1]["text"])

    def test_manual_run_leaves_schedule_untouched(self):
        loop = self._activate_loop()
        scheduled_for = loop.next_run_at

        self.assertTrue(self.controller.fire_loop_now(loop))

        current = self.store.get_loop(loop.loop_id)
        assert current is not None
        self.assertEqual(current.next_run_at, scheduled_for)
        run = self.store.running_loop_run(loop.loop_id)
        assert run is not None
        self.assertEqual(run.kind, LoopRunKind.MANUAL)

    def test_concurrent_manual_runs_start_once_and_record_overlap(self):
        loop = self._activate_loop()
        starts_before = len(self.runtime.started)
        barrier = threading.Barrier(3)
        results = []

        def fire():
            barrier.wait()
            results.append(self.controller.fire_loop_now(loop))

        workers = [threading.Thread(target=fire) for _ in range(2)]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join(timeout=2)

        self.assertFalse(any(worker.is_alive() for worker in workers))
        self.assertEqual(results, [True, True])
        runs = self.store.list_loop_runs(loop.loop_id)
        self.assertEqual(
            [run.status for run in runs].count(LoopRunStatus.RUNNING),
            1,
        )
        self.assertEqual(
            [run.status for run in runs].count(LoopRunStatus.SKIPPED),
            1,
        )
        self.assertEqual(len(self.runtime.started), starts_before + 1)

    def test_loop_summary_finalizes_run_and_writes_durable_journal(self):
        loop = self._activate_loop()
        self.controller.fire_loop_now(loop)
        task, agent, run, thread = self._running_task_and_run(loop)
        signal = AGENT_LOOP_SUMMARY_SIGNAL_PREFIX + json.dumps(
            {
                "summary": "Checked recent billing and found no anomaly.",
                "status": "ok",
                "carry": {"last_invoice": "example-001"},
            }
        )
        self.controller.handle_runtime_agent_control(task, agent, thread, signal)

        self.controller.handle_runtime_task_done(task, agent, thread)

        finished = self.store.get_loop_run(run.run_id)
        assert finished is not None
        self.assertEqual(finished.status, LoopRunStatus.DONE)
        self.assertIsNotNone(finished.finished_at)
        self.assertEqual(
            self.store.get_agent_task(task.task_id).status,
            AgentTaskStatus.DONE,
        )
        entries = self.store.list_loop_journal(loop.loop_id)
        summaries = [entry for entry in entries if entry.kind == "run_summary"]
        self.assertEqual(len(summaries), 1)
        self.assertIn("[ok] Checked recent billing", summaries[0].content)
        self.assertIn('Carried state: {"last_invoice": "example-001"}', summaries[0].content)
        self.assertIn("https://example.slack.com/archives/", summaries[0].content)

    def test_invalid_loop_summary_posts_only_one_nudge(self):
        loop = self._activate_loop()
        self.controller.fire_loop_now(loop)
        task, agent, _, thread = self._running_task_and_run(loop)
        replies_before = len(self.gateway.thread_replies)

        for _ in range(2):
            self.controller.handle_runtime_agent_control(
                task,
                agent,
                thread,
                AGENT_LOOP_SUMMARY_SIGNAL_PREFIX + '{"summary":""}',
            )

        nudges = self.gateway.thread_replies[replies_before:]
        self.assertEqual(len(nudges), 1)
        self.assertIn("could not record that run summary", nudges[0]["text"])

    def test_failed_mission_summary_is_a_successful_harness_run(self):
        loop = self._activate_loop()
        self.store.record_loop_failure(loop.loop_id, "prior runtime failure")
        self.store.record_loop_failure(loop.loop_id, "prior runtime failure")
        loop = self.store.get_loop(loop.loop_id)
        assert loop is not None
        self.controller.fire_loop_now(loop)
        task, agent, run, thread = self._running_task_and_run(loop)
        self.controller.handle_runtime_agent_control(
            task,
            agent,
            thread,
            AGENT_LOOP_SUMMARY_SIGNAL_PREFIX
            + '{"summary":"The mission found a service failure.","status":"failed"}',
        )

        self.controller.handle_runtime_task_done(task, agent, thread)

        finished = self.store.get_loop_run(run.run_id)
        current = self.store.get_loop(loop.loop_id)
        assert finished is not None and current is not None
        self.assertEqual(finished.status, LoopRunStatus.DONE)
        self.assertEqual(current.status, LoopStatus.ACTIVE)
        self.assertEqual(current.consecutive_failures, 0)

    def test_thread_done_finalizes_loop_without_posting_roster(self):
        loop = self._activate_loop()
        self.controller.fire_loop_now(loop)
        task, agent, run, thread = self._running_task_and_run(loop)
        self.controller.handle_runtime_agent_control(
            task,
            agent,
            thread,
            AGENT_LOOP_SUMMARY_SIGNAL_PREFIX + '{"summary":"thread complete"}',
        )
        posts_before = len(self.gateway.posts)

        handled = self.controller.handle_runtime_agent_control(
            task,
            agent,
            thread,
            "SLACKGENTIC: THREAD_DONE",
        )

        self.assertTrue(handled)
        finished = self.store.get_loop_run(run.run_id)
        assert finished is not None
        self.assertEqual(finished.status, LoopRunStatus.DONE)
        self.assertEqual(len(self.gateway.posts), posts_before)

    def test_loop_run_swallow_generic_task_control_signals(self):
        loop = self._activate_loop()
        self.controller.fire_loop_now(loop)
        task, agent, _, thread = self._running_task_and_run(loop)
        with patch.object(
            self.controller,
            "_schedule_agent_timer",
            side_effect=AssertionError("loop runs cannot schedule generic timers"),
        ):
            handled = self.controller.handle_runtime_agent_control(
                task,
                agent,
                thread,
                'SLACKGENTIC: TIMER {"delay_seconds":60,"prompt":"delegate"}',
            )
        self.assertTrue(handled)

    def test_missing_summary_gets_one_nudge_then_falls_back(self):
        loop = self._activate_loop()
        self.controller.fire_loop_now(loop)
        task, agent, run, thread = self._running_task_and_run(loop)

        self.controller.handle_runtime_task_done(task, agent, thread)

        still_running = self.store.get_loop_run(run.run_id)
        assert still_running is not None
        self.assertEqual(still_running.status, LoopRunStatus.RUNNING)
        nudged_task = self.store.get_agent_task(task.task_id)
        assert nudged_task is not None
        self.assertTrue(nudged_task.metadata["loop_summary_nudge"])
        self.assertIn(AGENT_LOOP_SUMMARY_SIGNAL_PREFIX, self.runtime.sent[-1][1])

        self.runtime.send_to_task = lambda task_id, message: False
        self.controller.handle_runtime_task_done(nudged_task, agent, thread)
        finished = self.store.get_loop_run(run.run_id)
        assert finished is not None
        self.assertEqual(finished.status, LoopRunStatus.DONE)
        entries = self.store.list_loop_journal(loop.loop_id)
        self.assertIn("completed without a summary", entries[-1].content)

    def test_spawn_failure_marks_run_failed(self):
        loop = self._activate_loop()
        self.runtime.start_task = lambda task, agent, thread: False

        self.assertTrue(self.controller.fire_loop_now(loop))

        run = self.store.list_loop_runs(loop.loop_id)[0]
        self.assertEqual(run.status, LoopRunStatus.FAILED)
        current = self.store.get_loop(loop.loop_id)
        assert current is not None
        self.assertEqual(current.consecutive_failures, 1)
        self.assertIn("runtime could not start", current.last_error)

    def test_three_cancelled_runs_pause_loop_and_notify_owner(self):
        loop = self._activate_loop()
        for expected_failures in range(1, 4):
            current = self.store.get_loop(loop.loop_id)
            assert current is not None
            self.assertTrue(self.controller.fire_loop_now(current))
            task, agent, run, thread = self._running_task_and_run(current)
            self.store.update_agent_task_status(task.task_id, AgentTaskStatus.CANCELLED)
            cancelled = self.store.get_agent_task(task.task_id)
            assert cancelled is not None
            self.controller.handle_runtime_task_done(cancelled, agent, thread)
            failed = self.store.get_loop_run(run.run_id)
            assert failed is not None
            self.assertEqual(failed.status, LoopRunStatus.FAILED)
            after = self.store.get_loop(loop.loop_id)
            assert after is not None
            self.assertEqual(after.consecutive_failures, expected_failures)

        paused = self.store.get_loop(loop.loop_id)
        assert paused is not None
        self.assertEqual(paused.status, LoopStatus.PAUSED)
        self.assertTrue(
            any(
                "Paused after 3 consecutive failed runs" in post["text"]
                for post in self.gateway.posts
            )
        )

    def test_inline_compaction_supersedes_run_memory_but_preserves_owner_notes(self):
        loop = self._activate_loop()
        now = utc_now()
        old_summary = LoopJournalEntry(
            "journal_old_summary",
            loop.loop_id,
            "run_summary",
            "run #0: old detail",
            now - timedelta(minutes=2),
        )
        owner_note = LoopJournalEntry(
            "journal_owner_note",
            loop.loop_id,
            "owner_note",
            "Never discard this standing instruction.",
            now - timedelta(minutes=1),
        )
        self.store.add_loop_journal_entry(old_summary)
        self.store.add_loop_journal_entry(owner_note)
        self.controller.fire_loop_now(loop)
        task, agent, run, thread = self._running_task_and_run(loop)

        handled = self.controller.handle_runtime_agent_control(
            task,
            agent,
            thread,
            AGENT_LOOP_COMPACT_SIGNAL_PREFIX
            + json.dumps({"snapshot": "Durable compacted memory."}),
        )

        self.assertTrue(handled)
        all_entries = self.store.list_loop_journal(
            loop.loop_id,
            include_superseded=True,
            limit=100,
        )
        old = next(entry for entry in all_entries if entry.entry_id == old_summary.entry_id)
        owner = next(entry for entry in all_entries if entry.entry_id == owner_note.entry_id)
        snapshot = next(entry for entry in all_entries if entry.kind == "compaction")
        self.assertEqual(old.superseded_by, snapshot.entry_id)
        self.assertIsNone(owner.superseded_by)
        self.assertEqual(snapshot.run_id, run.run_id)
        visible = self.store.list_loop_journal(loop.loop_id)
        self.assertIn(owner_note, visible)
        self.assertTrue(any(entry.content == "Durable compacted memory." for entry in visible))

    def test_finalization_queues_and_runner_starts_automatic_compaction(self):
        loop = self._activate_loop()
        self.store.add_loop_journal_entry(
            LoopJournalEntry(
                "journal_large_system",
                loop.loop_id,
                "system",
                "x" * (LOOP_COMPACTION_TRIGGER_CHARS + 1),
                utc_now(),
            )
        )
        self.controller.fire_loop_now(loop)
        task, agent, _, thread = self._running_task_and_run(loop)
        self.controller.handle_runtime_agent_control(
            task,
            agent,
            thread,
            AGENT_LOOP_SUMMARY_SIGNAL_PREFIX + '{"summary":"run complete"}',
        )
        self.controller.handle_runtime_task_done(task, agent, thread)
        pending = self.store.get_loop(loop.loop_id)
        assert pending is not None
        self.assertTrue(pending.metadata["compaction_pending"])
        starts_before = len(self.runtime.started)

        runner = LoopRunner(self.store, self.controller, poll_seconds=0.01)
        self.assertEqual(runner.sync_once(), 1)

        compacting = self.store.running_loop_run(loop.loop_id)
        assert compacting is not None
        self.assertEqual(compacting.kind, LoopRunKind.COMPACTION)
        self.assertEqual(len(self.runtime.started), starts_before + 1)
        compaction_task = self.store.get_agent_task(compacting.task_id)
        assert compaction_task is not None
        self.assertIn("[LOOP HARNESS: memory compaction]", compaction_task.prompt)
        self.assertIn("x" * 100, compaction_task.prompt)
        current = self.store.get_loop(loop.loop_id)
        assert current is not None
        self.assertNotIn("compaction_pending", current.metadata)

        compaction_agent = self.store.get_team_agent(loop.agent_id)
        assert compaction_agent is not None
        compaction_thread = SlackThreadRef(
            loop.channel_id,
            compacting.thread_ts,
            compacting.thread_ts,
        )
        self.controller.handle_runtime_agent_control(
            compaction_task,
            compaction_agent,
            compaction_thread,
            AGENT_LOOP_COMPACT_SIGNAL_PREFIX + '{"snapshot":"automatic snapshot"}',
        )
        self.controller.handle_runtime_task_done(
            compaction_task,
            compaction_agent,
            compaction_thread,
        )
        compacted = self.store.get_loop_run(compacting.run_id)
        assert compacted is not None
        self.assertEqual(compacted.status, LoopRunStatus.DONE)
        self.assertFalse(
            any(
                entry.kind == "run_summary" and entry.run_id == compacting.run_id
                for entry in self.store.list_loop_journal(
                    loop.loop_id,
                    include_superseded=True,
                    limit=100,
                )
            )
        )

    def test_loop_fetch_budget_is_durable_across_stale_task_callbacks(self):
        loop = self._activate_loop()
        task, agent, run = self._create_running_loop_task(loop)
        self.gateway.thread_history_messages[(loop.channel_id, run.thread_ts)] = [
            {"ts": run.thread_ts, "user": "UOWNER", "text": "trusted history"}
        ]
        thread = SlackThreadRef(loop.channel_id, run.thread_ts)

        for _ in range(6):
            self.controller.handle_runtime_agent_control(
                task,
                agent,
                thread,
                AGENT_LOOP_FETCH_SIGNAL_PREFIX + '{"run":1}',
            )

        fetch_calls = [
            call
            for call in self.gateway.thread_message_calls
            if call[0] == loop.channel_id and call[1] == run.thread_ts
        ]
        self.assertEqual(len(fetch_calls), 5)
        self.assertIn("budget exhausted", self.runtime.sent[-1][1])
        current_task = self.store.get_agent_task(task.task_id)
        assert current_task is not None
        self.assertEqual(current_task.metadata["loop_fetch_count"], 5)

    def test_restart_reconciliation_finalizes_terminal_loop_run(self):
        loop = self._activate_loop()
        self.controller.fire_loop_now(loop)
        task, agent, run, thread = self._running_task_and_run(loop)
        self.controller.handle_runtime_agent_control(
            task,
            agent,
            thread,
            AGENT_LOOP_SUMMARY_SIGNAL_PREFIX + '{"summary":"recovered summary"}',
        )
        self.store.update_agent_task_status(task.task_id, AgentTaskStatus.DONE)

        self.assertEqual(self.controller.reconcile_loop_runs(), 1)

        reconciled = self.store.get_loop_run(run.run_id)
        assert reconciled is not None
        self.assertEqual(reconciled.status, LoopRunStatus.DONE)

    def test_dead_loop_channel_pauses_and_notifies_anchor_thread(self):
        loop = self._activate_loop()

        class DeadChannelError(RuntimeError):
            def __init__(self, message):
                super().__init__(message)
                self.response = {"error": "is_archived"}

        with patch.object(
            self.gateway,
            "post_session_parent",
            side_effect=DeadChannelError("archived"),
        ):
            self.assertTrue(self.controller.fire_loop_now(loop))

        paused = self.store.get_loop(loop.loop_id)
        assert paused is not None
        self.assertEqual(paused.status, LoopStatus.PAUSED)
        self.assertIn("is_archived", paused.last_error)
        self.assertTrue(
            any(
                reply["thread"].channel_id == loop.anchor_channel_id and "Paused" in reply["text"]
                for reply in self.gateway.thread_replies
            )
        )


if __name__ == "__main__":
    unittest.main()
