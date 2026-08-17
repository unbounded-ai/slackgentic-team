import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent_harness.loops import (
    AGENT_LOOP_FETCH_SIGNAL_PREFIX,
    AGENT_LOOP_SIGNAL_PREFIX,
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
from agent_harness.slack.app import SlackMessageBackfill, SlackTeamController
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


if __name__ == "__main__":
    unittest.main()
