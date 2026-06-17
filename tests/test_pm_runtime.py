from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from agent_harness.models import (
    AgentSession,
    AgentTaskStatus,
    AssignmentMode,
    DeferredWorkStatus,
    PmInitiativeStatus,
    Provider,
    SessionStatus,
    SlackThreadRef,
    TeamAgentKind,
    WorkRequest,
    utc_now,
)
from agent_harness.pm import (
    AGENT_PM_PLAN_SIGNAL_PREFIX,
    PM_EXTENSION_KNOWN_IDS_METADATA_KEY,
    PM_INITIATIVE_ID_METADATA_KEY,
    PM_RESOLUTION_ATTEMPTS_METADATA_KEY,
    PM_RESOLUTION_METADATA_KEY,
    PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY,
    PM_SUBTASK_LOCAL_ID_METADATA_KEY,
)
from agent_harness.runtime.tasks import (
    AGENT_THREAD_DONE_SIGNAL,
    MANAGED_RUN_MAX_STALL_RECOVERIES,
    MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY,
    MANAGED_RUN_STARTED_METADATA_KEY,
    build_task_prompt,
)
from agent_harness.slack import encode_action_value
from agent_harness.slack.app import (
    TASK_REACTION_DONE,
    TASK_REACTION_IN_PROGRESS,
    PMInitiativeRunner,
    SlackTeamController,
)
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team, create_agent_task, hire_team_agents
from tests.test_slack_app import FakeGateway, FakeRuntime


def _build_plan_signal(initiative_id: str, *, handles: list[str]) -> str:
    payload = {
        "title": "Ship feature X",
        "summary": "Roll out feature X.",
        "subtasks": [
            {
                "id": "investigate",
                "title": "Investigate",
                "task": "Look at the current state and write a one-paragraph summary.",
                "target": "somebody",
                "depends_on": [],
            },
            {
                "id": "implement",
                "title": "Implement",
                "task": "Implement feature X based on the investigation summary.",
                "target": "somebody",
                "depends_on": ["investigate"],
            },
        ],
    }
    return f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}"


def _build_large_plan_signal() -> str:
    subtasks = []
    for index in range(20):
        local_id = f"s{index:02d}"
        title = f"Step {index:02d} " + ("with detailed approval context " * 8)
        subtasks.append(
            {
                "id": local_id,
                "title": title,
                "task": f"Complete {local_id}.",
                "target": "somebody",
                "depends_on": [f"s{index - 1:02d}"] if index else [],
            }
        )
    payload = {
        "title": "Large PM plan",
        "summary": "A plan large enough to require multiple Slack section blocks.",
        "subtasks": subtasks,
    }
    return f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}"


class PmDispatchTests(unittest.TestCase):
    def test_pm_message_creates_initiative_and_starts_resolver(self):
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
                            "text": "pm: ship a new logging stack",
                            "ts": "171.000001",
                        }
                    }
                )

                initiatives = store.list_pm_initiatives(
                    statuses=(PmInitiativeStatus.PLANNING, PmInitiativeStatus.ACTIVE)
                )
                self.assertEqual(len(initiatives), 1)
                self.assertEqual(initiatives[0].status, PmInitiativeStatus.PLANNING)
                self.assertEqual(initiatives[0].summary, "ship a new logging stack")
                self.assertEqual(initiatives[0].requested_by_slack_user, "U1")
                self.assertEqual(len(runtime.started), 1)
                started_task, started_agent, _thread = runtime.started[0]
                self.assertTrue(started_task.metadata.get(PM_RESOLUTION_METADATA_KEY))
                self.assertEqual(
                    started_task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY),
                    initiatives[0].initiative_id,
                )
                self.assertEqual(started_agent.agent_id, agent.agent_id)
                self.assertIn(
                    "ship a new logging stack",
                    gateway.thread_replies[-1]["text"],
                )
            finally:
                store.close()

    def test_pm_message_does_not_match_non_pm_messages(self):
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
                            "text": "please open the PR",
                            "ts": "171.000002",
                        }
                    }
                )

                self.assertEqual(store.list_pm_initiatives(), [])
            finally:
                store.close()


class PmChunkDispatchTests(unittest.TestCase):
    def test_post_agent_chunk_with_pm_plan_signal_parks_plan_immediately(self):
        """End-to-end: a PM agent emits a chunk containing the PM_PLAN
        control line, ManagedTaskRuntime._post_agent_chunk extracts the
        signal, and SlackTeamController.handle_runtime_agent_control
        parks the plan + posts the approval card — all on the SAME chunk,
        without waiting for the managed process to exit.

        Regression test: PM_PLAN was previously queued in
        `running.control_signals` and only flushed when the worker
        completed. With managed Claude staying alive across turns
        (--input-format stream-json), the queue was never flushed and
        the approval card never posted, even though the extractor
        correctly stripped the signal line.
        """
        import threading

        from agent_harness.config import AgentCommandConfig
        from agent_harness.runtime.tasks import ManagedTaskRuntime, RunningTask
        from tests.test_task_runtime import OneShotProcess

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Ship feature X",
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                )
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=OneShotProcess,
                    poll_seconds=0.01,
                    on_agent_control=controller.handle_runtime_agent_control,
                )
                running = RunningTask(
                    task=pm_task,
                    agent=agent,
                    process=OneShotProcess(None),
                    thread=thread_ref,
                    worker=threading.Thread(),
                )

                plan_signal = _build_plan_signal(
                    initiative.initiative_id,
                    handles=[agent.handle],
                )
                chunk = f"Plan ready — two subtasks: investigate, then implement.\n{plan_signal}\n"

                runtime._post_agent_chunk(running, chunk)

                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, PmInitiativeStatus.AWAITING_APPROVAL)
                self.assertIsNotNone(refreshed.pending_plan_json)
                approval_reply = next(
                    item
                    for item in gateway.thread_replies
                    if "Start executing" in (item.get("text") or "")
                )
                self.assertIsNotNone(approval_reply.get("blocks"))
                self.assertNotIn(
                    "SLACKGENTIC: PM_PLAN",
                    " ".join(item.get("text") or "" for item in gateway.thread_replies),
                    msg=(
                        "PM_PLAN control line must not leak into Slack — the "
                        "extractor should have stripped it before the agent "
                        "chunk was posted."
                    ),
                )
            finally:
                store.close()


class PmFullLifecycleTests(unittest.TestCase):
    def test_pm_lifecycle_post_to_plan_to_approve_to_status_to_replan(self):
        """Drive a full PM initiative end-to-end through the actual handlers
        (no mocking of the dispatcher): post a brief, have the PM emit a
        PM_PLAN chunk through the real runtime, click ``Start executing``,
        ask for status, and request a replan. Verifies that each Slack-
        visible artifact lands and each state transition fires.

        This is the regression net that #105 and #106 needed before
        either was claimed shippable.
        """
        import threading

        from agent_harness.config import AgentCommandConfig
        from agent_harness.models import TeamAgentKind
        from agent_harness.runtime.tasks import ManagedTaskRuntime, RunningTask
        from agent_harness.team import hire_team_agents
        from tests.test_task_runtime import OneShotProcess

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                engineer = build_initial_model_team(0, 1)[0]
                store.upsert_team_agent(engineer)
                pm_agent = hire_team_agents(
                    [engineer],
                    1,
                    None,
                    seed="lifecycle-test",
                    kind=TeamAgentKind.PM,
                )[0]
                store.upsert_team_agent(pm_agent)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                # ---- STEP 1: post a `pm: ...` brief.
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "pm: ship the new logging stack",
                            "ts": "171.000001",
                        }
                    }
                )

                initiatives = store.list_pm_initiatives()
                self.assertEqual(len(initiatives), 1)
                initiative = initiatives[0]
                self.assertEqual(initiative.status, PmInitiativeStatus.PLANNING)
                self.assertEqual(initiative.pm_agent_id, pm_agent.agent_id)
                self.assertEqual(len(runtime.started), 1)
                pm_task, started_pm, pm_thread = runtime.started[0]
                self.assertEqual(started_pm.handle, pm_agent.handle)
                self.assertEqual(
                    pm_task.metadata[PM_INITIATIVE_ID_METADATA_KEY],
                    initiative.initiative_id,
                )
                self.assertIn("Active Slackgentic worker handles:", pm_task.prompt)
                self.assertIn(f"@{engineer.handle}", pm_task.prompt)
                self.assertNotIn(f"@{pm_agent.handle}", pm_task.prompt)
                # The real ManagedTaskRuntime marks a task ACTIVE when it
                # starts the worker; FakeRuntime is a record-only stub, so
                # mirror that side effect here. The replan path (step 5)
                # below early-returns on a QUEUED resolver task.
                store.update_agent_task_status(pm_task.task_id, AgentTaskStatus.ACTIVE)
                pm_task = store.get_agent_task(pm_task.task_id) or pm_task

                # ---- STEP 2: PM emits a chunk containing PM_PLAN.
                # Drive a real ManagedTaskRuntime so the
                # extractor → dispatcher chain actually fires; this is the
                # path PR #105 and PR #106 jumped over.
                managed_runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=OneShotProcess,
                    poll_seconds=0.01,
                    on_agent_control=controller.handle_runtime_agent_control,
                )
                running = RunningTask(
                    task=pm_task,
                    agent=pm_agent,
                    process=OneShotProcess(None),
                    thread=pm_thread,
                    worker=threading.Thread(),
                )
                plan_signal = _build_plan_signal(
                    initiative.initiative_id,
                    handles=[pm_agent.handle, engineer.handle],
                )
                managed_runtime._post_agent_chunk(
                    running,
                    f"Plan ready — investigate, then implement.\n{plan_signal}\n",
                )

                parked = store.get_pm_initiative(initiative.initiative_id)
                assert parked is not None
                self.assertEqual(parked.status, PmInitiativeStatus.AWAITING_APPROVAL)
                self.assertIsNotNone(parked.pending_plan_json)
                approval_reply = next(
                    item
                    for item in gateway.thread_replies
                    if "Start executing" in (item.get("text") or "")
                )
                self.assertIsNotNone(approval_reply.get("blocks"))
                self.assertNotIn(
                    "SLACKGENTIC: PM_PLAN",
                    " ".join(item.get("text") or "" for item in gateway.thread_replies),
                )
                request_reactions = {
                    item["name"] for item in gateway._current_reactions("C1", "171.000001")
                }
                self.assertIn(TASK_REACTION_IN_PROGRESS, request_reactions)
                self.assertNotIn(TASK_REACTION_DONE, request_reactions)

                # ---- STEP 3: user clicks ``Start executing``.
                approve_action = encode_action_value(
                    "pm_initiative.start",
                    initiative_id=initiative.initiative_id,
                )
                controller.handle_block_action(
                    {
                        "actions": [{"action_id": "pm_initiative.start", "value": approve_action}],
                        "channel": {"id": "C1"},
                        "message": {"ts": approval_reply["ts"]},
                        "user": {"id": "U1"},
                    }
                )

                approved = store.get_pm_initiative(initiative.initiative_id)
                assert approved is not None
                self.assertEqual(approved.status, PmInitiativeStatus.ACTIVE)
                self.assertIsNone(approved.pending_plan_json)
                strip_updates = [
                    update for update in gateway.updates if update["ts"] == approval_reply["ts"]
                ]
                self.assertTrue(strip_updates)
                self.assertTrue(strip_updates[-1]["blocks"])
                self.assertFalse(
                    any(block.get("type") == "actions" for block in strip_updates[-1]["blocks"])
                )
                subtasks = store.list_pm_subtasks(initiative.initiative_id)
                self.assertEqual(
                    [s.local_id for s in subtasks],
                    ["investigate", "implement"],
                )
                root_deferred = store.get_deferred_work(subtasks[0].deferred_id)
                child_deferred = store.get_deferred_work(subtasks[1].deferred_id)
                assert root_deferred is not None
                assert child_deferred is not None
                self.assertIn(
                    root_deferred.status,
                    {DeferredWorkStatus.READY, DeferredWorkStatus.DONE},
                )
                self.assertEqual(child_deferred.status, DeferredWorkStatus.WAITING_DEPS)

                # ---- STEP 4: ask for status in the initiative thread.
                replies_before_status = len(gateway.thread_replies)
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "pm status",
                            "ts": "171.000020",
                            "thread_ts": pm_thread.thread_ts,
                        }
                    }
                )
                status_replies = gateway.thread_replies[replies_before_status:]
                self.assertTrue(status_replies, "expected a status reply")
                status_text = " ".join(item.get("text") or "" for item in status_replies)
                self.assertIn("investigate", status_text)
                self.assertIn("implement", status_text)

                # ---- STEP 5: ask for a replan.
                started_before_replan = len(runtime.started)
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "pm replan: also add a deploy step at the end",
                            "ts": "171.000030",
                            "thread_ts": pm_thread.thread_ts,
                        }
                    }
                )
                # A replan re-dispatches the PM resolver task with the
                # new user context (either by restarting the existing
                # PM task or starting a fresh one) — the runtime must
                # see at least one additional start.
                self.assertGreater(len(runtime.started), started_before_replan)
            finally:
                store.close()


class PmEntrySurfaceTests(unittest.TestCase):
    """Lifecycle coverage for every PM entry-point surface."""

    def _hire_pm_agent(self, store):
        from agent_harness.models import TeamAgentKind
        from agent_harness.team import hire_team_agents

        engineer = build_initial_model_team(0, 1)[0]
        store.upsert_team_agent(engineer)
        pm_agent = hire_team_agents(
            [engineer],
            1,
            None,
            seed="entry-surface-test",
            kind=TeamAgentKind.PM,
        )[0]
        store.upsert_team_agent(pm_agent)
        return pm_agent, engineer

    def _drive_plan_chunk_and_assert_parked(
        self,
        *,
        store,
        gateway,
        controller,
        pm_task,
        pm_agent,
        pm_thread,
        initiative,
        engineer,
    ):
        import threading

        from agent_harness.config import AgentCommandConfig
        from agent_harness.runtime.tasks import ManagedTaskRuntime, RunningTask
        from tests.test_task_runtime import OneShotProcess

        store.update_agent_task_status(pm_task.task_id, AgentTaskStatus.ACTIVE)
        managed_runtime = ManagedTaskRuntime(
            store,
            gateway,
            AgentCommandConfig(),
            process_factory=OneShotProcess,
            poll_seconds=0.01,
            on_agent_control=controller.handle_runtime_agent_control,
        )
        running = RunningTask(
            task=pm_task,
            agent=pm_agent,
            process=OneShotProcess(None),
            thread=pm_thread,
            worker=threading.Thread(),
        )
        plan_signal = _build_plan_signal(
            initiative.initiative_id,
            handles=[pm_agent.handle, engineer.handle],
        )
        managed_runtime._post_agent_chunk(
            running,
            f"Plan ready — investigate, then implement.\n{plan_signal}\n",
        )
        parked = store.get_pm_initiative(initiative.initiative_id)
        assert parked is not None
        self.assertEqual(parked.status, PmInitiativeStatus.AWAITING_APPROVAL)
        return next(
            item for item in gateway.thread_replies if "Start executing" in (item.get("text") or "")
        )

    def test_lifecycle_from_at_pm_handle_text_mention(self):
        """Entry point: `@<pm-handle> <brief>` plain-text mention.

        Same lifecycle the `pm: ...` form goes through, but via the
        `@<pm-handle>` routing branch in `_handle_pm_request`.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                pm_agent, engineer = self._hire_pm_agent(store)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store, gateway, default_channel_id="C1", runtime=runtime
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": f"@{pm_agent.handle} ship the new logging stack",
                            "ts": "171.000001",
                        }
                    }
                )

                initiatives = store.list_pm_initiatives()
                self.assertEqual(len(initiatives), 1)
                self.assertEqual(initiatives[0].pm_agent_id, pm_agent.agent_id)
                pm_task, _, pm_thread = runtime.started[0]
                approval_reply = self._drive_plan_chunk_and_assert_parked(
                    store=store,
                    gateway=gateway,
                    controller=controller,
                    pm_task=pm_task,
                    pm_agent=pm_agent,
                    pm_thread=pm_thread,
                    initiative=initiatives[0],
                    engineer=engineer,
                )
                self.assertIsNotNone(approval_reply.get("blocks"))
            finally:
                store.close()

    def test_lifecycle_from_roster_modal_now(self):
        """Entry point: roster modal `Assign work` → `now` targeting a PM.

        Verifies the consolidated routing redirect lands the same
        approval card as the text entry points.
        """
        from tests.test_slack_app import _roster_work_submission_payload

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                pm_agent, engineer = self._hire_pm_agent(store)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store, gateway, default_channel_id="C1", runtime=runtime
                )

                response = controller.handle_view_submission(
                    _roster_work_submission_payload(
                        pm_agent,
                        prompt="ship the new logging stack",
                        timing="now",
                    )
                )

                self.assertIsNone(response)
                initiatives = store.list_pm_initiatives()
                self.assertEqual(len(initiatives), 1)
                self.assertEqual(initiatives[0].pm_agent_id, pm_agent.agent_id)
                pm_task, _, pm_thread = runtime.started[0]
                approval_reply = self._drive_plan_chunk_and_assert_parked(
                    store=store,
                    gateway=gateway,
                    controller=controller,
                    pm_task=pm_task,
                    pm_agent=pm_agent,
                    pm_thread=pm_thread,
                    initiative=initiatives[0],
                    engineer=engineer,
                )
                self.assertIsNotNone(approval_reply.get("blocks"))
            finally:
                store.close()


class PmStallRecoveryTests(unittest.TestCase):
    def test_pm_resolver_stops_after_parking_plan_not_stall_recovered(self):
        """Parking a plan is a terminal resolver action.

        The PM initiative may still be waiting for the human to press Start
        executing, but the managed PM worker must not stay ACTIVE and later
        trip the stall recovery prompt.
        """
        import time

        from agent_harness.config import AgentCommandConfig
        from agent_harness.runtime.tasks import (
            MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY,
            ManagedTaskRuntime,
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            requests = []
            processes = []

            class PlanThenHoldingProcess:
                def __init__(self, request):
                    self.request = request
                    self.reads = 0
                    self.alive = True

                def start(self):
                    pass

                def read_available(self, max_reads=20, timeout=0.05):
                    if self.reads:
                        return ""
                    self.reads += 1
                    plan_signal = _build_plan_signal(
                        initiative.initiative_id,
                        handles=[pm_agent.handle, engineer.handle],
                    )
                    message = "Plan ready.\n" + plan_signal
                    return (
                        '{"type":"thread.started","thread_id":"codex-thread-1"}\n'
                        '{"type":"event_msg","payload":{"type":"agent_message","message":'
                        f"{json.dumps(message)}"
                        "}}\n"
                    )

                def is_alive(self):
                    return self.alive

                def terminate(self):
                    self.alive = False

            def process_factory(request):
                requests.append(request)
                process = PlanThenHoldingProcess(request)
                processes.append(process)
                return process

            try:
                store.init_schema()
                engineer = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(engineer)
                pm_agent = hire_team_agents(
                    [engineer],
                    1,
                    Provider.CODEX,
                    seed="stall-test",
                    kind=TeamAgentKind.PM,
                )[0]
                store.upsert_team_agent(pm_agent)
                thread_ref = SlackThreadRef("C1", "171.000001", "171.000001")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Plan the migration",
                    summary="plan the migration",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(pm_agent, "plan the migration", "C1"),
                    thread_ts=thread_ref.thread_ts,
                    parent_message_ts=thread_ref.message_ts,
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "plan the migration",
                    },
                )
                store.upsert_agent_task(pm_task)
                gateway = FakeGateway()
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                runtime = ManagedTaskRuntime(
                    store,
                    gateway,
                    AgentCommandConfig(),
                    process_factory=process_factory,
                    poll_seconds=0.01,
                    agent_progress_timeout=timedelta(minutes=30),
                    agent_stall_timeout=timedelta(seconds=0),
                    max_stall_recoveries=1,
                    on_agent_control=controller.handle_runtime_agent_control,
                )
                controller.runtime = runtime

                runtime.start_task(pm_task, pm_agent, thread_ref)
                for _ in range(200):
                    parked = store.get_pm_initiative(initiative.initiative_id)
                    current_task = store.get_agent_task(pm_task.task_id)
                    if (
                        parked is not None
                        and parked.status == PmInitiativeStatus.AWAITING_APPROVAL
                        and current_task is not None
                        and current_task.status == AgentTaskStatus.DONE
                        and not runtime.has_running_tasks()
                    ):
                        break
                    time.sleep(0.01)

                self.assertEqual(len(requests), 1)
                self.assertFalse(processes[0].is_alive())
                parked = store.get_pm_initiative(initiative.initiative_id)
                assert parked is not None
                self.assertEqual(parked.status, PmInitiativeStatus.AWAITING_APPROVAL)
                resolved_task = store.get_agent_task(pm_task.task_id)
                assert resolved_task is not None
                self.assertEqual(resolved_task.status, AgentTaskStatus.DONE)
                self.assertEqual(
                    resolved_task.metadata.get(MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY),
                    None,
                )
                reply_texts = [item.get("text") or "" for item in gateway.thread_replies]
                self.assertTrue(any("Start executing" in text for text in reply_texts))
                self.assertFalse(
                    any("restarted it with a status request" in text for text in reply_texts)
                )
            finally:
                if "runtime" in locals():
                    runtime.stop_all_running_tasks(status=AgentTaskStatus.CANCELLED)
                store.close()


class PmRuntimeTests(unittest.TestCase):
    def test_pm_plan_signal_parks_for_approval_without_firing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Ship feature X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                signal = _build_plan_signal(
                    initiative.initiative_id,
                    handles=[agent.handle],
                )
                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    agent,
                    thread_ref,
                    signal,
                )

                self.assertTrue(handled)
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, PmInitiativeStatus.AWAITING_APPROVAL)
                self.assertIsNotNone(refreshed.pending_plan_json)
                # No subtasks should be persisted, and nothing should fire.
                self.assertEqual(store.list_pm_subtasks(initiative.initiative_id), [])
                self.assertEqual(
                    [t for t, *_ in runtime.started if t.task_id != pm_task.task_id], []
                )
                # Plan ack mentions the approval gate.
                plan_message = gateway.thread_replies[-1]
                self.assertIn("Start executing", plan_message.get("text", ""))
            finally:
                store.close()

    def test_large_pm_plan_keeps_start_button_and_renders_dag_chart(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Large PM plan",
                    summary="Plan many steps.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Large PM plan",
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=FakeRuntime(),
                )

                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    agent,
                    thread_ref,
                    _build_large_plan_signal(),
                )

                self.assertTrue(handled)
                plan_message = gateway.thread_replies[-1]
                self.assertIn("DAG chart:", plan_message["text"])
                self.assertIn("s00 - Step 00", plan_message["text"])
                self.assertIn("`-- s01 - Step 01", plan_message["text"])
                blocks = plan_message["blocks"]
                self.assertIsNotNone(blocks)
                assert blocks is not None
                section_lengths = [
                    len(block["text"]["text"]) for block in blocks if block.get("type") == "section"
                ]
                self.assertGreater(len(section_lengths), 1)
                self.assertTrue(all(length <= 3000 for length in section_lengths))
                action_ids = [
                    element["action_id"]
                    for block in blocks
                    if block.get("type") == "actions"
                    for element in block.get("elements", [])
                ]
                self.assertIn("pm_initiative.start", action_ids)
                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.start",
                                "value": json.dumps(
                                    {
                                        "v": 1,
                                        "action": "pm_initiative.start",
                                        "initiative_id": initiative.initiative_id,
                                    }
                                ),
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": plan_message["ts"]},
                        "user": {"id": "U2"},
                    }
                )
                strip_updates = [
                    update for update in gateway.updates if update["ts"] == plan_message["ts"]
                ]
                self.assertTrue(strip_updates)
                self.assertLessEqual(len(strip_updates[-1]["text"]), 2602)
                self.assertFalse(
                    any(block.get("type") == "actions" for block in strip_updates[-1]["blocks"])
                )
            finally:
                store.close()

    def test_pm_initiative_start_button_dispatches_root_subtasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Ship feature X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                signal = _build_plan_signal(
                    initiative.initiative_id,
                    handles=[agent.handle],
                )
                controller.handle_runtime_agent_control(pm_task, agent, thread_ref, signal)
                parked = store.get_pm_initiative(initiative.initiative_id)
                assert parked is not None
                self.assertEqual(parked.status, PmInitiativeStatus.AWAITING_APPROVAL)

                action_value = json.dumps(
                    {
                        "v": 1,
                        "action": "pm_initiative.start",
                        "initiative_id": initiative.initiative_id,
                    }
                )
                controller.handle_block_action(
                    {
                        "actions": [{"action_id": "pm_initiative.start", "value": action_value}],
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.plan"},
                        "user": {"id": "U2"},
                    }
                )

                approved = store.get_pm_initiative(initiative.initiative_id)
                assert approved is not None
                self.assertEqual(approved.status, PmInitiativeStatus.ACTIVE)
                self.assertIsNone(approved.pending_plan_json)
                subtasks = store.list_pm_subtasks(initiative.initiative_id)
                self.assertEqual([s.local_id for s in subtasks], ["investigate", "implement"])
                root_deferred = store.get_deferred_work(subtasks[0].deferred_id)
                assert root_deferred is not None
                self.assertIn(
                    root_deferred.status,
                    {DeferredWorkStatus.DONE, DeferredWorkStatus.READY},
                )
                child_deferred = store.get_deferred_work(subtasks[1].deferred_id)
                assert child_deferred is not None
                self.assertEqual(child_deferred.status, DeferredWorkStatus.WAITING_DEPS)
                subtask_posts = [
                    item for item in gateway.posts if item["text"].startswith("PM subtask `")
                ]
                self.assertEqual(len(subtask_posts), 2)
                self.assertEqual(root_deferred.thread_ts, subtask_posts[0]["ts"])
                self.assertEqual(child_deferred.thread_ts, subtask_posts[1]["ts"])
                self.assertNotEqual(root_deferred.thread_ts, "171.thread")
                self.assertIsNotNone(root_deferred.last_task_id)
                self.assertIsNotNone(child_deferred.last_task_id)
                root_task = store.get_agent_task(root_deferred.last_task_id or "")
                child_task = store.get_agent_task(child_deferred.last_task_id or "")
                assert root_task is not None
                assert child_task is not None
                self.assertEqual(
                    root_task.metadata.get(PM_SUBTASK_LOCAL_ID_METADATA_KEY),
                    "investigate",
                )
                self.assertEqual(
                    child_task.metadata.get(PM_SUBTASK_LOCAL_ID_METADATA_KEY),
                    "implement",
                )
                prompt = build_task_prompt(agent, root_task)
                self.assertIn("single-purpose PM subtask thread", prompt)
                self.assertIn(AGENT_THREAD_DONE_SIGNAL, prompt)
                started_subtasks = [
                    item for item in runtime.started if item[0].task_id != pm_task.task_id
                ]
                self.assertEqual(len(started_subtasks), 1)
                self.assertEqual(started_subtasks[0][2].thread_ts, root_task.thread_ts)
                self.assertNotEqual(started_subtasks[0][2].thread_ts, "171.thread")
                pm_status_messages = [
                    item["text"]
                    for item in gateway.thread_replies
                    if item["thread"].thread_ts == "171.thread"
                    and item["text"].startswith("PM initiative status")
                ]
                self.assertTrue(pm_status_messages)
                self.assertIn("investigate", pm_status_messages[-1])
                self.assertIn(":large_blue_circle: active", pm_status_messages[-1])
                self.assertIn(":bookmark_tabs: reserved", pm_status_messages[-1])
                self.assertIn(f"owner=@{agent.handle}", pm_status_messages[-1])
                self.assertNotIn("owner=unassigned", pm_status_messages[-1])
                self.assertIn("<https://example.slack.com/archives/C1/p", pm_status_messages[-1])
                status_reply = next(
                    item
                    for item in gateway.thread_replies
                    if item["thread"].thread_ts == "171.thread"
                    and item["text"].startswith("PM initiative status")
                )
                self.assertFalse(status_reply["unfurl_links"])
                self.assertFalse(status_reply["unfurl_media"])
            finally:
                store.close()

    def test_pm_dependent_subtask_starts_in_precreated_thread_after_root_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Ship feature X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                signal = _build_plan_signal(initiative.initiative_id, handles=[agent.handle])
                controller.handle_runtime_agent_control(pm_task, agent, thread_ref, signal)
                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.start",
                                "value": json.dumps(
                                    {
                                        "v": 1,
                                        "action": "pm_initiative.start",
                                        "initiative_id": initiative.initiative_id,
                                    }
                                ),
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.plan"},
                        "user": {"id": "U2"},
                    }
                )
                subtasks = store.list_pm_subtasks(initiative.initiative_id)
                root_deferred = store.get_deferred_work(subtasks[0].deferred_id)
                child_deferred = store.get_deferred_work(subtasks[1].deferred_id)
                assert root_deferred is not None
                assert child_deferred is not None
                root_task = store.get_agent_task(root_deferred.last_task_id or "")
                child_task = store.get_agent_task(child_deferred.last_task_id or "")
                assert root_task is not None
                assert child_task is not None
                assert root_task.thread_ts is not None
                gateway.thread_history_messages[("C1", root_task.thread_ts)] = [
                    {
                        "username": "Avery",
                        "text": "Root result: preserve dependency thread context.",
                        "ts": "171.ctx",
                    }
                ]

                controller.handle_runtime_task_done(
                    root_task,
                    agent,
                    SlackThreadRef("C1", root_task.thread_ts or "", root_task.parent_message_ts),
                )
                child_started = store.get_deferred_work(child_deferred.deferred_id)
                assert child_started is not None
                self.assertEqual(child_started.status, DeferredWorkStatus.DONE)

                self.assertEqual(runtime.started[-1][0].task_id, child_task.task_id)
                self.assertEqual(runtime.started[-1][2].thread_ts, child_task.thread_ts)
                child_context = runtime.started[-1][0].metadata.get("thread_context")
                self.assertIsInstance(child_context, str)
                self.assertIn("PM dependency subtask `investigate`", child_context)
                self.assertIn("Root result: preserve dependency thread context.", child_context)
                self.assertNotEqual(runtime.started[-1][2].thread_ts, "171.thread")
                finished_root = store.get_agent_task(root_task.task_id)
                assert finished_root is not None
                self.assertEqual(finished_root.status, AgentTaskStatus.DONE)
                self.assertTrue(
                    any(
                        update["text"].startswith("PM initiative status")
                        and "done" in update["text"]
                        and "owner=@" in update["text"]
                        and ":white_check_mark:" in update["text"]
                        and update["attachments"] == []
                        for update in gateway.updates
                    )
                )
            finally:
                store.close()

    def test_pm_subtask_thread_done_signal_starts_dependent_subtask(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Ship feature X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                signal = _build_plan_signal(initiative.initiative_id, handles=[agent.handle])
                controller.handle_runtime_agent_control(pm_task, agent, thread_ref, signal)
                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.start",
                                "value": json.dumps(
                                    {
                                        "v": 1,
                                        "action": "pm_initiative.start",
                                        "initiative_id": initiative.initiative_id,
                                    }
                                ),
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.plan"},
                        "user": {"id": "U2"},
                    }
                )
                subtasks = store.list_pm_subtasks(initiative.initiative_id)
                root_deferred = store.get_deferred_work(subtasks[0].deferred_id)
                child_deferred = store.get_deferred_work(subtasks[1].deferred_id)
                assert root_deferred is not None
                assert child_deferred is not None
                root_task = store.get_agent_task(root_deferred.last_task_id or "")
                child_task = store.get_agent_task(child_deferred.last_task_id or "")
                assert root_task is not None
                assert child_task is not None

                handled = controller.handle_runtime_agent_control(
                    root_task,
                    agent,
                    SlackThreadRef("C1", root_task.thread_ts or "", root_task.parent_message_ts),
                    AGENT_THREAD_DONE_SIGNAL,
                )

                self.assertTrue(handled)
                finished_root = store.get_agent_task(root_task.task_id)
                assert finished_root is not None
                self.assertEqual(finished_root.status, AgentTaskStatus.DONE)
                child_started = store.get_deferred_work(child_deferred.deferred_id)
                assert child_started is not None
                self.assertEqual(child_started.status, DeferredWorkStatus.DONE)
                self.assertEqual(runtime.started[-1][0].task_id, child_task.task_id)
                self.assertEqual(runtime.started[-1][2].thread_ts, child_task.thread_ts)
                self.assertTrue(
                    any(
                        update["text"].startswith("PM initiative status")
                        and ":white_check_mark: done" in update["text"]
                        and ":large_blue_circle: active" in update["text"]
                        for update in gateway.updates
                    )
                )
            finally:
                store.close()

    def test_pm_cancelled_dependency_blocks_downstream_subtask(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Ship feature X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                signal = _build_plan_signal(initiative.initiative_id, handles=[agent.handle])
                controller.handle_runtime_agent_control(pm_task, agent, thread_ref, signal)
                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.start",
                                "value": json.dumps(
                                    {
                                        "v": 1,
                                        "action": "pm_initiative.start",
                                        "initiative_id": initiative.initiative_id,
                                    }
                                ),
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.plan"},
                        "user": {"id": "U2"},
                    }
                )
                subtasks = store.list_pm_subtasks(initiative.initiative_id)
                root_deferred = store.get_deferred_work(subtasks[0].deferred_id)
                child_deferred = store.get_deferred_work(subtasks[1].deferred_id)
                assert root_deferred is not None
                assert child_deferred is not None
                root_task = store.get_agent_task(root_deferred.last_task_id or "")
                child_task = store.get_agent_task(child_deferred.last_task_id or "")
                assert root_task is not None
                assert child_task is not None

                store.update_agent_task_status(root_task.task_id, AgentTaskStatus.CANCELLED)
                started_before = len(runtime.started)

                promoted = controller.evaluate_pending_deferred_work(child_deferred.deferred_id)

                self.assertEqual(promoted, 0)
                self.assertEqual(len(runtime.started), started_before)
                still_waiting = store.get_deferred_work(child_deferred.deferred_id)
                assert still_waiting is not None
                self.assertEqual(still_waiting.status, DeferredWorkStatus.WAITING_DEPS)
                blocker_text = "\n".join(reply["text"] for reply in gateway.thread_replies)
                self.assertIn("dependency `investigate` was cancelled", blocker_text)
                self.assertIn("will not start downstream work", blocker_text)
                reply_count = len(gateway.thread_replies)
                update_count = len(gateway.updates)

                promoted_again = controller.evaluate_pending_deferred_work(
                    child_deferred.deferred_id
                )

                self.assertEqual(promoted_again, 0)
                self.assertEqual(len(gateway.thread_replies), reply_count)
                self.assertEqual(len(gateway.updates), update_count)
            finally:
                store.close()

    def test_pm_watchdog_surfaces_cancelled_subtask_even_when_deferred_was_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                    requested_by_slack_user="U1",
                )
                store.update_pm_initiative_status(
                    initiative.initiative_id,
                    PmInitiativeStatus.ACTIVE,
                )
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                subtask = store.add_pm_subtask_dispatch(
                    initiative=refreshed,
                    local_id="integrate",
                    title="Integrate pieces",
                    request=WorkRequest(
                        prompt="Wire the pieces together.",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                task = replace(
                    create_agent_task(agent, "Wire the pieces together.", "C1"),
                    status=AgentTaskStatus.CANCELLED,
                    thread_ts="171.integrate",
                    parent_message_ts="171.integrate",
                    metadata={
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_SUBTASK_LOCAL_ID_METADATA_KEY: "integrate",
                    },
                )
                store.upsert_agent_task(task)
                store.complete_deferred_work(subtask.deferred_id, last_task_id=task.task_id)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=FakeRuntime(),
                )

                surfaced = controller.watch_pm_initiatives()

                self.assertEqual(surfaced, 1)
                updated_deferred = store.get_deferred_work(subtask.deferred_id)
                assert updated_deferred is not None
                self.assertEqual(updated_deferred.status, DeferredWorkStatus.CANCELLED)
                blocker_text = "\n".join(reply["text"] for reply in gateway.thread_replies)
                self.assertIn("subtask `integrate` was cancelled", blocker_text)
                self.assertIn("will not treat it as successful", blocker_text)
                updated_initiative = store.get_pm_initiative(initiative.initiative_id)
                assert updated_initiative is not None
                self.assertEqual(updated_initiative.status, PmInitiativeStatus.CANCELLED)
            finally:
                store.close()

    def test_pm_status_marks_exhausted_stall_recovery_as_stalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                    requested_by_slack_user="U1",
                )
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                subtask = store.add_pm_subtask_dispatch(
                    initiative=refreshed,
                    local_id="draft",
                    title="Draft design",
                    request=WorkRequest(
                        prompt="Draft the design.",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                task = replace(
                    create_agent_task(agent, "Draft the design.", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.subtask",
                    parent_message_ts="171.subtask",
                    metadata={
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_SUBTASK_LOCAL_ID_METADATA_KEY: "draft",
                        MANAGED_RUN_STARTED_METADATA_KEY: utc_now().isoformat(),
                        MANAGED_RUN_STALL_RECOVERIES_METADATA_KEY: (
                            MANAGED_RUN_MAX_STALL_RECOVERIES
                        ),
                    },
                )
                store.upsert_agent_task(task)
                store.update_deferred_work_last_task(subtask.deferred_id, last_task_id=task.task_id)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=FakeRuntime(),
                )

                view = controller._build_pm_initiative_plan_view(initiative.initiative_id)

                assert view is not None
                self.assertIn("draft: Draft design [:warning: stalled", view)
            finally:
                store.close()

    def test_pm_approval_prompts_to_hire_when_external_session_fills_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                workers = build_initial_model_team(3, 0)
                pm_agent = hire_team_agents(workers, 1, kind=TeamAgentKind.PM)[0]
                for agent in [*workers, pm_agent]:
                    store.upsert_team_agent(agent)
                busy_worker = workers[0]
                provider = busy_worker.provider_preference or Provider.CODEX
                store.upsert_session(
                    AgentSession(
                        provider=provider,
                        session_id="external-busy",
                        transcript_path=Path(tmp) / "external-busy.jsonl",
                        status=SessionStatus.ACTIVE,
                    )
                )
                store.set_setting(
                    f"external_session_agent.{provider.value}.external-busy",
                    busy_worker.agent_id,
                )
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Capacity constrained plan",
                    summary="Run three independent tasks.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(pm_agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Capacity constrained plan",
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                payload = {
                    "title": "Capacity constrained plan",
                    "summary": "Run three independent tasks.",
                    "subtasks": [
                        {
                            "id": f"root{index}",
                            "title": f"Root {index}",
                            "task": f"Run independent task {index}.",
                            "target": "somebody",
                            "depends_on": [],
                        }
                        for index in range(3)
                    ],
                }
                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    pm_agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}",
                )
                self.assertTrue(handled)
                plan_message = gateway.thread_replies[-1]

                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.start",
                                "value": json.dumps(
                                    {
                                        "v": 1,
                                        "action": "pm_initiative.start",
                                        "initiative_id": initiative.initiative_id,
                                    }
                                ),
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": plan_message["ts"]},
                        "user": {"id": "U2"},
                    }
                )

                still_parked = store.get_pm_initiative(initiative.initiative_id)
                assert still_parked is not None
                self.assertEqual(still_parked.status, PmInitiativeStatus.AWAITING_APPROVAL)
                self.assertIsNotNone(still_parked.pending_plan_json)
                self.assertEqual(store.list_pm_subtasks(initiative.initiative_id), [])
                self.assertEqual(runtime.started, [])
                capacity_message = gateway.thread_replies[-1]["text"]
                self.assertIn("not enough free worker agents", capacity_message)
                self.assertIn("needs 3 available worker agents", capacity_message)
                self.assertIn("only 2 worker agents are free", capacity_message)
                self.assertIn("Hire and reserve 1 agent", capacity_message)
                self.assertNotIn("team hire", capacity_message)
                capacity_blocks = gateway.thread_replies[-1].get("blocks")
                self.assertIsNotNone(capacity_blocks)
                action_ids = [
                    element["action_id"]
                    for block in capacity_blocks or []
                    if block.get("type") == "actions"
                    for element in block.get("elements", [])
                ]
                self.assertIn("pm_initiative.hire_and_start", action_ids)
            finally:
                store.close()

    def test_pm_specific_worker_shortfall_keeps_actionable_retry_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                workers = build_initial_model_team(2, 0)
                pm_agent = hire_team_agents(workers, 1, kind=TeamAgentKind.PM)[0]
                for agent in [*workers, pm_agent]:
                    store.upsert_team_agent(agent)
                busy_worker = workers[0]
                busy_task = replace(
                    create_agent_task(busy_worker, "busy elsewhere", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.busy",
                    parent_message_ts="171.busy",
                )
                store.upsert_agent_task(busy_task)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Specific worker plan",
                    summary="Run one pinned task.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(pm_agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Specific worker plan",
                        "request_message_ts": "171.thread",
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                payload = {
                    "title": "Specific worker plan",
                    "summary": "Run one pinned task.",
                    "subtasks": [
                        {
                            "id": "pinned",
                            "title": "Pinned work",
                            "task": "Run the pinned work.",
                            "target": busy_worker.handle,
                            "depends_on": [],
                        }
                    ],
                }
                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    pm_agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}",
                )
                self.assertTrue(handled)
                plan_message = gateway.thread_replies[-1]
                request_reactions = {
                    item["name"] for item in gateway._current_reactions("C1", "171.thread")
                }
                self.assertIn(TASK_REACTION_IN_PROGRESS, request_reactions)
                self.assertNotIn(TASK_REACTION_DONE, request_reactions)

                start_value = next(
                    element["value"]
                    for block in plan_message.get("blocks") or []
                    if block.get("type") == "actions"
                    for element in block.get("elements", [])
                    if element["action_id"] == "pm_initiative.start"
                )
                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.start",
                                "value": start_value,
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": plan_message["ts"]},
                        "user": {"id": "U2"},
                    }
                )

                still_parked = store.get_pm_initiative(initiative.initiative_id)
                assert still_parked is not None
                self.assertEqual(still_parked.status, PmInitiativeStatus.AWAITING_APPROVAL)
                self.assertEqual(store.list_pm_subtasks(initiative.initiative_id), [])
                capacity_message = gateway.thread_replies[-1]
                self.assertIn("reserves specific worker agents", capacity_message["text"])
                self.assertIn("The plan is still parked", capacity_message["text"])
                self.assertIn("Try start again", capacity_message["text"])
                retry_blocks = capacity_message.get("blocks")
                self.assertIsNotNone(retry_blocks)
                retry_actions = [
                    element
                    for block in retry_blocks or []
                    if block.get("type") == "actions"
                    for element in block.get("elements", [])
                ]
                self.assertEqual(
                    [element["action_id"] for element in retry_actions],
                    ["pm_initiative.start", "pm_initiative.cancel"],
                )

                store.update_agent_task_status(busy_task.task_id, AgentTaskStatus.DONE)
                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.start",
                                "value": retry_actions[0]["value"],
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": capacity_message["ts"]},
                        "user": {"id": "U2"},
                    }
                )

                promoted = store.get_pm_initiative(initiative.initiative_id)
                assert promoted is not None
                self.assertEqual(promoted.status, PmInitiativeStatus.ACTIVE)
                self.assertEqual(len(store.list_pm_subtasks(initiative.initiative_id)), 1)
                stripped_by_ts = {update["ts"]: update for update in gateway.updates}
                self.assertIn(plan_message["ts"], stripped_by_ts)
                self.assertIn(capacity_message["ts"], stripped_by_ts)
                for update in stripped_by_ts.values():
                    if update["ts"] not in {plan_message["ts"], capacity_message["ts"]}:
                        continue
                    self.assertFalse(
                        any(block.get("type") == "actions" for block in update["blocks"])
                    )
            finally:
                store.close()

    def test_pm_capacity_button_hires_and_reserves_before_unrelated_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                workers = build_initial_model_team(1, 0)
                pm_agent = hire_team_agents(workers, 1, kind=TeamAgentKind.PM)[0]
                for agent in [*workers, pm_agent]:
                    store.upsert_team_agent(agent)

                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Capacity constrained plan",
                    summary="Run three independent tasks.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(pm_agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Capacity constrained plan",
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                payload = {
                    "title": "Capacity constrained plan",
                    "summary": "Run three independent tasks.",
                    "subtasks": [
                        {
                            "id": f"root{index}",
                            "title": f"Root {index}",
                            "task": f"Run independent task {index}.",
                            "target": "somebody",
                            "depends_on": [],
                        }
                        for index in range(3)
                    ],
                }
                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    pm_agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}",
                )
                self.assertTrue(handled)
                plan_message = gateway.thread_replies[-1]

                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.start",
                                "value": json.dumps(
                                    {
                                        "v": 1,
                                        "action": "pm_initiative.start",
                                        "initiative_id": initiative.initiative_id,
                                    }
                                ),
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": plan_message["ts"]},
                        "user": {"id": "U2"},
                    }
                )
                capacity_message = gateway.thread_replies[-1]
                capacity_blocks = capacity_message.get("blocks")
                self.assertIsNotNone(capacity_blocks)

                store.upsert_session(
                    AgentSession(
                        provider=Provider.CODEX,
                        session_id="outside",
                        transcript_path=Path(tmp) / "outside.jsonl",
                        status=SessionStatus.IDLE,
                    )
                )
                store.set_setting("external_session_pending.codex.outside", "now")
                store.create_pending_work_request(
                    SlackThreadRef("C1", "171.unrelated", "171.unrelated"),
                    WorkRequest(
                        prompt="unrelated pending work",
                        assignment_mode=AssignmentMode.ANYONE,
                    ),
                )
                unrelated_deferred = store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.deferred", "171.deferred"),
                    WorkRequest(
                        prompt="unrelated deferred work",
                        assignment_mode=AssignmentMode.ANYONE,
                    ),
                    depends_on=(),
                )
                due_at = utc_now() - timedelta(seconds=5)
                store.mark_deferred_work_ready(
                    unrelated_deferred.deferred_id,
                    fire_at=due_at,
                    deps_satisfied_at=due_at,
                )

                action_value = next(
                    element["value"]
                    for block in capacity_blocks or []
                    if block.get("type") == "actions"
                    for element in block.get("elements", [])
                    if element["action_id"] == "pm_initiative.hire_and_start"
                )
                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.hire_and_start",
                                "value": action_value,
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": capacity_message["ts"]},
                        "user": {"id": "U2"},
                    }
                )

                agents = store.list_team_agents()
                workers_after_hire = [agent for agent in agents if not agent.is_pm]
                self.assertEqual(len(agents), 4)
                self.assertEqual(len(workers_after_hire), 3)
                promoted = store.get_pm_initiative(initiative.initiative_id)
                assert promoted is not None
                self.assertEqual(promoted.status, PmInitiativeStatus.ACTIVE)
                self.assertIsNone(promoted.pending_plan_json)

                subtasks = store.list_pm_subtasks(initiative.initiative_id)
                self.assertEqual(len(subtasks), 3)
                reserved_tasks = []
                for subtask in subtasks:
                    deferred = store.get_deferred_work(subtask.deferred_id)
                    assert deferred is not None
                    self.assertIsNotNone(deferred.last_task_id)
                    reserved = store.get_agent_task(str(deferred.last_task_id))
                    assert reserved is not None
                    reserved_tasks.append(reserved)
                self.assertEqual(len({task.agent_id for task in reserved_tasks}), 3)
                self.assertEqual(
                    {
                        task.metadata.get(PM_SUBTASK_LOCAL_ID_METADATA_KEY)
                        for task, _, _ in runtime.started
                    },
                    {"root0", "root1", "root2"},
                )
                self.assertNotIn(
                    "unrelated pending work",
                    {task.prompt for task, _, _ in runtime.started},
                )
                self.assertEqual(len(store.list_pending_work_requests()), 1)
                self.assertIsNone(store.get_setting("external_session_agent.codex.outside"))
                self.assertEqual(
                    store.get_setting("external_session_pending.codex.outside"),
                    "now",
                )
                latest_unrelated = store.get_deferred_work(unrelated_deferred.deferred_id)
                assert latest_unrelated is not None
                self.assertEqual(latest_unrelated.status, DeferredWorkStatus.READY)
            finally:
                store.close()

    def test_team_hire_in_pm_thread_promotes_plan_before_unrelated_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                workers = build_initial_model_team(1, 0)
                pm_agent = hire_team_agents(workers, 1, kind=TeamAgentKind.PM)[0]
                for agent in [*workers, pm_agent]:
                    store.upsert_team_agent(agent)

                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Capacity constrained plan",
                    summary="Run three independent tasks.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(pm_agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Capacity constrained plan",
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                payload = {
                    "title": "Capacity constrained plan",
                    "summary": "Run three independent tasks.",
                    "subtasks": [
                        {
                            "id": f"root{index}",
                            "title": f"Root {index}",
                            "task": f"Run independent task {index}.",
                            "target": "somebody",
                            "depends_on": [],
                        }
                        for index in range(3)
                    ],
                }
                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    pm_agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}",
                )
                self.assertTrue(handled)
                plan_message = gateway.thread_replies[-1]

                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.start",
                                "value": json.dumps(
                                    {
                                        "v": 1,
                                        "action": "pm_initiative.start",
                                        "initiative_id": initiative.initiative_id,
                                    }
                                ),
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": plan_message["ts"]},
                        "user": {"id": "U2"},
                    }
                )
                still_parked = store.get_pm_initiative(initiative.initiative_id)
                assert still_parked is not None
                self.assertEqual(still_parked.status, PmInitiativeStatus.AWAITING_APPROVAL)
                self.assertIsNotNone(still_parked.pending_plan_json)
                self.assertEqual(runtime.started, [])

                store.upsert_session(
                    AgentSession(
                        provider=Provider.CODEX,
                        session_id="outside",
                        transcript_path=Path(tmp) / "outside.jsonl",
                        status=SessionStatus.IDLE,
                    )
                )
                store.set_setting("external_session_pending.codex.outside", "now")
                store.create_pending_work_request(
                    SlackThreadRef("C1", "171.unrelated", "171.unrelated"),
                    WorkRequest(
                        prompt="unrelated pending work",
                        assignment_mode=AssignmentMode.ANYONE,
                    ),
                )
                unrelated_deferred = store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.deferred", "171.deferred"),
                    WorkRequest(
                        prompt="unrelated deferred work",
                        assignment_mode=AssignmentMode.ANYONE,
                    ),
                    depends_on=(),
                )
                due_at = utc_now() - timedelta(seconds=5)
                store.mark_deferred_work_ready(
                    unrelated_deferred.deferred_id,
                    fire_at=due_at,
                    deps_satisfied_at=due_at,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U2",
                            "text": "team hire 2",
                            "thread_ts": "171.thread",
                            "ts": "171.hire",
                        }
                    }
                )

                agents = store.list_team_agents()
                workers_after_hire = [agent for agent in agents if not agent.is_pm]
                self.assertEqual(len(agents), 4)
                self.assertEqual(len(workers_after_hire), 3)
                promoted = store.get_pm_initiative(initiative.initiative_id)
                assert promoted is not None
                self.assertEqual(promoted.status, PmInitiativeStatus.ACTIVE)
                self.assertIsNone(promoted.pending_plan_json)

                subtasks = store.list_pm_subtasks(initiative.initiative_id)
                self.assertEqual(len(subtasks), 3)
                reserved_tasks = []
                for subtask in subtasks:
                    deferred = store.get_deferred_work(subtask.deferred_id)
                    assert deferred is not None
                    self.assertIsNotNone(deferred.last_task_id)
                    reserved = store.get_agent_task(str(deferred.last_task_id))
                    assert reserved is not None
                    self.assertEqual(
                        reserved.metadata.get(PM_SUBTASK_LOCAL_ID_METADATA_KEY),
                        subtask.local_id,
                    )
                    reserved_tasks.append(reserved)
                self.assertEqual(len({task.agent_id for task in reserved_tasks}), 3)
                self.assertEqual(
                    {
                        task.metadata.get(PM_SUBTASK_LOCAL_ID_METADATA_KEY)
                        for task, _, _ in runtime.started
                    },
                    {"root0", "root1", "root2"},
                )
                self.assertNotIn(
                    "unrelated pending work",
                    {task.prompt for task, _, _ in runtime.started},
                )
                self.assertEqual(len(store.list_pending_work_requests()), 1)
                self.assertIsNone(store.get_setting("external_session_agent.codex.outside"))
                self.assertEqual(
                    store.get_setting("external_session_pending.codex.outside"),
                    "now",
                )
                latest_unrelated = store.get_deferred_work(unrelated_deferred.deferred_id)
                assert latest_unrelated is not None
                self.assertEqual(latest_unrelated.status, DeferredWorkStatus.READY)
            finally:
                store.close()

    def test_pm_approval_prompts_when_codesign_model_agent_is_not_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                workers = build_initial_model_team(1, 1)
                codex_worker = next(
                    agent for agent in workers if agent.provider_preference == Provider.CODEX
                )
                claude_worker = next(
                    agent for agent in workers if agent.provider_preference == Provider.CLAUDE
                )
                pm_agent = hire_team_agents(workers, 1, kind=TeamAgentKind.PM)[0]
                for agent in [*workers, pm_agent]:
                    store.upsert_team_agent(agent)
                busy_task = replace(
                    create_agent_task(codex_worker, "external project", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.busy",
                    parent_message_ts="171.busy",
                )
                store.upsert_agent_task(busy_task)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Capacity constrained co-design",
                    summary="Run co-design.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(pm_agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Capacity constrained co-design",
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                payload = {
                    "title": "Capacity constrained co-design",
                    "summary": "Run co-design.",
                    "subtasks": [
                        {
                            "id": "design",
                            "title": "Design",
                            "task": "Design the API.",
                            "co_designers": [codex_worker.handle, claude_worker.handle],
                            "depends_on": [],
                        }
                    ],
                }
                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    pm_agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}",
                )
                self.assertTrue(handled)
                plan_message = gateway.thread_replies[-1]

                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.start",
                                "value": json.dumps(
                                    {
                                        "v": 1,
                                        "action": "pm_initiative.start",
                                        "initiative_id": initiative.initiative_id,
                                    }
                                ),
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": plan_message["ts"]},
                        "user": {"id": "U2"},
                    }
                )

                still_parked = store.get_pm_initiative(initiative.initiative_id)
                assert still_parked is not None
                self.assertEqual(still_parked.status, PmInitiativeStatus.AWAITING_APPROVAL)
                self.assertEqual(store.list_pm_subtasks(initiative.initiative_id), [])
                self.assertEqual(runtime.started, [])
                capacity_message = gateway.thread_replies[-1]["text"]
                self.assertIn("reserves specific worker agents", capacity_message)
                self.assertIn(f"@{codex_worker.handle}", capacity_message)
                self.assertIn("not free right now", capacity_message)
            finally:
                store.close()

    def test_pm_approval_prompts_to_hire_when_active_tasks_fill_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            runtime = FakeRuntime()
            try:
                store.init_schema()
                workers = build_initial_model_team(3, 0)
                pm_agent = hire_team_agents(workers, 1, kind=TeamAgentKind.PM)[0]
                for agent in [*workers, pm_agent]:
                    store.upsert_team_agent(agent)
                for index, worker in enumerate(workers):
                    busy_task = replace(
                        create_agent_task(worker, f"busy task {index}", "C1"),
                        status=AgentTaskStatus.ACTIVE,
                        thread_ts=f"171.busy{index}",
                        parent_message_ts=f"171.busy{index}",
                    )
                    store.upsert_agent_task(busy_task)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Capacity constrained plan",
                    summary="Run three independent tasks.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(pm_agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Capacity constrained plan",
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                payload = {
                    "title": "Capacity constrained plan",
                    "summary": "Run three independent tasks.",
                    "subtasks": [
                        {
                            "id": f"root{index}",
                            "title": f"Root {index}",
                            "task": f"Run independent task {index}.",
                            "target": "somebody",
                            "depends_on": [],
                        }
                        for index in range(3)
                    ],
                }
                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    pm_agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}",
                )
                self.assertTrue(handled)
                plan_message = gateway.thread_replies[-1]

                controller.handle_block_action(
                    {
                        "actions": [
                            {
                                "action_id": "pm_initiative.start",
                                "value": json.dumps(
                                    {
                                        "v": 1,
                                        "action": "pm_initiative.start",
                                        "initiative_id": initiative.initiative_id,
                                    }
                                ),
                            }
                        ],
                        "channel": {"id": "C1"},
                        "message": {"ts": plan_message["ts"]},
                        "user": {"id": "U2"},
                    }
                )

                still_parked = store.get_pm_initiative(initiative.initiative_id)
                assert still_parked is not None
                self.assertEqual(still_parked.status, PmInitiativeStatus.AWAITING_APPROVAL)
                self.assertIsNotNone(still_parked.pending_plan_json)
                self.assertEqual(store.list_pm_subtasks(initiative.initiative_id), [])
                self.assertEqual(runtime.started, [])
                capacity_message = gateway.thread_replies[-1]["text"]
                self.assertIn("not enough free worker agents", capacity_message)
                self.assertIn("needs 3 available worker agents", capacity_message)
                self.assertIn("only 0 worker agents are free", capacity_message)
                self.assertIn("Hire and reserve 3 agents", capacity_message)
                self.assertNotIn("team hire", capacity_message)
                capacity_blocks = gateway.thread_replies[-1].get("blocks")
                self.assertIsNotNone(capacity_blocks)
                action_ids = [
                    element["action_id"]
                    for block in capacity_blocks or []
                    if block.get("type") == "actions"
                    for element in block.get("elements", [])
                ]
                self.assertIn("pm_initiative.hire_and_start", action_ids)
            finally:
                store.close()

    def test_pm_initiative_cancel_button_cancels_without_firing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Ship feature X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )
                signal = _build_plan_signal(
                    initiative.initiative_id,
                    handles=[agent.handle],
                )
                controller.handle_runtime_agent_control(pm_task, agent, thread_ref, signal)

                action_value = json.dumps(
                    {
                        "v": 1,
                        "action": "pm_initiative.cancel",
                        "initiative_id": initiative.initiative_id,
                    }
                )
                controller.handle_block_action(
                    {
                        "actions": [{"action_id": "pm_initiative.cancel", "value": action_value}],
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.plan"},
                        "user": {"id": "U2"},
                    }
                )

                cancelled = store.get_pm_initiative(initiative.initiative_id)
                assert cancelled is not None
                self.assertEqual(cancelled.status, PmInitiativeStatus.CANCELLED)
                self.assertEqual(store.list_pm_subtasks(initiative.initiative_id), [])
                self.assertEqual(
                    [t for t, *_ in runtime.started if t.task_id != pm_task.task_id], []
                )
            finally:
                store.close()

    def test_invalid_pm_plan_retries_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Ship feature X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{{not json",
                )

                self.assertTrue(handled)
                self.assertEqual(len(runtime.started), 1)
                retry_task = runtime.started[0][0]
                self.assertIn("previous PM_PLAN control line was invalid", retry_task.prompt)
                self.assertEqual(retry_task.metadata[PM_RESOLUTION_ATTEMPTS_METADATA_KEY], 1)
            finally:
                store.close()

    def test_invalid_pm_plan_at_retry_cap_cancels_initiative(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(thread_ref, title="T", summary="S")
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 3,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{{not json",
                )

                self.assertTrue(handled)
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, PmInitiativeStatus.CANCELLED)
                resolver_task = store.get_agent_task(pm_task.task_id)
                assert resolver_task is not None
                self.assertEqual(resolver_task.status, AgentTaskStatus.CANCELLED)
                self.assertEqual(runtime.started, [])
            finally:
                store.close()

    def test_pm_watchdog_marks_initiative_done_when_subtasks_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref, title="T", summary="S", requested_by_slack_user="U1"
                )
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed_initiative = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed_initiative is not None
                subtask = store.add_pm_subtask_dispatch(
                    initiative=refreshed_initiative,
                    local_id="s1",
                    title="Solo",
                    request=WorkRequest(prompt="solo task", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                store.update_deferred_work_status(subtask.deferred_id, DeferredWorkStatus.DONE)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                )

                runner = PMInitiativeRunner(store, controller, poll_seconds=0.01)
                surfaced = runner.sync_once()

                self.assertGreaterEqual(surfaced, 1)
                final = store.get_pm_initiative(initiative.initiative_id)
                assert final is not None
                self.assertEqual(final.status, PmInitiativeStatus.DONE)
                # A recap message was posted in the initiative thread.
                self.assertTrue(any("complete" in r["text"] for r in gateway.thread_replies))
                self.assertTrue(
                    any(
                        "PM initiative status" in item["text"]
                        and "status: :white_check_mark: done" in item["text"]
                        for item in gateway.thread_replies
                    )
                )
            finally:
                store.close()

    def test_pm_watchdog_fires_ready_subtasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref, title="T", summary="S", requested_by_slack_user="U1"
                )
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed_initiative = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed_initiative is not None
                subtask = store.add_pm_subtask_dispatch(
                    initiative=refreshed_initiative,
                    local_id="s1",
                    title="Ready",
                    request=WorkRequest(prompt="ready task", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                runner = PMInitiativeRunner(store, controller, poll_seconds=0.01)
                runner.sync_once()

                self.assertEqual(len(runtime.started), 1)
                started_task, _started_agent, started_thread = runtime.started[0]
                self.assertEqual(started_thread.thread_ts, "171.thread")
                self.assertEqual(started_task.prompt, "ready task")
                fired = store.get_deferred_work(subtask.deferred_id)
                assert fired is not None
                self.assertEqual(fired.status, DeferredWorkStatus.DONE)
            finally:
                store.close()

    def test_pm_watchdog_surfaces_stalled_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="T",
                    summary="S",
                    requested_by_slack_user="U1",
                )
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed_initiative = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed_initiative is not None
                subtask = store.add_pm_subtask_dispatch(
                    initiative=refreshed_initiative,
                    local_id="s1",
                    title="Block on approval",
                    request=WorkRequest(
                        prompt="touch CI",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                # Simulate the deferred row firing into a task in an ACTIVE
                # state by hand.
                subtask_task = replace(
                    create_agent_task(agent, "touch CI", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.subtask",
                    parent_message_ts="171.subtask-parent",
                )
                store.upsert_agent_task(subtask_task)
                store.update_deferred_work_last_task(
                    subtask.deferred_id, last_task_id=subtask_task.task_id
                )
                # An old, unresolved Slack approval request blocks the subtask.
                store.create_slack_agent_request(
                    token="tok-1",
                    provider_label="Codex",
                    method="codex/approve",
                    params={},
                    thread=SlackThreadRef("C1", "171.subtask", "171.approve"),
                    message_ts="171.approve",
                )
                old_created = (utc_now() - timedelta(minutes=10)).isoformat()
                store.conn.execute(
                    "UPDATE slack_agent_requests SET created_at = ? WHERE token = ?",
                    (old_created, "tok-1"),
                )
                store.conn.commit()
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                runner = PMInitiativeRunner(store, controller, poll_seconds=0.01)
                runner.sync_once()

                surfaced_texts = [reply["text"] for reply in gateway.thread_replies]
                self.assertTrue(
                    any("waiting on `codex/approve`" in text for text in surfaced_texts),
                    surfaced_texts,
                )
                # Second tick should NOT post the same blocker twice.
                first_count = len(gateway.thread_replies)
                runner.sync_once()
                self.assertEqual(len(gateway.thread_replies), first_count)
            finally:
                store.close()


class _FakeSlackApiError(Exception):
    """Stand-in for slack_sdk.errors.SlackApiError used by the dead-thread tests."""

    def __init__(self, error_code: str):
        super().__init__(error_code)
        self.response = {"error": error_code}


class PmWatchdogAdversarialTests(unittest.TestCase):
    """Adversarial coverage: lost threads, fired PMs, watchdog crashes."""

    def _setup_active_initiative(self, store: Store, *, agent_handle: str):
        team = build_initial_model_team(1, 0)
        agent = team[0]
        # Force the handle so the test reads cleanly.
        agent = replace(agent, handle=agent_handle)
        store.upsert_team_agent(agent)
        thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
        initiative = store.create_pm_initiative(
            thread_ref, title="T", summary="S", requested_by_slack_user="U1"
        )
        store.update_pm_initiative_status(initiative.initiative_id, PmInitiativeStatus.ACTIVE)
        refreshed = store.get_pm_initiative(initiative.initiative_id)
        assert refreshed is not None
        return refreshed, agent

    def test_watchdog_cancels_initiative_when_blocker_thread_is_unreachable(self):
        """When the watchdog tries to surface a blocker against a deleted /
        archived thread, it should mark the initiative CANCELLED so subsequent
        ticks do not loop forever against the same dead thread."""

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")

            class DeadThreadGateway(FakeGateway):
                def post_thread_reply(self, *args, **kwargs):
                    raise _FakeSlackApiError("channel_not_found")

            gateway = DeadThreadGateway()
            try:
                store.init_schema()
                initiative, agent = self._setup_active_initiative(store, agent_handle="alice")
                subtask = store.add_pm_subtask_dispatch(
                    initiative=initiative,
                    local_id="s1",
                    title="Stalled",
                    request=WorkRequest(
                        prompt="touch CI",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                # Drive the subtask into an ACTIVE task with a stale updated_at
                # so the watchdog tries to surface a `stalled_task` blocker.
                stalled_task = replace(
                    create_agent_task(agent, "touch CI", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.sub",
                    parent_message_ts="171.sub-parent",
                    updated_at=utc_now() - timedelta(minutes=45),
                )
                store.upsert_agent_task(stalled_task)
                store.update_deferred_work_last_task(
                    subtask.deferred_id, last_task_id=stalled_task.task_id
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                runner = PMInitiativeRunner(store, controller, poll_seconds=0.01)
                with self.assertLogs("agent_harness.slack.app", level="WARNING") as logs:
                    runner.sync_once()
                final = store.get_pm_initiative(initiative.initiative_id)
                assert final is not None
                self.assertEqual(final.status, PmInitiativeStatus.CANCELLED)
                self.assertIn(
                    "thread is unreachable (channel_not_found); cancelling",
                    "\n".join(logs.output),
                )
            finally:
                store.close()

    def test_watchdog_heartbeat_not_updated_on_exception(self):
        """``mark_pm_initiative_watchdog_run`` must skip the heartbeat update
        when ``_watch_one_pm_initiative`` raises, so a monitor that watches the
        timestamp can still notice the failure."""

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                initiative, _ = self._setup_active_initiative(store, agent_handle="alice")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                def boom(_initiative):
                    raise RuntimeError("simulated watchdog failure")

                controller._watch_one_pm_initiative = boom  # type: ignore[assignment]
                runner = PMInitiativeRunner(store, controller, poll_seconds=0.01)
                runner.sync_once()
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                self.assertIsNone(refreshed.watchdog_last_run_at)
            finally:
                store.close()

    def test_pm_agent_fired_during_active_keeps_initiative_visible(self):
        """If the user fires the PM mid-initiative, the watchdog should not
        crash. The initiative stays ACTIVE until subtasks reach terminal state,
        and the watchdog still posts a recap."""

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                initiative, agent = self._setup_active_initiative(store, agent_handle="alice")
                subtask = store.add_pm_subtask_dispatch(
                    initiative=initiative,
                    local_id="s1",
                    title="Solo",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                store.update_deferred_work_status(subtask.deferred_id, DeferredWorkStatus.DONE)
                # Fire the PM mid-run.
                store.fire_team_agent(agent.agent_id)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                runner = PMInitiativeRunner(store, controller, poll_seconds=0.01)
                runner.sync_once()
                final = store.get_pm_initiative(initiative.initiative_id)
                assert final is not None
                # All terminal, so the watchdog should close the initiative as DONE.
                self.assertEqual(final.status, PmInitiativeStatus.DONE)
            finally:
                store.close()


class PmRetryFailureReasonTests(unittest.TestCase):
    def test_retry_start_task_failure_surfaces_validator_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(thread_ref, title="T", summary="S")
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "ship X",
                    },
                )
                store.upsert_agent_task(pm_task)

                class FailingRuntime(FakeRuntime):
                    def start_task(self, task, agent, thread):
                        self.started.append((task, agent, thread))
                        return False

                runtime = FailingRuntime()
                controller = SlackTeamController(
                    store, gateway, default_channel_id="C1", runtime=runtime
                )
                # Trigger the retry path with an unparseable PM_PLAN.
                controller.handle_runtime_agent_control(
                    pm_task,
                    agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{{not json",
                )
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, PmInitiativeStatus.CANCELLED)
                cancel_message = gateway.thread_replies[-1]["text"]
                # The user-visible message must include the underlying
                # validator error so they know why the retry failed.
                self.assertIn("last validation error", cancel_message)
                self.assertIn("invalid PM_PLAN JSON", cancel_message)
            finally:
                store.close()


class PmStatusCommandTests(unittest.TestCase):
    def test_pm_status_command_renders_plan_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref, title="Ship feature X", summary="S"
                )
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                store.add_pm_subtask_dispatch(
                    initiative=refreshed,
                    local_id="investigate",
                    title="Investigate",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=FakeRuntime(),
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "pm status",
                            "ts": "171.000010",
                            "thread_ts": "171.thread",
                        }
                    }
                )
                self.assertTrue(gateway.thread_replies, "expected a thread reply")
                latest = gateway.thread_replies[-1]["text"]
                self.assertIn("investigate", latest)
                self.assertIn("Ship feature X", latest)
            finally:
                store.close()


class PmReplanRuntimeTests(unittest.TestCase):
    def test_pm_replan_cancels_pending_and_restarts_resolver(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                from agent_harness.models import TeamAgentKind

                team = build_initial_model_team(1, 0)
                agent = replace(team[0], kind=TeamAgentKind.PM)
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(thread_ref, title="T", summary="ship X")
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                # Two subtasks: one DONE, one still pending.
                done_subtask = store.add_pm_subtask_dispatch(
                    initiative=refreshed,
                    local_id="done1",
                    title="Done step",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                store.update_deferred_work_status(done_subtask.deferred_id, DeferredWorkStatus.DONE)
                pending = store.add_pm_subtask_dispatch(
                    initiative=refreshed,
                    local_id="pending1",
                    title="Pending step",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[done_subtask],
                    after_delay_seconds=0,
                    sort_order=1,
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "ship X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store, gateway, default_channel_id="C1", runtime=runtime
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "pm replan: take a different approach",
                            "ts": "171.000020",
                            "thread_ts": "171.thread",
                        }
                    }
                )
                pending_after = store.get_deferred_work(pending.deferred_id)
                assert pending_after is not None
                self.assertEqual(pending_after.status, DeferredWorkStatus.CANCELLED)
                # Done one stays DONE.
                done_after = store.get_deferred_work(done_subtask.deferred_id)
                assert done_after is not None
                self.assertEqual(done_after.status, DeferredWorkStatus.DONE)
                # Initiative is back in PLANNING.
                planning = store.get_pm_initiative(initiative.initiative_id)
                assert planning is not None
                self.assertEqual(planning.status, PmInitiativeStatus.PLANNING)
                # Resolver was restarted with the new context.
                self.assertEqual(len(runtime.started), 1)
                started_task, _, _ = runtime.started[0]
                self.assertIn("REPLAN", started_task.prompt)
                self.assertIn("take a different approach", started_task.prompt)
            finally:
                store.close()

    def test_pm_replan_clears_invalid_pm_external_session_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                team = build_initial_model_team(1, 0)
                agent = replace(team[0], kind=TeamAgentKind.PM)
                store.upsert_team_agent(agent)
                provider = agent.provider_preference or Provider.CODEX
                store.upsert_session(
                    AgentSession(
                        provider=provider,
                        session_id="busy-pm",
                        transcript_path=Path(tmp) / "busy-pm.jsonl",
                        status=SessionStatus.ACTIVE,
                    )
                )
                store.set_setting(
                    f"external_session_agent.{provider.value}.busy-pm",
                    agent.agent_id,
                )
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(thread_ref, title="T", summary="ship X")
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.DONE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "ship X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store, gateway, default_channel_id="C1", runtime=runtime
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "pm replan: take a different approach",
                            "ts": "171.000020",
                            "thread_ts": "171.thread",
                        }
                    }
                )

                self.assertEqual(len(runtime.started), 1)
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, PmInitiativeStatus.PLANNING)
                self.assertIsNone(
                    store.get_setting(f"external_session_agent.{provider.value}.busy-pm")
                )
                self.assertIsNotNone(
                    store.get_setting(f"external_session_ignored.{provider.value}.busy-pm")
                )
                self.assertIn("Replanning this initiative", gateway.thread_replies[-1]["text"])
            finally:
                store.close()


class PmExtensionRuntimeTests(unittest.TestCase):
    def test_pm_extension_command_seeds_known_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                from agent_harness.models import TeamAgentKind

                team = build_initial_model_team(1, 0)
                agent = replace(team[0], kind=TeamAgentKind.PM)
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(thread_ref, title="T", summary="ship X")
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                store.add_pm_subtask_dispatch(
                    initiative=refreshed,
                    local_id="existing1",
                    title="Existing",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "ship X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store, gateway, default_channel_id="C1", runtime=runtime
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "pm extend: also write integration tests",
                            "ts": "171.000030",
                            "thread_ts": "171.thread",
                        }
                    }
                )
                # Resolver restarted, prompt mentions EXTENSION mode and existing id.
                self.assertEqual(len(runtime.started), 1)
                started_task, _, _ = runtime.started[0]
                self.assertIn("EXTENSION", started_task.prompt)
                self.assertIn("existing1", started_task.prompt)
                # Metadata carries the known ids for the next plan-parse round.
                self.assertEqual(
                    list(started_task.metadata.get(PM_EXTENSION_KNOWN_IDS_METADATA_KEY) or []),
                    ["existing1"],
                )
                # Existing subtask stays untouched.
                existing_after = store.list_pm_subtasks(initiative.initiative_id)
                self.assertEqual([s.local_id for s in existing_after], ["existing1"])
            finally:
                store.close()


class PmPlanAckEstimateTests(unittest.TestCase):
    def test_estimate_surfaces_in_plan_approval_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(thread_ref, title="T", summary="S")
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "ship X",
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store, gateway, default_channel_id="C1", runtime=FakeRuntime()
                )
                signal = _build_plan_signal(initiative.initiative_id, handles=[agent.handle])
                controller.handle_runtime_agent_control(pm_task, agent, thread_ref, signal)
                plan_message = gateway.thread_replies[-1]["text"]
                self.assertIn("Estimate:", plan_message)
                self.assertIn("subtasks", plan_message)
                self.assertIn("critical path", plan_message)
            finally:
                store.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
