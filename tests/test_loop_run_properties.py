"""Property tests for loop runs: summaries, carry, THREAD_DONE, cards, and memory.

These are heavy, so CI skips them; every bug they find gets a fast regression test
in test_loop_app.py or test_loops.py instead. Run them locally with

    SLACKGENTIC_PROPERTY_TESTS=1 PYTHONPATH=src python -m pytest tests/test_loop_run_properties.py

They use a fixed seed so a run is reproducible. Set SLACKGENTIC_LOOP_PROPERTY_SEEDED=0
to explore new cases, and SLACKGENTIC_LOOP_PROPERTY_MAX_EXAMPLES to search deeper.
"""

import json
import os
import tempfile
import threading
import unittest
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

from agent_harness import loops as loop_logic
from agent_harness.config import AgentCommandConfig
from agent_harness.loops import (
    AGENT_LOOP_COMPACT_SIGNAL_PREFIX,
    AGENT_LOOP_FETCH_SIGNAL_PREFIX,
    AGENT_LOOP_SUMMARY_SIGNAL_PREFIX,
    LOOP_CARRY_MAX_CHARS,
    LOOP_CARRY_WARN_RATIO,
    LOOP_COMPACT_SNAPSHOT_MAX_CHARS,
    LOOP_HEADLINE_MAX_CHARS,
    LOOP_SUMMARY_STATUSES,
    loop_summary_from_json,
    parse_agent_loop_summary_signal,
)
from agent_harness.models import (
    LOOP_QUIET_OUTPUT_METADATA_KEY,
    LOOP_SILENT_OUTPUT_METADATA_KEY,
    AgentTaskStatus,
    Loop,
    LoopOverlapPolicy,
    LoopRunKind,
    LoopRunStatus,
    LoopStatus,
    LoopVisibility,
    PermissionMode,
    Provider,
    SlackThreadRef,
    TeamAgentKind,
    utc_now,
)
from agent_harness.runtime.tasks import (
    AGENT_THREAD_DONE_SIGNAL,
    ManagedTaskRuntime,
    RunningTask,
)
from agent_harness.slack.app import (
    LOOP_RUN_STRANDED_GRACE,
    SETTING_LOOP_THREAD_DONE_DEFERRED_PREFIX,
    LoopRunner,
    SlackTeamController,
)
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team, create_agent_task
from tests.polling import shut_down_runtime
from tests.test_slack_app import FakeGateway, FakeRuntime
from tests.test_task_runtime import FakeGateway as FakeRuntimeGateway
from tests.test_task_runtime import OneShotProcess

RUN_PROPERTY_TESTS = os.environ.get("SLACKGENTIC_PROPERTY_TESTS") == "1"
heavy = unittest.skipUnless(
    RUN_PROPERTY_TESTS, "heavy property suite; set SLACKGENTIC_PROPERTY_TESTS=1 to run it"
)
MAX_EXAMPLES = int(os.environ.get("SLACKGENTIC_LOOP_PROPERTY_MAX_EXAMPLES", "1000"))
PROPERTY_SETTINGS = settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    # Fixed seed by default so CI never flakes; set to 0 to explore new cases.
    derandomize=os.environ.get("SLACKGENTIC_LOOP_PROPERTY_SEEDED", "1") != "0",
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.differing_executors],
)
STATEFUL_SETTINGS = settings(
    PROPERTY_SETTINGS,
    max_examples=MAX_EXAMPLES * 2,
    stateful_step_count=50,
)

PROSE_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-()"
PROSE = st.text(PROSE_ALPHABET, min_size=1, max_size=80).filter(lambda value: value.strip())
JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(10**6), 10**6),
    st.text(max_size=40),
)
JSON_VALUES = st.recursive(
    JSON_SCALARS,
    lambda inner: st.one_of(
        st.lists(inner, max_size=4),
        st.dictionaries(st.text(max_size=12), inner, max_size=4),
    ),
    max_leaves=12,
)


def _carry_size(carry: dict) -> int:
    # The harness measures the carry exactly like this.
    return len(json.dumps(carry, sort_keys=True))


@st.composite
def carries(draw):
    """Carry objects of every shape, with sizes clustered around the limit."""
    carry = draw(st.dictionaries(st.text(max_size=12), JSON_VALUES, max_size=5))
    target = draw(
        st.one_of(
            st.none(),
            st.integers(0, 400),
            st.integers(LOOP_CARRY_MAX_CHARS - 3, LOOP_CARRY_MAX_CHARS + 3),
            st.integers(
                int(LOOP_CARRY_MAX_CHARS * LOOP_CARRY_WARN_RATIO) - 3, LOOP_CARRY_MAX_CHARS
            ),
            st.integers(0, LOOP_CARRY_MAX_CHARS * 3),
        )
    )
    if target is None:
        return carry
    carry = {key: value for key, value in carry.items() if key != "pad"}
    padding = target - _carry_size({**carry, "pad": ""})
    if padding > 0:
        carry["pad"] = "x" * padding
    return carry


@st.composite
def summary_payloads(draw):
    payload = {
        "summary": draw(st.text(min_size=1, max_size=300).filter(lambda value: value.strip())),
        "status": draw(st.sampled_from(sorted(LOOP_SUMMARY_STATUSES))),
    }
    if draw(st.booleans()):
        payload["headline"] = draw(st.text(max_size=LOOP_HEADLINE_MAX_CHARS * 2))
    if draw(st.booleans()):
        payload["report"] = draw(st.text(max_size=400))
    if draw(st.booleans()):
        payload["carry"] = draw(carries())
    return payload


@heavy
class LoopSummaryParsingProperties(unittest.TestCase):
    @PROPERTY_SETTINGS
    @given(summary_payloads(), st.booleans())
    def test_carry_size_never_costs_the_run_its_summary(self, payload, ensure_ascii):
        signal = AGENT_LOOP_SUMMARY_SIGNAL_PREFIX + json.dumps(payload, ensure_ascii=ensure_ascii)

        parsed = parse_agent_loop_summary_signal(signal)

        self.assertIsNone(parsed.error)
        summary = parsed.summary
        assert summary is not None
        self.assertEqual(summary.summary, payload["summary"].strip())
        self.assertEqual(summary.status, payload["status"])
        report = (payload.get("report") or "").strip()
        self.assertEqual(summary.report, report or None)
        if summary.headline is not None:
            self.assertLessEqual(len(summary.headline), LOOP_HEADLINE_MAX_CHARS)
        carry = payload.get("carry")
        if carry is None:
            self.assertIsNone(summary.carry)
            self.assertIsNone(summary.carry_overflow_chars)
        elif _carry_size(carry) <= LOOP_CARRY_MAX_CHARS:
            self.assertEqual(summary.carry, carry)
            self.assertIsNone(summary.carry_overflow_chars)
        else:
            self.assertIsNone(summary.carry)
            self.assertEqual(summary.carry_overflow_chars, _carry_size(carry))
        if carry and summary.carry_chars is not None:
            self.assertEqual(summary.carry_chars, _carry_size(carry))
        # What the harness stores must read back as the same summary.
        stored = json.dumps(summary.to_payload(), sort_keys=True)
        self.assertEqual(loop_summary_from_json(stored), summary)

    @PROPERTY_SETTINGS
    @given(
        st.one_of(st.none(), st.integers(0, LOOP_CARRY_MAX_CHARS * 3)),
        st.one_of(st.none(), st.integers(0, LOOP_COMPACT_SNAPSHOT_MAX_CHARS)),
    )
    def test_run_prompt_shows_memory_sizes_and_warns_exactly_when_due(self, previous, snapshot):
        loop = SimpleNamespace(
            title="Example Watch",
            mission="Report on example metrics.",
            recurrence={"frequency": "daily", "time": "09:00"},
            timezone="UTC",
            metadata={},
        )
        run = SimpleNamespace(run_number=4, due_at=datetime(2026, 1, 5, 9, tzinfo=UTC))

        prompt = loop_logic.build_loop_run_prompt(
            loop,
            run,
            journal_rendered="(memory)",
            now=datetime(2026, 1, 5, 9, tzinfo=UTC),
            previous_carry_chars=previous,
            snapshot_chars=snapshot,
        )

        self.assertIn(f"at most {LOOP_CARRY_MAX_CHARS} characters as compact JSON", prompt)
        self.assertEqual("previous run's carry was" in prompt, bool(previous))
        over = bool(previous) and previous > LOOP_CARRY_MAX_CHARS
        self.assertEqual(f"over the {LOOP_CARRY_MAX_CHARS} limit" in prompt, over)
        near = bool(previous) and previous >= LOOP_CARRY_MAX_CHARS * LOOP_CARRY_WARN_RATIO
        self.assertEqual("Prune it this run" in prompt, near and not over)
        self.assertEqual("long-term memory snapshot is" in prompt, bool(snapshot))
        snapshot_near = (
            bool(snapshot) and snapshot >= LOOP_COMPACT_SNAPSHOT_MAX_CHARS * LOOP_CARRY_WARN_RATIO
        )
        self.assertEqual("When you next compact" in prompt, snapshot_near)


SIGNALS_THAT_CLOSE_A_RUN = (
    AGENT_LOOP_SUMMARY_SIGNAL_PREFIX + '{"summary": "Done."}',
    AGENT_THREAD_DONE_SIGNAL,
)
SIGNALS_THAT_DO_NOT = (
    "SLACKGENTIC: ROSTER Checking telemetry (1/4)",
    AGENT_LOOP_COMPACT_SIGNAL_PREFIX + '{"snapshot": "memory"}',
)


@heavy
class QuietRunOutputProperties(unittest.TestCase):
    @PROPERTY_SETTINGS
    @given(
        st.lists(PROSE, min_size=1, max_size=3),
        st.lists(st.sampled_from(SIGNALS_THAT_CLOSE_A_RUN + SIGNALS_THAT_DO_NOT), max_size=3),
        st.sampled_from(("quiet", "silent", "normal")),
        st.randoms(use_true_random=False),
    )
    def test_quiet_runs_post_none_of_the_agents_text(self, prose, signals, mode, rng):
        lines = [*prose, *signals]
        rng.shuffle(lines)
        chunk = "\n".join(lines)
        metadata = {
            "quiet": {LOOP_QUIET_OUTPUT_METADATA_KEY: True},
            "silent": {LOOP_SILENT_OUTPUT_METADATA_KEY: True},
            "normal": {},
        }[mode]
        store = Store(Path(":memory:"))
        runtime = None
        try:
            store.init_schema()
            agent = build_initial_model_team(codex_count=0, claude_count=1)[0]
            store.upsert_team_agent(agent)
            task = replace(create_agent_task(agent, "loop run", "C1"), metadata=metadata)
            store.upsert_agent_task(task)
            gateway = FakeRuntimeGateway()
            runtime = ManagedTaskRuntime(
                store,
                gateway,
                AgentCommandConfig(),
                process_factory=OneShotProcess,
                poll_seconds=0.01,
                on_agent_control=lambda *args: True,
            )
            running = RunningTask(
                task=task,
                agent=agent,
                process=OneShotProcess(None),
                thread=SlackThreadRef("C1", "171.panel"),
                worker=threading.Thread(),
            )

            runtime._post_agent_chunk(running, chunk)

            posts = {"quiet": False, "silent": False, "normal": True}[mode]
            self.assertEqual(bool(gateway.replies), posts)
            for reply in gateway.replies:
                self.assertNotIn("SLACKGENTIC", reply)
            if mode == "quiet":
                # Withheld narration still counts as a response, so an exit without
                # THREAD_DONE finalizes the run instead of cancelling it as silent.
                self.assertEqual(running.visible_message_count, 1)
        finally:
            if runtime is not None:
                shut_down_runtime(runtime)
            store.close()


class LoopGateway(FakeGateway):
    """FakeGateway plus the message deletion a quiet thread rollover uses."""

    def __init__(self):
        super().__init__()
        self.deleted: list[tuple[str, str]] = []

    def delete_message(self, channel_id, ts, *args, **kwargs):
        self.deleted.append((channel_id, ts))
        return True


class WorkerRuntime(FakeRuntime):
    """Fake runtime that models worker lifetimes like managed Claude.

    A send to a live worker only queues input: the agent reads it when its next
    turn starts, so stopping a worker drops whatever it has not read yet. A real
    loop run lost its summary feedback exactly this way, so every dropped message
    is recorded in ``lost`` with whether dropping it was acceptable (``excused``).
    A worker that is being stopped refuses new input, like ManagedTaskRuntime.
    """

    def __init__(self, excused):
        super().__init__()
        self.excused = excused
        self.alive: set[str] = set()
        self.stopping: set[str] = set()
        self.inbox: dict[str, list[str]] = {}
        self.lost: list[tuple[str, bool]] = []
        self.starts: dict[str, int] = {}
        # Task ids are random, so workers are ordered by first start instead;
        # anything else would make Hypothesis replay a case differently.
        self.start_order: list[str] = []
        # Workers started while this is set behave like Codex: no live input
        # mid-turn, but an interrupt lets the next message through.
        self.codex_next = False
        self.codex: set[str] = set()
        # Workers whose turn ended and who wait for input, and those of them that
        # have waited past the harness's idle grace.
        self.idle: set[str] = set()
        self.idle_long: set[str] = set()

    def _wake(self, task_id):
        self.idle.discard(task_id)
        self.idle_long.discard(task_id)

    def end_turn(self, task_id):
        # With input already queued, Claude starts its next turn right away.
        if task_id in self.alive and task_id not in self.stopping and not self.unread(task_id):
            self.idle.add(task_id)

    def let_time_pass(self):
        self.idle_long |= self.idle

    def task_idle_seconds(self, task_id):
        if task_id not in self.alive or task_id in self.stopping or task_id not in self.idle:
            return None
        return 10**6 if task_id in self.idle_long else 0.0

    def start_task(self, task, agent, thread):
        super().start_task(task, agent, thread)
        self.starts[task.task_id] = self.starts.get(task.task_id, 0) + 1
        if task.task_id not in self.start_order:
            self.start_order.append(task.task_id)
        self.alive.add(task.task_id)
        self.stopping.discard(task.task_id)
        self.inbox[task.task_id] = []
        self._wake(task.task_id)
        if self.codex_next:
            self.codex.add(task.task_id)
        else:
            self.codex.discard(task.task_id)
        return True

    def send_to_task(self, task_id, message):
        if task_id in self.codex:
            return False
        return self._deliver(task_id, message)

    def _deliver(self, task_id, message):
        # Like ManagedTaskRuntime: a worker being stopped refuses input, so the
        # harness falls back to resuming the session.
        if task_id in self.stopping:
            return False
        if task_id in self.alive:
            self.sent.append((task_id, message))
            self.inbox.setdefault(task_id, []).append(message)
            self._wake(task_id)
            return True
        return False

    def send_to_interrupted_task(self, task_id, message):
        return self._deliver(task_id, message)

    def interrupt_task(self, task_id):
        return task_id in self.alive and task_id not in self.stopping

    def is_task_running(self, task_id):
        return task_id in self.alive

    def stop_task(self, task_id, status=AgentTaskStatus.CANCELLED):
        self.stopped.append((task_id, status))
        if task_id in self.alive:
            self.lost.extend(
                (message, self.excused(task_id, message)) for message in self.inbox.pop(task_id, [])
            )
            self.stopping.add(task_id)
            self._wake(task_id)
        return True

    def read_inbox(self, task_id) -> list[str]:
        return self.inbox.pop(task_id, [])

    def unread(self, task_id) -> bool:
        return bool(self.inbox.get(task_id))

    def exit(self, task_id):
        assert not self.unread(task_id), "a worker with unread input takes another turn"
        self.alive.discard(task_id)
        self.stopping.discard(task_id)
        self._wake(task_id)

    def crash(self, task_id):
        # A crash loses queued input through no fault of the harness.
        self.inbox.pop(task_id, None)
        self.alive.discard(task_id)
        self.stopping.discard(task_id)
        self._wake(task_id)

    def kill_all(self):
        for task_id in list(self.alive):
            self.crash(task_id)


SUMMARY_LINES = {
    "ok": (True, {"summary": "All clear.", "status": "ok", "carry": {"seen": 1}}),
    "issue": (
        True,
        {"summary": "Error spike.", "status": "found_issue", "headline": "5xx up 12x"},
    ),
    "positive": (
        True,
        {"summary": "Release landed.", "status": "notable_positive", "headline": "v2 is live"},
    ),
    "severe": (
        True,
        {"summary": "Outage.", "status": "found_very_severe_issue", "headline": "prod is down"},
    ),
    "resolved": (True, {"summary": "Spike cleared.", "status": "resolved"}),
    "action": (True, {"summary": "Restarted it.", "status": "action_taken"}),
    "failed": (True, {"summary": "Could not reach telemetry.", "status": "failed"}),
    "oversized_carry": (
        True,
        {
            "summary": "Clean, big carry.",
            "status": "ok",
            "carry": {"pad": "x" * LOOP_CARRY_MAX_CHARS},
        },
    ),
    "long_headline": (
        True,
        {"summary": "Clean, wordy.", "status": "ok", "headline": "y" * 400},
    ),
    "empty": (False, {"summary": ""}),
    "bad_status": (False, {"summary": "Odd.", "status": "maybe"}),
    "not_json": (False, None),
}
COMPACT_LINES = {
    "valid": {"snapshot": "Baselines and open issues."},
    "oversized": {"snapshot": "x" * (loop_logic.LOOP_COMPACT_SNAPSHOT_MAX_CHARS + 1)},
    "empty": {"snapshot": ""},
}
FETCH_LINES = {"known": {"run": 1}, "unknown": {"run": 999}}
# Small enough that a handful of runs queues a harness compaction run.
COMPACTION_TRIGGER_CHARS = 150


def summary_line(choice: str) -> str:
    _, payload = SUMMARY_LINES[choice]
    if payload is None:
        return AGENT_LOOP_SUMMARY_SIGNAL_PREFIX + '{"summary": "unterminated'
    return AGENT_LOOP_SUMMARY_SIGNAL_PREFIX + json.dumps(payload)


@dataclass
class RunModel:
    """What the agent did during one run, as the harness should have seen it."""

    run_id: str
    task_id: str | None
    kind: LoopRunKind
    quiet: bool
    accepted_summary: str | None = None
    accepted_status: str | None = None
    said_done: bool = False
    valid_compactions: int = 0
    deferrals: int = 0
    # Set once something that may legitimately end this run has happened.
    may_end: bool = False


@dataclass
class WorkerModel:
    """Runtime-side state of one worker process, reset when it restarts."""

    starts: int
    terminal_handled: bool = False
    pending: list[str] = field(default_factory=list)


class LoopLifecycle(RuleBasedStateMachine):
    """Plays arbitrary agent, owner, scheduler, and daemon behavior against a real
    controller across consecutive runs of one loop. The invariants are the
    promises a loop makes no matter what happens."""

    def __init__(self):
        super().__init__()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.compaction_trigger = patch(
            "agent_harness.slack.app.LOOP_COMPACTION_TRIGGER_CHARS", COMPACTION_TRIGGER_CHARS
        )
        self.compaction_trigger.start()
        # Roll the quiet panel thread over every two quiet runs instead of ten.
        self.rollover_floor = patch("agent_harness.loops.LOOP_THREAD_ROLLOVER_MIN_RUNS", 1)
        self.rollover_floor.start()
        self.store = Store(Path(":memory:"))
        self.store.init_schema()
        self.gateway = LoopGateway()
        self.runtime = WorkerRuntime(self._loss_excused)
        self.controller = self._new_controller()
        self.runs: dict[str, RunModel] = {}
        self.workers: dict[str, WorkerModel] = {}
        self.loop_stopped = False
        self.loop_id: str | None = None
        self.owner_messages = 0
        # Workers the owner woke up for a finished run, to answer a question.
        self.followups: set[str] = set()

    def _new_controller(self) -> SlackTeamController:
        return SlackTeamController(
            self.store,
            self.gateway,
            default_channel_id="CMAIN",
            runtime=self.runtime,
            home=self.root,
            default_cwd=self.root,
            ignored_bot_id="BOWN",
        )

    @initialize(quiet=st.booleans())
    def create_loop(self, quiet):
        agent = replace(
            build_initial_model_team(codex_count=0, claude_count=1)[0],
            agent_id="loopagent_prop",
            handle="prop-loop",
            kind=TeamAgentKind.LOOP,
        )
        self.store.upsert_team_agent(agent)
        now = utc_now()
        loop = self.store.create_loop(
            Loop(
                loop_id="loop_prop",
                agent_id=agent.agent_id,
                owner_slack_user_id="UOWNER",
                title="Prod Watch",
                mission="Check prod for unexpected errors.",
                channel_id="CLOOP",
                channel_name="loop-prod-watch",
                visibility=LoopVisibility.PRIVATE,
                provider=Provider.CLAUDE,
                permission_mode=PermissionMode.SAFE_AUTO,
                recurrence={"frequency": "interval", "interval_seconds": 3600},
                timezone=None,
                next_run_at=now - timedelta(seconds=1),
                status=LoopStatus.ACTIVE,
                overlap_policy=LoopOverlapPolicy.SKIP,
                anchor_channel_id="CMAIN",
                anchor_thread_ts="100.000001",
                # Every approved loop has a pinned panel; quiet runs work in its thread.
                charter_message_ts="100.000002",
                created_at=now,
                updated_at=now,
                metadata={"quiet": quiet, "thread_rollover_runs": 2},
            )
        )
        self.loop_id = loop.loop_id
        self.quiet = quiet
        self.agent = agent
        self._scheduler_tick()

    # -- helpers ----------------------------------------------------------------

    def _loop(self) -> Loop:
        loop = self.store.get_loop("loop_prop")
        assert loop is not None
        return loop

    def _all_runs(self):
        # Every run, not the store's default page: a long example outgrows it.
        return self.store.list_loop_runs("loop_prop", limit=10_000)

    def _run_for_task(self, task_id: str):
        return next((run for run in self._all_runs() if run.task_id == task_id), None)

    def _thread_for(self, task_id: str) -> SlackThreadRef:
        task = self.store.get_agent_task(task_id)
        assert task is not None and task.thread_ts is not None
        return SlackThreadRef(task.channel_id, task.thread_ts, task.thread_ts)

    def _loss_excused(self, task_id: str, message: str) -> bool:
        # The owner stopping the loop, the harness having already given the run
        # its one extra turn, or the agent ending the run in the same message it
        # asked for an earlier run (so it never waited for the answer) makes
        # dropping unread input acceptable.
        if self.loop_stopped or message.startswith(("[LOOP FETCH RESULT", "Loop fetch")):
            return True
        run = self._run_for_task(task_id)
        if run is None:
            return True
        return bool(
            self.store.get_setting(f"{SETTING_LOOP_THREAD_DONE_DEFERRED_PREFIX}{run.run_id}")
        )

    def _sync(self):
        """Pick up new runs and fresh workers after anything the harness did."""
        for run in self._all_runs():
            if run.run_id not in self.runs:
                self.runs[run.run_id] = RunModel(
                    run_id=run.run_id,
                    task_id=run.task_id,
                    kind=run.kind,
                    # Quiet runs have no header message of their own.
                    quiet=run.kind != LoopRunKind.COMPACTION
                    and run.thread_ts is None
                    and run.status == LoopRunStatus.RUNNING,
                    may_end=run.status != LoopRunStatus.RUNNING,
                )
                self._check_new_run_prompt(run)
        for task_id, starts in self.runtime.starts.items():
            worker = self.workers.get(task_id)
            if worker is None or worker.starts != starts:
                self.workers[task_id] = WorkerModel(starts=starts)

    def _check_new_run_prompt(self, run):
        # A run must see how full its memory is: an over-limit carry from the run
        # before must be called out in its prompt.
        if run.kind == LoopRunKind.COMPACTION or not run.task_id:
            return
        task = self.store.get_agent_task(run.task_id)
        if task is None:
            return
        earlier = sorted(
            (item for item in self._all_runs() if item.run_id != run.run_id),
            key=lambda item: item.run_number,
            reverse=True,
        )
        previous = next(
            (
                summary
                for item in earlier
                if item.kind != LoopRunKind.COMPACTION
                and (summary := loop_summary_from_json(item.summary_json)) is not None
            ),
            None,
        )
        if previous is not None and previous.carry_overflow_chars:
            assert f"over the {LOOP_CARRY_MAX_CHARS} limit" in task.prompt

    def _allow_ending(self, *, task_id: str | None = None) -> None:
        """Record that open runs (or the run of ``task_id``) may now end."""
        for run in self._all_runs():
            if run.status != LoopRunStatus.RUNNING:
                continue
            if task_id is None or run.task_id == task_id:
                model = self.runs.get(run.run_id)
                if model is not None:
                    model.may_end = True

    def _control(self, task_id: str, signal: str) -> bool:
        task = self.store.get_agent_task(task_id)
        assert task is not None
        handled = self.controller.handle_runtime_agent_control(
            task, self.agent, self._thread_for(task_id), signal
        )
        self._sync()
        return handled

    def _live_workers(self) -> list[str]:
        return [task_id for task_id in self.runtime.start_order if task_id in self.runtime.alive]

    def _speaking_worker(self, older: bool = False) -> str | None:
        """The worker whose output the next agent message belongs to.

        Usually the newest one; ``older`` picks an earlier live worker, such as an
        owner follow-up on a finished quiet run that shares the panel thread."""
        running = [
            task_id for task_id in self._live_workers() if task_id not in self.runtime.stopping
        ]
        candidates = running or self._live_workers()
        if not candidates:
            return None
        # An older worker may still be shutting down and emit trailing output.
        return self._live_workers()[0] if older else candidates[-1]

    def _scheduler_tick(self):
        # The scheduler reconciles open runs before firing new ones.
        self._allow_ending()
        loop = self._loop()
        if loop.status == LoopStatus.ACTIVE:
            self.store.update_loop_schedule(
                loop.loop_id,
                recurrence=loop.recurrence,
                timezone=loop.timezone,
                next_run_at=utc_now() - timedelta(seconds=1),
            )
        LoopRunner(self.store, self.controller).sync_once()
        self._sync()

    # -- agent behavior ------------------------------------------------------------

    @precondition(lambda self: self._speaking_worker() is not None)
    @rule(
        summary=st.one_of(st.none(), st.sampled_from(sorted(SUMMARY_LINES))),
        compact=st.one_of(st.none(), st.sampled_from(sorted(COMPACT_LINES))),
        fetch=st.one_of(st.none(), st.none(), st.sampled_from(sorted(FETCH_LINES))),
        order=st.permutations(["summary", "compact", "fetch"]),
        thread_done=st.booleans(),
        older=st.booleans(),
    )
    def agent_message(self, summary, compact, fetch, order, thread_done, older):
        task_id = self._speaking_worker(older)
        assert task_id is not None
        self._agent_turn(task_id, summary, compact, fetch, order, thread_done)

    def _agent_turn(self, task_id, summary, compact, fetch, order, thread_done):
        self.runtime._wake(task_id)
        try:
            self._agent_turn_lines(task_id, summary, compact, fetch, order, thread_done)
        finally:
            self.runtime.end_turn(task_id)

    def _agent_turn_lines(self, task_id, summary, compact, fetch, order, thread_done):
        # A new turn reads everything queued before it.
        self.runtime.read_inbox(task_id)
        run = self._run_for_task(task_id)
        model = self.runs.get(run.run_id) if run is not None else None
        # The agent wrote the whole message while the run was open, so every valid
        # line in it counts, even one the harness reads after an earlier line.
        accepting = run is not None and run.status == LoopRunStatus.RUNNING
        choices = {"summary": summary, "compact": compact, "fetch": fetch}
        # The runtime dispatches loop control lines in the order they are written
        # and THREAD_DONE after the rest of the message, wherever it appears.
        for line in order:
            choice = choices[line]
            if choice is None:
                continue
            if line == "fetch":
                self._control(
                    task_id, AGENT_LOOP_FETCH_SIGNAL_PREFIX + json.dumps(FETCH_LINES[choice])
                )
            elif line == "compact":
                if choice == "valid":
                    self._allow_ending(task_id=task_id)
                self._control(
                    task_id, AGENT_LOOP_COMPACT_SIGNAL_PREFIX + json.dumps(COMPACT_LINES[choice])
                )
                if choice == "valid" and accepting and model is not None:
                    model.valid_compactions += 1
            else:
                self._control(task_id, summary_line(choice))
                valid, payload = SUMMARY_LINES[choice]
                if (
                    valid
                    and accepting
                    and model is not None
                    and model.kind != LoopRunKind.COMPACTION
                ):
                    model.accepted_summary = payload["summary"]
                    model.accepted_status = payload["status"]
        if not thread_done:
            return
        if model is not None:
            model.said_done = True
        self._allow_ending(task_id=task_id)
        handled = self._control(task_id, AGENT_THREAD_DONE_SIGNAL)
        worker = self.workers[task_id]
        after = self._run_for_task(task_id)
        if handled:
            worker.terminal_handled = True
            assert after is None or after.status != LoopRunStatus.RUNNING, (
                "a handled THREAD_DONE must finish the run"
            )
            return
        worker.pending.append(AGENT_THREAD_DONE_SIGNAL)
        if model is not None and after is not None and after.status == LoopRunStatus.RUNNING:
            model.deferrals += 1

    @precondition(lambda self: self._speaking_worker() is not None)
    @rule(
        summary=st.sampled_from(
            ["ok", "ok", "issue", "oversized_carry", "oversized_carry", "empty", "not_json"]
        ),
        compact=st.sampled_from([None, None, "valid", "oversized"]),
        compact_first=st.booleans(),
        older=st.booleans(),
    )
    def agent_finishes_run(self, summary, compact, compact_first, older):
        """The usual ending: a summary, maybe a snapshot, then THREAD_DONE."""
        task_id = self._speaking_worker(older)
        assert task_id is not None
        order = (
            ["compact", "summary", "fetch"] if compact_first else ["summary", "compact", "fetch"]
        )
        self._agent_turn(task_id, summary, compact, None, order, True)

    def _worker_with_feedback(self) -> str | None:
        for task_id in reversed(self._live_workers()):
            if any(
                message.startswith(("[LOOP HARNESS]", "Emit only the required"))
                for message in self.runtime.inbox.get(task_id, [])
            ):
                return task_id
        return None

    @precondition(lambda self: self._worker_with_feedback() is not None)
    @rule(
        summary=st.sampled_from(["ok", "ok", "issue", "oversized_carry", "empty"]),
        compact=st.sampled_from([None, "valid", "valid", "oversized"]),
        compact_first=st.booleans(),
        thread_done=st.sampled_from([True, True, True, False]),
    )
    def agent_answers_feedback(self, summary, compact, compact_first, thread_done):
        """How agents usually answer the harness: the corrected lines, then done."""
        task_id = self._worker_with_feedback()
        assert task_id is not None
        order = (
            ["compact", "summary", "fetch"] if compact_first else ["summary", "compact", "fetch"]
        )
        self._agent_turn(task_id, summary, compact, None, order, thread_done)

    @precondition(
        lambda self: any(not self.runtime.unread(task_id) for task_id in self._live_workers())
    )
    @rule(data=st.data())
    def worker_exits(self, data):
        idle = [task_id for task_id in self._live_workers() if not self.runtime.unread(task_id)]
        self._exit_worker(data.draw(st.sampled_from(idle)))

    def _exit_worker(self, task_id: str):
        """Mirror ManagedTaskRuntime's process-exit path."""
        self._allow_ending(task_id=task_id)
        starts_before = self.runtime.starts.get(task_id)
        self.runtime.exit(task_id)
        worker = self.workers[task_id]
        if worker.terminal_handled:
            return
        task = self.store.get_agent_task(task_id)
        assert task is not None
        thread = self._thread_for(task_id)
        handled: set[str] = set()
        for signal in dict.fromkeys(worker.pending):
            if self.controller.handle_runtime_agent_control(task, self.agent, thread, signal):
                handled.add(signal)
        restarted = self.runtime.starts.get(task_id) != starts_before
        if AGENT_THREAD_DONE_SIGNAL not in handled and not restarted:
            task = self.store.get_agent_task(task_id) or task
            self.controller.handle_runtime_task_done(task, self.agent, thread)
        self._sync()

    @precondition(lambda self: bool(self._live_workers()))
    @rule(data=st.data())
    def worker_crashes(self, data):
        task_id = data.draw(st.sampled_from(self._live_workers()))
        self.runtime.crash(task_id)
        self.runtime.alive.add(task_id)
        self._exit_worker(task_id)

    @rule()
    def daemon_restarts(self):
        # Every worker dies with no exit handling, then a fresh controller takes
        # over the same state.
        self.runtime.kill_all()
        self.controller = self._new_controller()
        self._sync()

    @rule(codex=st.booleans())
    def next_workers_are_codex(self, codex):
        # Codex workers cannot take input mid-turn; the harness interrupts them.
        self.runtime.codex_next = codex

    # -- time, scheduler, and owner --------------------------------------------------

    @rule()
    def time_passes(self):
        self.runtime.let_time_pass()
        for run in self._all_runs():
            if run.status == LoopRunStatus.RUNNING and run.task_id:
                task = self.store.get_agent_task(run.task_id)
                if task is not None:
                    stale = utc_now() - LOOP_RUN_STRANDED_GRACE - timedelta(seconds=1)
                    self.store.upsert_agent_task(replace(task, updated_at=stale))
        self._allow_ending()
        self.controller.reconcile_loop_runs()
        self._sync()

    @rule()
    def scheduler_tick(self):
        self._scheduler_tick()

    # Owner actions are rare (1 in 8 draws act) so most cases build long histories.
    @precondition(lambda self: self._loop().status == LoopStatus.ACTIVE)
    @rule(odds=st.integers(0, 7))
    def owner_pauses(self, odds):
        if odds == 0:
            self.controller._pause_loop(self._loop())

    @precondition(lambda self: self._loop().status == LoopStatus.PAUSED)
    @rule()
    def owner_resumes(self):
        self.controller._resume_loop(self._loop())

    @precondition(lambda self: not self.loop_stopped)
    @rule(odds=st.integers(0, 7))
    def owner_stops(self, odds):
        if odds != 0:
            return
        self.loop_stopped = True
        self._allow_ending()
        loop = self._loop()
        self.controller._stop_loop(
            loop, archive=False, channel_id=loop.channel_id, message_ts=None, announce=False
        )
        self._sync()

    @precondition(lambda self: bool(self._all_runs()))
    @rule(latest_finished=st.booleans(), odds=st.integers(0, 3))
    def owner_replies(self, latest_finished, odds):
        """The owner writes in the thread of the open run, or of the last finished one."""
        if odds != 0:
            return
        runs = self._all_runs()
        running = [run for run in runs if run.status == LoopRunStatus.RUNNING]
        finished = [run for run in runs if run.status != LoopRunStatus.RUNNING and run.task_id]
        pool = finished if latest_finished and finished else running or finished
        if not pool:
            return
        run = max(pool, key=lambda item: item.run_number)
        task = self.store.get_agent_task(run.task_id) if run.task_id else None
        thread_ts = run.thread_ts or (task.thread_ts if task is not None else None)
        if not thread_ts:
            return
        self.owner_messages += 1
        if run.status != LoopRunStatus.RUNNING and run.task_id:
            self.followups.add(run.task_id)
        self.controller.handle_event(
            {
                "event": {
                    "type": "message",
                    "channel": self._loop().channel_id,
                    "ts": f"300.{self.owner_messages:06d}",
                    "thread_ts": thread_ts,
                    "user": "UOWNER",
                    "text": f"Owner question {self.owner_messages}: anything new?",
                }
            }
        )
        self._sync()

    @rule(on=st.booleans(), odds=st.integers(0, 3))
    def owner_toggles_quiet(self, on, odds):
        if odds != 0:
            return
        self.owner_messages += 1
        self.controller.handle_event(
            {
                "event": {
                    "type": "message",
                    "channel": self._loop().channel_id,
                    "ts": f"300.{self.owner_messages:06d}",
                    "user": "UOWNER",
                    "text": f"loop quiet: {'on' if on else 'off'}",
                }
            }
        )
        self.quiet = self._loop().metadata.get("quiet") is True
        self._sync()

    # -- promises ----------------------------------------------------------------------

    @invariant()
    def at_most_one_run_is_open(self):
        running = [run for run in self._all_runs() if run.status == LoopRunStatus.RUNNING]
        assert len(running) <= 1, running

    @invariant()
    def the_first_feedback_always_reaches_the_agent(self):
        lost = [message for message, excused in self.runtime.lost if not excused]
        assert lost == [], f"feedback lost before the agent could read it: {lost}"

    @invariant()
    def thread_done_is_held_back_at_most_once_per_run(self):
        for model in self.runs.values():
            assert model.deferrals <= 1, model

    @invariant()
    def an_accepted_summary_is_never_lost(self):
        for run in self._all_runs():
            model = self.runs.get(run.run_id)
            if model is None or model.accepted_summary is None:
                continue
            if run.status != LoopRunStatus.DONE:
                continue
            stored = loop_summary_from_json(run.summary_json)
            assert stored is not None, f"run {run.run_number} lost the summary it accepted"
            assert stored.summary == model.accepted_summary

    @invariant()
    def a_valid_compaction_sent_during_a_run_is_saved(self):
        entries = self.store.list_loop_journal("loop_prop", include_superseded=True, limit=10_000)
        for model in self.runs.values():
            saved = [
                entry
                for entry in entries
                if entry.kind == "compaction" and entry.run_id == model.run_id
            ]
            # A run keeps its first valid snapshot and ignores repeats.
            assert len(saved) == min(model.valid_compactions, 1), (model, len(saved))

    @invariant()
    def a_run_ends_only_for_its_own_reasons(self):
        # Nothing another run's worker does may end this run.
        for run in self._all_runs():
            model = self.runs.get(run.run_id)
            if model is not None and run.status != LoopRunStatus.RUNNING:
                assert model.may_end, f"run {run.run_number} was ended by something else"

    @invariant()
    def compaction_runs_record_no_summary(self):
        for run in self._all_runs():
            if run.kind == LoopRunKind.COMPACTION:
                assert not run.summary_json, f"compaction run {run.run_number} kept a summary"

    @invariant()
    def a_finished_run_leaves_no_live_worker(self):
        working = self.runtime.alive - self.runtime.stopping
        for run in self._all_runs():
            if run.status != LoopRunStatus.RUNNING and run.task_id not in self.followups:
                assert run.task_id not in working, (
                    f"run {run.run_number} ended, worker kept running"
                )

    @invariant()
    def a_finished_run_is_reported_truthfully(self):
        for run in self._all_runs():
            model = self.runs.get(run.run_id)
            if model is None or run.status != LoopRunStatus.DONE:
                continue
            if model.kind == LoopRunKind.COMPACTION:
                assert run.thread_ts is None, "a compaction run posted to the channel"
                continue
            summary = loop_summary_from_json(run.summary_json)
            # The panel and history never show an unknown result as all clear.
            emoji = self.controller._loop_run_emoji(run)
            if summary is None:
                assert emoji == "⚠️", emoji
            if model.quiet:
                needs_attention = summary is None or summary.status != "ok"
                assert (run.thread_ts is not None) == needs_attention, (run, summary)
            if run.thread_ts is None:
                continue
            cards = [
                str(item.get("blocks")) + str(item.get("text"))
                # A card starts as a post and is then updated in place.
                for item in [*self.gateway.posts, *self.gateway.updates]
                if item.get("ts") == run.thread_ts
            ]
            assert cards, f"run {run.run_number} has no card"
            if summary is None:
                assert "No report" in cards[-1] and "✅" not in cards[-1], cards[-1]
            elif summary.status == "ok":
                assert "✅" in cards[-1], cards[-1]

    @invariant()
    def quiet_threads_hold_only_the_run_log(self):
        # The only thing written into the panel thread quiet runs work in is the
        # harness's run log: one line per finished run, none while it runs. A run
        # the owner's stop cut short is failed in place, without a card or a line.
        quiet_threads: set[str] = set()
        logged: dict[str, int] = {}
        for run in self._all_runs():
            task = self.store.get_agent_task(run.task_id) if run.task_id else None
            if run.kind == LoopRunKind.COMPACTION or task is None or not task.thread_ts:
                continue
            if task.thread_ts == run.thread_ts:
                continue  # a run with a header message of its own
            quiet_threads.add(task.thread_ts)
            finalized = run.status == LoopRunStatus.DONE or (
                run.status == LoopRunStatus.FAILED and run.thread_ts is not None
            )
            if finalized:
                marker = f" Run #{run.run_number} "
                logged[marker] = logged.get(marker, 0) + 1
        replies = [
            reply["text"]
            for reply in self.gateway.thread_replies
            if reply["thread"].thread_ts in quiet_threads
        ]
        for text in replies:
            assert any(marker in text for marker in logged), text
        for marker, expected in logged.items():
            assert sum(marker in text for text in replies) == expected, (marker, replies)

    def _record_coverage(self):
        """Report which states this example reached (see --hypothesis-show-statistics)."""
        runs = self._all_runs()
        event(f"runs: {min(len(runs), 5)}{'+' if len(runs) >= 5 else ''}")
        for label, reached in (
            ("THREAD_DONE held back", any(model.deferrals for model in self.runs.values())),
            ("compaction run", any(run.kind == LoopRunKind.COMPACTION for run in runs)),
            ("skipped run", any(run.status == LoopRunStatus.SKIPPED for run in runs)),
            ("failed run", any(run.status == LoopRunStatus.FAILED for run in runs)),
            (
                "no-report run",
                any(run.status == LoopRunStatus.DONE and not run.summary_json for run in runs),
            ),
            (
                "carry dropped",
                any("carry_overflow_chars" in (run.summary_json or "") for run in runs),
            ),
            ("excused loss", any(excused for _, excused in self.runtime.lost)),
            ("worker restarted", any(starts > 1 for starts in self.runtime.starts.values())),
            ("owner replied", self.owner_messages > 0),
            ("thread rolled over", self._loop().charter_message_ts != "100.000002"),
            ("quiet and loud runs", len({model.quiet for model in self.runs.values()}) > 1),
        ):
            if reached:
                event(label)

    def teardown(self):
        try:
            if self.loop_id is None:
                return
            # Liveness. A worker still in its turn ends it, and queued input gets
            # a turn; in the worst case the agent answers with nothing and then
            # idles forever, as managed Claude does. The run must still end once
            # time passes, whether or not the agent ever said THREAD_DONE.
            for _ in range(8):
                running = [run for run in self._all_runs() if run.status == LoopRunStatus.RUNNING]
                if not running:
                    break
                run = running[0]
                task_id = run.task_id
                working = task_id not in self.runtime.idle and task_id not in self.runtime.stopping
                if task_id in self.runtime.alive and working:
                    self._agent_turn(task_id, None, None, None, [], False)
                self.time_passes()
            open_runs = [run for run in self._all_runs() if run.status == LoopRunStatus.RUNNING]
            assert open_runs == [], f"a run never finished: {open_runs}"
            self._record_coverage()
            for task_id in [t for t in self.runtime.start_order if t in self.runtime.stopping]:
                self._exit_worker(task_id)
            self.the_first_feedback_always_reaches_the_agent()
            self.an_accepted_summary_is_never_lost()
            self.a_finished_run_is_reported_truthfully()
        finally:
            self.compaction_trigger.stop()
            self.rollover_floor.stop()
            self.store.close()
            self.tmp.cleanup()


LoopLifecycle.TestCase.settings = STATEFUL_SETTINGS
TestLoopLifecycle = heavy(LoopLifecycle.TestCase)


if __name__ == "__main__":
    unittest.main()
