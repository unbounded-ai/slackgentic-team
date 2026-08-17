import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent_harness.models import (
    Loop,
    LoopJournalEntry,
    LoopOverlapPolicy,
    LoopRun,
    LoopRunKind,
    LoopRunStatus,
    LoopStatus,
    LoopVisibility,
    PermissionMode,
    Provider,
    TeamAgentKind,
)
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team


class LoopStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tempdir.name) / "state.sqlite")
        self.store.init_schema()
        agent = replace(
            build_initial_model_team(codex_count=1, claude_count=0)[0],
            agent_id="loopagent_test",
            handle="billing-loop",
            kind=TeamAgentKind.LOOP,
        )
        self.store.upsert_team_agent(agent)
        self.now = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
        self.due_at = self.now - timedelta(minutes=1)
        self.loop = Loop(
            loop_id="loop_test",
            agent_id=agent.agent_id,
            owner_slack_user_id="UOWNER",
            title="Billing Watch",
            mission="Check billing anomalies.",
            channel_id="CLOOP",
            channel_name="loop-billing-watch",
            visibility=LoopVisibility.PRIVATE,
            provider=Provider.CODEX,
            permission_mode=PermissionMode.SAFE_AUTO,
            recurrence={"frequency": "interval", "interval_seconds": 3600},
            timezone=None,
            next_run_at=self.due_at,
            status=LoopStatus.ACTIVE,
            overlap_policy=LoopOverlapPolicy.SKIP,
            anchor_channel_id="CMAIN",
            anchor_thread_ts="100.001",
            created_at=self.now,
            updated_at=self.now,
            metadata={"badge": "money"},
        )

    def tearDown(self):
        self.store.close()
        self.tempdir.cleanup()

    def test_loop_round_trip_filters_and_atomic_claim(self):
        created = self.store.create_loop(self.loop)

        self.assertEqual(self.store.get_loop(created.loop_id), created)
        self.assertEqual(self.store.get_loop_by_agent(created.agent_id), created)
        self.assertEqual(self.store.get_loop_by_channel("CLOOP"), created)
        self.assertEqual(self.store.list_loop_channel_ids(), ["CLOOP"])
        self.assertEqual(
            [loop.loop_id for loop in self.store.list_due_loops(now=self.now)],
            [created.loop_id],
        )

        next_run_at = self.now + timedelta(hours=1)
        claimed = self.store.claim_due_loop(
            created.loop_id,
            expected_next_run_at=self.due_at,
            new_next_run_at=next_run_at,
        )

        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed.run_count, 1)
        self.assertEqual(claimed.next_run_at, next_run_at)
        self.assertIsNotNone(claimed.last_run_at)
        self.assertIsNone(
            self.store.claim_due_loop(
                created.loop_id,
                expected_next_run_at=self.due_at,
                new_next_run_at=next_run_at + timedelta(hours=1),
            )
        )
        self.assertEqual(self.store.get_loop(created.loop_id).run_count, 1)

    def test_claim_due_loop_has_one_winner_across_store_connections(self):
        self.store.create_loop(self.loop)
        other_store = Store(Path(self.tempdir.name) / "state.sqlite")
        other_store.init_schema()
        barrier = threading.Barrier(2)
        results = []

        def claim(store: Store) -> None:
            barrier.wait()
            results.append(
                store.claim_due_loop(
                    self.loop.loop_id,
                    expected_next_run_at=self.due_at,
                    new_next_run_at=self.now + timedelta(hours=1),
                )
            )

        try:
            threads = (
                threading.Thread(target=claim, args=(self.store,)),
                threading.Thread(target=claim, args=(other_store,)),
            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            other_store.close()

        self.assertEqual(sum(result is not None for result in results), 1)
        claimed = self.store.get_loop(self.loop.loop_id)
        assert claimed is not None
        self.assertEqual(claimed.run_count, 1)

    def test_loop_updates_run_queries_and_failure_counter(self):
        self.store.create_loop(self.loop)
        self.store.update_loop_identity(
            self.loop.loop_id,
            title="Cost Watch",
            mission="Review daily spend.",
            channel_name="loop-cost-watch",
            visibility=LoopVisibility.PUBLIC,
            provider=Provider.CLAUDE,
            model="example-model",
            permission_mode=PermissionMode.LOCKED,
            cwd="/workspace/repos/example-project",
        )
        self.store.update_loop_schedule(
            self.loop.loop_id,
            recurrence={
                "frequency": "daily",
                "time": "09:00",
                "timezone": "America/New_York",
            },
            timezone="America/New_York",
            next_run_at=self.now + timedelta(days=1),
        )
        self.store.update_loop_pending_spec(
            self.loop.loop_id,
            '{"title":"Cost Watch"}',
            preview_message_ts="100.002",
        )

        updated = self.store.get_loop(self.loop.loop_id)
        assert updated is not None
        self.assertEqual(updated.title, "Cost Watch")
        self.assertEqual(updated.provider, Provider.CLAUDE)
        self.assertEqual(updated.visibility, LoopVisibility.PUBLIC)
        self.assertEqual(updated.preview_message_ts, "100.002")
        self.assertEqual(self.store.record_loop_failure(self.loop.loop_id, "first"), 1)
        self.assertEqual(self.store.record_loop_failure(self.loop.loop_id, "second"), 2)
        self.store.reset_loop_failures(self.loop.loop_id)
        self.assertEqual(self.store.get_loop(self.loop.loop_id).consecutive_failures, 0)

        run = LoopRun(
            run_id="lrun_test",
            loop_id=self.loop.loop_id,
            run_number=1,
            kind=LoopRunKind.SCHEDULED,
            due_at=self.due_at,
            started_at=self.now,
            status=LoopRunStatus.RUNNING,
            task_id="task_test",
            thread_ts="200.001",
            created_at=self.now,
            updated_at=self.now,
        )
        self.store.create_loop_run(run)
        self.assertEqual(self.store.running_loop_run(self.loop.loop_id), run)
        self.assertEqual(self.store.get_loop_run_by_task("task_test"), run)
        self.assertEqual(self.store.get_loop_run_by_thread(self.loop.loop_id, "200.001"), run)
        self.assertEqual(self.store.get_loop_run_by_number(self.loop.loop_id, 1), run)

        finished_at = self.now + timedelta(minutes=5)
        self.store.update_loop_run(
            run.run_id,
            status=LoopRunStatus.DONE,
            finished_at=finished_at,
            summary_json='{"summary":"No anomaly."}',
        )
        finished = self.store.get_loop_run(run.run_id)
        assert finished is not None
        self.assertEqual(finished.status, LoopRunStatus.DONE)
        self.assertEqual(finished.finished_at, finished_at)
        self.assertIsNone(self.store.running_loop_run(self.loop.loop_id))

    def test_compaction_supersedes_memory_but_not_owner_notes_or_itself(self):
        self.store.create_loop(self.loop)
        entries = (
            LoopJournalEntry(
                entry_id="lj_summary",
                loop_id=self.loop.loop_id,
                kind="run_summary",
                run_id="lrun_test",
                thread_ts="200.001",
                content="run summary",
                created_at=self.now,
            ),
            LoopJournalEntry(
                entry_id="lj_owner",
                loop_id=self.loop.loop_id,
                kind="owner_note",
                content="standing instruction",
                created_at=self.now + timedelta(seconds=1),
            ),
            LoopJournalEntry(
                entry_id="lj_compaction",
                loop_id=self.loop.loop_id,
                kind="compaction",
                content="long-term snapshot",
                created_at=self.now + timedelta(seconds=2),
            ),
        )
        for entry in entries:
            self.store.add_loop_journal_entry(entry)

        changed = self.store.supersede_loop_journal_entries(
            self.loop.loop_id,
            before=entries[-1].created_at,
            compaction_entry_id=entries[-1].entry_id,
        )

        self.assertEqual(changed, 1)
        visible_ids = {entry.entry_id for entry in self.store.list_loop_journal(self.loop.loop_id)}
        self.assertEqual(visible_ids, {"lj_owner", "lj_compaction"})
        all_entries = self.store.list_loop_journal(
            self.loop.loop_id,
            include_superseded=True,
        )
        superseded = {entry.entry_id: entry.superseded_by for entry in all_entries}
        self.assertEqual(superseded["lj_summary"], "lj_compaction")
        self.assertIsNone(superseded["lj_owner"])
        self.assertIsNone(superseded["lj_compaction"])


if __name__ == "__main__":
    unittest.main()
