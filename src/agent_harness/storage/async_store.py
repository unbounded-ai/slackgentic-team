from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from agent_harness.models import AgentSession
from agent_harness.storage.store import SCHEMA


class AsyncStore:
    def __init__(self, path: Path):
        self.path = path
        self.conn: aiosqlite.Connection | None = None

    async def __aenter__(self) -> AsyncStore:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        if self.conn:
            await self.conn.close()

    async def init_schema(self) -> None:
        conn = self._conn()
        await conn.executescript(SCHEMA)
        await conn.commit()

    async def upsert_session(self, session: AgentSession) -> None:
        conn = self._conn()
        await conn.execute(
            """
            INSERT INTO sessions (
              provider, session_id, transcript_path, cwd, started_at, last_seen_at,
              status, control_mode, model, git_branch, permission_mode, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, session_id) DO UPDATE SET
              transcript_path = excluded.transcript_path,
              cwd = excluded.cwd,
              started_at = excluded.started_at,
              last_seen_at = excluded.last_seen_at,
              status = excluded.status,
              control_mode = excluded.control_mode,
              model = excluded.model,
              git_branch = excluded.git_branch,
              permission_mode = excluded.permission_mode,
              metadata_json = excluded.metadata_json
            """,
            (
                session.provider.value,
                session.session_id,
                str(session.transcript_path),
                str(session.cwd) if session.cwd else None,
                session.started_at.isoformat() if session.started_at else None,
                session.last_seen_at.isoformat() if session.last_seen_at else None,
                session.status.value,
                session.control_mode.value,
                session.model,
                session.git_branch,
                session.permission_mode,
                json.dumps(session.metadata, sort_keys=True),
            ),
        )
        await conn.commit()

    def _conn(self) -> aiosqlite.Connection:
        if self.conn is None:
            raise RuntimeError("AsyncStore is not open")
        return self.conn
