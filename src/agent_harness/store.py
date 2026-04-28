from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from agent_harness.models import (
    AgentSession,
    AgentTask,
    AgentTaskKind,
    AgentTaskStatus,
    ControlMode,
    Persona,
    Provider,
    SessionDependency,
    SessionStatus,
    SlackThreadRef,
    TeamAgent,
    TeamAgentStatus,
    parse_timestamp,
    utc_now,
)

SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sessions (
  provider TEXT NOT NULL,
  session_id TEXT NOT NULL,
  transcript_path TEXT NOT NULL,
  cwd TEXT,
  started_at TEXT,
  last_seen_at TEXT,
  status TEXT NOT NULL,
  control_mode TEXT NOT NULL,
  model TEXT,
  git_branch TEXT,
  permission_mode TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (provider, session_id)
);

CREATE TABLE IF NOT EXISTS personas (
  provider TEXT NOT NULL,
  session_id TEXT NOT NULL,
  full_name TEXT NOT NULL,
  username TEXT NOT NULL,
  initials TEXT NOT NULL,
  color_hex TEXT NOT NULL,
  avatar_slug TEXT NOT NULL,
  icon_emoji TEXT NOT NULL,
  PRIMARY KEY (provider, session_id)
);

CREATE TABLE IF NOT EXISTS team_agents (
  agent_id TEXT NOT NULL PRIMARY KEY,
  handle TEXT NOT NULL UNIQUE,
  full_name TEXT NOT NULL,
  initials TEXT NOT NULL,
  color_hex TEXT NOT NULL,
  avatar_slug TEXT NOT NULL,
  icon_emoji TEXT NOT NULL,
  role TEXT NOT NULL,
  personality TEXT NOT NULL,
  voice TEXT NOT NULL,
  unique_strength TEXT NOT NULL,
  reaction_names_json TEXT NOT NULL,
  sort_order INTEGER NOT NULL,
  provider_preference TEXT,
  status TEXT NOT NULL,
  hired_at TEXT,
  fired_at TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_team_agents_status_sort
  ON team_agents(status, sort_order);

CREATE TABLE IF NOT EXISTS agent_tasks (
  task_id TEXT NOT NULL PRIMARY KEY,
  agent_id TEXT NOT NULL,
  prompt TEXT NOT NULL,
  channel_id TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'work',
  thread_ts TEXT,
  parent_message_ts TEXT,
  requested_by_slack_user TEXT,
  status TEXT NOT NULL,
  session_provider TEXT,
  session_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY(agent_id) REFERENCES team_agents(agent_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_tasks_agent_status
  ON agent_tasks(agent_id, status);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT NOT NULL PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS slack_threads (
  provider TEXT NOT NULL,
  session_id TEXT NOT NULL,
  team_id TEXT NOT NULL,
  channel_id TEXT NOT NULL,
  thread_ts TEXT NOT NULL,
  parent_ts TEXT,
  daily_usage_day TEXT,
  PRIMARY KEY (team_id, channel_id, thread_ts)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_slack_threads_session_channel
  ON slack_threads(provider, session_id, team_id, channel_id);

CREATE TABLE IF NOT EXISTS session_mirror_cursors (
  provider TEXT NOT NULL,
  session_id TEXT NOT NULL,
  last_line_number INTEGER NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (provider, session_id)
);

CREATE TABLE IF NOT EXISTS session_bridge_prompts (
  provider TEXT NOT NULL,
  session_id TEXT NOT NULL,
  prompt_text TEXT NOT NULL,
  consumed_at TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_session_bridge_prompts_pending
  ON session_bridge_prompts(provider, session_id, consumed_at);

CREATE TABLE IF NOT EXISTS claude_channel_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  target_pid INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  content TEXT NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}',
  delivered_at TEXT,
  cancelled_at TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_claude_channel_messages_pending
  ON claude_channel_messages(target_pid, delivered_at, cancelled_at, created_at);

CREATE TABLE IF NOT EXISTS slack_agent_requests (
  token TEXT PRIMARY KEY,
  provider_label TEXT NOT NULL,
  method TEXT NOT NULL,
  params_json TEXT NOT NULL,
  thread_channel_id TEXT NOT NULL,
  thread_ts TEXT NOT NULL,
  message_ts TEXT,
  answers_json TEXT NOT NULL DEFAULT '{}',
  response_json TEXT,
  resolved_at TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mirrored_slack_messages (
  channel_id TEXT NOT NULL,
  message_ts TEXT NOT NULL,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (channel_id, message_ts)
);

CREATE TABLE IF NOT EXISTS dependencies (
  blocked_session_id TEXT NOT NULL,
  blocking_channel_id TEXT NOT NULL,
  blocking_thread_ts TEXT NOT NULL,
  blocking_message_ts TEXT,
  permalink TEXT,
  created_by_slack_user TEXT,
  reason TEXT,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (blocked_session_id, blocking_channel_id, blocking_thread_ts)
);
"""


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def upsert_session(self, session: AgentSession) -> None:
        self.conn.execute(
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
        self.conn.commit()

    def upsert_persona(self, persona: Persona) -> None:
        self.conn.execute(
            """
            INSERT INTO personas (
              provider, session_id, full_name, username, initials, color_hex,
              avatar_slug, icon_emoji
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, session_id) DO UPDATE SET
              full_name = excluded.full_name,
              username = excluded.username,
              initials = excluded.initials,
              color_hex = excluded.color_hex,
              avatar_slug = excluded.avatar_slug,
              icon_emoji = excluded.icon_emoji
            """,
            (
                persona.provider.value,
                persona.session_id,
                persona.full_name,
                persona.username,
                persona.initials,
                persona.color_hex,
                persona.avatar_slug,
                persona.icon_emoji,
            ),
        )
        self.conn.commit()

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              value = excluded.value,
              updated_at = excluded.updated_at
            """,
            (key, value, utc_now().isoformat()),
        )
        self.conn.commit()

    def get_setting(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def delete_setting(self, key: str) -> None:
        self.conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        self.conn.commit()

    def upsert_team_agent(self, agent: TeamAgent) -> None:
        self.conn.execute(
            """
            INSERT INTO team_agents (
              agent_id, handle, full_name, initials, color_hex, avatar_slug,
              icon_emoji, role, personality, voice, unique_strength,
              reaction_names_json, sort_order, provider_preference, status,
              hired_at, fired_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
              handle = excluded.handle,
              full_name = excluded.full_name,
              initials = excluded.initials,
              color_hex = excluded.color_hex,
              avatar_slug = excluded.avatar_slug,
              icon_emoji = excluded.icon_emoji,
              role = excluded.role,
              personality = excluded.personality,
              voice = excluded.voice,
              unique_strength = excluded.unique_strength,
              reaction_names_json = excluded.reaction_names_json,
              sort_order = excluded.sort_order,
              provider_preference = excluded.provider_preference,
              status = excluded.status,
              hired_at = excluded.hired_at,
              fired_at = excluded.fired_at,
              metadata_json = excluded.metadata_json
            """,
            _team_agent_values(agent),
        )
        self.conn.commit()

    def list_team_agents(self, include_fired: bool = False) -> list[TeamAgent]:
        where = "" if include_fired else "WHERE status = ?"
        params: tuple[str, ...] = () if include_fired else (TeamAgentStatus.ACTIVE.value,)
        rows = self.conn.execute(
            f"SELECT * FROM team_agents {where} ORDER BY sort_order, full_name",
            params,
        ).fetchall()
        return [_team_agent_from_row(row) for row in rows]

    def idle_team_agents(self) -> list[TeamAgent]:
        rows = self.conn.execute(
            """
            SELECT ta.*
            FROM team_agents ta
            WHERE ta.status = ?
              AND NOT EXISTS (
                SELECT 1
                FROM agent_tasks task
                WHERE task.agent_id = ta.agent_id
                  AND task.status IN (?, ?)
              )
            ORDER BY ta.sort_order, ta.full_name
            """,
            (
                TeamAgentStatus.ACTIVE.value,
                AgentTaskStatus.QUEUED.value,
                AgentTaskStatus.ACTIVE.value,
            ),
        ).fetchall()
        return [_team_agent_from_row(row) for row in rows]

    def get_team_agent(self, handle_or_id: str, include_fired: bool = False) -> TeamAgent | None:
        normalized = handle_or_id.strip().lstrip("@").lower()
        row = self.conn.execute(
            """
            SELECT *
            FROM team_agents
            WHERE agent_id = ? OR lower(handle) = ?
            """,
            (handle_or_id, normalized),
        ).fetchone()
        if row is None:
            return None
        agent = _team_agent_from_row(row)
        if not include_fired and agent.status != TeamAgentStatus.ACTIVE:
            return None
        return agent

    def fire_team_agent(self, handle_or_id: str) -> TeamAgent | None:
        agent = self.get_team_agent(handle_or_id, include_fired=True)
        if agent is None:
            return None
        fired_at = utc_now().isoformat()
        self.conn.execute(
            "UPDATE team_agents SET status = ?, fired_at = ? WHERE agent_id = ?",
            (TeamAgentStatus.FIRED.value, fired_at, agent.agent_id),
        )
        self.conn.commit()
        return self.get_team_agent(agent.agent_id, include_fired=True)

    def next_team_sort_order(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order FROM team_agents"
        ).fetchone()
        return int(row["next_order"])

    def upsert_agent_task(self, task: AgentTask) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO agent_tasks (
                  task_id, agent_id, prompt, channel_id, kind, thread_ts, parent_message_ts,
                  requested_by_slack_user, status, session_provider, session_id,
                  created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                  agent_id = excluded.agent_id,
                  prompt = excluded.prompt,
                  channel_id = excluded.channel_id,
                  kind = excluded.kind,
                  thread_ts = excluded.thread_ts,
                  parent_message_ts = excluded.parent_message_ts,
                  requested_by_slack_user = excluded.requested_by_slack_user,
                  status = excluded.status,
                  session_provider = excluded.session_provider,
                  session_id = excluded.session_id,
                  updated_at = excluded.updated_at,
                  metadata_json = excluded.metadata_json
                """,
                _agent_task_values(task),
            )
            self.conn.commit()

    def update_agent_task_thread(
        self, task_id: str, thread_ts: str, parent_message_ts: str | None = None
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE agent_tasks
                SET thread_ts = ?, parent_message_ts = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (thread_ts, parent_message_ts, utc_now().isoformat(), task_id),
            )
            self.conn.commit()

    def update_agent_task_status(self, task_id: str, status: AgentTaskStatus) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE agent_tasks
                SET status = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (status.value, utc_now().isoformat(), task_id),
            )
            self.conn.commit()

    def update_agent_task_session(
        self,
        task_id: str,
        provider: Provider,
        session_id: str | None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE agent_tasks
                SET session_provider = ?, session_id = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (provider.value, session_id, utc_now().isoformat(), task_id),
            )
            self.conn.commit()

    def get_agent_task(self, task_id: str) -> AgentTask | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return _agent_task_from_row(row) if row else None

    def get_session(self, provider: Provider, session_id: str) -> AgentSession | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM sessions
            WHERE provider = ? AND session_id = ?
            """,
            (provider.value, session_id),
        ).fetchone()
        return _session_from_row(row) if row else None

    def list_sessions(self, provider: Provider | None = None) -> list[AgentSession]:
        where = ""
        params: tuple[str, ...] = ()
        if provider is not None:
            where = "WHERE provider = ?"
            params = (provider.value,)
        rows = self.conn.execute(
            f"""
            SELECT *
            FROM sessions
            {where}
            ORDER BY COALESCE(last_seen_at, started_at, '') DESC, session_id
            """,
            params,
        ).fetchall()
        return [_session_from_row(row) for row in rows]

    def get_agent_task_by_thread(self, channel_id: str, thread_ts: str) -> AgentTask | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM agent_tasks
            WHERE channel_id = ? AND thread_ts = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (channel_id, thread_ts),
        ).fetchone()
        return _agent_task_from_row(row) if row else None

    def get_original_agent_task_by_thread(
        self, channel_id: str, thread_ts: str
    ) -> AgentTask | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM agent_tasks
            WHERE channel_id = ? AND thread_ts = ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (channel_id, thread_ts),
        ).fetchone()
        return _agent_task_from_row(row) if row else None

    def get_slack_thread_for_session(
        self,
        provider: Provider,
        session_id: str,
        team_id: str,
        channel_id: str,
    ) -> SlackThreadRef | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM slack_threads
            WHERE provider = ? AND session_id = ? AND team_id = ? AND channel_id = ?
            LIMIT 1
            """,
            (provider.value, session_id, team_id, channel_id),
        ).fetchone()
        if row is None:
            return None
        return SlackThreadRef(
            channel_id=row["channel_id"],
            thread_ts=row["thread_ts"],
            message_ts=row["parent_ts"],
        )

    def get_session_for_slack_thread(
        self,
        team_id: str,
        channel_id: str,
        thread_ts: str,
    ) -> AgentSession | None:
        row = self.conn.execute(
            """
            SELECT session.*
            FROM slack_threads thread
            JOIN sessions session
              ON session.provider = thread.provider
             AND session.session_id = thread.session_id
            WHERE thread.team_id = ?
              AND thread.channel_id = ?
              AND thread.thread_ts = ?
            LIMIT 1
            """,
            (team_id, channel_id, thread_ts),
        ).fetchone()
        return _session_from_row(row) if row else None

    def upsert_slack_thread_for_session(
        self,
        provider: Provider,
        session_id: str,
        team_id: str,
        thread: SlackThreadRef,
        daily_usage_day: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO slack_threads (
              provider, session_id, team_id, channel_id, thread_ts, parent_ts,
              daily_usage_day
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, session_id, team_id, channel_id) DO UPDATE SET
              thread_ts = excluded.thread_ts,
              parent_ts = excluded.parent_ts,
              daily_usage_day = excluded.daily_usage_day
            """,
            (
                provider.value,
                session_id,
                team_id,
                thread.channel_id,
                thread.thread_ts,
                thread.message_ts,
                daily_usage_day,
            ),
        )
        self.conn.commit()

    def add_session_bridge_prompt(
        self,
        provider: Provider,
        session_id: str,
        prompt_text: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO session_bridge_prompts (
              provider, session_id, prompt_text, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (provider.value, session_id, prompt_text, utc_now().isoformat()),
        )
        self.conn.commit()

    def consume_session_bridge_prompt(
        self,
        provider: Provider,
        session_id: str,
        prompt_text: str,
    ) -> bool:
        rows = self.conn.execute(
            """
            SELECT rowid, prompt_text
            FROM session_bridge_prompts
            WHERE provider = ?
              AND session_id = ?
              AND consumed_at IS NULL
            ORDER BY created_at
            """,
            (provider.value, session_id),
        ).fetchall()
        row = next(
            (
                item
                for item in rows
                if _bridge_prompt_matches(str(item["prompt_text"]), prompt_text)
            ),
            None,
        )
        if row is None:
            return False
        self.conn.execute(
            """
            UPDATE session_bridge_prompts
            SET consumed_at = ?
            WHERE rowid = ?
            """,
            (utc_now().isoformat(), row["rowid"]),
        )
        self.conn.commit()
        return True

    def enqueue_claude_channel_message(
        self,
        target_pid: int,
        session_id: str,
        content: str,
        meta: dict[str, str],
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO claude_channel_messages (
              target_pid, session_id, content, meta_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                target_pid,
                session_id,
                content,
                json.dumps(meta, sort_keys=True),
                utc_now().isoformat(),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def pending_claude_channel_messages(
        self,
        target_pid: int,
        limit: int = 10,
    ) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT id, session_id, content, meta_json
                FROM claude_channel_messages
                WHERE target_pid = ?
                  AND delivered_at IS NULL
                  AND cancelled_at IS NULL
                ORDER BY created_at, id
                LIMIT ?
                """,
                (target_pid, limit),
            ).fetchall()
        )

    def mark_claude_channel_message_delivered(self, message_id: int) -> None:
        self.conn.execute(
            """
            UPDATE claude_channel_messages
            SET delivered_at = ?
            WHERE id = ? AND delivered_at IS NULL
            """,
            (utc_now().isoformat(), message_id),
        )
        self.conn.commit()

    def cancel_claude_channel_message(self, message_id: int) -> None:
        self.conn.execute(
            """
            UPDATE claude_channel_messages
            SET cancelled_at = ?
            WHERE id = ? AND delivered_at IS NULL AND cancelled_at IS NULL
            """,
            (utc_now().isoformat(), message_id),
        )
        self.conn.commit()

    def is_claude_channel_message_delivered(self, message_id: int) -> bool:
        row = self.conn.execute(
            """
            SELECT delivered_at
            FROM claude_channel_messages
            WHERE id = ?
            """,
            (message_id,),
        ).fetchone()
        return bool(row and row["delivered_at"])

    def create_slack_agent_request(
        self,
        token: str,
        provider_label: str,
        method: str,
        params: dict,
        thread: SlackThreadRef,
        *,
        message_ts: str | None = None,
        answers: dict[str, str] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO slack_agent_requests (
              token, provider_label, method, params_json, thread_channel_id,
              thread_ts, message_ts, answers_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token,
                provider_label,
                method,
                json.dumps(params, sort_keys=True),
                thread.channel_id,
                thread.thread_ts,
                message_ts,
                json.dumps(answers or {}, sort_keys=True),
                utc_now().isoformat(),
            ),
        )
        self.conn.commit()

    def update_slack_agent_request_message_ts(self, token: str, message_ts: str) -> None:
        self.conn.execute(
            """
            UPDATE slack_agent_requests
            SET message_ts = ?
            WHERE token = ?
            """,
            (message_ts, token),
        )
        self.conn.commit()

    def update_slack_agent_request_answers(self, token: str, answers: dict[str, str]) -> None:
        self.conn.execute(
            """
            UPDATE slack_agent_requests
            SET answers_json = ?
            WHERE token = ? AND resolved_at IS NULL
            """,
            (json.dumps(answers, sort_keys=True), token),
        )
        self.conn.commit()

    def resolve_slack_agent_request(self, token: str, response: object) -> None:
        self.conn.execute(
            """
            UPDATE slack_agent_requests
            SET response_json = ?, resolved_at = ?
            WHERE token = ? AND resolved_at IS NULL
            """,
            (json.dumps(response, sort_keys=True), utc_now().isoformat(), token),
        )
        self.conn.commit()

    def get_slack_agent_request(self, token: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT token, provider_label, method, params_json, thread_channel_id,
                   thread_ts, message_ts, answers_json, response_json, resolved_at
            FROM slack_agent_requests
            WHERE token = ?
            """,
            (token,),
        ).fetchone()

    def get_slack_agent_request_response(self, token: str) -> tuple[bool, object]:
        row = self.conn.execute(
            """
            SELECT response_json, resolved_at
            FROM slack_agent_requests
            WHERE token = ?
            """,
            (token,),
        ).fetchone()
        if row is None or not row["resolved_at"]:
            return False, None
        try:
            return True, json.loads(row["response_json"])
        except (TypeError, json.JSONDecodeError):
            return True, None

    def mark_slack_message_mirrored(
        self,
        channel_id: str,
        message_ts: str,
        source: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO mirrored_slack_messages (
              channel_id, message_ts, source, created_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(channel_id, message_ts) DO UPDATE SET
              source = excluded.source
            """,
            (channel_id, message_ts, source, utc_now().isoformat()),
        )
        self.conn.commit()

    def is_mirrored_slack_message(self, channel_id: str, message_ts: str | None) -> bool:
        if not message_ts:
            return False
        row = self.conn.execute(
            """
            SELECT 1
            FROM mirrored_slack_messages
            WHERE channel_id = ? AND message_ts = ?
            """,
            (channel_id, message_ts),
        ).fetchone()
        return row is not None

    def get_session_mirror_cursor(self, provider: Provider, session_id: str) -> int:
        row = self.conn.execute(
            """
            SELECT last_line_number
            FROM session_mirror_cursors
            WHERE provider = ? AND session_id = ?
            """,
            (provider.value, session_id),
        ).fetchone()
        return int(row["last_line_number"]) if row else 0

    def set_session_mirror_cursor(
        self,
        provider: Provider,
        session_id: str,
        last_line_number: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO session_mirror_cursors (
              provider, session_id, last_line_number, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(provider, session_id) DO UPDATE SET
              last_line_number = excluded.last_line_number,
              updated_at = excluded.updated_at
            """,
            (provider.value, session_id, last_line_number, utc_now().isoformat()),
        )
        self.conn.commit()

    def active_task_for_agent(self, agent_id: str) -> AgentTask | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM agent_tasks
            WHERE agent_id = ? AND status IN (?, ?)
            ORDER BY created_at
            LIMIT 1
            """,
            (agent_id, AgentTaskStatus.QUEUED.value, AgentTaskStatus.ACTIVE.value),
        ).fetchone()
        return _agent_task_from_row(row) if row else None

    def list_agent_tasks(self, include_done: bool = False) -> list[AgentTask]:
        where = "" if include_done else "WHERE status IN (?, ?)"
        params: tuple[str, ...] = (
            () if include_done else (AgentTaskStatus.QUEUED.value, AgentTaskStatus.ACTIVE.value)
        )
        rows = self.conn.execute(
            f"SELECT * FROM agent_tasks {where} ORDER BY created_at",
            params,
        ).fetchall()
        return [_agent_task_from_row(row) for row in rows]

    def add_dependency(self, dependency: SessionDependency) -> None:
        thread = dependency.blocking_thread
        self.conn.execute(
            """
            INSERT INTO dependencies (
              blocked_session_id, blocking_channel_id, blocking_thread_ts,
              blocking_message_ts, permalink, created_by_slack_user, reason, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(blocked_session_id, blocking_channel_id, blocking_thread_ts)
            DO UPDATE SET
              blocking_message_ts = excluded.blocking_message_ts,
              permalink = excluded.permalink,
              created_by_slack_user = excluded.created_by_slack_user,
              reason = excluded.reason,
              status = excluded.status
            """,
            (
                dependency.blocked_session_id,
                thread.channel_id,
                thread.thread_ts,
                thread.message_ts,
                thread.permalink,
                dependency.created_by_slack_user,
                dependency.reason,
                dependency.status,
            ),
        )
        self.conn.commit()

    def dependencies_for(self, blocked_session_id: str) -> list[SessionDependency]:
        rows = self.conn.execute(
            "SELECT * FROM dependencies WHERE blocked_session_id = ? ORDER BY created_at",
            (blocked_session_id,),
        ).fetchall()
        return [
            SessionDependency(
                blocked_session_id=row["blocked_session_id"],
                blocking_thread=SlackThreadRef(
                    channel_id=row["blocking_channel_id"],
                    thread_ts=row["blocking_thread_ts"],
                    message_ts=row["blocking_message_ts"],
                    permalink=row["permalink"],
                ),
                created_by_slack_user=row["created_by_slack_user"],
                reason=row["reason"],
                status=row["status"],
            )
            for row in rows
        ]


def _team_agent_values(agent: TeamAgent) -> tuple[object, ...]:
    return (
        agent.agent_id,
        agent.handle,
        agent.full_name,
        agent.initials,
        agent.color_hex,
        agent.avatar_slug,
        agent.icon_emoji,
        agent.role,
        agent.personality,
        agent.voice,
        agent.unique_strength,
        json.dumps(list(agent.reaction_names), sort_keys=True),
        agent.sort_order,
        agent.provider_preference.value if agent.provider_preference else None,
        agent.status.value,
        agent.hired_at.isoformat() if agent.hired_at else None,
        agent.fired_at.isoformat() if agent.fired_at else None,
        json.dumps(agent.metadata, sort_keys=True),
    )


def _session_from_row(row: sqlite3.Row) -> AgentSession:
    return AgentSession(
        provider=Provider(row["provider"]),
        session_id=row["session_id"],
        transcript_path=Path(row["transcript_path"]),
        cwd=Path(row["cwd"]) if row["cwd"] else None,
        started_at=parse_timestamp(row["started_at"]),
        last_seen_at=parse_timestamp(row["last_seen_at"]),
        status=SessionStatus(row["status"]),
        control_mode=ControlMode(row["control_mode"]),
        model=row["model"],
        git_branch=row["git_branch"],
        permission_mode=row["permission_mode"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _bridge_prompt_matches(expected: str, actual: str) -> bool:
    expected_clean = _normalize_bridge_prompt(expected)
    actual_clean = _normalize_bridge_prompt(actual)
    return actual_clean == expected_clean or actual_clean.startswith(
        (f"{expected_clean}\n", f"{expected_clean} ")
    )


def _normalize_bridge_prompt(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines()).strip()


def _team_agent_from_row(row: sqlite3.Row) -> TeamAgent:
    provider_preference = row["provider_preference"]
    return TeamAgent(
        agent_id=row["agent_id"],
        handle=row["handle"],
        full_name=row["full_name"],
        initials=row["initials"],
        color_hex=row["color_hex"],
        avatar_slug=row["avatar_slug"],
        icon_emoji=row["icon_emoji"],
        role=row["role"],
        personality=row["personality"],
        voice=row["voice"],
        unique_strength=row["unique_strength"],
        reaction_names=tuple(json.loads(row["reaction_names_json"] or "[]")),
        sort_order=int(row["sort_order"]),
        provider_preference=Provider(provider_preference) if provider_preference else None,
        status=TeamAgentStatus(row["status"]),
        hired_at=parse_timestamp(row["hired_at"]),
        fired_at=parse_timestamp(row["fired_at"]),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _agent_task_values(task: AgentTask) -> tuple[object, ...]:
    return (
        task.task_id,
        task.agent_id,
        task.prompt,
        task.channel_id,
        task.kind.value,
        task.thread_ts,
        task.parent_message_ts,
        task.requested_by_slack_user,
        task.status.value,
        task.session_provider.value if task.session_provider else None,
        task.session_id,
        task.created_at.isoformat(),
        task.updated_at.isoformat(),
        json.dumps(task.metadata, sort_keys=True),
    )


def _agent_task_from_row(row: sqlite3.Row) -> AgentTask:
    session_provider = row["session_provider"]
    return AgentTask(
        task_id=row["task_id"],
        agent_id=row["agent_id"],
        prompt=row["prompt"],
        channel_id=row["channel_id"],
        kind=AgentTaskKind(row["kind"]),
        thread_ts=row["thread_ts"],
        parent_message_ts=row["parent_message_ts"],
        requested_by_slack_user=row["requested_by_slack_user"],
        status=AgentTaskStatus(row["status"]),
        session_provider=Provider(session_provider) if session_provider else None,
        session_id=row["session_id"],
        created_at=parse_timestamp(row["created_at"]) or utc_now(),
        updated_at=parse_timestamp(row["updated_at"]) or utc_now(),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )
