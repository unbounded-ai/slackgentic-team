from __future__ import annotations

import asyncio

from agent_harness.async_store import AsyncStore
from agent_harness.config import AppConfig, load_config_from_env
from agent_harness.providers import ClaudeProvider, CodexProvider
from agent_harness.watcher import TranscriptWatcher


class AgentDaemon:
    def __init__(self, config: AppConfig):
        self.config = config
        self.codex = CodexProvider(home=config.home)
        self.claude = ClaudeProvider(home=config.home)

    async def index_once(self) -> int:
        count = 0
        async with AsyncStore(self.config.state_db) as store:
            await store.init_schema()
            for provider in (self.codex, self.claude):
                for session in provider.discover():
                    await store.upsert_session(session)
                    count += 1
        return count

    async def watch(self) -> None:
        roots = [self.codex.sessions_root, self.claude.projects_root]
        watcher = TranscriptWatcher(roots)
        await self.index_once()
        async for _path in watcher.changes():
            await self.index_once()


async def async_main() -> int:
    daemon = AgentDaemon(load_config_from_env())
    count = await daemon.index_once()
    print(f"indexed {count} sessions")
    return 0


def main() -> int:
    return asyncio.run(async_main())
