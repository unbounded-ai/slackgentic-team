from __future__ import annotations

from agent_harness.models import Provider
from agent_harness.storage.store import Store

MANAGED_SESSION_AGENT_PREFIX = "managed_session_agent."
MANAGED_SESSION_DANGEROUS_PREFIX = "managed_session_dangerous_mode."


def _session_key(provider: Provider, session_id: str) -> str:
    return f"{provider.value}.{session_id}"


def managed_session_agent_key(provider: Provider, session_id: str) -> str:
    return f"{MANAGED_SESSION_AGENT_PREFIX}{_session_key(provider, session_id)}"


def managed_session_dangerous_key(provider: Provider, session_id: str) -> str:
    return f"{MANAGED_SESSION_DANGEROUS_PREFIX}{_session_key(provider, session_id)}"


def record_managed_session(
    store: Store,
    provider: Provider,
    session_id: str,
    agent_id: str,
    *,
    dangerous_mode: bool,
) -> None:
    if not session_id:
        return
    store.set_setting(managed_session_agent_key(provider, session_id), agent_id)
    if dangerous_mode:
        store.set_setting(managed_session_dangerous_key(provider, session_id), "1")
    else:
        store.delete_setting(managed_session_dangerous_key(provider, session_id))


def clear_managed_session(store: Store, provider: Provider, session_id: str) -> None:
    if not session_id:
        return
    store.delete_setting(managed_session_agent_key(provider, session_id))
    store.delete_setting(managed_session_dangerous_key(provider, session_id))


def managed_session_is_dangerous(
    store: Store,
    provider: Provider,
    session_id: str,
) -> bool:
    if not session_id:
        return False
    return store.get_setting(managed_session_dangerous_key(provider, session_id)) == "1"


def managed_session_agents(store: Store) -> dict[tuple[Provider, str], str]:
    """Return a mapping of (provider, session_id) → agent_id for tracked managed sessions."""
    result: dict[tuple[Provider, str], str] = {}
    for key, agent_id in store.list_settings(MANAGED_SESSION_AGENT_PREFIX).items():
        suffix = key.removeprefix(MANAGED_SESSION_AGENT_PREFIX)
        provider_text, separator, session_id = suffix.partition(".")
        if not separator or not session_id:
            continue
        try:
            provider = Provider(provider_text)
        except ValueError:
            continue
        result[(provider, session_id)] = agent_id
    return result
