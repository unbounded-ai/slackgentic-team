from __future__ import annotations

from agent_harness.slack.agent_requests import (
    LEGACY_CODEX_REQUEST_ACTION,
    PendingAgentRequest,
    SlackAgentRequestHandler,
)

CODEX_REQUEST_ACTION = LEGACY_CODEX_REQUEST_ACTION
PendingCodexRequest = PendingAgentRequest


class SlackCodexRequestHandler(SlackAgentRequestHandler):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("provider_label", "Codex")
        kwargs.setdefault("action_name", LEGACY_CODEX_REQUEST_ACTION)
        super().__init__(*args, **kwargs)
