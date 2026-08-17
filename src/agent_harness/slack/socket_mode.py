from __future__ import annotations

import threading

from slack_sdk.socket_mode import SocketModeClient


class SupervisedSocketModeClient(SocketModeClient):
    """A Socket Mode client whose reconnect lifecycle is owned by Slackgentic.

    ``slack_sdk`` can reconnect from its close callback, monitor thread, and
    ``disconnect`` envelope handler. Slackgentic also needs to recover transports
    that the SDK still reports as connected. Letting both layers replace the same
    session caused races and reconnect storms, so SDK reconnect requests are
    converted into a signal for the outer supervisor instead.
    """

    def __init__(self, *args, **kwargs):
        self.reconnect_requested = threading.Event()
        close_listeners = list(kwargs.pop("on_close_listeners", None) or [])
        close_listeners.append(self._request_reconnect_after_close)
        kwargs["auto_reconnect_enabled"] = False
        kwargs["on_close_listeners"] = close_listeners
        super().__init__(*args, **kwargs)

    def connect_to_new_endpoint(self, force: bool = False) -> None:
        """Hand SDK reconnect requests to Slackgentic's single supervisor."""

        del force
        self.reconnect_requested.set()

    def close(self) -> None:
        """Close every SDK worker before the supervisor discards this generation."""

        try:
            super().close()
        finally:
            # The built-in client leaves this runner alive after close. A fresh
            # client per reconnect would otherwise leak one thread per generation
            # even though its transport and other workers were closed.
            session_runner = getattr(self, "current_session_runner", None)
            is_alive = getattr(session_runner, "is_alive", None)
            if callable(is_alive) and is_alive():
                session_runner.shutdown()

    def _request_reconnect_after_close(self, code: int, reason: str | None = None) -> None:
        del code, reason
        self.reconnect_requested.set()
