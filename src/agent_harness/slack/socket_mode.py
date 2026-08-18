from __future__ import annotations

import socket
import threading
from contextlib import suppress

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
        """Retire this generation without waiting on stale SDK worker locks."""

        # A socket retained across system sleep can remain non-None while its
        # receive thread is stuck inside SSL I/O holding the SDK's receive lock.
        # SocketModeClient.close() waits indefinitely for its monitor and session
        # runners, which can deadlock the supervisor before it creates the
        # replacement connection. Signal those workers, then close the raw socket
        # without entering the SDK lock hierarchy. They are daemon threads and
        # exit asynchronously once the socket wakes them.
        self.closed = True
        self.auto_reconnect_enabled = False
        session_state = getattr(self, "current_session_state", None)
        if session_state is not None:
            session_state.terminated = True
        for runner_name in (
            "current_app_monitor",
            "message_processor",
            "current_session_runner",
        ):
            runner = getattr(self, runner_name, None)
            event = getattr(runner, "event", None)
            set_event = getattr(event, "set", None)
            if callable(set_event):
                set_event()
        session = getattr(self, "current_session", None)
        active_socket = getattr(session, "sock", None)
        shutdown = getattr(active_socket, "shutdown", None)
        if callable(shutdown):
            with suppress(OSError):
                shutdown(socket.SHUT_RDWR)
        close_socket = getattr(active_socket, "close", None)
        if callable(close_socket):
            with suppress(OSError):
                close_socket()
        if session is not None:
            session.sock = None
        message_workers = getattr(self, "message_workers", None)
        shutdown_workers = getattr(message_workers, "shutdown", None)
        if callable(shutdown_workers):
            shutdown_workers(wait=False, cancel_futures=True)

    def _request_reconnect_after_close(self, code: int, reason: str | None = None) -> None:
        del code, reason
        self.reconnect_requested.set()
