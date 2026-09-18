"""Deadline-based waiting for tests that observe background threads.

Attempt-counted polls such as ``for _ in range(100): ...; time.sleep(0.01)``
give a background thread about a second to make progress. A loaded CI runner
can stall a thread for longer than that, so waits are bounded by a generous
wall-clock deadline instead. Every poll still ends as soon as its condition
holds, so the deadline only costs time when a test is genuinely failing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

POLL_TIMEOUT_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 0.01


def poll_attempts(timeout: float = POLL_TIMEOUT_SECONDS) -> Iterator[None]:
    """Yield until ``timeout`` elapses, for loops that break once satisfied."""
    deadline = time.monotonic() + timeout
    while True:
        yield
        if time.monotonic() >= deadline:
            return


def wait_until(predicate: Callable[[], object], timeout: float = POLL_TIMEOUT_SECONDS) -> bool:
    """Poll ``predicate`` until it is truthy; return whether it ever was."""
    for _ in poll_attempts(timeout):
        if predicate():
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    return bool(predicate())


def shut_down_runtime(runtime, timeout: float = POLL_TIMEOUT_SECONDS) -> None:
    """Stop a ManagedTaskRuntime's tasks and wait for its worker threads to exit.

    Call this before ``store.close()``. Workers keep using the store until they
    exit, and closing a sqlite connection that another thread is still inside
    can crash the interpreter instead of failing the test.
    """
    deadline = time.monotonic() + timeout
    # A retry can hand the task to a new worker after a stop request, so keep
    # asking until no worker is left.
    while not runtime.join_workers(0.0):
        if time.monotonic() >= deadline:
            raise AssertionError("managed task workers were still running at test shutdown")
        runtime.stop_all_running_tasks(status=None, join_timeout=1.0)
        runtime.join_workers(1.0)
