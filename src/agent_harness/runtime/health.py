from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class LoopBackoff:
    def __init__(
        self,
        *,
        base_seconds: float = 1.0,
        max_seconds: float = 60.0,
        multiplier: float = 2.0,
    ):
        self.base_seconds = max(0.0, base_seconds)
        self.max_seconds = max(self.base_seconds, max_seconds)
        self.multiplier = max(1.0, multiplier)
        self.failures = 0
        self._next_delay = self.base_seconds

    @property
    def next_delay(self) -> float:
        return self._next_delay

    def reset(self) -> None:
        self.failures = 0
        self._next_delay = self.base_seconds

    def record_failure(self) -> float:
        delay = self._next_delay
        self.failures += 1
        self._next_delay = min(self.max_seconds, max(self.base_seconds, delay * self.multiplier))
        return delay

    def wait(self, stop_event: threading.Event) -> bool:
        return stop_event.wait(self.record_failure())


def log_loop_failure(logger: logging.Logger, message: str, backoff: LoopBackoff) -> None:
    if backoff.failures == 0:
        logger.exception(message)
        return
    logger.warning(
        "%s; backing off for %.1fs after %d consecutive failures",
        message,
        backoff.next_delay,
        backoff.failures + 1,
    )


class ProcessCpuWatchdog:
    def __init__(
        self,
        *,
        interval_seconds: float = 60.0,
        threshold: float = 0.9,
        consecutive_samples: int = 5,
        process_time: Callable[[], float] = time.process_time,
        monotonic: Callable[[], float] = time.monotonic,
        on_unhealthy: Callable[[float], None] | None = None,
    ):
        self.interval_seconds = max(1.0, interval_seconds)
        self.threshold = max(0.0, threshold)
        self.consecutive_samples = max(1, consecutive_samples)
        self.process_time = process_time
        self.monotonic = monotonic
        self.on_unhealthy = on_unhealthy or _exit_for_sustained_cpu
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_cpu: float | None = None
        self._last_wall: float | None = None
        self._hot_samples = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="slackgentic-cpu-watchdog",
        )
        self._thread.start()

    def stop(self) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            return not self._thread.is_alive()
        return True

    def sample_once(self) -> float | None:
        now_wall = self.monotonic()
        now_cpu = self.process_time()
        if self._last_wall is None or self._last_cpu is None:
            self._last_wall = now_wall
            self._last_cpu = now_cpu
            return None
        wall_delta = max(0.001, now_wall - self._last_wall)
        cpu_delta = max(0.0, now_cpu - self._last_cpu)
        self._last_wall = now_wall
        self._last_cpu = now_cpu
        ratio = cpu_delta / wall_delta
        if ratio >= self.threshold:
            self._hot_samples += 1
        else:
            self._hot_samples = 0
        if self._hot_samples >= self.consecutive_samples:
            self.on_unhealthy(ratio)
            self._hot_samples = 0
        return ratio

    def _run(self) -> None:
        self.sample_once()
        while not self._stop.wait(self.interval_seconds):
            self.sample_once()


def _exit_for_sustained_cpu(ratio: float) -> None:
    LOGGER.critical(
        "Slackgentic daemon CPU watchdog tripped after sustained %.0f%% process CPU; exiting "
        "so the service manager can restart the daemon",
        ratio * 100,
    )
    os._exit(70)
