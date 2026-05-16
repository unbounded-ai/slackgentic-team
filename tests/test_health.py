import threading
import time
import unittest

from agent_harness.runtime.health import LoopBackoff, ProcessCpuWatchdog


class HealthTests(unittest.TestCase):
    def test_loop_backoff_increases_and_resets(self):
        backoff = LoopBackoff(base_seconds=0.1, max_seconds=1.0)

        self.assertAlmostEqual(backoff.record_failure(), 0.1)
        self.assertAlmostEqual(backoff.record_failure(), 0.2)
        self.assertAlmostEqual(backoff.record_failure(), 0.4)
        self.assertEqual(backoff.failures, 3)

        backoff.reset()

        self.assertEqual(backoff.failures, 0)
        self.assertAlmostEqual(backoff.next_delay, 0.1)

    def test_loop_backoff_wait_honors_stop_event(self):
        backoff = LoopBackoff(base_seconds=10.0, max_seconds=10.0)
        stop = threading.Event()
        stop.set()

        start = time.monotonic()
        self.assertTrue(backoff.wait(stop))

        self.assertLess(time.monotonic() - start, 0.5)

    def test_cpu_watchdog_triggers_after_consecutive_hot_samples(self):
        process_values = iter([0.0, 1.0, 2.0, 3.0])
        wall_values = iter([0.0, 1.0, 2.0, 3.0])
        triggered: list[float] = []
        watchdog = ProcessCpuWatchdog(
            threshold=0.9,
            consecutive_samples=2,
            process_time=lambda: next(process_values),
            monotonic=lambda: next(wall_values),
            on_unhealthy=triggered.append,
        )

        self.assertIsNone(watchdog.sample_once())
        self.assertEqual(watchdog.sample_once(), 1.0)
        self.assertEqual(watchdog.sample_once(), 1.0)

        self.assertEqual(triggered, [1.0])


if __name__ == "__main__":
    unittest.main()
