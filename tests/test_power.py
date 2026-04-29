import unittest
from subprocess import CompletedProcess

from agent_harness.runtime.power import (
    ActiveSessionAwakeKeeper,
    format_power_doctor_lines,
    inspect_macos_power,
    parse_pmset_custom,
    parse_scheduled_wakes,
    parse_wake_on_wireless,
)


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if not self.terminated and not self.killed else 0

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


class ActiveSessionAwakeKeeperTests(unittest.TestCase):
    def test_starts_and_stops_caffeinate_when_active_state_changes(self):
        states = [True, False]
        started = []

        def is_active():
            return states[0]

        def popen(args, stdout=None, stderr=None):
            process = FakeProcess()
            started.append((args, process))
            return process

        keeper = ActiveSessionAwakeKeeper(
            is_active,
            system="darwin",
            popen_factory=popen,
            which=lambda command: "/usr/bin/caffeinate",
        )

        keeper.sync_once()
        states[0] = False
        keeper.sync_once()

        self.assertEqual(started[0][0], ["caffeinate", "-dimsu"])
        self.assertTrue(started[0][1].terminated)

    def test_ignores_non_macos_systems(self):
        started = []
        keeper = ActiveSessionAwakeKeeper(
            lambda: True,
            system="linux",
            popen_factory=lambda *args, **kwargs: started.append(args),
        )

        keeper.sync_once()

        self.assertEqual(started, [])

    def test_start_syncs_immediately(self):
        started = []

        def popen(args, stdout=None, stderr=None):
            process = FakeProcess()
            started.append((args, process))
            return process

        keeper = ActiveSessionAwakeKeeper(
            lambda: True,
            system="darwin",
            popen_factory=popen,
            which=lambda command: "/usr/bin/caffeinate",
        )

        keeper.start()
        keeper.stop()

        self.assertEqual(started[0][0], ["caffeinate", "-dimsu"])


class MacPowerStatusTests(unittest.TestCase):
    def test_parse_pmset_custom(self):
        output = """
Battery Power:
 womp                 0
 tcpkeepalive         1
AC Power:
 womp                 1
 tcpkeepalive         1
"""

        settings = parse_pmset_custom(output)

        self.assertEqual(settings["Battery Power"]["womp"], "0")
        self.assertEqual(settings["AC Power"]["tcpkeepalive"], "1")

    def test_parse_wake_on_wireless(self):
        output = """
Wi-Fi:
          Wake On Wireless: Supported
"""

        self.assertEqual(parse_wake_on_wireless(output), "Supported")

    def test_parse_scheduled_wakes(self):
        output = """
Scheduled power events:
 [0]  wake at 04/29/2026 01:17:42 by 'com.apple.foo'
 [1]  shutdown at 04/29/2026 03:00:00 by 'com.apple.foo'
"""

        self.assertEqual(
            parse_scheduled_wakes(output),
            ["[0]  wake at 04/29/2026 01:17:42 by 'com.apple.foo'"],
        )

    def test_inspect_macos_power_formats_network_wake_status(self):
        responses = {
            ("pmset", "-g", "custom"): """
Battery Power:
 womp                 0
 tcpkeepalive         1
AC Power:
 womp                 1
 tcpkeepalive         1
""",
            ("system_profiler", "SPAirPortDataType"): """
Wi-Fi:
          Wake On Wireless: Supported
""",
            ("pmset", "-g", "sched"): """
Scheduled power events:
 [0]  wake at 04/29/2026 01:17:42 by 'com.apple.foo'
""",
        }

        def run(command, stdout=None, stderr=None, text=None, timeout=None):
            return CompletedProcess(command, 0, responses.get(tuple(command), ""))

        status = inspect_macos_power(
            system="darwin",
            which=lambda command: f"/usr/bin/{command}",
            run=run,
        )

        lines = "\n".join(format_power_doctor_lines(status))
        self.assertIn("ok Wake for network access on power adapter (1)", lines)
        self.assertIn("off Wake for network access on battery (0)", lines)
        self.assertIn("ok Wi-Fi wake support (Supported)", lines)
        self.assertIn("info scheduled wake events (1)", lines)
        self.assertIn("cannot guarantee wake", lines)


if __name__ == "__main__":
    unittest.main()
