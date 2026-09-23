import unittest
from datetime import UTC, datetime

from agent_harness.models import Provider
from agent_harness.providers.claude import claude_usage_from_record
from agent_harness.providers.codex import parse_token_count
from agent_harness.providers.usage import format_daily_usage


class UsageTests(unittest.TestCase):
    def test_parse_codex_token_count(self):
        record = {
            "timestamp": "2026-04-26T17:00:00.000Z",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 5,
                        "output_tokens": 3,
                        "reasoning_output_tokens": 2,
                        "total_tokens": 13,
                    },
                    "model_context_window": 100,
                },
                "rate_limits": {
                    "primary": {
                        "used_percent": 25.0,
                        "window_minutes": 300,
                        "resets_at": 1777240000,
                    },
                    "plan_type": "team",
                },
            },
        }
        snapshot = parse_token_count(record, "sid")
        self.assertEqual(snapshot.provider, Provider.CODEX)
        self.assertEqual(snapshot.usage.total_tokens, 13)
        self.assertEqual(snapshot.context_window, 100)
        self.assertEqual(snapshot.remaining_description, "75.0% primary window remaining")
        self.assertEqual(
            snapshot.as_of,
            datetime(2026, 4, 26, 17, 0, tzinfo=UTC),
        )

    def test_parse_claude_usage(self):
        record = {
            "message": {
                "usage": {
                    "input_tokens": 1,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 3,
                    "output_tokens": 4,
                }
            }
        }
        usage = claude_usage_from_record(record)
        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.total_tokens, 10)

    def test_parse_claude_top_level_usage(self):
        record = {
            "usage": {
                "input_tokens": 2,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 5,
                "output_tokens": 7,
            }
        }

        usage = claude_usage_from_record(record)

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.total_tokens, 17)

    def test_format_daily_usage_shows_codex_quota_and_tokens(self):
        codex = parse_token_count(
            {
                "timestamp": "2026-04-26T17:00:00.000Z",
                "payload": {
                    "type": "token_count",
                    "info": {"total_token_usage": {"total_tokens": 18}},
                    "rate_limits": {"primary": {"used_percent": 25.0, "window_minutes": 300}},
                },
            },
            "codex-session",
        )

        text = format_daily_usage("2026-04-26", [codex], [codex])

        self.assertIn("Agent status · 2026-04-26", text)
        self.assertIn("Codex CLI", text)
        self.assertIn("5h   ▰▰▰▱▱▱▱▱▱▱ 25%", text)
        self.assertIn("Today 18 tokens · 1 session · week 18", text)
        self.assertIn("Claude CLI + app", text)
        self.assertIn("Quota unavailable right now", text)


if __name__ == "__main__":
    unittest.main()
