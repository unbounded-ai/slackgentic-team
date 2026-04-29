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

    def test_format_daily_usage_emphasizes_bars_and_percentages(self):
        codex = parse_token_count(
            {
                "timestamp": "2026-04-26T17:00:00.000Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 10,
                            "cached_input_tokens": 5,
                            "output_tokens": 3,
                            "total_tokens": 18,
                        }
                    },
                    "rate_limits": {"primary": {"used_percent": 25.0}},
                },
            },
            "codex-session",
        )
        claude_usage = claude_usage_from_record(
            {
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "output_tokens": 2,
                    }
                }
            }
        )
        assert claude_usage is not None
        from agent_harness.models import UsageSnapshot

        claude = UsageSnapshot(
            provider=Provider.CLAUDE,
            session_id="claude-session",
            as_of=datetime(2026, 4, 26, 18, 0, tzinfo=UTC),
            usage=claude_usage,
        )

        text = format_daily_usage("2026-04-26", [codex, claude], [codex, claude])

        self.assertIn("Agent status", text)
        self.assertIn("quota window", text)
        self.assertIn("█", text)
        self.assertIn("week share", text)
        self.assertIn("Claude", text)
        self.assertIn("```", text)
        self.assertIn("Mix:", text)
        self.assertIn("Tokens: `18` today / `18` week", text)
        self.assertIn("Updated: `2026-04-26 17:00 UTC`", text)


if __name__ == "__main__":
    unittest.main()
