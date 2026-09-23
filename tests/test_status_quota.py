import json
import subprocess
import tempfile
import unittest
import unittest.mock
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent_harness.models import Provider, RateLimitWindow, TokenUsage, UsageSnapshot
from agent_harness.providers.quota import (
    ClaudeQuota,
    ClaudeSignIns,
    QuotaWindow,
    claude_plan_label,
    parse_claude_rate_limit_event,
    plan_label,
    probe_claude_quota,
    read_claude_sign_ins,
    window_label,
)
from agent_harness.providers.usage import build_status_report, format_daily_usage
from agent_harness.slack import STATUS_REFRESH_ACTION, build_status_blocks, decode_action_value

NOW = datetime(2026, 4, 26, 17, 0, tzinfo=UTC)


def _rate_limit_line(five_hour=0.03, seven_day=0.83):
    return json.dumps(
        {
            "type": "rate_limit_event",
            "rate_limit_info": {
                "status": "allowed_warning",
                "rateLimitType": "seven_day",
                "utilization": seven_day,
                "isUsingOverage": False,
                "unifiedWindows": {
                    "five_hour": {"utilization": five_hour, "resetsAt": 1777226400},
                    "seven_day": {"utilization": seven_day, "resetsAt": 1777500000},
                },
            },
        }
    )


def _claude(session_id, tokens, surface, day=NOW):
    return UsageSnapshot(
        provider=Provider.CLAUDE,
        session_id=session_id,
        as_of=day,
        usage=TokenUsage(total_tokens=tokens),
        surface=surface,
    )


def _codex(session_id, tokens, surface, weekly_percent=52.0, as_of=NOW):
    return UsageSnapshot(
        provider=Provider.CODEX,
        session_id=session_id,
        as_of=as_of,
        usage=TokenUsage(total_tokens=tokens),
        primary_limit=RateLimitWindow(
            used_percent=weekly_percent,
            window_minutes=10080,
            resets_at=NOW + timedelta(days=3),
        ),
        plan_type="self_serve_business_prolite",
        surface=surface,
    )


class ClaudeQuotaTests(unittest.TestCase):
    def test_rate_limit_event_reports_both_windows(self):
        output = "\n".join(
            [
                '{"type":"system","subtype":"init"}',
                "not json",
                _rate_limit_line(),
                '{"type":"result","subtype":"success"}',
            ]
        )

        quota = parse_claude_rate_limit_event(output, now=NOW)

        assert quota is not None
        self.assertEqual([window.label for window in quota.windows], ["5h", "Week"])
        self.assertAlmostEqual(quota.windows[0].used_percent, 3.0)
        self.assertAlmostEqual(quota.windows[1].used_percent, 83.0)
        self.assertEqual(quota.windows[0].resets_at, datetime.fromtimestamp(1777226400, UTC))
        self.assertEqual(ClaudeQuota.from_json(quota.to_json()), quota)

    def test_older_event_without_unified_windows_keeps_the_reported_window(self):
        line = json.dumps(
            {
                "type": "rate_limit_event",
                "rate_limit_info": {"rateLimitType": "five_hour", "utilization": 0.5},
            }
        )

        quota = parse_claude_rate_limit_event(line, now=NOW)

        assert quota is not None
        self.assertEqual(quota.windows, (QuotaWindow("5h", 50.0, None),))

    def test_output_without_rate_limits_has_no_quota(self):
        self.assertIsNone(parse_claude_rate_limit_event('{"type":"result"}\n', now=NOW))
        self.assertIsNone(ClaudeQuota.from_json("not json"))

    def test_probe_is_a_tiny_isolated_request_without_inherited_sign_in(self):
        calls = []

        def runner(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, stdout=_rate_limit_line(), stderr="")

        with unittest.mock.patch.dict(
            "os.environ",
            {"CLAUDE_CODE_OAUTH_TOKEN": "example", "ANTHROPIC_BASE_URL": "http://127.0.0.1:1"},
        ):
            quota = probe_claude_quota("claude", runner=runner)

        self.assertIsNotNone(quota)
        args, kwargs = calls[0]
        for flag in ("--no-session-persistence", "--strict-mcp-config", "--max-turns"):
            self.assertIn(flag, args)
        self.assertEqual(args[args.index("--tools") + 1], "")
        self.assertEqual(args[args.index("--output-format") + 1], "stream-json")
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", kwargs["env"])
        self.assertNotIn("ANTHROPIC_BASE_URL", kwargs["env"])

    def test_probe_failure_has_no_quota(self):
        def runner(args, **kwargs):
            raise subprocess.TimeoutExpired(args, 1)

        self.assertIsNone(probe_claude_quota("claude", runner=runner))


class SignInTests(unittest.TestCase):
    def _home(self, tmp, cli_uuid, app_uuid):
        home = Path(tmp)
        (home / ".claude.json").write_text(
            json.dumps(
                {
                    "oauthAccount": {
                        "accountUuid": cli_uuid,
                        "displayName": "Sam Example",
                        "organizationType": "claude_max",
                        "userRateLimitTier": "default_claude_max_20x",
                    }
                }
            )
        )
        app_dir = home / "Library" / "Application Support" / "Claude"
        app_dir.mkdir(parents=True)
        (app_dir / "config.json").write_text(json.dumps({"lastKnownAccountUuid": app_uuid}))
        return home

    def test_app_signed_in_to_another_account_is_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            sign_ins = read_claude_sign_ins(self._home(tmp, "account-a", "account-b"))

        self.assertEqual(sign_ins, ClaudeSignIns("Sam", "Max 20x", app_is_separate=True))

    def test_app_on_the_same_account_is_not_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            sign_ins = read_claude_sign_ins(self._home(tmp, "account-a", "account-a"))

        self.assertFalse(sign_ins.app_is_separate)

    def test_unknown_accounts_are_never_assumed_to_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_claude_sign_ins(Path(tmp)), ClaudeSignIns(app_is_separate=True))

    def test_plan_and_window_labels(self):
        self.assertEqual(claude_plan_label("claude_max", "default_claude_max_5x"), "Max 5x")
        self.assertEqual(claude_plan_label("claude_pro", None), "Pro")
        self.assertIsNone(claude_plan_label(None, None))
        self.assertEqual(plan_label("self_serve_business_prolite"), "Business Pro Lite")
        self.assertEqual(plan_label("plus"), "Plus")
        self.assertEqual(window_label(300), "5h")
        self.assertEqual(window_label(10080), "Week")
        self.assertEqual(window_label(None), "Window")


class StatusReportTests(unittest.TestCase):
    def _report(self, snapshots, **kwargs):
        kwargs.setdefault(
            "claude_quota",
            parse_claude_rate_limit_event(_rate_limit_line(), now=NOW),
        )
        return build_status_report(
            "2026-04-26",
            snapshots,
            snapshots,
            now=NOW,
            **kwargs,
        )

    def test_separate_claude_app_sign_in_gets_its_own_account(self):
        accounts = self._report(
            [
                _claude("cli-1", 900, "cli"),
                _claude("worker-1", 100, "sdk-cli"),
                _claude("app-1", 400, "claude-desktop"),
            ],
            claude_sign_ins=ClaudeSignIns(cli_plan="Max 20x", app_is_separate=True),
        )

        by_key = {account.key: account for account in accounts}
        cli, app = by_key["claude.cli"], by_key["claude.app"]
        self.assertEqual((cli.surfaces, cli.plan), ("CLI", "Max 20x"))
        self.assertEqual(cli.today_tokens, 1000)
        self.assertEqual(cli.session_count, 2)
        self.assertEqual([window.label for window in cli.windows], ["5h", "Week"])
        self.assertEqual(app.today_tokens, 400)
        self.assertEqual(app.windows, ())
        self.assertIn("Claude app", app.quota_note)

    def test_shared_sign_in_merges_cli_and_app(self):
        accounts = self._report(
            [_claude("cli-1", 900, "cli"), _claude("app-1", 400, "claude-desktop")],
            claude_sign_ins=ClaudeSignIns(app_is_separate=False),
        )

        claude = [account for account in accounts if account.provider == Provider.CLAUDE]
        self.assertEqual(len(claude), 1)
        self.assertEqual(claude[0].surfaces, "CLI + app")
        self.assertEqual(claude[0].today_tokens, 1300)

    def test_codex_cli_and_app_share_one_account_with_live_weekly_quota(self):
        accounts = self._report(
            [
                _codex(
                    "tui-1", 700, "codex-tui", weekly_percent=40.0, as_of=NOW - timedelta(hours=1)
                ),
                _codex("app-1", 300, "Codex Desktop", weekly_percent=52.0),
            ]
        )

        codex = next(account for account in accounts if account.provider == Provider.CODEX)
        self.assertEqual(codex.surfaces, "CLI + app")
        self.assertEqual(codex.plan, "Business Pro Lite")
        self.assertEqual(codex.windows, (QuotaWindow("Week", 52.0, NOW + timedelta(days=3)),))
        self.assertEqual(codex.today_tokens, 1000)

    def test_codex_app_on_a_different_account_is_shown_separately(self):
        cli = _codex("tui-1", 700, "codex-tui", weekly_percent=40.0)
        other_reset = replace(
            _codex("app-1", 300, "Codex Desktop", weekly_percent=10.0),
            primary_limit=RateLimitWindow(10.0, 10080, NOW + timedelta(days=5)),
        )
        other_plan = replace(_codex("app-2", 300, "Codex Desktop"), plan_type="plus")
        stale_app = replace(
            _codex("app-3", 300, "Codex Desktop"),
            primary_limit=RateLimitWindow(90.0, 10080, NOW - timedelta(hours=1)),
        )
        for app in (other_reset, other_plan, stale_app):
            with self.subTest(app=app.session_id):
                accounts = self._report([cli, app])

                codex = {
                    account.key: account
                    for account in accounts
                    if account.provider == Provider.CODEX
                }
                self.assertEqual(sorted(codex), ["codex.app", "codex.cli"])
                self.assertEqual(codex["codex.cli"].surfaces, "CLI")
                self.assertEqual(codex["codex.cli"].today_tokens, 700)
                self.assertEqual(codex["codex.app"].surfaces, "app")
                self.assertEqual(codex["codex.app"].today_tokens, 300)

    def test_codex_app_alone_is_labelled_as_the_app(self):
        accounts = self._report([_codex("app-1", 300, "Codex Desktop")])

        codex = next(account for account in accounts if account.provider == Provider.CODEX)
        self.assertEqual((codex.key, codex.surfaces), ("codex", "app"))

    def test_windows_that_already_reset_are_dropped(self):
        stale = ClaudeQuota(
            windows=(
                QuotaWindow("5h", 99.0, NOW - timedelta(minutes=1)),
                QuotaWindow("Week", 40.0, NOW + timedelta(days=2)),
            ),
            as_of=NOW - timedelta(hours=6),
        )

        accounts = self._report([_claude("cli-1", 10, "cli")], claude_quota=stale)

        self.assertEqual([window.label for window in accounts[0].windows], ["Week"])

    def test_top_sessions_use_labels_and_skip_slivers(self):
        accounts = self._report(
            [
                _claude("big", 9_000, "cli"),
                _claude("small", 1_000, "cli"),
                _claude("tiny", 5, "cli"),
            ],
            session_label=lambda snapshot: (
                "Ship the status card" if snapshot.session_id == "big" else None
            ),
        )

        top = accounts[0].top_sessions
        self.assertEqual(
            [session.label for session in top], ["Ship the status card", "session small"]
        )
        self.assertAlmostEqual(top[0].percent, 9_000 / 10_005 * 100)

    def test_missing_quota_says_so(self):
        accounts = self._report([_claude("cli-1", 10, "cli")], claude_quota=None)

        self.assertEqual(accounts[0].windows, ())
        self.assertEqual(accounts[0].quota_note, "Quota unavailable right now")

    def test_plain_text_status_lists_accounts_and_quota(self):
        text = format_daily_usage(
            "2026-04-26",
            [_claude("cli-1", 1_500_000, "cli"), _codex("tui-1", 2_000, "codex-tui")],
            claude_quota=parse_claude_rate_limit_event(_rate_limit_line(), now=NOW),
            claude_sign_ins=ClaudeSignIns(cli_plan="Max 20x"),
            now=NOW,
        )

        self.assertIn("Agent status · 2026-04-26", text)
        self.assertIn("Claude CLI + app · Max 20x", text)
        self.assertIn("5h   ▰▱▱▱▱▱▱▱▱▱ 3%", text)
        self.assertIn("Week ▰▰▰▰▰▰▰▰▱▱ 83%", text)
        self.assertIn("Today 1.5M tokens · 1 session · week 1.5M", text)
        self.assertIn("Codex CLI · Business Pro Lite", text)


class StatusBlockTests(unittest.TestCase):
    def test_blocks_show_bars_reset_times_and_a_refresh_button(self):
        accounts = build_status_report(
            "2026-04-26",
            [_claude("cli-1", 1_000, "cli"), _claude("app-1", 500, "claude-desktop")],
            claude_quota=parse_claude_rate_limit_event(_rate_limit_line(0.95, 0.5), now=NOW),
            claude_sign_ins=ClaudeSignIns(cli_plan="Max 20x", app_is_separate=True),
            now=NOW,
        )

        blocks = build_status_blocks(accounts, day_text="Sunday, Apr 26", updated_at=NOW)

        self.assertEqual(blocks[0]["type"], "header")
        sections = {block["block_id"]: block for block in blocks if block["type"] == "section"}
        cli = sections["usage.account.claude.cli"]["text"]["text"]
        self.assertIn("*Claude CLI · Max 20x*", cli)
        self.assertIn("🟥" * 10 + "  *95%* 5h", cli)
        self.assertIn("🟩" * 5 + "⬜" * 5 + "  *50%* Week", cli)
        self.assertIn("resets <!date^1777226400^{date_short_pretty} at {time}|", cli)
        app = sections["usage.account.claude.app"]["text"]["text"]
        self.assertIn("*Claude app*", app)
        self.assertIn("quota shows only in the Claude app", app)
        button = blocks[-1]["elements"][0]
        self.assertEqual(button["action_id"], STATUS_REFRESH_ACTION)
        self.assertEqual(decode_action_value(button["value"])["action"], STATUS_REFRESH_ACTION)


if __name__ == "__main__":
    unittest.main()
