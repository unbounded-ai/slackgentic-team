import json
import tempfile
import unittest
from pathlib import Path

from agent_harness.storage.jsonl import iter_jsonl, last_jsonl_line_number, tail_jsonl_records


class JsonlTests(unittest.TestCase):
    def test_iter_jsonl_after_line_skips_old_records_without_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"broken"',
                        json.dumps({"value": "old"}),
                        json.dumps({"value": "new"}),
                    ]
                )
                + "\n"
            )

            self.assertEqual(list(iter_jsonl(path, after_line=2)), [(3, {"value": "new"})])

    def test_last_jsonl_line_number_counts_complete_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"value": "old"}),
                        '{"broken"',
                        json.dumps({"value": "new"}),
                    ]
                )
                + "\n"
            )

            self.assertEqual(last_jsonl_line_number(path, chunk_size=8), 3)

    def test_last_jsonl_line_number_ignores_partial_trailing_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(json.dumps({"value": "old"}) + "\n" + '{"partial"')

            self.assertEqual(last_jsonl_line_number(path, chunk_size=8), 1)

    def test_last_jsonl_line_number_counts_valid_trailing_record_without_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(json.dumps({"value": "old"}) + "\n" + json.dumps({"value": "new"}))

            self.assertEqual(last_jsonl_line_number(path, chunk_size=8), 2)

    def test_tail_jsonl_records_reads_latest_valid_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            records = [{"index": index, "text": "x" * 128} for index in range(200)]
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            self.assertEqual(
                tail_jsonl_records(path, limit=3, chunk_size=128),
                records[-3:],
            )

    def test_tail_jsonl_records_handles_missing_final_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            records = [{"index": index} for index in range(5)]
            path.write_text("\n".join(json.dumps(record) for record in records))

            self.assertEqual(tail_jsonl_records(path, limit=2, chunk_size=16), records[-2:])


if __name__ == "__main__":
    unittest.main()
