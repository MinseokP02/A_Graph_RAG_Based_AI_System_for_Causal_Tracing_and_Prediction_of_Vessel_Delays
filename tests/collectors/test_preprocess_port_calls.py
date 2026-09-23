"""Verify lossless selection of the requested port-call columns."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocess_port_calls import OUTPUT_COLUMNS, preprocess


class PreprocessTests(unittest.TestCase):
    def test_selection_preserves_order_values_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.csv"
            row = {key: "" for key in OUTPUT_COLUMNS}
            row.update(prtAgCd="020", grtg="0", clsgn=" A B ", record_id="id", etryptYear="2026", etryptCo="006")
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS + ["extra"])
                writer.writeheader()
                writer.writerows([row, row])
            rows, summary = preprocess(path)
            self.assertEqual(rows, [row, row])
            self.assertEqual(list(rows[0]), OUTPUT_COLUMNS)
            self.assertEqual(summary["columns"], 34)

    def test_missing_columns_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.csv"
            path.write_text("record_id\nx\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing required columns"):
                preprocess(path)

    def test_empty_input_keeps_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.csv"
            path.write_text(",".join(OUTPUT_COLUMNS) + "\n", encoding="utf-8")
            rows, summary = preprocess(path)
            self.assertEqual(rows, [])
            self.assertEqual(summary["columns"], 34)
