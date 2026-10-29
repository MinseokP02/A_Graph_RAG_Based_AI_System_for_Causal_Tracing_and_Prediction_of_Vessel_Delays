"""Offline tests for the standalone MOF collector (no model or API calls)."""

import csv
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit


SPEC = importlib.util.spec_from_file_location(
    "port_calls", Path(__file__).resolve().parents[2] / "collect_port_calls.py"
)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def page(items="", total=0, number=1):
    return f"""<response><header><resultCode>00</resultCode></header>
    <body><items>{items}</items><totalCount>{total}</totalCount>
    <pageNo>{number}</pageNo></body></response>""".encode()


ITEM = """<item><prtAgCd>020</prtAgCd><clsgn>TEST</clsgn>
<vsslNm>테스트</vsslNm><details>
<detail><etryndNm>입항</etryndNm><etryptDt>2026-01-01</etryptDt></detail>
<detail><etryndNm>출항</etryndNm><tkoffDt>2026-01-02</tkoffDt></detail>
</details></item>"""


class PortCallsTests(unittest.TestCase):
    def test_nested_details_preserve_parent_fields(self):
        rows, count, total, number = collector.parse_page(page(ITEM, 1))
        self.assertEqual((len(rows), count, total, number), (2, 1, 1, 1))
        self.assertEqual(rows[0]["prtAgCd"], "020")
        self.assertEqual(rows[1]["vsslNm"], "테스트")
        self.assertNotEqual(rows[0]["record_id"], rows[1]["record_id"])

    def test_pagination_counts_parent_items_not_details(self):
        calls = []

        def fetch(params):
            calls.append(params)
            n = int(params["pageNo"])
            return page(ITEM if n == 1 else ITEM.replace("TEST", "OTHER"), 2, n)

        rows = collector.collect({"prtAgCd": "020"}, delay=0, fetch=fetch)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(calls), 2)

    def test_no_results(self):
        self.assertEqual(collector.collect({}, fetch=lambda _: page()), [])

    def test_missing_details(self):
        rows, *_ = collector.parse_page(page("<item><clsgn>ABC</clsgn></item>", 1))
        self.assertEqual(rows[0]["clsgn"], "ABC")

    def test_api_errors_and_malformed_responses(self):
        for payload in [
            b"<html/>",
            b"broken",
            b"<response><header><resultCode>30</resultCode></header></response>",
            b"<OpenAPI_ServiceResponse><cmmMsgHeader><returnReasonCode>30</returnReasonCode></cmmMsgHeader></OpenAPI_ServiceResponse>",
        ]:
            with (
                self.subTest(payload=payload),
                self.assertRaises(collector.CollectionError),
            ):
                collector.parse_page(payload)

    def test_repeated_page_rejected(self):
        with self.assertRaises(collector.CollectionError):
            collector.collect({}, delay=0, fetch=lambda p: page(ITEM, 3, p["pageNo"]))

    def test_incomplete_and_changing_totals_rejected(self):
        for fetch in [
            lambda p: page("", 1),
            lambda p: page(
                ITEM.replace("TEST", str(p["pageNo"])),
                int(p["pageNo"]) + 1,
                p["pageNo"],
            ),
        ]:
            with self.assertRaises(collector.CollectionError):
                collector.collect({}, delay=0, fetch=fetch)

    def test_exact_duplicate_details_removed(self):
        self.assertEqual(
            len(collector.collect({}, fetch=lambda _: page(ITEM + ITEM, 2))), 2
        )

    def test_csv_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "result.csv"
            rows, *_ = collector.parse_page(page(ITEM, 1))
            collector.write_csv(target, rows)
            with target.open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]["prtAgCd"], "020")
            before = target.read_bytes()
            with self.assertRaises(collector.CollectionError):
                collector.write_csv(target, [])
            self.assertEqual(target.read_bytes(), before)
            collector.write_csv(target, [], overwrite=True)
            self.assertIn("record_id", target.read_text(encoding="utf-8-sig"))

    def test_encoded_and_decoded_key(self):
        for key in ["abc+def/==", "abc%2Bdef%2F%3D%3D"]:
            with patch.dict(os.environ, {collector.KEY_ENV: key}):
                self.assertEqual(collector.load_key(Path("missing")), "abc+def/==")

    def test_env_file(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {}, clear=True),
        ):
            path = Path(folder) / ".env"
            path.write_text('MOF_SERVICE_KEY="test+key"\n', encoding="utf-8")
            self.assertEqual(collector.load_key(path), "test+key")

    def test_request_parameters_encoded_once(self):
        with patch.object(collector, "urlopen") as opened:
            opened.return_value.__enter__.return_value.read.return_value = page()
            collector.fetch_page({"serviceKey": "abc+def/==", "prtAgCd": "020"})
            query = parse_qs(urlsplit(opened.call_args.args[0]).query)
            self.assertEqual(query["serviceKey"], ["abc+def/=="])

    def test_dates(self):
        self.assertEqual(collector.date_arg("2026-01-31"), "20260131")
        with self.assertRaises(collector.argparse.ArgumentTypeError):
            collector.date_arg("20260230")

    def test_timeout_retries_without_exposing_key(self):
        with (
            patch.object(
                collector, "urlopen", side_effect=TimeoutError("secret-url")
            ) as opened,
            patch.object(collector.time, "sleep"),
        ):
            with self.assertRaises(collector.CollectionError) as caught:
                collector.fetch_page({"serviceKey": "secret-key"}, retries=1)
            self.assertEqual(opened.call_count, 2)
            self.assertIn("timed out", str(caught.exception))
            self.assertNotIn("secret", str(caught.exception))

    def test_incomplete_response_retried(self):
        from http.client import IncompleteRead

        with (
            patch.object(collector, "urlopen") as opened,
            patch.object(collector.time, "sleep"),
        ):
            opened.return_value.__enter__.return_value.read.side_effect = [
                IncompleteRead(b"partial"),
                page(),
            ]
            self.assertEqual(collector.fetch_page({}, retries=1), page())
            self.assertEqual(opened.call_count, 2)

    def test_user_interrupt_has_clean_exit(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(collector, "load_key", return_value="test"),
            patch.object(collector, "collect", side_effect=KeyboardInterrupt),
        ):
            output = Path(folder) / "out.csv"
            result = collector.main(
                [
                    "--port",
                    "020",
                    "--start",
                    "20260801",
                    "--end",
                    "20260801",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(result, 130)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
