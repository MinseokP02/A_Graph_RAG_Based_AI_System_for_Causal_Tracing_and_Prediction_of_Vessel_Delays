"""Offline control-event API contract and CSV tests."""

import csv

import collect_vessel_control as collector
from collect_port_calls import collect


def test_control_cli_parameters_and_empty_header(tmp_path, monkeypatch):
    output = tmp_path / "control.csv"
    captured = {}

    def fake_key(path, key_env):
        assert key_env == "MOF_CONTROL_SERVICE_KEY"
        return "test-key"

    monkeypatch.setattr(collector, "load_key", fake_key)

    def fake_collect(params, **kwargs):
        captured.update(params)
        assert kwargs["fetch"].keywords["endpoint"] == collector.ENDPOINT
        return []

    monkeypatch.setattr(collector, "collect", fake_collect)
    assert (
        collector.main(
            [
                "--port",
                "020",
                "--start",
                "20260801",
                "--end",
                "20260831",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert captured["sde"] == "20260801"
    assert "deGb" not in captured
    with output.open(encoding="utf-8-sig") as stream:
        assert next(csv.reader(stream)) == collector.CONTROL_FIELDS


def test_nested_control_events_keep_visit_and_leading_zeros():
    xml = b"""<response><header><resultCode>00</resultCode></header>
    <body><totalCount>1</totalCount><pageNo>1</pageNo><items><item>
    <prtAgCd>020</prtAgCd><clsgn>TEST</clsgn><etryptYear>2026</etryptYear>
    <etryptCo>006</etryptCo><details>
    <detail><commCo>001</commCo><cntrlSe>03</cntrlSe></detail>
    <detail><commCo>002</commCo><cntrlSe>04</cntrlSe></detail>
    </details></item></items></body></response>"""
    rows = collect({}, fetch=lambda params: xml)
    assert len(rows) == 2
    assert all(row["etryptCo"] == "006" for row in rows)
    assert rows[0]["record_id"] != rows[1]["record_id"]
