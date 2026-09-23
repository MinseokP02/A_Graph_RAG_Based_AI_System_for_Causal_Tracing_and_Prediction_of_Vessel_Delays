"""Batch protocol, graph identities, error boundaries and driver parameters."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from middleware.config import Config
from middleware.graph_schema import CONSTRAINTS, port_query
from middleware.importer import run
from middleware.neo4j_client import write_batch
from middleware.preprocessing import port, prepare_row
from middleware.utils import Stats


def test_port_namespaces_and_fallback():
    assert (
        port("020", "Busan", "authority")["portKey"]
        != port("020", "Busan", "route")["portKey"]
    )
    assert port(None, " A  B ", "route") == port(None, "a b", "route")
    assert port(None, None, "route") is None


def test_berth_scope_and_routes(row):
    config = Config(Path("x"))
    first = prepare_row(row, config, Stats(), 1)
    row["prtAgCd"] = "030"
    second = prepare_row(row, config, Stats(), 2)
    assert first["berth"]["berthKey"] != second["berth"]["berthKey"]
    assert set(first["ports"]) == {"AT_PORT", "PREVIOUS_PORT"}
    assert "NEXT_PORT" not in first["ports"]


def test_missing_current_port_skips_berth(row):
    row.update(prtAgCd="", prtAgNm="")
    record = prepare_row(row, Config(Path("x")), Stats(), 1)
    assert record["berth"] is None
    assert "PREVIOUS_PORT" in record["ports"]


def test_parameterized_transaction(row):
    row["vsslNm"] = "Robert'); MATCH (n) DETACH DELETE n; //"
    record = prepare_row(row, Config(Path("x")), Stats(), 1)
    tx = MagicMock()
    write_batch(tx, [record, record])
    for call in tx.run.call_args_list:
        assert "Robert" not in call.args[0]
        assert "UNWIND $rows" in call.args[0]
        assert "MERGE" in call.args[0]
        assert "rows" in call.kwargs
    assert "datetime(" in tx.run.call_args_list[0].args[0]
    assert len(CONSTRAINTS) == 4
    assert all("IS UNIQUE" in q for q in CONSTRAINTS)
    with pytest.raises(ValueError):
        port_query("UNTRUSTED")


def test_batches_and_duplicate_id_mapping(row, csv_file):
    config = Config(csv_file([row, row, row]), batch_size=2)
    client = MagicMock()
    stats = Stats()
    run(config, stats, client)
    assert [len(c.args[0]) for c in client.import_batch.call_args_list] == [2, 1]
    assert stats.summary()["unique_port_calls"] == 1
    assert stats.summary()["processed_rows"] == 3


def test_failed_batch_not_counted(row, csv_file):
    client = MagicMock()
    client.import_batch.side_effect = RuntimeError("secret-password")
    stats = Stats()
    with pytest.raises(RuntimeError, match="Batch 1 failed") as exc:
        run(Config(csv_file([row])), stats, client)
    assert "secret" not in str(exc.value)
    assert stats.counts["processed_rows"] == 0


def test_dry_run(row, csv_file):
    stats = Stats()
    run(Config(csv_file([row])), stats)
    assert stats.counts["batches"] == 1
    assert stats.summary()["unique_berths"] == 1
