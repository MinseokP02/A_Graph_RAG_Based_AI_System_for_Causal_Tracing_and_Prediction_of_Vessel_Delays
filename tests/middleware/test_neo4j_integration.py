"""Opt-in, real Neo4j idempotence and rollback checks with isolated fixture IDs."""

import os
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest

from middleware.config import Config
from middleware.neo4j_client import Neo4jClient
from middleware.preprocessing import prepare_row
from middleware.utils import Stats


@pytest.mark.skipif(
    not os.getenv("PORTMIS_TEST_NEO4J_URI"),
    reason="Dedicated test Neo4j not configured",
)
def test_reimport_native_datetime_and_rollback(row):
    """Import twice, assert graph cardinalities/types, and roll back a bad batch."""
    from neo4j.exceptions import Neo4jError
    prefix = "middleware-test-" + uuid4().hex
    row.update(
        record_id=prefix,
        clsgn=prefix,
        prtAgCd=prefix,
        prvsDpmprtNatPrtCd=prefix + "prev",
        etryptDt="2026-08-01 10:00:00",
        tkoffDt="2026-08-01 13:00:00",
        tkoffPrrrnDt="2026-08-01 11:30:00",
    )
    config = Config(
        Path("fixture"),
        uri=os.environ["PORTMIS_TEST_NEO4J_URI"],
        password=os.getenv("PORTMIS_TEST_NEO4J_PASSWORD", "test-only"),
    )
    payload = prepare_row(row, config, Stats(), 1)
    keys = [payload["vessel"]["vesselKey"], payload["berth"]["berthKey"]]
    keys += [p["portKey"] for p in payload["ports"].values()]
    with Neo4jClient(config) as client:
        client.initialize_schema()
        try:
            client.import_batch([payload, payload])
            client.import_batch([payload])
            with client.driver.session(database=config.database) as session:
                result = session.run(
                    """
                    MATCH (v:Vessel)-[r:MADE_PORT_CALL]->(pc:PortCall {recordId:$id})
                    RETURN count(pc) AS n, collect(pc.arrivalTime)[0] AS arrival,
                           collect(pc.portStayMinutes)[0] AS stay,
                           collect(pc.departureDelayMinutes)[0] AS delay
                """,
                    id=prefix,
                ).single()
                assert result["n"] == 1
                assert result["arrival"].hour == 10
                assert result["arrival"].utcoffset().total_seconds() == 32400
                assert result["stay"] == 180
                assert result["delay"] == 90
                assert (
                    session.run(
                        "MATCH (pc:PortCall {recordId:$id})-[r]->() RETURN count(r) AS n",
                        id=prefix,
                    ).single()["n"]
                    == 3
                )
            broken = deepcopy(payload)
            broken["portCall"].update(recordId=prefix + "bad", arrivalTime="invalid")
            with pytest.raises(Neo4jError):
                client.import_batch([broken])
            with client.driver.session(database=config.database) as session:
                assert (
                    session.run(
                        "MATCH (p:PortCall {recordId:$id}) RETURN count(p) AS n",
                        id=prefix + "bad",
                    ).single()["n"]
                    == 0
                )
        finally:
            with client.driver.session(database=config.database) as session:
                session.run(
                    """MATCH (n) WHERE n.recordId IN $ids OR
                    n.vesselKey IN $keys OR n.portKey IN $keys OR n.berthKey IN $keys
                    DETACH DELETE n""",
                    ids=[prefix, prefix + "bad"],
                    keys=keys,
                ).consume()
