"""Official driver lifecycle and atomic batch writes."""

from typing import Any

from .config import Config
from .graph_schema import (
    BASE_QUERY,
    BERTH_QUERY,
    CONSTRAINTS,
    PORT_FIELDS,
    TIME_FIELDS,
    port_query,
)


def write_batch(tx: Any, rows: list[dict]) -> None:
    """Write every entity/relation in a batch within one managed transaction."""
    time_sets = "\n".join(
        f"SET pc.{prop} = CASE WHEN row.portCall.{prop} IS NULL THEN null "
        f"ELSE datetime(row.portCall.{prop}) END"
        for prop in TIME_FIELDS
    )
    tx.run(BASE_QUERY + time_sets, rows=rows).consume()
    for relation in PORT_FIELDS:
        links = [
            {"recordId": r["portCall"]["recordId"], "port": r["ports"][relation]}
            for r in rows
            if relation in r["ports"]
        ]
        if links:
            tx.run(port_query(relation), rows=links).consume()
    berths = [
        {
            "recordId": r["portCall"]["recordId"],
            "berth": r["berth"],
            "portKey": r["ports"]["AT_PORT"]["portKey"],
        }
        for r in rows
        if r["berth"]
    ]
    if berths:
        tx.run(BERTH_QUERY, rows=berths).consume()


class Neo4jClient:
    """No connection is created until explicitly entering the context manager."""

    def __init__(self, config: Config) -> None:
        """Keep settings and an initially absent driver."""
        self.config = config
        self.driver = None

    def __enter__(self) -> "Neo4jClient":  # noqa: PYI034 -- retain Python 3.10 support without typing_extensions
        """Connect and verify authentication without exposing credentials in errors."""
        from neo4j import GraphDatabase

        if not self.config.password:
            raise ValueError("NEO4J_PASSWORD is required for import")
        self.driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.username, self.config.password),
            connection_timeout=15,
            max_transaction_retry_time=30,
        )
        try:
            self.driver.verify_connectivity()
        except Exception:
            self.driver.close()
            raise
        return self

    def initialize_schema(self) -> None:
        """Create four uniqueness constraints, also supplying their backing indexes."""
        with self.driver.session(database=self.config.database) as session:
            for query in CONSTRAINTS:
                session.run(query).consume()

    def import_batch(self, rows: list[dict]) -> None:
        """Commit or roll back the whole batch; driver handles transient retries."""
        with self.driver.session(database=self.config.database) as session:
            session.execute_write(write_batch, rows)

    def __exit__(self, *_: object) -> None:
        """Release sockets even when an import fails."""
        if self.driver:
            self.driver.close()
