"""Control evidence enrichment preserves report cardinality and sampling metadata."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from middleware.graph_retriever import CONTROL_EVENTS, GraphRetriever
from middleware.query_config import QueryConfig
from middleware.query_models import QueryPlan


def test_enrich_reports_with_bounded_control_events():
    summary = AsyncMock()
    summary.single.return_value = {"matchingRecords": 2}
    details = AsyncMock()
    details.data.return_value = [{"recordId": "r1"}, {"recordId": "r2"}]
    events = AsyncMock()
    events.data.return_value = [
        {"recordId": "r1", "total": 12, "events": [{"eventId": "e1"}]}
    ]
    tx = AsyncMock()
    tx.run.side_effect = [summary, details, events]
    session = AsyncMock()
    async def execute_read(callback):
        return await callback(tx)

    session.execute_read.side_effect = execute_read
    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=False)
    result = asyncio.run(
        GraphRetriever(driver, QueryConfig()).retrieve(QueryPlan(intent="control"), 2)
    )
    assert len(result["records"]) == 2
    assert result["records"][0]["controlEventsTruncated"]
    assert result["records"][1]["controlEvents"] == []
    assert result["records"][1]["controlEventCount"] == 0
    tx.run.assert_awaited_with(CONTROL_EVENTS, record_ids=["r1", "r2"], event_limit=10)
