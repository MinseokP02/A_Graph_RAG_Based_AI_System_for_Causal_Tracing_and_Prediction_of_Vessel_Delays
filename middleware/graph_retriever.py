"""Read-only, parameterized retrieval from the existing domain graph."""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from .query_config import QueryConfig
from .query_models import QueryPlan
from .utils import name, vessel_key

MATCH = """
MATCH (v:Vessel)-[:MADE_PORT_CALL]->(pc:PortCall)
WHERE ($call_sign IS NULL OR v.callSign = $call_sign)
  AND ($vessel_name IS NULL OR v.name = $vessel_name)
  AND ($port_code IS NULL OR EXISTS {
    MATCH (pc)-[:AT_PORT]->(p:Port) WHERE p.code = $port_code })
  AND ($port_name IS NULL OR EXISTS {
    MATCH (pc)-[:AT_PORT]->(p:Port) WHERE p.name = $port_name })
  AND ($berth_name IS NULL OR EXISTS {
    MATCH (pc)-[:BERTHED_AT]->(b:Berth) WHERE b.name = $berth_name })
WITH DISTINCT v, pc,
  CASE $date_basis WHEN 'arrival' THEN pc.arrivalTime
    WHEN 'departure' THEN pc.departureTime
    ELSE coalesce(pc.arrivalTime, pc.departureTime) END AS eventTime
WHERE ($start IS NULL OR eventTime >= datetime($start))
  AND ($end IS NULL OR eventTime < datetime($end))
"""

# Keeping the optional call-sign predicate in MATCH prevents Neo4j from reliably
# selecting the unique Vessel(vesselKey) index on larger imports. Use this
# constant-only query variant when a call sign is present; input stays parameterized.
CALL_SIGN_MATCH = """
MATCH (v:Vessel {vesselKey: $vessel_key})
WHERE v.callSign = $call_sign
WITH v
MATCH (v)-[:MADE_PORT_CALL]->(pc:PortCall)
WHERE ($vessel_name IS NULL OR v.name = $vessel_name)
  AND ($port_code IS NULL OR EXISTS {
    MATCH (pc)-[:AT_PORT]->(p:Port) WHERE p.code = $port_code })
  AND ($port_name IS NULL OR EXISTS {
    MATCH (pc)-[:AT_PORT]->(p:Port) WHERE p.name = $port_name })
  AND ($berth_name IS NULL OR EXISTS {
    MATCH (pc)-[:BERTHED_AT]->(b:Berth) WHERE b.name = $berth_name })
WITH DISTINCT v, pc,
  CASE $date_basis WHEN 'arrival' THEN pc.arrivalTime
    WHEN 'departure' THEN pc.departureTime
    ELSE coalesce(pc.arrivalTime, pc.departureTime) END AS eventTime
WHERE ($start IS NULL OR eventTime >= datetime($start))
  AND ($end IS NULL OR eventTime < datetime($end))
"""
SUMMARY = (
    MATCH
    + """
RETURN count(DISTINCT pc) AS matchingRecords, count(DISTINCT v) AS vessels,
       count(pc.departureDelayMinutes) AS knownDelayRecords,
       count(pc.portStayMinutes) AS knownStayRecords,
       sum(CASE WHEN pc.isDepartureDelayed = true THEN 1 ELSE 0 END) AS delayedRecords
"""
)
DETAILS = (
    MATCH
    + """
AND ($delayed_only = false OR pc.isDepartureDelayed = true)
RETURN v.callSign AS callSign, v.name AS vesselName,
       pc.recordId AS recordId, toString(pc.arrivalTime) AS arrivalTime,
       toString(pc.departureTime) AS departureTime,
       toString(pc.plannedDepartureTime) AS plannedDepartureTime,
       pc.portStayMinutes AS portStayMinutes, pc.departureDelayMinutes AS departureDelayMinutes,
       pc.isDepartureDelayed AS isDepartureDelayed, pc.cargoOperationTon AS cargoOperationTon,
       pc.purpose AS purpose, pc.sourceFile AS sourceFile, pc.sourceRow AS sourceRow,
       [(pc)-[:AT_PORT]->(p:Port) | p{.code, .name}] AS ports,
       [(pc)-[:FIRST_PORT]->(p:Port) | p{.code, .name}] AS firstPorts,
       [(pc)-[:PREVIOUS_PORT]->(p:Port) | p{.code, .name}] AS previousPorts,
       [(pc)-[:NEXT_PORT]->(p:Port) | p{.code, .name}] AS nextPorts,
       [(pc)-[:DESTINATION_PORT]->(p:Port) | p{.code, .name}] AS destinations,
       [(pc)-[:BERTHED_AT]->(b:Berth) | b{.code, .subCode, .name}] AS berths,
       [(pc)-[:HAS_DEPARTURE_WEATHER]->(w:WeatherObservation) |
         w{weatherTime: toString(w.observedAt), .temperatureC, .windSpeedMs,
           .rainMm, .portCode, .source, .datasetType}] AS departureWeather
ORDER BY eventTime, pc.recordId
LIMIT $limit
"""
)


def retrieval_queries(plan: QueryPlan) -> tuple[str, str]:
    """Choose an index-friendly, constant query template for exact call signs."""
    match = CALL_SIGN_MATCH if plan.call_sign else MATCH
    summary_tail = SUMMARY[len(MATCH) :]
    details_tail = DETAILS[len(MATCH) :]
    return match + summary_tail, match + details_tail


CONTROL_EVENTS = """
UNWIND $record_ids AS recordId
MATCH (pc:PortCall {recordId: recordId})
CALL {
    WITH pc
    MATCH (pc)-[:HAS_CONTROL_EVENT]->(e:ControlEvent)
    RETURN count(DISTINCT e) AS total
}
CALL {
    WITH pc
    MATCH (pc)-[:HAS_CONTROL_EVENT]->(e:ControlEvent)
    WITH DISTINCT e ORDER BY e.eventTime, e.eventId LIMIT $event_limit
    RETURN collect(e{.eventId, .controlCode, .controlName,
        eventTime: toString(e.eventTime), .facilityCode, .facilitySubCode,
        .facilityName, .communicationCount, .sourceFile, .sourceRow}) AS events
}
RETURN recordId, total, events
"""


def parameters(plan: QueryPlan, limit: int, timezone: str) -> dict:
    """Convert inclusive calendar dates to timezone-aware half-open bounds."""
    zone = ZoneInfo(timezone)
    return {
        **{
            k: name(getattr(plan, k))
            for k in (
                "call_sign",
                "vessel_name",
                "port_code",
                "port_name",
                "berth_name",
            )
        },
        "date_basis": plan.date_basis,
        "start": datetime.combine(plan.start_date, time.min, zone).isoformat()
        if plan.start_date
        else None,
        "end": datetime.combine(
            plan.end_date + timedelta(days=1), time.min, zone
        ).isoformat()
        if plan.end_date
        else None,
        "delayed_only": plan.intent == "delays",
        "limit": limit,
        "vessel_key": vessel_key(plan.call_sign, None),
    }


class GraphRetriever:
    """Use an official async driver and read-mode session; no schema mutations."""

    def __init__(self, driver, config: QueryConfig) -> None:
        """Inject the managed driver for lifecycle control and testing."""
        self.driver, self.config = driver, config

    async def retrieve(self, plan: QueryPlan, limit: int) -> dict:
        """Return counts over the filter and a bounded, deterministic record sample."""
        params = parameters(plan, limit, self.config.timezone)
        summary_query, details_query = retrieval_queries(plan)
        async with self.driver.session(
            database=self.config.neo4j_database, default_access_mode="READ"
        ) as session:

            async def read(tx):
                """Run both read queries in one managed transaction."""
                summary_result = await tx.run(summary_query, **params)
                summary = dict(await summary_result.single())
                detail_result = await tx.run(details_query, **params)
                details = await detail_result.data()
                if details:
                    event_result = await tx.run(
                        CONTROL_EVENTS,
                        record_ids=[r["recordId"] for r in details],
                        event_limit=10,
                    )
                    events_by_report = {
                        r["recordId"]: r for r in await event_result.data()
                    }
                    for record in details:
                        event_data = events_by_report.get(record["recordId"], {})
                        record["controlEvents"] = event_data.get("events", [])
                        record["controlEventCount"] = event_data.get("total", 0)
                        record["controlEventsTruncated"] = record[
                            "controlEventCount"
                        ] > len(record["controlEvents"])
                return {"summary": summary, "records": details}

            # Managed transaction timeout covers both queries, including scanning.
            from neo4j import unit_of_work

            return await session.execute_read(unit_of_work(timeout=20)(read))
