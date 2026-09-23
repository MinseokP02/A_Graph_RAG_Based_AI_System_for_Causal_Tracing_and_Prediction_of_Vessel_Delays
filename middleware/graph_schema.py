"""Static, allowlisted schema and parameterized batch Cypher."""

VESSEL_FIELDS = {
    "callSign": "clsgn",
    "name": "vsslNm",
    "nationalityCode": "vsslNltyCd",
    "vesselTypeCode": "vsslKndCd",
    "vesselTypeName": "vsslKndNm",
}
TIME_FIELDS = {
    "arrivalTime": "etryptDt",
    "departureTime": "tkoffDt",
    "plannedDepartureTime": "tkoffPrrrnDt",
    "destinationEta": "dstnEtryptDt",
}
NUMERIC_FIELDS = {
    "loadedTon": "ldadngTon",
    "transshipmentTon": "trnpdtTon",
    "unloadedCargoTon": "landngFrghtTon",
    "loadedCargoTon": "ldFrghtTon",
}
PORT_FIELDS = {
    "AT_PORT": ("prtAgCd", "prtAgNm", "authority"),
    "FIRST_PORT": ("frstDpmprtNatPrtCd", "frstDpmprtPrtNm", "route"),
    "PREVIOUS_PORT": ("prvsDpmprtNatPrtCd", "prvsDpmprtPrtNm", "route"),
    "NEXT_PORT": ("nxlnptNatPrtCd", "nxlnptPrtNm", "route"),
    "DESTINATION_PORT": ("dstnNatPrtCd", "dstnPrtNm", "route"),
}
NODE_KEYS = {
    "Vessel": "vesselKey",
    "PortCall": "recordId",
    "Port": "portKey",
    "Berth": "berthKey",
}
CONSTRAINTS = [
    f"CREATE CONSTRAINT portmis_{label.lower()}_unique IF NOT EXISTS "
    f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
    for label, prop in NODE_KEYS.items()
]

BASE_QUERY = """
UNWIND $rows AS row
MERGE (v:Vessel {vesselKey: row.vessel.vesselKey})
ON CREATE SET v += row.vessel
MERGE (pc:PortCall {recordId: row.portCall.recordId})
SET pc += row.portCall
MERGE (v)-[:MADE_PORT_CALL]->(pc)
"""


def port_query(relation: str) -> str:
    """Only interpolate compile-time relationship names, never CSV content."""
    if relation not in PORT_FIELDS:
        raise ValueError("Unsupported port relation")
    return f"""
    UNWIND $rows AS row
    MATCH (pc:PortCall {{recordId: row.recordId}})
    MERGE (p:Port {{portKey: row.port.portKey}})
    ON CREATE SET p += row.port
    MERGE (pc)-[:{relation}]->(p)
    """


BERTH_QUERY = """
UNWIND $rows AS row
MATCH (pc:PortCall {recordId: row.recordId})
MATCH (p:Port {portKey: row.portKey})
MERGE (b:Berth {berthKey: row.berth.berthKey})
ON CREATE SET b += row.berth
MERGE (pc)-[:BERTHED_AT]->(b)
MERGE (p)-[:HAS_BERTH]->(b)
"""
