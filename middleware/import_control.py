"""Import control-event CSV: python -m middleware.import_control --csv FILE."""

import argparse
import csv
import json
from pathlib import Path

from .config import Config
from .neo4j_client import Neo4jClient
from .preprocessing import timestamp
from .utils import Stats, key, name, text

QUERY = """
UNWIND $rows AS row
MERGE (e:ControlEvent {eventId: row.eventId})
SET e += row
SET e.eventTime = CASE WHEN row.eventTime IS NULL THEN null ELSE datetime(row.eventTime) END
WITH e, row
OPTIONAL MATCH (v:Vessel)-[:MADE_PORT_CALL]->(pc:PortCall)-[:AT_PORT]->(p:Port)
WHERE v.callSign = row.callSign AND p.code = row.portCode
  AND p.codeScheme = 'authority'
  AND pc.entryYear = row.entryYear AND pc.entryCount = row.entryCount
WITH e, collect(DISTINCT pc) AS reports
FOREACH (pc IN reports | MERGE (pc)-[:HAS_CONTROL_EVENT]->(e))
RETURN count(e) AS events,
       sum(CASE WHEN size(reports) > 0 THEN 1 ELSE 0 END) AS matched,
       sum(size(reports)) AS reportLinks
"""


def prepare(path: Path, config: Config):
    """Validate the entire file before writing; retain source record identities."""
    rows = {}
    stats = Stats()
    required = {
        "record_id",
        "prtAgCd",
        "clsgn",
        "etryptYear",
        "etryptCo",
        "cntrlSe",
        "cntrlOpertDt",
        "commCo",
        "fcltyCd",
        "fcltySubCd",
    }
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or required - set(headers):
            raise ValueError("Missing or duplicate control CSV columns")
        for number, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed CSV at line {number}")
            record = text(row["record_id"])
            if not record:
                raise ValueError(f"Missing record_id at line {number}")
            event_time = timestamp(
                row["cntrlOpertDt"], config, stats, record, "cntrlOpertDt"
            )
            if text(row["cntrlOpertDt"]) and event_time is None:
                raise ValueError(f"Invalid event timestamp at line {number}")
            event = {
                "eventId": key("control-record", record),
                "recordId": record,
                "portCode": text(row["prtAgCd"]),
                "callSign": name(row["clsgn"]),
                "entryYear": text(row["etryptYear"]),
                "entryCount": text(row["etryptCo"]),
                "controlCode": text(row["cntrlSe"]),
                "controlName": text(row.get("cntrlNm")),
                "communicationCount": text(row["commCo"]),
                "eventTime": event_time.isoformat() if event_time else None,
                "facilityCode": text(row["fcltyCd"]),
                "facilitySubCode": text(row["fcltySubCd"]),
                "facilityName": text(row.get("fcltyNm")),
                "sourceFile": path.name,
                "sourceRow": number,
                "source": "CntlVssl2/Info",
            }
            rows[event["eventId"]] = event
    return list(rows.values())


def run(config, rows):
    """Idempotent event writes; unmatched events remain available for later linking."""
    summary = {"events": 0, "matched": 0, "reportLinks": 0}
    with Neo4jClient(config) as client:  # noqa: SIM117 - keep driver lifecycle explicit
        with client.driver.session(database=config.database) as session:
            session.run(
                "CREATE CONSTRAINT portmis_control_event_unique IF NOT EXISTS FOR (e:ControlEvent) REQUIRE e.eventId IS UNIQUE"
            ).consume()
            for start in range(0, len(rows), config.batch_size):

                def write(tx, start=start):
                    return dict(
                        tx.run(
                            QUERY, rows=rows[start : start + config.batch_size]
                        ).single()
                    )

                result = session.execute_write(write)
                for k in summary:
                    summary[k] += result[k]
    summary["unmatched"] = summary["events"] - summary["matched"]
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--env-file", type=Path, default=Path(__file__).with_name(".env")
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
        config = Config.from_env(args.csv, args.batch_size)
        rows = prepare(args.csv, config)
        summary = {"validatedEvents": len(rows)} if args.dry_run else run(config, rows)
        print(json.dumps(summary))
        return 0
    except Exception as exc:  # noqa: BLE001 - sanitize credential-bearing driver errors
        print(
            f"Control import failed ({type(exc).__name__}); earlier batches may remain committed."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
