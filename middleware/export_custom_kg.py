"""Export the domain graph into a grounded LightRAG custom-KG payload."""

import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from .config import Config
from .neo4j_client import Neo4jClient

KEYS = {
    "Vessel": "vesselKey",
    "PortCall": "recordId",
    "Port": "portKey",
    "Berth": "berthKey",
    "ControlEvent": "eventId",
}
RELATIONS = [
    "MADE_PORT_CALL",
    "AT_PORT",
    "FIRST_PORT",
    "PREVIOUS_PORT",
    "NEXT_PORT",
    "DESTINATION_PORT",
    "BERTHED_AT",
    "HAS_BERTH",
    "HAS_CONTROL_EVENT",
]


def convert(nodes, relations):
    result = {"chunks": [], "entities": [], "relationships": []}
    names = {}
    for node in nodes:
        label, props = node["label"], dict(node["props"])
        identity = props.get(KEYS[label])
        if not identity:
            raise ValueError("Domain node lacks a stable identifier")
        digest = hashlib.sha256(f"{label}:{identity}".encode()).hexdigest()
        entity = f"PORTMIS:{label}:{identity}"
        names[node["id"]] = entity
        source = "portmis-node-" + digest
        if props.get("sourceFile"):
            props["sourceFile"] = (
                str(props["sourceFile"]).replace("\\", "/").split("/")[-1]
            )
        description = f"PORT-MIS {label}: " + json.dumps(
            props, ensure_ascii=False, default=str, sort_keys=True
        )
        result["chunks"].append(
            {
                "content": description,
                "source_id": source,
                "file_path": "neo4j-portmis-export.json",
            }
        )
        result["entities"].append(
            {
                "entity_name": entity,
                "entity_type": label,
                "description": description,
                "source_id": source,
            }
        )
    grouped = {}
    for relation in relations:
        pair = tuple(sorted((relation["src"], relation["dst"])))
        grouped.setdefault(pair, []).append(relation)
    for group in grouped.values():
        relation = group[0]
        src, dst = names[relation["src"]], names[relation["dst"]]
        kind = ",".join(sorted({r["kind"] for r in group}))
        description = "\n".join(
            f"{names[r['src']]} --{r['kind']}--> {names[r['dst']]}" for r in group
        ) + ". Recorded relationships; not proof of delay causality."
        source = "portmis-rel-" + hashlib.sha256(description.encode()).hexdigest()
        result["chunks"].append(
            {
                "content": description,
                "source_id": source,
                "file_path": "neo4j-portmis-export.json",
            }
        )
        result["relationships"].append(
            {
                "src_id": src,
                "tgt_id": dst,
                "description": description,
                "keywords": kind,
                "weight": 1.0,
                "source_id": source,
            }
        )
    return result


def main():
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    config = Config.from_env(Path("unused.csv"))
    with Neo4jClient(config) as client:  # noqa: SIM117 - explicit driver lifecycle
        with client.driver.session(
            database=config.database, default_access_mode="READ"
        ) as session:

            def read(tx):
                nodes = tx.run(
                    "MATCH (n) UNWIND labels(n) AS label WITH n,label WHERE label IN $labels RETURN elementId(n) AS id,label,properties(n) AS props",
                    labels=list(KEYS),
                ).data()
                ids = [n["id"] for n in nodes]
                relations = tx.run(
                    "MATCH (a)-[r]->(b) WHERE elementId(a) IN $ids AND elementId(b) IN $ids AND type(r) IN $types RETURN DISTINCT elementId(a) AS src,elementId(b) AS dst,type(r) AS kind",
                    ids=ids,
                    types=RELATIONS,
                ).data()
                return convert(nodes, relations)

            payload = session.execute_read(read)
    path = Path("data/portmis_custom_kg.json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps({k: len(v) for k, v in payload.items()}))


if __name__ == "__main__":
    main()
