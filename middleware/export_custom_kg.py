"""Export the domain graph into a grounded LightRAG custom-KG payload."""

import argparse
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
    "WeatherObservation": "observationKey",
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
    "HAS_DEPARTURE_WEATHER",
    "OBSERVED_AT",
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
        description = (
            "\n".join(
                f"{names[r['src']]} --{r['kind']}--> {names[r['dst']]}" for r in group
            )
            + ". Recorded relationships; not proof of delay causality."
        )
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


def weather_delta(payload):
    """Keep new weather entities/relations while referring to existing graph endpoints."""
    prefix = "PORTMIS:WeatherObservation:"
    entities = [e for e in payload["entities"] if e["entity_name"].startswith(prefix)]
    relationships = [
        r
        for r in payload["relationships"]
        if r["src_id"].startswith(prefix) or r["tgt_id"].startswith(prefix)
    ]
    source_ids = {e["source_id"] for e in entities} | {
        r["source_id"] for r in relationships
    }
    return {
        "chunks": [c for c in payload["chunks"] if c["source_id"] in source_ids],
        "entities": entities,
        "relationships": relationships,
    }


def entity_neighborhood(payload, root):
    """Export one entity and its one-hop facts for a small incremental refresh."""
    relationships = [
        r for r in payload["relationships"] if root in (r["src_id"], r["tgt_id"])
    ]
    names = {root}
    for relation in relationships:
        names.update((relation["src_id"], relation["tgt_id"]))
    entities = [e for e in payload["entities"] if e["entity_name"] in names]
    source_ids = {e["source_id"] for e in entities} | {
        r["source_id"] for r in relationships
    }
    return {
        "chunks": [c for c in payload["chunks"] if c["source_id"] in source_ids],
        "entities": entities,
        "relationships": relationships,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather-only", action="store_true")
    parser.add_argument("--root-entity")
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    config = Config.from_env(Path("unused.csv"))
    with Neo4jClient(config) as client:  # noqa: SIM117 - explicit driver lifecycle
        with client.driver.session(
            database=config.database, default_access_mode="READ"
        ) as session:

            def read(tx):
                nodes = tx.run(
                    "MATCH (n) UNWIND labels(n) AS label WITH n,label "
                    "WHERE label IN $labels AND "
                    "(label <> 'WeatherObservation' OR "
                    "EXISTS { MATCH (:PortCall)-[:HAS_DEPARTURE_WEATHER]->(n) }) "
                    "RETURN elementId(n) AS id,label,properties(n) AS props",
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
    if args.weather_only:
        payload = weather_delta(payload)
    elif args.root_entity:
        payload = entity_neighborhood(payload, args.root_entity)
    path = Path(
        "data/portmis_weather_custom_kg.json"
        if args.weather_only
        else "data/portmis_entity_custom_kg.json"
        if args.root_entity
        else "data/portmis_custom_kg.json"
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps({k: len(v) for k, v in payload.items()}))


if __name__ == "__main__":
    main()
