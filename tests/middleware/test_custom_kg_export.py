"""Custom KG conversion preserves identities, evidence and parallel relations."""

from middleware.export_custom_kg import convert


def test_parallel_relations_preserved_with_grounded_sources():
    nodes = [
        {"id": "1", "label": "PortCall", "props": {"recordId": "report"}},
        {"id": "2", "label": "Port", "props": {"portKey": "port", "code": "020"}},
    ]
    relations = [
        {"src": "1", "dst": "2", "kind": "AT_PORT"},
        {"src": "1", "dst": "2", "kind": "DESTINATION_PORT"},
    ]
    payload = convert(nodes, relations)
    assert len(payload["relationships"]) == 1
    relation = payload["relationships"][0]
    assert "AT_PORT" in relation["description"]
    assert "DESTINATION_PORT" in relation["description"]
    sources = {c["source_id"] for c in payload["chunks"]}
    assert all(e["source_id"] in sources for e in payload["entities"])
    assert relation["source_id"] in sources
    assert payload == convert(nodes, relations)
