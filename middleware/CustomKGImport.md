# Neo4j to LightRAG custom KG

`python -m middleware.export_custom_kg` reads the domain Neo4j database using
`middleware/.env` and writes `data/portmis_custom_kg.json`. It includes Vessel,
PortCall, Port, Berth and ControlEvent nodes and the allowlisted relationships.
The export does not modify Neo4j.

Stable names are prefixed with `PORTMIS:` to avoid collisions with extracted
document entities. Node properties become descriptions; each description and
relationship has a supporting chunk. Parallel relationships between the same
pair are combined, with original directions and types retained in the description,
because LightRAG's graph backend may store a single undirected edge per pair.

Stop the normal LightRAG server and back up `data/rag_storage` before importing.
Copy the export to `data/inputs/portmis_custom_kg.json`, then run:

```powershell
docker compose -f docker-compose.yml -f docker-compose.portmis.yml run --rm --no-deps lightrag python -m middleware.load_custom_kg
docker compose -f docker-compose.yml -f docker-compose.portmis.yml up -d --no-build
```

The loader constructs the server's configured LightRAG instance, initializes its
storages through the application lifespan, and calls `ainsert_custom_kg` with
the snapshot. The configured embedding model generates chunk, entity and relation
vectors; LLM entity extraction is not needed. Import may take considerable time.
`portmis_custom_kg_import.json` in rag_storage is written only after completion.

The import is a snapshot, not automatic synchronization. Reimports can update
matching identities but do not remove deleted Neo4j records. Keep the backup for
recovery; the custom KG path does not provide the normal document-upload lifecycle.
Do not submit the exported JSON through document scanning or upload: it has already
been imported through the custom KG method.

The existing internal Port-MIS hook remains enabled. `mix` searches the imported
LightRAG graph and chunks, while the direct Neo4j query still supplies exact facts.
`naive` searches chunks only. Graph edges here describe recorded associations,
not established delay causes.
