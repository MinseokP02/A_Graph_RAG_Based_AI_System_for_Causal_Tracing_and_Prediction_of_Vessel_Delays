# Internal LightRAG integration

The existing LightRAG chat endpoint now supports this optional flow:

`WebUI -> aquery_llm -> Neo4j read + aquery_data -> configured query model -> response`

No middleware HTTP server or call to `/query/data` is required. Neo4j is read
through its Bolt driver. The existing model binding still calls Ollama (or the
configured provider). This does not import the domain graph into LightRAG's
entity schema or create graph embeddings.

## Docker

Keep the existing Neo4j credentials in `middleware/.env`. From the LightRAG root:

```powershell
docker compose -f docker-compose.yml -f docker-compose.portmis.yml up -d --no-build
```

Open `http://localhost:9621` and use a retrieval mode such as `mix` or `naive`.
The override mounts the local integration code and middleware read-only, and
uses `host.docker.internal` for Neo4j. `middleware.internal_server` attaches the
hook to the image's existing class without replacing its core source files.
The image must support `aquery_llm`, `aquery_data`, role-based query models and
include the Neo4j Python driver. No port 9622 is required.
To disable, start with only the base compose file again.

## Local Python server

Install `middleware/requirements.txt` into the LightRAG environment. Add to the
root `.env` (not middleware/.env):

```dotenv
PORTMIS_GRAPH_ENABLED=true
PORTMIS_ENV_FILE=middleware/.env
```

Start the usual LightRAG server from the repository root. Optional
`PORTMIS_NEO4J_URI`, `PORTMIS_NEO4J_USERNAME`, `PORTMIS_NEO4J_PASSWORD` and
`PORTMIS_NEO4J_DATABASE` override the middleware file without changing LightRAG's
own graph storage configuration.

## Behavior and limits

- All retrieval-mode chat requests use the planner when enabled. `bypass` skips
  retrieval. Plain `aquery_data` retains its existing document retrieval behavior.
- Fixed parameterized Cypher returns a filter summary and at most 10 records.
  Unsupported or ambiguous filters produce a clarification in the answer context.
- Existing document chunks are retrieved internally in the selected mode. Empty
  document results still allow graph answers. Failures are stated in context.
- The final answer uses the existing query model, history and streaming support.
  Evidence is included in the model input so graph updates change the cache input.
- Context includes whole evidence items within a token budget, with 2000 tokens
  reserved for instructions. Oversized history/custom prompts still require the
  usual model context configuration.
- Structured evidence and warnings are available in `aquery_llm` metadata under
  `portmis` and `warnings`; individual HTTP routes may omit custom metadata.
- Graph citations use `[G1]` and `[GRAPH SUMMARY]`. A report is not necessarily a
  unique visit. Missing delay/stay values are unknown; this is evidence retrieval,
  not a trained delay predictor or proof of delay causes.
