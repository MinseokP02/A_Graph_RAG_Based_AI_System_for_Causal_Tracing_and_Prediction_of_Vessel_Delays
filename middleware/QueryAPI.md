# Neo4j + Qwen + LightRAG question API

The query service runs beside the existing importer and LightRAG server.
It reads the domain graph already loaded into Neo4j; it does not rebuild,
re-embed, or migrate that graph.

## Start

From `LightRAG`, use the same interpreter for installation and execution:

```powershell
python -m pip install -r middleware/requirements.txt
python -m middleware.query_api
```

If dependencies are already installed in the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m middleware.query_api
```

Open <http://localhost:9622/docs>. Expand `POST /ask`, click **Try it out**,
paste one of the request examples below and click **Execute**.
The existing LightRAG UI on port 9621 is unchanged; its chat does not
automatically call this new API. Use `/ask` or integrate it into your client.

The default middleware environment file is `middleware/.env`. Keep the working
Neo4j credentials and add these optional settings (defaults shown):

```dotenv
OLLAMA_URL=http://localhost:11434
QWEN_MODEL=qwen2.5:7b-instruct
LIGHTRAG_URL=http://localhost:9621
LIGHTRAG_QUERY_API_KEY=
MIDDLEWARE_HOST=127.0.0.1
MIDDLEWARE_PORT=9622
MIDDLEWARE_API_KEY=
QUERY_TIMEOUT=120
GRAPH_CONTEXT_CHARS=16000
DOCUMENT_CONTEXT_CHARS=8000
```

`SOURCE_TIMEZONE` continues to default to `Asia/Seoul`. Query configuration
does not require `CSV_PATH`. Override the dotenv location using
`MIDDLEWARE_ENV_FILE`; actual process environment variables take precedence.
No credentials are copied automatically from the main LightRAG `.env`.
If LightRAG requires an API key, put that key in `LIGHTRAG_QUERY_API_KEY`.
It is sent only to the configured LightRAG server as `X-API-Key`.
If `MIDDLEWARE_API_KEY` is configured, callers must send the corresponding
`X-API-Key` header to `/ask`. The API binds to loopback by default.

Neo4j and Ollama must be running. For stage 2, LightRAG must also be running
and documents must already have been ingested and embedded there. Only the
Neo4j domain graph is insufficient to supply LightRAG document context.

## Stage 1: Graph retrieval and Qwen answer

```json
{
  "question": "호출부호 202809 선박의 2026년 8월 입출항과 사용 계선시설을 알려줘.",
  "plan": {
    "intent": "berths",
    "call_sign": "202809",
    "start_date": "2026-08-01",
    "end_date": "2026-08-31",
    "date_basis": "event"
  },
  "use_documents": false,
  "max_records": 10
}
```

The explicit `plan` bypasses the planning model call. When `plan` is omitted,
Qwen extracts it with a validated JSON schema before retrieval:

```json
{
  "question": "호출부호 202809 선박의 2026년 8월 입출항 정보를 알려줘.",
  "use_documents": false
}
```

The common fully matched form `호출부호 … 선박의 YYYY년 M월 입출항 정보를 알려줘`
(including the supplied berth example) is parsed deterministically without a
planning model call. Other questions use Qwen's constrained JSON planner.
The extracted plan is returned so callers can verify/correct it. An ambiguous
question (e.g. a month without a year) should yield `clarification_required`.
Malformed model output fails with HTTP 502 instead of silently querying the
entire graph. Model extraction can still misinterpret a question; use an
explicit plan for reproducible filtering. Explicit plans are authoritative
even if their filters differ from the natural-language question.

Supported intents: `visits`, `routes`, `berths`, `delays`, `cargo`, `count`.
All use a bounded record retrieval template; `delays` additionally filters
detail rows to `isDepartureDelayed=true`. The summary always describes all
records matching the other filters, including unknown delay values. `count`
uses the returned summary, not a count of the bounded sample.

Supported exact-match filters: `call_sign`, `vessel_name`, `port_code`,
`port_name`, `berth_name`, `start_date`, `end_date`, `date_basis`.
Names and call signs are normalized as in import. Port filters refer to
`AT_PORT`, not next/previous ports. No fuzzy search or arbitrary Cypher is
accepted. Unsupported filters/aggregations/predictions should request
clarification rather than produce an unfiltered result.

Dates are inclusive calendar days in `SOURCE_TIMEZONE`; SQL-style half-open
bounds include the whole end date. `event` means arrival time if present,
otherwise departure time **on that same declaration row**. `arrival` or
`departure` searches only that field and excludes rows with missing values
when a date bound is supplied. No cross-row arrival/departure join is performed.

## Stage 2: Include LightRAG document context

Set `use_documents` to `true` (the default). The request first retrieves graph
facts, then calls LightRAG's **`POST /query/data`** with `mode=naive`,
`chunk_top_k=5`, reranking disabled and a bounded retrieval context.
It takes original text chunks and their source paths/IDs, not a second LLM
answer. Both evidence sets are passed to Qwen in the final prompt.

```powershell
$body = Get-Content middleware/examples/ask_with_documents.json -Raw -Encoding UTF8
Invoke-RestMethod -Uri http://localhost:9622/ask -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Use `middleware/examples/ask_graph.json` for graph-only mode. Add
`-Headers @{ 'X-API-Key' = 'your_middleware_key' }` if enabled.
Responses include:

- `status`: `ok` or `clarification_required`.
- `answer`: Qwen's Korean answer, requested to cite `[G1]`, `[D1]`, `[SUMMARY]`.
- `plan`: the actual filters used.
- `graph.summary`: matching declaration rows, vessels, known delay/stay and
  delayed counts across those filters.
- `evidence`: the exact graph records/document excerpts supplied to Qwen,
  including record IDs, source files, row numbers or document chunk IDs.
- `warnings`: sampling, missing delay values, context truncation or optional
  document retrieval failure.

Neo4j or Qwen failure returns HTTP 502. LightRAG failure allows a graph-only
answer with an explicit warning. HTTP 422 means invalid input; 401 means an
invalid middleware API key. `/health` is **liveness only**, not proof of
upstream availability. Questions can take time while Ollama loads the model.

## Stage 3: Future graph-description embeddings

No graph embedding job runs now. `RagService` accepts optional
`additional_retrievers`, each implementing async `retrieve(question)` and
returning dictionaries with `content`, `source`, and a `kind` such as
`graph_description`. Their evidence passes through the same document budget
and final answer pipeline. This injection point is tested.

To enable later: generate descriptions with stable graph IDs, embed them
using the selected embedding model, maintain an index and source/version
metadata, then implement this retrieval interface and pass it when constructing
`RagService` in the API lifespan. Exact dates/counts should continue to come
from Neo4j. Changing embedding models requires rebuilding that future index.

## Boundaries

- Only fixed read queries execute. User/model text is passed as parameters;
  it cannot become a label, relationship name or Cypher fragment. A dedicated
  DB read-only account is recommended for deployment.
- A PortCall remains one declaration row. Unknown times/delays remain unknown,
  not zero. The current dataset cannot produce row-level stay/delay metrics
  because the required timestamps are split between separate rows.
- Graph records are sampled (maximum 50); counts come from a separate aggregate
  over the full filter. Sampling does not establish a population trend, rank,
  average or correlation. The two queries share a managed transaction but
  concurrent updates are not a guaranteed immutable historical snapshot.
- Context budgets count characters, not exact model tokens. Truncated evidence
  is marked. Qwen uses `num_ctx=16384` and bounded output; model responses remain
  probabilistic and citations are requested, not independently fact-verified.
- Source content is explicitly treated as untrusted evidence. General document
  causes do not establish causation for a particular ship. Prediction, AIS
  joining, automatic entity resolution and causal models are not implemented.
- The endpoint is nonstreaming and separate from LightRAG's built-in chat API.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -c tests/middleware/pytest.ini tests/middleware/test_query_service.py --confcutdir=tests/middleware -q
```

Tests cover stages 1/2/3 injection, date bounds, constrained planning, API auth,
input limits, graph failure, document fallback, context budgets and HTTP contracts.

Validation: the new query-service suite passed 12 tests, and Ruff passed.
A live in-process `/ask` request returned HTTP 200
using the existing local Neo4j, Qwen and LightRAG: call sign 202809 matched
10 declaration rows; the bounded prompt contained two graph records and two
document chunks. The request returned Korean text and citations. The model
can still make narrative errors; always inspect the returned structured evidence.
No graph data was written during query verification. TestClient emitted one
upstream httpx deprecation warning; it did not affect the tests.

Sources: [Ollama chat API](https://docs.ollama.com/api/chat),
[Neo4j managed transactions](https://neo4j.com/docs/python-manual/current/transactions/).
LightRAG's request/response contract was checked directly in this repository's
`lightrag/api/routers/query_routes.py`.
