# MOF port-call collector

After importing the graph, use the [Graph + Qwen + LightRAG question API](../middleware/QueryAPI.md)
to retrieve graph facts and document context through `POST /ask`.

For typed Neo4j graph ingestion of the selected 32-column CSV, see
[Port-MIS middleware](../middleware/README.md). It includes dry-run, batch
import, derived metrics and a four-label domain graph schema.

## Select columns for graph preparation

From the `LightRAG` directory:

```powershell
python .\preprocess_port_calls.py --input .\data\port_calls\port_calls_020_20260801_20260831_I.csv
```

The output is `data/port_calls/processed/<input_stem>_selected.csv`.
Use `--output` for another destination and `--overwrite` to replace an existing
selected file. The source cannot be overwritten.

The script selects exactly the 32 columns in `OUTPUT_COLUMNS`, in that order.
All source rows and values are preserved, including blank cells, leading zeros,
whitespace, dates and record IDs. No derived graph IDs or quality flags are
added. Missing requested columns or malformed records stop processing.
The earlier `_graph.csv` output is retained separately; it is not regenerated
by this version. The previous `--as-of` option is no longer used.

This is column selection only; it does not create a Neo4j graph or enforce
node identities when uploaded as ordinary text to LightRAG.

`collect_port_calls.py` retrieves the Ministry of Oceans and Fisheries vessel
arrival/departure reports and exports UTF-8-BOM CSV using Python's standard
library. Python 3.10+ is required. No Ollama, LightRAG server, or extra package
is required.

## Official contract

Verified on 2026-09-11 against the embedded Swagger and attached
`DAT211_오픈API활용가이드_선박운항정보.docx` on the
[official dataset page](https://www.data.go.kr/data/15006353/openapi.do).

- Endpoint: `https://apis.data.go.kr/1192000/VsslEtrynd5/Info5`
- Required: `serviceKey`, `prtAgCd`, `sde`, `ede`.
- Dates: `YYYYMMDD`; CLI also accepts `YYYY-MM-DD`.
- `deGb`: `I` for arrival date, `O` for departure date.
- Optional call sign: `clsgn`.
- Pagination: `pageNo`, `numOfRows` (maximum 50).
- Busan authority code: `020`, as shown in the official guide.
- Response: XML, with `items/item/details/detail` nesting.

This is declaration/report data, not live AIS coordinates. Historical coverage
and maximum supported query interval are not assumed by the collector.

## Run in PowerShell

Activate this API through the portal, then add the following to `LightRAG/.env`:

```dotenv
MOF_SERVICE_KEY=your_service_key
```

The decoded key is preferred; a URL-encoded key is also accepted. An existing
`MOF_SERVICE_KEY` environment variable takes precedence. The collector never
prints the key, request URL, or response error body.

From the `LightRAG` directory:

```powershell
python .\collect_port_calls.py --port 020 --start 2026-08-01 --end 2026-08-31
```

Restrict to a call sign (replace `YOUR_CALL_SIGN` with a real one):

```powershell
python .\collect_port_calls.py --port 020 --start 2026-08-01 --end 2026-08-31 --call-sign YOUR_CALL_SIGN --output .\data\port_calls\selected_ship.csv
```

Query by departure date:

```powershell
python .\collect_port_calls.py --port 020 --start 2026-08-01 --end 2026-08-31 --date-basis O
```

The default destination is
`data/port_calls/port_calls_020_20260801_20260831_I.csv`, relative to the script.
It is deliberately outside the LightRAG input scanner. Each invocation produces
a complete snapshot; existing files require `--overwrite` or another output path.
The script does not append to an existing CSV. Use distinct paths for different
call-sign filters. A successful zero-result query produces a header-only CSV.

## Data and failure behavior

Requests print the current page and attempt before waiting. Use `--timeout 60`
to change the socket timeout and `--retries 1` to limit retries. The timeout
is per socket operation, not a total collection deadline. Ctrl+C stops cleanly
with exit code 130. To diagnose a slow query, first try one day with
`--page-size 10 --timeout 60 --retries 1`.

- Every nested detail becomes a separate CSV row with the parent vessel/port
  fields copied onto it. Pagination counts parent reports, not flattened rows.
- Original API field names and string values are preserved, including leading
  zeros. In spreadsheet software import identifier columns as text.
- Unknown additional fields are included as columns. No inferred MMSI or
  timestamp timezone conversion is added.
- `record_id` is a SHA-256 hash of the entire flattened row. Exact duplicates
  within a run are removed. It is a content identity, **not** a permanent vessel
  or port-call identity; amended declarations remain distinct records.
- Transient HTTP/network failures are retried. API errors, repeated pages,
  changed totals, premature empty pages, and page-limit exhaustion stop the run.
  No partial result replaces the destination. Exit status is 1 on failure.
- A live service can change records without changing `totalCount`; pagination
  cannot guarantee a transactional snapshot. Re-query a narrower historical
  period when consistency is important.
- Large queries are accumulated in memory. Narrow the period to bound memory
  and API traffic; `--max-pages` defaults to 10000.

This collector only fetches and exports reports. Neo4j insertion and LightRAG
vector/provenance updates are separate integration work. For later graph use,
retain `prtAgCd`, `clsgn`, `etryptYear`, `etryptCo`, and detail fields; validate
their identity semantics before deciding an upsert key. Do not merge a call
sign directly with an AIS MMSI.

## Offline verification

```powershell
python -m unittest discover -s tests/collectors -p "test_port_calls.py" -v
```

Tests use synthetic API-shaped XML and mocked HTTP calls. Authenticated live
retrieval must be verified with an approved service key.
