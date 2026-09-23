# Vessel control collection and visit identifiers

Run from the LightRAG repository root. The collector reads `MOF_CONTROL_SERVICE_KEY`
from the root `.env` or the environment. `--env-file` selects another file.
Activate the separate MOF vessel-control API in the public data portal first.

```powershell
python collect_vessel_control.py --port 020 --start 20260801 --end 20260831
```

Output: `data/vessel_control/vessel_control_020_20260801_20260831.csv`.
Use `--call-sign 202809` to narrow the query; specify `--output` to retain multiple
subsets for the same period. Existing files require explicit `--overwrite`.
Options include `--timeout 60 --retries 3 --max-pages 10000`.

The collector uses the existing XML detail flattening, pagination consistency
checks, exact-content deduplication, bounded retries and atomic CSV publication.
No partial CSV is published on collection failure. Empty results retain headers.
API credentials and request URLs containing credentials are not logged.

Official specification:
https://www.data.go.kr/data/15006354/openapi.do

Endpoint: `https://apis.data.go.kr/1192000/CntlVssl2/Info`.
Parameters: `serviceKey`, `prtAgCd`, `sde`, `ede`, optional `clsgn`,
`pageNo`, `numOfRows` (maximum 50). No `deGb` parameter is sent.
Output preserves control type (`cntrlSe`, `cntrlNm`), event time
(`cntrlOpertDt`), communication count (`commCo`), facility identifiers and
the original visit identifiers. A control event is not automatically a delay cause.

## Restored visit identifiers

`preprocess_port_calls.py` now selects 34 columns including `etryptYear` and
`etryptCo` after `prtAgNm`. All original values, row order and `record_id` values
are preserved. Regenerate older 32-column files from the raw collector CSV:

```powershell
python preprocess_port_calls.py --input data/port_calls/port_calls_020_20260801_20260831_I.csv --overwrite
```

Facility-use requests require `prtAgCd`, `etryptYear`, `etryptCo`, `clsgn`.
Keep them as strings, including leading zeros. Compare these fields when linking
control events; missing/ambiguous identifiers must not be guessed.

The middleware importer also preserves the restored fields as `PortCall.entryYear`
and `PortCall.entryCount`. Existing report IDs and graph relationships are unchanged.
To update already stored reports, rerun the existing importer explicitly:

```powershell
python -m middleware.main --csv data/port_calls/processed/port_calls_020_20260801_20260831_I_selected.csv
```

## Import control events into Neo4j

```powershell
python -m middleware.import_control --csv data/vessel_control/vessel_control_020_20260801_20260831_refresh.csv --dry-run
python -m middleware.import_control --csv data/vessel_control/vessel_control_020_20260801_20260831_refresh.csv
```

The importer reads `middleware/.env`, validates the entire CSV before writing,
and creates unique `ControlEvent` nodes with typed `eventTime` timestamps.
`(PortCall)-[:HAS_CONTROL_EVENT]->(ControlEvent)` links use exact matches on
port authority, call sign, entry year and entry count. Multiple report rows from
the same visit can link to the same event. This does not assert a delay cause.
Unmatched events remain stored; rerun after importing missing port calls to link
them. Repeated imports use MERGE and do not duplicate events or relationships.
Each batch is atomic; earlier batches remain committed if a later batch fails.
Facility codes/names are retained on events without guessing berth equivalence.

```cypher
MATCH (pc:PortCall)-[:HAS_CONTROL_EVENT]->(e:ControlEvent)
RETURN pc, e LIMIT 100;
```

The existing LightRAG graph retriever now enriches each sampled report with
`controlEvents` (up to 10, sorted by eventTime and eventId), `controlEventCount`
(the full linked count), and `controlEventsTruncated`. Events are fetched only
for the reports selected by the existing query, in the same read transaction.
The date filter selects reports, not individual event timestamps. Events shared
by multiple reports retain their eventId and must not be counted twice.
Unlinked events are not returned by this report-based retrieval. A missing link
is not proof that no event occurred. Events cite the parent graph evidence [G#].
The existing context budget may further omit whole reports. Control observations
do not establish delay causes; no new delay metrics are inferred.
`control` is an accepted planner intent for control-history questions; event-date
and control-code filters remain unsupported.
Facility-use API collection is not implemented here.

