"""Collect MOF VsslEtrynd5/Info5 port-call reports using only the stdlib."""

import argparse
import csv
from datetime import datetime
import hashlib
from functools import partial
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET


ENDPOINT = "https://apis.data.go.kr/1192000/VsslEtrynd5/Info5"
KEY_ENV = "MOF_SERVICE_KEY"
BASE_FIELDS = [
    "record_id", "prtAgCd", "prtAgNm", "etryptYear", "etryptCo", "clsgn",
    "vsslNm", "vsslNltyCd", "vsslNltyNm", "vsslKndCd", "vsslKndNm",
    "etryptPurpsCd", "etryptPurpsNm", "frstDpmprtNatPrtCd", "frstDpmprtPrtNm",
    "prvsDpmprtNatPrtCd", "prvsDpmprtPrtNm", "nxlnptNatPrtCd", "nxlnptPrtNm",
    "dstnNatPrtCd", "dstnPrtNm", "reqstSeNm", "etryndNm", "etryptDt",
    "tkoffDt", "ibobprtNm", "laidupFcltyCd", "laidupFcltySubCd",
    "laidupFcltyNm", "tugYn", "piltgYn", "ldadngFrghtClCd", "ldadngTon",
    "trnpdtTon", "landngFrghtTon", "ldFrghtTon", "grtg", "intrlGrtg",
    "satmntEntrpsNm", "crewCo", "koranCrewCo", "frgnrCrewCo", "mrNum",
    "tkoffPrrrnDt", "dstnEtryptDt",
]


class CollectionError(Exception):
    """Safe-to-display failure without a request URL or service key."""


def date_arg(value):
    compact = value.replace("-", "")
    if not re.fullmatch(r"\d{8}", compact):
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD or YYYYMMDD.")
    try:
        datetime.strptime(compact, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid calendar date.") from exc
    return compact


def load_key(env_file, key_env=KEY_ENV):
    """Environment wins; read only this key from an optional dotenv file."""
    value = os.environ.get(key_env, "").strip()
    if not value and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, sep, candidate = line.strip().removeprefix("export ").partition("=")
            if sep and name.strip() == key_env:
                value = candidate.strip().strip("\"'")
    if not value:
        raise CollectionError(f"Set {key_env} in the environment or --env-file.")
    # unquote (not unquote_plus) preserves '+' in decoded service keys.
    return unquote(value)


def parse_page(payload):
    """Return flattened detail rows, parent item count, total and page number."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise CollectionError("API returned invalid XML.") from exc
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    gateway = root.find(".//returnReasonCode")
    if gateway is not None:
        code = (gateway.text or "").strip()
        raise CollectionError(f"API gateway rejected request (code {code[:8] if code.isdigit() else 'unknown'}). Check service activation/key/quota.")
    code = root.findtext("./header/resultCode", "").strip()
    if code != "00":
        raise CollectionError(f"API rejected request (code {code[:8] if code.isdigit() else 'unknown'}). Check parameters/key/quota.")
    try:
        total = int(root.findtext("./body/totalCount", ""))
        page = int(root.findtext("./body/pageNo", ""))
        if total < 0 or page < 1:
            raise ValueError
    except ValueError as exc:
        raise CollectionError("Missing or invalid pagination metadata.") from exc
    items = root.findall("./body/items/item")
    rows = []
    for item in items:
        parent = {n.tag: (n.text or "").strip() for n in item if n.tag != "details"}
        details = item.findall("./details/detail")
        for detail in details or [None]:
            row = dict(parent)
            if detail is not None:
                row.update({n.tag: (n.text or "").strip() for n in detail})
            # Exact-content identity: corrections remain distinct, never silently merged.
            canonical = json.dumps(row, sort_keys=True, ensure_ascii=False)
            row["record_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            rows.append(row)
    return rows, len(items), total, page


def fetch_page(params, timeout=30, retries=3, *, endpoint=ENDPOINT):
    url = endpoint + "?" + urlencode(params)
    for attempt in range(retries + 1):
        print(
            f"Requesting page {params.get('pageNo', '1')} "
            f"(attempt {attempt + 1}/{retries + 1}, socket timeout {timeout}s)...",
            file=sys.stderr, flush=True,
        )
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if status not in (429, 500, 502, 503, 504) or attempt == retries:
                raise CollectionError(f"HTTP {status}; collection stopped.") from None
        except (URLError, TimeoutError, OSError, HTTPException) as exc:
            reason = exc.reason if isinstance(exc, URLError) else exc
            category = "Request timed out" if isinstance(reason, TimeoutError) else "Connection or HTTP response failed"
            if attempt == retries:
                raise CollectionError(f"{category}; retry a shorter date range or check connectivity.") from None
            print(f"{category}.", file=sys.stderr, flush=True)
        print(f"Retrying in {min(2 ** attempt, 8)}s...", file=sys.stderr, flush=True)
        time.sleep(min(2 ** attempt, 8))


def collect(params, *, delay=0.2, max_pages=10000, fetch=fetch_page):
    rows, seen_pages = {}, set()
    received = 0
    initial_total = None
    for page in range(1, max_pages + 1):
        payload = fetch({**params, "pageNo": str(page)})
        batch, count, total, returned_page = parse_page(payload)
        if returned_page != page:
            raise CollectionError("API returned the wrong page; refusing incomplete output.")
        if initial_total is None:
            initial_total = total
        elif total != initial_total:
            raise CollectionError("API total changed during collection; rerun a shorter period.")
        fingerprint = tuple(row["record_id"] for row in batch)
        if count and fingerprint in seen_pages:
            raise CollectionError("API repeated a page; refusing incomplete output.")
        seen_pages.add(fingerprint)
        received += count
        for row in batch:
            rows[row["record_id"]] = row
        print(f"Page {page}: {received}/{total} reports, {len(rows)} unique detail rows", file=sys.stderr)
        if received == total:
            return list(rows.values())
        if not count or received > total:
            raise CollectionError("API item count is inconsistent; collection stopped.")
        time.sleep(delay)
    raise CollectionError("Page limit reached; increase --max-pages or shorten the period.")


def write_csv(path, rows, overwrite=False, fieldnames=None):
    """Publish only a complete result, preserving an existing file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise CollectionError("Output already exists; choose another path or use --overwrite.")
    fields = fieldnames if fieldnames is not None else BASE_FIELDS + sorted({k for row in rows for k in row} - set(BASE_FIELDS))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if overwrite:
            os.replace(temp_path, path)
        else:
            # Atomic no-clobber publication, including concurrent writers.
            os.link(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Port authority code, e.g. 020 (Busan)")
    parser.add_argument("--start", required=True, type=date_arg)
    parser.add_argument("--end", required=True, type=date_arg)
    parser.add_argument("--call-sign", help="Optional vessel call sign")
    parser.add_argument("--date-basis", choices=("I", "O"), default="I", help="Arrival (I) or departure (O) date")
    parser.add_argument("--page-size", type=int, choices=range(1, 51), default=50, metavar="1..50")
    parser.add_argument("--max-pages", type=int, default=10000)
    parser.add_argument("--timeout", type=int, default=30, help="Socket timeout in seconds (default: 30)")
    parser.add_argument("--retries", type=int, default=3, help="Retries per page (default: 3)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must not be after --end")
    if not re.fullmatch(r"\d{3}", args.port):
        parser.error("--port must be a three-digit port authority code")
    if args.max_pages < 1:
        parser.error("--max-pages must be positive")
    if args.timeout < 1 or args.retries < 0:
        parser.error("--timeout must be positive and --retries must be nonnegative")
    output = args.output or Path(__file__).parent / "data" / "port_calls" / f"port_calls_{args.port}_{args.start}_{args.end}_{args.date_basis}.csv"
    try:
        if output.exists() and not args.overwrite:
            raise CollectionError("Output already exists; choose another path or use --overwrite.")
        params = {"serviceKey": load_key(args.env_file), "prtAgCd": args.port, "sde": args.start, "ede": args.end, "deGb": args.date_basis, "numOfRows": str(args.page_size)}
        if args.call_sign:
            params["clsgn"] = args.call_sign.strip()
        rows = collect(params, max_pages=args.max_pages, fetch=partial(fetch_page, timeout=args.timeout, retries=args.retries))
        write_csv(output, rows, overwrite=args.overwrite)
    except KeyboardInterrupt:
        print("Stopped by user; collection interrupted.", file=sys.stderr)
        return 130
    except CollectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print("ERROR: Unable to read or write a local file.", file=sys.stderr)
        return 1
    print(f"Saved {len(rows)} detail rows to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
