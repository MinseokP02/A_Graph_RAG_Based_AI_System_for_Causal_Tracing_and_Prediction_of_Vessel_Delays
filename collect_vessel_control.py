"""Collect MOF vessel-control events as CSV; no model calls or database writes."""

import argparse
import re
import sys
from functools import partial
from pathlib import Path

from collect_port_calls import (
    CollectionError,
    collect,
    date_arg,
    fetch_page,
    load_key,
    write_csv,
)

ENDPOINT = "https://apis.data.go.kr/1192000/CntlVssl2/Info"
CONTROL_FIELDS = [
    "record_id",
    "prtAgCd",
    "prtAgNm",
    "etryptYear",
    "etryptCo",
    "clsgn",
    "vsslKndCd",
    "vsslKndNm",
    "vsslNm",
    "vsslGrtg",
    "vsslNltyCd",
    "vsslNltyNm",
    "vsslSatmntAgCd",
    "satmntEtryptYear",
    "satmntEtryptCo",
    "aprtfEtryptDt",
    "tkoffDt",
    "harborEntrpsCd",
    "harborEntrpsNm",
    "cntrlNm",
    "commCo",
    "cntrlSe",
    "cntrlOpertDt",
    "fcltyCd",
    "fcltySubCd",
    "fcltyNm",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", required=True, help="Port authority code, e.g. 020 (Busan)"
    )
    parser.add_argument("--start", required=True, type=date_arg)
    parser.add_argument("--end", required=True, type=date_arg)
    parser.add_argument("--call-sign", help="Optional vessel call sign")
    parser.add_argument(
        "--page-size", type=int, choices=range(1, 51), default=50, metavar="1..50"
    )
    parser.add_argument("--max-pages", type=int, default=10000)
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Socket timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Retries per page (default: 3)"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--env-file", type=Path, default=Path(__file__).with_name(".env")
    )
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
    output = (
        args.output
        or Path(__file__).parent
        / "data"
        / "vessel_control"
        / f"vessel_control_{args.port}_{args.start}_{args.end}.csv"
    )
    try:
        if output.exists() and not args.overwrite:
            raise CollectionError(
                "Output already exists; choose another path or use --overwrite."
            )
        params = {
            "serviceKey": load_key(args.env_file, key_env="MOF_CONTROL_SERVICE_KEY"),
            "prtAgCd": args.port,
            "sde": args.start,
            "ede": args.end,
            "numOfRows": str(args.page_size),
        }
        if args.call_sign:
            params["clsgn"] = args.call_sign.strip()
        rows = collect(
            params,
            max_pages=args.max_pages,
            fetch=partial(
                fetch_page,
                timeout=args.timeout,
                retries=args.retries,
                endpoint=ENDPOINT,
            ),
        )
        fields = CONTROL_FIELDS + sorted(
            {k for row in rows for k in row} - set(CONTROL_FIELDS)
        )
        write_csv(output, rows, overwrite=args.overwrite, fieldnames=fields)
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
