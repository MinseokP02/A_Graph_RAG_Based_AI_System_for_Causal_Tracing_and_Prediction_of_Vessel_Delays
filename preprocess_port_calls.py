"""Select 34 port-call columns, including visit identifiers.

No model calls or database writes. Blank CSV cells represent null values.
"""

import argparse
import csv
import json
from pathlib import Path

from collect_port_calls import CollectionError, write_csv


OUTPUT_COLUMNS = """
record_id prtAgCd prtAgNm etryptYear etryptCo clsgn vsslNm vsslNltyCd vsslKndCd vsslKndNm
grtg etryptPurpsNm frstDpmprtNatPrtCd frstDpmprtPrtNm
prvsDpmprtNatPrtCd prvsDpmprtPrtNm nxlnptNatPrtCd nxlnptPrtNm
dstnNatPrtCd dstnPrtNm etryptDt tkoffDt tkoffPrrrnDt dstnEtryptDt
laidupFcltyCd laidupFcltySubCd laidupFcltyNm tugYn piltgYn
ldadngFrghtClCd ldadngTon trnpdtTon landngFrghtTon ldFrghtTon
""".split()


def preprocess(source: Path) -> tuple[list[dict], dict]:
    """Select the requested columns without changing values or dropping rows."""
    rows = []
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError("Duplicate column names in input")
        missing = set(OUTPUT_COLUMNS) - set(headers)
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        for number, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV record at data row {number}")
            rows.append({key: row[key] for key in OUTPUT_COLUMNS})
    return rows, {
        "source": str(source.resolve()),
        "rows": len(rows),
        "columns": len(OUTPUT_COLUMNS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or args.input.parent / "processed" / (args.input.stem + "_selected.csv")
    if output.resolve() == args.input.resolve():
        parser.error("Output must not replace the source CSV")
    try:
        rows, summary = preprocess(args.input)
        write_csv(output, rows, overwrite=args.overwrite, fieldnames=OUTPUT_COLUMNS)
    except (OSError, ValueError, csv.Error, CollectionError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
