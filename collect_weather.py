"""Collect hourly regional weather for a port-call dataset, preserving provenance."""

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
COORDINATE_SOURCE = "https://github.com/datasets/un-locode"
REGIONS = [
    ("frstDpmprtNatPrtCd", "frstDpmprtPrtNm"),
    ("prvsDpmprtNatPrtCd", "prvsDpmprtPrtNm"),
    ("nxlnptNatPrtCd", "nxlnptPrtNm"),
    ("dstnNatPrtCd", "dstnPrtNm"),
]
AUTHORITY_UNLOCODE = {
    "020": "KRPUS",
    "030": "KRINC",
}
logger = logging.getLogger(__name__)


def coordinates(value):
    """Parse UN/LOCODE degrees/minutes without guessing missing locations."""
    match = re.fullmatch(r"(\d{2})(\d{2})([NS]) (\d{3})(\d{2})([EW])", value)
    if not match:
        return None
    a, b, ns, c, d, ew = match.groups()
    if int(b) >= 60 or int(d) >= 60:
        return None
    lat = (int(a) + int(b) / 60) * (-1 if ns == "S" else 1)
    lon = (int(c) + int(d) / 60) * (-1 if ew == "W" else 1)
    return (lat, lon) if abs(lat) <= 90 and abs(lon) <= 180 else None


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, fields, rows):
    """Replace the output only after completing the new file."""
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def inventory(rows, locations, supplemental=None, *, current_port_only=False):
    regions = defaultdict(lambda: {"names": set(), "roles": set(), "references": 0})
    for row in rows:
        if not current_port_only:
            for code_field, name_field in REGIONS:
                code = row.get(code_field, "").strip().upper()
                if code:
                    regions[code]["names"].add(row.get(name_field, ""))
                    regions[code]["roles"].add(code_field)
                    regions[code]["references"] += 1
        authority = row.get("prtAgCd", "").strip()
        if unlocode := AUTHORITY_UNLOCODE.get(authority):
            regions[unlocode]["names"].add(row.get("prtAgNm", ""))
            regions[unlocode]["roles"].add(f"prtAgCd:{authority}")
            regions[unlocode]["references"] += 1
        elif row.get("prtAgCd"):
            code = "authority:" + row["prtAgCd"]
            regions[code]["names"].add(row.get("prtAgNm", ""))
            regions[code]["roles"].add("prtAgCd")
            regions[code]["references"] += 1
    result = []
    for code, info in sorted(regions.items()):
        location = locations.get(code, {})
        point = coordinates(location.get("Coordinates", ""))
        coordinate_source = COORDINATE_SOURCE
        extra = (supplemental or {}).get(code, {})
        if not point and len(extra.get("coordinates", [])) == 2:
            lon, lat = extra["coordinates"]
            if (
                isinstance(lat, (int, float))
                and isinstance(lon, (int, float))
                and abs(lat) <= 90
                and abs(lon) <= 180
            ):
                point = lat, lon
                coordinate_source = "https://github.com/marchah/sea-ports (community regional coordinates)"
        if code.endswith("ZZZ") or code == "OCZZZ":
            point = None
        result.append(
            {
                "port_code": code,
                "dataset_names": " | ".join(sorted(info["names"])),
                "roles": " | ".join(sorted(info["roles"])),
                "references": info["references"],
                "location_name": location.get("Name", "") or extra.get("name", ""),
                "latitude": point[0] if point else "",
                "longitude": point[1] if point else "",
                "coordinate_source": coordinate_source,
                "status": "resolved" if point else "unresolved_coordinates",
            }
        )
    return result


def fetch(client, params):
    for attempt in range(4):
        response = client.get(ENDPOINT, params=params)
        if (response.status_code == 429 or response.status_code >= 500) and attempt < 3:
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Weather request retries exhausted")


def weather_rows(payload, region, collected_at):
    units = payload["hourly_units"]
    if (
        units.get("temperature_2m") != "°C"
        or units.get("wind_speed_10m") != "m/s"
        or units.get("rain") != "mm"
    ):
        raise ValueError("Unexpected weather units")
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Expected UTC weather timestamps")
    hourly = payload["hourly"]
    times = hourly["time"]
    fields = ("temperature_2m", "wind_speed_10m", "rain")
    if any(len(hourly[f]) != len(times) for f in fields) or len(set(times)) != len(
        times
    ):
        raise ValueError("Malformed hourly arrays")
    for i, timestamp in enumerate(times):
        values = [hourly[f][i] for f in fields]
        if any(
            v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v))
            for v in values
        ):
            raise ValueError("Invalid weather value")
        yield {
            "port_code": region["port_code"],
            "weather_time_utc": timestamp + "+00:00",
            "temperature_2m_c": values[0],
            "wind_speed_10m_ms": values[1],
            "rain_mm": values[2],
            "requested_latitude": region["latitude"],
            "requested_longitude": region["longitude"],
            "grid_latitude": payload["latitude"],
            "grid_longitude": payload["longitude"],
            "source": "Open-Meteo",
            "dataset_type": "reanalysis",
            "model": "best_match",
            "collected_at_utc": collected_at,
        }


def enrich_departures(rows, weather):
    """Join only known current ports and actual departure times, never route dates."""
    index = {(r["port_code"], r["weather_time_utc"]): r for r in weather}
    output = []
    for source in rows:
        row = dict(source)
        row.update(
            departure_weather_time_utc="",
            departure_temperature_2m_c="",
            departure_wind_speed_10m_ms="",
            departure_rain_mm="",
            departure_weather_status="missing_departure_time",
        )
        if source.get("tkoffDt"):
            stamp = datetime.fromisoformat(source["tkoffDt"])
            if stamp.tzinfo is None:
                row["departure_weather_status"] = "missing_timezone"
            elif source.get("prtAgCd") not in AUTHORITY_UNLOCODE:
                row["departure_weather_status"] = "unmapped_current_port"
            else:
                hour = stamp.astimezone(timezone.utc).replace(
                    minute=0, second=0, microsecond=0
                )
                key = (
                    AUTHORITY_UNLOCODE[source["prtAgCd"]],
                    hour.isoformat(timespec="minutes"),
                )
                found = index.get(key)
                row["departure_weather_status"] = (
                    "matched" if found else "outside_collected_period"
                )
                if found:
                    row.update(
                        departure_weather_time_utc=found["weather_time_utc"],
                        departure_temperature_2m_c=found["temperature_2m_c"],
                        departure_wind_speed_10m_ms=found["wind_speed_10m_ms"],
                        departure_rain_mm=found["rain_mm"],
                    )
                    if any(
                        found[f] is None
                        for f in ("temperature_2m_c", "wind_speed_10m_ms", "rain_mm")
                    ):
                        row["departure_weather_status"] = "partial_weather"
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--locations", type=Path, default=Path("data/weather/unlocode_source.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("data/weather"))
    parser.add_argument(
        "--supplemental", type=Path, default=Path("data/weather/ports_source.json")
    )
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument(
        "--current-port-only",
        action="store_true",
        help="Collect weather only for the Port-MIS authority, excluding route ports",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    rows = read_csv(args.csv)
    arrivals = [
        datetime.fromisoformat(r["etryptDt"]).date() for r in rows if r.get("etryptDt")
    ]
    start, end = args.start or min(arrivals), args.end or max(arrivals)
    if start > end or end >= datetime.now(timezone.utc).date():
        raise ValueError("Historical range must be ordered and before today")
    args.output.mkdir(parents=True, exist_ok=True)
    locations = {r["Country"] + r["Location"]: r for r in read_csv(args.locations)}
    supplemental = (
        json.loads(args.supplemental.read_text(encoding="utf-8"))
        if args.supplemental.exists()
        else {}
    )
    regions = inventory(
        rows,
        locations,
        supplemental,
        current_port_only=args.current_port_only,
    )
    save_csv(args.output / "regions.csv", list(regions[0]), regions)
    resolved = [r for r in regions if r["status"] == "resolved"]
    raw = args.output / "raw"
    raw.mkdir(exist_ok=True)
    records, failures = [], []
    # Include the preceding UTC day to cover August 1 in positive-offset regions.
    with httpx.Client(timeout=90) as client:
        for index, region in enumerate(resolved, 1):
            params = {
                "latitude": region["latitude"],
                "longitude": region["longitude"],
                "start_date": (start - timedelta(days=1)).isoformat(),
                "end_date": end.isoformat(),
                "hourly": "temperature_2m,wind_speed_10m,rain",
                "wind_speed_unit": "ms",
                "temperature_unit": "celsius",
                "precipitation_unit": "mm",
                "timezone": "UTC",
            }
            cache_key = hashlib.sha256(
                json.dumps(params, sort_keys=True).encode()
            ).hexdigest()[:16]
            cache = raw / f"{region['port_code']}_{cache_key}.json"
            try:
                if cache.exists():
                    saved = json.loads(cache.read_text(encoding="utf-8"))
                else:
                    saved = {
                        "params": params,
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                        "response": fetch(client, params),
                    }
                    cache.write_text(json.dumps(saved), encoding="utf-8")
                    time.sleep(0.25)
                records.extend(
                    list(weather_rows(saved["response"], region, saved["collected_at"]))
                )
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                failures.append(
                    {"port_code": region["port_code"], "error": type(exc).__name__}
                )
            logger.info(
                "Weather %d/%d (%s), failures=%d",
                index,
                len(resolved),
                region["port_code"],
                len(failures),
            )
    if records:
        save_csv(args.output / "weather_hourly.csv", list(records[0]), records)
    enriched = enrich_departures(rows, records)
    save_csv(args.output / "port_calls_with_weather.csv", list(enriched[0]), enriched)
    summary = {
        "input": str(args.csv),
        "input_rows": len(rows),
        "period_start": str(start),
        "period_end": str(end),
        "utc_start": str(start - timedelta(days=1)),
        "utc_end": str(end),
        "regions_total": len(regions),
        "regions_resolved": len(resolved),
        "regions_collected": len({r["port_code"] for r in records}),
        "regions_unresolved": [
            r["port_code"] for r in regions if r["status"] != "resolved"
        ],
        "hourly_rows": len(records),
        "departure_join_status": {
            status: sum(r["departure_weather_status"] == status for r in enriched)
            for status in sorted({r["departure_weather_status"] for r in enriched})
        },
        "failures": failures,
        "null_values": {
            f: sum(r[f] is None for r in records)
            for f in ("temperature_2m_c", "wind_speed_10m_ms", "rain_mm")
        },
        "notes": [
            "Regional period data; route port visit times are not supplied by this dataset.",
            "UN/LOCODE location coordinates are representative, not berth coordinates.",
            "rain_mm is the preceding hour's rain, not total precipitation including snow.",
            "Open-Meteo historical reanalysis is not a direct station observation.",
        ],
        "source": ENDPOINT,
        "coordinate_source": COORDINATE_SOURCE,
    }
    (args.output / "collection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(
        "Saved %d hourly records for %d regions",
        len(records),
        summary["regions_collected"],
    )


if __name__ == "__main__":
    main()
