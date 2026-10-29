"""Authenticated Port-MIS dashboard routes and local AIS track access."""

import csv
import os
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import logger


def _optional_float(row: dict[str, str | None], name: str) -> float | None:
    """Parse an optional CSV number without rejecting the whole AIS row."""
    value = row.get(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class PortMISQueryPlan(BaseModel):
    """Dashboard filters forwarded to the constrained Port-MIS planner."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: Literal[
        "visits",
        "routes",
        "berths",
        "delays",
        "cargo",
        "count",
        "control",
        "delay_forecast",
    ] = "delays"
    call_sign: str | None = Field(default=None, min_length=1, max_length=100)
    vessel_name: str | None = Field(default=None, min_length=1, max_length=200)
    port_code: str | None = Field(default=None, min_length=1, max_length=100)
    port_name: str | None = Field(default=None, min_length=1, max_length=200)
    berth_name: str | None = Field(default=None, min_length=1, max_length=200)
    start_date: date | None = None
    end_date: date | None = None
    date_basis: Literal["event", "arrival", "departure"] = "departure"

    @model_validator(mode="after")
    def validate_dates(self) -> "PortMISQueryPlan":
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must not exceed end_date")
        return self


class PortMISAskRequest(BaseModel):
    """Subset of the middleware request contract used by the dashboard."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=2000)
    mode: Literal["naive", "hybrid", "neo4j_mix"] = "neo4j_mix"
    plan: PortMISQueryPlan | None = None
    use_documents: bool = True
    max_records: int = Field(default=25, ge=1, le=50)
    forecast_steps: int = Field(default=1, ge=1, le=24)
    delay_threshold_minutes: float = Field(default=60.0, ge=0, le=10080)


def create_portmis_routes(api_key: str | None = None) -> APIRouter:
    """Expose Port-MIS through the main API so the WebUI remains same-origin."""

    router = APIRouter(prefix="/portmis", tags=["portmis"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/ais/tracks", dependencies=[Depends(combined_auth)])
    def get_ais_tracks(
        hours: int = Query(default=24, ge=1, le=168),
        mmsi: str | None = Query(default=None, pattern=r"^\d{9}$"),
        max_points: int = Query(default=5000, ge=1, le=20000),
    ) -> dict:
        """Return recent Korea-area AIS positions grouped into vessel tracks."""
        source = Path(
            os.getenv("AIS_CSV_PATH", "data/ais/korea_ais_stream.csv")
        ).resolve()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        positions: deque[dict] = deque(maxlen=max_points)
        identities: dict[str, dict[str, str | None]] = {}
        if not source.is_file():
            return {
                "status": "empty",
                "region": "korea",
                "hours": hours,
                "vessels": [],
                "summary": {"vessels": 0, "points": 0},
            }

        def parse_time(value: str | None) -> datetime | None:
            if not value:
                return None
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

        try:
            with source.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    vessel_mmsi = (row.get("mmsi") or "").strip()
                    if not vessel_mmsi or (mmsi and vessel_mmsi != mmsi):
                        continue
                    identity = identities.setdefault(
                        vessel_mmsi,
                        {"shipName": None, "callSign": None, "imoNumber": None},
                    )
                    for source_name, target_name in (
                        ("ship_name", "shipName"),
                        ("call_sign", "callSign"),
                        ("imo_number", "imoNumber"),
                    ):
                        if row.get(source_name):
                            identity[target_name] = row[source_name].strip()
                    observed_at = parse_time(row.get("time_utc")) or parse_time(
                        row.get("received_at_utc")
                    )
                    if observed_at is None or observed_at < cutoff:
                        continue
                    try:
                        latitude = float(row.get("latitude") or "")
                        longitude = float(row.get("longitude") or "")
                    except ValueError:
                        continue
                    if not (33 <= latitude <= 39 and 124 <= longitude <= 132):
                        continue

                    positions.append(
                        {
                            "mmsi": vessel_mmsi,
                            "timestamp": observed_at.isoformat(),
                            "latitude": latitude,
                            "longitude": longitude,
                            "sog": _optional_float(row, "sog"),
                            "cog": _optional_float(row, "cog"),
                            "heading": _optional_float(row, "true_heading"),
                        }
                    )
        except (OSError, csv.Error):
            logger.warning("Unable to read local AIS track data")
            raise HTTPException(
                status_code=503, detail="AIS track data is unavailable"
            ) from None

        grouped: dict[str, list[dict]] = {}
        for point in positions:
            grouped.setdefault(point["mmsi"], []).append(point)
        vessels = []
        for vessel_mmsi, points in grouped.items():
            points.sort(key=lambda item: item["timestamp"])
            vessels.append(
                {
                    "mmsi": vessel_mmsi,
                    **identities.get(vessel_mmsi, {}),
                    "latest": points[-1],
                    "points": points,
                }
            )
        vessels.sort(key=lambda item: item["latest"]["timestamp"], reverse=True)
        return {
            "status": "ok" if vessels else "empty",
            "region": "korea",
            "hours": hours,
            "bounds": {"north": 39, "south": 33, "west": 124, "east": 132},
            "vessels": vessels,
            "summary": {
                "vessels": len(vessels),
                "points": sum(len(item["points"]) for item in vessels),
                "limited": len(positions) == max_points,
            },
        }

    @router.post("/ask", dependencies=[Depends(combined_auth)])
    async def ask_portmis(request: PortMISAskRequest) -> dict:
        base_url = os.getenv("PORTMIS_API_URL", "http://127.0.0.1:9622").rstrip("/")
        headers = {}
        middleware_api_key = os.getenv("PORTMIS_API_KEY")
        if middleware_api_key:
            headers["X-API-Key"] = middleware_api_key

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(90, connect=10), trust_env=False
            ) as client:
                response = await client.post(
                    f"{base_url}/ask",
                    json=request.model_dump(mode="json", exclude_none=True),
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Port-MIS dashboard request failed (%s)", type(exc).__name__)
            raise HTTPException(
                status_code=502,
                detail="Port-MIS query service is unavailable",
            ) from None

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=502,
                detail="Port-MIS query service returned an invalid response",
            )
        return payload

    return router
