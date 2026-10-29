"""Tests for the same-origin Port-MIS dashboard proxy."""

import asyncio
import sys
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import HTTPException

pytest.importorskip("jwt", reason="API extras are not installed")

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.routers import portmis_routes

sys.argv = _original_argv

pytestmark = pytest.mark.offline


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Client:
    response = _Response({"status": "ok", "answer": "분석 결과"})
    request = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, **kwargs):
        type(self).request = (url, kwargs)
        return type(self).response


def _endpoint():
    router = portmis_routes.create_portmis_routes()
    return next(
        route.endpoint for route in router.routes if route.path == "/portmis/ask"
    )


def _tracks_endpoint():
    router = portmis_routes.create_portmis_routes()
    return next(
        route.endpoint for route in router.routes if route.path == "/portmis/ais/tracks"
    )


def test_portmis_proxy_forwards_dashboard_filters(monkeypatch):
    monkeypatch.setattr(portmis_routes.httpx, "AsyncClient", _Client)
    monkeypatch.setenv("PORTMIS_API_URL", "http://middleware:9622/")
    monkeypatch.setenv("PORTMIS_API_KEY", "secret")

    request = portmis_routes.PortMISAskRequest(
        question="출항 지연 원인을 분석해줘",
        delay_threshold_minutes=90,
        plan=portmis_routes.PortMISQueryPlan(
            vessel_name="TEST SHIP",
            start_date="2026-10-01",
            end_date="2026-10-27",
        ),
    )
    result = asyncio.run(_endpoint()(request))

    assert result == {"status": "ok", "answer": "분석 결과"}
    url, kwargs = _Client.request
    assert url == "http://middleware:9622/ask"
    assert kwargs["headers"] == {"X-API-Key": "secret"}
    assert kwargs["json"]["plan"]["vessel_name"] == "TEST SHIP"
    assert kwargs["json"]["plan"]["date_basis"] == "departure"
    assert kwargs["json"]["delay_threshold_minutes"] == 90


def test_portmis_proxy_sanitizes_upstream_errors(monkeypatch):
    class FailingClient(_Client):
        async def post(self, url, **kwargs):
            raise httpx.ConnectError("credential-bearing upstream failure")

    monkeypatch.setattr(portmis_routes.httpx, "AsyncClient", FailingClient)
    request = portmis_routes.PortMISAskRequest(question="지연 원인 분석")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_endpoint()(request))

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Port-MIS query service is unavailable"


def test_ais_tracks_groups_recent_positions(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(minutes=5)).isoformat()
    old = (now - timedelta(days=2)).isoformat()
    source = tmp_path / "korea.csv"
    source.write_text(
        "time_utc,received_at_utc,message_type,mmsi,ship_name,call_sign,imo_number,latitude,longitude,sog,cog,true_heading\n"
        f"{old},,PositionReport,440000001,OLD,,,,37.4,126.5,1,2,3\n"
        f"{recent},,PositionReport,440000001,TEST SHIP,D7TEST,9876543,37.45,126.60,8.2,123.4,120\n"
        f"2026-10-29 02:43:52 +0000 UTC,{recent},PositionReport,440000001,TEST SHIP,D7TEST,9876543,37.46,126.61,8.5,124.0,121\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AIS_CSV_PATH", str(source))

    result = _tracks_endpoint()(hours=24, mmsi=None, max_points=100)

    assert result["status"] == "ok"
    assert result["summary"] == {"vessels": 1, "points": 2, "limited": False}
    assert result["vessels"][0]["mmsi"] == "440000001"
    assert result["vessels"][0]["callSign"] == "D7TEST"
    assert result["vessels"][0]["latest"]["longitude"] == 126.61
