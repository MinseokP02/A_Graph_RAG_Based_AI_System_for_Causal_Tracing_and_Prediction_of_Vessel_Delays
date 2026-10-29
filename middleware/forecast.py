"""Typed TFT boundary shared by the query API and independent forecast service."""

import re
from datetime import date
from itertools import pairwise
from typing import Literal

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .query_config import QueryConfig
from .query_models import QueryPlan


def is_forecast_question(question: str) -> bool:
    """Route every Korean prediction request deterministically."""
    return "예측" in question


def extract_forecast_date(question: str) -> date | None:
    """Extract one explicit calendar date without interpreting relative dates."""

    patterns = (
        r"(?<!\d)(\d{4})[-./](\d{1,2})[-./](\d{1,2})(?!\d)",
        r"(?<!\d)(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일",
    )
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, question):
            try:
                values.append(date(*(int(value) for value in match.groups())))
            except ValueError:
                continue
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else None


class ForecastRequest(BaseModel):
    """Request predictions for a bounded number of model time steps."""

    model_config = ConfigDict(extra="forbid")
    plan: QueryPlan
    steps: int = Field(default=1, ge=1, le=24)
    delay_threshold_minutes: float = Field(default=60.0, ge=0, le=10080)


class ForecastPoint(BaseModel):
    """Departure delay quantiles in minutes, not observed facts."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    timestamp: AwareDatetime
    p10: float
    p50: float
    p90: float
    delay_probability_percent: float | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def ordered_quantiles(self) -> "ForecastPoint":
        """Reject malformed or crossed quantiles instead of silently relabeling."""
        if not self.p10 <= self.p50 <= self.p90:
            raise ValueError("Forecast quantiles must be ordered")
        return self


class ForecastBasis(BaseModel):
    """Point-in-time inputs shown to users as forecast provenance."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    planned_departure: AwareDatetime | None = None
    feature_cutoff: AwareDatetime | None = None
    ais_observation_count: float | None = Field(default=None, ge=0)
    latest_sog_knots: float | None = Field(default=None, ge=0)
    latest_latitude: float | None = Field(default=None, ge=-90, le=90)
    latest_longitude: float | None = Field(default=None, ge=-180, le=180)
    gross_tonnage: float | None = Field(default=None, ge=0)
    cargo_ton: float | None = Field(default=None, ge=0)
    hours_to_arrival: float | None = None
    weather_age_hours: float | None = Field(default=None, ge=0)
    temperature_2m_c: float | None = None
    wind_speed_10m_ms: float | None = Field(default=None, ge=0)
    rain_mm: float | None = Field(default=None, ge=0)


class ForecastResult(BaseModel):
    """Non-ready states never carry numeric predictions."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "model_not_ready", "insufficient_data", "unavailable"]
    model: Literal["TFT"] = "TFT"
    model_version: str | None = Field(default=None, max_length=100)
    as_of: AwareDatetime | None = None
    target: Literal["departure_delay_minutes"] = "departure_delay_minutes"
    series_id: str | None = Field(default=None, max_length=200)
    predictions: list[ForecastPoint] = Field(default_factory=list, max_length=24)
    delay_threshold_minutes: float | None = Field(default=None, ge=0, le=10080)
    probability_method: Literal["quantile_linear_interpolation_capped"] | None = None
    basis: ForecastBasis | None = None

    @model_validator(mode="after")
    def consistent_result(self) -> "ForecastResult":
        """Validate provenance and chronology at the HTTP boundary."""
        if self.status == "ok":
            if (
                not self.predictions
                or not self.model_version
                or not self.series_id
                or not self.as_of
            ):
                raise ValueError(
                    "Successful forecasts require predictions and provenance"
                )
            times = [point.timestamp for point in self.predictions]
            if times[0] <= self.as_of or any(a >= b for a, b in pairwise(times)):
                raise ValueError("Forecast timestamps must increase after as_of")
        elif self.predictions:
            raise ValueError("Unavailable forecasts cannot contain predictions")
        return self


FORECAST_MESSAGES = {
    "model_not_ready": "TFT 학습 모델이 아직 준비되지 않아 지연 예측값을 제공할 수 없습니다.",
    "insufficient_data": "요청 조건에 맞는 TFT 입력 시계열이 부족하거나 예측 범위를 지원하지 않습니다.",
    "unavailable": "TFT 예측 서비스에 연결할 수 없거나 유효한 예측 결과를 받지 못했습니다.",
}


class TFTClient:
    """Call the sidecar without exposing transport errors or credentials."""

    def __init__(self, http: httpx.AsyncClient, config: QueryConfig) -> None:
        self.http, self.config = http, config

    async def predict(self, request: ForecastRequest) -> ForecastResult:
        """Validate all predictions before allowing them into LLM context."""
        try:
            response = await self.http.post(
                self.config.tft_url + "/predict",
                json=request.model_dump(mode="json"),
                headers={"X-API-Key": self.config.tft_api_key}
                if self.config.tft_api_key
                else {},
                timeout=self.config.tft_timeout,
            )
            response.raise_for_status()
            result = ForecastResult.model_validate(response.json())
            if result.status == "ok" and len(result.predictions) != request.steps:
                raise ValueError("Unexpected forecast horizon")
            return result
        except (httpx.HTTPError, ValueError, TypeError):
            return ForecastResult(status="unavailable")
