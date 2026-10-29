"""Validated API requests and a constrained query plan (never generated Cypher)."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueryPlan(BaseModel):
    """Filters accepted by the fixed Neo4j query templates."""

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
    ] = "visits"
    call_sign: str | None = Field(default=None, min_length=1, max_length=100)
    vessel_name: str | None = Field(default=None, min_length=1, max_length=200)
    port_code: str | None = Field(default=None, min_length=1, max_length=100)
    port_name: str | None = Field(default=None, min_length=1, max_length=200)
    berth_name: str | None = Field(default=None, min_length=1, max_length=200)
    start_date: date | None = None
    end_date: date | None = None
    date_basis: Literal["event", "arrival", "departure"] = "event"
    clarification: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def ordered_dates(self) -> "QueryPlan":
        """Reject inverted ranges and dates too large for exclusive-end conversion."""
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must not exceed end_date")
        if self.end_date == date.max:
            raise ValueError("end_date is out of range")
        return self


class AskRequest(BaseModel):
    """A question with optional explicit filters that bypass LLM planning."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=3, max_length=2000)
    mode: Literal["naive", "hybrid", "neo4j_mix"] = "neo4j_mix"
    plan: QueryPlan | None = None
    use_documents: bool = True
    max_records: int = Field(default=10, ge=1, le=50)
    forecast_steps: int = Field(default=1, ge=1, le=24)
    delay_threshold_minutes: float = Field(default=60.0, ge=0, le=10080)
