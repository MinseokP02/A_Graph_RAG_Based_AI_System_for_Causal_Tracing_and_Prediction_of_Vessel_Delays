"""Validated environment and CLI configuration."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Config:
    """Runtime settings; credentials are excluded from representations."""

    csv_path: Path
    batch_size: int = 1000
    delay_threshold_minutes: float = 60
    source_timezone: str = "Asia/Seoul"
    uri: str = field(default="bolt://localhost:7687", repr=False)
    username: str = field(default="neo4j", repr=False)
    password: str = field(default="", repr=False)
    database: str = "neo4j"

    def __post_init__(self) -> None:
        """Reject invalid settings before processing or opening a connection."""
        if self.batch_size <= 0:
            raise ValueError("BATCH_SIZE must be positive")
        if (
            not math.isfinite(self.delay_threshold_minutes)
            or self.delay_threshold_minutes < 0
        ):
            raise ValueError("DELAY_THRESHOLD_MINUTES must be finite and nonnegative")
        ZoneInfo(self.source_timezone)

    @classmethod
    def from_env(
        cls, csv_path: Path | None = None, batch_size: int | None = None
    ) -> "Config":
        """Read settings after dotenv loading; explicit CLI values take precedence."""
        path = csv_path or os.getenv("CSV_PATH")
        if not path:
            raise ValueError("Provide --csv or CSV_PATH")
        return cls(
            csv_path=Path(path),
            batch_size=batch_size
            if batch_size is not None
            else int(os.getenv("BATCH_SIZE", "1000")),
            delay_threshold_minutes=float(os.getenv("DELAY_THRESHOLD_MINUTES", "60")),
            source_timezone=os.getenv("SOURCE_TIMEZONE", "Asia/Seoul"),
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )
