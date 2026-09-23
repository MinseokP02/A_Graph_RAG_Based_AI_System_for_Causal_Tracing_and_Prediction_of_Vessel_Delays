"""Normalization, extensible entity keys and import diagnostics."""

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("portmis.middleware")


def text(value: Any) -> str | None:
    """Normalize null sentinels without removing leading zeros from strings."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    result = str(value).strip()
    return (
        None
        if result.casefold() in ("", "null", "none", "nan", "nat", "<na>")
        else result
    )


def name(value: Any) -> str | None:
    """Trim, collapse whitespace and uppercase a display/matching name."""
    value = text(value)
    return " ".join(value.split()).upper() if value else None


def key(namespace: str, *parts: str) -> str:
    """Encode composite identity without separator collisions."""
    return (
        namespace + ":" + json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    )


def vessel_key(call_sign: Any, vessel_name: Any) -> str | None:
    """Prefer call sign, then normalized name; extend here for MMSI/IMO resolution."""
    sign = name(call_sign)
    normalized_name = name(vessel_name)
    if sign:
        return key("callsign", sign)
    return key("vessel-name", normalized_name) if normalized_name else None


@dataclass
class Stats:
    """Exact input entity counts; memory grows with distinct entity keys."""

    counts: Counter = field(default_factory=Counter)
    entities: dict[str, set[str]] = field(
        default_factory=lambda: {
            label: set() for label in ("vessels", "ports", "port_calls", "berths")
        }
    )

    def issue(self, category: str, record: str, column: str) -> None:
        """Count every issue but limit repeated field warnings to keep logs usable."""
        self.counts[category] += 1
        self.counts["invalid_fields"] += 1
        if self.counts[category] <= 10:
            logger.warning("%s record=%r field=%s", category, record, column)

    def summary(self) -> dict[str, int]:
        """Return JSON-friendly counts, including zero-valued required metrics."""
        required = (
            "input_rows",
            "valid_rows",
            "skipped_rows",
            "invalid_rows",
            "invalid_datetime",
            "invalid_numeric",
            "invalid_boolean",
            "missing_vessel_identifier",
            "batches",
            "processed_rows",
            "invalid_fields",
            "available_portStayMinutes",
            "available_departureDelayMinutes",
        )
        result = {k: self.counts[k] for k in required}
        result.update(self.counts)
        result.update({"unique_" + k: len(v) for k, v in self.entities.items()})
        return result
