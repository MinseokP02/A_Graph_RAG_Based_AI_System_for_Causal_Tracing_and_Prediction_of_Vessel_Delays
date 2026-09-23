"""Bounded batch orchestration shared by dry-run and actual import."""

from typing import Protocol

from .config import Config
from .preprocessing import records
from .utils import Stats, logger


class BatchClient(Protocol):
    """Small interface for the database client and offline tests."""

    def import_batch(self, rows: list[dict]) -> None:
        """Persist a batch atomically."""
        ...


def run(
    config: Config,
    stats: Stats,
    client: BatchClient | None = None,
    limit: int | None = None,
) -> None:
    """Process bounded batches; None client means no database interaction."""

    def flush(batch: list[dict]) -> None:
        """Commit a batch and log its exact source-row range on failure."""
        index = stats.counts["batches"] + 1
        if client is not None:
            try:
                client.import_batch(batch)
            except Exception as exc:  # noqa: BLE001 -- sanitize arbitrary driver errors at the batch boundary
                logger.error(
                    "Batch %d failed, source rows=%s..%s, record IDs=%r..%r, error type=%s; earlier batches remain committed",
                    index,
                    batch[0]["portCall"]["sourceRow"],
                    batch[-1]["portCall"]["sourceRow"],
                    batch[0]["portCall"]["recordId"],
                    batch[-1]["portCall"]["recordId"],
                    type(exc).__name__,
                )
                raise RuntimeError(
                    f"Batch {index} failed; see source range in log"
                ) from None
        stats.counts["batches"] += 1
        stats.counts["processed_rows"] += len(batch)
        logger.info(
            "%s batch=%d rows=%d processed=%d",
            "dry-run" if client is None else "committed",
            index,
            len(batch),
            stats.counts["processed_rows"],
        )

    batch = []
    for record in records(config, stats, limit):
        batch.append(record)
        if len(batch) == config.batch_size:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
