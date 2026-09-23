"""CLI: python -m middleware.main --csv path.csv [--dry-run]."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config
from .importer import run
from .utils import Stats, logger


def main(argv: list[str] | None = None) -> int:
    """Load environment, validate settings and run the same pipeline in both modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit", type=int, help="Maximum input rows, including skipped rows"
    )
    parser.add_argument(
        "--env-file", type=Path, default=Path(__file__).with_name(".env")
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stats = Stats()
    try:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
        config = Config.from_env(args.csv, args.batch_size)
        if args.dry_run:
            run(config, stats, limit=args.limit)
        else:
            from .neo4j_client import Neo4jClient

            with Neo4jClient(config) as client:
                client.initialize_schema()
                run(config, stats, client, args.limit)
    except ModuleNotFoundError as exc:
        logger.error(
            "Missing Python module %r. Interpreter: %s. "
            "Install middleware dependencies with this interpreter: "
            '"%s" -m pip install -r "%s"',
            exc.name,
            sys.executable,
            sys.executable,
            Path(__file__).with_name("requirements.txt"),
        )
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted; earlier committed batches remain in Neo4j")
        return 130
    except Exception as exc:  # noqa: BLE001 -- CLI must not expose credential-bearing driver messages
        # Never echo driver/OS error messages that might include credentials.
        logger.error(
            "Import failed (%s). Check CSV schema/path, settings, driver installation and database availability.",
            type(exc).__name__,
        )
        return 1
    finally:
        logger.info(
            "Summary: %s",
            json.dumps(stats.summary(), ensure_ascii=False, sort_keys=True),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
