"""Publish failed JUnit test cases as GitHub Actions error annotations."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _escape_command(value: str) -> str:
    """Escape data embedded in a GitHub workflow command."""
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def failed_cases(report: Path) -> list[tuple[str, str]]:
    """Return test names and bounded failure details from a JUnit report."""
    root = ET.parse(report).getroot()
    failures: list[tuple[str, str]] = []
    for case in root.iter("testcase"):
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        if failure is None:
            continue
        classname = case.attrib.get("classname", "")
        name = case.attrib.get("name", "unknown test")
        test_id = f"{classname}::{name}" if classname else name
        detail = (
            failure.attrib.get("message") or failure.text or "test failed"
        ).strip()
        failures.append((test_id, detail[:4000]))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)

    if not args.report.is_file():
        print("::error title=Offline tests failed::JUnit report was not generated")
        return 1

    failures = failed_cases(args.report)
    if not failures:
        print("::error title=Offline tests failed::No failed JUnit cases were found")
        return 1

    for test_id, detail in failures[:50]:
        print(f"::error title={_escape_command(test_id)}::{_escape_command(detail)}")
    if len(failures) > 50:
        print(f"::error title=Additional failures::{len(failures) - 50} more failures")
    return 1


if __name__ == "__main__":
    sys.exit(main())
